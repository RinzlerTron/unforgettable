"""The agent turn loop: recall memories, respond, write new memories.

One turn does, in order:
  1. recall     - build a MemoryBundle from CockroachDB for the user's text
  2. remember   - store the user's message as an episode
  3. learn      - extract facts (with provenance to that episode) and tasks
  4. respond    - hand the prompt + memory context to the configured LLM
  5. remember   - store the reply as an episode
  6. audit      - record which memory rows the reply used (recall_traces),
                  so timetravel.explain_reply() can justify any past answer
Because every step reads and writes CockroachDB rows, an agent process can
be killed and restarted - or the database node under it can die - without
losing a single memory. Nothing is cached in process memory.

Invoked by: chat_cli.py, web.py, tools/chaos_demo.py.
Inputs: conversation id + user text. Outputs: reply + recall trace.
"""

import logging

import config
import extract
import llm
from db import Database
from memory_store import MemoryStore
from recall import Recall

log = logging.getLogger("agent")


def format_memory_context(bundle):
    """Render the MemoryBundle as the MEMORY section of the system prompt."""
    lines = ["MEMORY (recalled from CockroachDB for this turn):"]
    facts = bundle.get("facts", [])
    if facts:
        lines.append("Known facts:")
        for fact in facts:
            lines.append("- {0} (confidence {1:.2f})".format(
                fact["content"], fact["confidence"]))
    tasks = bundle.get("tasks", [])
    if tasks:
        lines.append("Open tasks:")
        for task in tasks:
            lines.append("- {0}".format(task["title"]))
    episodes = bundle.get("episodes", [])
    if episodes:
        lines.append("Related moments from past conversations:")
        for episode in episodes:
            lines.append("- [{0}] {1}".format(
                episode["role"], episode["content"][:200]))
    if len(lines) == 1:
        lines.append("(memory is empty so far)")
    return "\n".join(lines)


class Agent:

    def __init__(self, database=None, client=None):
        self.db = database or Database()
        self.db.ensure_schema()
        self.store = MemoryStore(self.db)
        self.recaller = Recall(self.db)
        self.client = client or llm.get_client()

    def new_conversation(self, title=""):
        return self.store.create_conversation(title)

    def turn(self, conversation_id, user_text):
        """Run one full agent turn; returns a dict with reply and trace."""
        user_text = (user_text or "").strip()
        if not user_text:
            raise ValueError("empty user message")

        bundle = self.recaller.recall(user_text, conversation_id)

        episode_id = self.store.add_episode(conversation_id, "user", user_text)
        provenance = {"method": "turn-extraction", "episode_id": episode_id,
                      "backend": self.client.name}

        stored_facts = []
        for fact in self.client.extract_facts(user_text):
            fact_id, merged = self.store.add_fact(
                fact["subject"], fact["content"], fact["confidence"],
                provenance)
            stored_facts.append({"id": fact_id, "content": fact["content"],
                                 "merged": merged})

        stored_tasks = []
        for title in extract.extract_tasks(user_text):
            task_id, duplicate = self.store.add_task(
                title, conversation_id=conversation_id,
                payload={"episode_id": episode_id})
            stored_tasks.append({"id": task_id, "title": title,
                                 "duplicate": duplicate})

        system = llm.SYSTEM_PROMPT + "\n\n" + format_memory_context(bundle)
        messages = [{"role": t["role"], "content": t["content"]}
                    for t in bundle["recent_turns"]
                    if t["role"] in ("user", "assistant")]
        messages.append({"role": "user", "content": user_text})
        reply = self.client.respond(system, messages, bundle=bundle)

        reply_episode_id = self.store.add_episode(
            conversation_id, "assistant", reply,
            meta={"in_reply_to": episode_id})

        # Decision audit: record exactly which memory rows this reply used,
        # so timetravel.explain_reply() can answer "why did it say that?"
        recalled = {
            "facts": [{"id": f["id"], "content": f["content"],
                       "score": round(f["score"], 4)}
                      for f in bundle["facts"]],
            "episodes": [{"id": e["id"], "content": e["content"],
                          "score": round(e["score"], 4)}
                         for e in bundle["episodes"]],
            "tasks": bundle["tasks"],
        }
        self.store.add_recall_trace(conversation_id, episode_id,
                                    reply_episode_id, user_text, recalled)

        return {
            "reply": reply,
            "reply_episode_id": reply_episode_id,
            "recalled": recalled,
            "stored": {"facts": stored_facts, "tasks": stored_tasks,
                       "episode_id": episode_id},
        }

    def status(self):
        """Health snapshot: which node we talk to and how much we remember."""
        node_id, url_index = self.db.node_info()
        counts = self.store.counts()
        return {
            "node_id": node_id,
            "url": self.db.urls[url_index],
            "urls_configured": len(self.db.urls),
            "llm_backend": self.client.name,
            "embedding_backend": config.EMBED_BACKEND,
            "counts": counts,
        }
