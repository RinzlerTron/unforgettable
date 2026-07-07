"""Memory consolidation: distill old episodes into semantic facts.

Run periodically (or via `./run.sh consolidate`). Takes episodes older than
CONSOLIDATE_MIN_AGE_MINUTES that have not been consolidated, extracts
durable facts from the user's own words, writes a per-conversation summary
fact, and stamps the episodes so they are never processed twice. Facts
created here carry provenance {"method": "consolidation", "episode_ids":
[...]} so any memory can be traced back to the exact conversation rows
that produced it.

Invoked by: run.sh consolidate, agent maintenance, tests.
Inputs: Database. Outputs: number of facts written.
"""

import logging

import config
import extract
import llm
from memory_store import MemoryStore

log = logging.getLogger("consolidate")


def _summarize_heuristic(user_lines):
    """Deterministic fallback summary: the distinct topics the user raised."""
    seen = []
    for line in user_lines:
        first_sentence = line.split(".")[0].strip()
        if first_sentence and first_sentence not in seen:
            seen.append(first_sentence)
        if len(seen) >= 4:
            break
    if not seen:
        return None
    return "In an earlier conversation the user said: " + "; ".join(seen) + "."


def consolidate(database, min_age_minutes=None, client=None):
    """Consolidate eligible episodes; returns count of facts written/merged."""
    min_age = min_age_minutes
    if min_age is None:
        min_age = config.CONSOLIDATE_MIN_AGE_MINUTES
    store = MemoryStore(database)
    client = client or llm.get_client()

    rows = database.execute(
        "SELECT id, conversation_id, role, content FROM episodes"
        " WHERE consolidated_at IS NULL"
        " AND created_at < now() - %s * INTERVAL '1 minute'"
        " ORDER BY conversation_id, created_at"
        " LIMIT %s",
        (min_age, config.CONSOLIDATE_BATCH), fetch="all") or []
    if not rows:
        return 0

    by_conversation = {}
    for episode_id, conversation_id, role, content in rows:
        by_conversation.setdefault(str(conversation_id), []).append(
            (str(episode_id), role, content))

    facts_written = 0
    for conversation_id, episodes in by_conversation.items():
        episode_ids = [e[0] for e in episodes]
        user_lines = [e[2] for e in episodes if e[1] == "user"]
        provenance = {"method": "consolidation",
                      "conversation_id": conversation_id,
                      "episode_ids": episode_ids[:50]}

        # 1. Re-run fact extraction over the user's own words. Facts that
        #    were already captured live simply merge (provenance grows).
        for line in user_lines:
            for fact in extract.extract_facts(line):
                store.add_fact(fact["subject"], fact["content"],
                               fact["confidence"], provenance)
                facts_written += 1

        # 2. One summary fact per consolidated batch, LLM-written when a
        #    model is configured, heuristic otherwise.
        summary = None
        if user_lines:
            summary = client.summarize(user_lines)
            if not summary:
                summary = _summarize_heuristic(user_lines)
        if summary:
            store.add_fact("conversation.summary", summary, 0.7, provenance)
            facts_written += 1

        placeholders = ",".join(["%s"] * len(episode_ids))
        database.execute(
            "UPDATE episodes SET consolidated_at = now()"
            " WHERE id IN ({0})".format(placeholders),
            tuple(episode_ids))
        log.info("consolidated %d episodes from conversation %s",
                 len(episode_ids), conversation_id)

    return facts_written
