"""Memory writes: validate, embed, and persist episodes, facts, and tasks.

Semantic memory is APPEND-ONLY. A belief is never updated in place:
  - a repeated statement writes a new version (merged provenance, the
    higher confidence) and stamps the old row's superseded_at,
  - a contradicting statement on a single-valued subject (name, location,
    employer, "my X is Y" attributes) supersedes the old belief with a new
    row linked through replaces_id - a belief flip that timetravel.py can
    show and diff later.
Content and confidence of past versions are never destroyed, so the
agent's belief state at any moment stays reconstructable forever (and
recent moments can be read natively with AS OF SYSTEM TIME).

Every write validates before persisting; every fact carries provenance:
which episodes taught it and which method produced it.

Invoked by: agent.py, consolidate.py, tools/chaos_demo.py.
Inputs: a Database from db.py + memory payloads. Outputs: row UUIDs.
"""

import json

import embeddings

VALID_ROLES = ("user", "assistant", "system")
FACT_MERGE_SIMILARITY = 0.92

# Subjects that may hold several beliefs at once. Everything else (name,
# location, work, "my X is Y" attributes) is single-valued: a new belief
# supersedes the old one.
MULTI_VALUE_SUBJECTS = {"user.preference", "user.health",
                        "conversation.summary"}


class MemoryStore:

    def __init__(self, database):
        self.db = database

    # -- conversations ----------------------------------------------------

    def create_conversation(self, title=""):
        row = self.db.execute(
            "INSERT INTO conversations (title) VALUES (%s) RETURNING id",
            (title,), fetch="one")
        return str(row[0])

    def conversation_exists(self, conversation_id):
        row = self.db.execute(
            "SELECT 1 FROM conversations WHERE id = %s",
            (conversation_id,), fetch="one")
        return row is not None

    # -- episodic memory ---------------------------------------------------

    def add_episode(self, conversation_id, role, content, meta=None):
        if role not in VALID_ROLES:
            raise ValueError("invalid role: {0}".format(role))
        content = (content or "").strip()
        if not content:
            raise ValueError("episode content must not be empty")
        vec = embeddings.embed(content)
        row = self.db.execute(
            "INSERT INTO episodes"
            " (conversation_id, role, content, embedding, embedding_model, meta)"
            " VALUES (%s, %s, %s, %s::vector, %s, %s)"
            " RETURNING id",
            (conversation_id, role, content,
             embeddings.vector_literal(vec), embeddings.model_name(),
             json.dumps(meta or {})),
            fetch="one")
        return str(row[0])

    # -- semantic memory (append-only versions) ------------------------------

    def _insert_fact_version(self, subject, content, confidence, sources,
                             vec, replaces_id=None, supersede_id=None):
        """Atomically stamp the old version (if any) and insert the new one."""
        statements = []
        if supersede_id is not None:
            statements.append((
                "UPDATE facts SET superseded_at = now()"
                " WHERE id = %s AND superseded_at IS NULL",
                (supersede_id,), "none"))
        statements.append((
            "INSERT INTO facts (subject, content, confidence, provenance,"
            " embedding, embedding_model, replaces_id)"
            " VALUES (%s, %s, %s, %s, %s::vector, %s, %s) RETURNING id",
            (subject, content, confidence, json.dumps({"sources": sources}),
             embeddings.vector_literal(vec), embeddings.model_name(),
             replaces_id), "one"))
        results = self.db.transaction(statements)
        return str(results[-1][0])

    def add_fact(self, subject, content, confidence, provenance):
        """Record a belief. Returns (fact_id, merged_bool).

        merged=True means the statement repeated an existing belief: a new
        version row was written with combined provenance and the higher
        confidence, and the old version was superseded. A contradiction on
        a single-valued subject also writes a new version (belief flip),
        but reports merged=False because the content changed.
        """
        subject = (subject or "").strip()
        content = (content or "").strip()
        if not subject or not content:
            raise ValueError("fact subject and content must not be empty")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be within [0, 1]")

        vec = embeddings.embed(content)
        current = self.db.execute(
            "SELECT id, content, confidence, provenance FROM facts"
            " WHERE subject = %s AND superseded_at IS NULL"
            " AND embedding_model = %s",
            (subject, embeddings.model_name()), fetch="all") or []

        # 1. Repetition of an existing belief -> reinforce (new version).
        for fact_id, old_content, old_conf, old_prov in current:
            similarity = embeddings.cosine_similarity(
                vec, embeddings.embed(old_content))
            if old_content.lower() == content.lower() \
                    or similarity >= FACT_MERGE_SIMILARITY:
                old_sources = []
                if isinstance(old_prov, dict):
                    old_sources = old_prov.get("sources", [])
                sources = (old_sources + [provenance])[-20:]
                new_id = self._insert_fact_version(
                    subject, old_content, max(confidence, old_conf),
                    sources, embeddings.embed(old_content),
                    replaces_id=fact_id, supersede_id=fact_id)
                return new_id, True

        # 2. Contradiction on a single-valued subject -> belief flip.
        if subject not in MULTI_VALUE_SUBJECTS and current:
            old_id = str(current[0][0])
            new_id = self._insert_fact_version(
                subject, content, confidence, [provenance], vec,
                replaces_id=old_id, supersede_id=old_id)
            return new_id, False

        # 3. Genuinely new belief.
        new_id = self._insert_fact_version(
            subject, content, confidence, [provenance], vec)
        return new_id, False

    def supersede_fact(self, fact_id):
        """Retire a belief without replacement (append-only: the row stays)."""
        self.db.execute(
            "UPDATE facts SET superseded_at = now()"
            " WHERE id = %s AND superseded_at IS NULL", (fact_id,))

    # -- task state ---------------------------------------------------------

    def add_task(self, title, conversation_id=None, payload=None):
        title = (title or "").strip()
        if not title:
            raise ValueError("task title must not be empty")
        duplicate = self.db.execute(
            "SELECT id FROM tasks WHERE status = 'open' AND lower(title) = lower(%s)",
            (title,), fetch="one")
        if duplicate:
            return str(duplicate[0]), True
        row = self.db.execute(
            "INSERT INTO tasks (conversation_id, title, payload)"
            " VALUES (%s, %s, %s) RETURNING id",
            (conversation_id, title, json.dumps(payload or {})),
            fetch="one")
        return str(row[0]), False

    def set_task_status(self, task_id, status):
        if status not in ("open", "done", "cancelled"):
            raise ValueError("invalid task status: {0}".format(status))
        self.db.execute(
            "UPDATE tasks SET status = %s, updated_at = now() WHERE id = %s",
            (status, task_id))

    def open_tasks(self, limit=20):
        rows = self.db.execute(
            "SELECT id, title, created_at FROM tasks WHERE status = 'open'"
            " ORDER BY created_at DESC LIMIT %s", (limit,), fetch="all") or []
        return [{"id": str(r[0]), "title": r[1], "created_at": r[2]}
                for r in rows]

    # -- recall traces (decision audit) ---------------------------------------

    def add_recall_trace(self, conversation_id, user_episode_id,
                         reply_episode_id, query, recalled):
        row = self.db.execute(
            "INSERT INTO recall_traces (conversation_id, user_episode_id,"
            " reply_episode_id, query, recalled)"
            " VALUES (%s, %s, %s, %s, %s) RETURNING id",
            (conversation_id, user_episode_id, reply_episode_id, query,
             json.dumps(recalled)),
            fetch="one")
        return str(row[0])

    # -- counters (status panel) ---------------------------------------------

    def counts(self):
        row = self.db.execute(
            "SELECT (SELECT count(*) FROM episodes),"
            " (SELECT count(*) FROM facts WHERE superseded_at IS NULL),"
            " (SELECT count(*) FROM facts),"
            " (SELECT count(*) FROM tasks WHERE status = 'open')",
            fetch="one")
        return {"episodes": row[0], "facts": row[1],
                "fact_versions": row[2], "open_tasks": row[3]}
