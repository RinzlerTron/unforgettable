"""Memory recall: hybrid retrieval (vector + keyword + recency) and ranking.

For each turn the agent asks for a MemoryBundle:
  - the verbatim tail of the current conversation (short-term context),
  - similar past episodes found by vector search over the whole history,
  - relevant facts found by vector search plus keyword match,
  - open tasks.
Vector search runs in CockroachDB (ORDER BY embedding <-> query, served by
the distributed vector index - both embedding backends are unit-normalized,
so L2 order equals cosine order and the query stays on the index; a test
asserts the EXPLAIN plan says "vector search", not FULL SCAN). Candidates
are then re-ranked in Python with a transparent score combining similarity,
recency, and keyword overlap.

Invoked by: agent.py (every turn), web.py (memory inspector).
Inputs: query text + Database. Outputs: MemoryBundle dict.
"""

import datetime
import math

import config
import embeddings


def score_memory(similarity, age_hours, keyword_overlap):
    """Pure ranking function; unit-tested directly.

    similarity: cosine similarity in [-1, 1] (vector distance rebased).
    age_hours: hours since the memory was written.
    keyword_overlap: fraction of query terms present in the memory text.
    """
    recency = math.exp(-max(age_hours, 0.0) / config.RECENCY_HALF_LIFE_HOURS)
    return (config.W_SIMILARITY * similarity
            + config.W_RECENCY * recency
            + config.W_KEYWORD * keyword_overlap)


def keyword_overlap(query_terms, text):
    if not query_terms:
        return 0.0
    text_terms = set(embeddings.tokenize(text))
    hits = sum(1 for term in query_terms if term in text_terms)
    return hits / len(query_terms)


def _age_hours(created_at, now):
    if created_at is None:
        return 0.0
    return max((now - created_at).total_seconds() / 3600.0, 0.0)


def _l2_to_similarity(distance):
    """Cosine similarity recovered from L2 distance of unit vectors:
    ||a-b||^2 = 2 - 2*cos(a,b), so cos = 1 - d^2/2."""
    return 1.0 - (distance * distance) / 2.0


class Recall:

    def __init__(self, database):
        self.db = database

    def recent_turns(self, conversation_id, limit=None):
        limit = limit or config.RECENT_TURNS
        rows = self.db.execute(
            "SELECT role, content, created_at FROM episodes"
            " WHERE conversation_id = %s"
            " ORDER BY created_at DESC LIMIT %s",
            (conversation_id, limit), fetch="all") or []
        return [{"role": r[0], "content": r[1], "created_at": r[2]}
                for r in reversed(rows)]

    def similar_episodes(self, query_vec, exclude_conversation_id):
        """Vector search over past episodes, excluding the live conversation
        (its tail is already provided verbatim by recent_turns).

        The SQL filters match episodes_ann_idx exactly (prefix column +
        partial predicate), so the query is served by the vector index; a
        conversation_id != filter in SQL would demote it to a full scan
        (verified with EXPLAIN). The exclusion therefore happens in Python
        over an oversampled pool, sized so even a live conversation longer
        than the whole candidate top-k cannot crowd out every
        cross-conversation memory."""
        rows = self.db.execute(
            "SELECT id, conversation_id, role, content, created_at,"
            " embedding <-> %s::vector AS distance"
            " FROM episodes"
            " WHERE embedding_model = %s AND role IN ('user', 'assistant')"
            " ORDER BY embedding <-> %s::vector"
            " LIMIT %s",
            (embeddings.vector_literal(query_vec), embeddings.model_name(),
             embeddings.vector_literal(query_vec),
             config.VECTOR_CANDIDATES * config.EPISODE_POOL_FACTOR),
            fetch="all") or []
        return [
            {"id": str(r[0]), "conversation_id": str(r[1]), "role": r[2],
             "content": r[3], "created_at": r[4],
             "similarity": _l2_to_similarity(float(r[5]))}
            for r in rows
            if str(r[1]) != str(exclude_conversation_id)
        ][: config.VECTOR_CANDIDATES]

    def candidate_facts(self, query_vec, query_terms):
        """Union of vector-nearest facts and keyword-matched facts."""
        by_id = {}
        rows = self.db.execute(
            "SELECT id, subject, content, confidence, valid_from,"
            " embedding <-> %s::vector AS distance"
            " FROM facts"
            " WHERE superseded_at IS NULL AND embedding_model = %s"
            " ORDER BY embedding <-> %s::vector"
            " LIMIT %s",
            (embeddings.vector_literal(query_vec), embeddings.model_name(),
             embeddings.vector_literal(query_vec), config.VECTOR_CANDIDATES),
            fetch="all") or []
        for r in rows:
            by_id[str(r[0])] = {
                "id": str(r[0]), "subject": r[1], "content": r[2],
                "confidence": float(r[3]), "created_at": r[4],
                "similarity": _l2_to_similarity(float(r[5]))}
        for term in list(query_terms)[:8]:
            rows = self.db.execute(
                "SELECT id, subject, content, confidence, valid_from"
                " FROM facts WHERE superseded_at IS NULL"
                " AND (content ILIKE %s OR subject ILIKE %s) LIMIT 10",
                ("%" + term + "%", "%" + term + "%"), fetch="all") or []
            for r in rows:
                if str(r[0]) not in by_id:
                    by_id[str(r[0])] = {
                        "id": str(r[0]), "subject": r[1], "content": r[2],
                        "confidence": float(r[3]), "created_at": r[4],
                        "similarity": 0.0}
        return list(by_id.values())

    def recall(self, query_text, conversation_id):
        """Build the MemoryBundle the agent injects into its prompt."""
        now = datetime.datetime.now(datetime.timezone.utc)
        query_vec = embeddings.embed(query_text)
        query_terms = set(embeddings.tokenize(query_text))

        episodes = self.similar_episodes(query_vec, conversation_id)
        for item in episodes:
            item["score"] = score_memory(
                item["similarity"], _age_hours(item["created_at"], now),
                keyword_overlap(query_terms, item["content"]))
        episodes.sort(key=lambda item: item["score"], reverse=True)

        facts = self.candidate_facts(query_vec, query_terms)
        for item in facts:
            item["score"] = score_memory(
                item["similarity"], _age_hours(item["created_at"], now),
                keyword_overlap(query_terms, item["content"]))
            # Confidence scales the whole score: shaky facts rank lower.
            item["score"] *= item["confidence"]
        facts.sort(key=lambda item: item["score"], reverse=True)

        open_tasks = self.db.execute(
            "SELECT id, title FROM tasks WHERE status = 'open'"
            " ORDER BY created_at DESC LIMIT 10", fetch="all") or []

        return {
            "recent_turns": self.recent_turns(conversation_id),
            "episodes": episodes[: config.RECALL_EPISODES],
            "facts": facts[: config.RECALL_FACTS],
            "tasks": [{"id": str(r[0]), "title": r[1]} for r in open_tasks],
        }
