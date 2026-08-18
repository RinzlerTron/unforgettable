"""Recall ranking: the pure scoring function and end-to-end DB retrieval."""

import config
import embeddings
import recall
from memory_store import MemoryStore
from recall import Recall, score_memory


def test_similarity_dominates():
    high_sim = score_memory(0.9, age_hours=100.0, keyword_overlap=0.0)
    low_sim = score_memory(0.1, age_hours=100.0, keyword_overlap=0.0)
    assert high_sim > low_sim


def test_recency_breaks_ties():
    fresh = score_memory(0.5, age_hours=1.0, keyword_overlap=0.0)
    stale = score_memory(0.5, age_hours=500.0, keyword_overlap=0.0)
    assert fresh > stale


def test_keyword_overlap_adds():
    with_kw = score_memory(0.5, age_hours=10.0, keyword_overlap=1.0)
    without_kw = score_memory(0.5, age_hours=10.0, keyword_overlap=0.0)
    assert with_kw > without_kw


def test_keyword_overlap_fraction():
    terms = {"cat", "name"}
    assert recall.keyword_overlap(terms, "my cat is called Miso") == 0.5
    assert recall.keyword_overlap(terms, "the weather is nice") == 0.0
    assert recall.keyword_overlap(set(), "anything") == 0.0


def test_recall_ranks_relevant_fact_first(database):
    store = MemoryStore(database)
    store.add_fact("user.cat", "The user's cat is called Miso.", 0.9,
                   {"method": "test"})
    store.add_fact("user.location", "The user lives in Singapore.", 0.9,
                   {"method": "test"})
    store.add_fact("user.preference", "The user likes laksa.", 0.9,
                   {"method": "test"})
    conversation_id = store.create_conversation("test")

    bundle = Recall(database).recall("what is my cat's name",
                                     conversation_id)
    assert bundle["facts"], "expected facts to be recalled"
    assert "Miso" in bundle["facts"][0]["content"]


def test_recall_excludes_live_conversation_from_episodes(database):
    store = MemoryStore(database)
    past = store.create_conversation("past")
    live = store.create_conversation("live")
    store.add_episode(past, "user", "I adopted a cat named Miso last year")
    store.add_episode(live, "user", "my cat again")

    bundle = Recall(database).recall("tell me about my cat", live)
    for episode in bundle["episodes"]:
        assert episode["conversation_id"] == past


def test_live_conversation_cannot_crowd_out_past_memories(database):
    """Regression: the live conversation must be excluded in SQL, before
    the vector top-k. A live conversation longer than the candidate pool
    used to push every cross-conversation memory out of the candidates."""
    store = MemoryStore(database)
    past = store.create_conversation("past")
    live = store.create_conversation("live")
    store.add_episode(past, "user", "I adopted a cat named Miso last year")
    for i in range(config.VECTOR_CANDIDATES + 6):
        store.add_episode(live, "user",
                          "my cat again, message {0}".format(i))

    bundle = Recall(database).recall("tell me about my cat", live)
    assert any("Miso" in e["content"] for e in bundle["episodes"])


def test_vector_index_serves_recall_queries(database):
    """The distributed vector index must actually serve the recall
    queries: EXPLAIN must say "vector search", never FULL SCAN. An index
    that exists but is not in the plan is decoration, not
    infrastructure."""
    store = MemoryStore(database)
    conversation_id = store.create_conversation("t")
    for i in range(10):
        store.add_fact("user.item{0}".format(i),
                       "Fact number {0}.".format(i), 0.9, {"method": "t"})
        store.add_episode(conversation_id, "user",
                          "episode number {0}".format(i))
    query_vec = embeddings.vector_literal(embeddings.embed("number 5"))
    model = embeddings.model_name()

    fact_plan = "\n".join(r[0] for r in database.execute(
        "EXPLAIN SELECT id FROM facts"
        " WHERE superseded_at IS NULL AND embedding_model = %s"
        " ORDER BY embedding <-> %s::vector LIMIT %s",
        (model, query_vec, config.VECTOR_CANDIDATES), fetch="all"))
    assert "vector search" in fact_plan, fact_plan
    assert "FULL SCAN" not in fact_plan, fact_plan

    episode_plan = "\n".join(r[0] for r in database.execute(
        "EXPLAIN SELECT id FROM episodes"
        " WHERE embedding_model = %s AND role IN ('user', 'assistant')"
        " ORDER BY embedding <-> %s::vector LIMIT %s",
        (model, query_vec,
         config.VECTOR_CANDIDATES * config.EPISODE_POOL_FACTOR),
        fetch="all"))
    assert "vector search" in episode_plan, episode_plan
    assert "FULL SCAN" not in episode_plan, episode_plan


def test_recall_includes_open_tasks(database):
    store = MemoryStore(database)
    conversation_id = store.create_conversation("test")
    store.add_task("renew my passport", conversation_id=conversation_id)
    bundle = Recall(database).recall("hello", conversation_id)
    assert any("passport" in t["title"] for t in bundle["tasks"])
