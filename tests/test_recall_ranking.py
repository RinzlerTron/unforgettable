"""Recall ranking: the pure scoring function and end-to-end DB retrieval."""

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


def test_recall_includes_open_tasks(database):
    store = MemoryStore(database)
    conversation_id = store.create_conversation("test")
    store.add_task("renew my passport", conversation_id=conversation_id)
    bundle = Recall(database).recall("hello", conversation_id)
    assert any("passport" in t["title"] for t in bundle["tasks"])
