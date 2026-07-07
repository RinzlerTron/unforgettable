"""Consolidation: old episodes distilled into facts with full provenance."""

import consolidate
from memory_store import MemoryStore


def _backdate_episodes(database, hours=2):
    database.execute(
        "UPDATE episodes SET created_at = now() - %s * INTERVAL '1 hour'",
        (hours,))


def test_consolidation_writes_facts_with_provenance(database):
    store = MemoryStore(database)
    conversation_id = store.create_conversation("old chat")
    episode_ids = [
        store.add_episode(conversation_id, "user",
                          "My name is Priya and I live in Singapore."),
        store.add_episode(conversation_id, "assistant", "Nice to meet you."),
        store.add_episode(conversation_id, "user",
                          "My cat is called Miso."),
    ]
    _backdate_episodes(database)

    written = consolidate.consolidate(database, min_age_minutes=30)
    assert written > 0

    rows = database.execute(
        "SELECT subject, provenance FROM facts", fetch="all")
    subjects = [r[0] for r in rows]
    assert "user.name" in subjects
    assert "user.cat" in subjects
    assert "conversation.summary" in subjects

    for _, provenance in rows:
        source = provenance["sources"][0]
        assert source["method"] == "consolidation"
        assert set(source["episode_ids"]) == set(episode_ids)


def test_consolidation_marks_episodes_done(database):
    store = MemoryStore(database)
    conversation_id = store.create_conversation("old chat")
    store.add_episode(conversation_id, "user", "I work at Standard Chartered.")
    _backdate_episodes(database)

    consolidate.consolidate(database, min_age_minutes=30)
    unconsolidated = database.execute(
        "SELECT count(*) FROM episodes WHERE consolidated_at IS NULL",
        fetch="one")[0]
    assert unconsolidated == 0


def test_consolidation_is_idempotent(database):
    store = MemoryStore(database)
    conversation_id = store.create_conversation("old chat")
    store.add_episode(conversation_id, "user", "My name is Priya.")
    _backdate_episodes(database)

    consolidate.consolidate(database, min_age_minutes=30)
    second_run = consolidate.consolidate(database, min_age_minutes=30)
    assert second_run == 0


def test_fresh_episodes_left_alone(database):
    store = MemoryStore(database)
    conversation_id = store.create_conversation("live chat")
    store.add_episode(conversation_id, "user", "My name is Priya.")

    written = consolidate.consolidate(database, min_age_minutes=30)
    assert written == 0
    unconsolidated = database.execute(
        "SELECT count(*) FROM episodes WHERE consolidated_at IS NULL",
        fetch="one")[0]
    assert unconsolidated == 1
