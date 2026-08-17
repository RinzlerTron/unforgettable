"""Memory writes: validation before persist, fact merging, task dedupe."""

import pytest

from memory_store import MemoryStore


def test_episode_requires_valid_role(database):
    store = MemoryStore(database)
    conversation_id = store.create_conversation("test")
    with pytest.raises(ValueError):
        store.add_episode(conversation_id, "robot", "hello")


def test_episode_requires_content(database):
    store = MemoryStore(database)
    conversation_id = store.create_conversation("test")
    with pytest.raises(ValueError):
        store.add_episode(conversation_id, "user", "   ")


def test_episode_roundtrip_with_embedding(database):
    store = MemoryStore(database)
    conversation_id = store.create_conversation("test")
    episode_id = store.add_episode(conversation_id, "user",
                                   "my cat is called Miso", meta={"k": 1})
    row = database.execute(
        "SELECT role, content, embedding IS NOT NULL, meta->>'k'"
        " FROM episodes WHERE id = %s", (episode_id,), fetch="one")
    assert row == ("user", "my cat is called Miso", True, "1")


def test_fact_confidence_bounds(database):
    store = MemoryStore(database)
    with pytest.raises(ValueError):
        store.add_fact("user.name", "The user's name is Alice.", 1.5, {})


def test_fact_requires_subject_and_content(database):
    store = MemoryStore(database)
    with pytest.raises(ValueError):
        store.add_fact("", "The user's name is Alice.", 0.9, {})
    with pytest.raises(ValueError):
        store.add_fact("user.name", "", 0.9, {})


def test_duplicate_fact_merges_into_new_version(database):
    store = MemoryStore(database)
    first_id, merged_first = store.add_fact(
        "user.name", "The user's name is Alice.", 0.8,
        {"method": "turn-extraction", "episode_id": "ep1"})
    assert merged_first is False

    second_id, merged_second = store.add_fact(
        "user.name", "The user's name is Alice.", 0.95,
        {"method": "consolidation", "episode_id": "ep2"})
    assert merged_second is True
    assert second_id != first_id  # append-only: a new version row

    new_row = database.execute(
        "SELECT confidence, provenance, replaces_id, superseded_at"
        " FROM facts WHERE id = %s", (second_id,), fetch="one")
    assert new_row[0] == 0.95  # kept the higher confidence
    methods = [s["method"] for s in new_row[1]["sources"]]
    assert methods == ["turn-extraction", "consolidation"]
    assert str(new_row[2]) == first_id
    assert new_row[3] is None  # the new version is the current belief

    old_row = database.execute(
        "SELECT confidence, superseded_at FROM facts WHERE id = %s",
        (first_id,), fetch="one")
    assert old_row[0] == 0.8   # history is never rewritten
    assert old_row[1] is not None

    active = database.execute(
        "SELECT count(*) FROM facts WHERE subject = 'user.name'"
        " AND superseded_at IS NULL", fetch="one")[0]
    total = database.execute(
        "SELECT count(*) FROM facts WHERE subject = 'user.name'",
        fetch="one")[0]
    assert (active, total) == (1, 2)


def test_contradiction_flips_single_valued_belief(database):
    store = MemoryStore(database)
    old_id, _ = store.add_fact(
        "user.location", "The user lives in Singapore.", 0.9, {"m": 1})
    new_id, merged = store.add_fact(
        "user.location", "The user lives in Chennai.", 0.9, {"m": 2})
    assert merged is False
    assert new_id != old_id

    active = database.execute(
        "SELECT content, replaces_id FROM facts"
        " WHERE subject = 'user.location' AND superseded_at IS NULL",
        fetch="all")
    assert len(active) == 1
    assert "Chennai" in active[0][0]
    assert str(active[0][1]) == old_id


def test_multi_valued_subject_keeps_both_beliefs(database):
    store = MemoryStore(database)
    store.add_fact("user.preference", "The user likes laksa.", 0.9, {})
    _, merged = store.add_fact(
        "user.preference", "The user dislikes durian.", 0.9, {})
    assert merged is False
    active = database.execute(
        "SELECT count(*) FROM facts WHERE subject = 'user.preference'"
        " AND superseded_at IS NULL", fetch="one")[0]
    assert active == 2


def test_task_dedupe_on_open_title(database):
    store = MemoryStore(database)
    first_id, duplicate_first = store.add_task("renew my passport")
    second_id, duplicate_second = store.add_task("Renew My Passport")
    assert duplicate_first is False
    assert duplicate_second is True
    assert first_id == second_id

    store.set_task_status(first_id, "done")
    third_id, duplicate_third = store.add_task("renew my passport")
    assert duplicate_third is False
    assert third_id != first_id


def test_task_status_validation(database):
    store = MemoryStore(database)
    task_id, _ = store.add_task("water the plants")
    with pytest.raises(ValueError):
        store.set_task_status(task_id, "abandoned")
