"""Time travel: rewind beliefs, diff belief changes, explain past replies."""

import time

import timetravel
from agent import Agent
from memory_store import MemoryStore


def _db_now(database):
    return database.execute("SELECT now()", fetch="one")[0]


def test_beliefs_at_rewinds_a_flip(database):
    store = MemoryStore(database)
    t0 = _db_now(database)
    time.sleep(0.05)
    store.add_fact("user.name", "The user's name is Priya.", 0.9, {"m": 1})
    time.sleep(0.05)
    t1 = _db_now(database)
    time.sleep(0.05)
    store.add_fact("user.name", "The user's name is Anya.", 0.9, {"m": 2})
    time.sleep(0.05)
    t2 = _db_now(database)

    at_t0 = timetravel.beliefs_at(database, t0)
    assert at_t0["beliefs"] == []
    assert at_t0["mechanism"] in ("as_of_system_time",
                                  "version_reconstruction")

    at_t1 = timetravel.beliefs_at(database, t1)
    assert len(at_t1["beliefs"]) == 1
    assert "Priya" in at_t1["beliefs"][0]["content"]

    at_t2 = timetravel.beliefs_at(database, t2)
    assert len(at_t2["beliefs"]) == 1
    assert "Anya" in at_t2["beliefs"][0]["content"]


def test_beliefs_at_falls_back_beyond_gc_window(database):
    snapshot = timetravel.beliefs_at(database, "2000-01-01T00:00:00+00:00")
    assert snapshot["mechanism"] == "version_reconstruction"
    assert snapshot["beliefs"] == []


def test_belief_diff_classifies_changes(database):
    store = MemoryStore(database)
    t0 = _db_now(database)
    time.sleep(0.05)
    store.add_fact("user.name", "The user's name is Priya.", 0.9, {"m": 1})
    store.add_fact("user.preference", "The user likes laksa.", 0.8, {"m": 1})
    time.sleep(0.05)
    t1 = _db_now(database)
    time.sleep(0.05)
    store.add_fact("user.name", "The user's name is Anya.", 0.9, {"m": 2})
    store.add_fact("user.preference", "The user dislikes durian.", 0.8,
                   {"m": 2})
    time.sleep(0.05)
    t2 = _db_now(database)

    first_window = timetravel.belief_diff(database, t0, t1)
    assert len(first_window["learned"]) == 2
    assert first_window["revised"] == []
    assert first_window["retired"] == []

    second_window = timetravel.belief_diff(database, t1, t2)
    learned_contents = [f["content"] for f in second_window["learned"]]
    assert any("durian" in c for c in learned_contents)
    assert len(second_window["revised"]) == 1
    flip = second_window["revised"][0]
    assert flip["change"] == "changed_belief"
    assert "Priya" in flip["before"]["content"]
    assert "Anya" in flip["after"]["content"]
    assert second_window["retired"] == []


def test_belief_diff_shows_reinforcement(database):
    store = MemoryStore(database)
    store.add_fact("user.cat", "The user's cat is Miso.", 0.7, {"m": 1})
    time.sleep(0.05)
    t1 = _db_now(database)
    time.sleep(0.05)
    store.add_fact("user.cat", "The user's cat is Miso.", 0.95, {"m": 2})
    time.sleep(0.05)
    t2 = _db_now(database)

    diff = timetravel.belief_diff(database, t1, t2)
    assert len(diff["revised"]) == 1
    change = diff["revised"][0]
    assert change["change"] == "reinforced"
    assert change["before"]["confidence"] == 0.7
    assert change["after"]["confidence"] == 0.95


def test_explain_reply_traces_memory_to_source(database):
    agent = Agent(database=database)
    conversation_id = agent.new_conversation("audit test")
    agent.turn(conversation_id, "My cat is called Miso.")
    result = agent.turn(conversation_id, "What is my cat's name?")

    audit = timetravel.explain_reply(database, result["reply_episode_id"])
    assert audit is not None
    assert audit["user_message"] == "What is my cat's name?"
    assert audit["reply"] == result["reply"]
    used = [f for f in audit["used_facts"] if "Miso" in f["content"]]
    assert used, "the cat fact should appear in the audit"
    assert used[0]["taught_by"] is not None
    assert "Miso" in used[0]["taught_by"]["content"]


def test_explain_reply_missing_trace_returns_none(database):
    assert timetravel.explain_reply(
        database, "00000000-0000-0000-0000-000000000000") is None
