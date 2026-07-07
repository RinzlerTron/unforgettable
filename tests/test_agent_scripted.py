"""Full agent turns in scripted mode: memory survives process restarts."""

import json

from agent import Agent
from conftest import FIXTURES


def _fixture():
    return json.loads((FIXTURES / "conversation_priya.json").read_text())


def test_memory_survives_new_agent_instance(database):
    fixture = _fixture()
    first_agent = Agent(database=database)
    conversation_id = first_agent.new_conversation("session one")
    for line in fixture["setup_lines"]:
        first_agent.turn(conversation_id, line)

    # A brand-new Agent object simulates a process restart: nothing is
    # carried over in RAM, only rows in CockroachDB.
    second_agent = Agent(database=database)
    new_conversation = second_agent.new_conversation("session two")
    for probe in fixture["probes"]:
        result = second_agent.turn(new_conversation, probe["ask"])
        assert probe["expect_in_reply"] in result["reply"].lower(), \
            "probe failed: {0}".format(probe["ask"])


def test_turn_stores_episodes_facts_and_tasks(database):
    agent = Agent(database=database)
    conversation_id = agent.new_conversation("test")
    result = agent.turn(conversation_id,
                        "My name is Priya. Remind me to renew my passport.")

    assert result["stored"]["facts"], "expected a fact to be stored"
    assert result["stored"]["tasks"], "expected a task to be stored"

    counts = agent.store.counts()
    assert counts["episodes"] == 2  # user turn + assistant reply
    assert counts["open_tasks"] == 1


def test_recall_trace_is_exposed(database):
    agent = Agent(database=database)
    conversation_id = agent.new_conversation("test")
    agent.turn(conversation_id, "My cat is called Miso.")
    result = agent.turn(conversation_id, "What is my cat's name?")
    recalled_facts = [f["content"] for f in result["recalled"]["facts"]]
    assert any("Miso" in content for content in recalled_facts)
    assert "miso" in result["reply"].lower()


def test_empty_message_rejected(database):
    agent = Agent(database=database)
    conversation_id = agent.new_conversation("test")
    try:
        agent.turn(conversation_id, "   ")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_status_reports_node_and_counts(database):
    agent = Agent(database=database)
    status = agent.status()
    assert status["node_id"] >= 1
    assert status["llm_backend"] == "scripted"
    assert set(status["counts"]) == {"episodes", "facts", "fact_versions",
                                     "open_tasks"}
