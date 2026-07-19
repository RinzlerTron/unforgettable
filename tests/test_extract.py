"""Rule-based fact and task extraction against saved real inputs."""

import json

import extract
from conftest import FIXTURES


def _fixture_messages():
    return json.loads((FIXTURES / "user_messages.json").read_text())


def test_fixture_extractions():
    for case in _fixture_messages():
        facts = extract.extract_facts(case["text"])
        subjects = [f["subject"] for f in facts]
        for expected in case["expected_fact_subjects"]:
            assert expected in subjects, \
                "missing {0} for: {1}".format(expected, case["text"])
        tasks = extract.extract_tasks(case["text"])
        for expected in case["expected_tasks"]:
            assert expected in tasks, \
                "missing task '{0}' for: {1}".format(expected, case["text"])


def test_no_facts_from_smalltalk():
    assert extract.extract_facts("What is the weather like today?") == []
    assert extract.extract_tasks("thanks, that is all") == []


def test_fact_shape_and_confidence():
    facts = extract.extract_facts("My name is Priya")
    assert facts, "name should be extracted"
    fact = facts[0]
    assert fact["subject"] == "user.name"
    assert "Priya" in fact["content"]
    assert 0.0 <= fact["confidence"] <= 1.0


def test_attribute_stoplist_blocks_noise():
    assert extract.extract_facts("my question is why") == []


def test_duplicate_facts_deduped_within_message():
    facts = extract.extract_facts("My name is Priya. My name is Priya.")
    names = [f for f in facts if f["subject"] == "user.name"]
    assert len(names) == 1


def test_value_stops_at_sentence_boundary():
    facts = extract.extract_facts(
        "my cat is called Miso. Remind me to renew my passport")
    cats = [f for f in facts if f["subject"] == "user.cat"]
    assert cats and cats[0]["content"] == "The user's cat is Miso."


def test_value_stops_at_new_clause_with_my():
    facts = extract.extract_facts(
        "I live in Singapore, my cat is called Miso")
    homes = [f for f in facts if f["subject"] == "user.location"]
    assert homes and homes[0]["content"] == "The user lives in Singapore."


def test_value_drops_trailing_time_phrase():
    facts = extract.extract_facts("Actually I moved to Chennai last month")
    homes = [f for f in facts if f["subject"] == "user.location"]
    assert homes and homes[0]["content"] == "The user lives in Chennai."
