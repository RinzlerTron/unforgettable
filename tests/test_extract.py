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
