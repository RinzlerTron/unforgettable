"""Rule-based extraction of durable facts and task intents from user text.

This is the deterministic path used when MEM_LLM=off, and the fallback when
an LLM's structured extraction cannot be parsed. Patterns are deliberately
conservative: better to miss a fact than to store a wrong one (facts get
confidence scores; rule-extracted facts start at 0.9 because they quote the
user directly).

Invoked by: agent.py (every user turn), consolidate.py, llm.py (fallback).
Inputs: one user message. Outputs: list of fact dicts / task titles.
"""

import re

# Each pattern maps to (subject, content template). {0} is the captured value.
_FACT_PATTERNS = [
    (re.compile(r"\bmy name is ([A-Za-z][A-Za-z .'-]{0,60})", re.I),
     "user.name", "The user's name is {0}."),
    (re.compile(r"\bcall me ([A-Za-z][A-Za-z .'-]{0,60})", re.I),
     "user.name", "The user prefers to be called {0}."),
    (re.compile(r"\bi live in ([A-Za-z][A-Za-z .,'-]{0,80})", re.I),
     "user.location", "The user lives in {0}."),
    (re.compile(r"\bi(?:'m| am) from ([A-Za-z][A-Za-z .,'-]{0,80})", re.I),
     "user.location", "The user is from {0}."),
    (re.compile(r"\bi (?:just |recently )?moved to ([A-Za-z][A-Za-z .,'-]{0,80})", re.I),
     "user.location", "The user lives in {0}."),
    (re.compile(r"\bi work (?:at|for) ([A-Za-z0-9][A-Za-z0-9 .,&'-]{0,80})", re.I),
     "user.work", "The user works at {0}."),
    (re.compile(r"\bi(?:'m| am) an? ([A-Za-z][A-Za-z -]{0,50}?(?:er|or|ist|ian|eer))\b", re.I),
     "user.work", "The user is a {0}."),
    (re.compile(r"\bmy (\w{2,30}) is (?:called |named )?([A-Za-z0-9][A-Za-z0-9 .'-]{0,60})", re.I),
     "user.attribute", None),  # handled specially below
    (re.compile(r"\bmy (\w{2,30})'s name is ([A-Za-z0-9][A-Za-z0-9 .'-]{0,60})", re.I),
     "user.attribute", None),  # "my cat's name is Miso"
    (re.compile(r"\bi (?:really )?(love|like|prefer|enjoy) ([a-z0-9][a-z0-9 .,'-]{1,80})", re.I),
     "user.preference", "The user {0}s {1}." ),
    (re.compile(r"\bi (?:hate|dislike|can't stand|cannot stand) ([a-z0-9][a-z0-9 .,'-]{1,80})", re.I),
     "user.preference", "The user dislikes {0}."),
    (re.compile(r"\bi(?:'m| am) allergic to ([a-z0-9][a-z0-9 .,'-]{1,60})", re.I),
     "user.health", "The user is allergic to {0}."),
]

_TASK_PATTERNS = [
    re.compile(r"\bremind me to ([^.!?,\n]{3,120})", re.I),
    re.compile(r"\bdon'?t let me forget (?:to )?([^.!?,\n]{3,120})", re.I),
    re.compile(r"\badd (?:a )?(?:task|todo)(?: to)?:? ([^.!?,\n]{3,120})", re.I),
]

# Attribute words that are not durable facts about the user.
_ATTRIBUTE_STOPLIST = {
    "question", "point", "problem", "answer", "guess", "turn", "message",
    "favorite", "name",
}

# A captured value ends where a new clause begins ("Priya and I live in...").
_CLAUSE_STOP_WORDS = {"and", "but", "or", "so", "because", "while", "since",
                      "i", "im", "also", "which", "who", "though"}
# Fact values additionally shed trailing time phrases ("moved to Chennai
# last month" -> Chennai). Task titles keep them ("renew my passport
# next week" is the whole task).
_TIME_STOP_WORDS = {"last", "next", "this", "yesterday", "today",
                    "tomorrow", "recently"}
_VALUE_STOP_WORDS = _CLAUSE_STOP_WORDS | _TIME_STOP_WORDS


def _clean(value, stop_words=_VALUE_STOP_WORDS):
    # A value never crosses a sentence boundary ("Miso. Remind me to..."
    # is the cat's name plus the start of the next sentence), nor a
    # ", my ..." clause ("Singapore, my cat is called Miso").
    value = re.split(r"[.!?](?=\s|$)", value)[0]
    value = re.split(r",\s*my\b", value, flags=re.I)[0]
    value = re.sub(r"\s+", " ", value).strip(" .,'")
    kept = []
    for word in value.split(" "):
        if word.lower().strip(".,'") in stop_words:
            break
        kept.append(word)
    return " ".join(kept).strip(" .,'")


def extract_facts(text):
    """Return a list of {subject, content, confidence} dicts found in text."""
    facts = []
    seen = set()
    for pattern, subject, template in _FACT_PATTERNS:
        for match in pattern.finditer(text):
            groups = [_clean(g) for g in match.groups()]
            if subject == "user.attribute":
                attribute, value = groups[0].lower(), groups[1]
                if attribute in _ATTRIBUTE_STOPLIST or not value:
                    continue
                content = "The user's {0} is {1}.".format(attribute, value)
                subject_key = "user." + attribute
            else:
                if not all(groups):
                    continue
                content = template.format(*groups)
                subject_key = subject
            key = (subject_key, content.lower())
            if key in seen:
                continue
            seen.add(key)
            facts.append({
                "subject": subject_key,
                "content": content,
                "confidence": 0.9,
            })
    return facts


def extract_tasks(text):
    """Return a list of task title strings found in text."""
    titles = []
    for pattern in _TASK_PATTERNS:
        for match in pattern.finditer(text):
            title = _clean(match.group(1), stop_words=_CLAUSE_STOP_WORDS)
            if title and title not in titles:
                titles.append(title)
    return titles
