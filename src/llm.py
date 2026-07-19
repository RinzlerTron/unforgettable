"""Pluggable LLM backends: AWS Bedrock, direct Anthropic API, or scripted.

All three implement the same small interface so the agent never knows which
one is running:
  respond(system, messages, bundle)  -> reply text
  extract_facts(text)                -> list of {subject, content, confidence}
  summarize(user_lines)              -> short summary string or None

ScriptedClient (MEM_LLM=off) is fully deterministic and needs no keys or
network: it answers from the recalled MemoryBundle, which exercises the
entire memory pipeline end to end. It exists so judges can run the whole
project - including the chaos demo - with zero credentials.

Invoked by: agent.py, consolidate.py.
"""

import json
import logging
import re

import config
import extract

log = logging.getLogger("llm")

SYSTEM_PROMPT = (
    "You are Unforgettable, an assistant whose entire memory lives in a "
    "CockroachDB cluster. The MEMORY section below was recalled from that "
    "database for this turn. Rely on it: refer to remembered facts "
    "naturally, never claim to be stateless, and keep replies concise. If "
    "the user asks what you remember, answer from the MEMORY section only."
)

_EXTRACT_PROMPT = (
    "Extract durable facts about the user from the message below. Reply "
    "with a JSON array only; each item has keys subject (a short dotted "
    "key like user.name), content (one full sentence), confidence (0-1). "
    "Reply [] if there is nothing durable.\n\nMessage: {0}"
)

_SUMMARY_PROMPT = (
    "Summarize the durable, useful information in these user statements "
    "in at most two sentences. Reply with the summary only.\n\n{0}"
)


def _parse_fact_json(text):
    match = re.search(r"\[.*\]", text, re.S)
    if not match:
        return None
    try:
        items = json.loads(match.group(0))
    except ValueError:
        return None
    facts = []
    for item in items:
        if not isinstance(item, dict):
            continue
        subject = str(item.get("subject", "")).strip()
        content = str(item.get("content", "")).strip()
        try:
            confidence = float(item.get("confidence", 0.7))
        except (TypeError, ValueError):
            confidence = 0.7
        if subject and content:
            facts.append({"subject": subject, "content": content,
                          "confidence": min(max(confidence, 0.0), 1.0)})
    return facts


class _ModelClient:
    """Shared plumbing for the two real-model backends."""

    name = "model"

    def _complete(self, system, messages):
        raise NotImplementedError

    def respond(self, system, messages, bundle=None):
        return self._complete(system, messages)

    def extract_facts(self, text):
        reply = self._complete(
            "You extract structured memory. Reply with JSON only.",
            [{"role": "user", "content": _EXTRACT_PROMPT.format(text)}])
        facts = _parse_fact_json(reply)
        if facts is None:
            log.warning("fact extraction JSON unparseable; using rules")
            return extract.extract_facts(text)
        return facts

    def summarize(self, user_lines):
        joined = "\n".join("- " + line for line in user_lines[:30])
        reply = self._complete(
            "You write terse factual summaries.",
            [{"role": "user", "content": _SUMMARY_PROMPT.format(joined)}])
        reply = reply.strip()
        return reply or None


class BedrockClient(_ModelClient):
    """Claude on AWS Bedrock via the boto3 Converse API."""

    name = "bedrock"

    def __init__(self):
        import boto3
        self._client = boto3.client("bedrock-runtime",
                                    region_name=config.AWS_REGION)

    def _complete(self, system, messages):
        response = self._client.converse(
            modelId=config.BEDROCK_MODEL,
            system=[{"text": system}],
            messages=[{"role": m["role"], "content": [{"text": m["content"]}]}
                      for m in messages],
            inferenceConfig={"maxTokens": config.LLM_MAX_TOKENS},
        )
        parts = response["output"]["message"]["content"]
        return "".join(p.get("text", "") for p in parts)


class AnthropicClient(_ModelClient):
    """Direct Anthropic API via the official SDK."""

    name = "anthropic"

    def __init__(self):
        import anthropic
        self._client = anthropic.Anthropic()

    def _complete(self, system, messages):
        response = self._client.messages.create(
            model=config.ANTHROPIC_MODEL,
            max_tokens=config.LLM_MAX_TOKENS,
            system=system,
            messages=messages,
        )
        return "".join(block.text for block in response.content
                       if block.type == "text")


class ScriptedClient:
    """Deterministic responder driven entirely by recalled memory."""

    name = "scripted"

    def respond(self, system, messages, bundle=None):
        bundle = bundle or {"facts": [], "tasks": [], "episodes": []}
        text = messages[-1]["content"] if messages else ""
        lower = text.lower()

        name_fact = self._find_fact(bundle, "user.name")
        if re.search(r"what('?s| is) my name|who am i\b", lower):
            if name_fact:
                return ("I remember: {0} That memory is a row in "
                        "CockroachDB.".format(name_fact))
            return ("I do not have your name in memory yet. Tell me and I "
                    "will keep it in CockroachDB.")

        if re.search(r"what do you (know|remember)|what have you learned", lower):
            facts = [f["content"] for f in bundle.get("facts", [])[:5]]
            tasks = [t["title"] for t in bundle.get("tasks", [])[:5]]
            if not facts and not tasks:
                return "My memory tables have nothing about you yet."
            lines = ["Here is what my memory holds:"]
            lines += ["- " + f for f in facts]
            lines += ["- Open task: " + t for t in tasks]
            return "\n".join(lines)

        task_titles = extract.extract_tasks(text)
        if task_titles:
            return ("Noted. I stored a task to {0}. It will survive even if "
                    "a database node goes down.".format(task_titles[0]))

        new_facts = extract.extract_facts(text)
        if new_facts:
            return ("Got it. I will remember that ({0}). Stored durably in "
                    "CockroachDB.".format(new_facts[0]["content"].rstrip(".")))

        if "?" in text:
            best = self._best_fact(bundle)
            if best:
                return "Based on what I remember: {0}".format(best)
            return ("I do not have a memory that answers that yet. Tell me "
                    "more and I will remember it.")

        if re.search(r"\b(hi|hello|hey|good (morning|afternoon|evening))\b", lower):
            if name_fact:
                return "Hello again. {0}".format(name_fact)
            return "Hello. I am Unforgettable; everything you tell me is remembered."

        # Nothing matched the conservative patterns: be transparent
        # about what was and was not stored, instead of implying a
        # belief was formed.
        return ("Stored as an episode, but I did not extract a durable "
                "belief from that. In this zero-key scripted mode I only "
                "recognize simple statements like 'my cat is called Miso' "
                "or 'I live in Singapore'; in Bedrock/Anthropic mode "
                "Claude does the extraction and understands free phrasing.")

    def extract_facts(self, text):
        return extract.extract_facts(text)

    def summarize(self, user_lines):
        return None  # consolidate.py falls back to its heuristic summary

    @staticmethod
    def _find_fact(bundle, subject):
        for fact in bundle.get("facts", []):
            if fact.get("subject") == subject:
                return fact["content"]
        return None

    @staticmethod
    def _best_fact(bundle):
        facts = bundle.get("facts", [])
        if not facts:
            return None
        top = facts[0]
        if top.get("score", 0.0) < 0.15:
            return None
        return top["content"]


def get_client():
    if config.LLM_BACKEND == "bedrock":
        return BedrockClient()
    if config.LLM_BACKEND == "anthropic":
        return AnthropicClient()
    return ScriptedClient()
