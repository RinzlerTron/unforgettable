"""Time-travel memory: rewind, diff, and explain the agent's beliefs.

Three capabilities, all served by CockroachDB:
  beliefs_at    - the agent's belief state at any past timestamp. Recent
                  moments are read natively with AS OF SYSTEM TIME (a
                  consistent historical snapshot, no locks, served by any
                  node); moments beyond the MVCC garbage-collection window
                  are reconstructed from the append-only version columns
                  (valid_from / superseded_at), which work forever.
  belief_diff   - what changed between two timestamps: beliefs learned,
                  revised (flips and confidence reinforcement, linked
                  through replaces_id), and retired.
  explain_reply - decision audit for any past answer: which memory rows
                  the reply retrieved (recall_traces) and which episode
                  originally taught each of those facts (provenance).

Invoked by: web.py (time-travel panel), chat_cli.py, tests.
Inputs: Database + timestamps / episode ids. Outputs: plain dicts.
"""

import datetime
import logging

import psycopg

log = logging.getLogger("timetravel")

_FACT_COLUMNS = "id, subject, content, confidence, valid_from"


def _iso(value):
    if isinstance(value, str):
        value = datetime.datetime.fromisoformat(value)
    if value.tzinfo is None:
        value = value.replace(tzinfo=datetime.timezone.utc)
    return value.astimezone(datetime.timezone.utc)


def _fact_dicts(rows):
    return [{"id": str(r[0]), "subject": r[1], "content": r[2],
             "confidence": float(r[3]), "valid_from": r[4].isoformat()}
            for r in rows]


def beliefs_at(database, at):
    """Belief state at timestamp `at`; tries AS OF SYSTEM TIME first."""
    at = _iso(at)
    # AS OF SYSTEM TIME requires a constant expression, so the (validated,
    # re-serialized) timestamp is embedded in the SQL text, never raw input.
    aost_sql = (
        "SELECT {0} FROM facts AS OF SYSTEM TIME '{1}'"
        " WHERE superseded_at IS NULL ORDER BY subject, valid_from"
    ).format(_FACT_COLUMNS, at.isoformat())
    try:
        rows = database.execute(aost_sql, fetch="all") or []
        return {"at": at.isoformat(), "mechanism": "as_of_system_time",
                "beliefs": _fact_dicts(rows)}
    except psycopg.Error as error:
        # Timestamp outside the GC window, before the table existed, or in
        # the future: fall back to the append-only reconstruction.
        log.info("AS OF SYSTEM TIME unavailable for %s (%s); "
                 "using bitemporal columns", at.isoformat(), error)
    rows = database.execute(
        "SELECT {0} FROM facts"
        " WHERE valid_from <= %s"
        " AND (superseded_at IS NULL OR superseded_at > %s)"
        " ORDER BY subject, valid_from".format(_FACT_COLUMNS),
        (at, at), fetch="all") or []
    return {"at": at.isoformat(), "mechanism": "bitemporal",
            "beliefs": _fact_dicts(rows)}


def belief_diff(database, start, end):
    """Classify belief changes in (start, end]: learned/revised/retired."""
    start, end = _iso(start), _iso(end)
    rows = database.execute(
        "SELECT id, subject, content, confidence, valid_from,"
        " superseded_at, replaces_id, provenance"
        " FROM facts WHERE valid_from <= %s ORDER BY valid_from",
        (end,), fetch="all") or []

    facts = {}
    for r in rows:
        facts[str(r[0])] = {
            "id": str(r[0]), "subject": r[1], "content": r[2],
            "confidence": float(r[3]), "valid_from": r[4],
            "superseded_at": r[5],
            "replaces_id": str(r[6]) if r[6] else None,
            "provenance": r[7] if isinstance(r[7], dict) else {},
        }

    def active_at(fact, moment):
        if fact["valid_from"] > moment:
            return False
        return fact["superseded_at"] is None or fact["superseded_at"] > moment

    set_start = {fid for fid, f in facts.items() if active_at(f, start)}
    set_end = {fid for fid, f in facts.items() if active_at(f, end)}
    added = set_end - set_start
    retired = set_start - set_end

    def public(fact):
        sources = fact["provenance"].get("sources", [])
        return {"subject": fact["subject"], "content": fact["content"],
                "confidence": fact["confidence"],
                "valid_from": fact["valid_from"].isoformat(),
                "learned_by": sources[-1].get("method") if sources else None}

    learned, revised = [], []
    for fid in sorted(added, key=lambda i: facts[i]["valid_from"]):
        fact = facts[fid]
        old_id = fact["replaces_id"]
        if old_id in retired:
            old = facts[old_id]
            retired.discard(old_id)
            change = "reinforced" \
                if old["content"].lower() == fact["content"].lower() \
                else "changed_belief"
            revised.append({
                "subject": fact["subject"], "change": change,
                "before": {"content": old["content"],
                           "confidence": old["confidence"]},
                "after": {"content": fact["content"],
                          "confidence": fact["confidence"]},
                "learned_by": public(fact)["learned_by"],
            })
        else:
            learned.append(public(fact))

    retired_list = [public(facts[fid]) for fid in
                    sorted(retired, key=lambda i: facts[i]["valid_from"])]
    return {"start": start.isoformat(), "end": end.isoformat(),
            "learned": learned, "revised": revised, "retired": retired_list}


def explain_reply(database, reply_episode_id):
    """Decision audit: why did the agent say what it said?"""
    trace = database.execute(
        "SELECT query, recalled, created_at FROM recall_traces"
        " WHERE reply_episode_id = %s", (reply_episode_id,), fetch="one")
    if trace is None:
        return None
    reply = database.execute(
        "SELECT content, created_at FROM episodes WHERE id = %s",
        (reply_episode_id,), fetch="one")
    query, recalled, traced_at = trace

    used_facts = []
    for item in recalled.get("facts", []):
        fact = database.execute(
            "SELECT subject, content, confidence, provenance FROM facts"
            " WHERE id = %s", (item.get("id"),), fetch="one")
        if fact is None:
            continue
        subject, content, confidence, provenance = fact
        taught_by = None
        sources = provenance.get("sources", []) \
            if isinstance(provenance, dict) else []
        for source in sources:
            episode_id = source.get("episode_id")
            if episode_id:
                row = database.execute(
                    "SELECT content, created_at FROM episodes WHERE id = %s",
                    (episode_id,), fetch="one")
                if row:
                    taught_by = {"episode_id": episode_id,
                                 "content": row[0],
                                 "at": row[1].isoformat()}
                    break
        used_facts.append({
            "subject": subject, "content": content,
            "confidence": float(confidence),
            "score": item.get("score"), "taught_by": taught_by})

    return {
        "reply_episode_id": str(reply_episode_id),
        "reply": reply[0] if reply else None,
        "answered_at": reply[1].isoformat() if reply else None,
        "user_message": query,
        "used_facts": used_facts,
        "used_episodes": recalled.get("episodes", []),
        "used_tasks": recalled.get("tasks", []),
        "traced_at": traced_at.isoformat(),
    }
