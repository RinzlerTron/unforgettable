"""Terminal chat front-end for the agent, with time-travel commands.

Commands inside the chat:
  /status        - connected node, backend, memory row counts
  /facts         - current beliefs (active semantic memory)
  /tasks         - open task state
  /beliefs N     - what the agent believed N minutes ago
  /diff N [M]    - belief changes between N and M minutes ago (M default 0)
  /why           - decision audit for the last reply
  /new           - start a fresh conversation (memory persists across it)
  /quit          - exit

Invoked by: ./run.sh cli  (or: python src/chat_cli.py).
Inputs: stdin. Outputs: stdout.
"""

import datetime
import sys

import config
import timetravel
from agent import Agent


def _minutes_ago(minutes):
    return (datetime.datetime.now(datetime.timezone.utc)
            - datetime.timedelta(minutes=minutes))


def print_status(agent):
    status = agent.status()
    print("node id      : {0}".format(status["node_id"]))
    print("connected to : {0}".format(status["url"]))
    print("llm backend  : {0}".format(status["llm_backend"]))
    print("embeddings   : {0}".format(status["embedding_backend"]))
    print("memory rows  : {0} episodes, {1} beliefs ({2} versions), "
          "{3} open tasks".format(
              status["counts"]["episodes"], status["counts"]["facts"],
              status["counts"]["fact_versions"],
              status["counts"]["open_tasks"]))


def print_facts(agent):
    rows = agent.db.execute(
        "SELECT subject, content, confidence FROM facts"
        " WHERE superseded_at IS NULL ORDER BY valid_from DESC LIMIT 30",
        fetch="all") or []
    if not rows:
        print("(no beliefs yet)")
    for subject, content, confidence in rows:
        print("[{0:.2f}] {1}: {2}".format(confidence, subject, content))


def print_tasks(agent):
    tasks = agent.store.open_tasks()
    if not tasks:
        print("(no open tasks)")
    for task in tasks:
        print("- {0}".format(task["title"]))


def print_beliefs(agent, minutes):
    snapshot = timetravel.beliefs_at(agent.db, _minutes_ago(minutes))
    print("beliefs {0} minute(s) ago (via {1}):".format(
        minutes, snapshot["mechanism"]))
    if not snapshot["beliefs"]:
        print("(the agent believed nothing at that moment)")
    for belief in snapshot["beliefs"]:
        print("[{0:.2f}] {1}".format(belief["confidence"], belief["content"]))


def print_diff(agent, start_minutes, end_minutes):
    diff = timetravel.belief_diff(
        agent.db, _minutes_ago(start_minutes), _minutes_ago(end_minutes))
    for fact in diff["learned"]:
        print("learned : {0}".format(fact["content"]))
    for change in diff["revised"]:
        print("{0}: '{1}' ({2:.2f}) -> '{3}' ({4:.2f})".format(
            change["change"], change["before"]["content"],
            change["before"]["confidence"], change["after"]["content"],
            change["after"]["confidence"]))
    for fact in diff["retired"]:
        print("retired : {0}".format(fact["content"]))
    if not (diff["learned"] or diff["revised"] or diff["retired"]):
        print("(no beliefs changed in that window)")


def print_why(agent, reply_episode_id):
    if not reply_episode_id:
        print("(no reply yet in this session)")
        return
    audit = timetravel.explain_reply(agent.db, reply_episode_id)
    if audit is None:
        print("(no recall trace for the last reply)")
        return
    print("asked  : {0}".format(audit["user_message"]))
    print("reply  : {0}".format(audit["reply"]))
    for fact in audit["used_facts"]:
        print("used   : {0}".format(fact["content"]))
        if fact["taught_by"]:
            print("         learned from: '{0}' at {1}".format(
                fact["taught_by"]["content"], fact["taught_by"]["at"]))
    for episode in audit["used_episodes"]:
        print("used past moment: {0}".format(episode["content"]))
    if not audit["used_facts"] and not audit["used_episodes"]:
        print("(the reply used no long-term memory)")


def main():
    config.setup_logging()
    agent = Agent()
    conversation_id = agent.new_conversation(title="cli session")
    last_reply_episode = None
    print("Unforgettable - memory lives in CockroachDB "
          "({0} node URL(s) configured)".format(len(agent.db.urls)))
    print("Commands: /status /facts /tasks /beliefs N /diff N [M] /why "
          "/new /quit\n")

    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        parts = text.split()
        command = parts[0]
        if command in ("/quit", "/exit"):
            break
        if command == "/status":
            print_status(agent)
        elif command == "/facts":
            print_facts(agent)
        elif command == "/tasks":
            print_tasks(agent)
        elif command == "/beliefs":
            print_beliefs(agent, float(parts[1]) if len(parts) > 1 else 5.0)
        elif command == "/diff":
            start = float(parts[1]) if len(parts) > 1 else 5.0
            end = float(parts[2]) if len(parts) > 2 else 0.0
            print_diff(agent, start, end)
        elif command == "/why":
            print_why(agent, last_reply_episode)
        elif command == "/new":
            conversation_id = agent.new_conversation(title="cli session")
            print("(new conversation started; long-term memory kept)")
        else:
            result = agent.turn(conversation_id, text)
            last_reply_episode = result["reply_episode_id"]
            print("agent> {0}\n".format(result["reply"]))


if __name__ == "__main__":
    sys.exit(main())
