#!/usr/bin/env bash
# Front door for Unforgettable. One command per job:
#   ./run.sh setup        create venv + install pinned dependencies
#   ./run.sh cli          terminal chat (needs a reachable CockroachDB)
#   ./run.sh web          web chat UI on http://127.0.0.1:8400
#   ./run.sh chaos        3-node cluster + kill-a-node resilience demo
#   ./run.sh consolidate  run the memory consolidation job once
#   ./run.sh reset-demo   wipe all memory, reseed the clean demo state
#   ./run.sh test         run the test suite
#
# Inputs: .env (optional; see .env.example). Outputs: see each command.

set -euo pipefail
cd "$(dirname "$0")"

PY="python3"
if [ -x "venv/bin/python" ]; then PY="venv/bin/python"; fi
export PYTHONPATH="$(pwd)/src"

case "${1:-help}" in
  setup)
    python3 -m venv venv
    venv/bin/pip install -r requirements.txt
    echo "Done. Optional: cp .env.example .env and edit."
    ;;
  cli)
    exec "$PY" src/chat_cli.py
    ;;
  web)
    exec "$PY" src/web.py
    ;;
  chaos)
    exec "$PY" tools/chaos_demo.py "${@:2}"
    ;;
  consolidate)
    exec "$PY" -c "import config, db, consolidate; config.setup_logging(); \
d = db.Database(); d.ensure_schema(); \
print('facts written:', consolidate.consolidate(d))"
    ;;
  reset-demo)
    # For a public demo URL: visitors can teach the agent things, so this
    # wipes every memory table and reseeds the clean demo conversation.
    exec "$PY" - <<'EOF'
import config, db
from agent import Agent
config.setup_logging()
d = db.Database()
d.ensure_schema()
for table in ("recall_traces", "tasks", "facts", "episodes", "conversations"):
    d.execute("TRUNCATE TABLE {0} CASCADE".format(table))
agent = Agent(database=d)
conversation_id = agent.new_conversation("demo")
for line in (
    "Hi there. My name is Priya Nair and I live in Singapore.",
    "I work at Meridian Health. My risk tolerance is conservative.",
    "Remind me to review the fund disclosure documents.",
):
    agent.turn(conversation_id, line)
counts = agent.store.counts()
print("Demo reset: {0} episodes, {1} beliefs, {2} open task(s) seeded.".format(
    counts["episodes"], counts["facts"], counts["open_tasks"]))
EOF
    ;;
  test)
    exec "$PY" -m pytest tests/ -v "${@:2}"
    ;;
  *)
    grep '^#   ' "$0" | sed 's/^#   //'
    ;;
esac
