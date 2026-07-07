#!/usr/bin/env bash
# Front door for Unforgettable. One command per job:
#   ./run.sh setup        create venv + install pinned dependencies
#   ./run.sh cli          terminal chat (needs a reachable CockroachDB)
#   ./run.sh web          web chat UI on http://127.0.0.1:8400
#   ./run.sh chaos        3-node cluster + kill-a-node resilience demo
#   ./run.sh consolidate  run the memory consolidation job once
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
  test)
    exec "$PY" -m pytest tests/ -v "${@:2}"
    ;;
  *)
    grep '^#   ' "$0" | sed 's/^#   //'
    ;;
esac
