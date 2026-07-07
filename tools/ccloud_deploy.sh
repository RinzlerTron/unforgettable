#!/usr/bin/env bash
# Provision a CockroachDB Cloud cluster for Unforgettable with the ccloud CLI
# (the agent-ready CLI: noun-verb commands, JSON output, service accounts).
#
# Prerequisites:
#   - ccloud CLI installed: https://www.cockroachlabs.com/docs/cockroachcloud/ccloud-get-started
#   - authenticated once:   ccloud auth login
#
# What it does:
#   1. creates a Basic (serverless) cluster on AWS,
#   2. creates a SQL user for the agent,
#   3. prints the connection URL to export as MEM_DB_URLS.
#
# Invoked by: a human (or an agent) deploying to CockroachDB Cloud.
# Inputs: CLUSTER_NAME / REGION / SQL_USER env vars (defaults below).
# Outputs: connection URL on stdout.

set -euo pipefail

CLUSTER_NAME="${CLUSTER_NAME:-unforgettable}"
REGION="${REGION:-us-east-1}"
SQL_USER="${SQL_USER:-memory_agent}"

echo "== Creating Basic cluster '${CLUSTER_NAME}' on AWS ${REGION}"
ccloud cluster create basic "${CLUSTER_NAME}" \
    --cloud AWS \
    --regions "${REGION}"

echo "== Creating SQL user '${SQL_USER}' (password will be printed once)"
ccloud cluster user create "${CLUSTER_NAME}" "${SQL_USER}"

echo "== Connection URL (put this in .env as MEM_DB_URLS)"
ccloud cluster sql "${CLUSTER_NAME}" --connection-url

cat <<'NOTES'

Next steps:
  1. Set MEM_DB_URLS in .env to the URL above (add the password, and change
     the database name to 'unforgettable' after: ccloud cluster sql <name>
     then CREATE DATABASE unforgettable;).
  2. The schema is applied automatically on first agent start.
  3. Optional: connect Claude Code or Cursor to the same cluster read-only
     through the managed MCP server - see tools/mcp_config.example.json.
NOTES
