# Deployment

Reproduce from a fresh machine. Three paths, fastest first.

## Path 1 - fully local, zero accounts (about 3 minutes)

Prerequisites: Python 3.10+, curl.

    # 1. CockroachDB binary (Linux; see cockroachlabs.com/docs/releases for
    #    macOS/Windows tarballs)
    curl -s https://binaries.cockroachdb.com/cockroach-v25.2.2.linux-amd64.tgz \
      | tar xz
    export PATH="$PWD/cockroach-v25.2.2.linux-amd64:$PATH"

    # 2. Project
    git clone https://github.com/RinzlerTron/unforgettable unforgettable && cd unforgettable
    ./run.sh setup

    # 3. The resilience + time-travel showcase (starts its own 3-node
    #    cluster, needs nothing else)
    ./run.sh chaos

    # 4. Interactive use: start a node, then the web UI
    cockroach start-single-node --insecure \
      --store=path=data/local-store --listen-addr=localhost:26257 \
      --http-addr=localhost:8080 --background
    cockroach sql --insecure -e "CREATE DATABASE IF NOT EXISTS unforgettable"
    ./run.sh web        # http://127.0.0.1:8400

Defaults (`MEM_LLM=off`, `MEM_EMBEDDINGS=local`) need no keys: the scripted
client exercises the full memory pipeline deterministically.

## Path 1b - Docker, one command

Prerequisites: Docker with the compose plugin.

    git clone https://github.com/RinzlerTron/unforgettable unforgettable && cd unforgettable
    docker compose up --build     # agent + one CockroachDB node
                                  # web UI: http://localhost:8400

The recipe is the repo's `Dockerfile` and `compose.yaml` - read them first
if you like; the app container is the same stateless process as Path 1.
Restore the clean seeded demo state at any time:

    docker compose exec app ./run.sh reset-demo

The same image is the AWS deployment artifact: run it on ECS, App Runner,
or EC2 with `MEM_DB_URLS` pointing at CockroachDB Cloud (path 2) and the
Bedrock env vars (path 3); give the task role Bedrock invoke permissions.

## Path 2 - CockroachDB Cloud (the judged configuration)

Prerequisites: ccloud CLI (https://www.cockroachlabs.com/docs/cockroachcloud/ccloud-get-started),
a free CockroachDB Cloud account.

    ccloud auth login
    ./tools/ccloud_deploy.sh          # creates a Basic cluster on AWS +
                                      # SQL user, prints the connection URL
    cp .env.example .env              # set MEM_DB_URLS to the printed URL
    ./run.sh web

The schema (including vector indexes) is applied automatically on first
start. On CockroachDB Cloud the `feature.vector_index.enabled` setting may
not be writable from SQL; if vector indexes cannot be created the app logs
a warning and similarity search transparently falls back to a scan.

Optional, recommended for judging: connect Claude Code or Cursor to the
same cluster through the CockroachDB Cloud managed MCP server (read-only
by default, fully audited) using `tools/mcp_config.example.json` - then ask
your editor to show the `facts` table and watch the agent's beliefs, with
provenance, as plain rows.

## Path 3 - full AWS configuration (Bedrock LLM + embeddings)

Prerequisites: AWS credentials with Bedrock model access enabled for
Anthropic Claude and Amazon Titan Text Embeddings V2 in your region.

    # .env
    MEM_DB_URLS=<CockroachDB Cloud URL from path 2>
    MEM_LLM=bedrock
    MEM_EMBEDDINGS=bedrock
    AWS_REGION=us-east-1
    MEM_BEDROCK_MODEL=us.anthropic.claude-sonnet-4-5-20250929-v1:0

    ./run.sh web

Notes:
- The agent uses the Bedrock Converse API (`bedrock-runtime`), so any
  Claude model id/inference profile enabled in your account works; set
  `MEM_BEDROCK_MODEL` accordingly.
- `MEM_LLM=anthropic` with `ANTHROPIC_API_KEY` is an alternative backend.
- To host the agent itself on AWS, run `./run.sh web` on any EC2 instance
  or container (ECS/App Runner): it is a single stateless uvicorn process;
  give the instance role Bedrock invoke permissions and set the env vars
  above. Bind `MEM_WEB_HOST=0.0.0.0` behind your load balancer.

## Switching embedding backends

Embeddings from different models are never mixed: rows record their
`embedding_model` and recall filters on the active one. If you switch
`MEM_EMBEDDINGS` on an existing database, old rows stop matching vector
search (keyword recall still finds facts); re-ingest or stay consistent.

## Tests

    ./run.sh test

DB-backed tests use, in order: `MEM_TEST_DSN`, a local node on 26257, or a
disposable single node started automatically if `cockroach` is on PATH.
Without any of those, DB tests skip with an explanatory message.
