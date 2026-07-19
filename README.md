# Unforgettable

![Python](https://img.shields.io/badge/python-3.10+-blue) ![CockroachDB](https://img.shields.io/badge/CockroachDB-VECTOR%20%7C%20AOST%20%7C%20JSONB-6933ff) ![AWS Bedrock](https://img.shields.io/badge/AWS-Bedrock-orange) ![License](https://img.shields.io/badge/license-MIT-green)

An AI agent whose memory you can rewind, diff, and audit - built for the
CockroachDB x AWS Hackathon: Build with Agentic Memory.

Agents forget, and worse: when they misbehave, nobody can say what they
believed at the moment they acted. Take a wealth-advisory chatbot that
tells a client on Monday their portfolio matches a conservative risk
tolerance, then on Thursday recommends a high-volatility product. The
client complains, and compliance has to reconstruct what the agent
believed about that client's risk profile at each moment - today that
means grepping prompt logs with no ground truth. With Unforgettable it's
`beliefs_at(Monday 10:02)` vs `beliefs_at(Thursday 15:41)`, a `belief_diff`
showing exactly which belief flipped and which message taught it, and
`explain_reply` tracing each answer to the memory rows behind it - under a
minute, from SQL. Systems like MemGPT/Letta, Zep, and mem0 can tell you
what an agent remembers now; none expose a queryable, diffable belief
history that answers what it believed then, what changed, and why -
that history layer is what CockroachDB's `AS OF SYSTEM TIME` and
multi-node resilience make possible, with the diff and audit layer built
on top as this project's contribution. Unforgettable keeps an agent's
entire memory - episodes, versioned beliefs, tasks, and a per-reply
decision audit - as SQL rows in CockroachDB, so you can time-travel to
any past belief state (`AS OF SYSTEM TIME` + append-only versions), diff
exactly which belief flipped between two answers, trace any reply to the
message that taught it, and kill a database node mid-conversation without
losing a thing.

## Try It (For Judges)

Running locally in about 3 minutes, zero accounts or keys:

    curl -s https://binaries.cockroachdb.com/cockroach-v25.2.2.linux-amd64.tgz | tar xz
    export PATH="$PWD/cockroach-v25.2.2.linux-amd64:$PATH"
    git clone <repo-url> unforgettable && cd unforgettable
    ./run.sh setup
    ./run.sh chaos      # 3-node cluster, conversation, node kill, verdict

Then interactively (`./run.sh web`, after starting a node per
[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)) try:

1. Tell it things: "My name is Priya, I live in Singapore, my cat is
   called Miso. Remind me to renew my passport." Then ask "what do you
   remember about me?"
2. Change your mind ("Actually I moved to Chennai"), then use the
   time-travel panel: beliefs at a minute ago vs now, and the diff showing
   the location belief flip.
3. Click any agent reply to see its decision audit: which memory rows it
   used and which message taught each fact.

Manual fallback: `./run.sh test` (44 tests, self-starts a disposable
CockroachDB node). Default mode is `MEM_LLM=off` - a deterministic
scripted client that exercises the full memory pipeline; set
`MEM_LLM=bedrock` for Claude on AWS Bedrock (see docs/DEPLOYMENT.md).

## Architecture

    Agent (FastAPI/CLI, stateless) --- SQL ---> CockroachDB cluster
      recall -> respond -> remember            episodes | facts (append-only
      Bedrock Claude / Anthropic / scripted    versions) | tasks | recall_traces
      time-travel API                          VECTOR indexes, JSONB, AOST
                 |
                 +--> AWS Bedrock (Claude Converse API, Titan embeddings)

Detail in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Results

| Metric | Value |
|---|---|
| Node killed mid-conversation | 0 memories lost; failover on the next statement (chaos demo asserts it) |
| Time travel | Belief state at any past moment; AS OF SYSTEM TIME inside the GC window, bitemporal columns forever |
| Decision audit | 100% of replies traceable to the exact memory rows used and the messages that taught them |
| Test suite | 44 tests against a real CockroachDB node |
| Keys needed for the full demo | none (scripted mode + local embeddings) |

## Challenge Compliance

| Requirement / criterion | Where this submission delivers it |
|---|---|
| CockroachDB as persistent memory layer | All memory is SQL rows: `episodes`, `facts`, `tasks`, `recall_traces` (src/schema.sql); the agent process is stateless |
| CockroachDB tool 1: Distributed Vector Indexing | `VECTOR(256)` columns + `CREATE VECTOR INDEX` on episodes and facts; every recall runs `ORDER BY embedding <=> query` in-database (src/db.py, src/recall.py) |
| CockroachDB tool 2: ccloud CLI | `tools/ccloud_deploy.sh` provisions the cluster, SQL user, and connection URL end to end |
| CockroachDB tool 3: Managed MCP Server | `tools/mcp_config.example.json` connects Claude Code/Cursor to the memory cluster read-only, so judges can inspect the agent's beliefs live |
| AWS service: Amazon Bedrock | Claude via the Converse API + Titan Text Embeddings V2 (src/llm.py, src/embeddings.py); agent hosts on EC2/ECS (docs/DEPLOYMENT.md) |
| Agentic Memory Design | Episodic + semantic + task memory with confidence and provenance; consolidation job distills episodes into beliefs; append-only versioning makes belief history first-class |
| Technical Implementation | Native AS OF SYSTEM TIME, vector indexes, JSONB provenance, SQLSTATE 40001 retries, multi-node client failover, transactional versioning |
| Real-World Impact | Memory audit and debugging: "what did the agent believe when it did that, and why" - the missing tool for agents in production |
| Production Readiness | Chaos-tested node failure, retry/backoff, validation before persist, schema bootstrap idempotent, secrets via env only, 44 tests |
| Creativity & Originality | Time-travel memory: rewind, belief diff, and per-reply decision audit built directly on CockroachDB primitives |

## Limitations

Deliberate scope decisions, not gaps we missed:

- Multi-valued subjects are a fixed allowlist (`user.preference`,
  `user.health`, `conversation.summary`); a second "my brother is X" fact
  supersedes rather than coexists. Widening this to arbitrary multi-valued
  facts is a schema change, not a redesign.
- No confidence decay over time - a belief learned a year ago keeps its
  confidence until reinforced or contradicted. Decay is on the roadmap
  but was cut to keep versioning semantics provable in the time
  available.
- In zero-key scripted mode, fact extraction uses deliberately conservative
  rule-based patterns; full natural-language extraction needs the Bedrock
  or Anthropic mode, where Claude does the extraction instead of regex.

## Project structure

    src/
    ├── config.py        # paths, env, tunables (imported by all)
    ├── schema.sql       # memory tables            -> applied by db.py
    ├── db.py            # failover + retries       -> everything below
    ├── embeddings.py    # Titan / local embeddings -> memory_store, recall
    ├── extract.py       # rule-based facts/tasks   -> agent, consolidate
    ├── memory_store.py  # validated, versioned writes
    ├── recall.py        # vector + keyword + recency retrieval
    ├── timetravel.py    # AOST reads, belief diff, decision audit
    ├── consolidate.py   # episodes -> facts job
    ├── llm.py           # bedrock | anthropic | scripted
    ├── agent.py         # the turn loop            -> cli, web, chaos
    ├── chat_cli.py      # terminal chat
    └── web.py           # web chat + time-travel console
    tools/               # chaos_demo.py, ccloud_deploy.sh, MCP config
    tests/               # 44 tests + fixtures (real saved inputs)
    docs/                # ARCHITECTURE.md, DEPLOYMENT.md

## Author

Sanjay - CockroachDB x AWS Hackathon: Build with Agentic Memory, 2026
