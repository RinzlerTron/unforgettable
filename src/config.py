"""Single configuration module: paths, tunables, env vars, logging.

Every other module imports from here; nothing else reads os.environ for
project settings. All paths are anchored to the project root so the code
runs identically from any working directory.

Invoked by: every module in src/ and tools/.
Inputs: environment variables (optionally loaded from a .env file).
Outputs: module-level constants and setup_logging().
"""

import logging
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
DATA_DIR = ROOT / "data"
SCHEMA_PATH = SRC_DIR / "schema.sql"

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass


def _env(name, default):
    value = os.environ.get(name, "").strip()
    return value if value else default


# --- Database -------------------------------------------------------------
# Comma-separated connection URLs. More than one URL enables client-side
# failover: when the current node dies mid-conversation, db.py reconnects
# to the next URL and retries. The chaos demo passes all three node URLs.
DB_URLS = [u.strip() for u in _env(
    "MEM_DB_URLS",
    "postgresql://root@localhost:26257/unforgettable?sslmode=disable",
).split(",") if u.strip()]

DB_CONNECT_TIMEOUT_SECONDS = 3
DB_MAX_RETRIES = 6          # per statement, across failover + 40001 retries
DB_RETRY_BASE_SLEEP = 0.25  # exponential backoff base, seconds

# --- Embeddings -----------------------------------------------------------
EMBED_DIM = 256
EMBED_BACKEND = _env("MEM_EMBEDDINGS", "local")            # local | bedrock
LOCAL_EMBED_MODEL_NAME = "local-hash-v1"
BEDROCK_EMBED_MODEL = _env("MEM_BEDROCK_EMBED_MODEL", "amazon.titan-embed-text-v2:0")

# --- LLM ------------------------------------------------------------------
LLM_BACKEND = _env("MEM_LLM", "off")                       # off | bedrock | anthropic
AWS_REGION = _env("AWS_REGION", "us-east-1")
BEDROCK_MODEL = _env("MEM_BEDROCK_MODEL", "us.anthropic.claude-sonnet-4-5-20250929-v1:0")
ANTHROPIC_MODEL = _env("MEM_ANTHROPIC_MODEL", "claude-opus-4-8")
LLM_MAX_TOKENS = int(_env("MEM_LLM_MAX_TOKENS", "1024"))

# --- Recall ranking -------------------------------------------------------
# Final score = W_SIMILARITY * cosine_similarity
#             + W_RECENCY   * exp(-age_hours / RECENCY_HALF_LIFE_HOURS)
#             + W_KEYWORD   * keyword_overlap
W_SIMILARITY = 0.6
W_RECENCY = 0.2
W_KEYWORD = 0.2
RECENCY_HALF_LIFE_HOURS = 72.0

RECALL_EPISODES = 4      # similar past episodes injected per turn
RECALL_FACTS = 6         # relevant facts injected per turn
RECENT_TURNS = 8         # verbatim tail of the current conversation
VECTOR_CANDIDATES = 24   # rows fetched by vector search before re-ranking

# --- Consolidation --------------------------------------------------------
CONSOLIDATE_MIN_AGE_MINUTES = int(_env("MEM_CONSOLIDATE_MIN_AGE_MINUTES", "30"))
CONSOLIDATE_BATCH = 200

# --- Web UI ---------------------------------------------------------------
WEB_HOST = _env("MEM_WEB_HOST", "127.0.0.1")
WEB_PORT = int(_env("MEM_WEB_PORT", "8400"))


def setup_logging(level=logging.INFO):
    """Configure root logging once, idempotently."""
    root = logging.getLogger()
    if root.handlers:
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
