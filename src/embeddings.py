"""Text embeddings: Amazon Titan on Bedrock, or a pure-Python fallback.

The local backend needs no network or keys: a signed hashing-trick embedding
over word unigrams and bigrams, L2-normalized, deterministic across runs
(md5-based hashing, independent of PYTHONHASHSEED). It is intentionally
simple - good enough to rank "my cat is called Miso" above "the weather is
nice" for the query "what is my cat's name" - so the whole project runs
offline. Both backends emit config.EMBED_DIM dimensions so they share the
VECTOR(256) columns; rows record which model produced them and recall
filters on that, so the two spaces are never compared with each other.

Invoked by: memory_store.py, recall.py.
Inputs: text. Outputs: list[float] of length config.EMBED_DIM.
"""

import hashlib
import json
import math
import re

import config

_WORD_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    """Lowercase word tokens; the shared vocabulary unit for embeddings
    and keyword-overlap scoring."""
    return _WORD_RE.findall(text.lower())


def local_embed(text):
    """Hashed unigram + bigram embedding with sign trick, L2-normalized."""
    dim = config.EMBED_DIM
    vec = [0.0] * dim
    words = tokenize(text)
    terms = list(words)
    terms += [words[i] + "_" + words[i + 1] for i in range(len(words) - 1)]
    for term in terms:
        digest = hashlib.md5(term.encode("utf-8")).digest()
        slot = int.from_bytes(digest[0:4], "big") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vec[slot] += sign
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


def bedrock_embed(text):
    """Amazon Titan Text Embeddings V2 at 256 dimensions, normalized."""
    import boto3
    client = boto3.client("bedrock-runtime", region_name=config.AWS_REGION)
    body = json.dumps({
        "inputText": text[:8000],
        "dimensions": config.EMBED_DIM,
        "normalize": True,
    })
    response = client.invoke_model(
        modelId=config.BEDROCK_EMBED_MODEL, body=body)
    payload = json.loads(response["body"].read())
    return payload["embedding"]


def embed(text):
    """Embed text with the configured backend."""
    if config.EMBED_BACKEND == "bedrock":
        return bedrock_embed(text)
    return local_embed(text)


def model_name():
    """Identifier stored next to each embedding row."""
    if config.EMBED_BACKEND == "bedrock":
        return config.BEDROCK_EMBED_MODEL
    return config.LOCAL_EMBED_MODEL_NAME


def vector_literal(vec):
    """Format a vector as the '[0.1,0.2,...]' literal CockroachDB accepts."""
    return "[" + ",".join("{0:.6f}".format(v) for v in vec) + "]"


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
