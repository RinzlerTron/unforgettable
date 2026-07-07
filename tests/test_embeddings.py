"""Local embedding backend: shape, determinism, and usefulness."""

import math

import config
import embeddings


def test_dimension_and_normalization():
    vec = embeddings.local_embed("my cat is called Miso")
    assert len(vec) == config.EMBED_DIM
    norm = math.sqrt(sum(v * v for v in vec))
    assert abs(norm - 1.0) < 1e-6


def test_deterministic_across_calls():
    a = embeddings.local_embed("remind me to renew my passport")
    b = embeddings.local_embed("remind me to renew my passport")
    assert a == b


def test_empty_text_is_zero_vector():
    vec = embeddings.local_embed("")
    assert all(v == 0.0 for v in vec)


def test_similar_text_ranks_above_dissimilar():
    query = embeddings.local_embed("what is my cat's name")
    about_cat = embeddings.local_embed("my cat is called Miso")
    about_weather = embeddings.local_embed("the weather is sunny today")
    assert embeddings.cosine_similarity(query, about_cat) > \
        embeddings.cosine_similarity(query, about_weather)


def test_vector_literal_format():
    literal = embeddings.vector_literal([0.5, -0.25, 0.0])
    assert literal.startswith("[") and literal.endswith("]")
    parts = literal[1:-1].split(",")
    assert len(parts) == 3
    assert float(parts[0]) == 0.5
    assert float(parts[1]) == -0.25
