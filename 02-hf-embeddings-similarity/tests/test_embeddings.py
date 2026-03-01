"""
test_embeddings.py — TDD tests for the embeddings similarity lab.

Tests are written BEFORE implementation. Each test defines
exactly what the production code must do.

Red -> Green -> Refactor cycle:
  1. Run tests -> they fail (Red)
  2. Write minimum code to pass (Green)
  3. Clean up without breaking tests (Refactor)
"""

import numpy as np
import pytest
from pathlib import Path

from src.embedder import load_notes, load_model, embed_texts, search


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def notes():
    """Load notes once for the entire test module."""
    data_path = Path(__file__).parent.parent / "data" / "notes.json"
    return load_notes(data_path)


@pytest.fixture(scope="module")
def model():
    """Load model once for the entire test module."""
    return load_model()


@pytest.fixture(scope="module")
def embeddings(notes, model):
    """Generate embeddings once for the entire test module."""
    texts = [n["text"] for n in notes]
    return embed_texts(texts, model)


# ── load_notes() ──────────────────────────────────────────────────────────────


def test_load_notes_returns_list(notes):
    """load_notes() should return a list."""
    assert isinstance(notes, list)


def test_load_notes_has_correct_count(notes):
    """load_notes() should return all 10 notes."""
    assert len(notes) == 10


def test_load_notes_items_have_required_keys(notes):
    """Each note should have 'id' and 'text' keys."""
    for note in notes:
        assert "id" in note
        assert "text" in note


# ── load_model() ──────────────────────────────────────────────────────────────


def test_load_model_returns_object(model):
    """load_model() should return a non-None object."""
    assert model is not None


# ── embed_texts() ─────────────────────────────────────────────────────────────


def test_embed_texts_returns_numpy_array(embeddings):
    """embed_texts() should return a numpy array."""
    assert isinstance(embeddings, np.ndarray)


def test_embed_texts_correct_shape(notes, embeddings):
    """Embeddings shape should be (num_texts, embedding_dim)."""
    assert embeddings.shape[0] == len(notes)
    assert embeddings.ndim == 2


def test_embed_texts_are_normalized(embeddings):
    """Each embedding vector should have unit length (L2 norm ≈ 1.0)."""
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


# ── search() ──────────────────────────────────────────────────────────────────


def test_search_returns_list(notes, model, embeddings):
    """search() should return a list."""
    results = search("semantic search", notes, embeddings, model, top_k=3)
    assert isinstance(results, list)


def test_search_returns_correct_count(notes, model, embeddings):
    """search() should return exactly top_k results."""
    results = search("semantic search", notes, embeddings, model, top_k=3)
    assert len(results) == 3


def test_search_results_have_required_keys(notes, model, embeddings):
    """Each result should have 'score' and 'text' keys."""
    results = search("semantic search", notes, embeddings, model, top_k=3)
    for result in results:
        assert "score" in result
        assert "text" in result


def test_search_results_sorted_by_score(notes, model, embeddings):
    """Results should be sorted highest score first."""
    results = search("semantic search", notes, embeddings, model, top_k=5)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_search_semantic_relevance(notes, model, embeddings):
    """A query about embeddings should return embedding-related notes in top 3."""
    results = search(
        "how do embeddings represent text", notes, embeddings, model, top_k=3
    )
    top_texts = " ".join([r["text"] for r in results]).lower()
    assert any(word in top_texts for word in ["embedding", "vector", "semantic"])
