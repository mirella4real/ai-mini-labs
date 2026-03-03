"""
embedder.py — loads notes, generates embeddings, and performs semantic search.

Uses sentence-transformers for embedding and cosine similarity for ranking.
Kept as pure functions (no classes) for simplicity and testability.

Design decisions:
- normalize_embeddings=True enables cosine similarity via dot product
- Pure functions (no classes) make each step independently testable
- Embedding dimension is 384 for all-MiniLM-L6-v2
"""

import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 output dimension


def load_notes(path: Path) -> list[dict]:
    """
    Load notes from a JSON file.
    Returns a list of dicts with 'id' and 'text' keys.
    """
    with open(path, "r") as f:
        return json.load(f)


def load_model(model_name: str = MODEL_NAME) -> SentenceTransformer:
    """
    Load the sentence-transformers embedding model.
    Downloads and caches locally on first run (~90MB).
    """
    print(f"Loading embedding model: {model_name} ...")
    model = SentenceTransformer(model_name)
    print("Model loaded.")
    return model


def embed_texts(texts: list[str], model: SentenceTransformer) -> np.ndarray:
    """
    Convert a list of strings into normalized embedding vectors.

    normalize_embeddings=True ensures unit length vectors (L2 norm = 1.0),
    which makes cosine similarity equivalent to a simple dot product.

    Returns numpy array of shape (num_texts, EMBEDDING_DIM).
    """
    return model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )


def search(
    query: str,
    notes: list[dict],
    embeddings: np.ndarray,
    model: SentenceTransformer,
    top_k: int = 3,
) -> list[dict]:
    """
    Find the top-k most semantically similar notes to a query.

    Cosine similarity = dot product when vectors are normalized.
    Returns list of dicts with 'score', 'id', and 'text',
    sorted by score descending (most similar first).
    """
    query_embedding = embed_texts([query], model)[0]
    scores = np.dot(embeddings, query_embedding)
    top_indices = np.argsort(scores)[::-1][:top_k]

    return [
        {
            "score": float(scores[idx]),
            "id": notes[idx]["id"],
            "text": notes[idx]["text"],
        }
        for idx in top_indices
    ]
