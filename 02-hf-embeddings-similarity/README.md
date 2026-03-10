# Lab 02 — Embeddings Similarity

Build a semantic search tool over a local set of notes using HuggingFace
sentence-transformers. No API keys, no vector database, no cloud.

## What This Demonstrates

- Generating text embeddings with sentence-transformers
- Cosine similarity for semantic ranking
- Embedding cache to disk for performance
- TDD (Red → Green → Refactor) development workflow
- Pure functions for testability and composability

## Architecture
```
src/
  embedder.py  — load_notes, load_model, embed_texts, search, save/load cache
  search.py    — CLI that wires embedder functions with cache logic
data/
  notes.json        — 10 engineering notes (source of truth)
  embeddings.npy    — cached embeddings (generated, not committed)
tests/
  test_embeddings.py — 14 tests written before implementation (TDD)
```

## How to Run
```bash
# From the repo root
cd 02-hf-embeddings-similarity
uv pip install -r requirements.txt

# First run — computes and caches embeddings
python -m src.search --query "how does semantic search work" --top-k 3

# Subsequent runs — loads from cache (faster)
python -m src.search --query "how does semantic search work" --top-k 3

# Force recompute embeddings (use when notes.json changes)
python -m src.search --query "distributed systems" --rebuild-cache
```

## How Semantic Search Works

Traditional keyword search matches exact words. Semantic search matches
meaning. The model converts text into a vector of 384 numbers — an
embedding — that captures semantic content. Similar meanings produce
similar vectors.

Similarity is measured with cosine similarity: the angle between two
vectors. A score of 1.0 means identical meaning, 0.0 means unrelated.

Because embeddings are normalized (unit length), cosine similarity
reduces to a simple dot product — fast and clean.

## Model Choice

`sentence-transformers/all-MiniLM-L6-v2` — small (~90MB), fast, and
effective. Produces 384-dimensional embeddings. Ideal for semantic search
on a laptop.

## TDD Approach

Tests were written before implementation following Red → Green → Refactor:

- 🔴 Red — 14 tests written against non-existent code
- 🟢 Green — minimum implementation to pass all tests
- 🔵 Refactor — improved clarity and documentation, tests still passing

## Running Tests
```bash
cd 02-hf-embeddings-similarity
python -m pytest tests/ -v
```

14 tests covering: note loading, model loading, embedding shape and
normalization, search ranking, semantic relevance, and cache roundtrip.