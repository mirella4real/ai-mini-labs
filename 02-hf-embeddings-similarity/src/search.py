"""
search.py — command-line interface for semantic search over local notes.

Wires together embedder.py functions with embedding cache logic:
- On first run: embeds all notes and saves to disk
- On subsequent runs: loads embeddings from cache (much faster)
"""

import argparse
import sys
from pathlib import Path

from src.embedder import (
    load_notes,
    load_model,
    embed_texts,
    search,
    save_embeddings,
    load_embeddings,
)

DATA_PATH = Path(__file__).parent.parent / "data" / "notes.json"
CACHE_PATH = Path(__file__).parent.parent / "data" / "embeddings.npy"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Semantic search over local notes using embeddings."
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="The search query.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of results to return (default: 3).",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Force recompute embeddings even if cache exists.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load notes
    notes = load_notes(DATA_PATH)
    print(f"Loaded {len(notes)} notes.")

    # Load model
    model = load_model()

    # Load or compute embeddings
    embeddings = None if args.rebuild_cache else load_embeddings(CACHE_PATH)

    if embeddings is None:
        print("Computing embeddings...")
        texts = [n["text"] for n in notes]
        embeddings = embed_texts(texts, model)
        save_embeddings(embeddings, CACHE_PATH)
        print(f"Embeddings cached to {CACHE_PATH}")
    else:
        print("Loaded embeddings from cache.")

    # Search
    results = search(args.query, notes, embeddings, model, top_k=args.top_k)

    # Output
    print(f"\nQuery: {args.query}")
    print("-" * 40)
    for i, result in enumerate(results, 1):
        print(f"{i}. Score: {result['score']:.4f}")
        print(f"   {result['text']}\n")


if __name__ == "__main__":
    sys.exit(main())
