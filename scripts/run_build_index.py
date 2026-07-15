"""
Standalone script to build a FAISS index from pre-computed embeddings
without Django.

This mirrors the pattern of scripts/run_embedding_pipeline.py: it
imports from modapp.search (which has zero Django dependencies) and
runs the builder directly.  Use this when you want to build a FAISS
index without starting Django or running migrations.

Usage:
    python scripts/run_build_index.py /path/to/dataset_embeddings.npz
    python scripts/run_build_index.py /path/to/embeddings.npz --rebuild
    python scripts/run_build_index.py /path/to/embeddings.npz --index-path custom.index
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

# Make the project root importable (same approach as run_embedding_pipeline.py).
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from modapp.search.index_builder import DatasetIndexBuilder  # noqa: E402

# Configure logging so FaissManager / DatasetIndexBuilder messages are
# visible in the terminal.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-30s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ModaMind — Build a FAISS index from dataset embeddings.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/run_build_index.py ./dataset_embeddings.npz\n"
            "  python scripts/run_build_index.py ./embeddings.npz --rebuild\n"
            "  python scripts/run_build_index.py ./embeddings.npz "
            "--index-path ./custom.index\n"
        ),
    )
    parser.add_argument(
        "npz_path",
        help="Path to the .npz file produced by the embedding pipeline.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Overwrite any existing index at the target path.",
    )
    parser.add_argument(
        "--index-path",
        default=None,
        help="Custom output path for the .index file.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  ModaMind — FAISS Index Builder")
    print("=" * 60)
    print(f"  Source:  {os.path.abspath(args.npz_path)}")
    if args.index_path:
        print(f"  Output:  {os.path.abspath(args.index_path)}")
    if args.rebuild:
        print("  Mode:    REBUILD")
    print()

    builder = DatasetIndexBuilder(
        npz_path=args.npz_path,
        index_path=args.index_path,
        rebuild=args.rebuild,
    )

    try:
        result = builder.build()
    except (FileNotFoundError, ValueError) as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        sys.exit(1)

    # --- Summary ---
    print()
    print("-" * 60)
    print("  Index Build Summary")
    print("-" * 60)

    if result.skipped:
        print(
            f"  Skipped: valid index already exists "
            f"({result.indexed_count} vectors)."
        )
        print("  Use --rebuild to overwrite.")
        return

    print(f"  Embeddings loaded:      {result.total_embeddings}")
    print(f"  Vectors indexed:        {result.indexed_count}")
    print(f"  Embedding dimension:    {result.embedding_dim}")

    if result.categories_found:
        print(
            f"  Categories:             {', '.join(result.categories_found)}"
        )

    print(f"  Time elapsed:           {result.elapsed_seconds:.1f}s")

    if result.index_path:
        print(
            f"\n  Index saved to:    {result.index_path} "
            f"({result.index_size_mb:.2f} MB)"
        )
    if result.metadata_path:
        print(
            f"  Metadata saved to: {result.metadata_path} "
            f"({result.metadata_size_mb:.2f} MB)"
        )


if __name__ == "__main__":
    main()
