"""
Standalone script to run the ModaMind dataset embedding pipeline
without Django.

This mirrors the pattern of scripts/test_pipeline.py: it imports from
modapp.ml (which has zero Django dependencies) and runs the pipeline
directly. Use this when you want to process a dataset folder without
starting the Django server or running migrations.

Usage:
    python scripts/run_embedding_pipeline.py /path/to/dataset
    python scripts/run_embedding_pipeline.py /path/to/dataset --output /path/to/output.npz
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

# Make the project root importable (same approach as test_pipeline.py).
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from modapp.ml.embedding_pipeline import EmbeddingPipeline  # noqa: E402

# Configure logging so EmbeddingService / DatasetScanner / pipeline
# messages are visible in the terminal.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-30s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)


def cli_progress(current: int, total: int, path: str, success: bool) -> None:
    """Simple CLI progress indicator."""
    icon = "✓" if success else "✗"
    pct = (current / total) * 100
    print(f"  [{current:>4d}/{total}] ({pct:5.1f}%) {icon} {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ModaMind — Embed a dataset folder using ResNet50.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/run_embedding_pipeline.py ./test_images\n"
            "  python scripts/run_embedding_pipeline.py /data/fashion --output ./embeddings/fashion.npz\n"
        ),
    )
    parser.add_argument(
        "dataset_dir",
        help="Path to the dataset directory containing images.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output .npz file path. Default: <dataset_dir>/dataset_embeddings.npz",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  ModaMind — Dataset Embedding Pipeline")
    print("=" * 60)
    print(f"  Dataset:  {os.path.abspath(args.dataset_dir)}")
    if args.output:
        print(f"  Output:   {os.path.abspath(args.output)}")
    print()

    pipeline = EmbeddingPipeline(
        dataset_dir=args.dataset_dir,
        output_path=args.output,
        progress_callback=cli_progress,
    )

    try:
        result = pipeline.run()
    except (FileNotFoundError, NotADirectoryError) as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        sys.exit(1)

    # --- Summary ---
    print()
    print("-" * 60)
    print("  Pipeline Summary")
    print("-" * 60)
    print(f"  Total images found:     {result.total_images}")
    print(f"  Successfully embedded:  {result.embedded_count}")

    if result.failed_count:
        print(f"  Failed:                 {result.failed_count}")

    if result.skipped_by_scanner:
        print(f"  Skipped (non-image):    {result.skipped_by_scanner}")

    if result.categories_found:
        print(f"  Categories:             {', '.join(result.categories_found)}")

    print(f"  Time elapsed:           {result.elapsed_seconds:.1f}s")

    if result.output_path:
        print(f"\n  Output saved to: {result.output_path}")
    else:
        print("\n  No output file was created (all images failed).")
        sys.exit(1)


if __name__ == "__main__":
    main()
