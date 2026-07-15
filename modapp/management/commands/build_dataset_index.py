"""
Management command: build a FAISS index from a .npz embedding file
produced by the dataset embedding pipeline.

This is the counterpart to ``embed_dataset`` — after embedding a raw
dataset folder into a .npz file, run this command to build a
searchable FAISS index from those embeddings.

Usage:
    python manage.py build_dataset_index /path/to/dataset_embeddings.npz
    python manage.py build_dataset_index /path/to/embeddings.npz --rebuild
    python manage.py build_dataset_index /path/to/embeddings.npz --index-path custom/path.index
"""

from django.core.management.base import BaseCommand

from modapp.search.index_builder import DatasetIndexBuilder


class Command(BaseCommand):
    help = (
        "Build a persistent FAISS index from a .npz embedding file "
        "produced by the embed_dataset pipeline."
    )

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "npz_path",
            type=str,
            help="Path to the .npz file produced by embed_dataset.",
        )
        parser.add_argument(
            "--rebuild",
            action="store_true",
            help="Overwrite any existing index at the target path.",
        )
        parser.add_argument(
            "--index-path",
            type=str,
            default=None,
            help="Custom output path for the .index file. "
            "Default: modapp/search/indexes/fashion_dataset.index.",
        )

    def handle(self, *args, **options) -> None:
        npz_path: str = options["npz_path"]
        rebuild: bool = options["rebuild"]
        index_path: str | None = options["index_path"]

        self.stdout.write(
            self.style.MIGRATE_HEADING("ModaMind FAISS Index Builder")
        )
        self.stdout.write(f"  Source:   {npz_path}")
        if index_path:
            self.stdout.write(f"  Output:   {index_path}")
        if rebuild:
            self.stdout.write(
                self.style.WARNING(
                    "  Mode:     REBUILD (existing index will be overwritten)"
                )
            )
        self.stdout.write("")

        builder = DatasetIndexBuilder(
            npz_path=npz_path,
            index_path=index_path,
            rebuild=rebuild,
        )

        try:
            result = builder.build()
        except (FileNotFoundError, ValueError) as exc:
            self.stderr.write(self.style.ERROR(f"\nError: {exc}"))
            return

        # --- Summary ---
        self.stdout.write("")
        self.stdout.write(self.style.MIGRATE_HEADING("Index Build Summary"))

        if result.skipped:
            self.stdout.write(
                self.style.WARNING(
                    f"  Skipped: valid index already exists with "
                    f"{result.indexed_count} vectors."
                )
            )
            self.stdout.write(
                self.style.WARNING("  Use --rebuild to overwrite.")
            )
            return

        self.stdout.write(
            f"  Embeddings loaded:      {result.total_embeddings}"
        )
        self.stdout.write(
            self.style.SUCCESS(
                f"  Vectors indexed:        {result.indexed_count}"
            )
        )
        self.stdout.write(
            f"  Embedding dimension:    {result.embedding_dim}"
        )

        if result.categories_found:
            self.stdout.write(
                f"  Categories:             "
                f"{', '.join(result.categories_found)}"
            )

        self.stdout.write(
            f"  Time elapsed:           {result.elapsed_seconds:.1f}s"
        )

        self.stdout.write(
            self.style.SUCCESS(
                f"\n  Index saved to:    {result.index_path} "
                f"({result.index_size_mb:.2f} MB)"
            )
        )
        self.stdout.write(
            self.style.SUCCESS(
                f"  Metadata saved to: {result.metadata_path} "
                f"({result.metadata_size_mb:.2f} MB)"
            )
        )

        # Hint about next step
        self.stdout.write(
            self.style.WARNING(
                "\n  Next step: implement the Similarity Search Service "
                "(Milestone 5)."
            )
        )
