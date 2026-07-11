"""
Management command: embed a raw dataset folder into a .npz file for
later FAISS index construction.

Unlike `build_index` (which operates on ClothingItem database rows),
this command works on a raw image folder — the typical starting point
when you've just downloaded a fashion dataset.

Usage:
    python manage.py embed_dataset /path/to/dataset
    python manage.py embed_dataset /path/to/dataset --output embeddings/my_catalog.npz

The output .npz can later be loaded and fed into FaissManager.add_vectors()
during the index-building milestone.
"""

from django.core.management.base import BaseCommand

from modapp.ml.embedding_pipeline import EmbeddingPipeline


class Command(BaseCommand):
    help = (
        "Scan a dataset folder, embed every image with ResNet50, "
        "and save the embeddings as a .npz file for FAISS."
    )

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "dataset_dir",
            type=str,
            help="Path to the dataset root directory containing images "
            "(optionally organized into category subfolders).",
        )
        parser.add_argument(
            "--output",
            type=str,
            default=None,
            help="Output path for the .npz file. Defaults to "
            "<dataset_dir>/dataset_embeddings.npz.",
        )

    def handle(self, *args, **options) -> None:
        dataset_dir: str = options["dataset_dir"]
        output_path: str | None = options["output"]

        self.stdout.write(
            self.style.MIGRATE_HEADING(
                f"ModaMind Dataset Embedding Pipeline"
            )
        )
        self.stdout.write(f"  Dataset: {dataset_dir}")
        if output_path:
            self.stdout.write(f"  Output:  {output_path}")
        self.stdout.write("")

        def progress_callback(
            current: int, total: int, path: str, success: bool
        ) -> None:
            """Print progress to the management command's stdout."""
            status_icon = self.style.SUCCESS("✓") if success else self.style.ERROR("✗")
            pct = (current / total) * 100
            self.stdout.write(
                f"  [{current:>4d}/{total}] ({pct:5.1f}%) {status_icon} {path}"
            )

        pipeline = EmbeddingPipeline(
            dataset_dir=dataset_dir,
            output_path=output_path,
            progress_callback=progress_callback,
        )

        try:
            result = pipeline.run()
        except (FileNotFoundError, NotADirectoryError) as exc:
            self.stderr.write(self.style.ERROR(f"\nError: {exc}"))
            return

        # --- Summary ---
        self.stdout.write("")
        self.stdout.write(self.style.MIGRATE_HEADING("Pipeline Summary"))
        self.stdout.write(f"  Total images found:     {result.total_images}")
        self.stdout.write(
            self.style.SUCCESS(f"  Successfully embedded:  {result.embedded_count}")
        )

        if result.failed_count:
            self.stdout.write(
                self.style.ERROR(f"  Failed:                 {result.failed_count}")
            )

        if result.skipped_by_scanner:
            self.stdout.write(
                self.style.WARNING(
                    f"  Skipped (non-image):    {result.skipped_by_scanner}"
                )
            )

        if result.categories_found:
            self.stdout.write(
                f"  Categories:             {', '.join(result.categories_found)}"
            )

        self.stdout.write(f"  Time elapsed:           {result.elapsed_seconds:.1f}s")

        if result.output_path:
            self.stdout.write(
                self.style.SUCCESS(f"\n  Output saved to: {result.output_path}")
            )

            # Hint about what to do next
            self.stdout.write(
                self.style.WARNING(
                    "\n  Next step: use the output .npz file to build a FAISS index "
                    "(Milestone 4)."
                )
            )
        else:
            self.stdout.write(
                self.style.ERROR(
                    "\n  No output file was created (all images failed)."
                )
            )
