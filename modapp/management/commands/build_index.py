"""
Management command: build (or rebuild) the FAISS similarity index from
every ClothingItem currently in the database.

Usage:
    python manage.py build_index            # only embed un-indexed items
    python manage.py build_index --rebuild  # wipe and rebuild from scratch
"""

from django.core.management.base import BaseCommand, CommandError

from modapp.ml.embedding_service import EmbeddingService
from modapp.models import ClothingItem
from modapp.search.faiss_manager import FaissManager


class Command(BaseCommand):
    help = "Embed ClothingItem images with ResNet50 and add them to the FAISS index."

    def add_arguments(self, parser):
        parser.add_argument(
            "--rebuild",
            action="store_true",
            help="Discard the existing index and re-embed every ClothingItem.",
        )

    def handle(self, *args, **options):
        faiss_manager = FaissManager()
        embedder = EmbeddingService()

        if options["rebuild"]:
            self.stdout.write("Rebuilding index from scratch...")
            faiss_manager.reset_index()
            queryset = ClothingItem.objects.all()
            ClothingItem.objects.update(is_indexed=False)
        else:
            queryset = ClothingItem.objects.filter(is_indexed=False)

        total = queryset.count()
        if total == 0:
            self.stdout.write(self.style.WARNING("Nothing to index. All items are already indexed."))
            return

        self.stdout.write(f"Embedding {total} item(s)...")

        indexed_count = 0
        for item in queryset.iterator():
            if not item.image:
                self.stdout.write(self.style.WARNING(f"Skipping item #{item.id}: no image attached."))
                continue

            try:
                with item.image.open("rb") as image_file:
                    embedding = embedder.extract_embedding(image_file.read())
            except Exception as exc:  # noqa: BLE001
                raise CommandError(f"Failed to embed item #{item.id}: {exc}") from exc

            faiss_manager.add_vectors(embedding, ids=[item.id])

            item.is_indexed = True
            item.save(update_fields=["is_indexed"])

            indexed_count += 1
            self.stdout.write(f"  [{indexed_count}/{total}] indexed '{item}'")

        faiss_manager.save_index()

        self.stdout.write(
            self.style.SUCCESS(
                f"Done. Index now contains {faiss_manager.total_vectors} vector(s) "
                f"({indexed_count} newly added)."
            )
        )
