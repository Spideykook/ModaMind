"""
modapp/models.py — ModaMind relational database layer (Phase 3).

Three models, three layers of responsibility:

  Category          — normalised controlled vocabulary of garment types.
  ClothingItem      — one catalog entry, one image, one FAISS vector slot.
                      pk == FAISS vector id (IndexIDMap2 binding).
  EmbeddingMetadata — audit record created by build_index recording which
                      model version produced each stored vector.
"""

from __future__ import annotations

from django.conf import settings
from django.core.exceptions import ValidationError
from django.core.validators import FileExtensionValidator
from django.db import models


# ---------------------------------------------------------------------------
# Shared validators
# ---------------------------------------------------------------------------

def validate_image_file_size(file) -> None:
    """Reject uploads that exceed settings.MAX_UPLOAD_SIZE_BYTES."""
    if file.size > settings.MAX_UPLOAD_SIZE_BYTES:
        max_mb = settings.MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)
        raise ValidationError(
            f"Image file too large. Maximum allowed size is {max_mb} MB."
        )


# ---------------------------------------------------------------------------
# Category
# ---------------------------------------------------------------------------

class Category(models.Model):
    """
    Garment category, normalised into its own table so new categories can be
    added at runtime (via seed_catalog or admin) without a migration.
    """

    slug = models.SlugField(
        max_length=64,
        unique=True,
        help_text="Machine key, e.g. 'tops'. Inferred from dataset folder name.",
    )
    name = models.CharField(
        max_length=100,
        help_text="Human label, e.g. 'Tops'.",
    )

    class Meta:
        verbose_name = "Category"
        verbose_name_plural = "Categories"
        ordering = ["name"]
        indexes = [
            models.Index(fields=["slug"], name="category_slug_idx"),
        ]

    def __str__(self) -> str:
        return self.name

    def save(self, *args, **kwargs) -> None:
        if self.name and self.name == self.name.lower():
            self.name = self.name.title()
        super().save(*args, **kwargs)


# ---------------------------------------------------------------------------
# ClothingItem
# ---------------------------------------------------------------------------

class ClothingItem(models.Model):
    """
    A single catalog entry.  pk is deliberately used as the FAISS vector id
    (IndexIDMap2) so FAISS results resolve to a DB row via a single pk lookup.
    """

    name = models.CharField(
        max_length=255,
        blank=True,
        default="",
        help_text="Display name, e.g. 'Oversized Denim Jacket'.",
    )
    category = models.ForeignKey(
        Category,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="clothing_items",
        help_text="Garment category. NULL = uncategorised.",
    )
    brand = models.CharField(
        max_length=100,
        blank=True,
        default="",
        db_index=True,
    )
    color = models.CharField(
        max_length=50,
        blank=True,
        default="",
        db_index=True,
    )
    image = models.ImageField(
        upload_to="clothing_images/",
        validators=[
            FileExtensionValidator(
                allowed_extensions=settings.ALLOWED_IMAGE_EXTENSIONS
            ),
            validate_image_file_size,
        ],
        help_text="Catalog photo used to generate the ResNet50 embedding.",
    )
    is_indexed = models.BooleanField(
        default=False,
        db_index=True,
        help_text="True once the embedding is in the FAISS index.",
    )
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Clothing Item"
        verbose_name_plural = "Clothing Items"
        ordering = ["-created_at"]
        indexes = [
            models.Index(
                fields=["is_indexed", "created_at"],
                name="item_index_pending_idx",
            ),
            models.Index(
                fields=["category", "is_indexed"],
                name="item_category_indexed_idx",
            ),
        ]

    def __str__(self) -> str:
        parts = [p for p in (self.name, self.brand) if p]
        return " — ".join(parts) if parts else f"Item #{self.pk}"

    @property
    def display_category(self) -> str:
        return self.category.name if self.category_id else "Uncategorised"


# ---------------------------------------------------------------------------
# EmbeddingMetadata
# ---------------------------------------------------------------------------

class EmbeddingMetadata(models.Model):
    """
    Audit record written by build_index.  Records which model/version produced
    the stored vector so stale embeddings can be detected after a model upgrade.
    """

    clothing_item = models.OneToOneField(
        ClothingItem,
        on_delete=models.CASCADE,
        related_name="embedding_metadata",
    )
    model_name = models.CharField(max_length=100)
    embedding_version = models.CharField(max_length=200)
    created_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Embedding Metadata"
        verbose_name_plural = "Embedding Metadata"
        indexes = [
            models.Index(
                fields=["model_name", "embedding_version"],
                name="embedding_model_version_idx",
            ),
        ]

    def __str__(self) -> str:
        return (
            f"EmbeddingMetadata(item={self.clothing_item_id}, "
            f"model={self.model_name!r}, version={self.embedding_version!r})"
        )
