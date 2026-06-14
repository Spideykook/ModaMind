from django.conf import settings
from django.core.exceptions import ValidationError
from django.core.validators import FileExtensionValidator
from django.db import models


def validate_image_file_size(file) -> None:
    """
    Reject uploads larger than settings.MAX_UPLOAD_SIZE_BYTES.

    Applied to ClothingItem.image so oversized catalog photos are rejected
    in the admin (and any future management UI) the same way the
    /api/search/ endpoint rejects oversized query images.
    """
    if file.size > settings.MAX_UPLOAD_SIZE_BYTES:
        max_mb = settings.MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)
        raise ValidationError(f"Image file too large. Maximum allowed size is {max_mb}MB.")


class ClothingItem(models.Model):
    """
    A single catalog entry representing one piece of clothing.

    The primary key (`id`) doubles as the FAISS vector ID: when this item's
    image embedding is added to the index via FaissManager.add_vectors(),
    it is tagged with `item.id` (through IndexIDMap2). This means a FAISS
    search result can be resolved straight back to a row here with a single
    `ClothingItem.objects.get(pk=...)` lookup — no separate mapping table
    required.
    """

    CATEGORY_CHOICES = [
        ("tops", "Tops"),
        ("bottoms", "Bottoms"),
        ("dresses", "Dresses"),
        ("outerwear", "Outerwear"),
        ("footwear", "Footwear"),
        ("accessories", "Accessories"),
    ]

    image = models.ImageField(
        upload_to="clothing_images/",
        validators=[
            FileExtensionValidator(allowed_extensions=settings.ALLOWED_IMAGE_EXTENSIONS),
            validate_image_file_size,
        ],
        help_text="The catalog photo used to generate the ResNet50 embedding.",
    )
    title = models.CharField(max_length=255, blank=True, default="")
    category = models.CharField(
        max_length=32, choices=CATEGORY_CHOICES, blank=True, default=""
    )
    brand = models.CharField(max_length=100, blank=True, default="")
    description = models.TextField(blank=True, default="")

    is_indexed = models.BooleanField(
        default=False,
        help_text="True once this item's embedding has been added to the FAISS index.",
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Clothing Item"
        verbose_name_plural = "Clothing Items"

    def __str__(self) -> str:
        return self.title or f"ClothingItem #{self.pk}"
