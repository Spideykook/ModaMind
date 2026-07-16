"""
modapp/views.py — ModaMind views.

IndexView              → GET /    — SPA dashboard.
SimilaritySearchView   → POST /api/search/ — DRF endpoint (multipart upload).
"""

from __future__ import annotations

import io
import logging

from django.conf import settings
from django.views.generic import TemplateView
from PIL import Image, UnidentifiedImageError
from rest_framework import status
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import Category, ClothingItem
from .search.similarity_service import SimilaritySearchService

logger = logging.getLogger(__name__)


def _is_decodable_image(image_bytes: bytes) -> bool:
    """Verify bytes decode as a genuine image (anti disguised-file)."""
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            img.verify()
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        return False


class IndexView(TemplateView):
    template_name = "modapp/index.html"


class SimilaritySearchView(APIView):
    """POST /api/search/ — returns Top-K similar ClothingItems as JSON."""

    parser_classes = [MultiPartParser, FormParser]
    TOP_K = 5

    def post(self, request, *args, **kwargs) -> Response:
        uploaded_file = request.FILES.get("image")
        if uploaded_file is None:
            return Response(
                {"error": "No image provided. Send a multipart field named 'image'."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if uploaded_file.size > settings.MAX_UPLOAD_SIZE_BYTES:
            max_mb = settings.MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)
            return Response(
                {"error": f"Image exceeds the {max_mb} MB limit."},
                status=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            )

        if uploaded_file.content_type not in settings.ALLOWED_IMAGE_CONTENT_TYPES:
            allowed = ", ".join(settings.ALLOWED_IMAGE_CONTENT_TYPES)
            return Response(
                {"error": f"Unsupported file type '{uploaded_file.content_type}'. Allowed: {allowed}."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        image_bytes = uploaded_file.read()

        if not _is_decodable_image(image_bytes):
            return Response(
                {"error": "The uploaded file is not a valid image."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        service = SimilaritySearchService()

        # --- Optional category filter ---
        allowed_ids = None
        category_slug = request.data.get("category", "").strip().lower()
        if category_slug:
            try:
                cat = Category.objects.get(slug=category_slug)
            except Category.DoesNotExist:
                return Response(
                    {"error": f"Unknown category slug: '{category_slug}'."},
                    status=status.HTTP_400_BAD_REQUEST,
                )
            allowed_ids = set(
                ClothingItem.objects
                .filter(category=cat, is_indexed=True)
                .values_list("pk", flat=True)
            )

        search_response = service.search_by_image(
            image_bytes, top_k=self.TOP_K, allowed_ids=allowed_ids,
        )

        if not search_response.ok:
            http_status = (
                status.HTTP_503_SERVICE_UNAVAILABLE
                if "index is empty" in search_response.error.lower()
                else status.HTTP_400_BAD_REQUEST
            )
            return Response({"error": search_response.error}, status=http_status)

        results: list[dict] = []
        for match in search_response.results:
            try:
                item = ClothingItem.objects.select_related("category").get(pk=match.item_id)
            except ClothingItem.DoesNotExist:
                logger.warning(
                    "SimilaritySearchView: FAISS id=%s has no ClothingItem (stale index?). Skipping.",
                    match.item_id,
                )
                continue

            image_url = request.build_absolute_uri(item.image.url) if item.image else None
            results.append(
                {
                    "id": item.id,
                    "name": item.name or f"Item #{item.id}",
                    "category": item.display_category,
                    "brand": item.brand,
                    "color": item.color,
                    "image_url": image_url,
                    "similarity_score": match.similarity_score,
                }
            )

        return Response({"count": len(results), "results": results}, status=status.HTTP_200_OK)
