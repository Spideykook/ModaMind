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

from .ml.embedding_service import EmbeddingService
from .models import ClothingItem
from .search.faiss_manager import FaissManager

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

        try:
            embedder = EmbeddingService()
            query_embedding = embedder.extract_embedding(image_bytes)
        except Exception:
            logger.exception("SimilaritySearchView: embedding extraction failed")
            return Response(
                {"error": "Could not process the uploaded image."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        faiss_manager = FaissManager()
        if faiss_manager.total_vectors == 0:
            return Response(
                {"error": "The similarity index is empty. Seed the catalog and run build_index first."},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        matches = faiss_manager.search(query_embedding, top_k=self.TOP_K)

        results: list[dict] = []
        for item_id, score in matches:
            try:
                item = ClothingItem.objects.select_related("category").get(pk=item_id)
            except ClothingItem.DoesNotExist:
                logger.warning(
                    "SimilaritySearchView: FAISS id=%s has no ClothingItem (stale index?). Skipping.",
                    item_id,
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
                    "similarity_score": round(score, 4),
                }
            )

        return Response({"count": len(results), "results": results}, status=status.HTTP_200_OK)
