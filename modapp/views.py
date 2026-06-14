"""
Views for the ModaMind core app.

- IndexView: renders the single-page dashboard (modapp/templates/modapp/index.html).
- SimilaritySearchView: DRF APIView that accepts an uploaded image, runs it
  through EmbeddingService, queries FaissManager for the Top-K most similar
  catalog items, and returns their metadata as JSON for app.js to render.
"""

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
    """
    Quickly verify that `image_bytes` decodes as a genuine image.

    This guards against disguised-file attacks where a non-image payload
    is uploaded with a spoofed filename/Content-Type (e.g. a script
    renamed to "outfit.jpg"), before those bytes are ever handed to the
    PyTorch pipeline.

    Image.verify() performs a structural check without fully decoding
    pixel data, so it stays cheap even for large files. It also leaves the
    file object unusable afterwards, which is fine here since this function
    only opens a throwaway BytesIO copy.
    """
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            img.verify()
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        return False


class IndexView(TemplateView):
    """Renders the main ModaMind dashboard (upload zone + results grid)."""

    template_name = "modapp/index.html"


class SimilaritySearchView(APIView):
    """
    POST /api/search/

    Accepts a multipart/form-data request with a single image file under
    the 'image' field. Returns the Top-K visually similar ClothingItems
    from the catalog, ranked by cosine similarity.

    Example success response:
        {
            "count": 3,
            "results": [
                {
                    "id": 12,
                    "title": "Oversized Denim Jacket",
                    "category": "outerwear",
                    "brand": "Urban Thread",
                    "image_url": "http://localhost:8000/media/clothing_images/jacket12.jpg",
                    "similarity_score": 0.9123
                },
                ...
            ]
        }
    """

    parser_classes = [MultiPartParser, FormParser]

    TOP_K = 5

    def post(self, request, *args, **kwargs):
        uploaded_file = request.FILES.get("image")
        if uploaded_file is None:
            return Response(
                {"error": "No image file provided. Expected a multipart field named 'image'."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # --- Step 1: Reject oversized files up front ----------------------
        if uploaded_file.size > settings.MAX_UPLOAD_SIZE_BYTES:
            max_mb = settings.MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)
            return Response(
                {"error": f"Image exceeds the {max_mb}MB upload limit."},
                status=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            )

        # --- Step 2: Reject disallowed content types ----------------------
        if uploaded_file.content_type not in settings.ALLOWED_IMAGE_CONTENT_TYPES:
            allowed = ", ".join(settings.ALLOWED_IMAGE_CONTENT_TYPES)
            return Response(
                {"error": f"Unsupported file type '{uploaded_file.content_type}'. Allowed types: {allowed}."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        image_bytes = uploaded_file.read()

        # --- Step 3: Verify the bytes are actually a decodable image -------
        if not _is_decodable_image(image_bytes):
            return Response(
                {"error": "The uploaded file is not a valid image."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # --- Step 4: Extract the query embedding ---------------------------
        try:
            embedder = EmbeddingService()
            query_embedding = embedder.extract_embedding(image_bytes)
        except Exception:  # noqa: BLE001 - log internally, return a generic message
            logger.exception("ModaMind: embedding extraction failed")
            return Response(
                {"error": "Could not process the uploaded image."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # --- Step 5: Query the FAISS index ----------------------------------
        faiss_manager = FaissManager()
        if faiss_manager.total_vectors == 0:
            return Response(
                {
                    "error": (
                        "The similarity index is empty. Add catalog items via "
                        "/admin/ and run `python manage.py build_index` first."
                    )
                },
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        matches = faiss_manager.search(query_embedding, top_k=self.TOP_K)

        # --- Step 6: Resolve FAISS IDs back to ClothingItem rows -----------
        results = []
        for item_id, score in matches:
            try:
                item = ClothingItem.objects.get(pk=item_id)
            except ClothingItem.DoesNotExist:
                logger.warning(
                    "ModaMind: FAISS returned id=%s with no matching ClothingItem (stale index?).",
                    item_id,
                )
                continue

            image_url = request.build_absolute_uri(item.image.url) if item.image else None

            results.append(
                {
                    "id": item.id,
                    "title": item.title or f"Item #{item.id}",
                    "category": item.get_category_display() if item.category else "",
                    "brand": item.brand,
                    "image_url": image_url,
                    "similarity_score": round(score, 4),
                }
            )

        return Response({"count": len(results), "results": results}, status=status.HTTP_200_OK)
