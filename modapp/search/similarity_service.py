"""
Similarity Search Service — the core business-logic layer for finding
visually similar fashion items.

This module owns the end-to-end search pipeline:
    query image → embedding → FAISS lookup → result hydration

It is deliberately framework-agnostic (no Django imports). Consumers
include SimilaritySearchView (Django REST Framework), management
commands, and standalone scripts.

The strongly-typed dataclasses below define the data contract that
every consumer depends on. They follow the same pattern used elsewhere
in the codebase (PipelineResult, IndexBuildResult, ScanResult).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

from ..ml.embedding_service import EmbeddingService, ImageInput
from .faiss_manager import FaissManager

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """
    A single item returned by a similarity search.

    Attributes:
        item_id:          Primary key of the matched ClothingItem.
        similarity_score: Cosine similarity in [-1.0, 1.0].  1.0 means
                          the query and catalog vectors point in exactly
                          the same direction (identical visual features).
        name:             Display name of the clothing item.
        category:         Human-readable category label (e.g. 'Tops').
        brand:            Brand name (may be empty).
        color:            Color descriptor (may be empty).
        image_path:       Relative path or URL to the catalog image.
                          The view layer resolves this to an absolute URL.
    """

    item_id: int
    similarity_score: float
    name: str = ""
    category: str = ""
    brand: str = ""
    color: str = ""
    image_path: str = ""


@dataclass
class SearchResponse:
    """
    Complete response from a similarity search operation.

    Encapsulates both the result list and any metadata a consumer
    might need (total count, query timing, errors).

    Attributes:
        results:         Ordered list of SearchResult objects (most
                         similar first).
        total:           Number of results returned.
        query_time_ms:   Wall-clock time for the search in milliseconds.
        error:           If non-None, the search failed and this string
                         describes why. When error is set, results will
                         be empty.
    """

    results: List[SearchResult] = field(default_factory=list)
    total: int = 0
    query_time_ms: float = 0.0
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        """True if the search completed without error."""
        return self.error is None


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class SimilaritySearchService:
    """
    Orchestrates visual similarity search: image → embedding → FAISS → results.

    This class is the single entry point for all similarity search
    operations. It coordinates EmbeddingService (image-to-vector) and
    FaissManager (vector-to-top-k) behind a clean, one-method interface.

    The service returns *un-hydrated* SearchResult objects containing
    only ``item_id`` and ``similarity_score``.  Populating the remaining
    fields (name, category, brand, etc.) from a database or metadata
    sidecar is the caller's responsibility — this keeps the service
    framework-agnostic.

    Design decisions:
        - No Django imports.  The service can be used in standalone
          scripts, CLI tools, or non-Django web frameworks.
        - Errors are captured in ``SearchResponse.error`` rather than
          raised, so the caller decides how to surface them (HTTP 503,
          CLI warning, retry loop, etc.).
        - ``query_time_ms`` is measured inside the service for accurate
          observability regardless of network/serialization overhead.
        - ``default_top_k`` is configurable per instance so different
          consumers (API endpoint, batch evaluation, admin preview) can
          use different defaults without subclassing.

    Usage:
        service = SimilaritySearchService()
        response = service.search_by_image(image_bytes)
        if response.ok:
            for r in response.results:
                print(f"Item {r.item_id}: {r.similarity_score:.4f}")
    """

    DEFAULT_TOP_K = 5

    def __init__(self, default_top_k: int = DEFAULT_TOP_K) -> None:
        """
        Args:
            default_top_k: Number of results to return when the caller
                           does not specify ``top_k`` explicitly.
        """
        self.default_top_k = default_top_k

    def search_by_image(
        self,
        image_input: ImageInput,
        top_k: Optional[int] = None,
    ) -> SearchResponse:
        """
        Run the full similarity search pipeline for a query image.

        Steps:
            1. Extract a 2048-d L2-normalized embedding via EmbeddingService.
            2. Search the FAISS index for the top-k nearest neighbours.
            3. Wrap each (item_id, score) pair in a SearchResult.

        Args:
            image_input: A file path (str), raw bytes, or PIL.Image —
                         anything accepted by EmbeddingService.
            top_k:       Maximum number of results. Falls back to
                         ``self.default_top_k`` if not supplied.

        Returns:
            A SearchResponse.  On success, ``response.ok`` is True and
            ``response.results`` contains up to ``top_k`` SearchResult
            objects ordered by descending similarity.  On failure,
            ``response.ok`` is False and ``response.error`` describes
            the problem.
        """
        effective_top_k = top_k if top_k is not None else self.default_top_k
        start = time.perf_counter()

        # --- Step 1: Extract embedding ---
        try:
            embedder = EmbeddingService()
            query_embedding = embedder.extract_embedding(image_input)
        except Exception:
            logger.exception("SimilaritySearchService: embedding extraction failed")
            return SearchResponse(
                error="Could not process the provided image.",
                query_time_ms=_elapsed_ms(start),
            )

        # --- Step 2: FAISS search ---
        faiss_manager = FaissManager()

        if faiss_manager.total_vectors == 0:
            logger.warning("SimilaritySearchService: index is empty.")
            return SearchResponse(
                error="The similarity index is empty. "
                      "Seed the catalog and run build_index first.",
                query_time_ms=_elapsed_ms(start),
            )

        matches = faiss_manager.search(query_embedding, top_k=effective_top_k)

        # --- Step 3: Build results ---
        results = [
            SearchResult(
                item_id=item_id,
                similarity_score=round(score, 4),
            )
            for item_id, score in matches
        ]

        elapsed = _elapsed_ms(start)

        logger.info(
            "SimilaritySearchService: returned %d result(s) in %.1f ms.",
            len(results),
            elapsed,
        )

        return SearchResponse(
            results=results,
            total=len(results),
            query_time_ms=round(elapsed, 2),
        )


def _elapsed_ms(start: float) -> float:
    """Convert a perf_counter start time to elapsed milliseconds."""
    return (time.perf_counter() - start) * 1000.0
