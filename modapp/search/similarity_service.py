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

from dataclasses import dataclass, field
from typing import List, Optional


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
