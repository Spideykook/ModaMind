"""
FAISS-backed vector index for fashion item similarity search.

This module has no Django dependencies. FaissManager wraps a
faiss.IndexFlatIP(2048) inside an IndexIDMap2 so that vector IDs map
1:1 onto ClothingItem.id values from modapp.models, with no separate
position-tracking table required.
"""

import logging
import os
import threading
from typing import List, Optional, Tuple

import faiss
import numpy as np

logger = logging.getLogger(__name__)


class FaissManager:
    """
    Thread-safe singleton wrapper around a FAISS IndexFlatIP for cosine
    similarity search over 2048-dimensional, L2-normalized embeddings.

    Because IndexFlatIP computes the inner product between vectors, and
    EmbeddingService guarantees every stored/query vector has unit L2
    norm, the inner product here is mathematically equivalent to cosine
    similarity (range: -1.0 to 1.0, where 1.0 = identical direction).
    """

    _instance: Optional["FaissManager"] = None
    _instance_lock = threading.Lock()

    EMBEDDING_DIM = 2048
    DEFAULT_INDEX_PATH = os.path.join(
        os.path.dirname(__file__), "indexes", "fashion_items.index"
    )

    def __new__(cls, *args, **kwargs):
        # Singleton guard: only the first call's index_path takes effect.
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, index_path: Optional[str] = None) -> None:
        if self._initialized:
            return

        self.index_path = index_path or self.DEFAULT_INDEX_PATH
        self._op_lock = threading.Lock()
        self.index: faiss.IndexIDMap2 = None  # type: ignore[assignment]

        self._load_or_create_index()
        self._initialized = True

    # ------------------------------------------------------------------
    # Initialization / persistence
    # ------------------------------------------------------------------
    def _load_or_create_index(self) -> None:
        if os.path.exists(self.index_path):
            self.load_index()
        else:
            logger.info(
                "FaissManager: no index found at '%s' - creating a new IndexFlatIP(%d).",
                self.index_path,
                self.EMBEDDING_DIM,
            )
            flat_index = faiss.IndexFlatIP(self.EMBEDDING_DIM)
            self.index = faiss.IndexIDMap2(flat_index)

    def load_index(self) -> None:
        """Load the index from disk, validating type and dimensionality."""
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"No FAISS index file found at '{self.index_path}'.")

        with self._op_lock:
            loaded = faiss.read_index(self.index_path)

            # Defensive checks: refuse to operate on an index that doesn't
            # match the structure this codebase expects.
            if not isinstance(loaded, faiss.IndexIDMap2):
                raise TypeError(
                    "Loaded FAISS index is not an IndexIDMap2 - refusing to use it. "
                    "This file may have been created by an incompatible version."
                )
            if loaded.d != self.EMBEDDING_DIM:
                raise ValueError(
                    f"FAISS index dimension mismatch: expected {self.EMBEDDING_DIM}, "
                    f"got {loaded.d}."
                )

            self.index = loaded

        logger.info(
            "FaissManager: loaded index from '%s' (%d vectors).",
            self.index_path,
            self.index.ntotal,
        )

    def save_index(self) -> None:
        """Persist the current index to disk, creating directories as needed."""
        with self._op_lock:
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            faiss.write_index(self.index, self.index_path)

        logger.info("FaissManager: saved index to '%s'.", self.index_path)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------
    def add_vectors(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        """
        Add one or more L2-normalized embedding vectors to the index.

        Args:
            vectors: shape (N, 2048) or (2048,), dtype convertible to float32.
            ids: shape (N,) or scalar, dtype convertible to int64. These
                should correspond to ClothingItem.id values.

        Raises:
            ValueError: if vector dimensionality is wrong or the number of
                ids does not match the number of vectors.
        """
        vectors = np.ascontiguousarray(vectors, dtype="float32")
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        if vectors.shape[1] != self.EMBEDDING_DIM:
            raise ValueError(
                f"Expected vectors of dimension {self.EMBEDDING_DIM}, "
                f"got {vectors.shape[1]}."
            )

        ids_arr = np.ascontiguousarray(ids, dtype="int64").reshape(-1)
        if ids_arr.shape[0] != vectors.shape[0]:
            raise ValueError(
                f"Number of ids ({ids_arr.shape[0]}) must match number of "
                f"vectors ({vectors.shape[0]})."
            )

        with self._op_lock:
            self.index.add_with_ids(vectors, ids_arr)

        logger.info(
            "FaissManager: added %d vector(s). Total in index: %d.",
            vectors.shape[0],
            self.index.ntotal,
        )

    def reset_index(self) -> None:
        """
        Discard all vectors and start from a fresh, empty IndexFlatIP.

        Used by `manage.py build_index --rebuild` to re-embed the entire
        catalog from scratch (e.g. after changing the embedding model).
        """
        with self._op_lock:
            flat_index = faiss.IndexFlatIP(self.EMBEDDING_DIM)
            self.index = faiss.IndexIDMap2(flat_index)

        logger.info("FaissManager: index reset to empty.")

    def remove_vector(self, item_id: int) -> int:
        """
        Remove a single vector by its ID (e.g. when a ClothingItem is
        deleted from the catalog).

        Returns:
            The number of vectors actually removed (0 or 1).
        """
        with self._op_lock:
            removed = self.index.remove_ids(np.array([item_id], dtype="int64"))

        logger.info("FaissManager: removed %d vector(s) for id=%d.", removed, item_id)
        return int(removed)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Perform a Top-K cosine similarity search.

        Args:
            query_embedding: shape (2048,) or (1, 2048), L2-normalized.
            top_k: maximum number of results to return.

        Returns:
            A list of (item_id, similarity_score) tuples, ordered from
            most to least similar. similarity_score is the raw inner
            product (cosine similarity for unit-norm vectors), in [-1, 1].
            Returns an empty list if the index has no vectors yet.
        """
        if self.index.ntotal == 0:
            logger.warning("FaissManager: search() called on an empty index.")
            return []

        query = np.ascontiguousarray(query_embedding, dtype="float32").reshape(1, -1)
        if query.shape[1] != self.EMBEDDING_DIM:
            raise ValueError(
                f"Expected query of dimension {self.EMBEDDING_DIM}, got {query.shape[1]}."
            )

        with self._op_lock:
            scores, ids = self.index.search(query, top_k)

        results: List[Tuple[int, float]] = []
        for score, item_id in zip(scores[0], ids[0]):
            if item_id == -1:
                # FAISS pads with -1 when fewer than top_k results exist.
                continue
            results.append((int(item_id), float(score)))

        return results

    @property
    def total_vectors(self) -> int:
        return self.index.ntotal if self.index is not None else 0
