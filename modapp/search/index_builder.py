"""
Builds a persistent FAISS index from pre-computed dataset embeddings.

This module bridges the EmbeddingPipeline output (.npz file) and the
FaissManager (FAISS binary index).  It loads embeddings, assigns
sequential integer IDs, populates a FAISS index via FaissManager, and
saves a JSON metadata sidecar so that search results can map FAISS
integer IDs back to image paths and categories.

This module has no Django dependencies.

File layout after a build:
    indexes/
    ├── fashion_dataset.index          — FAISS binary index
    └── fashion_dataset.metadata.json  — ID → image_path + category map

Loading the output:
    from modapp.search.index_builder import DatasetIndexBuilder
    metadata = DatasetIndexBuilder.load_metadata(
        "indexes/fashion_dataset.metadata.json"
    )
    # metadata["entries"]["0"]["image_path"]  → "tops/img001.jpg"
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np

from .faiss_manager import FaissManager

logger = logging.getLogger(__name__)


@dataclass
class IndexBuildResult:
    """
    Summary of a completed index build operation.

    Attributes:
        total_embeddings:  Number of embeddings loaded from the .npz file.
        indexed_count:     Number of vectors successfully added to the index.
        index_path:        Where the FAISS index was saved.
        metadata_path:     Where the metadata sidecar was saved.
        categories_found:  Unique category names in the indexed data.
        embedding_dim:     Dimensionality of the indexed vectors.
        source_npz:        Path to the source .npz file.
        elapsed_seconds:   Wall-clock time for the full build operation.
        index_size_mb:     Size of the saved .index file in megabytes.
        metadata_size_mb:  Size of the saved .metadata.json file in megabytes.
        skipped:           True if the build was skipped (valid index exists).
    """

    total_embeddings: int = 0
    indexed_count: int = 0
    index_path: Optional[str] = None
    metadata_path: Optional[str] = None
    categories_found: List[str] = field(default_factory=list)
    embedding_dim: int = 0
    source_npz: Optional[str] = None
    elapsed_seconds: float = 0.0
    index_size_mb: float = 0.0
    metadata_size_mb: float = 0.0
    skipped: bool = False


class DatasetIndexBuilder:
    """
    Builds a persistent FAISS index from a .npz embedding file produced
    by EmbeddingPipeline.

    The builder reuses the existing FaissManager (which provides validated
    IndexIDMap2 wrapping IndexFlatIP, thread-safe operations, and
    dimension checks) and adds a JSON metadata sidecar for mapping
    FAISS integer IDs back to image file paths and categories.

    Design decisions:
        - Sequential 0-based integer IDs are assigned to each embedding
          vector.  The metadata sidecar maps these IDs to the original
          image paths.  This differs from the DB-backed build_index
          command which uses ClothingItem.pk as IDs.
        - The metadata sidecar is a plain JSON file (not SQLite, not
          pickle) so it remains human-readable, framework-agnostic, and
          easy to inspect or debug.
        - If a valid index already exists and rebuild is not requested,
          the builder returns early to avoid wasting time on redundant
          work.  This makes the management command idempotent.

    Usage:
        builder = DatasetIndexBuilder(
            npz_path="/path/to/dataset_embeddings.npz",
        )
        result = builder.build()
        print(f"Indexed {result.indexed_count} vectors → {result.index_path}")
    """

    DEFAULT_INDEX_DIR = os.path.join(
        os.path.dirname(__file__), "indexes"
    )
    DEFAULT_INDEX_NAME = "fashion_dataset"
    METADATA_SUFFIX = ".metadata.json"

    def __init__(
        self,
        npz_path: str,
        index_path: Optional[str] = None,
        rebuild: bool = False,
    ) -> None:
        """
        Args:
            npz_path:    Path to the .npz file produced by EmbeddingPipeline.
            index_path:  Where to save the FAISS index.  Defaults to
                         <module_dir>/indexes/fashion_dataset.index.
            rebuild:     If True, overwrite any existing index at index_path.
                         If False and a valid index exists, skip building.
        """
        self.npz_path = os.path.abspath(npz_path)
        self.index_path = os.path.abspath(
            index_path
            or os.path.join(
                self.DEFAULT_INDEX_DIR, self.DEFAULT_INDEX_NAME + ".index"
            )
        )

        # Derive metadata sidecar path from index path.
        # e.g. "fashion_dataset.index" → "fashion_dataset.metadata.json"
        base, _ = os.path.splitext(self.index_path)
        self.metadata_path = base + self.METADATA_SUFFIX

        self.rebuild = rebuild

    def build(self) -> IndexBuildResult:
        """
        Execute the full index build pipeline.

        Steps:
            1. Check for an existing valid index (skip if rebuild=False).
            2. Load and validate the .npz embedding file.
            3. Reset the FaissManager singleton for a clean build.
            4. Add all embedding vectors to the FAISS index.
            5. Save the FAISS index to disk.
            6. Save the JSON metadata sidecar.

        Returns:
            An IndexBuildResult summarizing the build.

        Raises:
            FileNotFoundError: If the .npz file does not exist.
            ValueError: If the .npz file has invalid structure or dimensions.
        """
        start_time = time.time()
        result = IndexBuildResult(
            source_npz=self.npz_path,
            index_path=self.index_path,
            metadata_path=self.metadata_path,
        )

        # --- Step 1: Check for existing index ---
        if not self.rebuild and os.path.exists(self.index_path):
            is_valid, msg = self.validate_index_files(self.index_path)
            if is_valid:
                logger.info(
                    "DatasetIndexBuilder: valid index already exists at '%s'. "
                    "Use rebuild=True to overwrite. (%s)",
                    self.index_path,
                    msg,
                )
                existing_meta = self.load_metadata(self.metadata_path)
                result.total_embeddings = existing_meta.get("total_vectors", 0)
                result.indexed_count = existing_meta.get("total_vectors", 0)
                result.categories_found = existing_meta.get("categories", [])
                result.embedding_dim = existing_meta.get("embedding_dim", 0)
                result.skipped = True
                result.elapsed_seconds = time.time() - start_time
                return result

        # --- Step 2: Load and validate .npz ---
        logger.info(
            "DatasetIndexBuilder: loading embeddings from '%s'...",
            self.npz_path,
        )
        embeddings, image_paths, categories = self._load_npz()
        self._validate_embeddings(embeddings)

        num_vectors = embeddings.shape[0]
        result.total_embeddings = num_vectors
        result.embedding_dim = embeddings.shape[1]

        # --- Step 3: Prepare FaissManager ---
        # Reset the singleton so we get a fresh instance at our index_path.
        # This is safe because index building is a batch operation that runs
        # in its own process (management command or standalone script), never
        # concurrently with the web server.
        FaissManager._instance = None
        faiss_mgr = FaissManager(index_path=self.index_path)
        faiss_mgr.reset_index()

        # --- Step 4: Add vectors ---
        ids = np.arange(num_vectors, dtype="int64")
        logger.info(
            "DatasetIndexBuilder: adding %d vectors (dim=%d) to FAISS index...",
            num_vectors,
            result.embedding_dim,
        )
        faiss_mgr.add_vectors(embeddings, ids)
        result.indexed_count = faiss_mgr.total_vectors

        # --- Step 5: Save FAISS index ---
        faiss_mgr.save_index()
        result.index_size_mb = os.path.getsize(self.index_path) / (1024 * 1024)

        # --- Step 6: Save metadata sidecar ---
        metadata = self._build_metadata(image_paths, categories, ids)
        self._save_metadata(metadata)
        result.metadata_size_mb = (
            os.path.getsize(self.metadata_path) / (1024 * 1024)
        )

        result.categories_found = sorted(set(categories))
        result.elapsed_seconds = time.time() - start_time

        logger.info(
            "DatasetIndexBuilder: done. %d vectors indexed in %.1fs. "
            "Index: '%s' (%.2f MB), Metadata: '%s' (%.2f MB).",
            result.indexed_count,
            result.elapsed_seconds,
            self.index_path,
            result.index_size_mb,
            self.metadata_path,
            result.metadata_size_mb,
        )

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_npz(self) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Load the .npz file and extract the three expected arrays.

        Returns:
            (embeddings, image_paths, categories) where:
              - embeddings:  np.ndarray shape (N, 2048), dtype float32
              - image_paths: list of N relative path strings
              - categories:  list of N category strings

        Raises:
            FileNotFoundError: If the .npz file does not exist.
            ValueError: If the file is missing required arrays or is
                        corrupted.
        """
        if not os.path.exists(self.npz_path):
            raise FileNotFoundError(
                f"Embedding file not found: '{self.npz_path}'"
            )

        try:
            data = np.load(self.npz_path, allow_pickle=True)
        except Exception as exc:
            raise ValueError(
                f"Failed to load .npz file '{self.npz_path}': {exc}"
            ) from exc

        required_keys = {"embeddings", "image_paths", "categories"}
        missing = required_keys - set(data.files)
        if missing:
            raise ValueError(
                f"The .npz file is missing required arrays: {missing}. "
                f"Found: {data.files}. "
                f"Expected output from EmbeddingPipeline."
            )

        embeddings = data["embeddings"]
        image_paths = [str(p) for p in data["image_paths"]]
        categories = [str(c) for c in data["categories"]]

        logger.info(
            "DatasetIndexBuilder: loaded %d embeddings from '%s'. "
            "Shape: %s, dtype: %s.",
            len(image_paths),
            self.npz_path,
            embeddings.shape,
            embeddings.dtype,
        )

        return embeddings, image_paths, categories

    def _validate_embeddings(self, embeddings: np.ndarray) -> None:
        """
        Validate the embedding matrix shape and dtype.

        Raises:
            ValueError: If the matrix has wrong rank, is empty, or has
                        a dimension mismatch with FaissManager.EMBEDDING_DIM.
        """
        if embeddings.ndim != 2:
            raise ValueError(
                f"Expected a 2D embedding matrix, got {embeddings.ndim}D "
                f"with shape {embeddings.shape}."
            )

        if embeddings.shape[0] == 0:
            raise ValueError("Embedding matrix is empty (0 rows).")

        if embeddings.shape[1] != FaissManager.EMBEDDING_DIM:
            raise ValueError(
                f"Embedding dimension mismatch: expected "
                f"{FaissManager.EMBEDDING_DIM}, got {embeddings.shape[1]}."
            )

        # Warn (but don't fail) if dtype needs conversion — FaissManager's
        # add_vectors handles the cast to float32 via np.ascontiguousarray.
        if embeddings.dtype != np.float32:
            logger.warning(
                "DatasetIndexBuilder: embeddings dtype is %s; "
                "FaissManager will convert to float32.",
                embeddings.dtype,
            )

    def _build_metadata(
        self,
        image_paths: List[str],
        categories: List[str],
        ids: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Build the metadata dictionary that maps FAISS integer IDs to
        image paths and categories.

        The dictionary is structured for fast lookup by ID (string key)
        and also stores aggregate information (categories list, counts)
        for quick inspection without iterating all entries.
        """
        entries: Dict[str, Dict[str, str]] = {}
        for idx, (img_path, category) in enumerate(
            zip(image_paths, categories)
        ):
            entries[str(ids[idx])] = {
                "image_path": img_path,
                "category": category,
            }

        return {
            "version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_npz": self.npz_path,
            "index_path": self.index_path,
            "total_vectors": len(entries),
            "embedding_dim": FaissManager.EMBEDDING_DIM,
            "categories": sorted(set(categories)),
            "entries": entries,
        }

    def _save_metadata(self, metadata: Dict[str, Any]) -> None:
        """Persist the metadata sidecar to disk as formatted JSON."""
        os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)

        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(
            "DatasetIndexBuilder: saved metadata to '%s' (%d entries).",
            self.metadata_path,
            len(metadata["entries"]),
        )

    # ------------------------------------------------------------------
    # Static utilities
    # ------------------------------------------------------------------

    @staticmethod
    def load_metadata(metadata_path: str) -> Dict[str, Any]:
        """
        Load and return a previously saved metadata sidecar.

        This is the primary way to access ID-to-image mappings at search
        time without loading the full .npz file.

        Args:
            metadata_path: Path to the .metadata.json file.

        Returns:
            The parsed metadata dictionary.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file contains invalid JSON.
        """
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"Metadata file not found: '{metadata_path}'"
            )

        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON in metadata file '{metadata_path}': {exc}"
            ) from exc

        logger.info(
            "DatasetIndexBuilder: loaded metadata from '%s' (%d entries).",
            metadata_path,
            len(metadata.get("entries", {})),
        )
        return metadata

    @staticmethod
    def validate_index_files(index_path: str) -> Tuple[bool, str]:
        """
        Check whether a FAISS index and its metadata sidecar are valid
        and consistent with each other.

        This is used to detect missing, invalid, or corrupted index files
        without triggering a full rebuild.  The management command calls
        this before deciding whether to skip or rebuild.

        Args:
            index_path: Path to the .index file.

        Returns:
            A (is_valid, message) tuple.  is_valid is True only if both
            the index and metadata exist, are parseable, and agree on
            the vector count.
        """
        # --- Check index file ---
        if not os.path.exists(index_path):
            return False, f"Index file not found: '{index_path}'"

        # --- Derive and check metadata path ---
        base, _ = os.path.splitext(index_path)
        metadata_path = base + DatasetIndexBuilder.METADATA_SUFFIX

        if not os.path.exists(metadata_path):
            return False, f"Metadata sidecar not found: '{metadata_path}'"

        # --- Validate FAISS index structure ---
        try:
            loaded = faiss.read_index(index_path)
        except Exception as exc:
            return False, f"Failed to read FAISS index: {exc}"

        if not isinstance(loaded, faiss.IndexIDMap2):
            return False, (
                f"Index is not an IndexIDMap2 "
                f"(got {type(loaded).__name__}). "
                f"The file may have been created by an incompatible version."
            )

        if loaded.d != FaissManager.EMBEDDING_DIM:
            return False, (
                f"Dimension mismatch: expected {FaissManager.EMBEDDING_DIM}, "
                f"got {loaded.d}."
            )

        # --- Validate metadata sidecar ---
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            return False, f"Failed to read metadata sidecar: {exc}"

        if "entries" not in metadata:
            return False, "Metadata is missing the 'entries' key."

        # --- Cross-check counts ---
        index_count = loaded.ntotal
        metadata_count = len(metadata["entries"])
        if index_count != metadata_count:
            return False, (
                f"Count mismatch: index has {index_count} vectors but "
                f"metadata has {metadata_count} entries."
            )

        return True, f"Valid index with {index_count} vectors."
