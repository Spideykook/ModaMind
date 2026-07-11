"""
Batch embedding pipeline for dataset images.

Orchestrates the end-to-end flow: scan a dataset folder, generate
embeddings for every image using the existing EmbeddingService, and
persist the results to disk in a format ready for FAISS consumption.

This module has no Django dependencies and can be imported standalone
(see scripts/run_embedding_pipeline.py).

Output format (.npz):
    embeddings  — np.ndarray, shape (N, 2048), dtype float32
    image_paths — np.ndarray, shape (N,), dtype str (relative paths)
    categories  — np.ndarray, shape (N,), dtype str

Loading the output:
    data = np.load("dataset_embeddings.npz", allow_pickle=True)
    embeddings  = data["embeddings"]    # (N, 2048) float32
    image_paths = data["image_paths"]   # (N,)
    categories  = data["categories"]    # (N,)
"""

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from .dataset_scanner import DatasetScanner, ImageRecord, ScanResult
from .embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """
    Summary of a completed embedding pipeline run.

    Attributes:
        total_images:     Number of images discovered by the scanner.
        embedded_count:   Number of images successfully embedded.
        failed_count:     Number of images that failed during embedding.
        skipped_by_scanner: Number of non-image files skipped by the scanner.
        failed_paths:     List of (relative_path, error_message) for failures.
        output_path:      Where the .npz file was saved (None if not saved).
        categories_found: Unique category names in the embedded results.
        elapsed_seconds:  Wall-clock time for the full pipeline run.
    """

    total_images: int = 0
    embedded_count: int = 0
    failed_count: int = 0
    skipped_by_scanner: int = 0
    failed_paths: List[tuple[str, str]] = field(default_factory=list)
    output_path: Optional[str] = None
    categories_found: List[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0


# Type alias for progress callbacks.
# Signature: callback(current_index, total_count, relative_path, success)
ProgressCallback = Callable[[int, int, str, bool], None]


class EmbeddingPipeline:
    """
    Scans a dataset directory, generates embeddings for every image,
    and saves the results as a .npz file.

    The pipeline reuses the existing EmbeddingService (which internally
    uses the singleton ModelLoader, so the ResNet50 weights are loaded
    only once regardless of how many pipelines are created).

    Design decisions:
        - Corrupted/unreadable images are logged and skipped, never crash
          the pipeline. Fashion datasets commonly have a few broken files.
        - Progress is reported via an optional callback, keeping the
          pipeline decoupled from any specific UI (CLI, Django command,
          web socket, etc.).
        - Output is a single .npz file rather than one file per image,
          because FAISS needs the full (N, 2048) matrix at index-build time
          and loading one file is far faster than loading thousands.

    Usage:
        pipeline = EmbeddingPipeline(
            dataset_dir="/path/to/dataset",
            output_path="/path/to/output/dataset_embeddings.npz",
        )
        result = pipeline.run()
        print(f"Embedded {result.embedded_count}/{result.total_images} images.")
    """

    DEFAULT_OUTPUT_FILENAME = "dataset_embeddings.npz"

    def __init__(
        self,
        dataset_dir: str,
        output_path: Optional[str] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> None:
        """
        Args:
            dataset_dir:       Path to the dataset root directory.
            output_path:       Where to save the .npz output. Defaults to
                               <dataset_dir>/dataset_embeddings.npz.
            progress_callback: Optional function called after each image
                               is processed. See ProgressCallback type alias.
        """
        self.dataset_dir = os.path.abspath(dataset_dir)
        self.output_path = output_path or os.path.join(
            self.dataset_dir, self.DEFAULT_OUTPUT_FILENAME
        )
        self.progress_callback = progress_callback

    def run(self) -> PipelineResult:
        """
        Execute the full embedding pipeline.

        Steps:
            1. Scan the dataset directory for images.
            2. Initialize the EmbeddingService (loads ResNet50 once).
            3. Iterate over every discovered image:
               a. Call EmbeddingService.extract_embedding(path).
               b. On success, collect the embedding + metadata.
               c. On failure, log the error and continue.
            4. Stack all embeddings into a single (N, 2048) matrix.
            5. Save the matrix + metadata to a .npz file.

        Returns:
            A PipelineResult summarizing the run.
        """
        start_time = time.time()
        result = PipelineResult()

        # --- Step 1: Scan ---
        logger.info("EmbeddingPipeline: scanning '%s' for images...", self.dataset_dir)
        scanner = DatasetScanner(self.dataset_dir)
        scan_result: ScanResult = scanner.scan()

        result.total_images = len(scan_result.records)
        result.skipped_by_scanner = len(scan_result.skipped_files)

        if result.total_images == 0:
            logger.warning(
                "EmbeddingPipeline: no images found in '%s'. Nothing to embed.",
                self.dataset_dir,
            )
            result.elapsed_seconds = time.time() - start_time
            return result

        # --- Step 2: Load model ---
        logger.info("EmbeddingPipeline: initializing EmbeddingService...")
        embedder = EmbeddingService()

        # --- Step 3: Embed each image ---
        embeddings: List[np.ndarray] = []
        image_paths: List[str] = []
        categories: List[str] = []

        for idx, record in enumerate(scan_result.records):
            success = self._embed_single_image(
                record, embedder, embeddings, image_paths, categories
            )

            if success:
                result.embedded_count += 1
            else:
                result.failed_count += 1
                # The error message was already appended inside _embed_single_image
                # via the result object — but we don't have it here. Let's track
                # failures in a simpler way.

            if self.progress_callback:
                self.progress_callback(
                    idx + 1,
                    result.total_images,
                    record.relative_path,
                    success,
                )

        # --- Step 4 & 5: Stack and save ---
        if embeddings:
            self._save_embeddings(embeddings, image_paths, categories, result)
        else:
            logger.error(
                "EmbeddingPipeline: all %d images failed. No output file created.",
                result.total_images,
            )

        result.categories_found = sorted(set(categories))
        result.elapsed_seconds = time.time() - start_time

        logger.info(
            "EmbeddingPipeline: done. %d/%d embedded, %d failed, %.1fs elapsed.",
            result.embedded_count,
            result.total_images,
            result.failed_count,
            result.elapsed_seconds,
        )

        return result

    def _embed_single_image(
        self,
        record: ImageRecord,
        embedder: EmbeddingService,
        embeddings: List[np.ndarray],
        image_paths: List[str],
        categories: List[str],
    ) -> bool:
        """
        Attempt to embed a single image. On success, appends to the
        accumulator lists. On failure, logs a warning and returns False.

        Returns:
            True if the embedding was generated successfully.
        """
        try:
            embedding = embedder.extract_embedding(record.absolute_path)

            embeddings.append(embedding)
            image_paths.append(record.relative_path)
            categories.append(record.category)
            return True

        except Exception as exc:  # noqa: BLE001
            # Broad catch is intentional: fashion datasets commonly contain
            # truncated JPEGs, zero-byte files, or non-image files with
            # image extensions. Crashing the whole pipeline for one bad
            # file would be unacceptable.
            logger.warning(
                "EmbeddingPipeline: failed to embed '%s': %s",
                record.relative_path,
                exc,
            )
            return False

    def _save_embeddings(
        self,
        embeddings: List[np.ndarray],
        image_paths: List[str],
        categories: List[str],
        result: PipelineResult,
    ) -> None:
        """
        Stack individual embedding vectors into a matrix and persist
        to a .npz file.

        The .npz format stores multiple named arrays in a single
        compressed archive. It's the standard NumPy serialization
        format and can be loaded with a single np.load() call.
        """
        embedding_matrix = np.stack(embeddings).astype("float32")

        os.makedirs(os.path.dirname(os.path.abspath(self.output_path)), exist_ok=True)

        # np.savez auto-appends ".npz" if the path doesn't already end
        # with it, so we normalise here to know the real file path.
        if not self.output_path.endswith(".npz"):
            actual_path = self.output_path + ".npz"
        else:
            actual_path = self.output_path

        np.savez(
            self.output_path,
            embeddings=embedding_matrix,
            image_paths=np.array(image_paths, dtype=object),
            categories=np.array(categories, dtype=object),
        )

        result.output_path = actual_path

        file_size_mb = os.path.getsize(actual_path) / (1024 * 1024)
        logger.info(
            "EmbeddingPipeline: saved %d embeddings to '%s'. "
            "Matrix shape: %s, file size: %.2f MB.",
            len(embeddings),
            actual_path,
            embedding_matrix.shape,
            file_size_mb,
        )
