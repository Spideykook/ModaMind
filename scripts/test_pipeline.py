"""
Standalone verification script for ModaMind's ML + FAISS pipeline.

This runs entirely outside Django: it imports `modapp.ml.embedding_service`
and `modapp.search.faiss_manager` directly, both of which have zero Django
dependencies by design. Use this to sanity-check the embedding and
similarity-search logic in isolation, before wiring up HTTP endpoints or
running the dev server.

Usage:
    python scripts/test_pipeline.py [path/to/image_dir]

If no directory is given, defaults to <project_root>/test_images/.
Populate that folder with a handful of JPG/PNG/WEBP images - ideally
including at least two visually similar items - before running.

What this checks:
    1. EmbeddingService loads ResNet50 once and produces a 2048-d,
       unit-L2-norm vector for every image.
    2. FaissManager.add_vectors() / save_index() / search() round-trip
       correctly using a throwaway index file (the real
       modapp/search/indexes/fashion_items.index is never touched).
    3. A "self-similarity" sanity check: searching with an image that is
       already in the index should return that same image as the #1
       match with a similarity score of ~1.0.
"""

from __future__ import annotations

import os
import sys

import numpy as np

# Make the project root (the directory containing the `modapp` package)
# importable without requiring Django to be configured. modapp.ml and
# modapp.search are plain Python packages, so no `django.setup()` call
# is needed here.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from modapp.ml.embedding_service import EmbeddingService  # noqa: E402
from modapp.search.faiss_manager import FaissManager  # noqa: E402

SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
DEFAULT_IMAGE_DIR = os.path.join(PROJECT_ROOT, "test_images")

# A throwaway index, separate from modapp/search/indexes/fashion_items.index,
# so running this script never affects the real catalog index. Removed at
# the end of a successful run.
TEST_INDEX_PATH = os.path.join(PROJECT_ROOT, "scripts", "_test_pipeline.index")


def discover_images(image_dir: str) -> list[str]:
    """Return sorted image filenames in `image_dir`, or raise if none exist."""
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(
            f"Image directory not found: '{image_dir}'.\n"
            f"Create it and add a few JPG/PNG/WEBP images, or pass a "
            f"different path as the first argument:\n"
            f"  python scripts/test_pipeline.py path/to/images"
        )

    files = sorted(
        f for f in os.listdir(image_dir) if f.lower().endswith(SUPPORTED_EXTENSIONS)
    )

    if not files:
        raise FileNotFoundError(
            f"No images found in '{image_dir}'. Supported extensions: "
            f"{', '.join(SUPPORTED_EXTENSIONS)}."
        )

    return files


def main() -> None:
    image_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE_DIR
    image_files = discover_images(image_dir)

    print(f"Found {len(image_files)} image(s) in '{image_dir}'.\n")

    print("Loading EmbeddingService (ResNet50, classification head removed)...")
    embedder = EmbeddingService()
    print(f"  -> running on device: {embedder.device}\n")

    # Note: FaissManager is a process-wide singleton. The index_path passed
    # on its *first* construction in this process wins, which is fine here
    # since this script is the only thing running.
    if os.path.exists(TEST_INDEX_PATH):
        os.remove(TEST_INDEX_PATH)
    faiss_manager = FaissManager(index_path=TEST_INDEX_PATH)

    vectors: list[np.ndarray] = []
    ids: list[int] = []
    id_to_filename: dict[int, str] = {}

    print("Embedding images:")
    for i, filename in enumerate(image_files):
        path = os.path.join(image_dir, filename)
        embedding = embedder.extract_embedding(path)

        norm = float(np.linalg.norm(embedding))
        print(
            f"  [{i}] {filename:<30s} shape={embedding.shape}  "
            f"||v||={norm:.6f}  (expect ~1.0)"
        )

        vectors.append(embedding)
        ids.append(i)
        id_to_filename[i] = filename

    vectors_matrix = np.stack(vectors)

    print("\nAdding vectors to FAISS IndexFlatIP (wrapped in IndexIDMap2)...")
    faiss_manager.add_vectors(vectors_matrix, ids=np.array(ids))
    faiss_manager.save_index()
    print(f"  -> index now contains {faiss_manager.total_vectors} vector(s).\n")

    # --- Self-similarity sanity check ---------------------------------
    # Querying with an image that's already in the index should return
    # that same image as the #1 match with a score very close to 1.0.
    query_index = 0
    query_filename = id_to_filename[query_index]
    query_embedding = vectors_matrix[query_index]

    top_k = min(3, len(image_files))
    print(f"Top-{top_k} matches for query image: '{query_filename}'")

    results = faiss_manager.search(query_embedding, top_k=top_k)
    for rank, (item_id, score) in enumerate(results, start=1):
        marker = "  <- self" if item_id == query_index else ""
        print(f"  {rank}. {id_to_filename[item_id]:<30s} similarity={score:.4f}{marker}")

    best_match_id, best_score = results[0]
    if best_match_id == query_index and best_score > 0.99:
        print("\n[PASS] Self-similarity check: the query image matched itself with score ~1.0.")
    else:
        print(
            "\n[WARN] Self-similarity check did not return the query image as the "
            "top match with score ~1.0. Review the embedding/index logic."
        )

    # Clean up the throwaway index so repeated runs start fresh.
    if os.path.exists(TEST_INDEX_PATH):
        os.remove(TEST_INDEX_PATH)


if __name__ == "__main__":
    main()
