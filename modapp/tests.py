"""
Security and correctness tests for the ModaMind core app.

These tests deliberately avoid loading the real ResNet50 model (which would
require downloading pretrained weights on first use). Wherever a view or
model under test depends on EmbeddingService, it is mocked. FaissManager
itself has zero PyTorch dependencies and is tested directly against
temporary index files on disk.

Run with:
    python manage.py test modapp
"""

from __future__ import annotations

import io
import os
import tempfile
from unittest import mock

import faiss
import numpy as np
from django.core.exceptions import ValidationError
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.urls import reverse
from PIL import Image

from .models import ClothingItem
from .search.faiss_manager import FaissManager


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _make_image_bytes(fmt: str = "JPEG", size: tuple[int, int] = (64, 64), noisy: bool = False) -> bytes:
    """Return raw encoded bytes for a small in-memory image."""
    if noisy:
        # Random noise compresses poorly, which is useful for reliably
        # producing a file *larger* than a given byte threshold.
        rng = np.random.default_rng(0)
        array = (rng.random((size[1], size[0], 3)) * 255).astype("uint8")
        image = Image.fromarray(array, "RGB")
    else:
        image = Image.new("RGB", size, color=(255, 0, 128))

    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


def _make_uploaded_image(
    name: str = "test.jpg",
    fmt: str = "JPEG",
    content_type: str = "image/jpeg",
    size: tuple[int, int] = (64, 64),
    noisy: bool = False,
) -> SimpleUploadedFile:
    data = _make_image_bytes(fmt=fmt, size=size, noisy=noisy)
    return SimpleUploadedFile(name, data, content_type=content_type)


# --------------------------------------------------------------------------
# FaissManager: index integrity & validation
# --------------------------------------------------------------------------
class FaissManagerTests(TestCase):
    """FaissManager has no Django/PyTorch dependencies -> tested directly."""

    def setUp(self):
        # FaissManager is a process-wide singleton. Reset it before each
        # test so tests don't leak index state into one another.
        FaissManager._instance = None
        self._tmpdir = tempfile.TemporaryDirectory()
        self.index_path = os.path.join(self._tmpdir.name, "test.index")

    def tearDown(self):
        FaissManager._instance = None
        self._tmpdir.cleanup()

    def test_starts_empty(self):
        manager = FaissManager(index_path=self.index_path)
        self.assertEqual(manager.total_vectors, 0)
        self.assertEqual(manager.search(np.zeros(2048, dtype="float32")), [])

    def test_add_and_search_round_trip(self):
        manager = FaissManager(index_path=self.index_path)

        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((3, 2048)).astype("float32")
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)  # unit-norm, like EmbeddingService output

        manager.add_vectors(vectors, ids=np.array([10, 20, 30]))
        self.assertEqual(manager.total_vectors, 3)

        results = manager.search(vectors[0], top_k=1)
        self.assertEqual(results[0][0], 10)
        self.assertAlmostEqual(results[0][1], 1.0, places=4)

    def test_save_and_load_round_trip(self):
        manager = FaissManager(index_path=self.index_path)

        vectors = np.eye(2048, dtype="float32")[:2]  # two orthogonal unit vectors
        manager.add_vectors(vectors, ids=np.array([1, 2]))
        manager.save_index()

        # Simulate a fresh process picking the persisted index back up.
        FaissManager._instance = None
        reloaded = FaissManager(index_path=self.index_path)
        self.assertEqual(reloaded.total_vectors, 2)

    def test_add_vectors_rejects_wrong_dimension(self):
        manager = FaissManager(index_path=self.index_path)
        bad_vector = np.zeros((16,), dtype="float32")
        with self.assertRaises(ValueError):
            manager.add_vectors(bad_vector, ids=np.array([1]))

    def test_search_rejects_wrong_dimension(self):
        manager = FaissManager(index_path=self.index_path)
        manager.add_vectors(np.eye(2048, dtype="float32")[:1], ids=np.array([1]))
        with self.assertRaises(ValueError):
            manager.search(np.zeros(16, dtype="float32"))

    def test_load_index_rejects_dimension_mismatch(self):
        """A structurally valid IndexIDMap2 with the wrong dimensionality must be refused."""
        wrong_dim_index = faiss.IndexIDMap2(faiss.IndexFlatIP(16))
        faiss.write_index(wrong_dim_index, self.index_path)

        FaissManager._instance = None
        with self.assertRaises(ValueError):
            FaissManager(index_path=self.index_path)

    def test_load_index_rejects_non_idmap_index(self):
        """A bare IndexFlatIP (not wrapped in IndexIDMap2) must be refused."""
        plain_index = faiss.IndexFlatIP(2048)
        faiss.write_index(plain_index, self.index_path)

        FaissManager._instance = None
        with self.assertRaises(TypeError):
            FaissManager(index_path=self.index_path)


# --------------------------------------------------------------------------
# ClothingItem: upload validators
# --------------------------------------------------------------------------
class ClothingItemValidationTests(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._media_tmpdir = tempfile.TemporaryDirectory()
        cls._media_override = override_settings(MEDIA_ROOT=cls._media_tmpdir.name)
        cls._media_override.enable()

    @classmethod
    def tearDownClass(cls):
        cls._media_override.disable()
        cls._media_tmpdir.cleanup()
        super().tearDownClass()

    def test_rejects_disallowed_extension(self):
        """GIF is a valid image format, but isn't in ALLOWED_IMAGE_EXTENSIONS."""
        gif_file = _make_uploaded_image(name="photo.gif", fmt="GIF", content_type="image/gif")
        item = ClothingItem(image=gif_file, title="Bad extension")
        with self.assertRaises(ValidationError):
            item.full_clean()

    @override_settings(MAX_UPLOAD_SIZE_BYTES=1024)
    def test_rejects_oversized_image(self):
        big_file = _make_uploaded_image(size=(256, 256), noisy=True)
        self.assertGreater(big_file.size, 1024)

        item = ClothingItem(image=big_file, title="Too big")
        with self.assertRaises(ValidationError):
            item.full_clean()

    def test_accepts_valid_jpeg(self):
        good_file = _make_uploaded_image()
        item = ClothingItem(image=good_file, title="Good", category="tops")
        item.full_clean()  # should not raise


# --------------------------------------------------------------------------
# SimilaritySearchView: request validation & happy path
# --------------------------------------------------------------------------
class SimilaritySearchViewTests(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._media_tmpdir = tempfile.TemporaryDirectory()
        cls._media_override = override_settings(MEDIA_ROOT=cls._media_tmpdir.name)
        cls._media_override.enable()

    @classmethod
    def tearDownClass(cls):
        cls._media_override.disable()
        cls._media_tmpdir.cleanup()
        super().tearDownClass()

    def setUp(self):
        self.url = reverse("modapp:similarity-search")

    def test_missing_image_field_returns_400(self):
        response = self.client.post(self.url, {})
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json())

    @override_settings(MAX_UPLOAD_SIZE_BYTES=1024)
    def test_oversized_file_returns_413(self):
        oversized = _make_uploaded_image(size=(256, 256), noisy=True)
        self.assertGreater(oversized.size, 1024)

        response = self.client.post(self.url, {"image": oversized})
        self.assertEqual(response.status_code, 413)
        self.assertIn("error", response.json())

    def test_disallowed_content_type_returns_400(self):
        text_file = SimpleUploadedFile("notes.txt", b"hello world", content_type="text/plain")
        response = self.client.post(self.url, {"image": text_file})
        self.assertEqual(response.status_code, 400)
        self.assertIn("Unsupported file type", response.json()["error"])

    def test_disguised_non_image_returns_400(self):
        """A correctly-named/typed file whose bytes are not a real image must be rejected."""
        fake_image = SimpleUploadedFile("outfit.jpg", b"not-an-image", content_type="image/jpeg")
        response = self.client.post(self.url, {"image": fake_image})
        self.assertEqual(response.status_code, 400)
        self.assertIn("not a valid image", response.json()["error"])

    @mock.patch("modapp.views.FaissManager")
    @mock.patch("modapp.views.EmbeddingService")
    def test_empty_index_returns_503(self, mock_embedding_service, mock_faiss_manager):
        mock_embedding_service.return_value.extract_embedding.return_value = np.zeros(2048, dtype="float32")
        mock_faiss_manager.return_value.total_vectors = 0

        response = self.client.post(self.url, {"image": _make_uploaded_image()})

        self.assertEqual(response.status_code, 503)
        self.assertIn("empty", response.json()["error"])

    @mock.patch("modapp.views.FaissManager")
    @mock.patch("modapp.views.EmbeddingService")
    def test_successful_search_returns_matching_items(self, mock_embedding_service, mock_faiss_manager):
        item = ClothingItem.objects.create(
            image=_make_uploaded_image(name="jacket.jpg"),
            title="Oversized Denim Jacket",
            category="outerwear",
            brand="Urban Thread",
            is_indexed=True,
        )

        mock_embedding_service.return_value.extract_embedding.return_value = np.zeros(2048, dtype="float32")
        mock_faiss_manager.return_value.total_vectors = 1
        mock_faiss_manager.return_value.search.return_value = [(item.id, 0.9123)]

        response = self.client.post(self.url, {"image": _make_uploaded_image()})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["count"], 1)

        result = payload["results"][0]
        self.assertEqual(result["id"], item.id)
        self.assertEqual(result["title"], "Oversized Denim Jacket")
        self.assertEqual(result["category"], "Outerwear")
        self.assertEqual(result["similarity_score"], 0.9123)
        self.assertIn("/media/", result["image_url"])

    @mock.patch("modapp.views.FaissManager")
    @mock.patch("modapp.views.EmbeddingService")
    def test_stale_index_entry_is_skipped(self, mock_embedding_service, mock_faiss_manager):
        """A FAISS id with no matching ClothingItem row is dropped, not a 500."""
        mock_embedding_service.return_value.extract_embedding.return_value = np.zeros(2048, dtype="float32")
        mock_faiss_manager.return_value.total_vectors = 1
        mock_faiss_manager.return_value.search.return_value = [(999_999, 0.5)]

        response = self.client.post(self.url, {"image": _make_uploaded_image()})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"count": 0, "results": []})


# --------------------------------------------------------------------------
# IndexView: dashboard page
# --------------------------------------------------------------------------
class IndexViewTests(TestCase):
    def test_dashboard_renders(self):
        response = self.client.get(reverse("modapp:index"))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "modapp/index.html")
        # The CSRF meta tag must be present for app.js's getCsrfToken() to work.
        self.assertContains(response, 'meta name="csrf-token"')
