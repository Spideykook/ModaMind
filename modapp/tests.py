"""
modapp/tests.py — Security and correctness tests for ModaMind.

Coverage:
  - FaissManager: index integrity and round-trip persistence.
  - Category model: __str__, auto-capitalise, slug uniqueness.
  - ClothingItem model: upload validators, __str__, display_category.
  - EmbeddingMetadata model: OneToOne relationship and cascade.
  - SimilaritySearchView: every request-validation branch + happy path
    + category-filtered search (valid slug, unknown slug, empty category).
  - IndexView: dashboard renders with CSRF meta tag.

SimilaritySearchService is mocked at the view layer so the
suite runs without downloading ResNet50 weights.
"""
from __future__ import annotations
import io, os, tempfile
from unittest import mock
import faiss, numpy as np
from django.core.exceptions import ValidationError
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.urls import reverse
from PIL import Image
from .models import Category, ClothingItem, EmbeddingMetadata
from .search.faiss_manager import FaissManager
from .search.similarity_service import SearchResponse, SearchResult


# ── helpers ──────────────────────────────────────────────────────────────────
def _img_bytes(fmt="JPEG", size=(64, 64), noisy=False):
    if noisy:
        rng = np.random.default_rng(0)
        arr = (rng.random((size[1], size[0], 3)) * 255).astype("uint8")
        img = Image.fromarray(arr, "RGB")
    else:
        img = Image.new("RGB", size, color=(200, 100, 50))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def _upload(name="t.jpg", fmt="JPEG", ct="image/jpeg", size=(64,64), noisy=False):
    return SimpleUploadedFile(name, _img_bytes(fmt=fmt, size=size, noisy=noisy), content_type=ct)


def _cat(slug="tops", name="Tops"):
    return Category.objects.create(slug=slug, name=name)


# ── FaissManager ─────────────────────────────────────────────────────────────
class FaissManagerTests(TestCase):
    def setUp(self):
        FaissManager._instance = None
        self._tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._tmp.name, "t.index")

    def tearDown(self):
        FaissManager._instance = None
        self._tmp.cleanup()

    def test_starts_empty(self):
        m = FaissManager(index_path=self.path)
        self.assertEqual(m.total_vectors, 0)
        self.assertEqual(m.search(np.zeros(2048, dtype="float32")), [])

    def test_add_search_round_trip(self):
        m = FaissManager(index_path=self.path)
        rng = np.random.default_rng(7)
        v = rng.standard_normal((3, 2048)).astype("float32")
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        m.add_vectors(v, ids=np.array([10, 20, 30]))
        self.assertEqual(m.total_vectors, 3)
        res = m.search(v[0], top_k=1)
        self.assertEqual(res[0][0], 10)
        self.assertAlmostEqual(res[0][1], 1.0, places=4)

    def test_save_load(self):
        m = FaissManager(index_path=self.path)
        m.add_vectors(np.eye(2048, dtype="float32")[:2], ids=np.array([1, 2]))
        m.save_index()
        FaissManager._instance = None
        m2 = FaissManager(index_path=self.path)
        self.assertEqual(m2.total_vectors, 2)

    def test_rejects_wrong_dim_add(self):
        m = FaissManager(index_path=self.path)
        with self.assertRaises(ValueError):
            m.add_vectors(np.zeros(16, dtype="float32"), ids=np.array([1]))

    def test_rejects_wrong_dim_search(self):
        m = FaissManager(index_path=self.path)
        m.add_vectors(np.eye(2048, dtype="float32")[:1], ids=np.array([1]))
        with self.assertRaises(ValueError):
            m.search(np.zeros(16, dtype="float32"))

    def test_rejects_dim_mismatch_on_load(self):
        faiss.write_index(faiss.IndexIDMap2(faiss.IndexFlatIP(16)), self.path)
        FaissManager._instance = None
        with self.assertRaises(ValueError):
            FaissManager(index_path=self.path)

    def test_rejects_non_idmap_on_load(self):
        faiss.write_index(faiss.IndexFlatIP(2048), self.path)
        FaissManager._instance = None
        with self.assertRaises(TypeError):
            FaissManager(index_path=self.path)


# ── Category ──────────────────────────────────────────────────────────────────
class CategoryTests(TestCase):
    def test_str(self):
        self.assertEqual(str(Category(slug="tops", name="Tops")), "Tops")

    def test_auto_capitalise(self):
        c = Category.objects.create(slug="dresses", name="dresses")
        c.refresh_from_db()
        self.assertEqual(c.name, "Dresses")

    def test_slug_unique(self):
        Category.objects.create(slug="tops", name="Tops")
        from django.db import IntegrityError
        with self.assertRaises(IntegrityError):
            Category.objects.create(slug="tops", name="Tops2")


# ── ClothingItem ──────────────────────────────────────────────────────────────
class ClothingItemTests(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._tmp = tempfile.TemporaryDirectory()
        cls._ov  = override_settings(MEDIA_ROOT=cls._tmp.name)
        cls._ov.enable()

    @classmethod
    def tearDownClass(cls):
        cls._ov.disable()
        cls._tmp.cleanup()
        super().tearDownClass()

    def test_str_name_and_brand(self):
        i = ClothingItem(name="Silk Blouse", brand="Acne")
        self.assertEqual(str(i), "Silk Blouse — Acne")

    def test_str_name_only(self):
        i = ClothingItem(name="Trousers")
        self.assertEqual(str(i), "Trousers")

    def test_str_fallback(self):
        i = ClothingItem(); i.pk = 3
        self.assertEqual(str(i), "Item #3")

    def test_display_category_with(self):
        c = _cat("bottoms", "Bottoms")
        i = ClothingItem(name="Jeans", category=c)
        self.assertEqual(i.display_category, "Bottoms")

    def test_display_category_without(self):
        self.assertEqual(ClothingItem(name="X").display_category, "Uncategorised")

    def test_rejects_gif(self):
        i = ClothingItem(image=_upload("p.gif","GIF","image/gif"), name="bad")
        with self.assertRaises(ValidationError):
            i.full_clean()

    @override_settings(MAX_UPLOAD_SIZE_BYTES=1024)
    def test_rejects_oversized(self):
        big = _upload(size=(256,256), noisy=True)
        self.assertGreater(big.size, 1024)
        i = ClothingItem(image=big, name="big")
        with self.assertRaises(ValidationError):
            i.full_clean()

    def test_accepts_valid_jpeg(self):
        c = _cat()
        i = ClothingItem(image=_upload(), name="ok", category=c)
        i.full_clean(exclude=["category"])


# ── EmbeddingMetadata ─────────────────────────────────────────────────────────
class EmbeddingMetadataTests(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._tmp = tempfile.TemporaryDirectory()
        cls._ov  = override_settings(MEDIA_ROOT=cls._tmp.name)
        cls._ov.enable()

    @classmethod
    def tearDownClass(cls):
        cls._ov.disable()
        cls._tmp.cleanup()
        super().tearDownClass()

    def _item(self):
        return ClothingItem.objects.create(image=_upload(), name="Test", category=_cat())

    def test_str(self):
        m = EmbeddingMetadata(clothing_item_id=5, model_name="resnet50", embedding_version="v1")
        self.assertIn("resnet50", str(m))

    def test_one_to_one(self):
        item = self._item()
        meta = EmbeddingMetadata.objects.create(
            clothing_item=item, model_name="resnet50", embedding_version="v1")
        self.assertEqual(item.embedding_metadata, meta)

    def test_cascade_delete(self):
        item = self._item()
        EmbeddingMetadata.objects.create(
            clothing_item=item, model_name="resnet50", embedding_version="v1")
        pk = item.pk
        item.delete()
        self.assertFalse(EmbeddingMetadata.objects.filter(clothing_item_id=pk).exists())


# ── SimilaritySearchView ──────────────────────────────────────────────────────
class SearchViewTests(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._tmp = tempfile.TemporaryDirectory()
        cls._ov  = override_settings(MEDIA_ROOT=cls._tmp.name)
        cls._ov.enable()

    @classmethod
    def tearDownClass(cls):
        cls._ov.disable()
        cls._tmp.cleanup()
        super().tearDownClass()

    def setUp(self):
        self.url = reverse("modapp:similarity-search")

    def test_no_file_400(self):
        r = self.client.post(self.url, {})
        self.assertEqual(r.status_code, 400)
        self.assertIn("error", r.json())

    @override_settings(MAX_UPLOAD_SIZE_BYTES=1024)
    def test_oversized_413(self):
        big = _upload(size=(256,256), noisy=True)
        self.assertGreater(big.size, 1024)
        r = self.client.post(self.url, {"image": big})
        self.assertEqual(r.status_code, 413)

    def test_bad_content_type_400(self):
        txt = SimpleUploadedFile("n.txt", b"hi", content_type="text/plain")
        r = self.client.post(self.url, {"image": txt})
        self.assertEqual(r.status_code, 400)
        self.assertIn("Unsupported", r.json()["error"])

    def test_disguised_file_400(self):
        fake = SimpleUploadedFile("x.jpg", b"not-an-image", content_type="image/jpeg")
        r = self.client.post(self.url, {"image": fake})
        self.assertEqual(r.status_code, 400)
        self.assertIn("not a valid image", r.json()["error"])

    @mock.patch("modapp.views.SimilaritySearchService")
    def test_empty_index_503(self, mock_service_cls):
        mock_service_cls.return_value.search_by_image.return_value = SearchResponse(
            error="The similarity index is empty. Seed the catalog and run build_index first.",
        )
        r = self.client.post(self.url, {"image": _upload()})
        self.assertEqual(r.status_code, 503)

    @mock.patch("modapp.views.SimilaritySearchService")
    def test_happy_path(self, mock_service_cls):
        cat  = _cat("outerwear", "Outerwear")
        item = ClothingItem.objects.create(
            image=_upload(name="j.jpg"), name="Denim Jacket",
            category=cat, brand="Urban Thread", color="Indigo", is_indexed=True)
        mock_service_cls.return_value.search_by_image.return_value = SearchResponse(
            results=[SearchResult(item_id=item.id, similarity_score=0.91)],
            total=1,
        )

        r = self.client.post(self.url, {"image": _upload()})
        self.assertEqual(r.status_code, 200)
        res = r.json()["results"][0]
        self.assertEqual(res["name"],     "Denim Jacket")
        self.assertEqual(res["category"], "Outerwear")
        self.assertEqual(res["brand"],    "Urban Thread")
        self.assertEqual(res["color"],    "Indigo")
        self.assertAlmostEqual(res["similarity_score"], 0.91)

    @mock.patch("modapp.views.SimilaritySearchService")
    def test_stale_index_skipped(self, mock_service_cls):
        mock_service_cls.return_value.search_by_image.return_value = SearchResponse(
            results=[SearchResult(item_id=999_999, similarity_score=0.5)],
            total=1,
        )
        r = self.client.post(self.url, {"image": _upload()})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"count": 0, "results": []})

    # ── Category filter tests ────────────────────────────────────────────────

    def test_unknown_category_400(self):
        """Sending a category slug that doesn't exist should return 400."""
        r = self.client.post(self.url, {"image": _upload(), "category": "nonexistent"})
        self.assertEqual(r.status_code, 400)
        self.assertIn("Unknown category slug", r.json()["error"])

    @mock.patch("modapp.views.SimilaritySearchService")
    def test_category_filter_happy_path(self, mock_service_cls):
        """A valid category slug should pass allowed_ids to the service."""
        cat = _cat("outerwear", "Outerwear")
        item = ClothingItem.objects.create(
            image=_upload(name="c.jpg"), name="Parka",
            category=cat, brand="Snow", color="Black", is_indexed=True,
        )
        mock_service_cls.return_value.search_by_image.return_value = SearchResponse(
            results=[SearchResult(item_id=item.id, similarity_score=0.88)],
            total=1,
            category_filter=1,
        )

        r = self.client.post(self.url, {"image": _upload(), "category": "outerwear"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["results"][0]["name"], "Parka")

        # Verify allowed_ids was passed to the service.
        call_kwargs = mock_service_cls.return_value.search_by_image.call_args
        self.assertIn("allowed_ids", call_kwargs.kwargs)
        self.assertIsNotNone(call_kwargs.kwargs["allowed_ids"])
        self.assertIn(item.id, call_kwargs.kwargs["allowed_ids"])

    @mock.patch("modapp.views.SimilaritySearchService")
    def test_category_filter_no_indexed_items(self, mock_service_cls):
        """Category exists but has no indexed items → empty allowed_ids, empty results."""
        _cat("shoes", "Shoes")  # exists but no ClothingItem in it
        mock_service_cls.return_value.search_by_image.return_value = SearchResponse(
            results=[], total=0, category_filter=0,
        )

        r = self.client.post(self.url, {"image": _upload(), "category": "shoes"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["count"], 0)

        # Verify the service received an empty allowed_ids set.
        call_kwargs = mock_service_cls.return_value.search_by_image.call_args
        self.assertEqual(call_kwargs.kwargs["allowed_ids"], set())


# ── IndexView ─────────────────────────────────────────────────────────────────
class IndexViewTests(TestCase):
    def test_dashboard_renders(self):
        r = self.client.get(reverse("modapp:index"))
        self.assertEqual(r.status_code, 200)
        self.assertTemplateUsed(r, "modapp/index.html")
        self.assertContains(r, 'meta name="csrf-token"')
