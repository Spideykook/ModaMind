# ModaMind — AI Fashion Similarity Search & Stylist Assistant

ModaMind lets a user upload a photo of a clothing item and finds the most
visually similar items in a catalog, using a ResNet50 embedding pipeline and
a FAISS cosine-similarity index, exposed through a Django REST Framework API
and a vanilla HTML/CSS/JS dashboard.

This README covers **Phase 1** (ML + FAISS core) and **Phase 2**
(Django app, API, and frontend dashboard).

```
modamind/
├── manage.py
├── requirements.txt
├── .env / .env.example      <- environment-driven secrets (see Security)
├── config/                  <- project configuration
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
└── modapp/                  <- the single core Django app
    ├── models.py            <- ClothingItem (FAISS id == ClothingItem.id)
    ├── views.py             <- IndexView + SimilaritySearchView (DRF)
    ├── urls.py
    ├── admin.py
    ├── tests.py              <- security & correctness test suite
    ├── management/commands/
    │   └── build_index.py    <- embeds the catalog into FAISS
    ├── ml/                    <- PyTorch / ResNet50 (no Django deps)
    │   ├── transforms.py
    │   ├── model_loader.py
    │   └── embedding_service.py
    ├── search/                <- FAISS (no Django deps)
    │   ├── faiss_manager.py
    │   └── indexes/
    ├── templates/modapp/index.html
    └── static/modapp/{css,js}
scripts/
└── test_pipeline.py          <- standalone ML + FAISS smoke test
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env               # then edit .env for real deployments
python manage.py migrate
python manage.py createsuperuser   # to access /admin/
python manage.py runserver
```

Visit `http://127.0.0.1:8000/` for the dashboard and
`http://127.0.0.1:8000/admin/` to add catalog items.

## Populating the catalog & building the index

1. Add a few `ClothingItem` rows via `/admin/`, each with a photo.
2. Embed them into FAISS:

   ```bash
   python manage.py build_index            # embed only new/un-indexed items
   python manage.py build_index --rebuild  # wipe and re-embed everything
   ```

3. Upload a photo on the dashboard — `SimilaritySearchView` will return the
   Top-5 most visually similar catalog items.

## Verifying the ML/FAISS core in isolation

Before touching HTTP at all, drop a handful of JPG/PNG/WEBP images into
`test_images/` and run:

```bash
python scripts/test_pipeline.py
```

This loads `EmbeddingService` and `FaissManager` directly (no Django, no
running server), embeds every image, builds a throwaway FAISS index, and
runs a self-similarity sanity check (an image should match itself with a
score of ~1.0).

## Running the test suite

```bash
python manage.py test modapp
```

18 tests cover FAISS index integrity (dimension/type validation, add/search/
save/load round-trips), `ClothingItem` upload validators, and every
`SimilaritySearchView` response path (missing file, oversized file,
disallowed content type, disguised non-image file, empty index, stale index
entries, and a successful match). `EmbeddingService`/`FaissManager` are
mocked at the view layer so the suite runs without downloading model
weights.

## Security

ModaMind treats the `/api/search/` endpoint as a public, unauthenticated
surface that accepts file uploads and feeds them into a PyTorch model —
so it's hardened accordingly:

- **Secrets via environment, not source.** `SECRET_KEY`, `DEBUG`, and
  `ALLOWED_HOSTS` are loaded from `.env` (gitignored) via `python-dotenv`.
  If `DEBUG=False` and the secret key is still the insecure placeholder,
  `settings.py` raises `ImproperlyConfigured` instead of starting up.
- **Upload hardening, defense in depth.** Every image — both catalog
  uploads (`ClothingItem.image`) and search queries — passes through:
  1. A hard size cap (`MAX_UPLOAD_SIZE_BYTES`, default 10MB), enforced at
     both the Django (`DATA_UPLOAD_MAX_MEMORY_SIZE`) and view level.
  2. An extension allowlist (`FileExtensionValidator` on the model field;
     manual check in the view) restricted to `jpg/jpeg/png/webp`.
  3. A `Content-Type` allowlist (`image/jpeg`, `image/png`, `image/webp`).
  4. **Content verification** — `Image.open(...).verify()` actually decodes
     the file before it ever reaches the PyTorch pipeline, defeating
     disguised-file attacks (e.g. a script renamed `outfit.jpg`).
- **FAISS index integrity checks.** `FaissManager.load_index()` refuses to
  load a file that isn't an `IndexIDMap2` or whose dimensionality doesn't
  match `EMBEDDING_DIM` (2048), so a corrupted or incompatible index file
  fails loudly instead of producing silently-wrong search results.
- **Rate limiting.** DRF's `AnonRateThrottle` caps anonymous clients at
  `30/minute` on `/api/search/`, since each request triggers a full
  ResNet50 forward pass.
- **CSRF.** `index.html` renders a `<meta name="csrf-token">` tag; `app.js`
  reads it and sends `X-CSRFToken` on every POST.
- **XSS-safe rendering.** `app.js` builds result cards with
  `createElement`/`textContent` only — never `innerHTML` — so catalog data
  (titles, brands, categories) can never be interpreted as markup.
- **Generic error responses.** Internal exceptions are logged server-side
  (`logger.exception(...)`) but the client only ever sees a generic JSON
  `{"error": "..."}` message — no stack traces or internals leak out.
- **Cookie & transport hardening.** `SESSION_COOKIE_SAMESITE` /
  `CSRF_COOKIE_SAMESITE = "Lax"` and `SECURE_REFERRER_POLICY =
  "same-origin"` are always on. When `DEBUG=False`, `settings.py`
  additionally enables HSTS (1 year, with subdomains + preload), secure
  cookies, `SECURE_SSL_REDIRECT`, `X_FRAME_OPTIONS = "DENY"`, and
  MIME-sniffing protection. `CSRF_TRUSTED_ORIGINS` is environment-driven for
  deployments behind a TLS-terminating reverse proxy.
- **`.gitignore` discipline.** `.env`, `db.sqlite3`, uploaded media, and
  generated FAISS `.index` files are all excluded from version control.

## Roadmap (later phases)

- **Phase 3:** Migrate `DATABASES` from SQLite to PostgreSQL (only
  `config/settings.py` needs to change — the app code is database-agnostic).
- **Phase 4:** Add `apps/llm/` — a local Llama 3 (via Ollama) reasoning
  layer that takes the FAISS matches plus the query image's
  metadata/embedding neighborhood and generates a natural-language styling
  rationale ("Why these go together").
