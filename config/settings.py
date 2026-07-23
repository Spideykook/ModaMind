"""
Django settings for the ModaMind project.

Phase 1 / Phase 2 configuration, hardened with environment-driven secrets
and a production security baseline:

- SECRET_KEY, DEBUG, and ALLOWED_HOSTS are read from environment variables,
  loaded from a local .env file via python-dotenv. Never commit a real
  .env file - copy .env.example to .env and fill in real values.
- SQLite for local development (swap to PostgreSQL later by editing
  DATABASES).
- 'modapp' is the single core app, with its own templates/ and static/
  folders discovered automatically via APP_DIRS / AppDirectoriesFinder.
- DRF is configured for multipart image uploads, JSON-only responses, and
  anonymous rate limiting on the similarity-search endpoint.
- Upload size limits and an image content-type/extension allowlist guard
  both the catalog model and the ML pipeline against oversized or
  disguised files.
- When DEBUG is False, a baseline of production security headers (HSTS,
  secure cookies, clickjacking/MIME-sniffing protection) is enabled
  automatically, and a misconfigured SECRET_KEY raises an error instead
  of silently deploying with an insecure default.
"""

import os
from pathlib import Path

from django.core.exceptions import ImproperlyConfigured
from dotenv import load_dotenv

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from a local .env file (gitignored). In
# production, set these directly in the host environment instead
# (systemd unit, Docker env, platform secret manager, etc.).
load_dotenv(BASE_DIR / ".env")


# --------------------------------------------------------------------------
# Core / Security
# --------------------------------------------------------------------------
_INSECURE_DEFAULT_KEY = "django-insecure-change-me-before-deploying-modamind"

SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", _INSECURE_DEFAULT_KEY)

DEBUG = os.environ.get("DJANGO_DEBUG", "True") == "True"

ALLOWED_HOSTS = [
    host.strip()
    for host in os.environ.get("DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")
    if host.strip()
]

# Fail loudly rather than silently running a "production" deployment with
# the placeholder secret key baked into source control.
if not DEBUG and SECRET_KEY == _INSECURE_DEFAULT_KEY:
    raise ImproperlyConfigured(
        "DEBUG is False but DJANGO_SECRET_KEY is unset or still using the "
        "insecure default. Set a unique, random DJANGO_SECRET_KEY in your "
        "environment (see .env.example) before deploying."
    )


# --------------------------------------------------------------------------
# Cookie / referrer / cross-origin hardening (always on, dev and prod)
# --------------------------------------------------------------------------
# 'Lax' is the Django default for both, but set explicitly so the intent
# is visible in source rather than relying on framework defaults that could
# silently change between Django versions.
SESSION_COOKIE_SAMESITE = "Lax"
CSRF_COOKIE_SAMESITE = "Lax"

# Limits the Referer header sent on cross-origin navigations/requests,
# reducing the chance of leaking URLs (which may contain query params)
# to third-party sites.
SECURE_REFERRER_POLICY = "same-origin"

# Required if Django sits behind a reverse proxy / load balancer that
# terminates TLS, so browsers see an https:// origin even though Django's
# own ALLOWED_HOSTS/SECRET_KEY config is unaware of that. Comma-separated
# list of scheme://host entries, e.g.:
#   DJANGO_CSRF_TRUSTED_ORIGINS=https://modamind.example.com
CSRF_TRUSTED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("DJANGO_CSRF_TRUSTED_ORIGINS", "").split(",")
    if origin.strip()
]


# --------------------------------------------------------------------------
# Application definition
# --------------------------------------------------------------------------
INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    # Third-party
    "rest_framework",
    # Local
    "modapp.apps.ModappConfig",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        # Project-level templates dir (optional, for shared/base templates).
        "DIRS": [os.path.join(BASE_DIR, "templates")],
        # APP_DIRS=True means Django also looks inside modapp/templates/,
        # so modapp/templates/modapp/index.html is discovered automatically.
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"


# --------------------------------------------------------------------------
# Database
# --------------------------------------------------------------------------
# Phase 3: PostgreSQL. Connection credentials are sourced from environment
# variables, following the same pattern as SECRET_KEY and DEBUG above.
# Swap these values in your .env file for each deployment environment.
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": os.environ.get("DB_NAME", "modamind_db"),
        "USER": os.environ.get("DB_USER", "modamind_user"),
        "PASSWORD": os.environ.get("DB_PASSWORD", ""),
        "HOST": os.environ.get("DB_HOST", "localhost"),
        "PORT": os.environ.get("DB_PORT", "5432"),
        # Keep persistent DB connections alive for 10 minutes instead of
        # closing them after every request. Dramatically reduces connection
        # overhead under load (each new PostgreSQL connection takes ~5-10ms
        # for TCP + TLS handshake + authentication). Set to 0 to close
        # connections at the end of each request (Django default).
        "CONN_MAX_AGE": int(os.environ.get("DB_CONN_MAX_AGE", "600")),
    }
}

# Fail loudly if a production deployment has no database password configured.
# An empty password is acceptable for local development (peer/trust auth),
# but never for a deployed environment.
if not DEBUG and not DATABASES["default"]["PASSWORD"]:
    raise ImproperlyConfigured(
        "DEBUG is False but DB_PASSWORD is unset. Set a database password "
        "in your environment (see .env.example) before deploying."
    )


# --------------------------------------------------------------------------
# Password validation
# --------------------------------------------------------------------------
AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]


# --------------------------------------------------------------------------
# Internationalization
# --------------------------------------------------------------------------
LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True


# --------------------------------------------------------------------------
# Static files (CSS, JavaScript)
# --------------------------------------------------------------------------
# Served at /static/. Because 'modapp' is in INSTALLED_APPS and
# AppDirectoriesFinder is enabled by default, Django's runserver will
# automatically serve everything under modapp/static/ during development
# (e.g. modapp/static/modapp/css/style.css -> /static/modapp/css/style.css).
STATIC_URL = "static/"

# Reserved for any *project-level* static assets that live outside of an
# individual app (shared vendor libraries, global fonts, etc.). Empty for
# now since all Phase 1/2 assets live in modapp/static/.
STATICFILES_DIRS = []

# Where `collectstatic` gathers files for production deployment.
STATIC_ROOT = BASE_DIR / "staticfiles"


# --------------------------------------------------------------------------
# Media files (user-uploaded clothing images)
# --------------------------------------------------------------------------
MEDIA_URL = "/media/"
MEDIA_ROOT = BASE_DIR / "media"


# --------------------------------------------------------------------------
# Default primary key field type
# --------------------------------------------------------------------------
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"


# --------------------------------------------------------------------------
# Upload hardening
# --------------------------------------------------------------------------
# Hard cap on any single uploaded file / request body. DATA_UPLOAD_MAX_*
# settings make Django itself reject oversized requests before they reach
# a view; SimilaritySearchView additionally checks this for a friendlier
# JSON error message.
MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB

DATA_UPLOAD_MAX_MEMORY_SIZE = MAX_UPLOAD_SIZE_BYTES
FILE_UPLOAD_MAX_MEMORY_SIZE = MAX_UPLOAD_SIZE_BYTES

# Only these are accepted for ClothingItem.image (modapp/models.py) and for
# the similarity-search upload (modapp/views.py). Rejecting everything else
# closes off a common vector for disguised-file attacks (e.g. a script
# renamed with a .jpg extension or a spoofed Content-Type header).
ALLOWED_IMAGE_CONTENT_TYPES = ["image/jpeg", "image/png", "image/webp"]
ALLOWED_IMAGE_EXTENSIONS = ["jpg", "jpeg", "png", "webp"]


# --------------------------------------------------------------------------
# Django REST Framework
# --------------------------------------------------------------------------
REST_FRAMEWORK = {
    "DEFAULT_RENDERER_CLASSES": [
        "rest_framework.renderers.JSONRenderer",
    ],
    "DEFAULT_PARSER_CLASSES": [
        "rest_framework.parsers.MultiPartParser",
        "rest_framework.parsers.FormParser",
        "rest_framework.parsers.JSONParser",
    ],
    # Anonymous clients hitting /api/search/ are rate-limited so the
    # ResNet50 inference pipeline can't be turned into a DoS vector by
    # repeated large-image uploads.
    "DEFAULT_THROTTLE_CLASSES": [
        "rest_framework.throttling.AnonRateThrottle",
    ],
    "DEFAULT_THROTTLE_RATES": {
        "anon": "30/minute",
    },
}


# --------------------------------------------------------------------------
# ModaMind ML / Search configuration
# --------------------------------------------------------------------------
# Centralized path to the persisted FAISS index. FaissManager defaults to
# this location, but accepts an override for testing.
ML_INDEX_PATH = BASE_DIR / "modapp" / "search" / "indexes" / "fashion_items.index"

# Embedding dimensionality produced by EmbeddingService (ResNet50, fc removed).
EMBEDDING_DIM = 2048

# Stored in EmbeddingMetadata to detect stale vectors after a model upgrade.
EMBEDDING_MODEL_NAME = "resnet50"
EMBEDDING_VERSION = "resnet50-imagenet1k-v2-l2norm"


# --------------------------------------------------------------------------
# Ollama / LLM configuration (Phase 4)
# --------------------------------------------------------------------------
# FashionReasoningService uses these to connect to a local Ollama instance
# running Llama 3.  The LLM layer is optional — if Ollama is unreachable,
# the similarity search still works; the user just doesn't get AI styling
# advice (graceful degradation).

# Base URL for the Ollama REST API.
#   Local:  http://localhost:11434  (default Ollama port)
#   Remote: http://gpu-server:11434
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# Which model Ollama should use for generation.
#   ollama pull llama3       → "llama3"
#   ollama pull llama3:8b    → "llama3:8b"
#   ollama pull mistral      → "mistral"
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3")

# HTTP timeout (seconds) for a single Ollama /api/generate call.
# LLM inference is slow (5-30s typical, 60s+ on first cold start).
OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "120"))

# Sampling temperature (0.0 = deterministic, 1.0 = very creative).
# 0.7 balances coherence with variety for fashion advice.
OLLAMA_TEMPERATURE = float(os.environ.get("OLLAMA_TEMPERATURE", "0.7"))

# Feature toggle.  Set to "False" to disable AI reasoning entirely
# (useful if Ollama is not installed or if you want faster responses
# during development/testing).
OLLAMA_ENABLED = os.environ.get("OLLAMA_ENABLED", "True") == "True"

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
# Ensures EmbeddingService / FaissManager / build_index `logger.info(...)`
# and `logger.exception(...)` calls are actually visible during development
# instead of being silently dropped by Django's default logging config.
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {
        "console": {"class": "logging.StreamHandler"},
    },
    "root": {
        "handlers": ["console"],
        "level": "INFO",
    },
}


# --------------------------------------------------------------------------
# Production security baseline
# --------------------------------------------------------------------------
# These are no-ops in local development (DEBUG=True) but harden the app
# automatically the moment DEBUG is switched off for a real deployment.
# If TLS terminates at a reverse proxy that already handles HTTP->HTTPS
# redirects, set SECURE_SSL_REDIRECT = False via an env-driven override
# instead of editing this block.
if not DEBUG:
    SECURE_SSL_REDIRECT = True
    SESSION_COOKIE_SECURE = True
    CSRF_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SECURE_CONTENT_TYPE_NOSNIFF = True
    SECURE_BROWSER_XSS_FILTER = True
    X_FRAME_OPTIONS = "DENY"
    SECURE_HSTS_SECONDS = 31536000  # 1 year
    SECURE_HSTS_INCLUDE_SUBDOMAINS = True
    SECURE_HSTS_PRELOAD = True
