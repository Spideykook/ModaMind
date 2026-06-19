"use strict";

/**
 * ModaMind — Frontend interactivity (app.js)
 *
 * Completely decoupled from index.html. Reads the page via stable DOM IDs.
 *
 * Responsibilities:
 *   1. File intake  — click-to-browse + drag-and-drop, client-side validation.
 *   2. Preview      — FileReader thumbnail, reset flow.
 *   3. AJAX search  — async fetch POST to /api/search/ with X-CSRFToken header.
 *   4. State UI     — loader show/hide, error bar, results count label.
 *   5. Card render  — createElement / textContent only (no innerHTML, no XSS).
 *
 * Security notes:
 *   - getCsrfToken() reads from <meta name="csrf-token"> rendered by Django.
 *   - X-CSRFToken is sent on every POST so DRF's CsrfViewMiddleware passes.
 *   - All user-visible strings from the API (name, brand, category, color)
 *     are written via textContent, never innerHTML.
 *
 * Stack: Vanilla ES2017+, Fetch API, no build step, no frameworks.
 */
(() => {

  /* ── Constants ─────────────────────────────────────────────────────────── */
  const SEARCH_ENDPOINT     = "/api/search/";
  const MAX_FILE_BYTES      = 10 * 1024 * 1024;   // 10 MB — mirrors server setting
  const ALLOWED_MIME_TYPES  = ["image/jpeg", "image/png", "image/webp"];

  /* ── DOM refs ──────────────────────────────────────────────────────────── */
  const dropZone        = document.getElementById("drop-zone");
  const fileInput       = document.getElementById("file-input");
  const dropInner       = document.getElementById("drop-inner");
  const previewWrapper  = document.getElementById("preview-wrapper");
  const previewImage    = document.getElementById("preview-image");
  const resetBtn        = document.getElementById("reset-btn");
  const loaderSection   = document.getElementById("loader-section");
  const errorBanner     = document.getElementById("error-banner");
  const errorMsg        = document.getElementById("error-message");
  const resultsSection  = document.getElementById("results-section");
  const resultsGrid     = document.getElementById("results-grid");
  const resultsCount    = document.getElementById("results-count");

  /* ── CSRF ──────────────────────────────────────────────────────────────── */
  /**
   * Reads the Django CSRF token from <meta name="csrf-token">.
   * Django's CsrfViewMiddleware requires this on non-safe HTTP methods.
   * DRF's SimilaritySearchView (a POST endpoint) enforces it via the
   * middleware, so we always send X-CSRFToken with every fetch.
   */
  function getCsrfToken() {
    const meta = document.querySelector('meta[name="csrf-token"]');
    return meta ? meta.getAttribute("content") : "";
  }

  /* ── UI state helpers ──────────────────────────────────────────────────── */
  function showError(message) {
    errorMsg.textContent = message;
    errorBanner.hidden   = false;
    errorBanner.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function hideError() {
    errorBanner.hidden  = true;
    errorMsg.textContent = "";
  }

  function showLoader() { loaderSection.hidden = false; }
  function hideLoader() { loaderSection.hidden = true;  }

  function clearResults() {
    resultsGrid.replaceChildren();
    resultsSection.hidden = true;
    if (resultsCount) resultsCount.textContent = "";
  }

  /* ── Client-side validation ────────────────────────────────────────────── */
  /**
   * Returns a user-facing error string or null if the file passes.
   * These checks are a UX nicety — the server re-validates everything.
   */
  function validateFile(file) {
    if (!ALLOWED_MIME_TYPES.includes(file.type)) {
      return `Unsupported type "${file.type || "unknown"}". Use JPG, PNG, or WEBP.`;
    }
    if (file.size > MAX_FILE_BYTES) {
      return `File too large (${(file.size / 1024 / 1024).toFixed(1)} MB). Max is 10 MB.`;
    }
    return null;
  }

  /* ── Preview ───────────────────────────────────────────────────────────── */
  function showPreview(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
      previewImage.src    = e.target.result;
      previewImage.alt    = `Preview: ${file.name}`;
      dropZone.hidden     = true;
      previewWrapper.hidden = false;
    };
    reader.readAsDataURL(file);
  }

  function resetUpload() {
    fileInput.value = "";
    previewImage.removeAttribute("src");
    previewWrapper.hidden = true;
    dropZone.hidden       = false;
    hideError();
    hideLoader();
    clearResults();
  }

  /* ── Card construction ─────────────────────────────────────────────────── */
  /**
   * Builds one result card from an API result object.
   * All catalog strings are assigned via textContent — never innerHTML —
   * so they are always rendered as plain text, preventing XSS even if the
   * catalog ever contains untrusted data.
   *
   * @param {{ id:number, name:string, category:string, brand:string,
   *           color:string, image_url:string|null, similarity_score:number }} item
   * @param {number} rank  1-based position in the result list
   */
  function createResultCard(item, rank) {
    const score        = Math.max(0, Math.min(1, item.similarity_score ?? 0));
    const scorePct     = Math.round(score * 100);
    const metaParts    = [item.category, item.brand, item.color].filter(Boolean);

    /* ── Article shell ── */
    const card = document.createElement("article");
    card.className = "result-card";
    card.style.animationDelay = `${(rank - 1) * 55}ms`;

    /* ── Media block ── */
    const media = document.createElement("div");
    media.className = "result-card__media";

    const img = document.createElement("img");
    img.className   = "result-card__image";
    img.loading     = "lazy";
    img.decoding    = "async";
    img.alt         = item.name || `Item ${item.id}`;
    if (item.image_url) img.src = item.image_url;
    media.appendChild(img);

    /* Rank badge */
    const rankBadge = document.createElement("span");
    rankBadge.className       = "result-card__rank";
    rankBadge.setAttribute("aria-hidden", "true");
    rankBadge.textContent     = String(rank);
    media.appendChild(rankBadge);

    /* Score badge */
    const scoreBadge = document.createElement("span");
    scoreBadge.className   = "result-card__score";
    scoreBadge.textContent = `${scorePct}%`;
    media.appendChild(scoreBadge);

    card.appendChild(media);

    /* ── Body ── */
    const body = document.createElement("div");
    body.className = "result-card__body";

    const nameEl = document.createElement("p");
    nameEl.className   = "result-card__name";
    nameEl.textContent = item.name || `Item #${item.id}`;
    body.appendChild(nameEl);

    if (metaParts.length > 0) {
      const metaEl = document.createElement("p");
      metaEl.className   = "result-card__meta";
      metaEl.textContent = metaParts.join(" · ");
      body.appendChild(metaEl);
    }

    /* Match bar */
    const bar = document.createElement("div");
    bar.className = "match-bar";

    const track = document.createElement("div");
    track.className = "match-bar__track";
    const fill = document.createElement("div");
    fill.className = "match-bar__fill";
    track.appendChild(fill);
    bar.appendChild(track);

    const labels = document.createElement("div");
    labels.className = "match-bar__labels";
    const labelLeft = document.createElement("span");
    labelLeft.textContent = "Similarity";
    const labelRight = document.createElement("strong");
    labelRight.textContent = `${scorePct}%`;
    labels.appendChild(labelLeft);
    labels.appendChild(labelRight);
    bar.appendChild(labels);

    body.appendChild(bar);
    card.appendChild(body);

    /* Animate bar fill after paint */
    requestAnimationFrame(() => {
      fill.style.width = `${scorePct}%`;
    });

    return card;
  }

  /* ── Results rendering ─────────────────────────────────────────────────── */
  function renderResults(results) {
    resultsGrid.replaceChildren();

    if (!Array.isArray(results) || results.length === 0) {
      const empty = document.createElement("div");
      empty.className = "results-empty";
      const heading = document.createElement("strong");
      heading.textContent = "No close matches found";
      const hint = document.createElement("span");
      hint.textContent = "Try a different photo or add more items to the catalog.";
      empty.appendChild(heading);
      empty.appendChild(hint);
      resultsGrid.appendChild(empty);

      if (resultsCount) resultsCount.textContent = "0 results";
    } else {
      results.forEach((item, index) => {
        resultsGrid.appendChild(createResultCard(item, index + 1));
      });
      if (resultsCount) {
        resultsCount.textContent = `${results.length} result${results.length === 1 ? "" : "s"}`;
      }
    }

    resultsSection.hidden = false;
    resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  /* ── AJAX search (Fetch API + CSRF) ────────────────────────────────────── */
  /**
   * Posts the selected file to /api/search/ as multipart/form-data.
   *
   * CSRF flow:
   *   1. Django renders {{ csrf_token }} into <meta name="csrf-token"> in index.html.
   *   2. getCsrfToken() reads it here.
   *   3. X-CSRFToken header is sent with every POST so Django's
   *      CsrfViewMiddleware (and DRF's enforce_csrf) accept the request.
   *
   * DRF response shapes this view handles:
   *   200 { count, results }
   *   400 { error }  — bad file
   *   413 { error }  — file too large
   *   503 { error }  — index empty
   */
  async function runSearch(file) {
    hideError();
    clearResults();
    showLoader();

    const formData = new FormData();
    formData.append("image", file);

    try {
      const response = await fetch(SEARCH_ENDPOINT, {
        method:  "POST",
        headers: { "X-CSRFToken": getCsrfToken() },
        body:    formData,
        // Do NOT set Content-Type manually — the browser sets it to
        // multipart/form-data with the correct boundary automatically.
      });

      let payload = null;
      try {
        payload = await response.json();
      } catch {
        /* Non-JSON body (e.g. 502 gateway page) — fall through to generic msg */
      }

      if (!response.ok) {
        const msg = (payload && payload.error)
          || `Search failed (HTTP ${response.status}). Please try again.`;
        showError(msg);
        return;
      }

      renderResults(payload?.results ?? []);

    } catch {
      /* Network-level failure (offline, DNS, CORS, etc.) */
      showError("Couldn't reach the server. Check your connection and try again.");
    } finally {
      hideLoader();
    }
  }

  /* ── File intake ───────────────────────────────────────────────────────── */
  function handleFile(file) {
    if (!file) return;
    const err = validateFile(file);
    if (err) { showError(err); return; }
    hideError();
    showPreview(file);
    runSearch(file);
  }

  /* ── Event wiring ──────────────────────────────────────────────────────── */
  function init() {
    /* Click / keyboard → open file picker */
    dropZone.addEventListener("click",  () => fileInput.click());
    dropZone.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        fileInput.click();
      }
    });

    /* File input change */
    fileInput.addEventListener("change", (e) => {
      const [file] = e.target.files;
      handleFile(file);
    });

    /* Drag-and-drop */
    ["dragenter", "dragover"].forEach((evt) => {
      dropZone.addEventListener(evt, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.add("drop-zone--active");
      });
    });
    ["dragleave", "drop"].forEach((evt) => {
      dropZone.addEventListener(evt, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.remove("drop-zone--active");
      });
    });
    dropZone.addEventListener("drop", (e) => {
      const file = e.dataTransfer?.files?.[0];
      handleFile(file);
    });

    /* Reset button */
    resetBtn.addEventListener("click", resetUpload);
  }

  document.addEventListener("DOMContentLoaded", init);

})();
