"use strict";

/**
 * ModaMind dashboard interactivity.
 *
 * - Lets the user pick an image via drag-and-drop or click-to-browse.
 * - Validates the file client-side (type + size) before sending it.
 * - Shows an "AI analysis" loading state while the request is in flight.
 * - POSTs the image to /api/search/ as multipart/form-data via fetch.
 * - Renders the returned matches as result cards, building the DOM with
 *   createElement/textContent (never innerHTML) so catalog data such as
 *   item titles and brand names can never be interpreted as markup.
 *
 * No build step, no frameworks — plain ES2017+ running directly in the
 * browser, loaded as a classic <script> at the end of <body>.
 */
(() => {
  const SEARCH_ENDPOINT = "/api/search/";

  // Mirrors settings.MAX_UPLOAD_SIZE_BYTES / ALLOWED_IMAGE_CONTENT_TYPES.
  // Client-side checks are a UX nicety only — the server re-validates
  // everything regardless.
  const MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024; // 10 MB
  const ALLOWED_MIME_TYPES = ["image/jpeg", "image/png", "image/webp"];

  // --- DOM references -------------------------------------------------------
  const dropZone = document.getElementById("drop-zone");
  const fileInput = document.getElementById("file-input");
  const previewWrapper = document.getElementById("preview-wrapper");
  const previewImage = document.getElementById("preview-image");
  const resetButton = document.getElementById("reset-btn");

  const loaderSection = document.getElementById("loader-section");
  const errorBanner = document.getElementById("error-banner");
  const errorMessage = document.getElementById("error-message");
  const resultsSection = document.getElementById("results-section");
  const resultsGrid = document.getElementById("results-grid");

  // --- Small helpers ----------------------------------------------------

  /** Reads the CSRF token Django renders into a <meta> tag in <head>. */
  function getCsrfToken() {
    const meta = document.querySelector('meta[name="csrf-token"]');
    return meta ? meta.content : "";
  }

  function showError(message) {
    errorMessage.textContent = message;
    errorBanner.hidden = false;
  }

  function hideError() {
    errorBanner.hidden = true;
    errorMessage.textContent = "";
  }

  function showLoader() {
    loaderSection.hidden = false;
  }

  function hideLoader() {
    loaderSection.hidden = true;
  }

  function clearResults() {
    resultsGrid.replaceChildren();
    resultsSection.hidden = true;
  }

  // --- Client-side validation -------------------------------------------

  /** Returns a user-facing error string, or null if the file looks fine. */
  function validateFile(file) {
    if (!ALLOWED_MIME_TYPES.includes(file.type)) {
      return `Unsupported file type "${file.type || "unknown"}". Please use a JPG, PNG, or WEBP image.`;
    }
    if (file.size > MAX_FILE_SIZE_BYTES) {
      const maxMb = MAX_FILE_SIZE_BYTES / (1024 * 1024);
      return `That image is too large. Please choose a file under ${maxMb}MB.`;
    }
    return null;
  }

  // --- Preview / reset ----------------------------------------------------

  function showPreview(file) {
    const reader = new FileReader();
    reader.onload = (event) => {
      previewImage.src = event.target.result;
      dropZone.hidden = true;
      previewWrapper.hidden = false;
    };
    reader.readAsDataURL(file);
  }

  function resetUpload() {
    fileInput.value = "";
    previewImage.removeAttribute("src");
    previewWrapper.hidden = true;
    dropZone.hidden = false;
    hideError();
    hideLoader();
    clearResults();
  }

  // --- Result card construction -------------------------------------------

  /**
   * Builds a single result card via DOM APIs only. All catalog-provided
   * strings (title, category, brand) go through `textContent`, so they
   * are always rendered as plain text and can never inject HTML/scripts —
   * even if the admin catalog later contains untrusted data.
   */
  function createResultCard(item) {
    const card = document.createElement("article");
    card.className = "result-card";

    // --- Media: thumbnail + similarity badge -----------------------
    const media = document.createElement("div");
    media.className = "result-card__media";

    const img = document.createElement("img");
    img.className = "result-card__image";
    img.loading = "lazy";
    img.alt = item.title || "Clothing item";
    if (item.image_url) {
      img.src = item.image_url;
    }
    media.appendChild(img);

    // Cosine similarity is in [-1, 1]; clamp to [0, 1] for display.
    const clampedScore = Math.max(0, Math.min(1, item.similarity_score ?? 0));
    const scorePercent = Math.round(clampedScore * 100);

    const scoreBadge = document.createElement("span");
    scoreBadge.className = "result-card__score";
    scoreBadge.textContent = `${scorePercent}% match`;
    media.appendChild(scoreBadge);

    card.appendChild(media);

    // --- Body: title, metadata, match meter -------------------------
    const body = document.createElement("div");
    body.className = "result-card__body";

    const title = document.createElement("h3");
    title.className = "result-card__title";
    title.textContent = item.title || `Item #${item.id}`;
    body.appendChild(title);

    const metaParts = [item.category, item.brand].filter(Boolean);
    if (metaParts.length > 0) {
      const meta = document.createElement("p");
      meta.className = "result-card__meta";
      meta.textContent = metaParts.join(" · ");
      body.appendChild(meta);
    }

    const meter = document.createElement("div");
    meter.className = "match-meter";
    const fill = document.createElement("div");
    fill.className = "match-meter__fill";
    meter.appendChild(fill);
    body.appendChild(meter);

    card.appendChild(body);

    // Defer setting the width so the CSS transition animates the fill in.
    requestAnimationFrame(() => {
      fill.style.width = `${scorePercent}%`;
    });

    return card;
  }

  function renderResults(results) {
    resultsGrid.replaceChildren();

    if (!results || results.length === 0) {
      const empty = document.createElement("div");
      empty.className = "results-empty";

      const heading = document.createElement("strong");
      heading.textContent = "No close matches yet";
      empty.appendChild(heading);

      const hint = document.createElement("span");
      hint.textContent = "Try a different photo, or add more items to the catalog.";
      empty.appendChild(hint);

      resultsGrid.appendChild(empty);
    } else {
      results.forEach((item, index) => {
        const card = createResultCard(item);
        card.style.animationDelay = `${index * 60}ms`;
        resultsGrid.appendChild(card);
      });
    }

    resultsSection.hidden = false;
  }

  // --- Network request ----------------------------------------------------

  async function runSimilaritySearch(file) {
    hideError();
    clearResults();
    showLoader();

    const formData = new FormData();
    formData.append("image", file);

    try {
      const response = await fetch(SEARCH_ENDPOINT, {
        method: "POST",
        // DRF's anonymous SimilaritySearchView doesn't require this, but
        // sending it costs nothing and keeps the request CSRF-safe if
        // session authentication is ever added later.
        headers: {
          "X-CSRFToken": getCsrfToken(),
        },
        body: formData,
      });

      let payload = null;
      try {
        payload = await response.json();
      } catch {
        // Non-JSON response (e.g. a 5xx error page) — fall through to the
        // generic error message below.
      }

      if (!response.ok) {
        const message =
          (payload && payload.error) || `Search failed (HTTP ${response.status}). Please try again.`;
        showError(message);
        return;
      }

      renderResults(payload ? payload.results : []);
    } catch {
      showError("Couldn't reach ModaMind's server. Check your connection and try again.");
    } finally {
      hideLoader();
    }
  }

  // --- File intake -----------------------------------------------------

  function handleFile(file) {
    if (!file) {
      return;
    }

    const validationError = validateFile(file);
    if (validationError) {
      showError(validationError);
      return;
    }

    hideError();
    showPreview(file);
    runSimilaritySearch(file);
  }

  // --- Event wiring --------------------------------------------------------

  function init() {
    // Click or keyboard activation on the drop zone opens the file picker.
    dropZone.addEventListener("click", () => fileInput.click());
    dropZone.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        fileInput.click();
      }
    });

    fileInput.addEventListener("change", (event) => {
      const [file] = event.target.files;
      handleFile(file);
    });

    // Drag-and-drop visual feedback + intake.
    ["dragenter", "dragover"].forEach((eventName) => {
      dropZone.addEventListener(eventName, (event) => {
        event.preventDefault();
        event.stopPropagation();
        dropZone.classList.add("upload-zone--active");
      });
    });

    ["dragleave", "drop"].forEach((eventName) => {
      dropZone.addEventListener(eventName, (event) => {
        event.preventDefault();
        event.stopPropagation();
        dropZone.classList.remove("upload-zone--active");
      });
    });

    dropZone.addEventListener("drop", (event) => {
      const file = event.dataTransfer.files[0];
      handleFile(file);
    });

    resetButton.addEventListener("click", resetUpload);
  }

  document.addEventListener("DOMContentLoaded", init);
})();
