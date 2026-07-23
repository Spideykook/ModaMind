"""
Ollama HTTP client for LLM inference.

Handles all communication with the Ollama REST API (/api/generate).
This module has no Django dependencies and no fashion-domain knowledge —
it is a pure transport layer, analogous to how FaissManager is a pure
FAISS wrapper with no search-domain knowledge.

The client is intentionally NOT a singleton (unlike ModelLoader and
FaissManager) because:
  - It holds no heavyweight in-memory state (no model weights, no index).
  - Each call is a stateless HTTP request.
  - Configuration (URL, model, timeouts) may vary per caller.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response dataclass
# ---------------------------------------------------------------------------

@dataclass
class OllamaResponse:
    """
    Structured response from an Ollama /api/generate call.

    Attributes:
        text:           The generated text content from the LLM.
        model:          Which model produced the response (echoed back).
        total_duration: Total time reported by Ollama (nanoseconds).
        ok:             True if the call succeeded without error.
        error:          Error description if the call failed.
        raw:            Full JSON response dict from Ollama (for debugging).
    """

    text: str = ""
    model: str = ""
    total_duration: int = 0
    ok: bool = True
    error: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        """Total duration converted to milliseconds."""
        return self.total_duration / 1_000_000.0


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class OllamaClient:
    """
    HTTP client for the Ollama LLM inference API.

    Sends prompts to Ollama's /api/generate endpoint and returns
    structured responses.  Handles:
      - Connection errors (Ollama not running)
      - Timeouts (LLM taking too long)
      - HTTP errors (bad model name, server errors)
      - JSON parsing failures

    All errors are captured in OllamaResponse.error rather than raised,
    following the same pattern as SimilaritySearchService.SearchResponse.
    This lets the caller decide how to surface errors (HTTP 503, log
    warning, retry, etc.).

    Usage:
        client = OllamaClient(base_url="http://localhost:11434")
        response = client.generate(
            prompt="Why are these items similar?",
            model="llama3",
        )
        if response.ok:
            print(response.text)
        else:
            print(f"Error: {response.error}")
    """

    DEFAULT_BASE_URL = "http://localhost:11434"
    DEFAULT_MODEL = "llama3"
    DEFAULT_TIMEOUT = 120  # seconds — LLM inference can be slow
    GENERATE_ENDPOINT = "/api/generate"

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> None:
        """
        Args:
            base_url: Ollama server URL (e.g. "http://localhost:11434").
                      Falls back to DEFAULT_BASE_URL.
            model:    Model name to use (e.g. "llama3", "llama3:8b").
                      Falls back to DEFAULT_MODEL.
            timeout:  HTTP request timeout in seconds.
                      Falls back to DEFAULT_TIMEOUT.
        """
        self.base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self.model = model or self.DEFAULT_MODEL
        self.timeout = timeout if timeout is not None else self.DEFAULT_TIMEOUT

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> OllamaResponse:
        """
        Send a generation request to Ollama.

        Args:
            prompt:      The user-facing prompt text.
            system:      Optional system prompt (sets LLM persona/rules).
            temperature: Sampling temperature (0.0 = deterministic,
                         1.0 = creative). Default 0.7 balances coherence
                         with variety for fashion advice.
            max_tokens:  Maximum tokens to generate. None = model default.
            stream:      If True, Ollama streams tokens incrementally.
                         Currently we use False (non-streaming) for
                         simplicity; streaming can be added later without
                         changing the public interface.

        Returns:
            An OllamaResponse. On success, response.ok is True and
            response.text contains the generated output. On failure,
            response.ok is False and response.error describes the problem.
        """
        url = f"{self.base_url}{self.GENERATE_ENDPOINT}"
        start = time.perf_counter()

        # --- Build request payload ---
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature,
            },
        }

        if system:
            payload["system"] = system

        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens

        # --- Send request ---
        try:
            http_response = requests.post(
                url,
                json=payload,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"},
            )
        except requests.ConnectionError:
            elapsed = _elapsed_ms(start)
            logger.error(
                "OllamaClient: connection refused at '%s' (%.1f ms). "
                "Is Ollama running?",
                url, elapsed,
            )
            return OllamaResponse(
                ok=False,
                error=(
                    "Cannot connect to Ollama. Make sure Ollama is running "
                    f"at {self.base_url}. Start it with: ollama serve"
                ),
            )
        except requests.Timeout:
            elapsed = _elapsed_ms(start)
            logger.error(
                "OllamaClient: request timed out after %d s (%.1f ms).",
                self.timeout, elapsed,
            )
            return OllamaResponse(
                ok=False,
                error=(
                    f"Ollama request timed out after {self.timeout} seconds. "
                    "The model may be loading or the prompt may be too long."
                ),
            )
        except requests.RequestException as exc:
            elapsed = _elapsed_ms(start)
            logger.exception(
                "OllamaClient: unexpected HTTP error (%.1f ms).", elapsed,
            )
            return OllamaResponse(
                ok=False,
                error=f"Ollama request failed: {exc}",
            )

        # --- Parse response ---
        if http_response.status_code != 200:
            logger.error(
                "OllamaClient: HTTP %d from Ollama. Body: %s",
                http_response.status_code,
                http_response.text[:500],
            )
            return OllamaResponse(
                ok=False,
                error=(
                    f"Ollama returned HTTP {http_response.status_code}. "
                    "Check the model name and Ollama server logs."
                ),
            )

        try:
            data = http_response.json()
        except (json.JSONDecodeError, ValueError):
            logger.error(
                "OllamaClient: non-JSON response from Ollama. Body: %s",
                http_response.text[:500],
            )
            return OllamaResponse(
                ok=False,
                error="Ollama returned an invalid (non-JSON) response.",
            )

        # --- Handle Ollama-level errors ---
        if "error" in data:
            logger.error("OllamaClient: Ollama error: %s", data["error"])
            return OllamaResponse(
                ok=False,
                error=f"Ollama error: {data['error']}",
                raw=data,
            )

        # --- Build success response ---
        elapsed = _elapsed_ms(start)
        response = OllamaResponse(
            text=data.get("response", ""),
            model=data.get("model", self.model),
            total_duration=data.get("total_duration", 0),
            ok=True,
            raw=data,
        )

        logger.info(
            "OllamaClient: generated %d chars in %.1f ms (model=%s).",
            len(response.text),
            elapsed,
            response.model,
        )

        return response

    def is_available(self) -> bool:
        """
        Quick health check — can we reach the Ollama server?

        Sends a lightweight GET to the root endpoint.  Returns True if
        the server responds, False otherwise.  Useful for feature-toggle
        logic: if Ollama is down, skip LLM reasoning gracefully.
        """
        try:
            resp = requests.get(self.base_url, timeout=5)
            return resp.status_code == 200
        except requests.RequestException:
            return False


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _elapsed_ms(start: float) -> float:
    """Convert a perf_counter start time to elapsed milliseconds."""
    return (time.perf_counter() - start) * 1000.0
