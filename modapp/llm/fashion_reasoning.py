"""
Fashion Reasoning Service — orchestrates LLM-powered styling analysis.

This is the single entry point for all Llama 3 fashion reasoning.
It coordinates PromptBuilder (what to say) and OllamaClient (how to
say it) behind a clean, one-method interface.

Architecture parallel:
    SimilaritySearchService  orchestrates  EmbeddingService + FaissManager
    FashionReasoningService  orchestrates  PromptBuilder   + OllamaClient

This module has no Django dependencies.  Consumers include
SimilaritySearchView (Django REST Framework), management commands,
and standalone scripts — the same audience as SimilaritySearchService.

Graceful degradation:
    If Ollama is unavailable, the service returns a ReasoningResponse
    with ok=False.  The caller (view) can still return search results
    without AI commentary.  The user experience degrades gracefully
    from "similar items + AI analysis" to "similar items only."
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .ollama_client import OllamaClient
from .prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response dataclass
# ---------------------------------------------------------------------------

@dataclass
class ReasoningResponse:
    """
    Complete response from a fashion reasoning operation.

    Follows the same ok/error pattern as SearchResponse and OllamaResponse
    for consistent error handling across the codebase.

    Attributes:
        analysis:       The AI-generated fashion analysis text.
        ok:             True if the reasoning completed successfully.
        error:          Error description if reasoning failed.
        model:          Which LLM model produced the analysis.
        duration_ms:    Wall-clock time for the reasoning in milliseconds.
        items_analyzed: Number of search results that were analyzed.
    """

    analysis: str = ""
    ok: bool = True
    error: Optional[str] = None
    model: str = ""
    duration_ms: float = 0.0
    items_analyzed: int = 0

    @property
    def available(self) -> bool:
        """True if reasoning was attempted and succeeded."""
        return self.ok and bool(self.analysis)


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class FashionReasoningService:
    """
    Orchestrates LLM-powered fashion analysis from search results.

    End-to-end pipeline:
        result dicts → SearchResultContext → prompt → Ollama → analysis

    This class is the single entry point for all LLM reasoning
    operations.  It coordinates PromptBuilder and OllamaClient behind
    a clean, one-method interface, so the view layer only needs:

        service = FashionReasoningService()
        reasoning = service.analyze(result_dicts)
        if reasoning.ok:
            # include reasoning.analysis in the API response

    Design decisions:
        - No Django imports.  Can be used in scripts and CLI tools.
        - Errors are captured in ReasoningResponse.error rather than
          raised, matching SimilaritySearchService's pattern.
        - Graceful degradation: if Ollama is down, returns ok=False
          so the caller can still serve search results without AI.
        - Configuration is injected via constructor parameters so
          different consumers can use different settings without
          subclassing.

    Usage:
        # Basic usage (all defaults):
        service = FashionReasoningService()
        reasoning = service.analyze(result_dicts)

        # Custom configuration:
        service = FashionReasoningService(
            ollama_url="http://gpu-server:11434",
            model="llama3:70b",
            timeout=180,
        )
        reasoning = service.analyze(result_dicts)
    """

    def __init__(
        self,
        ollama_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
    ) -> None:
        """
        Args:
            ollama_url:    Ollama server URL.  Defaults to OllamaClient's default.
            model:         LLM model name.  Defaults to OllamaClient's default.
            timeout:       HTTP timeout in seconds.  Defaults to OllamaClient's default.
            temperature:   Sampling temperature for generation.
            max_tokens:    Maximum tokens to generate.  None = model default.
            system_prompt: Override the default fashion stylist prompt.
        """
        self.client = OllamaClient(
            base_url=ollama_url,
            model=model,
            timeout=timeout,
        )
        self.builder = PromptBuilder(system_prompt=system_prompt)
        self.temperature = temperature
        self.max_tokens = max_tokens

    def analyze(
        self,
        result_dicts: List[Dict],
        user_query: Optional[str] = None,
    ) -> ReasoningResponse:
        """
        Run the full fashion reasoning pipeline on search results.

        Steps:
            1. Convert result dicts to SearchResultContext objects.
            2. Build the (system_prompt, user_prompt) pair.
            3. Send to Ollama via OllamaClient.
            4. Wrap the response in a ReasoningResponse.

        Args:
            result_dicts:  List of result dicts from the view layer.
                           Expected keys: name, category, brand, color,
                           similarity_score (matching views.py output).
            user_query:    Optional free-text query from the user for
                           future chat-style interactions.

        Returns:
            A ReasoningResponse.  On success, response.ok is True and
            response.analysis contains the AI styling advice.  On failure,
            response.ok is False and response.error describes the problem.
        """
        start = time.perf_counter()

        # --- Guard: nothing to analyze ---
        if not result_dicts:
            return ReasoningResponse(
                ok=False,
                error="No search results to analyze.",
                duration_ms=_elapsed_ms(start),
            )

        # --- Step 1: Convert dicts to typed context ---
        try:
            contexts = PromptBuilder.from_result_dicts(result_dicts)
        except Exception:
            logger.exception(
                "FashionReasoningService: failed to convert result dicts."
            )
            return ReasoningResponse(
                ok=False,
                error="Could not process search results for AI analysis.",
                duration_ms=_elapsed_ms(start),
            )

        # --- Step 2: Build prompts ---
        system_prompt, user_prompt = self.builder.build(
            contexts, user_query=user_query,
        )

        # --- Step 3: Call Ollama ---
        ollama_response = self.client.generate(
            prompt=user_prompt,
            system=system_prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        elapsed = _elapsed_ms(start)

        # --- Step 4: Build response ---
        if not ollama_response.ok:
            logger.warning(
                "FashionReasoningService: Ollama call failed (%.1f ms): %s",
                elapsed,
                ollama_response.error,
            )
            return ReasoningResponse(
                ok=False,
                error=ollama_response.error,
                model=ollama_response.model,
                duration_ms=round(elapsed, 2),
                items_analyzed=len(result_dicts),
            )

        analysis_text = ollama_response.text.strip()

        if not analysis_text:
            logger.warning(
                "FashionReasoningService: Ollama returned empty response (%.1f ms).",
                elapsed,
            )
            return ReasoningResponse(
                ok=False,
                error="The AI stylist returned an empty response. Please try again.",
                model=ollama_response.model,
                duration_ms=round(elapsed, 2),
                items_analyzed=len(result_dicts),
            )

        logger.info(
            "FashionReasoningService: analysis complete — %d items, "
            "%d chars, %.1f ms (model=%s).",
            len(result_dicts),
            len(analysis_text),
            elapsed,
            ollama_response.model,
        )

        return ReasoningResponse(
            analysis=analysis_text,
            ok=True,
            model=ollama_response.model,
            duration_ms=round(elapsed, 2),
            items_analyzed=len(result_dicts),
        )

    def is_available(self) -> bool:
        """
        Check if the Ollama server is reachable.

        Delegates to OllamaClient.is_available().  Useful for the view
        layer to decide whether to attempt AI reasoning or skip it
        entirely for a faster response.
        """
        return self.client.is_available()


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _elapsed_ms(start: float) -> float:
    """Convert a perf_counter start time to elapsed milliseconds."""
    return (time.perf_counter() - start) * 1000.0
