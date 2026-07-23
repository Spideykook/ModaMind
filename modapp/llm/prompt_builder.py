"""
Fashion-focused prompt construction for LLM reasoning.

Converts similarity search results into structured prompts that guide
Llama 3 to act as a fashion stylist.  This module has no Django
dependencies and no knowledge of Ollama or HTTP — it is pure text
construction, analogous to how transforms.py is pure image preprocessing
with no knowledge of the model that consumes the tensors.

Separation rationale:
  - OllamaClient owns HOW to talk to the LLM (HTTP transport).
  - PromptBuilder owns WHAT to say to the LLM (prompt engineering).
  - FashionReasoningService combines both (orchestration).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data contract — what the builder expects from the caller
# ---------------------------------------------------------------------------

@dataclass
class SearchResultContext:
    """
    A single search result prepared for prompt inclusion.

    This is a lightweight transfer object that decouples the prompt
    builder from the view layer's result format.  The view (or any
    caller) maps its own result dicts into these before calling
    PromptBuilder.

    Attributes:
        rank:             1-based position in the result list.
        name:             Display name of the clothing item.
        category:         Garment category (e.g. "Tops", "Shoes").
        brand:            Brand name (may be empty).
        color:            Color descriptor (may be empty).
        similarity_score: Cosine similarity as a float in [0, 1].
    """

    rank: int
    name: str
    category: str = ""
    brand: str = ""
    color: str = ""
    similarity_score: float = 0.0


# ---------------------------------------------------------------------------
# System prompt — the fashion stylist persona
# ---------------------------------------------------------------------------

FASHION_SYSTEM_PROMPT = """You are ModaMind's AI Fashion Stylist — an expert fashion consultant with deep knowledge of clothing styles, trends, color theory, and outfit coordination.

Your role:
- Analyze the fashion items provided and explain their visual and stylistic similarities.
- Provide practical styling recommendations.
- Suggest how to incorporate these pieces into complete outfits.
- Identify relevant fashion trends.

Rules:
- ONLY reference the items provided below. Do NOT invent or hallucinate products that are not listed.
- Keep your analysis grounded in the actual item attributes (category, color, brand).
- Be concise, enthusiastic, and actionable.
- Write for a fashion-conscious audience, not technical experts.
- Use a warm, knowledgeable tone — like a personal stylist in a boutique.

Response format:
- Start with a brief style summary (2-3 sentences) of the overall aesthetic theme.
- Then provide a "Why These Match" section explaining the visual similarities.
- Then provide a "Styling Tips" section with 3-4 practical outfit suggestions.
- Keep the total response under 300 words."""


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

class PromptBuilder:
    """
    Constructs structured prompts for fashion reasoning from search results.

    Responsibilities:
      1. Convert a list of SearchResultContext objects into a readable
         text block that the LLM can reason about.
      2. Combine the context with the fashion system prompt.
      3. Return a (system_prompt, user_prompt) tuple ready for
         OllamaClient.generate().

    This class is stateless and has no side effects.  Every method is
    a pure function of its inputs, making it trivially testable.

    Usage:
        builder = PromptBuilder()
        items = [SearchResultContext(rank=1, name="Red Hoodie", ...)]
        system, user = builder.build(items)
        # Pass system and user to OllamaClient.generate()
    """

    def __init__(self, system_prompt: Optional[str] = None) -> None:
        """
        Args:
            system_prompt: Override the default fashion stylist prompt.
                           Useful for testing or for domain variants
                           (e.g. a streetwear-focused prompt).
        """
        self.system_prompt = system_prompt or FASHION_SYSTEM_PROMPT

    def build_context(self, results: List[SearchResultContext]) -> str:
        """
        Convert search results into a structured text block.

        Each item becomes a numbered entry with its metadata and
        similarity score.  The format is designed to be:
          - Easy for the LLM to parse (numbered, labeled fields)
          - Easy for humans to read (for debugging prompts)
          - Concise (no wasted tokens)

        Args:
            results: Ordered list of SearchResultContext objects
                     (most similar first).

        Returns:
            A multi-line string describing all items.
        """
        if not results:
            return "No similar items were found."

        lines: list[str] = []
        for item in results:
            parts = [f"Item {item.rank}: {item.name or 'Unnamed Item'}"]

            details: list[str] = []
            if item.category:
                details.append(f"Category: {item.category}")
            if item.brand:
                details.append(f"Brand: {item.brand}")
            if item.color:
                details.append(f"Color: {item.color}")

            score_pct = round(item.similarity_score * 100)
            details.append(f"Similarity: {score_pct}%")

            parts.append("  " + " | ".join(details))
            lines.append("\n".join(parts))

        return "\n\n".join(lines)

    def build_user_prompt(
        self,
        results: List[SearchResultContext],
        user_query: Optional[str] = None,
    ) -> str:
        """
        Build the complete user-facing prompt that combines context
        and task instruction.

        Args:
            results:    Search results to include as context.
            user_query: Optional free-text query from the user
                        (e.g. "How can I style this for a date night?").
                        If not provided, a default analysis request is used.

        Returns:
            A complete user prompt string ready for the LLM.
        """
        context = self.build_context(results)

        task = user_query or (
            "Analyze these visually similar fashion items. "
            "Explain why they match, identify the common style theme, "
            "and suggest how to style them into complete outfits."
        )

        prompt = (
            f"A user uploaded a fashion image and our AI found these "
            f"visually similar items from the catalog:\n\n"
            f"{context}\n\n"
            f"Task: {task}"
        )

        return prompt

    def build(
        self,
        results: List[SearchResultContext],
        user_query: Optional[str] = None,
    ) -> tuple[str, str]:
        """
        Build the complete (system_prompt, user_prompt) pair.

        This is the primary entry point.  Returns a tuple ready to be
        passed directly to OllamaClient.generate(prompt=user, system=system).

        Args:
            results:    Search results to include as context.
            user_query: Optional free-text query from the user.

        Returns:
            A (system_prompt, user_prompt) tuple.
        """
        system = self.system_prompt
        user = self.build_user_prompt(results, user_query)

        logger.info(
            "PromptBuilder: built prompt with %d items (%d chars).",
            len(results),
            len(user),
        )

        return system, user

    @staticmethod
    def from_result_dicts(
        result_dicts: List[Dict],
    ) -> List[SearchResultContext]:
        """
        Convert the result dicts produced by SimilaritySearchView into
        SearchResultContext objects.

        This is a convenience adapter so the view doesn't need to know
        about SearchResultContext internals.  It maps the JSON-serializable
        dict format used by the API response into the dataclass format
        used by the prompt builder.

        Expected dict keys (matching views.py output):
            id, name, category, brand, color, similarity_score

        Args:
            result_dicts: List of result dicts from the view layer.

        Returns:
            List of SearchResultContext objects, ranked by position.
        """
        contexts: list[SearchResultContext] = []
        for rank, item in enumerate(result_dicts, start=1):
            contexts.append(
                SearchResultContext(
                    rank=rank,
                    name=item.get("name", ""),
                    category=item.get("category", ""),
                    brand=item.get("brand", ""),
                    color=item.get("color", ""),
                    similarity_score=item.get("similarity_score", 0.0),
                )
            )
        return contexts
