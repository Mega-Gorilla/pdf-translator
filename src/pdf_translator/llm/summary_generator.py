# SPDX-License-Identifier: Apache-2.0
"""LLM-based summary generation and metadata extraction."""

from __future__ import annotations

import json
import logging
import re

from pdf_translator.llm.client import LLMClient, LLMConfig

logger = logging.getLogger(__name__)


class LLMSummaryGenerator:
    """Generate document summaries and extract metadata using LLM.

    Uses LiteLLM for unified access to multiple LLM providers:
    - Gemini (default): gemini/gemini-3.0-flash
    - OpenAI: openai/gpt-5-mini
    - Anthropic: anthropic/claude-sonnet-4-5
    """

    SUMMARY_PROMPT = """Summarize this academic paper in 3-5 sentences, covering:
- Main research objective
- Key methodology
- Important findings/conclusions

Paper content:
{content}"""

    METADATA_PROMPT = """Extract the following information from this academic paper's first page.
Return JSON format with null for missing fields.

Required fields:
- title: The main title of the paper
- abstract: The abstract/summary section (if present on first page)
- organization: The institution/company names (e.g., "Meta AI", "Google Research")

First page content:
{content}"""

    def __init__(self, config: LLMConfig) -> None:
        """Initialize LLMSummaryGenerator.

        Args:
            config: LLM configuration.
        """
        self._config = config
        self._client = (
            LLMClient(config) if (config.use_summary or config.use_fallback) else None
        )

    async def generate_summary(
        self,
        markdown_content: str,
    ) -> str | None:
        """Generate summary from original Markdown content.

        Args:
            markdown_content: Full original Markdown (images excluded).

        Returns:
            Summary text in original language, or None if disabled/failed.
        """
        if not self._config.use_summary or not self._client:
            return None

        # Remove image references from Markdown
        content = re.sub(r"!\[.*?\]\(.*?\)", "", markdown_content)

        # Truncate if too long (500K chars safety limit)
        if len(content) > 500_000:
            logger.warning(
                "Markdown content truncated from %d to 500K chars", len(content)
            )
            content = content[:500_000]

        prompt = self.SUMMARY_PROMPT.format(content=content)

        try:
            return await self._client.generate(prompt)
        except Exception as e:
            logger.warning("Failed to generate LLM summary: %s", e)
            return None

    async def extract_metadata_fallback(
        self,
        first_page_text: str,
    ) -> dict[str, str | None]:
        """Extract metadata from first page when layout analysis fails.

        Args:
            first_page_text: Text content of first page only.

        Returns:
            Dict with title, abstract, organization (any may be None).
        """
        if not self._config.use_fallback or not self._client:
            return {"title": None, "abstract": None, "organization": None}

        prompt = self.METADATA_PROMPT.format(content=first_page_text)

        try:
            text = await self._client.generate(prompt)

            # Parse JSON from response
            # Handle markdown code blocks if present
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]

            result: dict[str, str | None] = json.loads(text)
            return result
        except Exception as e:
            logger.warning("Failed to extract metadata via LLM: %s", e)
            return {"title": None, "abstract": None, "organization": None}
