# SPDX-License-Identifier: Apache-2.0
"""Summary extractor for document metadata and LLM-generated summaries."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from pdf_translator.core.models import Paragraph
from pdf_translator.llm.client import LLMConfig
from pdf_translator.llm.summary_generator import LLMSummaryGenerator
from pdf_translator.output.base_summary import BaseSummary
from pdf_translator.output.thumbnail_generator import ThumbnailConfig, ThumbnailGenerator

logger = logging.getLogger(__name__)


class SummaryExtractor:
    """Extract base document summary from paragraphs.

    Handles:
    - Title extraction from doc_title category (with LLM fallback)
    - Abstract extraction from abstract category (with LLM fallback)
    - Organization extraction (LLM only)
    - LLM summary generation from original Markdown (in source language)
    - Thumbnail generation from first page

    Note: Translation is handled separately in the pipeline.
    """

    TITLE_CATEGORY = "doc_title"
    ABSTRACT_CATEGORY = "abstract"

    def __init__(
        self,
        thumbnail_config: ThumbnailConfig | None = None,
        llm_config: LLMConfig | None = None,
    ) -> None:
        """Initialize SummaryExtractor.

        Args:
            thumbnail_config: Configuration for thumbnail generation.
            llm_config: Configuration for LLM integration.
        """
        self._thumbnail_config = thumbnail_config or ThumbnailConfig()
        self._llm_config = llm_config
        self._llm_generator = (
            LLMSummaryGenerator(llm_config) if llm_config else None
        )

    @property
    def llm_generator(self) -> LLMSummaryGenerator | None:
        """Get the LLM generator instance for use in pipeline."""
        return self._llm_generator

    async def extract(
        self,
        paragraphs: list[Paragraph],
        pdf_path: Path,
        output_dir: Path,
        output_stem: str,
        source_lang: str = "",
        page_count: int = 0,
        generate_thumbnail: bool = True,
        original_markdown: str | None = None,
    ) -> BaseSummary:
        """Extract base document summary from paragraphs.

        This method extracts original language content only.
        Translation is handled separately in the pipeline.

        Args:
            paragraphs: List of paragraphs (with original text).
            pdf_path: Path to original PDF (for thumbnail).
            output_dir: Directory to save thumbnail.
            output_stem: Base filename for outputs.
            source_lang: Source language code.
            page_count: Total page count.
            generate_thumbnail: Whether to generate thumbnail.
            original_markdown: Original Markdown content for LLM summary.

        Returns:
            BaseSummary with extracted information.
        """
        # Step 1: Extract from layout analysis
        title = self._find_and_merge_by_category(
            paragraphs, self.TITLE_CATEGORY
        )
        abstract = self._find_and_merge_by_category(
            paragraphs, self.ABSTRACT_CATEGORY
        )

        title_source: Literal["layout", "llm"] = "layout"
        abstract_source: Literal["layout", "llm"] = "layout"
        organization = None

        # Step 2: LLM metadata extraction
        # - Organization: Always extracted via LLM (not available in layout)
        # - Title/Abstract: Fallback when layout analysis fails
        if self._llm_generator:
            first_page_text = self._get_first_page_text(paragraphs)
            if first_page_text:
                llm_metadata = await self._llm_generator.extract_metadata_fallback(
                    first_page_text
                )

                # Organization is always from LLM (no layout category exists)
                organization = llm_metadata.get("organization")

                # Use LLM result only if layout failed
                if not title and llm_metadata.get("title"):
                    title = llm_metadata["title"]
                    title_source = "llm"

                if not abstract and llm_metadata.get("abstract"):
                    abstract = llm_metadata["abstract"]
                    abstract_source = "llm"

        # Step 3: Generate LLM summary from original Markdown (in source language)
        summary = None
        if self._llm_generator and original_markdown:
            summary = await self._llm_generator.generate_summary(
                original_markdown, target_lang=source_lang
            )

        # Step 4: Generate thumbnail
        thumbnail_path, thumbnail_bytes, thumb_width, thumb_height = (
            await self._generate_thumbnail(
                pdf_path, output_dir, output_stem, generate_thumbnail
            )
        )

        return BaseSummary(
            title=title,
            abstract=abstract,
            organization=organization,
            summary=summary,
            thumbnail_path=thumbnail_path,
            thumbnail_width=thumb_width,
            thumbnail_height=thumb_height,
            page_count=page_count,
            title_source=title_source,
            abstract_source=abstract_source,
            _thumbnail_bytes=thumbnail_bytes,
        )

    @staticmethod
    def _find_and_merge_by_category(
        paragraphs: list[Paragraph],
        category: str,
    ) -> str | None:
        """Find all paragraphs with category and merge them.

        Args:
            paragraphs: List of paragraphs to search.
            category: Category to find.

        Returns:
            Merged original text or None if not found.
        """
        matched = [p for p in paragraphs if p.category == category]

        if not matched:
            return None

        # Sort by page_number, then by y-coordinate (descending, PDF coordinates)
        matched.sort(key=lambda p: (p.page_number, -p.block_bbox.y1))

        original_parts = [p.text for p in matched if p.text]

        return "\n\n".join(original_parts) if original_parts else None

    @staticmethod
    def _get_first_page_text(paragraphs: list[Paragraph]) -> str:
        """Get concatenated text from first page for LLM fallback.

        Args:
            paragraphs: List of paragraphs.

        Returns:
            Concatenated text from first page.
        """
        first_page = [p for p in paragraphs if p.page_number == 0]
        first_page.sort(key=lambda p: -p.block_bbox.y1)
        return "\n\n".join(p.text for p in first_page if p.text)

    async def _generate_thumbnail(
        self,
        pdf_path: Path,
        output_dir: Path,
        output_stem: str,
        generate_thumbnail: bool,
    ) -> tuple[str | None, bytes | None, int, int]:
        """Generate thumbnail from PDF first page.

        Args:
            pdf_path: Path to PDF file.
            output_dir: Directory to save thumbnail.
            output_stem: Base filename for thumbnail.
            generate_thumbnail: Whether to generate thumbnail.

        Returns:
            Tuple of (thumbnail_filename, thumbnail_bytes, width, height).
        """
        if not generate_thumbnail or not pdf_path.exists():
            return None, None, 0, 0

        try:
            generator = ThumbnailGenerator(self._thumbnail_config)
            thumbnail_bytes, width, height = generator.generate(pdf_path)

            thumbnail_filename = f"{output_stem}_thumbnail.png"
            thumbnail_file = output_dir / thumbnail_filename
            # Ensure output directory exists before writing
            output_dir.mkdir(parents=True, exist_ok=True)
            thumbnail_file.write_bytes(thumbnail_bytes)

            logger.debug("Generated thumbnail: %s", thumbnail_file)
            return thumbnail_filename, thumbnail_bytes, width, height
        except Exception as e:
            logger.warning("Failed to generate thumbnail: %s", e)
            return None, None, 0, 0
