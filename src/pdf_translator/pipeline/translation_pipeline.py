# SPDX-License-Identifier: Apache-2.0
"""Translation pipeline for PDF documents.

This module provides the main TranslationPipeline class that orchestrates
the entire translation workflow: extraction, layout analysis, translation,
and PDF generation.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

from ..core.font_adjuster import FontSizeAdjuster
from ..core.layout_analyzer import LayoutAnalyzer
from ..core.layout_utils import match_text_with_layout
from ..core.models import PDFDocument, ProjectCategory, TextObject
from ..core.pdf_processor import PDFProcessor
from ..core.text_merger import TextMerger
from ..translators.base import ConfigurationError, TranslationError, TranslatorBackend
from .errors import (
    ExtractionError,
    FontAdjustmentError,
    LayoutAnalysisError,
    MergeError,
    PipelineError,
)
from .progress import ProgressCallback

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Pipeline configuration.

    Consolidates all configuration parameters for the translation pipeline.
    Individual components receive their needed values from here.

    Attributes:
        source_lang: Source language code (e.g., "en", "auto")
        target_lang: Target language code (e.g., "ja")
        use_layout_analysis: Whether to use layout analysis
        layout_containment_threshold: Threshold for layout matching
        line_y_tolerance: Y-coordinate tolerance for same-line detection (pt)
        merge_threshold_x: X gap threshold for same-line merge (pt)
        merge_threshold_y: Y gap threshold for next-line merge (pt)
        x_overlap_ratio: X overlap ratio required for next-line merge
        min_font_size: Minimum font size (pt)
        font_size_decrement: Step size for font reduction (pt)
        max_retries: Maximum retry attempts for translation
        retry_delay: Base delay between retries (seconds)
    """

    # Translation settings
    source_lang: str = "en"
    target_lang: str = "ja"

    # Layout analysis
    use_layout_analysis: bool = True
    layout_containment_threshold: float = 0.5

    # Text merger settings
    line_y_tolerance: float = 3.0
    merge_threshold_x: float = 20.0
    merge_threshold_y: float = 5.0
    x_overlap_ratio: float = 0.5

    # Font adjustment settings
    min_font_size: float = 6.0
    font_size_decrement: float = 0.1

    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0


@dataclass
class TranslationResult:
    """Result of translation pipeline.

    Attributes:
        pdf_bytes: Translated PDF as bytes
        stats: Translation statistics (optional)
    """

    pdf_bytes: bytes
    stats: Optional[dict[str, Any]] = field(default=None)


class TranslationPipeline:
    """PDF translation pipeline.

    Orchestrates the complete translation workflow:
    1. Extract text from PDF
    2. Analyze layout (via asyncio.to_thread)
    3. Match text with layout blocks
    4. Filter and sort translatable text objects
    5. Batch translate
    6. Adjust font sizes
    7. Apply to PDF

    Example:
        >>> from pdf_translator.translators import GoogleTranslator
        >>> translator = GoogleTranslator()
        >>> pipeline = TranslationPipeline(translator)
        >>> result = await pipeline.translate("input.pdf")
        >>> Path("output.pdf").write_bytes(result.pdf_bytes)
    """

    def __init__(
        self,
        translator: TranslatorBackend,
        config: Optional[PipelineConfig] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> None:
        """Initialize TranslationPipeline.

        Args:
            translator: Translation backend
            config: Pipeline configuration (defaults if None)
            progress_callback: Progress reporting callback
        """
        self._translator = translator
        self._config = config or PipelineConfig()
        self._progress_callback = progress_callback
        self._analyzer: Optional[LayoutAnalyzer] = None

    def _get_analyzer(self) -> LayoutAnalyzer:
        """Get or create LayoutAnalyzer (lazy initialization)."""
        if self._analyzer is None:
            self._analyzer = LayoutAnalyzer()
        return self._analyzer

    def _report_progress(
        self,
        stage: str,
        current: int,
        total: int,
        message: str = "",
    ) -> None:
        """Report progress if callback is set."""
        if self._progress_callback is not None:
            self._progress_callback(stage, current, total, message)

    async def translate(
        self,
        pdf_source: Union[Path, str, bytes],
        output_path: Optional[Path] = None,
    ) -> TranslationResult:
        """Translate a PDF document.

        Args:
            pdf_source: Input PDF (path or bytes)
            output_path: Output path (if specified, also saves to file)

        Returns:
            TranslationResult with translated PDF bytes

        Raises:
            PipelineError: On pipeline failure
            ExtractionError: On text extraction failure
            LayoutAnalysisError: On layout analysis failure
        """
        # Handle bytes input by writing to temp file (LayoutAnalyzer needs path)
        if isinstance(pdf_source, bytes):
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                f.write(pdf_source)
                temp_path = Path(f.name)
            try:
                return await self._translate_impl(temp_path, output_path)
            finally:
                temp_path.unlink(missing_ok=True)
        else:
            path = Path(pdf_source)
            return await self._translate_impl(path, output_path)

    async def _translate_impl(
        self,
        pdf_path: Path,
        output_path: Optional[Path],
    ) -> TranslationResult:
        """Internal implementation of translate.

        Args:
            pdf_path: Path to PDF file
            output_path: Optional output path

        Returns:
            TranslationResult
        """
        stats: dict[str, Any] = {
            "pages": 0,
            "text_objects": 0,
            "translated_objects": 0,
        }

        # Stage 1: Extract text objects
        pdf_doc, processor = await self._stage_extract(pdf_path)
        stats["pages"] = len(pdf_doc.pages)
        stats["text_objects"] = sum(len(p.text_objects) for p in pdf_doc.pages)

        try:
            # Stage 2: Layout analysis
            categories = await self._stage_analyze(pdf_path, pdf_doc)

            # Stage 3: Merge (filter and sort)
            sorted_objects_by_page = self._stage_merge(pdf_doc, categories)

            # Count translatable objects
            total_translatable = sum(len(objs) for objs in sorted_objects_by_page.values())
            stats["translated_objects"] = total_translatable

            # Stage 4: Translate
            await self._stage_translate(sorted_objects_by_page)

            # Stage 5: Font adjustment
            self._stage_font_adjust(sorted_objects_by_page)

            # Stage 6: Apply to PDF
            pdf_bytes = self._stage_apply(processor, pdf_doc)

            # Save to file if output_path specified
            if output_path:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(pdf_bytes)

            return TranslationResult(pdf_bytes=pdf_bytes, stats=stats)

        finally:
            processor.close()

    async def _stage_extract(
        self,
        pdf_path: Path,
    ) -> tuple[PDFDocument, PDFProcessor]:
        """Stage 1: Extract text objects from PDF.

        Args:
            pdf_path: Path to PDF

        Returns:
            Tuple of (PDFDocument, PDFProcessor)

        Raises:
            ExtractionError: On extraction failure
        """
        self._report_progress("extract", 0, 1, "Starting extraction...")

        processor: PDFProcessor | None = None
        try:
            processor = PDFProcessor(pdf_path)
            pdf_doc = processor.extract_text_objects()
            self._report_progress("extract", 1, 1, "Extraction complete")
            return pdf_doc, processor
        except Exception as e:
            # Clean up processor if extraction fails after opening
            if processor is not None:
                processor.close()
            raise ExtractionError(f"Failed to extract text from PDF: {e}", cause=e)

    async def _stage_analyze(
        self,
        pdf_path: Path,
        pdf_doc: PDFDocument,
    ) -> dict[str, ProjectCategory]:
        """Stage 2: Layout analysis and category matching.

        Args:
            pdf_path: Path to PDF
            pdf_doc: Extracted PDF document

        Returns:
            Mapping of TextObject.id -> ProjectCategory

        Raises:
            LayoutAnalysisError: On analysis failure
        """
        page_count = len(pdf_doc.pages)

        categories: dict[str, ProjectCategory] = {}

        if not self._config.use_layout_analysis:
            logger.warning(
                "Layout analysis disabled. All text objects will be translated. "
                "Formulas, tables, and other non-text elements may be incorrectly translated."
            )
            # All TextObjects treated as TEXT category
            for page in pdf_doc.pages:
                for obj in page.text_objects:
                    categories[obj.id] = ProjectCategory.TEXT
            return categories

        self._report_progress("analyze", 0, page_count, "Starting layout analysis...")

        try:
            # Run layout analysis in thread pool (CPU-bound)
            analyzer = self._get_analyzer()
            layout_by_page = await asyncio.to_thread(analyzer.analyze_all, pdf_path)

            # Match text with layout for each page
            for i, page in enumerate(pdf_doc.pages):
                page_blocks = layout_by_page.get(page.page_number, [])
                page_categories = match_text_with_layout(
                    page.text_objects,
                    page_blocks,
                    self._config.layout_containment_threshold,
                )
                categories.update(page_categories)
                self._report_progress(
                    "analyze", i + 1, page_count, f"Page {page.page_number + 1} analyzed"
                )

            return categories

        except Exception as e:
            raise LayoutAnalysisError(f"Layout analysis failed: {e}", cause=e)

    def _stage_merge(
        self,
        pdf_doc: PDFDocument,
        categories: dict[str, ProjectCategory],
    ) -> dict[int, list[TextObject]]:
        """Stage 3: Filter and sort text objects in reading order.

        Args:
            pdf_doc: PDF document
            categories: Category mapping

        Returns:
            Dict mapping page_number -> sorted translatable TextObjects

        Raises:
            MergeError: On merge/sort failure
        """
        page_count = len(pdf_doc.pages)
        self._report_progress("merge", 0, page_count, "Sorting text objects...")

        try:
            merger = TextMerger(
                line_y_tolerance=self._config.line_y_tolerance,
                merge_threshold_x=self._config.merge_threshold_x,
                merge_threshold_y=self._config.merge_threshold_y,
                x_overlap_ratio=self._config.x_overlap_ratio,
            )

            result: dict[int, list[TextObject]] = {}
            for i, page in enumerate(pdf_doc.pages):
                sorted_objects = merger.merge(page.text_objects, categories)
                result[page.page_number] = sorted_objects
                self._report_progress(
                    "merge", i + 1, page_count, f"Page {page.page_number + 1} sorted"
                )

            return result
        except Exception as e:
            raise MergeError(f"Failed to merge/sort text objects: {e}", cause=e)

    async def _stage_translate(
        self,
        sorted_objects_by_page: dict[int, list[TextObject]],
    ) -> None:
        """Stage 4: Translate text objects (in-place modification).

        Args:
            sorted_objects_by_page: Sorted TextObjects by page

        Note:
            This modifies TextObject.text in-place with translated text.
        """
        # Flatten all objects for batch translation
        all_objects: list[TextObject] = []
        for objects in sorted_objects_by_page.values():
            all_objects.extend(objects)

        total = len(all_objects)
        if total == 0:
            self._report_progress("translate", 0, 0, "No text to translate")
            return

        self._report_progress("translate", 0, total, "Starting translation...")

        # Extract texts
        texts = [obj.text for obj in all_objects]

        # Translate with retry
        translated = await self._translate_with_retry(texts)

        # Update text objects in-place
        for obj, translated_text in zip(all_objects, translated):
            obj.text = translated_text

        self._report_progress("translate", total, total, "Translation complete")

    async def _translate_with_retry(self, texts: list[str]) -> list[str]:
        """Translate texts with retry logic.

        Args:
            texts: Texts to translate

        Returns:
            Translated texts

        Raises:
            PipelineError: After max retries exceeded
            ConfigurationError: On configuration errors (not retried)
        """
        for attempt in range(self._config.max_retries + 1):
            try:
                return await self._translator.translate_batch(
                    texts,
                    self._config.source_lang,
                    self._config.target_lang,
                )
            except ConfigurationError:
                # Configuration errors are not retryable
                raise
            except TranslationError as e:
                if attempt < self._config.max_retries:
                    delay = self._config.retry_delay * (2**attempt)  # Exponential backoff
                    logger.warning(
                        "Translation attempt %d failed, retrying in %.1fs: %s",
                        attempt + 1,
                        delay,
                        e,
                    )
                    await asyncio.sleep(delay)
                else:
                    raise PipelineError(
                        f"Translation failed after {self._config.max_retries} retries: {e}",
                        stage="translate",
                        cause=e,
                    )

        # Should not reach here, but satisfy type checker
        raise PipelineError(
            "Translation failed unexpectedly",
            stage="translate",
        )

    def _stage_font_adjust(
        self,
        sorted_objects_by_page: dict[int, list[TextObject]],
    ) -> None:
        """Stage 5: Adjust font sizes for translated text.

        Args:
            sorted_objects_by_page: TextObjects with translated text

        Raises:
            FontAdjustmentError: On font adjustment failure
        """
        all_objects: list[TextObject] = []
        for objects in sorted_objects_by_page.values():
            all_objects.extend(objects)

        total = len(all_objects)
        if total == 0:
            self._report_progress("font_adjust", 0, 0, "No text to adjust")
            return

        self._report_progress("font_adjust", 0, total, "Adjusting font sizes...")

        try:
            adjuster = FontSizeAdjuster(
                min_font_size=self._config.min_font_size,
                font_size_decrement=self._config.font_size_decrement,
            )

            for i, obj in enumerate(all_objects):
                if obj.font is not None:
                    new_size = adjuster.calculate_font_size(
                        obj.text,
                        obj.bbox,
                        obj.font.size,
                        self._config.target_lang,
                    )
                    obj.font.size = new_size

                if (i + 1) % 100 == 0 or i == total - 1:
                    self._report_progress("font_adjust", i + 1, total, f"{i + 1}/{total} adjusted")

            self._report_progress("font_adjust", total, total, "Font adjustment complete")
        except Exception as e:
            raise FontAdjustmentError(f"Failed to adjust font sizes: {e}", cause=e)

    def _stage_apply(
        self,
        processor: PDFProcessor,
        pdf_doc: PDFDocument,
    ) -> bytes:
        """Stage 6: Apply changes to PDF.

        Args:
            processor: PDF processor
            pdf_doc: Modified PDF document

        Returns:
            PDF bytes
        """
        self._report_progress("apply", 0, 1, "Applying changes...")

        processor.apply(pdf_doc)
        pdf_bytes = processor.to_bytes()

        self._report_progress("apply", 1, 1, "PDF generation complete")

        return pdf_bytes
