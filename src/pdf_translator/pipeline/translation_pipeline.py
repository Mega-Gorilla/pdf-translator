# SPDX-License-Identifier: Apache-2.0
"""Translation pipeline implementation."""

from __future__ import annotations

import asyncio
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pdf_translator.core.font_subsetter import (
    CFF_OPTIMIZATION_WARNING,
    FontSubsetter,
    SubsetConfig,
    has_truetype_outlines,
)
from pdf_translator.core.layout_analyzer import LayoutAnalyzer
from pdf_translator.core.layout_utils import assign_categories
from pdf_translator.core.models import LayoutBlock, Paragraph
from pdf_translator.core.paragraph_extractor import ParagraphExtractor
from pdf_translator.core.paragraph_merger import MergeConfig, merge_adjacent_paragraphs
from pdf_translator.core.pdf_processor import PDFProcessor
from pdf_translator.core.side_by_side import (
    SideBySideConfig,
    SideBySideGenerator,
    SideBySideOrder,
)
from pdf_translator.pipeline.errors import (
    ExtractionError,
    LayoutAnalysisError,
    PipelineError,
)
from pdf_translator.pipeline.progress import ProgressCallback
from pdf_translator.translators.base import (
    ArrayLengthMismatchError,
    ConfigurationError,
    TranslationError,
    TranslatorBackend,
)

logger = logging.getLogger(__name__)

# Bundled Koruri font (Apache 2.0 licensed TrueType font with good PDF compatibility)
_BUNDLED_FONT_DIR = Path(__file__).parent.parent / "resources" / "fonts"
BUNDLED_KORURI_REGULAR = _BUNDLED_FONT_DIR / "Koruri-Regular.ttf"
BUNDLED_KORURI_BOLD = _BUNDLED_FONT_DIR / "Koruri-Bold.ttf"


@dataclass
class PipelineConfig:
    """Translation pipeline configuration."""

    source_lang: str = "en"
    target_lang: str = "ja"

    use_layout_analysis: bool = True
    layout_containment_threshold: float = 0.5

    # Categories to translate (raw_category strings)
    # If None, uses DEFAULT_TRANSLATABLE_RAW_CATEGORIES
    translatable_categories: frozenset[str] | None = None

    # Minimum font size for translated text (used by TextLayoutEngine)
    min_font_size: float = 6.0

    max_retries: int = 3
    retry_delay: float = 1.0
    translation_batch_size: int = 20
    translation_max_concurrent: int = 10  # Maximum concurrent API requests

    cjk_font_path: Path | None = None

    # Font optimization settings
    optimize_fonts: bool = True  # Enable font subsetting (reduces PDF size)
    font_subset_cache_dir: Path | None = None  # Cache directory for subset fonts
    cjk_font_number: int = 0  # Font index for TTC files (default: 0 = Japanese)

    # Paragraph merge settings
    merge_adjacent_paragraphs: bool = True  # Enabled by default
    merge_gap_tolerance: float = 0.5  # Gap <= font_size * 0.5
    merge_x_overlap_threshold: float = 0.7  # X overlap >= 70%
    merge_font_size_tolerance: float = 1.0  # Font size difference <= 1pt
    merge_width_tolerance: float = 0.95  # Width ratio >= 95%
    merge_max_length: int = 4000  # Max merged text length (Google limit: 5000)

    # Debug options
    debug_draw_bbox: bool = False

    # Side-by-side output options
    side_by_side: bool = False  # Generate side-by-side comparison PDF
    side_by_side_order: SideBySideOrder = SideBySideOrder.TRANSLATED_ORIGINAL
    side_by_side_gap: float = 0.0  # Gap between pages in points

    # Translation failure handling
    strict_mode: bool = False  # If True, raise error on single text failure


@dataclass
class TranslationResult:
    """Translation pipeline result."""

    pdf_bytes: bytes
    stats: dict[str, Any] | None = None
    side_by_side_pdf_bytes: bytes | None = None  # Only set if side_by_side is enabled


class TranslationPipeline:
    """PDF translation pipeline."""

    def __init__(
        self,
        translator: TranslatorBackend,
        config: PipelineConfig | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        """Initialize TranslationPipeline."""
        self._translator = translator
        self._config = config or PipelineConfig()
        self._progress_callback = progress_callback
        self._analyzer = LayoutAnalyzer()

    async def translate(
        self,
        pdf_source: Path | str | bytes,
        output_path: Path | None = None,
    ) -> TranslationResult:
        """Translate a PDF from path or bytes."""
        if isinstance(pdf_source, bytes):
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                f.write(pdf_source)
                temp_path = Path(f.name)
            try:
                return await self._translate_impl(temp_path, output_path)
            finally:
                temp_path.unlink(missing_ok=True)

        path = Path(pdf_source)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")
        return await self._translate_impl(path, output_path)

    async def _translate_impl(
        self,
        pdf_path: Path,
        output_path: Path | None,
    ) -> TranslationResult:
        paragraphs = await self._stage_extract(pdf_path)
        original_count = len(paragraphs)
        layout_blocks = await self._stage_analyze(pdf_path)
        self._stage_categorize(paragraphs, layout_blocks)

        # Save pre-merge paragraphs for debug overlay
        pre_merge_paragraphs = list(paragraphs) if self._config.debug_draw_bbox else None

        paragraphs = self._stage_merge(paragraphs)
        translatable = await self._stage_translate(paragraphs)
        pdf_bytes = self._stage_apply(pdf_path, paragraphs, pre_merge_paragraphs)

        # Generate side-by-side PDF if enabled
        side_by_side_bytes: bytes | None = None
        if self._config.side_by_side:
            side_by_side_bytes = self._stage_side_by_side(
                pdf_path, pdf_bytes, paragraphs, pre_merge_paragraphs
            )

        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(pdf_bytes)

            # Save side-by-side PDF with "_side_by_side" suffix
            if side_by_side_bytes is not None:
                sbs_output_path = output_path.with_stem(output_path.stem + "_side_by_side")
                sbs_output_path.write_bytes(side_by_side_bytes)

        stats = {
            "original_paragraphs": original_count,
            "paragraphs": len(paragraphs),
            "merged_paragraphs": original_count - len(paragraphs),
            "translated_paragraphs": len(translatable),
            "skipped_paragraphs": len(paragraphs) - len(translatable),
        }
        return TranslationResult(
            pdf_bytes=pdf_bytes,
            stats=stats,
            side_by_side_pdf_bytes=side_by_side_bytes,
        )

    async def _stage_extract(self, pdf_path: Path) -> list[Paragraph]:
        try:
            paragraphs = ParagraphExtractor.extract_from_pdf(pdf_path)
        except Exception as exc:
            raise ExtractionError(
                "Paragraph extraction failed", stage="extract", cause=exc
            ) from exc

        with PDFProcessor(pdf_path) as processor:
            page_count = processor.page_count
        self._notify("extract", page_count, page_count)
        return paragraphs

    async def _stage_analyze(self, pdf_path: Path) -> dict[int, list[LayoutBlock]]:
        if not self._config.use_layout_analysis:
            logger.warning(
                "Layout analysis disabled. All paragraphs will be translated. "
                "Formulas, tables, and figures may be incorrectly translated."
            )
            return {}

        try:
            result = await asyncio.to_thread(self._analyzer.analyze_all, pdf_path)
        except Exception as exc:
            raise LayoutAnalysisError("Layout analysis failed", stage="analyze", cause=exc) from exc

        self._notify("analyze", len(result), len(result))
        return result

    def _stage_categorize(
        self,
        paragraphs: list[Paragraph],
        layout_blocks: dict[int, list[LayoutBlock]],
    ) -> None:
        if not layout_blocks:
            return

        assign_categories(
            paragraphs,
            layout_blocks,
            threshold=self._config.layout_containment_threshold,
        )
        self._notify("categorize", len(paragraphs), len(paragraphs))

    def _stage_merge(self, paragraphs: list[Paragraph]) -> list[Paragraph]:
        """Merge adjacent same-category paragraphs.

        Only runs if merge_adjacent_paragraphs is True in config.
        """
        if not self._config.merge_adjacent_paragraphs:
            return paragraphs

        merge_config = MergeConfig(
            gap_tolerance=self._config.merge_gap_tolerance,
            x_overlap_threshold=self._config.merge_x_overlap_threshold,
            font_size_tolerance=self._config.merge_font_size_tolerance,
            translatable_categories=self._config.translatable_categories,
            width_tolerance=self._config.merge_width_tolerance,
            max_merged_length=self._config.merge_max_length,
        )

        original_count = len(paragraphs)
        merged = merge_adjacent_paragraphs(paragraphs, merge_config)
        self._notify("merge", len(merged), original_count)
        return merged

    async def _stage_translate(self, paragraphs: list[Paragraph]) -> list[Paragraph]:
        categories = self._config.translatable_categories
        translatable = [
            para for para in paragraphs
            if para.is_translatable(categories)
        ]
        if not translatable:
            return []

        texts = [para.text for para in translatable]
        chunks = self._chunk_texts(texts)
        total = len(texts)

        # Use semaphore to limit concurrent API requests
        # Ensure at least 1 to avoid Semaphore(0) causing infinite wait
        max_concurrent = max(1, self._config.translation_max_concurrent)
        semaphore = asyncio.Semaphore(max_concurrent)
        completed_count = 0

        async def translate_chunk(chunk: list[str], chunk_idx: int) -> tuple[int, list[str]]:
            """Translate a chunk with semaphore-based concurrency control."""
            nonlocal completed_count
            async with semaphore:
                chunk_translations = await self._translate_with_retry(chunk)
                if len(chunk_translations) != len(chunk):
                    raise PipelineError(
                        "Translator returned unexpected number of results",
                        stage="translate",
                    )
                completed_count += len(chunk)
                self._notify("translate", completed_count, total)
                return chunk_idx, chunk_translations

        # Execute all chunks in parallel with semaphore limiting concurrency
        tasks = [translate_chunk(chunk, idx) for idx, chunk in enumerate(chunks)]
        results = await asyncio.gather(*tasks)

        # Reconstruct translated_texts in original order
        results_sorted = sorted(results, key=lambda x: x[0])
        translated_texts: list[str] = []
        for _, chunk_translations in results_sorted:
            translated_texts.extend(chunk_translations)

        for para, translated in zip(translatable, translated_texts):
            para.translated_text = translated

        return translatable

    def _stage_apply(
        self,
        pdf_path: Path,
        paragraphs: list[Paragraph],
        pre_merge_paragraphs: list[Paragraph] | None = None,
    ) -> bytes:
        target_lang = self._config.target_lang.lower()
        font_path: Path | None = None
        if target_lang in {"ja", "zh", "ko"}:
            if self._config.cjk_font_path is not None:
                font_path = self._config.cjk_font_path
            elif BUNDLED_KORURI_REGULAR.exists():
                # Use bundled Koruri font as default (TrueType, good compatibility)
                font_path = BUNDLED_KORURI_REGULAR
                logger.info("Using bundled Koruri font for CJK text rendering.")
            else:
                logger.warning(
                    "Target language is CJK but no cjk_font_path is set "
                    "and bundled font not found; output may be garbled."
                )

        # Create font subsets if optimization is enabled
        font_subsets: dict[tuple[bool, bool], Path] = {}
        subsetter: FontSubsetter | None = None
        if self._config.optimize_fonts and font_path is not None:
            # Check for CFF font compatibility issues
            if not has_truetype_outlines(font_path, self._config.cjk_font_number):
                logger.warning(CFF_OPTIMIZATION_WARNING, font_path.name)

            # Collect texts per style variant for optimized subsetting
            texts_by_style: dict[tuple[bool, bool], list[str]] = {}
            for p in paragraphs:
                if p.translated_text:
                    key = (p.is_bold, p.is_italic)
                    if key not in texts_by_style:
                        texts_by_style[key] = []
                    texts_by_style[key].append(p.translated_text)

            if texts_by_style:
                subset_config = SubsetConfig(cache_dir=self._config.font_subset_cache_dir)
                subsetter = FontSubsetter(subset_config)

                # Create subset for each style variant with only its texts
                for (is_bold, is_italic), style_texts in texts_by_style.items():
                    subset_path = subsetter.subset_for_texts(
                        font_path=font_path,
                        texts=style_texts,  # Only texts for this style
                        font_number=self._config.cjk_font_number,
                        is_bold=is_bold,
                        is_italic=is_italic,
                    )
                    if subset_path:
                        font_subsets[(is_bold, is_italic)] = subset_path
                        logger.debug(
                            "Created font subset: bold=%s, italic=%s -> %s (%d chars)",
                            is_bold,
                            is_italic,
                            subset_path.name,
                            len(set("".join(style_texts))),
                        )

        # When side_by_side is enabled, draw debug boxes on original PDF instead
        draw_debug = pre_merge_paragraphs is not None and not self._config.side_by_side

        try:
            with PDFProcessor(pdf_path) as processor:
                processor.apply_paragraphs(
                    paragraphs,
                    font_path=font_path,
                    font_subsets=font_subsets if font_subsets else None,
                    min_font_size=self._config.min_font_size,
                )

                # Draw debug overlay showing original and merged paragraphs
                if draw_debug and pre_merge_paragraphs is not None:
                    processor.draw_merge_debug_overlay(
                        original_paragraphs=pre_merge_paragraphs,
                        merged_paragraphs=paragraphs,
                    )

                self._notify("apply", 1, 1)
                return processor.to_bytes()
        finally:
            # Clean up subset files if not using cache directory
            if subsetter and not self._config.font_subset_cache_dir:
                subsetter.cleanup()

    def _stage_side_by_side(
        self,
        original_pdf_path: Path,
        translated_pdf_bytes: bytes,
        paragraphs: list[Paragraph],
        pre_merge_paragraphs: list[Paragraph] | None = None,
    ) -> bytes:
        """Generate side-by-side comparison PDF.

        When debug_draw_bbox is enabled, debug boxes are drawn on the original
        PDF side instead of the translated PDF, allowing users to see clean
        translation results while viewing layout analysis on the original.

        Args:
            original_pdf_path: Path to the original PDF.
            translated_pdf_bytes: Bytes of the translated PDF.
            paragraphs: List of merged paragraphs.
            pre_merge_paragraphs: List of original paragraphs before merge
                (for debug overlay).

        Returns:
            bytes: The side-by-side PDF bytes.
        """
        config = SideBySideConfig(
            order=self._config.side_by_side_order,
            gap=self._config.side_by_side_gap,
        )
        generator = SideBySideGenerator(config)

        # Draw debug boxes on original PDF if enabled
        if pre_merge_paragraphs is not None:
            with PDFProcessor(original_pdf_path) as processor:
                processor.draw_merge_debug_overlay(
                    original_paragraphs=pre_merge_paragraphs,
                    merged_paragraphs=paragraphs,
                )
                original_bytes = processor.to_bytes()
        else:
            original_bytes = original_pdf_path.read_bytes()

        result = generator.generate(
            translated_pdf=translated_pdf_bytes,
            original_pdf=original_bytes,
        )
        self._notify("side_by_side", 1, 1)
        return result

    async def _translate_with_retry(self, texts: list[str]) -> list[str]:
        """Translate texts with retry and fallback logic.

        On ArrayLengthMismatchError, splits the batch and retries.
        """
        last_error: TranslationError | None = None
        for attempt in range(self._config.max_retries + 1):
            try:
                return await self._translator.translate_batch(
                    texts,
                    self._config.source_lang,
                    self._config.target_lang,
                )
            except ConfigurationError:
                raise
            except ArrayLengthMismatchError:
                # Batch split fallback (do not retry with same input)
                logger.warning(
                    "Array length mismatch, splitting batch of %d texts",
                    len(texts),
                )
                return await self._translate_with_split(texts)
            except TranslationError as exc:
                last_error = exc
                if attempt < self._config.max_retries:
                    delay = self._config.retry_delay * (2**attempt)
                    await asyncio.sleep(delay)

        raise PipelineError(
            f"Translation failed after {self._config.max_retries} retries",
            stage="translate",
            cause=last_error,
        )

    async def _translate_with_split(self, texts: list[str]) -> list[str]:
        """Translate texts by splitting batch recursively.

        When ArrayLengthMismatchError occurs, split the batch in half
        and retry. If batch size is 1, fall back to individual translation.
        """
        if len(texts) == 0:
            return []

        if len(texts) == 1:
            # Fall back to individual translation
            try:
                result = await self._translator.translate(
                    texts[0],
                    self._config.source_lang,
                    self._config.target_lang,
                )
                return [result]
            except TranslationError as exc:
                if self._config.strict_mode:
                    # Strict mode: re-raise error
                    raise
                else:
                    # Lenient mode (default): log warning and return original
                    logger.warning(
                        "Translation failed for text (len=%d), returning original: %s",
                        len(texts[0]),
                        exc,
                    )
                    return [texts[0]]

        # Split batch in half
        mid = len(texts) // 2
        left = await self._translate_with_retry(texts[:mid])
        right = await self._translate_with_retry(texts[mid:])
        return left + right

    def _chunk_texts(self, texts: list[str]) -> list[list[str]]:
        batch_size = max(1, int(self._config.translation_batch_size))
        return [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]

    def _notify(self, stage: str, current: int, total: int, message: str = "") -> None:
        if self._progress_callback is None:
            return
        self._progress_callback(stage, current, total, message)
