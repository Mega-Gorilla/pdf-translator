# SPDX-License-Identifier: Apache-2.0
"""Translation pipeline implementation."""

from __future__ import annotations

import asyncio
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pdf_translator.core.layout_analyzer import LayoutAnalyzer
from pdf_translator.core.layout_utils import assign_categories
from pdf_translator.core.models import LayoutBlock, Paragraph
from pdf_translator.core.paragraph_extractor import ParagraphExtractor
from pdf_translator.core.pdf_processor import PDFProcessor
from pdf_translator.pipeline.errors import (
    ExtractionError,
    LayoutAnalysisError,
    PipelineError,
)
from pdf_translator.pipeline.progress import ProgressCallback
from pdf_translator.translators.base import ConfigurationError, TranslationError, TranslatorBackend

logger = logging.getLogger(__name__)


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

    cjk_font_path: Path | None = None

    # Debug options
    debug_draw_bbox: bool = False


@dataclass
class TranslationResult:
    """Translation pipeline result."""

    pdf_bytes: bytes
    stats: dict[str, Any] | None = None


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
        layout_blocks = await self._stage_analyze(pdf_path)
        self._stage_categorize(paragraphs, layout_blocks)
        translatable = await self._stage_translate(paragraphs)
        pdf_bytes = self._stage_apply(pdf_path, paragraphs)

        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(pdf_bytes)

        stats = {
            "paragraphs": len(paragraphs),
            "translated_paragraphs": len(translatable),
            "skipped_paragraphs": len(paragraphs) - len(translatable),
        }
        return TranslationResult(pdf_bytes=pdf_bytes, stats=stats)

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

    async def _stage_translate(self, paragraphs: list[Paragraph]) -> list[Paragraph]:
        categories = self._config.translatable_categories
        translatable = [
            para for para in paragraphs
            if para.is_translatable(categories)
        ]
        if not translatable:
            return []

        texts = [para.text for para in translatable]
        translated_texts: list[str] = []
        total = len(texts)

        for chunk in self._chunk_texts(texts):
            chunk_translations = await self._translate_with_retry(chunk)
            if len(chunk_translations) != len(chunk):
                raise PipelineError(
                    "Translator returned unexpected number of results",
                    stage="translate",
                )
            translated_texts.extend(chunk_translations)
            self._notify("translate", len(translated_texts), total)

        for para, translated in zip(translatable, translated_texts):
            para.translated_text = translated

        return translatable

    def _stage_apply(self, pdf_path: Path, paragraphs: list[Paragraph]) -> bytes:
        target_lang = self._config.target_lang.lower()
        font_path = None
        if target_lang in {"ja", "zh", "ko"}:
            if self._config.cjk_font_path is not None:
                font_path = self._config.cjk_font_path
            else:
                logger.warning(
                    "Target language is CJK but no cjk_font_path is set; output may be garbled."
                )

        with PDFProcessor(pdf_path) as processor:
            processor.apply_paragraphs(
                paragraphs,
                font_path=font_path,
                debug_draw_bbox=self._config.debug_draw_bbox,
                min_font_size=self._config.min_font_size,
            )
            self._notify("apply", 1, 1)
            return processor.to_bytes()

    async def _translate_with_retry(self, texts: list[str]) -> list[str]:
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

    def _chunk_texts(self, texts: list[str]) -> list[list[str]]:
        batch_size = max(1, int(self._config.translation_batch_size))
        return [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]

    def _notify(self, stage: str, current: int, total: int, message: str = "") -> None:
        if self._progress_callback is None:
            return
        self._progress_callback(stage, current, total, message)
