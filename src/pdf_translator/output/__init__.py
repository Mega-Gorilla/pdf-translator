# SPDX-License-Identifier: Apache-2.0
"""Output format modules for PDF Translator.

This module provides output format support including Markdown generation,
image extraction, and table extraction.
"""

from pdf_translator.output.base_document import (
    BaseDocument,
    BaseDocumentMetadata,
)
from pdf_translator.output.base_document_writer import BaseDocumentWriter
from pdf_translator.output.base_summary import BaseSummary
from pdf_translator.output.image_extractor import (
    ExtractedImage,
    ImageExtractionConfig,
    ImageExtractor,
    extract_image_as_fallback,
)
from pdf_translator.output.markdown_writer import (
    DEFAULT_CATEGORY_MAPPING,
    DEFAULT_MARKDOWN_SKIP_CATEGORIES,
    DEFAULT_NONE_CATEGORY_MAPPING,
    MarkdownConfig,
    MarkdownOutputMode,
    MarkdownWriter,
)
from pdf_translator.output.summary_extractor import SummaryExtractor
from pdf_translator.output.table_extractor import (
    ExtractedTable,
    TableCell,
    TableExtractionConfig,
    TableExtractionError,
    TableExtractor,
)
from pdf_translator.output.thumbnail_generator import ThumbnailConfig, ThumbnailGenerator
from pdf_translator.output.translated_summary import TranslatedSummary
from pdf_translator.output.translation_document import (
    SCHEMA_VERSION,
    TranslationDocument,
)
from pdf_translator.output.translation_writer import TranslationWriter

__all__ = [
    # Base document
    "BaseDocument",
    "BaseDocumentMetadata",
    "BaseDocumentWriter",
    # Base summary
    "BaseSummary",
    # Image extractor
    "ImageExtractor",
    "ImageExtractionConfig",
    "ExtractedImage",
    "extract_image_as_fallback",
    # Table extractor
    "TableExtractor",
    "TableExtractionConfig",
    "TableExtractionError",
    "ExtractedTable",
    "TableCell",
    # Markdown writer
    "MarkdownWriter",
    "MarkdownConfig",
    "MarkdownOutputMode",
    "DEFAULT_CATEGORY_MAPPING",
    "DEFAULT_MARKDOWN_SKIP_CATEGORIES",
    "DEFAULT_NONE_CATEGORY_MAPPING",
    # Translation document
    "SCHEMA_VERSION",
    "TranslationDocument",
    "TranslationWriter",
    # Translated summary
    "TranslatedSummary",
    # Thumbnail generator
    "ThumbnailConfig",
    "ThumbnailGenerator",
    # Summary extractor
    "SummaryExtractor",
]
