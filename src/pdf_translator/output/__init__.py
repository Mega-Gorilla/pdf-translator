# SPDX-License-Identifier: Apache-2.0
"""Output format modules for PDF Translator.

This module provides output format support including Markdown generation,
image extraction, and table extraction.
"""

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
from pdf_translator.output.table_extractor import (
    ExtractedTable,
    TableCell,
    TableExtractionConfig,
    TableExtractionError,
    TableExtractor,
)
from pdf_translator.output.translated_document import (
    TRANSLATED_DOC_VERSION,
    TranslatedDocument,
    TranslatedDocumentMetadata,
)

__all__ = [
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
    # Translated document
    "TRANSLATED_DOC_VERSION",
    "TranslatedDocument",
    "TranslatedDocumentMetadata",
]
