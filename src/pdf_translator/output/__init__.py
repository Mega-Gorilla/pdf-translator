# SPDX-License-Identifier: Apache-2.0
"""Output format modules for PDF Translator.

This module provides output format support including Markdown generation,
image extraction, and table extraction.
"""

from pdf_translator.output.markdown_writer import (
    DEFAULT_CATEGORY_MAPPING,
    DEFAULT_NONE_CATEGORY_MAPPING,
    MarkdownConfig,
    MarkdownOutputMode,
    MarkdownWriter,
)
from pdf_translator.output.translated_document import (
    TRANSLATED_DOC_VERSION,
    TranslatedDocument,
    TranslatedDocumentMetadata,
)

__all__ = [
    # Markdown writer
    "MarkdownWriter",
    "MarkdownConfig",
    "MarkdownOutputMode",
    "DEFAULT_CATEGORY_MAPPING",
    "DEFAULT_NONE_CATEGORY_MAPPING",
    # Translated document
    "TRANSLATED_DOC_VERSION",
    "TranslatedDocument",
    "TranslatedDocumentMetadata",
]
