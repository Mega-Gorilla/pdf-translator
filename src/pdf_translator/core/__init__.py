# SPDX-License-Identifier: Apache-2.0
"""Core PDF processing modules."""

from .font_adjuster import FontSizeAdjuster
from .models import (
    TRANSLATABLE_CATEGORIES,
    BBox,
    CharPosition,
    Color,
    Font,
    LayoutBlock,
    Metadata,
    Page,
    PDFDocument,
    ProjectCategory,
    TextObject,
    Transform,
)
from .pdf_processor import PDFProcessor
from .text_merger import TextMerger

__all__ = [
    "BBox",
    "CharPosition",
    "Color",
    "Font",
    "FontSizeAdjuster",
    "LayoutBlock",
    "Metadata",
    "Page",
    "PDFDocument",
    "PDFProcessor",
    "ProjectCategory",
    "TextMerger",
    "TextObject",
    "TRANSLATABLE_CATEGORIES",
    "Transform",
]
