# SPDX-License-Identifier: Apache-2.0
"""Core PDF processing modules."""

from .font_subsetter import FontSubsetter, SubsetConfig
from .models import (
    BBox,
    CharPosition,
    Color,
    Font,
    LayoutBlock,
    Metadata,
    Page,
    Paragraph,
    PDFDocument,
    TextObject,
    Transform,
)
from .pdf_processor import PDFProcessor

__all__ = [
    "BBox",
    "CharPosition",
    "Color",
    "Font",
    "FontSubsetter",
    "LayoutBlock",
    "Metadata",
    "Page",
    "Paragraph",
    "PDFDocument",
    "PDFProcessor",
    "SubsetConfig",
    "TextObject",
    "Transform",
]
