# SPDX-License-Identifier: Apache-2.0
"""Translated document intermediate data format.

This module provides classes for saving and loading translated document data,
enabling Markdown regeneration without re-translation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from pdf_translator.core.models import Paragraph

TRANSLATED_DOC_VERSION = "1.0.0"


@dataclass
class TranslatedDocumentMetadata:
    """Metadata for a translated document."""

    source_file: str
    source_lang: str
    target_lang: str
    translated_at: str
    translator_backend: str
    page_count: int
    paragraph_count: int
    translated_count: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source_file": self.source_file,
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "translated_at": self.translated_at,
            "translator_backend": self.translator_backend,
            "page_count": self.page_count,
            "paragraph_count": self.paragraph_count,
            "translated_count": self.translated_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TranslatedDocumentMetadata:
        """Create from dictionary."""
        return cls(
            source_file=data["source_file"],
            source_lang=data["source_lang"],
            target_lang=data["target_lang"],
            translated_at=data["translated_at"],
            translator_backend=data["translator_backend"],
            page_count=int(data["page_count"]),
            paragraph_count=int(data["paragraph_count"]),
            translated_count=int(data["translated_count"]),
        )


@dataclass
class TranslatedDocument:
    """Translated document with paragraphs and metadata.

    This class enables saving translation results to JSON and regenerating
    Markdown output without re-translation.
    """

    metadata: TranslatedDocumentMetadata
    paragraphs: list[Paragraph]

    def to_json(self, indent: int = 2) -> str:
        """Export to JSON string.

        Args:
            indent: JSON indentation level.

        Returns:
            JSON string representation.
        """
        data = {
            "version": TRANSLATED_DOC_VERSION,
            "metadata": self.metadata.to_dict(),
            "paragraphs": [p.to_dict() for p in self.paragraphs],
        }
        return json.dumps(data, indent=indent, ensure_ascii=False)

    def save(self, path: Path) -> None:
        """Save to JSON file.

        Args:
            path: Output file path.
        """
        path.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def from_json(cls, json_str: str) -> TranslatedDocument:
        """Create from JSON string.

        Args:
            json_str: JSON string representation.

        Returns:
            TranslatedDocument instance.

        Raises:
            ValueError: If version is unsupported.
        """
        data = json.loads(json_str)
        version = data.get("version", "unknown")
        if version != TRANSLATED_DOC_VERSION:
            raise ValueError(
                f"Unsupported version: {version} (expected {TRANSLATED_DOC_VERSION})"
            )
        return cls(
            metadata=TranslatedDocumentMetadata.from_dict(data["metadata"]),
            paragraphs=[Paragraph.from_dict(p) for p in data["paragraphs"]],
        )

    @classmethod
    def load(cls, path: Path) -> TranslatedDocument:
        """Load from JSON file.

        Args:
            path: Input file path.

        Returns:
            TranslatedDocument instance.
        """
        return cls.from_json(path.read_text(encoding="utf-8"))

    @classmethod
    def from_pipeline_result(
        cls,
        paragraphs: list[Paragraph],
        source_file: str,
        source_lang: str,
        target_lang: str,
        translator_backend: str,
        total_pages: int,
    ) -> TranslatedDocument:
        """Create from pipeline result.

        Args:
            paragraphs: Translated paragraph list.
            source_file: Original PDF file name.
            source_lang: Source language code.
            target_lang: Target language code.
            translator_backend: Translator backend name.
            total_pages: Total PDF page count (including empty pages).

        Returns:
            TranslatedDocument instance.

        Note:
            page_count uses total_pages parameter instead of deriving from
            paragraphs to avoid gaps for empty pages.
        """
        translated_count = sum(1 for p in paragraphs if p.translated_text)
        metadata = TranslatedDocumentMetadata(
            source_file=source_file,
            source_lang=source_lang,
            target_lang=target_lang,
            translated_at=datetime.now().isoformat(),
            translator_backend=translator_backend,
            page_count=total_pages,
            paragraph_count=len(paragraphs),
            translated_count=translated_count,
        )
        return cls(metadata=metadata, paragraphs=paragraphs)
