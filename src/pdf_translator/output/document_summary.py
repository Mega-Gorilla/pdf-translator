# SPDX-License-Identifier: Apache-2.0
"""Document summary data model for web service integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class DocumentSummary:
    """Document summary for web service integration.

    Contains essential metadata extracted from translated documents
    for dashboard display, search, and document management.

    Attributes:
        title: Original document title (from doc_title category or LLM fallback).
        title_translated: Translated document title.
        abstract: Original abstract text (from abstract category or LLM fallback).
        abstract_translated: Translated abstract text.
        organization: Institution/company name (LLM extracted).
        summary: LLM-generated summary of the document.
        summary_translated: Translated LLM-generated summary.
        thumbnail_path: Relative path to thumbnail file (primary reference).
        thumbnail_width: Thumbnail width in pixels.
        thumbnail_height: Thumbnail height in pixels.
        page_count: Total number of pages in the document.
        source_lang: Source language code (e.g., "en").
        target_lang: Target language code (e.g., "ja").
        title_source: Source of title extraction ("layout" or "llm").
        abstract_source: Source of abstract extraction ("layout" or "llm").
    """

    # Title (from doc_title category or LLM fallback)
    title: str | None = None
    title_translated: str | None = None

    # Abstract (from abstract category or LLM fallback, may be merged from multiple paragraphs)
    abstract: str | None = None
    abstract_translated: str | None = None

    # Organization (LLM extracted only - not available via layout analysis)
    organization: str | None = None

    # LLM-generated summary (from full original Markdown)
    summary: str | None = None
    summary_translated: str | None = None

    # Thumbnail (first page of original PDF)
    thumbnail_path: str | None = None  # Relative path to thumbnail file
    thumbnail_width: int = 0
    thumbnail_height: int = 0

    # Metadata
    page_count: int = 0
    source_lang: str = ""
    target_lang: str = ""

    # Extraction source tracking (for debugging/quality monitoring)
    title_source: Literal["layout", "llm"] = "layout"
    abstract_source: Literal["layout", "llm"] = "layout"

    # Internal: thumbnail bytes (not serialized by default)
    _thumbnail_bytes: bytes | None = field(default=None, repr=False)

    def to_dict(self, include_thumbnail_base64: bool = False) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Args:
            include_thumbnail_base64: If True, include base64-encoded thumbnail.
                Default False - use thumbnail_path instead.

        Returns:
            Dictionary representation.
        """
        result: dict[str, Any] = {
            "title": self.title,
            "title_translated": self.title_translated,
            "abstract": self.abstract,
            "abstract_translated": self.abstract_translated,
            "organization": self.organization,
            "summary": self.summary,
            "summary_translated": self.summary_translated,
            "thumbnail_path": self.thumbnail_path,
            "thumbnail_width": self.thumbnail_width,
            "thumbnail_height": self.thumbnail_height,
            "page_count": self.page_count,
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "title_source": self.title_source,
            "abstract_source": self.abstract_source,
        }

        if include_thumbnail_base64 and self._thumbnail_bytes:
            import base64

            result["thumbnail_base64"] = base64.b64encode(
                self._thumbnail_bytes
            ).decode("ascii")

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DocumentSummary:
        """Create from dictionary.

        Args:
            data: Dictionary with summary fields.

        Returns:
            DocumentSummary instance.
        """
        thumbnail_bytes = None
        if "thumbnail_base64" in data:
            import base64

            thumbnail_bytes = base64.b64decode(data["thumbnail_base64"])

        return cls(
            title=data.get("title"),
            title_translated=data.get("title_translated"),
            abstract=data.get("abstract"),
            abstract_translated=data.get("abstract_translated"),
            organization=data.get("organization"),
            summary=data.get("summary"),
            summary_translated=data.get("summary_translated"),
            thumbnail_path=data.get("thumbnail_path"),
            thumbnail_width=data.get("thumbnail_width", 0),
            thumbnail_height=data.get("thumbnail_height", 0),
            page_count=data.get("page_count", 0),
            source_lang=data.get("source_lang", ""),
            target_lang=data.get("target_lang", ""),
            title_source=data.get("title_source", "layout"),
            abstract_source=data.get("abstract_source", "layout"),
            _thumbnail_bytes=thumbnail_bytes,
        )

    def has_content(self) -> bool:
        """Check if summary has any meaningful content.

        Returns:
            True if at least title, abstract, summary, or thumbnail is present.
        """
        return bool(
            self.title or self.abstract or self.summary or self.thumbnail_path
        )
