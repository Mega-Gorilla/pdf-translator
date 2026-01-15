# SPDX-License-Identifier: Apache-2.0
"""Tests for BaseDocument module."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from pdf_translator.core.models import BBox, Paragraph
from pdf_translator.output.base_document import (
    BaseDocument,
    BaseDocumentMetadata,
)
from pdf_translator.output.base_summary import BaseSummary


class TestBaseDocumentMetadata:
    """Test BaseDocumentMetadata class."""

    def test_metadata_to_dict(self) -> None:
        """Test metadata serialization."""
        meta = BaseDocumentMetadata(
            source_file="test.pdf",
            source_lang="en",
            page_count=10,
            paragraph_count=50,
        )
        d = meta.to_dict()

        assert d["source_file"] == "test.pdf"
        assert d["source_lang"] == "en"
        assert d["page_count"] == 10
        assert d["paragraph_count"] == 50

    def test_metadata_from_dict(self) -> None:
        """Test metadata deserialization."""
        d = {
            "source_file": "paper.pdf",
            "source_lang": "ja",
            "page_count": 5,
            "paragraph_count": 20,
        }
        meta = BaseDocumentMetadata.from_dict(d)

        assert meta.source_file == "paper.pdf"
        assert meta.source_lang == "ja"
        assert meta.page_count == 5
        assert meta.paragraph_count == 20


class TestBaseDocument:
    """Test BaseDocument class."""

    def _create_test_paragraphs(self) -> list[Paragraph]:
        """Create test paragraphs."""
        return [
            Paragraph(
                id="para_1",
                page_number=0,
                text="First paragraph",
                block_bbox=BBox(x0=10, y0=700, x1=200, y1=720),
                line_count=1,
                category="text",
                translated_text="最初の段落",  # Should be stripped in base document
            ),
            Paragraph(
                id="para_2",
                page_number=0,
                text="Second paragraph",
                block_bbox=BBox(x0=10, y0=650, x1=200, y1=690),
                line_count=2,
                category="text",
            ),
        ]

    def test_to_json_excludes_translated_text(self) -> None:
        """Test that translated_text is excluded from JSON."""
        paragraphs = self._create_test_paragraphs()
        doc = BaseDocument.from_pipeline_result(
            paragraphs=paragraphs,
            source_file="test.pdf",
            source_lang="en",
            total_pages=2,
        )
        json_str = doc.to_json()
        data = json.loads(json_str)

        assert data["schema_version"] == "2.0.0"
        assert data["metadata"]["source_file"] == "test.pdf"
        assert data["metadata"]["page_count"] == 2
        assert data["metadata"]["paragraph_count"] == 2
        assert len(data["paragraphs"]) == 2
        # translated_text should be excluded
        assert "translated_text" not in data["paragraphs"][0]

    def test_from_json(self) -> None:
        """Test JSON deserialization."""
        paragraphs = self._create_test_paragraphs()
        doc = BaseDocument.from_pipeline_result(
            paragraphs=paragraphs,
            source_file="test.pdf",
            source_lang="en",
            total_pages=2,
        )
        json_str = doc.to_json()
        restored = BaseDocument.from_json(json_str)

        assert restored.metadata.source_file == "test.pdf"
        assert len(restored.paragraphs) == 2

    def test_save_and_load(self) -> None:
        """Test file save and load."""
        paragraphs = self._create_test_paragraphs()
        doc = BaseDocument.from_pipeline_result(
            paragraphs=paragraphs,
            source_file="paper.pdf",
            source_lang="en",
            total_pages=5,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "paper.json"
            doc.save(path)

            assert path.exists()

            loaded = BaseDocument.load(path)
            assert loaded.metadata.source_file == "paper.pdf"
            assert loaded.metadata.page_count == 5
            assert len(loaded.paragraphs) == 2

    def test_with_summary(self) -> None:
        """Test with BaseSummary."""
        paragraphs = self._create_test_paragraphs()
        summary = BaseSummary(
            title="Test Title",
            abstract="Test abstract",
            page_count=2,
        )
        doc = BaseDocument.from_pipeline_result(
            paragraphs=paragraphs,
            source_file="test.pdf",
            source_lang="en",
            total_pages=2,
            summary=summary,
        )

        json_str = doc.to_json()
        data = json.loads(json_str)

        assert "summary" in data
        assert data["summary"]["title"] == "Test Title"
        assert data["summary"]["abstract"] == "Test abstract"

        # Roundtrip test
        restored = BaseDocument.from_json(json_str)
        assert restored.summary is not None
        assert restored.summary.title == "Test Title"
