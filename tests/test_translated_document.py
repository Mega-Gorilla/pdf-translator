# SPDX-License-Identifier: Apache-2.0
"""Tests for translated document module."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from pdf_translator.core.models import BBox, Color, Paragraph
from pdf_translator.output.translated_document import (
    TRANSLATED_DOC_VERSION,
    TranslatedDocument,
    TranslatedDocumentMetadata,
)


class TestParagraphSerialization:
    """Test Paragraph to_dict/from_dict methods."""

    def test_paragraph_to_dict_required_fields(self) -> None:
        """Test serialization of required fields."""
        para = Paragraph(
            id="test_1",
            page_number=0,
            text="Hello world",
            block_bbox=BBox(x0=10, y0=20, x1=100, y1=40),
            line_count=1,
        )
        d = para.to_dict()

        assert d["id"] == "test_1"
        assert d["page_number"] == 0
        assert d["text"] == "Hello world"
        assert d["block_bbox"] == {"x0": 10, "y0": 20, "x1": 100, "y1": 40}
        assert d["line_count"] == 1
        assert d["original_font_size"] == 12.0
        assert d["is_bold"] is False
        assert d["is_italic"] is False
        assert d["rotation"] == 0.0
        assert d["alignment"] == "left"

    def test_paragraph_to_dict_optional_fields(self) -> None:
        """Test serialization of optional fields."""
        para = Paragraph(
            id="test_1",
            page_number=1,
            text="Test text",
            block_bbox=BBox(x0=0, y0=0, x1=100, y1=50),
            line_count=2,
            layout_block_id="block_1",
            category="text",
            category_confidence=0.95,
            translated_text="翻訳されたテキスト",
            adjusted_font_size=10.5,
            is_bold=True,
            is_italic=True,
            font_name="Helvetica",
            text_color=Color(r=255, g=128, b=0),
            rotation=90.0,
            alignment="center",
        )
        d = para.to_dict()

        assert d["layout_block_id"] == "block_1"
        assert d["category"] == "text"
        assert d["category_confidence"] == 0.95
        assert d["translated_text"] == "翻訳されたテキスト"
        assert d["adjusted_font_size"] == 10.5
        assert d["is_bold"] is True
        assert d["is_italic"] is True
        assert d["font_name"] == "Helvetica"
        assert d["text_color"] == {"r": 255, "g": 128, "b": 0}
        assert d["rotation"] == 90.0
        assert d["alignment"] == "center"

    def test_paragraph_to_dict_excludes_none_optional(self) -> None:
        """Test that None optional fields are excluded."""
        para = Paragraph(
            id="test_1",
            page_number=0,
            text="Hello",
            block_bbox=BBox(x0=0, y0=0, x1=10, y1=10),
            line_count=1,
        )
        d = para.to_dict()

        assert "layout_block_id" not in d
        assert "category" not in d
        assert "category_confidence" not in d
        assert "translated_text" not in d
        assert "adjusted_font_size" not in d
        assert "font_name" not in d
        assert "text_color" not in d

    def test_paragraph_from_dict_required_fields(self) -> None:
        """Test deserialization of required fields."""
        d = {
            "id": "para_1",
            "page_number": 2,
            "text": "Sample text",
            "block_bbox": {"x0": 5.0, "y0": 10.0, "x1": 95.0, "y1": 30.0},
            "line_count": 3,
        }
        para = Paragraph.from_dict(d)

        assert para.id == "para_1"
        assert para.page_number == 2
        assert para.text == "Sample text"
        assert para.block_bbox.x0 == 5.0
        assert para.block_bbox.y1 == 30.0
        assert para.line_count == 3
        assert para.original_font_size == 12.0  # default
        assert para.layout_block_id is None
        assert para.category is None
        assert para.is_bold is False
        assert para.alignment == "left"

    def test_paragraph_from_dict_optional_fields(self) -> None:
        """Test deserialization of optional fields."""
        d = {
            "id": "para_1",
            "page_number": 0,
            "text": "Text",
            "block_bbox": {"x0": 0, "y0": 0, "x1": 100, "y1": 50},
            "line_count": 1,
            "original_font_size": 14.0,
            "layout_block_id": "lb_001",
            "category": "paragraph_title",
            "category_confidence": 0.88,
            "translated_text": "翻訳",
            "adjusted_font_size": 12.0,
            "is_bold": True,
            "is_italic": False,
            "font_name": "Arial",
            "text_color": {"r": 0, "g": 0, "b": 255},
            "rotation": 45.0,
            "alignment": "right",
        }
        para = Paragraph.from_dict(d)

        assert para.original_font_size == 14.0
        assert para.layout_block_id == "lb_001"
        assert para.category == "paragraph_title"
        assert para.category_confidence == 0.88
        assert para.translated_text == "翻訳"
        assert para.adjusted_font_size == 12.0
        assert para.is_bold is True
        assert para.is_italic is False
        assert para.font_name == "Arial"
        assert para.text_color is not None
        assert para.text_color.b == 255
        assert para.rotation == 45.0
        assert para.alignment == "right"

    def test_paragraph_roundtrip(self) -> None:
        """Test serialization/deserialization roundtrip."""
        original = Paragraph(
            id="roundtrip_test",
            page_number=5,
            text="Roundtrip text with 日本語",
            block_bbox=BBox(x0=1.5, y0=2.5, x1=99.5, y1=49.5),
            line_count=2,
            original_font_size=11.0,
            layout_block_id="block_123",
            category="abstract",
            category_confidence=0.92,
            translated_text="往復テスト",
            adjusted_font_size=10.0,
            is_bold=True,
            is_italic=True,
            font_name="NotoSans",
            text_color=Color(r=128, g=64, b=32),
            rotation=270.0,
            alignment="justify",
        )

        d = original.to_dict()
        restored = Paragraph.from_dict(d)

        assert restored.id == original.id
        assert restored.page_number == original.page_number
        assert restored.text == original.text
        assert restored.block_bbox.x0 == original.block_bbox.x0
        assert restored.block_bbox.y1 == original.block_bbox.y1
        assert restored.line_count == original.line_count
        assert restored.original_font_size == original.original_font_size
        assert restored.layout_block_id == original.layout_block_id
        assert restored.category == original.category
        assert restored.category_confidence == original.category_confidence
        assert restored.translated_text == original.translated_text
        assert restored.adjusted_font_size == original.adjusted_font_size
        assert restored.is_bold == original.is_bold
        assert restored.is_italic == original.is_italic
        assert restored.font_name == original.font_name
        assert restored.text_color is not None
        assert restored.text_color.r == original.text_color.r
        assert restored.rotation == original.rotation
        assert restored.alignment == original.alignment


class TestTranslatedDocumentMetadata:
    """Test TranslatedDocumentMetadata class."""

    def test_metadata_to_dict(self) -> None:
        """Test metadata serialization."""
        meta = TranslatedDocumentMetadata(
            source_file="test.pdf",
            source_lang="en",
            target_lang="ja",
            translated_at="2026-01-07T12:00:00",
            translator_backend="google",
            page_count=10,
            paragraph_count=50,
            translated_count=45,
        )
        d = meta.to_dict()

        assert d["source_file"] == "test.pdf"
        assert d["source_lang"] == "en"
        assert d["target_lang"] == "ja"
        assert d["translated_at"] == "2026-01-07T12:00:00"
        assert d["translator_backend"] == "google"
        assert d["page_count"] == 10
        assert d["paragraph_count"] == 50
        assert d["translated_count"] == 45

    def test_metadata_from_dict(self) -> None:
        """Test metadata deserialization."""
        d = {
            "source_file": "paper.pdf",
            "source_lang": "ja",
            "target_lang": "en",
            "translated_at": "2026-01-07T15:30:00",
            "translator_backend": "deepl",
            "page_count": 5,
            "paragraph_count": 20,
            "translated_count": 18,
        }
        meta = TranslatedDocumentMetadata.from_dict(d)

        assert meta.source_file == "paper.pdf"
        assert meta.source_lang == "ja"
        assert meta.target_lang == "en"
        assert meta.translator_backend == "deepl"
        assert meta.page_count == 5
        assert meta.paragraph_count == 20
        assert meta.translated_count == 18


class TestTranslatedDocument:
    """Test TranslatedDocument class."""

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
                translated_text="最初の段落",
            ),
            Paragraph(
                id="para_2",
                page_number=0,
                text="Second paragraph",
                block_bbox=BBox(x0=10, y0=650, x1=200, y1=690),
                line_count=2,
                category="text",
                translated_text="2番目の段落",
            ),
            Paragraph(
                id="para_3",
                page_number=1,
                text="Title",
                block_bbox=BBox(x0=50, y0=750, x1=150, y1=780),
                line_count=1,
                category="doc_title",
                # No translation for title
            ),
        ]

    def test_to_json(self) -> None:
        """Test JSON serialization."""
        paragraphs = self._create_test_paragraphs()
        doc = TranslatedDocument.from_pipeline_result(
            paragraphs=paragraphs,
            source_file="test.pdf",
            source_lang="en",
            target_lang="ja",
            translator_backend="google",
            total_pages=2,
        )
        json_str = doc.to_json()
        data = json.loads(json_str)

        assert data["version"] == TRANSLATED_DOC_VERSION
        assert data["metadata"]["source_file"] == "test.pdf"
        assert data["metadata"]["page_count"] == 2
        assert data["metadata"]["paragraph_count"] == 3
        assert data["metadata"]["translated_count"] == 2
        assert len(data["paragraphs"]) == 3

    def test_from_json(self) -> None:
        """Test JSON deserialization."""
        paragraphs = self._create_test_paragraphs()
        doc = TranslatedDocument.from_pipeline_result(
            paragraphs=paragraphs,
            source_file="test.pdf",
            source_lang="en",
            target_lang="ja",
            translator_backend="openai",
            total_pages=2,
        )
        json_str = doc.to_json()
        restored = TranslatedDocument.from_json(json_str)

        assert restored.metadata.source_file == "test.pdf"
        assert restored.metadata.translator_backend == "openai"
        assert len(restored.paragraphs) == 3
        assert restored.paragraphs[0].translated_text == "最初の段落"
        assert restored.paragraphs[2].category == "doc_title"

    def test_from_json_version_mismatch(self) -> None:
        """Test error on version mismatch."""
        bad_json = json.dumps(
            {
                "version": "0.0.1",
                "metadata": {},
                "paragraphs": [],
            }
        )
        with pytest.raises(ValueError, match="Unsupported version"):
            TranslatedDocument.from_json(bad_json)

    def test_save_and_load(self) -> None:
        """Test file save and load."""
        paragraphs = self._create_test_paragraphs()
        doc = TranslatedDocument.from_pipeline_result(
            paragraphs=paragraphs,
            source_file="paper.pdf",
            source_lang="en",
            target_lang="ja",
            translator_backend="deepl",
            total_pages=5,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "translated.json"
            doc.save(path)

            assert path.exists()

            loaded = TranslatedDocument.load(path)
            assert loaded.metadata.source_file == "paper.pdf"
            assert loaded.metadata.page_count == 5
            assert len(loaded.paragraphs) == 3

    def test_from_pipeline_result(self) -> None:
        """Test creation from pipeline result."""
        paragraphs = [
            Paragraph(
                id="p1",
                page_number=0,
                text="Text 1",
                block_bbox=BBox(x0=0, y0=0, x1=100, y1=50),
                line_count=1,
                translated_text="翻訳1",
            ),
            Paragraph(
                id="p2",
                page_number=0,
                text="Text 2",
                block_bbox=BBox(x0=0, y0=50, x1=100, y1=100),
                line_count=1,
                # No translation
            ),
            Paragraph(
                id="p3",
                page_number=1,
                text="Text 3",
                block_bbox=BBox(x0=0, y0=0, x1=100, y1=50),
                line_count=1,
                translated_text="翻訳3",
            ),
        ]
        doc = TranslatedDocument.from_pipeline_result(
            paragraphs=paragraphs,
            source_file="input.pdf",
            source_lang="en",
            target_lang="ja",
            translator_backend="google",
            total_pages=3,  # More pages than paragraphs cover
        )

        assert doc.metadata.source_file == "input.pdf"
        assert doc.metadata.source_lang == "en"
        assert doc.metadata.target_lang == "ja"
        assert doc.metadata.translator_backend == "google"
        assert doc.metadata.page_count == 3
        assert doc.metadata.paragraph_count == 3
        assert doc.metadata.translated_count == 2  # Only 2 have translated_text
        assert len(doc.paragraphs) == 3

    def test_roundtrip_with_all_fields(self) -> None:
        """Test complete roundtrip with all paragraph fields."""
        para = Paragraph(
            id="full_test",
            page_number=3,
            text="Full test paragraph",
            block_bbox=BBox(x0=10.5, y0=20.5, x1=300.5, y1=100.5),
            line_count=4,
            original_font_size=14.5,
            layout_block_id="lb_full",
            category="abstract",
            category_confidence=0.99,
            translated_text="完全テスト段落",
            adjusted_font_size=13.0,
            is_bold=True,
            is_italic=True,
            font_name="NotoSansCJK",
            text_color=Color(r=100, g=150, b=200),
            rotation=180.0,
            alignment="justify",
        )
        doc = TranslatedDocument.from_pipeline_result(
            paragraphs=[para],
            source_file="full.pdf",
            source_lang="en",
            target_lang="ja",
            translator_backend="openai",
            total_pages=10,
        )

        json_str = doc.to_json()
        restored = TranslatedDocument.from_json(json_str)
        rp = restored.paragraphs[0]

        assert rp.id == "full_test"
        assert rp.layout_block_id == "lb_full"
        assert rp.category == "abstract"
        assert rp.category_confidence == 0.99
        assert rp.translated_text == "完全テスト段落"
        assert rp.is_bold is True
        assert rp.is_italic is True
        assert rp.font_name == "NotoSansCJK"
        assert rp.text_color is not None
        assert rp.text_color.r == 100
        assert rp.rotation == 180.0
        assert rp.alignment == "justify"
