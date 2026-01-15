# SPDX-License-Identifier: Apache-2.0
"""Tests for DocumentSummary dataclass."""

import pytest

from pdf_translator.output.document_summary import DocumentSummary


class TestDocumentSummary:
    """Tests for DocumentSummary."""

    def test_default_values(self) -> None:
        """Test default values."""
        summary = DocumentSummary()

        assert summary.title is None
        assert summary.title_translated is None
        assert summary.abstract is None
        assert summary.abstract_translated is None
        assert summary.organization is None
        assert summary.summary is None
        assert summary.summary_translated is None
        assert summary.thumbnail_path is None
        assert summary.thumbnail_width == 0
        assert summary.thumbnail_height == 0
        assert summary.page_count == 0
        assert summary.source_lang == ""
        assert summary.target_lang == ""
        assert summary.title_source == "layout"
        assert summary.abstract_source == "layout"
        assert summary._thumbnail_bytes is None

    def test_with_values(self) -> None:
        """Test with explicit values."""
        summary = DocumentSummary(
            title="Test Title",
            title_translated="テストタイトル",
            abstract="Test abstract",
            abstract_translated="テスト要約",
            organization="Test Org",
            summary="Test summary",
            summary_translated="テストサマリー",
            thumbnail_path="thumbnail.png",
            thumbnail_width=400,
            thumbnail_height=518,
            page_count=10,
            source_lang="en",
            target_lang="ja",
            title_source="layout",
            abstract_source="llm",
        )

        assert summary.title == "Test Title"
        assert summary.title_translated == "テストタイトル"
        assert summary.organization == "Test Org"
        assert summary.page_count == 10
        assert summary.abstract_source == "llm"


class TestDocumentSummaryToDict:
    """Tests for to_dict method."""

    def test_to_dict_basic(self) -> None:
        """Test basic to_dict."""
        summary = DocumentSummary(
            title="Test Title",
            thumbnail_path="thumb.png",
            page_count=5,
            source_lang="en",
            target_lang="ja",
        )

        result = summary.to_dict()

        assert result["title"] == "Test Title"
        assert result["thumbnail_path"] == "thumb.png"
        assert result["page_count"] == 5
        assert result["source_lang"] == "en"
        assert result["target_lang"] == "ja"
        assert "thumbnail_base64" not in result

    def test_to_dict_with_thumbnail_base64(self) -> None:
        """Test to_dict with base64 thumbnail."""
        summary = DocumentSummary(
            title="Test Title",
            _thumbnail_bytes=b"PNG data",
        )

        result = summary.to_dict(include_thumbnail_base64=True)

        assert "thumbnail_base64" in result
        assert result["thumbnail_base64"] == "UE5HIGRhdGE="  # base64 of "PNG data"

    def test_to_dict_without_thumbnail_bytes(self) -> None:
        """Test to_dict with flag but no thumbnail bytes."""
        summary = DocumentSummary(title="Test Title")

        result = summary.to_dict(include_thumbnail_base64=True)

        assert "thumbnail_base64" not in result


class TestDocumentSummaryFromDict:
    """Tests for from_dict method."""

    def test_from_dict_basic(self) -> None:
        """Test basic from_dict."""
        data = {
            "title": "Test Title",
            "title_translated": "テストタイトル",
            "abstract": None,
            "abstract_translated": None,
            "organization": "Test Org",
            "summary": None,
            "summary_translated": None,
            "thumbnail_path": "thumb.png",
            "thumbnail_width": 400,
            "thumbnail_height": 518,
            "page_count": 5,
            "source_lang": "en",
            "target_lang": "ja",
            "title_source": "layout",
            "abstract_source": "layout",
        }

        summary = DocumentSummary.from_dict(data)

        assert summary.title == "Test Title"
        assert summary.title_translated == "テストタイトル"
        assert summary.organization == "Test Org"
        assert summary.thumbnail_path == "thumb.png"
        assert summary.thumbnail_width == 400
        assert summary.page_count == 5

    def test_from_dict_with_thumbnail_base64(self) -> None:
        """Test from_dict with base64 thumbnail."""
        data = {
            "title": "Test Title",
            "thumbnail_base64": "UE5HIGRhdGE=",  # base64 of "PNG data"
        }

        summary = DocumentSummary.from_dict(data)

        assert summary._thumbnail_bytes == b"PNG data"

    def test_from_dict_missing_optional_fields(self) -> None:
        """Test from_dict with missing optional fields."""
        data = {"title": "Test Title"}

        summary = DocumentSummary.from_dict(data)

        assert summary.title == "Test Title"
        assert summary.thumbnail_width == 0
        assert summary.page_count == 0
        assert summary.source_lang == ""
        assert summary.title_source == "layout"


class TestDocumentSummaryRoundTrip:
    """Tests for to_dict/from_dict round trip."""

    def test_round_trip(self) -> None:
        """Test round trip serialization."""
        original = DocumentSummary(
            title="Test Title",
            title_translated="テストタイトル",
            abstract="Test abstract",
            organization="Test Org",
            thumbnail_path="thumb.png",
            thumbnail_width=400,
            thumbnail_height=518,
            page_count=10,
            source_lang="en",
            target_lang="ja",
            title_source="layout",
            abstract_source="llm",
        )

        data = original.to_dict()
        restored = DocumentSummary.from_dict(data)

        assert restored.title == original.title
        assert restored.title_translated == original.title_translated
        assert restored.abstract == original.abstract
        assert restored.organization == original.organization
        assert restored.thumbnail_path == original.thumbnail_path
        assert restored.thumbnail_width == original.thumbnail_width
        assert restored.thumbnail_height == original.thumbnail_height
        assert restored.page_count == original.page_count
        assert restored.source_lang == original.source_lang
        assert restored.target_lang == original.target_lang
        assert restored.title_source == original.title_source
        assert restored.abstract_source == original.abstract_source


class TestDocumentSummaryHasContent:
    """Tests for has_content method."""

    def test_has_content_empty(self) -> None:
        """Test has_content with empty summary."""
        summary = DocumentSummary()
        assert summary.has_content() is False

    def test_has_content_with_title(self) -> None:
        """Test has_content with title."""
        summary = DocumentSummary(title="Test Title")
        assert summary.has_content() is True

    def test_has_content_with_abstract(self) -> None:
        """Test has_content with abstract."""
        summary = DocumentSummary(abstract="Test abstract")
        assert summary.has_content() is True

    def test_has_content_with_summary(self) -> None:
        """Test has_content with summary."""
        summary = DocumentSummary(summary="Test summary")
        assert summary.has_content() is True

    def test_has_content_with_thumbnail(self) -> None:
        """Test has_content with thumbnail_path."""
        summary = DocumentSummary(thumbnail_path="thumb.png")
        assert summary.has_content() is True

    def test_has_content_with_only_metadata(self) -> None:
        """Test has_content with only metadata (no content)."""
        summary = DocumentSummary(
            page_count=10,
            source_lang="en",
            target_lang="ja",
        )
        assert summary.has_content() is False
