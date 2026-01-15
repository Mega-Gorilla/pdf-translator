# SPDX-License-Identifier: Apache-2.0
"""Tests for BaseSummary dataclass."""

import pytest

from pdf_translator.output.base_summary import BaseSummary


class TestBaseSummary:
    """Tests for BaseSummary."""

    def test_default_values(self) -> None:
        """Test default values."""
        summary = BaseSummary()

        assert summary.title is None
        assert summary.abstract is None
        assert summary.organization is None
        assert summary.summary is None
        assert summary.thumbnail_path is None
        assert summary.thumbnail_width == 0
        assert summary.thumbnail_height == 0
        assert summary.page_count == 0
        assert summary.title_source == "layout"
        assert summary.abstract_source == "layout"
        assert summary._thumbnail_bytes is None

    def test_with_values(self) -> None:
        """Test with explicit values."""
        summary = BaseSummary(
            title="Test Title",
            abstract="Test abstract",
            organization="Test Org",
            summary="Test summary",
            thumbnail_path="thumbnail.png",
            thumbnail_width=400,
            thumbnail_height=518,
            page_count=10,
            title_source="layout",
            abstract_source="llm",
        )

        assert summary.title == "Test Title"
        assert summary.abstract == "Test abstract"
        assert summary.organization == "Test Org"
        assert summary.page_count == 10
        assert summary.abstract_source == "llm"


class TestBaseSummaryToDict:
    """Tests for to_dict method."""

    def test_to_dict_basic(self) -> None:
        """Test basic to_dict."""
        summary = BaseSummary(
            title="Test Title",
            thumbnail_path="thumb.png",
            page_count=5,
        )

        result = summary.to_dict()

        assert result["title"] == "Test Title"
        assert result["thumbnail_path"] == "thumb.png"
        assert result["page_count"] == 5
        assert "thumbnail_base64" not in result

    def test_to_dict_with_thumbnail_base64(self) -> None:
        """Test to_dict with base64 thumbnail."""
        summary = BaseSummary(
            title="Test Title",
            _thumbnail_bytes=b"PNG data",
        )

        result = summary.to_dict(include_thumbnail_base64=True)

        assert "thumbnail_base64" in result
        assert result["thumbnail_base64"] == "UE5HIGRhdGE="  # base64 of "PNG data"

    def test_to_dict_without_thumbnail_bytes(self) -> None:
        """Test to_dict with flag but no thumbnail bytes."""
        summary = BaseSummary(title="Test Title")

        result = summary.to_dict(include_thumbnail_base64=True)

        assert "thumbnail_base64" not in result


class TestBaseSummaryFromDict:
    """Tests for from_dict method."""

    def test_from_dict_basic(self) -> None:
        """Test basic from_dict."""
        data = {
            "title": "Test Title",
            "abstract": None,
            "organization": "Test Org",
            "summary": None,
            "thumbnail_path": "thumb.png",
            "thumbnail_width": 400,
            "thumbnail_height": 518,
            "page_count": 5,
            "title_source": "layout",
            "abstract_source": "layout",
        }

        summary = BaseSummary.from_dict(data)

        assert summary.title == "Test Title"
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

        summary = BaseSummary.from_dict(data)

        assert summary._thumbnail_bytes == b"PNG data"

    def test_from_dict_missing_optional_fields(self) -> None:
        """Test from_dict with missing optional fields."""
        data = {"title": "Test Title"}

        summary = BaseSummary.from_dict(data)

        assert summary.title == "Test Title"
        assert summary.thumbnail_width == 0
        assert summary.page_count == 0
        assert summary.title_source == "layout"


class TestBaseSummaryRoundTrip:
    """Tests for to_dict/from_dict round trip."""

    def test_round_trip(self) -> None:
        """Test round trip serialization."""
        original = BaseSummary(
            title="Test Title",
            abstract="Test abstract",
            organization="Test Org",
            thumbnail_path="thumb.png",
            thumbnail_width=400,
            thumbnail_height=518,
            page_count=10,
            title_source="layout",
            abstract_source="llm",
        )

        data = original.to_dict()
        restored = BaseSummary.from_dict(data)

        assert restored.title == original.title
        assert restored.abstract == original.abstract
        assert restored.organization == original.organization
        assert restored.thumbnail_path == original.thumbnail_path
        assert restored.thumbnail_width == original.thumbnail_width
        assert restored.thumbnail_height == original.thumbnail_height
        assert restored.page_count == original.page_count
        assert restored.title_source == original.title_source
        assert restored.abstract_source == original.abstract_source


class TestBaseSummaryHasContent:
    """Tests for has_content method."""

    def test_has_content_empty(self) -> None:
        """Test has_content with empty summary."""
        summary = BaseSummary()
        assert summary.has_content() is False

    def test_has_content_with_title(self) -> None:
        """Test has_content with title."""
        summary = BaseSummary(title="Test Title")
        assert summary.has_content() is True

    def test_has_content_with_abstract(self) -> None:
        """Test has_content with abstract."""
        summary = BaseSummary(abstract="Test abstract")
        assert summary.has_content() is True

    def test_has_content_with_summary(self) -> None:
        """Test has_content with summary."""
        summary = BaseSummary(summary="Test summary")
        assert summary.has_content() is True

    def test_has_content_with_thumbnail(self) -> None:
        """Test has_content with thumbnail_path."""
        summary = BaseSummary(thumbnail_path="thumb.png")
        assert summary.has_content() is True

    def test_has_content_with_only_metadata(self) -> None:
        """Test has_content with only metadata (no content)."""
        summary = BaseSummary(page_count=10)
        assert summary.has_content() is False
