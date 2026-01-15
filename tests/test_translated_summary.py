# SPDX-License-Identifier: Apache-2.0
"""Tests for TranslatedSummary dataclass."""

from pdf_translator.output.translated_summary import TranslatedSummary


class TestTranslatedSummary:
    """Tests for TranslatedSummary."""

    def test_default_values(self) -> None:
        """Test default values."""
        summary = TranslatedSummary()

        assert summary.title is None
        assert summary.abstract is None
        assert summary.summary is None

    def test_with_values(self) -> None:
        """Test with explicit values."""
        summary = TranslatedSummary(
            title="テストタイトル",
            abstract="テスト要約",
            summary="テストサマリー",
        )

        assert summary.title == "テストタイトル"
        assert summary.abstract == "テスト要約"
        assert summary.summary == "テストサマリー"


class TestTranslatedSummaryToDict:
    """Tests for to_dict method."""

    def test_to_dict_all_fields(self) -> None:
        """Test to_dict with all fields."""
        summary = TranslatedSummary(
            title="テストタイトル",
            abstract="テスト要約",
            summary="テストサマリー",
        )

        result = summary.to_dict()

        assert result["title"] == "テストタイトル"
        assert result["abstract"] == "テスト要約"
        assert result["summary"] == "テストサマリー"

    def test_to_dict_partial_fields(self) -> None:
        """Test to_dict with partial fields (None values excluded)."""
        summary = TranslatedSummary(title="テストタイトル")

        result = summary.to_dict()

        assert result["title"] == "テストタイトル"
        # None values are excluded from the dict
        assert "abstract" not in result
        assert "summary" not in result


class TestTranslatedSummaryFromDict:
    """Tests for from_dict method."""

    def test_from_dict_all_fields(self) -> None:
        """Test from_dict with all fields."""
        data = {
            "title": "テストタイトル",
            "abstract": "テスト要約",
            "summary": "テストサマリー",
        }

        summary = TranslatedSummary.from_dict(data)

        assert summary.title == "テストタイトル"
        assert summary.abstract == "テスト要約"
        assert summary.summary == "テストサマリー"

    def test_from_dict_missing_fields(self) -> None:
        """Test from_dict with missing fields."""
        data = {"title": "テストタイトル"}

        summary = TranslatedSummary.from_dict(data)

        assert summary.title == "テストタイトル"
        assert summary.abstract is None
        assert summary.summary is None

    def test_from_dict_empty(self) -> None:
        """Test from_dict with empty dict."""
        data: dict[str, str | None] = {}

        summary = TranslatedSummary.from_dict(data)

        assert summary.title is None
        assert summary.abstract is None
        assert summary.summary is None


class TestTranslatedSummaryRoundTrip:
    """Tests for to_dict/from_dict round trip."""

    def test_round_trip(self) -> None:
        """Test round trip serialization."""
        original = TranslatedSummary(
            title="テストタイトル",
            abstract="テスト要約",
            summary="テストサマリー",
        )

        data = original.to_dict()
        restored = TranslatedSummary.from_dict(data)

        assert restored.title == original.title
        assert restored.abstract == original.abstract
        assert restored.summary == original.summary
