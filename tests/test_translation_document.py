# SPDX-License-Identifier: Apache-2.0
"""Tests for TranslationDocument module."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from pdf_translator.output.translated_summary import TranslatedSummary
from pdf_translator.output.translation_document import (
    SCHEMA_VERSION,
    TranslationDocument,
)


class TestTranslationDocument:
    """Test TranslationDocument class."""

    def test_to_json(self) -> None:
        """Test JSON serialization."""
        doc = TranslationDocument.from_pipeline_result(
            paragraphs={"para_1": "翻訳1", "para_2": "翻訳2"},
            target_lang="ja",
            base_file="paper.json",
            translator_backend="google",
        )
        json_str = doc.to_json()
        data = json.loads(json_str)

        assert data["schema_version"] == SCHEMA_VERSION
        assert data["target_lang"] == "ja"
        assert data["base_file"] == "paper.json"
        assert data["translator_backend"] == "google"
        assert data["translated_count"] == 2
        assert data["paragraphs"]["para_1"] == "翻訳1"
        assert data["paragraphs"]["para_2"] == "翻訳2"

    def test_from_json(self) -> None:
        """Test JSON deserialization."""
        doc = TranslationDocument.from_pipeline_result(
            paragraphs={"para_1": "翻訳1", "para_2": "翻訳2"},
            target_lang="ja",
            base_file="paper.json",
            translator_backend="deepl",
        )
        json_str = doc.to_json()
        restored = TranslationDocument.from_json(json_str)

        assert restored.target_lang == "ja"
        assert restored.base_file == "paper.json"
        assert restored.translator_backend == "deepl"
        assert restored.translated_count == 2
        assert restored.paragraphs["para_1"] == "翻訳1"

    def test_from_json_version_mismatch(self) -> None:
        """Test error on version mismatch."""
        bad_json = json.dumps(
            {
                "schema_version": "1.0.0",
                "target_lang": "ja",
                "base_file": "paper.json",
                "translated_at": "2026-01-15T12:00:00",
                "translator_backend": "google",
                "translated_count": 0,
                "paragraphs": {},
            }
        )
        with pytest.raises(ValueError, match="Unsupported schema version"):
            TranslationDocument.from_json(bad_json)

    def test_save_and_load(self) -> None:
        """Test file save and load."""
        doc = TranslationDocument.from_pipeline_result(
            paragraphs={"para_1": "翻訳1"},
            target_lang="ja",
            base_file="paper.json",
            translator_backend="google",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "paper.ja.json"
            doc.save(path)

            assert path.exists()

            loaded = TranslationDocument.load(path)
            assert loaded.target_lang == "ja"
            assert loaded.translated_count == 1

    def test_with_summary(self) -> None:
        """Test with TranslatedSummary."""
        summary = TranslatedSummary(
            title="テストタイトル",
            abstract="テスト要約",
            summary="テストサマリー",
        )
        doc = TranslationDocument.from_pipeline_result(
            paragraphs={"para_1": "翻訳1"},
            target_lang="ja",
            base_file="paper.json",
            translator_backend="google",
            summary=summary,
        )

        json_str = doc.to_json()
        data = json.loads(json_str)

        assert "summary" in data
        assert data["summary"]["title"] == "テストタイトル"
        assert data["summary"]["abstract"] == "テスト要約"
        assert data["summary"]["summary"] == "テストサマリー"

        # Roundtrip test
        restored = TranslationDocument.from_json(json_str)
        assert restored.summary is not None
        assert restored.summary.title == "テストタイトル"

    def test_translated_count_matches_paragraphs(self) -> None:
        """Test that translated_count equals paragraph count."""
        doc = TranslationDocument.from_pipeline_result(
            paragraphs={"p1": "t1", "p2": "t2", "p3": "t3"},
            target_lang="ja",
            base_file="paper.json",
            translator_backend="google",
        )

        assert doc.translated_count == 3
        assert doc.translated_count == len(doc.paragraphs)


class TestTranslationDocumentRoundTrip:
    """Tests for complete round trip."""

    def test_roundtrip_with_all_fields(self) -> None:
        """Test complete roundtrip with all fields."""
        summary = TranslatedSummary(
            title="テストタイトル",
            abstract="テスト要約",
            summary="テストサマリー",
        )
        paragraphs = {
            "para_p0_b1": "最初の翻訳",
            "para_p0_b2": "二番目の翻訳",
            "para_p1_b0": "次のページの翻訳",
        }
        doc = TranslationDocument.from_pipeline_result(
            paragraphs=paragraphs,
            target_lang="ja",
            base_file="sample.json",
            translator_backend="openai",
            summary=summary,
        )

        json_str = doc.to_json()
        restored = TranslationDocument.from_json(json_str)

        assert restored.target_lang == "ja"
        assert restored.base_file == "sample.json"
        assert restored.translator_backend == "openai"
        assert restored.translated_count == 3
        assert restored.paragraphs == paragraphs
        assert restored.summary is not None
        assert restored.summary.title == "テストタイトル"
