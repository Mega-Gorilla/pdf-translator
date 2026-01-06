# SPDX-License-Identifier: Apache-2.0
"""Tests for pipeline text splitting functionality."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from pdf_translator.pipeline.translation_pipeline import (
    PipelineConfig,
    TranslationPipeline,
)
from pdf_translator.translators import GoogleTranslator


class MockTranslator:
    """Mock translator for testing."""

    def __init__(self, max_text_length: int | None = 100) -> None:
        self._max_text_length = max_text_length
        self.name = "mock"

    @property
    def max_text_length(self) -> int | None:
        return self._max_text_length

    async def translate(
        self, text: str, source_lang: str, target_lang: str
    ) -> str:
        return f"[translated]{text}"

    async def translate_batch(
        self, texts: list[str], source_lang: str, target_lang: str
    ) -> list[str]:
        return [f"[translated]{t}" for t in texts]


class MockOpenAITranslator:
    """Mock OpenAI translator with token-based limits for testing."""

    def __init__(
        self, max_tokens: int = 100_000, max_batch_tokens: int = 200_000
    ) -> None:
        """Initialize mock translator.

        Args:
            max_tokens: Token limit per text. Default 100K matches gpt-5-nano.
            max_batch_tokens: Token limit per batch. Default 200K (400K/2).
        """
        self._max_tokens = max_tokens
        self._max_batch_tokens = max_batch_tokens
        self.name = "openai"
        # Same ratio as real OpenAITranslator (conservative for CJK)
        self.CHARS_PER_TOKEN = 1.0

    @property
    def max_text_length(self) -> int | None:
        if self._max_tokens == 0:
            return None
        return int(self._max_tokens * self.CHARS_PER_TOKEN)

    @property
    def max_batch_tokens(self) -> int:
        return self._max_batch_tokens

    def count_tokens(self, text: str) -> int:
        """Count tokens (1 char = 1 token for simplicity)."""
        return len(text)

    async def translate(
        self, text: str, source_lang: str, target_lang: str
    ) -> str:
        return f"[translated]{text}"

    async def translate_batch(
        self, texts: list[str], source_lang: str, target_lang: str
    ) -> list[str]:
        return [f"[translated]{t}" for t in texts]


class TestSplitLongText:
    """Test _split_long_text method."""

    def test_short_text_no_split(self) -> None:
        """Text shorter than max_length should not be split."""
        pipeline = TranslationPipeline(MockTranslator(max_text_length=100))
        result = pipeline._split_long_text("Short text", 100)
        assert result == ["Short text"]

    def test_exact_max_length_no_split(self) -> None:
        """Text exactly at max_length should not be split."""
        pipeline = TranslationPipeline(MockTranslator(max_text_length=100))
        text = "a" * 100
        result = pipeline._split_long_text(text, 100)
        assert result == [text]

    def test_split_at_sentence_boundary(self) -> None:
        """Long text should split at sentence boundaries."""
        pipeline = TranslationPipeline(MockTranslator(max_text_length=50))
        text = "First sentence. Second sentence. Third sentence."
        result = pipeline._split_long_text(text, 30)
        assert len(result) >= 2
        # Check that splits happen at sentence boundaries
        for part in result:
            assert len(part) <= 30 or "." in part[:30]

    def test_split_at_word_boundary(self) -> None:
        """When no sentence boundary, split at word boundary."""
        pipeline = TranslationPipeline(MockTranslator(max_text_length=50))
        text = "This is a long text without any sentence boundary that needs splitting"
        result = pipeline._split_long_text(text, 30)
        assert len(result) >= 2
        # Verify no part exceeds max_length (except if forced)
        for part in result:
            # Parts should be at or near max_length
            assert len(part) <= 30 or " " not in part[:30]

    def test_split_japanese_sentence_boundary(self) -> None:
        """Japanese text should split at 。."""
        pipeline = TranslationPipeline(MockTranslator(max_text_length=50))
        text = "これは最初の文です。これは二番目の文です。これは三番目の文です。"
        result = pipeline._split_long_text(text, 25)
        assert len(result) >= 2

    def test_forced_split_no_boundary(self) -> None:
        """Text with no boundaries should force split at max_length."""
        pipeline = TranslationPipeline(MockTranslator(max_text_length=50))
        text = "a" * 100  # No spaces or sentence boundaries
        result = pipeline._split_long_text(text, 30)
        assert len(result) >= 3
        # First parts should be exactly max_length
        assert len(result[0]) == 30

    def test_whitespace_preservation(self) -> None:
        """Whitespace should be preserved during split."""
        pipeline = TranslationPipeline(MockTranslator(max_text_length=50))
        # Text with multiple spaces and newlines
        text = "First sentence.  Second sentence.\n\nThird sentence."
        result = pipeline._split_long_text(text, 20)
        # Joining parts should reconstruct original text
        assert "".join(result) == text

    def test_trailing_space_in_word_split(self) -> None:
        """Trailing space should be preserved in word boundary split."""
        pipeline = TranslationPipeline(MockTranslator(max_text_length=50))
        text = "Hello world this is a test"
        result = pipeline._split_long_text(text, 12)
        # Each part (except possibly last) should end with space
        # and joining should reconstruct original
        assert "".join(result) == text


class TestSplitTextsForApi:
    """Test _split_texts_for_api method."""

    def test_no_split_when_unlimited(self) -> None:
        """No splitting when max_text_length is None."""
        pipeline = TranslationPipeline(MockTranslator(max_text_length=None))
        texts = ["short", "a" * 10000]  # Very long text
        result_texts, mapping = pipeline._split_texts_for_api(texts)
        assert result_texts == texts
        assert mapping == [(0, 1), (1, 1)]

    def test_no_split_when_short(self) -> None:
        """No splitting when all texts are short."""
        pipeline = TranslationPipeline(MockTranslator(max_text_length=100))
        texts = ["short text", "another short"]
        result_texts, mapping = pipeline._split_texts_for_api(texts)
        assert result_texts == texts
        assert mapping == [(0, 1), (1, 1)]

    def test_split_long_text(self) -> None:
        """Long text should be split into multiple parts."""
        pipeline = TranslationPipeline(MockTranslator(max_text_length=50))
        texts = [
            "short",
            "This is a long text. That needs to be split. Into multiple parts.",
        ]
        result_texts, mapping = pipeline._split_texts_for_api(texts)
        assert len(result_texts) > 2
        assert mapping[0] == (0, 1)  # First text not split
        assert mapping[1][0] == 1  # Second text index
        assert mapping[1][1] > 1  # Second text was split

    def test_empty_list(self) -> None:
        """Empty list should return empty results."""
        pipeline = TranslationPipeline(MockTranslator(max_text_length=100))
        result_texts, mapping = pipeline._split_texts_for_api([])
        assert result_texts == []
        assert mapping == []


class TestRejoinTranslatedTexts:
    """Test _rejoin_translated_texts method."""

    def test_no_split_passthrough(self) -> None:
        """Non-split texts should pass through."""
        config = PipelineConfig(target_lang="en")
        pipeline = TranslationPipeline(MockTranslator(max_text_length=100), config)
        translated = ["text1", "text2", "text3"]
        mapping = [(0, 1), (1, 1), (2, 1)]
        result = pipeline._rejoin_translated_texts(translated, mapping)
        assert result == ["text1", "text2", "text3"]

    def test_rejoin_split_texts_english(self) -> None:
        """Split texts should be rejoined with space for English."""
        config = PipelineConfig(target_lang="en")
        pipeline = TranslationPipeline(MockTranslator(max_text_length=100), config)
        translated = ["part1", "part2", "other"]
        mapping = [(0, 2), (1, 1)]  # First text was split into 2 parts
        result = pipeline._rejoin_translated_texts(translated, mapping)
        assert result == ["part1 part2", "other"]

    def test_rejoin_split_texts_japanese(self) -> None:
        """Split texts should be rejoined without space for Japanese."""
        config = PipelineConfig(target_lang="ja")
        pipeline = TranslationPipeline(MockTranslator(max_text_length=100), config)
        translated = ["これは", "テストです", "他"]
        mapping = [(0, 2), (1, 1)]
        result = pipeline._rejoin_translated_texts(translated, mapping)
        assert result == ["これはテストです", "他"]

    def test_rejoin_split_texts_chinese(self) -> None:
        """Split texts should be rejoined without space for Chinese."""
        config = PipelineConfig(target_lang="zh")
        pipeline = TranslationPipeline(MockTranslator(max_text_length=100), config)
        translated = ["这是", "测试", "其他"]
        mapping = [(0, 2), (1, 1)]
        result = pipeline._rejoin_translated_texts(translated, mapping)
        assert result == ["这是测试", "其他"]

    def test_rejoin_split_texts_korean(self) -> None:
        """Split texts should be rejoined with space for Korean.

        Korean uses spaces between words (unlike Japanese/Chinese).
        """
        config = PipelineConfig(target_lang="ko")
        pipeline = TranslationPipeline(MockTranslator(max_text_length=100), config)
        translated = ["이것은", "테스트입니다", "기타"]
        mapping = [(0, 2), (1, 1)]
        result = pipeline._rejoin_translated_texts(translated, mapping)
        assert result == ["이것은 테스트입니다", "기타"]

    def test_multiple_splits(self) -> None:
        """Multiple texts split should all be rejoined correctly."""
        config = PipelineConfig(target_lang="en")
        pipeline = TranslationPipeline(MockTranslator(max_text_length=100), config)
        translated = ["a1", "a2", "b", "c1", "c2", "c3"]
        mapping = [(0, 2), (1, 1), (2, 3)]
        result = pipeline._rejoin_translated_texts(translated, mapping)
        assert result == ["a1 a2", "b", "c1 c2 c3"]


class TestIntegration:
    """Integration tests for text splitting in pipeline."""

    @pytest.mark.asyncio
    async def test_split_and_rejoin_in_translation_english(self) -> None:
        """Test that split texts are rejoined with space for English."""
        mock_translator = MagicMock()
        mock_translator.name = "mock"
        mock_translator.max_text_length = 50

        config = PipelineConfig(target_lang="en")
        pipeline = TranslationPipeline(mock_translator, config)

        texts = ["short", "a" * 100]  # Second text exceeds 50 chars
        split_texts, mapping = pipeline._split_texts_for_api(texts)

        # Verify splitting occurred
        assert len(split_texts) > 2
        assert mapping[0] == (0, 1)
        assert mapping[1][1] > 1

        # Simulate translation of split texts
        translated_split = [f"[EN]{t}" for t in split_texts]
        rejoined = pipeline._rejoin_translated_texts(translated_split, mapping)

        # Verify rejoining with space for English
        assert len(rejoined) == 2
        assert rejoined[0] == "[EN]short"
        assert " " in rejoined[1]  # Split parts are rejoined with space

    @pytest.mark.asyncio
    async def test_split_and_rejoin_in_translation_japanese(self) -> None:
        """Test that split texts are rejoined without space for Japanese."""
        mock_translator = MagicMock()
        mock_translator.name = "mock"
        mock_translator.max_text_length = 50

        config = PipelineConfig(target_lang="ja")
        pipeline = TranslationPipeline(mock_translator, config)

        texts = ["短い", "あ" * 100]  # Second text exceeds 50 chars
        split_texts, mapping = pipeline._split_texts_for_api(texts)

        # Verify splitting occurred
        assert len(split_texts) > 2
        assert mapping[0] == (0, 1)
        assert mapping[1][1] > 1

        # Simulate translation of split texts
        translated_split = [f"[JA]{t}" for t in split_texts]
        rejoined = pipeline._rejoin_translated_texts(translated_split, mapping)

        # Verify rejoining without space for Japanese
        assert len(rejoined) == 2
        assert rejoined[0] == "[JA]短い"
        assert " " not in rejoined[1]  # No space for CJK languages


class TestOpenAITokenBasedSplitting:
    """Test OpenAI translator with token-based limits."""

    def test_openai_max_text_length_default(self) -> None:
        """OpenAI mock should have default max_text_length matching gpt-5-nano."""
        translator = MockOpenAITranslator()
        # Default: 100K tokens × 1.0 chars/token = 100K characters (gpt-5-nano)
        assert translator.max_text_length == 100_000

    def test_openai_max_text_length_custom(self) -> None:
        """OpenAI mock should accept custom max_tokens."""
        translator = MockOpenAITranslator(max_tokens=32_000)
        # gpt-4o equivalent: 32K tokens
        assert translator.max_text_length == 32_000

    def test_openai_max_text_length_disabled(self) -> None:
        """OpenAI mock with max_tokens=0 should return None."""
        translator = MockOpenAITranslator(max_tokens=0)
        assert translator.max_text_length is None

    def test_openai_splitting_with_pipeline(self) -> None:
        """Pipeline should split texts based on OpenAI's token-derived limit."""
        # Use small token limit for testing (25 tokens × 1.0 = 25 chars)
        translator = MockOpenAITranslator(max_tokens=25)
        config = PipelineConfig(target_lang="en")
        pipeline = TranslationPipeline(translator, config)

        # Text exceeding 25 chars should be split
        texts = ["short", "a" * 100]
        split_texts, mapping = pipeline._split_texts_for_api(texts)

        assert len(split_texts) > 2
        assert mapping[0] == (0, 1)  # First text not split
        assert mapping[1][1] > 1  # Second text was split

    def test_openai_no_splitting_when_disabled(self) -> None:
        """Pipeline should not split when max_tokens=0 (unlimited)."""
        translator = MockOpenAITranslator(max_tokens=0)
        config = PipelineConfig(target_lang="en")
        pipeline = TranslationPipeline(translator, config)

        # Even very long text should not be split
        texts = ["short", "a" * 100000]
        split_texts, mapping = pipeline._split_texts_for_api(texts)

        assert split_texts == texts
        assert mapping == [(0, 1), (1, 1)]

    def test_openai_chunk_by_batch_tokens(self) -> None:
        """Pipeline should chunk by max_batch_tokens for OpenAI."""
        # Small batch token limit for testing
        translator = MockOpenAITranslator(max_tokens=100, max_batch_tokens=50)
        config = PipelineConfig(target_lang="en", translation_batch_size=10)
        pipeline = TranslationPipeline(translator, config)

        # 5 texts of 20 chars each = 100 tokens total
        # With max_batch_tokens=50, should be split into chunks
        texts = ["a" * 20 for _ in range(5)]
        chunks = pipeline._chunk_texts(texts)

        # Each chunk should have <= 50 tokens (2 texts of 20 each = 40 tokens)
        assert len(chunks) >= 2
        for chunk in chunks:
            total_tokens = sum(len(t) for t in chunk)
            assert total_tokens <= 50

    def test_openai_chunk_respects_both_limits(self) -> None:
        """Pipeline should respect both batch_size and max_batch_tokens."""
        # Large token limit but small batch size
        translator = MockOpenAITranslator(max_tokens=100, max_batch_tokens=1000)
        config = PipelineConfig(target_lang="en", translation_batch_size=2)
        pipeline = TranslationPipeline(translator, config)

        # 6 short texts - should be chunked by batch_size (2), not tokens
        texts = ["hi"] * 6
        chunks = pipeline._chunk_texts(texts)

        assert len(chunks) == 3
        assert all(len(chunk) == 2 for chunk in chunks)
