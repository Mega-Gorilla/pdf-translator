# SPDX-License-Identifier: Apache-2.0
"""Tests for LLM summary generator module."""

import pytest
from unittest.mock import AsyncMock, patch

from pdf_translator.llm.client import LLMConfig
from pdf_translator.llm.summary_generator import LLMSummaryGenerator


class TestLLMSummaryGeneratorInit:
    """Tests for LLMSummaryGenerator initialization."""

    def test_init_disabled(self) -> None:
        """Test initialization with both features disabled."""
        config = LLMConfig(use_summary=False, use_fallback=False)
        generator = LLMSummaryGenerator(config)

        assert generator._client is None

    def test_init_with_summary_enabled(self) -> None:
        """Test initialization with summary enabled."""
        config = LLMConfig(use_summary=True, use_fallback=False)
        # Client will be created but may fail if litellm not installed
        generator = LLMSummaryGenerator(config)
        # Just verify initialization doesn't crash


class TestLLMSummaryGeneratorPrompts:
    """Tests for prompt constants."""

    def test_summary_prompt_format(self) -> None:
        """Test summary prompt contains required placeholders."""
        assert "{content}" in LLMSummaryGenerator.SUMMARY_PROMPT

    def test_metadata_prompt_format(self) -> None:
        """Test metadata prompt contains required placeholders."""
        assert "{content}" in LLMSummaryGenerator.METADATA_PROMPT


class TestLLMSummaryGeneratorMocked:
    """Tests with mocked LLM client."""

    async def test_generate_summary_disabled(self) -> None:
        """Test generate_summary when disabled."""
        config = LLMConfig(use_summary=False)
        generator = LLMSummaryGenerator(config)

        result = await generator.generate_summary("Test content")

        assert result is None

    async def test_extract_metadata_fallback_disabled(self) -> None:
        """Test extract_metadata_fallback when disabled."""
        config = LLMConfig(use_fallback=False)
        generator = LLMSummaryGenerator(config)

        result = await generator.extract_metadata_fallback("Test content")

        assert result == {"title": None, "abstract": None, "organization": None}

    @patch("pdf_translator.llm.summary_generator.LLMClient")
    async def test_generate_summary_with_mock(self, MockClient: AsyncMock) -> None:
        """Test generate_summary with mocked client."""
        # Set up mock
        mock_instance = AsyncMock()
        mock_instance.generate = AsyncMock(return_value="Generated summary")
        MockClient.return_value = mock_instance

        config = LLMConfig(use_summary=True)
        generator = LLMSummaryGenerator(config)
        generator._client = mock_instance

        result = await generator.generate_summary("Test markdown content")

        assert result == "Generated summary"
        mock_instance.generate.assert_called_once()

    @patch("pdf_translator.llm.summary_generator.LLMClient")
    async def test_generate_summary_strips_images(self, MockClient: AsyncMock) -> None:
        """Test that image references are removed from content."""
        mock_instance = AsyncMock()
        mock_instance.generate = AsyncMock(return_value="Summary")
        MockClient.return_value = mock_instance

        config = LLMConfig(use_summary=True)
        generator = LLMSummaryGenerator(config)
        generator._client = mock_instance

        content = "Text before ![alt](image.png) text after"
        await generator.generate_summary(content)

        # Check that the prompt doesn't contain image reference
        call_args = mock_instance.generate.call_args[0][0]
        assert "![alt](image.png)" not in call_args
        assert "Text before" in call_args
        assert "text after" in call_args

    @patch("pdf_translator.llm.summary_generator.LLMClient")
    async def test_extract_metadata_json_response(self, MockClient: AsyncMock) -> None:
        """Test metadata extraction with JSON response."""
        mock_instance = AsyncMock()
        mock_instance.generate = AsyncMock(
            return_value='{"title": "Test Title", "abstract": "Test abstract", "organization": "Test Org"}'
        )
        MockClient.return_value = mock_instance

        config = LLMConfig(use_fallback=True)
        generator = LLMSummaryGenerator(config)
        generator._client = mock_instance

        result = await generator.extract_metadata_fallback("First page content")

        assert result["title"] == "Test Title"
        assert result["abstract"] == "Test abstract"
        assert result["organization"] == "Test Org"

    @patch("pdf_translator.llm.summary_generator.LLMClient")
    async def test_extract_metadata_json_code_block(self, MockClient: AsyncMock) -> None:
        """Test metadata extraction with JSON in code block."""
        mock_instance = AsyncMock()
        mock_instance.generate = AsyncMock(
            return_value='```json\n{"title": "Test", "abstract": null, "organization": "Org"}\n```'
        )
        MockClient.return_value = mock_instance

        config = LLMConfig(use_fallback=True)
        generator = LLMSummaryGenerator(config)
        generator._client = mock_instance

        result = await generator.extract_metadata_fallback("First page content")

        assert result["title"] == "Test"
        assert result["abstract"] is None
        assert result["organization"] == "Org"

    @patch("pdf_translator.llm.summary_generator.LLMClient")
    async def test_extract_metadata_error_handling(self, MockClient: AsyncMock) -> None:
        """Test metadata extraction error handling."""
        mock_instance = AsyncMock()
        mock_instance.generate = AsyncMock(side_effect=Exception("API error"))
        MockClient.return_value = mock_instance

        config = LLMConfig(use_fallback=True)
        generator = LLMSummaryGenerator(config)
        generator._client = mock_instance

        result = await generator.extract_metadata_fallback("First page content")

        # Should return default values on error
        assert result == {"title": None, "abstract": None, "organization": None}

    @patch("pdf_translator.llm.summary_generator.LLMClient")
    async def test_generate_summary_truncation(self, MockClient: AsyncMock) -> None:
        """Test content truncation for very long content."""
        mock_instance = AsyncMock()
        mock_instance.generate = AsyncMock(return_value="Summary")
        MockClient.return_value = mock_instance

        config = LLMConfig(use_summary=True)
        generator = LLMSummaryGenerator(config)
        generator._client = mock_instance

        # Create content longer than 500K chars
        long_content = "x" * 600_000
        await generator.generate_summary(long_content)

        # Verify the content was truncated
        call_args = mock_instance.generate.call_args[0][0]
        # The prompt template adds some text, but the content itself should be truncated
        assert len(call_args) < 550_000
