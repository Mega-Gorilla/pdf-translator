# SPDX-License-Identifier: Apache-2.0
"""Tests for LLM client module."""

import pytest

from pdf_translator.llm.client import LLMConfig


class TestLLMConfig:
    """Tests for LLMConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = LLMConfig()

        assert config.provider == "gemini"
        assert config.model is None
        assert config.api_key is None
        assert config.use_summary is False
        assert config.use_fallback is True

    def test_effective_model_gemini(self) -> None:
        """Test effective_model for Gemini provider."""
        config = LLMConfig(provider="gemini")
        assert config.effective_model == "gemini-3.0-flash"

    def test_effective_model_openai(self) -> None:
        """Test effective_model for OpenAI provider."""
        config = LLMConfig(provider="openai")
        assert config.effective_model == "gpt-5-mini"

    def test_effective_model_anthropic(self) -> None:
        """Test effective_model for Anthropic provider."""
        config = LLMConfig(provider="anthropic")
        assert config.effective_model == "claude-sonnet-4-5"

    def test_effective_model_custom(self) -> None:
        """Test effective_model with custom model."""
        config = LLMConfig(provider="gemini", model="gemini-1.5-pro")
        assert config.effective_model == "gemini-1.5-pro"

    def test_effective_model_unknown_provider(self) -> None:
        """Test effective_model with unknown provider."""
        config = LLMConfig(provider="unknown")
        assert config.effective_model == "gemini-3.0-flash"  # Fallback

    def test_litellm_model_default(self) -> None:
        """Test litellm_model string."""
        config = LLMConfig(provider="gemini")
        assert config.litellm_model == "gemini/gemini-3.0-flash"

    def test_litellm_model_custom(self) -> None:
        """Test litellm_model with custom model."""
        config = LLMConfig(provider="openai", model="gpt-4o")
        assert config.litellm_model == "openai/gpt-4o"

    def test_get_api_key_env_var_gemini(self) -> None:
        """Test API key environment variable for Gemini."""
        config = LLMConfig(provider="gemini")
        assert config.get_api_key_env_var() == "GEMINI_API_KEY"

    def test_get_api_key_env_var_openai(self) -> None:
        """Test API key environment variable for OpenAI."""
        config = LLMConfig(provider="openai")
        assert config.get_api_key_env_var() == "OPENAI_API_KEY"

    def test_get_api_key_env_var_anthropic(self) -> None:
        """Test API key environment variable for Anthropic."""
        config = LLMConfig(provider="anthropic")
        assert config.get_api_key_env_var() == "ANTHROPIC_API_KEY"

    def test_get_api_key_env_var_unknown(self) -> None:
        """Test API key environment variable for unknown provider."""
        config = LLMConfig(provider="custom")
        assert config.get_api_key_env_var() == "CUSTOM_API_KEY"


# LLMClient tests require litellm, which is optional
def _has_litellm() -> bool:
    """Check if litellm is available."""
    try:
        import litellm  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _has_litellm(), reason="litellm not installed")
class TestLLMClient:
    """Tests for LLMClient (requires litellm)."""

    def test_client_initialization(self) -> None:
        """Test LLMClient initialization."""
        from pdf_translator.llm.client import LLMClient

        config = LLMConfig(provider="gemini", api_key="test-key")
        client = LLMClient(config)

        assert client._config == config
