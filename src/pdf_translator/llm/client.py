# SPDX-License-Identifier: Apache-2.0
"""LLM client using LiteLLM for unified provider access."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import ClassVar

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for LLM integration via LiteLLM.

    Attributes:
        provider: LLM provider ("gemini", "openai", "anthropic", etc.).
        model: Model name within provider. If None, uses PROVIDER_DEFAULTS.
        api_key: API key (optional, can use environment variables).
        use_summary: Enable LLM summary generation.
        use_fallback: Enable LLM fallback for metadata extraction.
    """

    provider: str = "gemini"
    model: str | None = None  # None = use PROVIDER_DEFAULTS[provider]
    api_key: str | None = None
    use_summary: bool = False
    use_fallback: bool = False  # Opt-in: requires explicit --llm-fallback flag

    # Supported providers and their default models
    PROVIDER_DEFAULTS: ClassVar[dict[str, str]] = {
        "gemini": "gemini-3.0-flash",
        "openai": "gpt-5-mini",
        "anthropic": "claude-sonnet-4-5",
    }

    # Environment variable names for API keys
    API_KEY_ENV_VARS: ClassVar[dict[str, str]] = {
        "gemini": "GEMINI_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }

    @property
    def effective_model(self) -> str:
        """Get effective model name (resolves None to provider default)."""
        if self.model is not None:
            return self.model
        return self.PROVIDER_DEFAULTS.get(self.provider, "gemini-3.0-flash")

    @property
    def litellm_model(self) -> str:
        """Get LiteLLM model string (provider/model format)."""
        return f"{self.provider}/{self.effective_model}"

    def get_api_key_env_var(self) -> str:
        """Get environment variable name for API key."""
        return self.API_KEY_ENV_VARS.get(
            self.provider, f"{self.provider.upper()}_API_KEY"
        )


class LLMClient:
    """Unified LLM client using LiteLLM.

    Provides a simple interface for text generation across multiple providers.
    Supports: Gemini, OpenAI, Anthropic, AWS Bedrock, Azure, and 100+ others.
    """

    def __init__(self, config: LLMConfig) -> None:
        """Initialize LLMClient.

        Args:
            config: LLM configuration.

        Raises:
            ImportError: If litellm is not installed.
        """
        self._config = config
        self._setup_api_key()

    def _setup_api_key(self) -> None:
        """Set up API key in environment if provided."""
        if self._config.api_key:
            env_var = self._config.get_api_key_env_var()
            os.environ[env_var] = self._config.api_key

    async def generate(
        self,
        prompt: str,
        system: str | None = None,
    ) -> str:
        """Generate text from prompt using LiteLLM.

        Args:
            prompt: User prompt.
            system: Optional system prompt.

        Returns:
            Generated text.

        Raises:
            Exception: On LLM API errors.
        """
        from litellm import acompletion

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = await acompletion(
            model=self._config.litellm_model,
            messages=messages,
        )

        content: str = response.choices[0].message.content
        return content
