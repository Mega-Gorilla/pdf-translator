# SPDX-License-Identifier: Apache-2.0
"""LLM integration module for PDF Translator.

This module provides LLM-based functionality using LiteLLM:
- Summary generation from document content
- Metadata extraction fallback when layout analysis fails

Requires optional dependency: litellm>=1.80.0
"""

from pdf_translator.llm.client import LLMClient, LLMConfig
from pdf_translator.llm.summary_generator import LLMSummaryGenerator

__all__ = [
    "LLMConfig",
    "LLMClient",
    "LLMSummaryGenerator",
]


def _has_litellm() -> bool:
    """Check if litellm is available."""
    try:
        import litellm  # noqa: F401

        return True
    except ImportError:
        return False
