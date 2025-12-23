# SPDX-License-Identifier: Apache-2.0
"""Pipeline error definitions."""

from __future__ import annotations


class PipelineError(Exception):
    """Base exception for pipeline errors."""

    def __init__(
        self,
        message: str,
        stage: str,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.cause = cause


class ExtractionError(PipelineError):
    """Text extraction error."""


class LayoutAnalysisError(PipelineError):
    """Layout analysis error."""
