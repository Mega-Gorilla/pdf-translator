# SPDX-License-Identifier: Apache-2.0
"""Pipeline-specific error classes.

This module defines exception classes for the translation pipeline,
allowing for structured error handling with stage information.
"""

from __future__ import annotations

from typing import Optional


class PipelineError(Exception):
    """Base class for pipeline errors.

    Attributes:
        message: Error message
        stage: Pipeline stage where error occurred
        cause: Original exception that caused this error
    """

    def __init__(
        self,
        message: str,
        stage: str,
        cause: Optional[Exception] = None,
    ) -> None:
        """Initialize PipelineError.

        Args:
            message: Error message
            stage: Pipeline stage where error occurred
            cause: Original exception that caused this error
        """
        super().__init__(message)
        self.stage = stage
        self.cause = cause

    def __str__(self) -> str:
        """Return string representation with stage info."""
        base = f"[{self.stage}] {super().__str__()}"
        if self.cause:
            base += f" (caused by: {self.cause})"
        return base


class ExtractionError(PipelineError):
    """Error during PDF text extraction stage."""

    def __init__(
        self,
        message: str,
        cause: Optional[Exception] = None,
    ) -> None:
        """Initialize ExtractionError.

        Args:
            message: Error message
            cause: Original exception
        """
        super().__init__(message, stage="extract", cause=cause)


class LayoutAnalysisError(PipelineError):
    """Error during layout analysis stage."""

    def __init__(
        self,
        message: str,
        cause: Optional[Exception] = None,
    ) -> None:
        """Initialize LayoutAnalysisError.

        Args:
            message: Error message
            cause: Original exception
        """
        super().__init__(message, stage="analyze", cause=cause)


class MergeError(PipelineError):
    """Error during text merge/sorting stage."""

    def __init__(
        self,
        message: str,
        cause: Optional[Exception] = None,
    ) -> None:
        """Initialize MergeError.

        Args:
            message: Error message
            cause: Original exception
        """
        super().__init__(message, stage="merge", cause=cause)


class FontAdjustmentError(PipelineError):
    """Error during font size adjustment stage."""

    def __init__(
        self,
        message: str,
        cause: Optional[Exception] = None,
    ) -> None:
        """Initialize FontAdjustmentError.

        Args:
            message: Error message
            cause: Original exception
        """
        super().__init__(message, stage="font_adjust", cause=cause)
