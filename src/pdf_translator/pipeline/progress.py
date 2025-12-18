# SPDX-License-Identifier: Apache-2.0
"""Progress reporting for the translation pipeline.

This module defines the ProgressCallback protocol for reporting
pipeline progress to external consumers (CLI, GUI, etc.).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ProgressCallback(Protocol):
    """Protocol for progress reporting callbacks.

    Implementations can be used to update progress bars, log messages,
    or send updates to external systems.

    Stage names and their total values:
        - extract: PDF text extraction (total=1)
        - analyze: Layout analysis (total=page_count)
        - merge: Reading order sort (total=page_count)
        - translate: Batch translation (total=translatable_object_count)
        - font_adjust: Font size adjustment (total=translatable_object_count)
        - apply: PDF apply (total=1)

    Example:
        >>> def my_progress(stage: str, current: int, total: int, message: str = "") -> None:
        ...     print(f"{stage}: {current}/{total} {message}")
        >>> pipeline = TranslationPipeline(translator, progress_callback=my_progress)
    """

    def __call__(
        self,
        stage: str,
        current: int,
        total: int,
        message: str = "",
    ) -> None:
        """Report progress.

        Args:
            stage: Stage name (extract, analyze, merge, translate, font_adjust, apply)
            current: Current progress count
            total: Total count for this stage
            message: Optional status message
        """
        ...
