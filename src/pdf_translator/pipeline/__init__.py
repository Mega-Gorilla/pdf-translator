# SPDX-License-Identifier: Apache-2.0
"""Translation pipeline module.

This module provides the main translation pipeline for PDF documents.

Public API:
    - TranslationPipeline: Main pipeline class
    - PipelineConfig: Configuration dataclass
    - TranslationResult: Result dataclass
    - ProgressCallback: Progress reporting protocol
    - PipelineError: Base error class
    - ExtractionError: Text extraction error
    - LayoutAnalysisError: Layout analysis error
    - MergeError: Text merge error
    - FontAdjustmentError: Font adjustment error
"""

from .errors import (
    ExtractionError,
    FontAdjustmentError,
    LayoutAnalysisError,
    MergeError,
    PipelineError,
)
from .progress import ProgressCallback
from .translation_pipeline import (
    PipelineConfig,
    TranslationPipeline,
    TranslationResult,
)

__all__ = [
    # Main classes
    "TranslationPipeline",
    "PipelineConfig",
    "TranslationResult",
    # Progress
    "ProgressCallback",
    # Errors
    "PipelineError",
    "ExtractionError",
    "LayoutAnalysisError",
    "MergeError",
    "FontAdjustmentError",
]
