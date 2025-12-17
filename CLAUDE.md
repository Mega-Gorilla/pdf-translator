# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PDF Translator is a PDF translation tool that preserves document layout while translating content. It outputs both translated PDFs and Markdown files.

**Languages**: Python 3.11+
**License**: Apache-2.0

## Technology Stack

- **pypdfium2**: PDF processing (Apache-2.0 compatible)
- **PP-DocLayout / DocLayout-YOLO**: Document layout analysis
- **spaCy**: Natural language processing
- **deep-translator**: Google Translate backend

## Commands

### Installation
```bash
uv sync
```

### Run Translation
```bash
uv run translate-pdf paper.pdf
```

### Development
```bash
# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_pdf_processor.py

# Run a single test function
uv run pytest tests/test_pdf_processor.py::test_extract_text_objects -v

# Type checking
uv run mypy src/

# Linting
uv run ruff check src/

# Auto-fix lint issues
uv run ruff check src/ --fix
```

## Architecture

### Data Flow
```
PDF → PDFProcessor.extract_text_objects() → PDFDocument (intermediate JSON)
    → Translation → PDFProcessor.apply() → Modified PDF
```

### Key Components

**core/models.py** - Intermediate data format (PDFDocument schema v1.0.0):
- `PDFDocument` → `Page` → `TextObject` (with BBox, Font, Color, Transform)
- `LayoutBlock` for PP-DocLayout integration
- JSON serialization for pipeline data exchange

**core/pdf_processor.py** - PDF manipulation via pypdfium2:
- Template PDF approach: preserve non-text elements, modify text layer only
- Stable object IDs (`text_p{page}_i{index}`) for remove/insert workflow
- Control character normalization for PDF-specific encoding issues

**translators/** - Async translation backends (Protocol-based):
- `TranslatorBackend` protocol: `translate()` and `translate_batch()`
- Implementations: Google (default), DeepL, OpenAI
- Error types: `TranslationError` (retryable), `ConfigurationError` (not retryable)

### Directory Structure
```
src/pdf_translator/
├── core/           # PDF processing (pypdfium2), data models
├── nlp/            # Text processing (spaCy) - planned
├── translators/    # Translation backends (google, deepl, openai)
└── resources/      # Fonts, data files
```

## Code Style

- **License**: Add `# SPDX-License-Identifier: Apache-2.0` to source files
- **Type hints**: Required for all functions (mypy strict mode)
- **Docstrings**: Google style
- **Line length**: 100 characters (ruff)

## Testing

- Tests use `pytest-asyncio` with `asyncio_mode = "auto"`
- DeepL/OpenAI tests require API keys and are skipped via dependency guards when unavailable
- pypdfium2 finalizer warnings are suppressed in pytest config (tracked in Issue #10)

## Reference

The `_archive/` folder contains the previous implementation (Index_PDF_Translation) for reference. This code is gitignored and not part of the new implementation.
