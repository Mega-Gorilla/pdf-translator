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
- **litellm**: Unified LLM client (Gemini, OpenAI, Anthropic)
- **python-dotenv**: Environment variable loading from .env files

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
PDF → ParagraphExtractor → Paragraph[] → TranslationPipeline
    → Translation → PDF/Markdown/JSON output
```

### Key Components

**core/models.py** - Data models:
- `Paragraph` - Translated paragraph with layout info
- `BBox`, `Font`, `Color` - PDF object properties
- `LayoutBlock` for PP-DocLayout integration

**core/pdf_processor.py** - PDF manipulation via pypdfium2:
- Template PDF approach: preserve non-text elements, modify text layer only
- Stable object IDs (`text_p{page}_i{index}`) for remove/insert workflow
- Control character normalization for PDF-specific encoding issues

**pipeline/translation_pipeline.py** - Main orchestration:
- `TranslationPipeline` - Coordinates extraction, translation, and output
- `PipelineConfig` - Configuration for all pipeline options
- `TranslationResult` - Result with PDF bytes, Markdown, and JSON documents

**translators/** - Async translation backends (Protocol-based):
- `TranslatorBackend` protocol: `translate()` and `translate_batch()`
- Implementations: Google (default), DeepL, OpenAI
- Error types: `TranslationError` (retryable), `ConfigurationError` (not retryable)

**output/** - Output generation:
- `BaseSummary` / `TranslatedSummary` - Document summary dataclasses
- `BaseDocument` / `TranslationDocument` - JSON schema v2.0.0 dataclasses
- `SummaryExtractor` - Extract title, abstract from layout
- `MarkdownWriter` - Generate Markdown output
- `ThumbnailGenerator` - Generate PDF thumbnails

**llm/** - LLM integration:
- `LLMClient` - Unified client via litellm (Gemini, OpenAI, Anthropic)
- `LLMSummaryGenerator` - Generate document summaries

### Directory Structure
```
src/pdf_translator/
├── core/           # PDF processing (pypdfium2), data models
├── pipeline/       # Translation pipeline orchestration
├── translators/    # Translation backends (google, deepl, openai)
├── output/         # Output generation (Markdown, JSON, thumbnail)
├── llm/            # LLM integration (summary generation)
├── nlp/            # Text processing (spaCy)
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
