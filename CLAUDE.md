# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

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
# Run tests
uv run pytest

# Type checking
uv run mypy src/

# Linting
uv run ruff check src/
```

## Architecture

```
pdf-translator/
├── src/pdf_translator/
│   ├── core/           # PDF processing (pypdfium2)
│   ├── nlp/            # Text processing (spaCy)
│   ├── translators/    # Translation backends
│   └── resources/      # Fonts, data files
├── tests/
└── _archive/           # Reference only (gitignored)
```

## Code Style

- **License**: Add `# SPDX-License-Identifier: Apache-2.0` to source files
- **Type hints**: Required for all functions
- **Docstrings**: Google style

## Reference

The `_archive/` folder contains the previous implementation (Index_PDF_Translation) for reference. This code is gitignored and not part of the new implementation.
