# SPDX-License-Identifier: Apache-2.0
"""
PDF Translator - CLI Tool

Translates PDF documents while preserving layout. Outputs translated PDF,
and optionally Markdown files.

Usage:
    translate-pdf <input.pdf> [options]

Examples:
    translate-pdf paper.pdf                          # Basic translation
    translate-pdf paper.pdf --markdown               # With Markdown output
    translate-pdf paper.pdf -m --markdown-mode parallel  # Parallel mode
    translate-pdf paper.pdf -o ./translated.pdf
    translate-pdf paper.pdf --source en --target ja
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import NoReturn

from pdf_translator.output.markdown_writer import MarkdownOutputMode
from pdf_translator.pipeline.translation_pipeline import (
    PipelineConfig,
    TranslationPipeline,
)
from pdf_translator.translators.base import TranslatorBackend
from pdf_translator.translators.google import GoogleTranslator

# Optional backends
try:
    from pdf_translator.translators import get_deepl_translator, get_openai_translator
except ImportError:
    get_deepl_translator = None  # type: ignore[assignment]
    get_openai_translator = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Default output directory
DEFAULT_OUTPUT_DIR = "./output/"


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed argument Namespace.
    """
    parser = argparse.ArgumentParser(
        prog="translate-pdf",
        description="PDF Translation Tool - Translates PDF documents with layout preservation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s paper.pdf                              # Google Translate (default)
  %(prog)s paper.pdf --backend deepl              # DeepL (high quality)
  %(prog)s paper.pdf --backend openai             # OpenAI GPT
  %(prog)s paper.pdf -o result.pdf                # Specify output file
  %(prog)s paper.pdf -s en -t ja                  # English to Japanese
  %(prog)s paper.pdf --markdown                   # Generate Markdown
  %(prog)s paper.pdf -m --markdown-mode parallel  # Original + translation
  %(prog)s paper.pdf --save-intermediate          # Save JSON for regeneration

Environment Variables:
  DEEPL_API_KEY    DeepL API key (required for --backend deepl)
  OPENAI_API_KEY   OpenAI API key (required for --backend openai)
""",
    )

    # Input file
    parser.add_argument(
        "input",
        type=Path,
        help="Path to PDF file to translate",
    )

    # Output options
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help=f"Output file path (default: {DEFAULT_OUTPUT_DIR}<input>_translated.pdf)",
    )

    # Translation backend
    parser.add_argument(
        "-b",
        "--backend",
        default="google",
        choices=["google", "deepl", "openai"],
        help="Translation backend (default: google)",
    )

    # Language options
    parser.add_argument(
        "-s",
        "--source",
        default="en",
        help="Source language code (default: en)",
    )

    parser.add_argument(
        "-t",
        "--target",
        default="ja",
        help="Target language code (default: ja)",
    )

    # DeepL options
    deepl_group = parser.add_argument_group("DeepL options")
    deepl_group.add_argument(
        "--api-key",
        help="DeepL API key (or set DEEPL_API_KEY)",
    )
    deepl_group.add_argument(
        "--api-url",
        help="DeepL API URL (optional, for Pro users)",
    )

    # OpenAI options
    openai_group = parser.add_argument_group("OpenAI options")
    openai_group.add_argument(
        "--openai-api-key",
        help="OpenAI API key (or set OPENAI_API_KEY)",
    )
    openai_group.add_argument(
        "--openai-model",
        default="gpt-4o-mini",
        help="OpenAI model (default: gpt-4o-mini)",
    )
    openai_group.add_argument(
        "--openai-prompt",
        help="Custom system prompt for OpenAI",
    )
    openai_group.add_argument(
        "--openai-prompt-file",
        type=Path,
        help="File containing custom system prompt for OpenAI",
    )

    # Markdown options
    md_group = parser.add_argument_group("Markdown output options")
    md_group.add_argument(
        "-m",
        "--markdown",
        action="store_true",
        help="Generate Markdown output file",
    )
    md_group.add_argument(
        "--markdown-mode",
        default="translated_only",
        choices=["translated_only", "original_only", "parallel"],
        help="Markdown output mode (default: translated_only)",
    )
    md_group.add_argument(
        "--markdown-no-metadata",
        action="store_true",
        help="Disable YAML frontmatter in Markdown",
    )
    md_group.add_argument(
        "--markdown-no-page-breaks",
        action="store_true",
        help="Disable page break markers in Markdown",
    )
    md_group.add_argument(
        "--markdown-heading-offset",
        type=int,
        default=0,
        choices=[0, 1, 2, 3, 4, 5],
        help="Heading level offset (default: 0)",
    )
    md_group.add_argument(
        "--markdown-include-all",
        action="store_true",
        help="Include all categories in Markdown output (disable default skipping)",
    )
    md_group.add_argument(
        "--markdown-skip",
        type=str,
        metavar="CATEGORIES",
        help=(
            "Comma-separated categories to skip in Markdown output. "
            "Default: header,footer,page_number,aside_text,footnote,etc."
        ),
    )

    # Image extraction options (for Markdown)
    img_group = parser.add_argument_group("Image extraction options (with --markdown)")
    img_group.add_argument(
        "--no-extract-images",
        action="store_true",
        help="Disable image extraction from PDF",
    )
    img_group.add_argument(
        "--image-format",
        default="png",
        choices=["png", "jpeg"],
        help="Image output format (default: png)",
    )
    img_group.add_argument(
        "--image-quality",
        type=int,
        default=95,
        help="JPEG quality 1-100 (default: 95)",
    )
    img_group.add_argument(
        "--image-dpi",
        type=int,
        default=150,
        help="Image render resolution (default: 150)",
    )

    # Table extraction options (for Markdown)
    tbl_group = parser.add_argument_group("Table extraction options (with --markdown)")
    tbl_group.add_argument(
        "--no-extract-tables",
        action="store_true",
        help="Disable table extraction from PDF",
    )
    tbl_group.add_argument(
        "--table-mode",
        default="heuristic",
        choices=["heuristic", "pdfplumber", "image"],
        help=(
            "Table extraction mode (default: heuristic). "
            "Note: heuristic mode currently falls back to pdfplumber/image "
            "as TextObject integration is not yet implemented."
        ),
    )

    # Intermediate data options
    parser.add_argument(
        "--save-intermediate",
        action="store_true",
        help="Save intermediate JSON file for regeneration",
    )

    # Side-by-side options
    sbs_group = parser.add_argument_group("Side-by-side options")
    sbs_group.add_argument(
        "--side-by-side",
        action="store_true",
        help="Generate side-by-side comparison PDF",
    )

    # Debug options
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (draw bounding boxes)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    return parser.parse_args()


def create_translator(args: argparse.Namespace) -> TranslatorBackend:
    """Create translator based on backend selection.

    Args:
        args: Command line arguments.

    Returns:
        Translator instance.

    Raises:
        SystemExit: If required API key is missing.
    """
    if args.backend == "deepl":
        if get_deepl_translator is None:
            print(
                "Error: DeepL backend requires 'deepl' extra.\n"
                "  Install with: pip install pdf-translator[deepl]",
                file=sys.stderr,
            )
            sys.exit(1)

        api_key = args.api_key or os.environ.get("DEEPL_API_KEY", "")
        if not api_key:
            print(
                "Error: DeepL API key is required for --backend deepl.\n"
                "  Set --api-key option or DEEPL_API_KEY environment variable.\n"
                "  Or use --backend google for API-key-free translation.",
                file=sys.stderr,
            )
            sys.exit(1)

        DeepLTranslator = get_deepl_translator()
        api_url = args.api_url or os.environ.get("DEEPL_API_URL")
        translator: TranslatorBackend = DeepLTranslator(api_key=api_key, api_url=api_url)
        return translator

    elif args.backend == "openai":
        if get_openai_translator is None:
            print(
                "Error: OpenAI backend requires 'openai' extra.\n"
                "  Install with: pip install pdf-translator[openai]",
                file=sys.stderr,
            )
            sys.exit(1)

        api_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            print(
                "Error: OpenAI API key is required for --backend openai.\n"
                "  Set --openai-api-key option or OPENAI_API_KEY environment variable.\n"
                "  Or use --backend google for API-key-free translation.",
                file=sys.stderr,
            )
            sys.exit(1)

        # Load custom prompt
        system_prompt = None
        if args.openai_prompt_file:
            if not args.openai_prompt_file.exists():
                print(
                    f"Error: Prompt file not found: {args.openai_prompt_file}",
                    file=sys.stderr,
                )
                sys.exit(1)
            system_prompt = args.openai_prompt_file.read_text(encoding="utf-8")
        elif args.openai_prompt:
            system_prompt = args.openai_prompt

        OpenAITranslator = get_openai_translator()
        translator = OpenAITranslator(
            api_key=api_key,
            model=args.openai_model,
            system_prompt=system_prompt,
        )
        return translator

    else:
        # Default: Google Translate
        return GoogleTranslator()


def get_markdown_mode(mode_str: str) -> MarkdownOutputMode:
    """Convert mode string to MarkdownOutputMode enum.

    Args:
        mode_str: Mode string from CLI.

    Returns:
        MarkdownOutputMode enum value.
    """
    mode_map = {
        "translated_only": MarkdownOutputMode.TRANSLATED_ONLY,
        "original_only": MarkdownOutputMode.ORIGINAL_ONLY,
        "parallel": MarkdownOutputMode.PARALLEL,
    }
    return mode_map[mode_str]


def get_markdown_skip_categories(args: argparse.Namespace) -> frozenset[str] | None:
    """Get markdown skip categories from CLI arguments.

    Args:
        args: Command line arguments.

    Returns:
        frozenset of categories to skip, or None for default.
    """
    if args.markdown_include_all:
        # Include all categories (skip nothing)
        return frozenset()
    elif args.markdown_skip:
        # Custom skip categories
        categories = [cat.strip() for cat in args.markdown_skip.split(",") if cat.strip()]
        return frozenset(categories)
    else:
        # Use default (None will use DEFAULT_MARKDOWN_SKIP_CATEGORIES)
        return None


async def run(args: argparse.Namespace) -> int:
    """Execute translation pipeline.

    Args:
        args: Command line arguments.

    Returns:
        Exit code (0: success, 1: failure).
    """
    input_path: Path = args.input

    # Validate input file
    if not input_path.exists():
        print(f"Error: File not found: {input_path}", file=sys.stderr)
        return 1

    if input_path.suffix.lower() != ".pdf":
        print(f"Error: Not a PDF file: {input_path}", file=sys.stderr)
        return 1

    # Determine output path
    if args.output:
        output_path: Path = args.output
    else:
        output_dir = Path(DEFAULT_OUTPUT_DIR)
        output_path = output_dir / f"{input_path.stem}_translated.pdf"

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create translator
    translator = create_translator(args)

    # Create pipeline config
    config = PipelineConfig(
        source_lang=args.source,
        target_lang=args.target,
        # Markdown options
        markdown_output=args.markdown,
        markdown_mode=get_markdown_mode(args.markdown_mode),
        markdown_include_metadata=not args.markdown_no_metadata,
        markdown_include_page_breaks=not args.markdown_no_page_breaks,
        markdown_heading_offset=args.markdown_heading_offset,
        markdown_skip_categories=get_markdown_skip_categories(args),
        save_intermediate=args.save_intermediate,
        # Image extraction options
        extract_images=not args.no_extract_images,
        image_format=args.image_format,
        image_quality=args.image_quality,
        image_dpi=args.image_dpi,
        # Table extraction options
        extract_tables=not args.no_extract_tables,
        table_mode=args.table_mode,
        # Side-by-side options
        side_by_side=args.side_by_side,
        # Debug options
        debug_draw_bbox=args.debug,
    )

    # Display progress
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Backend: {args.backend}")
    if args.backend == "openai":
        print(f"Model: {args.openai_model}")
    print(f"Translation: {args.source.upper()} -> {args.target.upper()}")
    if args.markdown:
        print(f"Markdown: enabled (mode: {args.markdown_mode})")
    if args.save_intermediate:
        print("Intermediate JSON: enabled")
    if args.side_by_side:
        print("Side-by-side: enabled")
    if args.debug:
        print("Debug mode: enabled")
    print()

    # Create pipeline
    pipeline = TranslationPipeline(translator, config)

    # Execute translation
    try:
        print("Translating...")
        result = await pipeline.translate(input_path, output_path)
    except Exception as e:
        print(f"Error: Translation failed: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1

    # Display results
    print()
    print(f"Complete: {output_path}")
    if result.stats:
        stats = result.stats
        print(f"  Paragraphs: {stats.get('paragraphs', 0)}")
        print(f"  Translated: {stats.get('translated_paragraphs', 0)}")
        print(f"  Skipped: {stats.get('skipped_paragraphs', 0)}")

    if args.markdown and result.markdown:
        md_path = output_path.with_suffix(".md")
        print(f"  Markdown: {md_path}")

    if args.save_intermediate:
        json_path = output_path.with_suffix(".json")
        print(f"  JSON: {json_path}")

    if args.side_by_side and result.side_by_side_pdf_bytes:
        sbs_path = output_path.with_stem(output_path.stem + "_side_by_side")
        print(f"  Side-by-side: {sbs_path}")

    return 0


def main() -> NoReturn:
    """Main entry point."""
    args = parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    exit_code = asyncio.run(run(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
