# SPDX-License-Identifier: Apache-2.0
"""Tests for CLI argument parsing."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from pdf_translator.cli import (
    get_markdown_skip_categories,
    get_translatable_categories,
    parse_args,
)


class TestParseArgs:
    """Tests for parse_args function."""

    def test_basic_input(self) -> None:
        """Test basic input file argument."""
        with patch.object(sys, "argv", ["translate-pdf", "test.pdf"]):
            args = parse_args()
            assert args.input == Path("test.pdf")
            assert args.backend == "google"
            assert args.source == "en"
            assert args.target == "ja"

    def test_translate_all_flag(self) -> None:
        """Test --translate-all flag."""
        with patch.object(sys, "argv", ["translate-pdf", "test.pdf", "--translate-all"]):
            args = parse_args()
            assert args.translate_all is True

    def test_translate_all_default(self) -> None:
        """Test --translate-all default is False."""
        with patch.object(sys, "argv", ["translate-pdf", "test.pdf"]):
            args = parse_args()
            assert args.translate_all is False

    def test_translate_categories_option(self) -> None:
        """Test --translate-categories option."""
        with patch.object(
            sys,
            "argv",
            ["translate-pdf", "test.pdf", "--translate-categories", "text,abstract,doc_title"],
        ):
            args = parse_args()
            assert args.translate_categories == "text,abstract,doc_title"

    def test_translate_categories_default(self) -> None:
        """Test --translate-categories default is None."""
        with patch.object(sys, "argv", ["translate-pdf", "test.pdf"]):
            args = parse_args()
            assert args.translate_categories is None

    def test_markdown_include_all_flag(self) -> None:
        """Test --markdown-include-all flag."""
        with patch.object(
            sys, "argv", ["translate-pdf", "test.pdf", "-m", "--markdown-include-all"]
        ):
            args = parse_args()
            assert args.markdown_include_all is True

    def test_markdown_include_all_default(self) -> None:
        """Test --markdown-include-all default is False."""
        with patch.object(sys, "argv", ["translate-pdf", "test.pdf"]):
            args = parse_args()
            assert args.markdown_include_all is False

    def test_markdown_skip_option(self) -> None:
        """Test --markdown-skip option."""
        with patch.object(
            sys, "argv", ["translate-pdf", "test.pdf", "-m", "--markdown-skip", "header,footer"]
        ):
            args = parse_args()
            assert args.markdown_skip == "header,footer"

    def test_markdown_skip_default(self) -> None:
        """Test --markdown-skip default is None."""
        with patch.object(sys, "argv", ["translate-pdf", "test.pdf"]):
            args = parse_args()
            assert args.markdown_skip is None

    def test_markdown_mode_choices(self) -> None:
        """Test --markdown-mode accepts valid choices."""
        for mode in ["translated_only", "original_only", "parallel"]:
            with patch.object(
                sys, "argv", ["translate-pdf", "test.pdf", "-m", "--markdown-mode", mode]
            ):
                args = parse_args()
                assert args.markdown_mode == mode

    def test_markdown_heading_offset(self) -> None:
        """Test --markdown-heading-offset option."""
        with patch.object(
            sys, "argv", ["translate-pdf", "test.pdf", "-m", "--markdown-heading-offset", "2"]
        ):
            args = parse_args()
            assert args.markdown_heading_offset == 2


class TestGetTranslatableCategories:
    """Tests for get_translatable_categories helper function."""

    def test_default_returns_none(self) -> None:
        """Test default (no flags) returns None."""
        args = argparse.Namespace(translate_all=False, translate_categories=None)
        result = get_translatable_categories(args)
        assert result is None

    def test_translate_all_returns_all_categories(self) -> None:
        """Test --translate-all returns all RawLayoutCategory values."""
        from pdf_translator.core.models import RawLayoutCategory

        args = argparse.Namespace(translate_all=True, translate_categories=None)
        result = get_translatable_categories(args)

        assert result is not None
        assert isinstance(result, frozenset)
        # Should include all known categories
        expected = frozenset(cat.value for cat in RawLayoutCategory)
        assert result == expected

    def test_translate_categories_parses_comma_separated(self) -> None:
        """Test --translate-categories parses comma-separated values."""
        args = argparse.Namespace(translate_all=False, translate_categories="text,abstract,doc_title")
        result = get_translatable_categories(args)

        assert result is not None
        assert result == frozenset({"text", "abstract", "doc_title"})

    def test_translate_categories_strips_whitespace(self) -> None:
        """Test --translate-categories strips whitespace around values."""
        args = argparse.Namespace(
            translate_all=False, translate_categories=" text , abstract , doc_title "
        )
        result = get_translatable_categories(args)

        assert result is not None
        assert result == frozenset({"text", "abstract", "doc_title"})

    def test_translate_categories_ignores_empty(self) -> None:
        """Test --translate-categories ignores empty values."""
        args = argparse.Namespace(translate_all=False, translate_categories="text,,abstract,")
        result = get_translatable_categories(args)

        assert result is not None
        assert result == frozenset({"text", "abstract"})

    def test_translate_all_takes_precedence(self) -> None:
        """Test --translate-all takes precedence over --translate-categories."""
        from pdf_translator.core.models import RawLayoutCategory

        args = argparse.Namespace(translate_all=True, translate_categories="text,abstract")
        result = get_translatable_categories(args)

        # Should return all categories, not just the specified ones
        expected = frozenset(cat.value for cat in RawLayoutCategory)
        assert result == expected


class TestGetMarkdownSkipCategories:
    """Tests for get_markdown_skip_categories helper function."""

    def test_default_returns_none(self) -> None:
        """Test default (no flags) returns None."""
        args = argparse.Namespace(markdown_include_all=False, markdown_skip=None)
        result = get_markdown_skip_categories(args)
        assert result is None

    def test_markdown_include_all_returns_empty_frozenset(self) -> None:
        """Test --markdown-include-all returns empty frozenset (skip nothing)."""
        args = argparse.Namespace(markdown_include_all=True, markdown_skip=None)
        result = get_markdown_skip_categories(args)

        assert result is not None
        assert result == frozenset()

    def test_markdown_skip_parses_comma_separated(self) -> None:
        """Test --markdown-skip parses comma-separated values."""
        args = argparse.Namespace(markdown_include_all=False, markdown_skip="header,footer")
        result = get_markdown_skip_categories(args)

        assert result is not None
        assert result == frozenset({"header", "footer"})

    def test_markdown_skip_strips_whitespace(self) -> None:
        """Test --markdown-skip strips whitespace around values."""
        args = argparse.Namespace(markdown_include_all=False, markdown_skip=" header , footer ")
        result = get_markdown_skip_categories(args)

        assert result is not None
        assert result == frozenset({"header", "footer"})

    def test_markdown_skip_ignores_empty(self) -> None:
        """Test --markdown-skip ignores empty values."""
        args = argparse.Namespace(markdown_include_all=False, markdown_skip="header,,footer,")
        result = get_markdown_skip_categories(args)

        assert result is not None
        assert result == frozenset({"header", "footer"})

    def test_markdown_include_all_takes_precedence(self) -> None:
        """Test --markdown-include-all takes precedence over --markdown-skip."""
        args = argparse.Namespace(markdown_include_all=True, markdown_skip="header,footer")
        result = get_markdown_skip_categories(args)

        # Should return empty frozenset (include all), not the skip list
        assert result == frozenset()
