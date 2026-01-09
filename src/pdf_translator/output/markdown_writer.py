# SPDX-License-Identifier: Apache-2.0
"""Markdown output writer for translated documents."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from pdf_translator.core.models import BBox, ListMarker, Paragraph

if TYPE_CHECKING:
    from pdf_translator.output.image_extractor import ExtractedImage
    from pdf_translator.output.table_extractor import ExtractedTable


class MarkdownOutputMode(Enum):
    """Markdown output mode."""

    TRANSLATED_ONLY = "translated_only"  # Translation only (fallback: original)
    ORIGINAL_ONLY = "original_only"  # Original only
    PARALLEL = "parallel"  # Original and translation in parallel


# Default category to element type mapping for PP-DocLayoutV2
DEFAULT_CATEGORY_MAPPING: dict[str, str] = {
    # Text categories
    "doc_title": "h1",
    "paragraph_title": "h2",
    "text": "p",
    "vertical_text": "p",
    "abstract": "p",
    "aside_text": "blockquote",
    # Figure/table categories
    "figure_title": "caption",
    "table": "table",
    "chart": "image",
    "image": "image",
    # Formula categories
    "inline_formula": "code",
    "display_formula": "code_block",
    "algorithm": "code_block",
    "formula_number": "p",
    # Reference categories
    "reference": "p",
    "reference_content": "p",
    "footnote": "p",
    "vision_footnote": "p",
    # Navigation categories
    "header": "p",
    "header_image": "image",
    "footer": "p",
    "footer_image": "image",
    "number": "p",
    # Other
    "seal": "p",
    "content": "p",
    "unknown": "p",
}

# Default for None category
DEFAULT_NONE_CATEGORY_MAPPING: str = "p"

# Default categories to skip in Markdown output
# These are typically layout elements not part of the main document content
DEFAULT_MARKDOWN_SKIP_CATEGORIES: frozenset[str] = frozenset({
    # Layout elements (PDF page structure)
    "header",
    "header_image",
    "footer",
    "footer_image",
    "page_number",
    "number",
    # Supplementary text
    "aside_text",
    "footnote",
    "vision_footnote",
    # Other
    "formula_number",
    "seal",
})

# Element types where list conversion should be skipped
# These elements have their own formatting that shouldn't be prefixed with list markers
SKIP_LIST_CONVERSION: frozenset[str] = frozenset({
    "code",
    "code_block",
})


@dataclass
class MarkdownConfig:
    """Markdown output configuration."""

    output_mode: MarkdownOutputMode = MarkdownOutputMode.TRANSLATED_ONLY
    include_metadata: bool = True
    include_page_breaks: bool = True
    heading_offset: int = 0  # 0-5
    source_lang: Optional[str] = None
    target_lang: Optional[str] = None
    source_filename: Optional[str] = None
    category_mapping_overrides: Optional[dict[str, str]] = None
    # Categories to skip in output
    # None: use DEFAULT_MARKDOWN_SKIP_CATEGORIES
    # frozenset(): include all categories (skip nothing)
    # frozenset({...}): use custom skip categories
    skip_categories: Optional[frozenset[str]] = None


class MarkdownWriter:
    """Markdown output writer."""

    def __init__(self, config: Optional[MarkdownConfig] = None) -> None:
        """Initialize MarkdownWriter.

        Args:
            config: Markdown output configuration.
        """
        self._config = config or MarkdownConfig()

    def write(
        self,
        paragraphs: list[Paragraph],
        extracted_images: list[ExtractedImage] | None = None,
        extracted_tables: list[ExtractedTable] | None = None,
    ) -> str:
        """Generate Markdown string from paragraphs.

        Args:
            paragraphs: List of translated paragraphs.
            extracted_images: List of extracted images (for future use).
            extracted_tables: List of extracted tables (for future use).

        Returns:
            Markdown string.
        """
        parts: list[str] = []

        # Generate metadata header
        if self._config.include_metadata:
            metadata = self._generate_metadata()
            if metadata:
                parts.append(metadata)

        # Build image and table maps for lookup
        image_map = self._build_image_map(extracted_images)
        table_map = self._build_table_map(extracted_tables)

        # Track emitted images/tables to avoid duplicates
        emitted_keys: set[str] = set()

        # Process paragraphs
        current_page: int | None = None
        for paragraph in paragraphs:
            # Add page break if needed
            if self._config.include_page_breaks:
                if current_page is not None and paragraph.page_number != current_page:
                    # Page changed, add break
                    display_page = paragraph.page_number + 1  # 0-indexed to 1-indexed
                    parts.append(f"\n---\n<!-- Page {display_page} -->\n")
                current_page = paragraph.page_number

            # Convert paragraph to markdown
            md = self._paragraph_to_markdown(
                paragraph, image_map, table_map, emitted_keys
            )
            if md:
                parts.append(md)

        return "\n".join(parts)

    def write_to_file(
        self,
        paragraphs: list[Paragraph],
        output_path: Path,
        extracted_images: list[ExtractedImage] | None = None,
        extracted_tables: list[ExtractedTable] | None = None,
    ) -> None:
        """Write Markdown to file.

        Args:
            paragraphs: List of translated paragraphs.
            output_path: Output file path.
            extracted_images: List of extracted images (for future use).
            extracted_tables: List of extracted tables (for future use).
        """
        markdown = self.write(paragraphs, extracted_images, extracted_tables)
        output_path.write_text(markdown, encoding="utf-8")

    def _generate_metadata(self) -> str:
        """Generate YAML frontmatter metadata.

        Returns:
            YAML frontmatter string.
        """
        lines: list[str] = ["---"]

        if self._config.source_filename:
            lines.append(f"title: {self._config.source_filename}")
        if self._config.source_lang:
            lines.append(f"source_lang: {self._config.source_lang}")
        if self._config.target_lang:
            lines.append(f"target_lang: {self._config.target_lang}")

        lines.append(f"translated_at: {datetime.now().isoformat()}")
        lines.append("---")

        return "\n".join(lines) + "\n"

    def _build_key(
        self,
        layout_block_id: str | None,
        page_number: int,
        bbox: BBox,
    ) -> str:
        """Generate key for map lookup.

        Args:
            layout_block_id: LayoutBlock ID (preferred).
            page_number: Page number (fallback).
            bbox: Bounding box (fallback).

        Returns:
            Key string.
        """
        if layout_block_id:
            return layout_block_id
        # Fallback to position-based key
        return f"p{page_number}_{bbox.x0:.1f}_{bbox.y0:.1f}_{bbox.x1:.1f}_{bbox.y1:.1f}"

    def _build_image_map(
        self,
        extracted_images: list[ExtractedImage] | None,
    ) -> dict[str, ExtractedImage]:
        """Build image map for lookup.

        Args:
            extracted_images: List of extracted images.

        Returns:
            Key -> ExtractedImage map.
        """
        if not extracted_images:
            return {}
        result: dict[str, ExtractedImage] = {}
        for img in extracted_images:
            key = self._build_key(
                img.layout_block_id,
                img.page_number,
                img.bbox,
            )
            result[key] = img
        return result

    def _build_table_map(
        self,
        extracted_tables: list[ExtractedTable] | None,
    ) -> dict[str, ExtractedTable]:
        """Build table map for lookup.

        Args:
            extracted_tables: List of extracted tables.

        Returns:
            Key -> ExtractedTable map.
        """
        if not extracted_tables:
            return {}
        result: dict[str, ExtractedTable] = {}
        for table in extracted_tables:
            key = self._build_key(
                table.layout_block_id,
                table.page_number,
                table.bbox,
            )
            result[key] = table
        return result

    def _paragraph_to_markdown(
        self,
        paragraph: Paragraph,
        image_map: dict[str, ExtractedImage] | None = None,
        table_map: dict[str, ExtractedTable] | None = None,
        emitted_keys: set[str] | None = None,
    ) -> str:
        """Convert single paragraph to Markdown.

        Args:
            paragraph: Paragraph to convert.
            image_map: Key -> ExtractedImage map.
            table_map: Key -> ExtractedTable map.
            emitted_keys: Set of already emitted image/table keys.

        Returns:
            Markdown string.
        """
        # Skip if category is in skip list
        if self._should_skip_category(paragraph.category):
            return ""

        element_type = self._get_element_type(paragraph.category)

        # Get text based on output mode
        text = self._get_display_text(paragraph)
        if not text:
            return ""

        # Generate key for lookups
        key = self._build_key(
            paragraph.layout_block_id,
            paragraph.page_number,
            paragraph.block_bbox,
        )

        # Handle special element types
        if element_type == "image":
            # Skip if already emitted
            if emitted_keys is not None and key in emitted_keys:
                return ""
            # For image category, look up in image_map
            if image_map and key in image_map:
                if emitted_keys is not None:
                    emitted_keys.add(key)
                img = image_map[key]
                alt = getattr(img, "caption", None) or "Image"
                path = getattr(img, "relative_path", "")
                return f"![{alt}]({path})\n"
            # Fallback: skip if no image found
            return ""

        if element_type == "table":
            # Skip if already emitted
            if emitted_keys is not None and key in emitted_keys:
                return ""
            # For table category, look up in table_map first
            if table_map and key in table_map:
                if emitted_keys is not None:
                    emitted_keys.add(key)
                table = table_map[key]
                if hasattr(table, "to_markdown"):
                    return str(table.to_markdown()) + "\n"
            # Fallback to image_map for table image
            if image_map and key in image_map:
                if emitted_keys is not None:
                    emitted_keys.add(key)
                img = image_map[key]
                alt = "Table"
                path = getattr(img, "relative_path", "")
                return f"![{alt}]({path})\n"
            # Final fallback: output as paragraph
            return self._format_text(text, "p")

        # Handle list items (skip for code elements)
        if paragraph.list_marker and element_type not in SKIP_LIST_CONVERSION:
            return self._format_list_item(paragraph)

        # Format text based on element type
        return self._format_text(text, element_type, paragraph)

    def _format_as_list_item(
        self,
        text: str,
        list_marker: ListMarker | None,
    ) -> str:
        """Format text as a Markdown list item if applicable.

        Args:
            text: Text content to format.
            list_marker: List marker info, or None for regular text.

        Returns:
            Formatted text with Markdown list syntax if applicable.
        """
        if not list_marker:
            return text

        if list_marker.marker_type == "bullet":
            # Bullet list: "- item"
            return f"- {text}"

        elif list_marker.marker_type == "numbered":
            # Numbered list: "N. item"
            num = list_marker.number if list_marker.number is not None else 1
            return f"{num}. {text}"

        return text

    def _format_list_item(self, paragraph: Paragraph) -> str:
        """Format a list item paragraph to Markdown.

        Handles both regular and parallel output modes.

        Args:
            paragraph: Paragraph with list_marker to format.

        Returns:
            Markdown string for the list item.
        """
        list_marker = paragraph.list_marker
        text = self._get_display_text(paragraph)

        # Apply styling
        styled_text = self._apply_style(text, paragraph)

        # Format as list item
        formatted = self._format_as_list_item(styled_text, list_marker)

        # Handle parallel mode
        if (
            self._config.output_mode == MarkdownOutputMode.PARALLEL
            and paragraph.translated_text
        ):
            original = self._format_as_list_item(
                self._apply_style(paragraph.text, paragraph),
                list_marker,
            )
            translated = self._format_as_list_item(
                self._apply_style(paragraph.translated_text, paragraph),
                list_marker,
            )
            return f"{original}\n{translated}\n"

        return formatted + "\n"

    def _get_display_text(self, paragraph: Paragraph) -> str:
        """Get text to display based on output mode.

        Args:
            paragraph: Paragraph to get text from.

        Returns:
            Text string.
        """
        if self._config.output_mode == MarkdownOutputMode.TRANSLATED_ONLY:
            # Prefer translated, fallback to original
            return paragraph.translated_text or paragraph.text
        elif self._config.output_mode == MarkdownOutputMode.ORIGINAL_ONLY:
            return paragraph.text
        else:  # PARALLEL
            # Return original text (translation added separately)
            return paragraph.text

    def _format_text(
        self,
        text: str,
        element_type: str,
        paragraph: Paragraph | None = None,
    ) -> str:
        """Format text according to element type.

        Args:
            text: Text to format.
            element_type: Element type string.
            paragraph: Original paragraph (for parallel mode).

        Returns:
            Formatted Markdown string.
        """
        # Apply style formatting
        formatted = self._apply_style(text, paragraph)

        # Format based on element type
        if element_type.startswith("h") and len(element_type) == 2:
            level = int(element_type[1])
            level = self._apply_heading_offset(level)
            result = "#" * level + " " + formatted + "\n"
        elif element_type == "p":
            result = formatted + "\n"
        elif element_type == "blockquote":
            # Handle multi-line quotes
            lines = formatted.split("\n")
            result = "\n".join("> " + line for line in lines) + "\n"
        elif element_type == "code":
            result = f"`{formatted}`\n"
        elif element_type == "code_block":
            result = f"```\n{formatted}\n```\n"
        elif element_type == "caption":
            result = f"*{formatted}*\n"
        else:
            # Default to paragraph
            result = formatted + "\n"

        # Add translation for parallel mode
        if (
            self._config.output_mode == MarkdownOutputMode.PARALLEL
            and paragraph
            and paragraph.translated_text
        ):
            translated = self._apply_style(paragraph.translated_text, paragraph)
            if element_type.startswith("h") and len(element_type) == 2:
                level = int(element_type[1])
                level = self._apply_heading_offset(level)
                result += "#" * level + " " + translated + "\n"
            elif element_type == "blockquote":
                lines = translated.split("\n")
                result += "\n".join("> " + line for line in lines) + "\n"
            elif element_type == "caption":
                result += f"*{translated}*\n"
            else:
                result += translated + "\n"

        return result

    def _apply_style(self, text: str, paragraph: Paragraph | None) -> str:
        """Apply bold/italic styling to text.

        Args:
            text: Text to style.
            paragraph: Paragraph with style info.

        Returns:
            Styled text.
        """
        if not paragraph:
            return text

        if paragraph.is_bold and paragraph.is_italic:
            return f"***{text}***"
        elif paragraph.is_bold:
            return f"**{text}**"
        elif paragraph.is_italic:
            return f"*{text}*"
        return text

    def _get_element_type(self, category: str | None) -> str:
        """Get element type from category.

        Args:
            category: Category string.

        Returns:
            Element type string.
        """
        if category is None:
            return DEFAULT_NONE_CATEGORY_MAPPING

        # Check overrides first
        if self._config.category_mapping_overrides:
            if category in self._config.category_mapping_overrides:
                return self._config.category_mapping_overrides[category]

        # Use default mapping
        return DEFAULT_CATEGORY_MAPPING.get(category, "p")

    def _should_skip_category(self, category: str | None) -> bool:
        """Check if category should be skipped.

        Args:
            category: Category string.

        Returns:
            True if category should be skipped.
        """
        if category is None:
            return False

        # Get effective skip categories
        if self._config.skip_categories is not None:
            skip_categories = self._config.skip_categories
        else:
            skip_categories = DEFAULT_MARKDOWN_SKIP_CATEGORIES

        return category in skip_categories

    def _apply_heading_offset(self, level: int) -> int:
        """Apply heading offset.

        Args:
            level: Original heading level (1-6).

        Returns:
            Adjusted heading level (1-6).
        """
        new_level = level + self._config.heading_offset
        return max(1, min(6, new_level))
