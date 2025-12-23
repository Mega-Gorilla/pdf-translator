# SPDX-License-Identifier: Apache-2.0
"""Data models for PDF processing intermediate data format.

This module defines the intermediate data schema for PDF text processing,
enabling extraction, manipulation, and reinsertion of text objects.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

SCHEMA_VERSION = "1.0.0"


class RawLayoutCategory(str, Enum):
    """PP-DocLayoutV2 の生カテゴリ（モデル出力そのまま）.

    PP-DocLayoutV2 が出力する 25 種類のカテゴリをそのまま定義。
    公式 label_list: https://huggingface.co/PaddlePaddle/PP-DocLayoutV2/raw/main/config.json
    モデルバージョン変更時にはここを更新する。
    """

    # テキスト系
    TEXT = "text"
    VERTICAL_TEXT = "vertical_text"  # 縦書きテキスト
    PARAGRAPH_TITLE = "paragraph_title"
    DOC_TITLE = "doc_title"
    ABSTRACT = "abstract"
    ASIDE_TEXT = "aside_text"

    # 数式系
    INLINE_FORMULA = "inline_formula"
    DISPLAY_FORMULA = "display_formula"
    FORMULA_NUMBER = "formula_number"
    ALGORITHM = "algorithm"

    # 図表系
    TABLE = "table"
    IMAGE = "image"
    FIGURE_TITLE = "figure_title"
    CHART = "chart"

    # ナビゲーション系
    HEADER = "header"
    HEADER_IMAGE = "header_image"  # ヘッダー内画像
    FOOTER = "footer"
    FOOTER_IMAGE = "footer_image"  # フッター内画像
    NUMBER = "number"

    # 参照系
    REFERENCE = "reference"
    REFERENCE_CONTENT = "reference_content"
    FOOTNOTE = "footnote"
    VISION_FOOTNOTE = "vision_footnote"  # 視覚的脚注

    # その他
    SEAL = "seal"
    CONTENT = "content"

    # 未知（フォールバック用、公式 label_list には含まれない）
    UNKNOWN = "unknown"


class ProjectCategory(str, Enum):
    """プロジェクト内部の安定カテゴリ.

    翻訳ロジックが依存する安定したカテゴリ。
    モデルバージョンが変わっても、このカテゴリ体系は維持される。
    """

    # 翻訳対象
    TEXT = "text"  # 本文テキスト
    TITLE = "title"  # 見出し・タイトル
    CAPTION = "caption"  # キャプション（図表共通）

    # 翻訳除外
    FOOTNOTE = "footnote"  # 脚注
    FORMULA = "formula"  # 数式（inline/display 統合）
    CODE = "code"  # コード・アルゴリズム
    TABLE = "table"  # 表
    IMAGE = "image"  # 画像
    CHART = "chart"  # チャート
    HEADER = "header"  # ヘッダー・フッター
    REFERENCE = "reference"  # 参考文献

    # その他
    OTHER = "other"  # 上記以外


# Raw → Project カテゴリマッピング
RAW_TO_PROJECT_MAPPING: dict[RawLayoutCategory, ProjectCategory] = {
    # 翻訳対象
    RawLayoutCategory.TEXT: ProjectCategory.TEXT,
    RawLayoutCategory.VERTICAL_TEXT: ProjectCategory.TEXT,  # 縦書きも翻訳対象
    RawLayoutCategory.ABSTRACT: ProjectCategory.TEXT,
    RawLayoutCategory.ASIDE_TEXT: ProjectCategory.TEXT,
    RawLayoutCategory.PARAGRAPH_TITLE: ProjectCategory.TITLE,
    RawLayoutCategory.DOC_TITLE: ProjectCategory.TITLE,
    RawLayoutCategory.FIGURE_TITLE: ProjectCategory.CAPTION,
    RawLayoutCategory.FOOTNOTE: ProjectCategory.FOOTNOTE,
    RawLayoutCategory.VISION_FOOTNOTE: ProjectCategory.FOOTNOTE,  # 視覚的脚注
    # 翻訳除外
    RawLayoutCategory.INLINE_FORMULA: ProjectCategory.FORMULA,
    RawLayoutCategory.DISPLAY_FORMULA: ProjectCategory.FORMULA,
    RawLayoutCategory.FORMULA_NUMBER: ProjectCategory.FORMULA,
    RawLayoutCategory.ALGORITHM: ProjectCategory.CODE,
    RawLayoutCategory.TABLE: ProjectCategory.TABLE,
    RawLayoutCategory.IMAGE: ProjectCategory.IMAGE,
    RawLayoutCategory.CHART: ProjectCategory.CHART,
    RawLayoutCategory.HEADER: ProjectCategory.HEADER,
    RawLayoutCategory.HEADER_IMAGE: ProjectCategory.HEADER,  # ヘッダー画像
    RawLayoutCategory.FOOTER: ProjectCategory.HEADER,
    RawLayoutCategory.FOOTER_IMAGE: ProjectCategory.HEADER,  # フッター画像
    RawLayoutCategory.NUMBER: ProjectCategory.HEADER,
    RawLayoutCategory.REFERENCE: ProjectCategory.REFERENCE,
    RawLayoutCategory.REFERENCE_CONTENT: ProjectCategory.REFERENCE,
    # その他
    RawLayoutCategory.SEAL: ProjectCategory.OTHER,
    RawLayoutCategory.CONTENT: ProjectCategory.OTHER,
    RawLayoutCategory.UNKNOWN: ProjectCategory.OTHER,
}

# 翻訳対象カテゴリのセット
TRANSLATABLE_CATEGORIES: frozenset[ProjectCategory] = frozenset(
    {ProjectCategory.TEXT, ProjectCategory.TITLE, ProjectCategory.CAPTION}
)


@dataclass
class BBox:
    """Bounding box in PDF coordinate system (origin at bottom-left).

    Represents post-transform coordinates of an object.

    Attributes:
        x0: Left X coordinate
        y0: Bottom Y coordinate
        x1: Right X coordinate
        y1: Top Y coordinate
    """

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        """Width of the bounding box."""
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        """Height of the bounding box."""
        return self.y1 - self.y0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {"x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BBox:
        """Create from dictionary."""
        return cls(
            x0=float(data["x0"]),
            y0=float(data["y0"]),
            x1=float(data["x1"]),
            y1=float(data["y1"]),
        )


@dataclass
class Font:
    """Font information for text objects.

    Attributes:
        name: Font name (e.g., "Helvetica", "Times-Roman")
        size: Font size in points
        is_bold: Whether the font is bold
        is_italic: Whether the font is italic
    """

    name: str
    size: float
    is_bold: bool = False
    is_italic: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "size": self.size,
            "is_bold": self.is_bold,
            "is_italic": self.is_italic,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Font:
        """Create from dictionary."""
        return cls(
            name=data["name"],
            size=float(data["size"]),
            is_bold=data.get("is_bold", False),
            is_italic=data.get("is_italic", False),
        )


@dataclass
class Color:
    """RGB color value.

    Attributes:
        r: Red component (0-255)
        g: Green component (0-255)
        b: Blue component (0-255)
    """

    r: int = 0
    g: int = 0
    b: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {"r": self.r, "g": self.g, "b": self.b}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Color:
        """Create from dictionary."""
        return cls(
            r=int(data.get("r", 0)),
            g=int(data.get("g", 0)),
            b=int(data.get("b", 0)),
        )


@dataclass
class Transform:
    """Affine transformation matrix [a, b, c, d, e, f].

    Used for rotation, scale, and skew transformations.
    The matrix transforms coordinates as:
        x' = a*x + c*y + e
        y' = b*x + d*y + f

    bbox stores post-transform coordinates, while transform is used
    to reproduce the same transformation when reinserting text.

    Attributes:
        a: Horizontal scale
        b: Vertical skew
        c: Horizontal skew
        d: Vertical scale
        e: Horizontal translation
        f: Vertical translation
    """

    a: float = 1.0
    b: float = 0.0
    c: float = 0.0
    d: float = 1.0
    e: float = 0.0
    f: float = 0.0

    def is_identity(self) -> bool:
        """Check if this is an identity transformation."""
        return (
            abs(self.a - 1.0) < 1e-6
            and abs(self.b) < 1e-6
            and abs(self.c) < 1e-6
            and abs(self.d - 1.0) < 1e-6
            and abs(self.e) < 1e-6
            and abs(self.f) < 1e-6
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "a": self.a,
            "b": self.b,
            "c": self.c,
            "d": self.d,
            "e": self.e,
            "f": self.f,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Transform:
        """Create from dictionary."""
        return cls(
            a=float(data.get("a", 1.0)),
            b=float(data.get("b", 0.0)),
            c=float(data.get("c", 0.0)),
            d=float(data.get("d", 1.0)),
            e=float(data.get("e", 0.0)),
            f=float(data.get("f", 0.0)),
        )


@dataclass
class CharPosition:
    """Character-level position information.

    Attributes:
        char: Single character
        bbox: Bounding box of the character
    """

    char: str
    bbox: BBox

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {"char": self.char, "bbox": self.bbox.to_dict()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CharPosition:
        """Create from dictionary."""
        return cls(char=data["char"], bbox=BBox.from_dict(data["bbox"]))


@dataclass
class TextObject:
    """A text object extracted from PDF.

    Attributes:
        id: Unique identifier
        bbox: Bounding box (post-transform coordinates)
        text: Text content
        font: Font information (optional)
        color: Text color (optional)
        transform: Affine transformation matrix (optional)
        char_positions: Character-level positions (optional)
    """

    id: str
    bbox: BBox
    text: str
    font: Optional[Font] = None
    color: Optional[Color] = None
    transform: Optional[Transform] = None
    char_positions: Optional[list[CharPosition]] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result: dict[str, Any] = {
            "id": self.id,
            "bbox": self.bbox.to_dict(),
            "text": self.text,
        }
        if self.font is not None:
            result["font"] = self.font.to_dict()
        if self.color is not None:
            result["color"] = self.color.to_dict()
        if self.transform is not None:
            result["transform"] = self.transform.to_dict()
        if self.char_positions is not None:
            result["char_positions"] = [cp.to_dict() for cp in self.char_positions]
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TextObject:
        """Create from dictionary."""
        font = Font.from_dict(data["font"]) if "font" in data else None
        color = Color.from_dict(data["color"]) if "color" in data else None
        transform = (
            Transform.from_dict(data["transform"]) if "transform" in data else None
        )
        char_positions = (
            [CharPosition.from_dict(cp) for cp in data["char_positions"]]
            if "char_positions" in data
            else None
        )
        return cls(
            id=data["id"],
            bbox=BBox.from_dict(data["bbox"]),
            text=data["text"],
            font=font,
            color=color,
            transform=transform,
            char_positions=char_positions,
        )


@dataclass
class LayoutBlock:
    """Layout block from document layout analysis (PP-DocLayout).

    Attributes:
        id: Unique identifier
        bbox: Bounding box in PDF coordinates
        raw_category: PP-DocLayoutV2 の生カテゴリ（モデル出力そのまま）
        project_category: プロジェクト内部の正規化カテゴリ
        confidence: Detection confidence (0-1)
        page_num: Page number (0-indexed)
        text_object_ids: IDs of TextObjects contained in this block
    """

    id: str
    bbox: BBox
    raw_category: RawLayoutCategory
    project_category: ProjectCategory
    confidence: float = 0.0
    page_num: int = 0
    text_object_ids: list[str] = field(default_factory=list)

    @property
    def type(self) -> str:
        """Backward-compatible type property.

        Returns the raw_category value as string.
        """
        return self.raw_category.value

    @property
    def is_translatable(self) -> bool:
        """Check if this block should be translated."""
        return self.project_category in TRANSLATABLE_CATEGORIES

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "bbox": self.bbox.to_dict(),
            "raw_category": self.raw_category.value,
            "project_category": self.project_category.value,
            "confidence": self.confidence,
            "page_num": self.page_num,
            "text_object_ids": self.text_object_ids,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LayoutBlock:
        """Create from dictionary.

        Supports both new format (raw_category/project_category) and
        legacy format (type) for backward compatibility.
        """
        # Handle raw_category
        if "raw_category" in data:
            try:
                raw_category = RawLayoutCategory(data["raw_category"])
            except ValueError:
                raw_category = RawLayoutCategory.UNKNOWN
        elif "type" in data:
            # Legacy format: convert type string to RawLayoutCategory
            try:
                raw_category = RawLayoutCategory(data["type"])
            except ValueError:
                raw_category = RawLayoutCategory.UNKNOWN
        else:
            raw_category = RawLayoutCategory.UNKNOWN

        # Handle project_category
        if "project_category" in data:
            try:
                project_category = ProjectCategory(data["project_category"])
            except ValueError:
                project_category = RAW_TO_PROJECT_MAPPING.get(
                    raw_category, ProjectCategory.OTHER
                )
        else:
            # Derive from raw_category
            project_category = RAW_TO_PROJECT_MAPPING.get(
                raw_category, ProjectCategory.OTHER
            )

        return cls(
            id=data["id"],
            bbox=BBox.from_dict(data["bbox"]),
            raw_category=raw_category,
            project_category=project_category,
            confidence=float(data.get("confidence", 0.0)),
            page_num=int(data.get("page_num", 0)),
            text_object_ids=data.get("text_object_ids", []),
        )


@dataclass
class Paragraph:
    """Paragraph extracted from pdftext blocks.

    Represents a translation unit mapped to a block-level bounding box.
    """

    id: str
    page_number: int
    text: str
    block_bbox: BBox
    line_count: int
    original_font_size: float = 12.0
    category: Optional[ProjectCategory] = None
    translated_text: Optional[str] = None
    adjusted_font_size: Optional[float] = None

    @property
    def is_translatable(self) -> bool:
        """Check if this paragraph should be translated."""
        if self.category is None:
            return True
        return self.category in TRANSLATABLE_CATEGORIES


@dataclass
class Page:
    """A page in the PDF document.

    Attributes:
        page_number: Zero-indexed page number
        width: Page width in points
        height: Page height in points
        text_objects: List of text objects on the page
        rotation: Page rotation in degrees (0, 90, 180, 270)
        layout_blocks: Layout blocks from PP-DocLayout (optional)
    """

    page_number: int
    width: float
    height: float
    text_objects: list[TextObject] = field(default_factory=list)
    rotation: int = 0
    layout_blocks: Optional[list[LayoutBlock]] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result: dict[str, Any] = {
            "page_number": self.page_number,
            "width": self.width,
            "height": self.height,
            "rotation": self.rotation,
            "text_objects": [obj.to_dict() for obj in self.text_objects],
        }
        if self.layout_blocks is not None:
            result["layout_blocks"] = [lb.to_dict() for lb in self.layout_blocks]
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Page:
        """Create from dictionary."""
        layout_blocks = (
            [LayoutBlock.from_dict(lb) for lb in data["layout_blocks"]]
            if "layout_blocks" in data
            else None
        )
        return cls(
            page_number=int(data["page_number"]),
            width=float(data["width"]),
            height=float(data["height"]),
            rotation=int(data.get("rotation", 0)),
            text_objects=[TextObject.from_dict(obj) for obj in data["text_objects"]],
            layout_blocks=layout_blocks,
        )


@dataclass
class Metadata:
    """Document metadata.

    Attributes:
        source_file: Original PDF file name
        created_at: Creation timestamp
        page_count: Number of pages
        pdf_version: PDF version string (optional)
    """

    source_file: str
    created_at: str
    page_count: int = 0
    pdf_version: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result: dict[str, Any] = {
            "source_file": self.source_file,
            "created_at": self.created_at,
            "page_count": self.page_count,
        }
        if self.pdf_version is not None:
            result["pdf_version"] = self.pdf_version
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Metadata:
        """Create from dictionary."""
        return cls(
            source_file=data["source_file"],
            created_at=data["created_at"],
            page_count=int(data.get("page_count", 0)),
            pdf_version=data.get("pdf_version"),
        )


@dataclass
class PDFDocument:
    """Intermediate data representation of a PDF document.

    This class holds extracted text information for manipulation
    and reinsertion. Non-text elements (images, figures) are
    inherited from the original PDF template.

    Attributes:
        pages: List of pages with text objects
        metadata: Document metadata
    """

    pages: list[Page]
    metadata: Metadata

    def to_json(self, indent: int = 2) -> str:
        """Export to JSON string.

        Args:
            indent: JSON indentation level

        Returns:
            JSON string representation
        """
        data = {
            "version": SCHEMA_VERSION,
            "metadata": self.metadata.to_dict(),
            "pages": [page.to_dict() for page in self.pages],
        }
        return json.dumps(data, indent=indent, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> PDFDocument:
        """Create from JSON string.

        Args:
            json_str: JSON string representation

        Returns:
            PDFDocument instance

        Raises:
            ValueError: If version is unsupported
        """
        data = json.loads(json_str)
        version = data.get("version", "1.0.0")
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schema version: {version} (expected {SCHEMA_VERSION})"
            )
        return cls(
            metadata=Metadata.from_dict(data["metadata"]),
            pages=[Page.from_dict(page) for page in data["pages"]],
        )

    def get_text_object(self, object_id: str) -> Optional[TextObject]:
        """Find a text object by ID.

        Args:
            object_id: The ID to search for

        Returns:
            TextObject if found, None otherwise
        """
        for page in self.pages:
            for obj in page.text_objects:
                if obj.id == object_id:
                    return obj
        return None

    def get_all_text(self) -> str:
        """Get all text content concatenated.

        Returns:
            All text from all pages joined by newlines
        """
        texts = []
        for page in self.pages:
            for obj in page.text_objects:
                texts.append(obj.text)
        return "\n".join(texts)

    @classmethod
    def create_empty(cls, source_file: str) -> PDFDocument:
        """Create an empty document with metadata.

        Args:
            source_file: Source file name

        Returns:
            Empty PDFDocument instance
        """
        metadata = Metadata(
            source_file=source_file,
            created_at=datetime.now().isoformat(),
            page_count=0,
        )
        return cls(pages=[], metadata=metadata)
