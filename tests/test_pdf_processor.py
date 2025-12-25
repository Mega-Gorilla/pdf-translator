# SPDX-License-Identifier: Apache-2.0
"""Tests for PDF processor module."""

import json
import tempfile
from pathlib import Path

import pytest

from pdf_translator.core import (
    BBox,
    Color,
    Font,
    Metadata,
    Page,
    Paragraph,
    PDFDocument,
    PDFProcessor,
    TextObject,
    Transform,
)

# Test fixtures path
FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_PDF = FIXTURES_DIR / "sample_llama.pdf"


class TestModels:
    """Tests for data models."""

    def test_bbox_properties(self):
        """Test BBox width and height properties."""
        bbox = BBox(x0=10.0, y0=20.0, x1=100.0, y1=50.0)
        assert bbox.width == 90.0
        assert bbox.height == 30.0

    def test_bbox_serialization(self):
        """Test BBox to_dict and from_dict."""
        bbox = BBox(x0=10.0, y0=20.0, x1=100.0, y1=50.0)
        data = bbox.to_dict()
        restored = BBox.from_dict(data)
        assert restored.x0 == bbox.x0
        assert restored.y0 == bbox.y0
        assert restored.x1 == bbox.x1
        assert restored.y1 == bbox.y1

    def test_font_serialization(self):
        """Test Font to_dict and from_dict."""
        font = Font(name="Helvetica", size=12.0, is_bold=True, is_italic=False)
        data = font.to_dict()
        restored = Font.from_dict(data)
        assert restored.name == font.name
        assert restored.size == font.size
        assert restored.is_bold == font.is_bold
        assert restored.is_italic == font.is_italic

    def test_transform_identity(self):
        """Test Transform identity detection."""
        identity = Transform()
        assert identity.is_identity()

        non_identity = Transform(a=2.0)
        assert not non_identity.is_identity()

    def test_transform_serialization(self):
        """Test Transform to_dict and from_dict."""
        transform = Transform(a=1.0, b=0.5, c=-0.5, d=1.0, e=10.0, f=20.0)
        data = transform.to_dict()
        restored = Transform.from_dict(data)
        assert restored.a == transform.a
        assert restored.b == transform.b
        assert restored.c == transform.c
        assert restored.d == transform.d
        assert restored.e == transform.e
        assert restored.f == transform.f

    def test_text_object_serialization(self):
        """Test TextObject to_dict and from_dict."""
        text_obj = TextObject(
            id="test_001",
            bbox=BBox(72.0, 750.0, 300.0, 770.0),
            text="Sample text",
            font=Font(name="Times-Roman", size=14.0),
            color=Color(r=0, g=0, b=0),
        )
        data = text_obj.to_dict()
        restored = TextObject.from_dict(data)
        assert restored.id == text_obj.id
        assert restored.text == text_obj.text
        assert restored.bbox.x0 == text_obj.bbox.x0
        assert restored.font.name == text_obj.font.name

    def test_page_serialization(self):
        """Test Page to_dict and from_dict."""
        page = Page(
            page_number=0,
            width=595.0,
            height=842.0,
            text_objects=[
                TextObject(
                    id="text_001",
                    bbox=BBox(72.0, 750.0, 300.0, 770.0),
                    text="Sample",
                )
            ],
        )
        data = page.to_dict()
        restored = Page.from_dict(data)
        assert restored.page_number == page.page_number
        assert restored.width == page.width
        assert restored.height == page.height
        assert len(restored.text_objects) == 1
        assert restored.text_objects[0].text == "Sample"


class TestPDFDocument:
    """Tests for PDFDocument class."""

    def test_to_json_and_from_json(self):
        """Test JSON serialization roundtrip."""
        doc = PDFDocument(
            pages=[
                Page(
                    page_number=0,
                    width=595.0,
                    height=842.0,
                    text_objects=[
                        TextObject(
                            id="text_001",
                            bbox=BBox(72.0, 750.0, 300.0, 770.0),
                            text="Test Document",
                            font=Font(name="Helvetica", size=18.0, is_bold=True),
                        )
                    ],
                )
            ],
            metadata=Metadata(
                source_file="test.pdf",
                created_at="2025-01-15T10:00:00",
                page_count=1,
            ),
        )

        json_str = doc.to_json()
        restored = PDFDocument.from_json(json_str)

        assert restored.metadata.source_file == "test.pdf"
        assert restored.metadata.page_count == 1
        assert len(restored.pages) == 1
        assert restored.pages[0].text_objects[0].text == "Test Document"

    def test_get_text_object(self):
        """Test finding text object by ID."""
        doc = PDFDocument(
            pages=[
                Page(
                    page_number=0,
                    width=595.0,
                    height=842.0,
                    text_objects=[
                        TextObject(
                            id="text_001",
                            bbox=BBox(72.0, 750.0, 300.0, 770.0),
                            text="First",
                        ),
                        TextObject(
                            id="text_002",
                            bbox=BBox(72.0, 700.0, 300.0, 720.0),
                            text="Second",
                        ),
                    ],
                )
            ],
            metadata=Metadata(
                source_file="test.pdf",
                created_at="2025-01-15T10:00:00",
            ),
        )

        found = doc.get_text_object("text_002")
        assert found is not None
        assert found.text == "Second"

        not_found = doc.get_text_object("nonexistent")
        assert not_found is None

    def test_get_all_text(self):
        """Test getting all text content."""
        doc = PDFDocument(
            pages=[
                Page(
                    page_number=0,
                    width=595.0,
                    height=842.0,
                    text_objects=[
                        TextObject(
                            id="text_001",
                            bbox=BBox(72.0, 750.0, 300.0, 770.0),
                            text="Line 1",
                        ),
                        TextObject(
                            id="text_002",
                            bbox=BBox(72.0, 700.0, 300.0, 720.0),
                            text="Line 2",
                        ),
                    ],
                )
            ],
            metadata=Metadata(
                source_file="test.pdf",
                created_at="2025-01-15T10:00:00",
            ),
        )

        all_text = doc.get_all_text()
        assert "Line 1" in all_text
        assert "Line 2" in all_text

    def test_create_empty(self):
        """Test creating empty document."""
        doc = PDFDocument.create_empty("new.pdf")
        assert doc.metadata.source_file == "new.pdf"
        assert len(doc.pages) == 0

    def test_version_mismatch_error(self):
        """Test error on version mismatch."""
        json_str = '{"version": "2.0.0", "metadata": {}, "pages": []}'
        with pytest.raises(ValueError, match="Unsupported schema version"):
            PDFDocument.from_json(json_str)


@pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="Test fixture not available")
class TestPDFProcessor:
    """Tests for PDFProcessor class."""

    def test_open_pdf(self):
        """Test opening a PDF file."""
        with PDFProcessor(SAMPLE_PDF) as processor:
            assert processor.page_count > 0

    def test_file_not_found(self):
        """Test error on missing file."""
        with pytest.raises(FileNotFoundError):
            PDFProcessor("nonexistent.pdf")

    def test_invalid_source_type(self):
        """Test error on invalid pdf_source type."""
        with pytest.raises(TypeError, match="pdf_source must be Path, str, or bytes"):
            PDFProcessor(123)  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="pdf_source must be Path, str, or bytes"):
            PDFProcessor(["test.pdf"])  # type: ignore[arg-type]

    def test_extract_text_objects(self):
        """Test extracting text objects from PDF."""
        with PDFProcessor(SAMPLE_PDF) as processor:
            doc = processor.extract_text_objects()

            assert doc.metadata.source_file == "sample_llama.pdf"
            assert len(doc.pages) == processor.page_count
            assert len(doc.pages[0].text_objects) > 0

            # Check that text objects have required fields
            for text_obj in doc.pages[0].text_objects:
                assert text_obj.id is not None
                assert text_obj.text is not None
                assert text_obj.bbox is not None

    def test_extract_preserves_font_info(self):
        """Test that font information is extracted."""
        with PDFProcessor(SAMPLE_PDF) as processor:
            doc = processor.extract_text_objects()

            # At least some text objects should have font info
            fonts_found = [
                obj.font
                for obj in doc.pages[0].text_objects
                if obj.font is not None
            ]
            assert len(fonts_found) > 0

    def test_to_json(self):
        """Test JSON export."""
        with PDFProcessor(SAMPLE_PDF) as processor:
            json_str = processor.to_json()
            data = json.loads(json_str)

            assert "version" in data
            assert "metadata" in data
            assert "pages" in data

    def test_remove_all_text(self):
        """Test removing all text from a page."""
        with PDFProcessor(SAMPLE_PDF) as processor:
            # Get initial count
            doc_before = processor.extract_text_objects()
            initial_count = len(doc_before.pages[0].text_objects)

            # Remove all text
            removed = processor.remove_all_text(0)
            assert removed == initial_count

            # Verify removal
            doc_after = processor.extract_text_objects()
            assert len(doc_after.pages[0].text_objects) == 0

    def test_save_pdf(self):
        """Test saving PDF to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.pdf"

            with PDFProcessor(SAMPLE_PDF) as processor:
                processor.save(output_path)

            assert output_path.exists()
            assert output_path.stat().st_size > 0

    def test_insert_text_standard_font(self):
        """Test inserting text with standard font."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.pdf"

            with PDFProcessor(SAMPLE_PDF) as processor:
                # Remove existing text first
                processor.remove_all_text(0)

                # Insert new text
                obj_id = processor.insert_text_object(
                    page_num=0,
                    text="Inserted Text",
                    bbox=BBox(72.0, 750.0, 300.0, 770.0),
                    font=Font(name="Helvetica", size=14.0),
                )

                assert obj_id is not None
                processor.save(output_path)

            # Verify the saved PDF
            with PDFProcessor(output_path) as verify:
                doc = verify.extract_text_objects()
                texts = [obj.text for obj in doc.pages[0].text_objects]
                assert any("Inserted" in t for t in texts)

    def test_context_manager(self):
        """Test context manager protocol."""
        processor = PDFProcessor(SAMPLE_PDF)
        with processor:
            assert processor.page_count > 0
        # After exit, operations should fail or be cleaned up

    def test_stable_ids(self):
        """Test that object IDs are stable across extractions."""
        with PDFProcessor(SAMPLE_PDF) as processor:
            doc1 = processor.extract_text_objects()
            doc2 = processor.extract_text_objects()

            # IDs should be the same
            ids1 = [obj.id for obj in doc1.pages[0].text_objects]
            ids2 = [obj.id for obj in doc2.pages[0].text_objects]
            assert ids1 == ids2

            # IDs should follow the pattern text_p{page}_i{index}
            for obj in doc1.pages[0].text_objects[:5]:
                assert obj.id.startswith("text_p0_i")

    def test_remove_specific_objects_by_id(self):
        """Test removing specific text objects by their stable IDs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.pdf"

            with PDFProcessor(SAMPLE_PDF) as processor:
                doc = processor.extract_text_objects()
                initial_count = len(doc.pages[0].text_objects)

                # Get IDs of first 3 objects
                ids_to_remove = [obj.id for obj in doc.pages[0].text_objects[:3]]

                # Remove specific objects
                removed = processor.remove_text_objects(0, ids_to_remove)
                assert removed == 3

                # Verify removal
                doc_after = processor.extract_text_objects()
                assert len(doc_after.pages[0].text_objects) == initial_count - 3

                processor.save(output_path)


class TestTextLayerEdit:
    """Tests for text layer editing (template PDF approach)."""

    @pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="Test fixture not available")
    def test_text_layer_edit_workflow(self):
        """Test the complete text layer edit workflow.

        This is the main success criteria test:
        1. Extract text objects
        2. Modify text content
        3. Apply changes using template PDF approach
        4. Verify modifications
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "edited.pdf"

            # Step 1: Extract
            with PDFProcessor(SAMPLE_PDF) as processor:
                doc = processor.extract_text_objects()

                # Step 2: Modify (mark text as translated)
                for text_obj in doc.pages[0].text_objects[:3]:
                    text_obj.text = f"[TR] {text_obj.text}"

                # Step 3: Apply using template approach
                processor.apply(doc)
                processor.save(output_path)

            # Step 4: Verify
            with PDFProcessor(output_path) as verify:
                verify_doc = verify.extract_text_objects()
                modified_texts = [
                    obj.text for obj in verify_doc.pages[0].text_objects
                ]

                # Check that at least some texts were modified
                tr_count = sum(1 for t in modified_texts if "[TR]" in t)
                assert tr_count > 0, "No modified texts found"

    @pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="Test fixture not available")
    def test_json_roundtrip(self):
        """Test JSON export and import roundtrip."""
        with PDFProcessor(SAMPLE_PDF) as processor:
            # Export to JSON
            json_str = processor.to_json()

            # Parse back
            doc = PDFDocument.from_json(json_str)

            # Re-export
            json_str2 = doc.to_json()

            # Compare structures
            data1 = json.loads(json_str)
            data2 = json.loads(json_str2)

            assert data1["version"] == data2["version"]
            assert len(data1["pages"]) == len(data2["pages"])

            # Text content should match
            for p1, p2 in zip(data1["pages"], data2["pages"]):
                texts1 = {obj["text"] for obj in p1["text_objects"]}
                texts2 = {obj["text"] for obj in p2["text_objects"]}
                assert texts1 == texts2


class TestPDFProcessorParagraphs:
    """Tests for paragraph-based PDF operations."""

    def test_remove_text_in_bbox(self):
        """Ensure text objects are removed within a bounding box."""
        with PDFProcessor(SAMPLE_PDF) as processor:
            doc = processor.extract_text_objects()
            page = doc.pages[0]
            bbox = BBox(0, 0, page.width, page.height)
            removed = processor.remove_text_in_bbox(0, bbox, containment_threshold=0.5)
            assert removed > 0

    def test_apply_paragraphs_and_to_bytes(self):
        """Apply paragraphs and ensure PDF bytes are returned."""
        with PDFProcessor(SAMPLE_PDF) as processor:
            doc = processor.extract_text_objects()
            page = doc.pages[0]
            first_obj = page.text_objects[0]
            paragraph = Paragraph(
                id="para_p0_b0",
                page_number=0,
                text=first_obj.text,
                block_bbox=first_obj.bbox,
                line_count=1,
                original_font_size=first_obj.font.size if first_obj.font else 12.0,
                translated_text="Translated",
            )
            processor.apply_paragraphs([paragraph])
            pdf_bytes = processor.to_bytes()
            assert isinstance(pdf_bytes, bytes)
            assert len(pdf_bytes) > 0

    def test_find_font_variant(self, tmp_path):
        """Test font variant detection for Bold/Italic styles."""
        # Create mock font files
        base_font = tmp_path / "TestFont-Regular.ttf"
        bold_font = tmp_path / "TestFont-Bold.ttf"
        italic_font = tmp_path / "TestFont-Italic.ttf"
        bold_italic_font = tmp_path / "TestFont-BoldItalic.ttf"

        # Create empty files
        base_font.write_bytes(b"")
        bold_font.write_bytes(b"")
        italic_font.write_bytes(b"")
        bold_italic_font.write_bytes(b"")

        with PDFProcessor(SAMPLE_PDF) as processor:
            # Regular - should return base font
            result = processor._find_font_variant(base_font, False, False)
            assert result == base_font

            # Bold - should find bold variant
            result = processor._find_font_variant(base_font, True, False)
            assert result == bold_font

            # Italic - should find italic variant
            result = processor._find_font_variant(base_font, False, True)
            assert result == italic_font

            # BoldItalic - should find bold italic variant
            result = processor._find_font_variant(base_font, True, True)
            assert result == bold_italic_font

    def test_find_font_variant_fallback(self, tmp_path):
        """Test font variant falls back to base when variant not found."""
        # Create only base font
        base_font = tmp_path / "TestFont-Regular.ttf"
        base_font.write_bytes(b"")

        with PDFProcessor(SAMPLE_PDF) as processor:
            # Should fall back to base font when bold not found
            result = processor._find_font_variant(base_font, True, False)
            assert result == base_font

            # Should fall back when italic not found
            result = processor._find_font_variant(base_font, False, True)
            assert result == base_font

    def test_cover_bbox_with_rect(self):
        """Test covering a bbox area with a filled rectangle."""
        with PDFProcessor(SAMPLE_PDF) as processor:
            # Cover a small area on page 0
            bbox = BBox(x0=100.0, y0=100.0, x1=200.0, y1=200.0)
            result = processor.cover_bbox_with_rect(0, bbox)
            assert result is True

            # Verify PDF can be saved and is valid
            pdf_bytes = processor.to_bytes()
            assert isinstance(pdf_bytes, bytes)
            assert len(pdf_bytes) > 0

    def test_cover_bbox_with_rect_custom_color(self):
        """Test covering bbox with custom color."""
        with PDFProcessor(SAMPLE_PDF) as processor:
            bbox = BBox(x0=100.0, y0=100.0, x1=200.0, y1=200.0)
            custom_color = Color(r=128, g=128, b=128)
            result = processor.cover_bbox_with_rect(0, bbox, color=custom_color)
            assert result is True

    def test_cover_bbox_with_rect_invalid_page(self):
        """Test error on invalid page number."""
        with PDFProcessor(SAMPLE_PDF) as processor:
            bbox = BBox(x0=100.0, y0=100.0, x1=200.0, y1=200.0)
            with pytest.raises(IndexError):
                processor.cover_bbox_with_rect(999, bbox)


@pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="Test fixture not available")
class TestSpacingPreservation:
    """Tests for Issue #47: spacing preservation during translation.

    Verifies that cover_bbox_with_rect() preserves spacing in nearby text,
    unlike remove_text_in_bbox() which can corrupt spacing via gen_content().
    """

    def test_cover_bbox_preserves_nearby_text_spacing(self):
        """Verify that covering a bbox does not corrupt spacing in nearby text.

        This test covers the fix for Issue #47 where remove_text_in_bbox()
        caused "1 Introduction" to become "1Introduction" by corrupting
        TJ operator spacing data during gen_content().
        """
        from pdf_translator.core.paragraph_extractor import ParagraphExtractor

        # Find "1 Introduction" in original PDF
        original_paragraphs = ParagraphExtractor.extract_from_pdf(SAMPLE_PDF)
        intro_text = None
        for para in original_paragraphs:
            if "Introduction" in para.text and para.text.startswith("1"):
                intro_text = para.text
                break

        assert intro_text is not None, "Could not find '1 Introduction' in sample PDF"
        assert " " in intro_text, f"Original text should have space: {intro_text!r}"

        # Apply overlay approach (should preserve spacing)
        with PDFProcessor(SAMPLE_PDF) as processor:
            # Cover a nearby bbox (the one that caused issues in investigation)
            nearby_bbox = BBox(x0=70.5, y0=106.3, x1=291.0, y1=431.1)
            processor.cover_bbox_with_rect(0, nearby_bbox)
            pdf_bytes = processor.to_bytes()

        # Check that spacing is preserved in output
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(pdf_bytes)
            temp_path = Path(f.name)

        try:
            output_paragraphs = ParagraphExtractor.extract_from_pdf(temp_path)
            output_intro = None
            for para in output_paragraphs:
                if "Introduction" in para.text:
                    output_intro = para.text
                    break

            assert output_intro is not None, "Could not find 'Introduction' in output PDF"
            assert " " in output_intro, (
                f"Spacing was corrupted! Expected space in '{output_intro!r}'. "
                "This indicates the overlay approach failed to preserve spacing."
            )
        finally:
            temp_path.unlink()
