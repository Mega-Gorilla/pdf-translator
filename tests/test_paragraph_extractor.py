# SPDX-License-Identifier: Apache-2.0
"""Tests for ParagraphExtractor."""

from pdf_translator.core.paragraph_extractor import (
    ExtractorConfig,
    ParagraphExtractor,
)


def test_extract_merges_lines_and_converts_coords():
    extractor = ParagraphExtractor()
    pdftext_result = [
        {
            "bbox": [0, 0, 600, 800],
            "blocks": [
                {
                    "bbox": [10, 20, 100, 40],
                    "lines": [
                        {"spans": [{"text": "Hello", "font": {"size": 12}}]},
                        {"spans": [{"text": "world", "font": {"size": 12}}]},
                    ],
                }
            ],
        }
    ]

    paragraphs = extractor.extract(pdftext_result)

    assert len(paragraphs) == 1
    paragraph = paragraphs[0]
    assert paragraph.text == "Hello world"
    assert paragraph.block_bbox.x0 == 10
    assert paragraph.block_bbox.x1 == 100
    assert paragraph.block_bbox.y0 == 760
    assert paragraph.block_bbox.y1 == 780
    assert paragraph.line_count == 2
    assert paragraph.original_font_size == 12.0


def test_extract_page_range_filters():
    extractor = ParagraphExtractor()
    pdftext_result = [
        {"bbox": [0, 0, 600, 800], "blocks": []},
        {
            "bbox": [0, 0, 600, 800],
            "blocks": [
                {
                    "bbox": [0, 0, 100, 20],
                    "lines": [
                        {"spans": [{"text": "Page2", "font": {"size": 10}}]}
                    ],
                }
            ],
        },
    ]

    paragraphs = extractor.extract(pdftext_result, page_range=[1])
    assert len(paragraphs) == 1
    assert paragraphs[0].page_number == 1
    assert paragraphs[0].text == "Page2"


def test_font_size_weighted_mode():
    extractor = ParagraphExtractor()
    pdftext_result = [
        {
            "bbox": [0, 0, 600, 800],
            "blocks": [
                {
                    "bbox": [0, 0, 100, 20],
                    "lines": [
                        {
                            "spans": [
                                {"text": "AA", "font": {"size": 10}},
                                {"text": "BBBBB", "font": {"size": 12}},
                            ]
                        }
                    ],
                }
            ],
        }
    ]

    paragraphs = extractor.extract(pdftext_result)
    assert paragraphs[0].original_font_size == 12.0


def test_empty_block_skipped():
    extractor = ParagraphExtractor()
    pdftext_result = [
        {
            "bbox": [0, 0, 600, 800],
            "blocks": [
                {"bbox": [0, 0, 100, 20], "lines": []},
            ],
        }
    ]

    paragraphs = extractor.extract(pdftext_result)
    assert paragraphs == []


def test_is_bold_from_weight():
    """Test bold detection using font weight >= 700."""
    extractor = ParagraphExtractor()
    pdftext_result = [
        {
            "bbox": [0, 0, 600, 800],
            "blocks": [
                {
                    "bbox": [0, 0, 100, 20],
                    "lines": [
                        {"spans": [{"text": "Bold text", "font": {"size": 12, "weight": 700}}]}
                    ],
                }
            ],
        }
    ]

    paragraphs = extractor.extract(pdftext_result)
    assert paragraphs[0].is_bold is True


def test_is_bold_from_font_name():
    """Test bold detection from font name containing 'bold'."""
    extractor = ParagraphExtractor()
    pdftext_result = [
        {
            "bbox": [0, 0, 600, 800],
            "blocks": [
                {
                    "bbox": [0, 0, 100, 20],
                    "lines": [
                        {
                            "spans": [
                                {
                                    "text": "Bold text",
                                    "font": {"size": 12, "weight": 400, "name": "Helvetica-Bold"},
                                }
                            ]
                        }
                    ],
                }
            ],
        }
    ]

    paragraphs = extractor.extract(pdftext_result)
    assert paragraphs[0].is_bold is True


def test_is_italic_from_flags():
    """Test italic detection using font flags bit 6 (0x40)."""
    extractor = ParagraphExtractor()
    pdftext_result = [
        {
            "bbox": [0, 0, 600, 800],
            "blocks": [
                {
                    "bbox": [0, 0, 100, 20],
                    "lines": [
                        {"spans": [{"text": "Italic text", "font": {"size": 12, "flags": 0x40}}]}
                    ],
                }
            ],
        }
    ]

    paragraphs = extractor.extract(pdftext_result)
    assert paragraphs[0].is_italic is True


def test_is_italic_from_font_name():
    """Test italic detection from font name containing 'italic'."""
    extractor = ParagraphExtractor()
    pdftext_result = [
        {
            "bbox": [0, 0, 600, 800],
            "blocks": [
                {
                    "bbox": [0, 0, 100, 20],
                    "lines": [
                        {
                            "spans": [
                                {
                                    "text": "Italic text",
                                    "font": {"size": 12, "flags": 0, "name": "Times-Italic"},
                                }
                            ]
                        }
                    ],
                }
            ],
        }
    ]

    paragraphs = extractor.extract(pdftext_result)
    assert paragraphs[0].is_italic is True


def test_font_name_extraction():
    """Test font name extraction uses weighted mode."""
    extractor = ParagraphExtractor()
    pdftext_result = [
        {
            "bbox": [0, 0, 600, 800],
            "blocks": [
                {
                    "bbox": [0, 0, 100, 20],
                    "lines": [
                        {
                            "spans": [
                                {"text": "AB", "font": {"size": 12, "name": "Helvetica"}},
                                {"text": "CDEFGHIJ", "font": {"size": 12, "name": "Times-Roman"}},
                            ]
                        }
                    ],
                }
            ],
        }
    ]

    paragraphs = extractor.extract(pdftext_result)
    # Times-Roman has more text, so it should be selected
    assert paragraphs[0].font_name == "Times-Roman"


def test_rotation_extraction():
    """Test rotation extraction from spans.

    Note: pdftext returns rotation in radians, which is converted to degrees.
    """
    import math

    extractor = ParagraphExtractor()
    # pdftext returns rotation in radians (e.g., π/2 for 90°)
    rotation_radians = math.pi / 2  # ≈ 1.5708
    pdftext_result = [
        {
            "bbox": [0, 0, 600, 800],
            "blocks": [
                {
                    "bbox": [0, 0, 100, 20],
                    "lines": [
                        {"spans": [{"text": "Rotated", "font": {"size": 12}, "rotation": rotation_radians}]}
                    ],
                }
            ],
        }
    ]

    paragraphs = extractor.extract(pdftext_result)
    # Should be converted to degrees (90°)
    assert abs(paragraphs[0].rotation - 90.0) < 0.1


def test_alignment_left():
    """Test left alignment detection from consistent left edge."""
    extractor = ParagraphExtractor()
    pdftext_result = [
        {
            "bbox": [0, 0, 600, 800],
            "blocks": [
                {
                    "bbox": [10, 0, 200, 60],
                    "lines": [
                        {
                            "bbox": [10, 0, 150, 20],
                            "spans": [{"text": "Line 1", "font": {"size": 12}}],
                        },
                        {
                            "bbox": [10, 20, 120, 40],
                            "spans": [{"text": "Line 2", "font": {"size": 12}}],
                        },
                        {
                            "bbox": [10, 40, 180, 60],
                            "spans": [{"text": "Line 3", "font": {"size": 12}}],
                        },
                    ],
                }
            ],
        }
    ]

    paragraphs = extractor.extract(pdftext_result)
    assert paragraphs[0].alignment == "left"


def test_alignment_center():
    """Test center alignment detection from consistent center positions."""
    extractor = ParagraphExtractor()
    pdftext_result = [
        {
            "bbox": [0, 0, 600, 800],
            "blocks": [
                {
                    "bbox": [50, 0, 150, 40],
                    "lines": [
                        {
                            "bbox": [60, 0, 140, 20],
                            "spans": [{"text": "Centered", "font": {"size": 12}}],
                        },
                        {
                            "bbox": [70, 20, 130, 40],
                            "spans": [{"text": "Text", "font": {"size": 12}}],
                        },
                    ],
                }
            ],
        }
    ]

    paragraphs = extractor.extract(pdftext_result)
    assert paragraphs[0].alignment == "center"


def test_alignment_justify():
    """Test justify alignment detection from consistent left and right edges."""
    extractor = ParagraphExtractor()
    pdftext_result = [
        {
            "bbox": [0, 0, 600, 800],
            "blocks": [
                {
                    "bbox": [10, 0, 200, 60],
                    "lines": [
                        {
                            "bbox": [10, 0, 200, 20],
                            "spans": [{"text": "Line 1", "font": {"size": 12}}],
                        },
                        {
                            "bbox": [10, 20, 200, 40],
                            "spans": [{"text": "Line 2", "font": {"size": 12}}],
                        },
                        {
                            "bbox": [10, 40, 200, 60],
                            "spans": [{"text": "Line 3", "font": {"size": 12}}],
                        },
                    ],
                }
            ],
        }
    ]

    paragraphs = extractor.extract(pdftext_result)
    assert paragraphs[0].alignment == "justify"


def test_alignment_left_with_first_line_indent():
    """Test left alignment detection when first line has indent (common in body text)."""
    extractor = ParagraphExtractor()
    # First line is indented (x=30), but subsequent lines start at x=10
    # This is a common pattern in body text paragraphs
    pdftext_result = [
        {
            "bbox": [0, 0, 600, 800],
            "blocks": [
                {
                    "bbox": [10, 0, 200, 80],
                    "lines": [
                        {
                            "bbox": [30, 0, 180, 20],  # First line indented
                            "spans": [{"text": "First line with indent", "font": {"size": 12}}],
                        },
                        {
                            "bbox": [10, 20, 190, 40],  # Subsequent lines at left edge
                            "spans": [{"text": "Second line at margin", "font": {"size": 12}}],
                        },
                        {
                            "bbox": [10, 40, 170, 60],
                            "spans": [{"text": "Third line at margin", "font": {"size": 12}}],
                        },
                        {
                            "bbox": [10, 60, 185, 80],
                            "spans": [{"text": "Fourth line at margin", "font": {"size": 12}}],
                        },
                    ],
                }
            ],
        }
    ]

    paragraphs = extractor.extract(pdftext_result)
    # Should be detected as left-aligned (middle lines have consistent left edge)
    assert paragraphs[0].alignment == "left"


# === List marker detection tests ===


def test_detect_bullet_marker():
    """Bullet in separate span should be detected."""
    extractor = ParagraphExtractor()
    pdftext_result = [
        {
            "bbox": [0, 0, 600, 800],
            "blocks": [
                {
                    "bbox": [100, 200, 400, 220],
                    "lines": [
                        {
                            "bbox": [100, 200, 400, 220],
                            "spans": [
                                {"text": "•", "bbox": [100, 200, 108, 212], "font": {"size": 12}},
                                {"text": " ", "bbox": [108, 200, 111, 212], "font": {"size": 12}},
                                {"text": "Item content", "bbox": [111, 200, 400, 212], "font": {"size": 12}},
                            ],
                        }
                    ],
                }
            ],
        }
    ]

    paragraphs = extractor.extract(pdftext_result)
    assert len(paragraphs) == 1
    assert paragraphs[0].list_marker is not None
    assert paragraphs[0].list_marker.marker_type == "bullet"
    assert paragraphs[0].list_marker.marker_text == "•"
    assert paragraphs[0].text == "Item content"


def test_detect_numbered_marker():
    """Number with period should be detected."""
    extractor = ParagraphExtractor()
    pdftext_result = [
        {
            "bbox": [0, 0, 600, 800],
            "blocks": [
                {
                    "bbox": [100, 200, 400, 220],
                    "lines": [
                        {
                            "bbox": [100, 200, 400, 220],
                            "spans": [
                                {"text": "1.", "bbox": [100, 200, 115, 212], "font": {"size": 12}},
                                {"text": " ", "bbox": [115, 200, 118, 212], "font": {"size": 12}},
                                {"text": "First item", "bbox": [118, 200, 400, 212], "font": {"size": 12}},
                            ],
                        }
                    ],
                }
            ],
        }
    ]

    paragraphs = extractor.extract(pdftext_result)
    assert len(paragraphs) == 1
    assert paragraphs[0].list_marker is not None
    assert paragraphs[0].list_marker.marker_type == "numbered"
    assert paragraphs[0].list_marker.number == 1


def test_detect_numbered_paren_marker():
    """Number with parenthesis should be detected."""
    extractor = ParagraphExtractor()
    pdftext_result = [
        {
            "bbox": [0, 0, 600, 800],
            "blocks": [
                {
                    "bbox": [100, 200, 400, 220],
                    "lines": [
                        {
                            "bbox": [100, 200, 400, 220],
                            "spans": [
                                {"text": "2)", "bbox": [100, 200, 115, 212], "font": {"size": 12}},
                                {"text": " ", "bbox": [115, 200, 118, 212], "font": {"size": 12}},
                                {"text": "Second item", "bbox": [118, 200, 400, 212], "font": {"size": 12}},
                            ],
                        }
                    ],
                }
            ],
        }
    ]

    paragraphs = extractor.extract(pdftext_result)
    assert len(paragraphs) == 1
    assert paragraphs[0].list_marker is not None
    assert paragraphs[0].list_marker.marker_type == "numbered"
    assert paragraphs[0].list_marker.number == 2


def test_detect_circled_number_marker():
    """Circled number should be detected."""
    extractor = ParagraphExtractor()
    pdftext_result = [
        {
            "bbox": [0, 0, 600, 800],
            "blocks": [
                {
                    "bbox": [100, 200, 400, 220],
                    "lines": [
                        {
                            "bbox": [100, 200, 400, 220],
                            "spans": [
                                {"text": "③", "bbox": [100, 200, 112, 212], "font": {"size": 12}},
                                {"text": " ", "bbox": [112, 200, 115, 212], "font": {"size": 12}},
                                {"text": "Third item", "bbox": [115, 200, 400, 212], "font": {"size": 12}},
                            ],
                        }
                    ],
                }
            ],
        }
    ]

    paragraphs = extractor.extract(pdftext_result)
    assert len(paragraphs) == 1
    assert paragraphs[0].list_marker is not None
    assert paragraphs[0].list_marker.marker_type == "numbered"
    assert paragraphs[0].list_marker.number == 3


def test_list_block_creates_multiple_paragraphs():
    """Block with list items should create separate paragraphs."""
    extractor = ParagraphExtractor()
    pdftext_result = [
        {
            "bbox": [0, 0, 600, 800],
            "blocks": [
                {
                    "bbox": [100, 180, 400, 220],
                    "lines": [
                        {
                            "bbox": [100, 200, 400, 212],
                            "spans": [
                                {"text": "•", "bbox": [100, 200, 108, 212], "font": {"size": 12}},
                                {"text": " ", "bbox": [108, 200, 111, 212], "font": {"size": 12}},
                                {"text": "First item", "bbox": [111, 200, 400, 212], "font": {"size": 12}},
                            ],
                        },
                        {
                            "bbox": [100, 180, 400, 192],
                            "spans": [
                                {"text": "•", "bbox": [100, 180, 108, 192], "font": {"size": 12}},
                                {"text": " ", "bbox": [108, 180, 111, 192], "font": {"size": 12}},
                                {"text": "Second item", "bbox": [111, 180, 400, 192], "font": {"size": 12}},
                            ],
                        },
                    ],
                }
            ],
        }
    ]

    paragraphs = extractor.extract(pdftext_result)
    assert len(paragraphs) == 2
    assert paragraphs[0].list_marker is not None
    assert paragraphs[0].text == "First item"
    assert paragraphs[1].list_marker is not None
    assert paragraphs[1].text == "Second item"


def test_regular_block_no_list_marker():
    """Block without list markers should not have list_marker."""
    extractor = ParagraphExtractor()
    pdftext_result = [
        {
            "bbox": [0, 0, 600, 800],
            "blocks": [
                {
                    "bbox": [100, 200, 400, 220],
                    "lines": [
                        {
                            "bbox": [100, 200, 400, 220],
                            "spans": [
                                {"text": "Regular paragraph text", "bbox": [100, 200, 400, 212], "font": {"size": 12}},
                            ],
                        }
                    ],
                }
            ],
        }
    ]

    paragraphs = extractor.extract(pdftext_result)
    assert len(paragraphs) == 1
    assert paragraphs[0].list_marker is None
    assert paragraphs[0].text == "Regular paragraph text"


def test_single_span_not_list():
    """Single span line should not be detected as list."""
    extractor = ParagraphExtractor()
    pdftext_result = [
        {
            "bbox": [0, 0, 600, 800],
            "blocks": [
                {
                    "bbox": [100, 200, 400, 220],
                    "lines": [
                        {
                            "bbox": [100, 200, 400, 220],
                            "spans": [
                                {"text": "• Item without separation", "bbox": [100, 200, 400, 212], "font": {"size": 12}},
                            ],
                        }
                    ],
                }
            ],
        }
    ]

    paragraphs = extractor.extract(pdftext_result)
    assert len(paragraphs) == 1
    assert paragraphs[0].list_marker is None


def test_detect_lists_disabled():
    """When detect_lists=False, list markers should not be detected."""
    extractor = ParagraphExtractor()
    config = ExtractorConfig(detect_lists=False)
    pdftext_result = [
        {
            "bbox": [0, 0, 600, 800],
            "blocks": [
                {
                    "bbox": [100, 200, 400, 220],
                    "lines": [
                        {
                            "bbox": [100, 200, 400, 220],
                            "spans": [
                                {"text": "•", "bbox": [100, 200, 108, 212], "font": {"size": 12}},
                                {"text": " ", "bbox": [108, 200, 111, 212], "font": {"size": 12}},
                                {"text": "Item", "bbox": [111, 200, 400, 212], "font": {"size": 12}},
                            ],
                        }
                    ],
                }
            ],
        }
    ]

    paragraphs = extractor.extract(pdftext_result, config=config)
    assert len(paragraphs) == 1
    assert paragraphs[0].list_marker is None


def test_continuation_line_merged():
    """List item with continuation line should be merged."""
    extractor = ParagraphExtractor()
    pdftext_result = [
        {
            "bbox": [0, 0, 600, 800],
            "blocks": [
                {
                    "bbox": [100, 160, 400, 220],
                    "lines": [
                        {
                            "bbox": [100, 200, 400, 212],
                            "spans": [
                                {"text": "•", "bbox": [100, 200, 108, 212], "font": {"size": 12}},
                                {"text": " ", "bbox": [108, 200, 111, 212], "font": {"size": 12}},
                                {"text": "First part of item", "bbox": [111, 200, 400, 212], "font": {"size": 12}},
                            ],
                        },
                        {
                            "bbox": [111, 180, 400, 192],
                            "spans": [
                                {"text": "continuation text", "bbox": [111, 180, 400, 192], "font": {"size": 12}},
                            ],
                        },
                        {
                            "bbox": [100, 160, 400, 172],
                            "spans": [
                                {"text": "•", "bbox": [100, 160, 108, 172], "font": {"size": 12}},
                                {"text": " ", "bbox": [108, 160, 111, 172], "font": {"size": 12}},
                                {"text": "Second item", "bbox": [111, 160, 400, 172], "font": {"size": 12}},
                            ],
                        },
                    ],
                }
            ],
        }
    ]

    paragraphs = extractor.extract(pdftext_result)
    # Should create 2 paragraphs (continuation merged with first item)
    assert len(paragraphs) == 2
    assert paragraphs[0].list_marker is not None
    assert "First part of item" in paragraphs[0].text
    assert "continuation text" in paragraphs[0].text
    assert paragraphs[1].text == "Second item"
