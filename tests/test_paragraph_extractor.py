# SPDX-License-Identifier: Apache-2.0
"""Tests for ParagraphExtractor."""

from pdf_translator.core.paragraph_extractor import ParagraphExtractor


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
    """Test rotation extraction from spans."""
    extractor = ParagraphExtractor()
    pdftext_result = [
        {
            "bbox": [0, 0, 600, 800],
            "blocks": [
                {
                    "bbox": [0, 0, 100, 20],
                    "lines": [
                        {"spans": [{"text": "Rotated", "font": {"size": 12}, "rotation": 1.57}]}
                    ],
                }
            ],
        }
    ]

    paragraphs = extractor.extract(pdftext_result)
    assert paragraphs[0].rotation == 1.57


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
