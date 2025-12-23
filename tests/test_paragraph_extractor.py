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
