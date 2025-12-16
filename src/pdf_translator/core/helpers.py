# SPDX-License-Identifier: Apache-2.0
"""Helper functions for pypdfium2 raw API operations.

This module provides ctypes conversion utilities required for
pypdfium2's low-level PDFium API.
"""

import ctypes


def to_widestring(text: str) -> ctypes.Array:
    """Convert Python string to FPDF_WIDESTRING (UTF-16LE + null terminator).

    PDFium expects wide strings in UTF-16LE encoding with a null terminator.

    Args:
        text: Python string to convert

    Returns:
        ctypes array of c_ushort (UTF-16LE encoded)

    Example:
        >>> ws = to_widestring("Hello")
        >>> # ws can now be passed to FPDFText_SetText
    """
    # Encode as UTF-16LE and add null terminator (2 bytes)
    encoded = text.encode("utf-16-le") + b"\x00\x00"
    # Create ctypes array of unsigned shorts
    arr = (ctypes.c_ushort * (len(encoded) // 2))()
    for i in range(len(encoded) // 2):
        arr[i] = int.from_bytes(encoded[i * 2 : i * 2 + 2], "little")
    return arr


def to_byte_array(data: bytes) -> ctypes.Array:
    """Convert bytes to ctypes array of unsigned bytes.

    Used for passing binary data (e.g., font files) to PDFium.

    Args:
        data: Bytes to convert

    Returns:
        ctypes array of c_ubyte

    Example:
        >>> with open("font.ttf", "rb") as f:
        ...     font_data = f.read()
        >>> arr = to_byte_array(font_data)
        >>> # arr can now be passed to FPDFText_LoadFont
    """
    arr = (ctypes.c_ubyte * len(data))()
    for i, b in enumerate(data):
        arr[i] = b
    return arr
