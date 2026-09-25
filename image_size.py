"""An image part's pixel dimensions, read from its data's header (#21, #35).

The part is found by structure (``message_content``); its data is a data URL's base64
payload or an Anthropic ``source`` of type base64. The header of PNG, JPEG, GIF and
WebP gives the width and height. Anything else (a remote URL, a file id, another
format, a header that does not parse) has no dimensions here, and the estimate counts
that image as uncounted rather than guessing.
"""

from __future__ import annotations

import base64
import binascii
import struct
from typing import Any, Optional


def _data_bytes(part: dict) -> Optional[bytes]:
    payload: Optional[str] = None
    source = part.get("source")
    if isinstance(source, dict) and source.get("type") == "base64" and isinstance(source.get("data"), str):
        payload = source["data"]
    else:
        value: Any = part.get("image_url", part.get("url"))
        if isinstance(value, dict):
            value = value.get("url")
        if isinstance(value, str) and value.startswith("data:") and "," in value:
            header, _, data = value.partition(",")
            if ";base64" in header:
                payload = data
    if payload is None:
        return None
    try:
        return base64.b64decode(payload + "=" * (-len(payload) % 4), validate=False)
    except (binascii.Error, ValueError):
        return None


def _png(data: bytes) -> Optional[tuple[int, int]]:
    if data[:8] == b"\x89PNG\r\n\x1a\n" and data[12:16] == b"IHDR" and len(data) >= 24:
        return struct.unpack(">II", data[16:24])
    return None


def _gif(data: bytes) -> Optional[tuple[int, int]]:
    if data[:6] in (b"GIF87a", b"GIF89a") and len(data) >= 10:
        return struct.unpack("<HH", data[6:10])
    return None


def _webp(data: bytes) -> Optional[tuple[int, int]]:
    if data[:4] != b"RIFF" or data[8:12] != b"WEBP" or len(data) < 30:
        return None
    chunk = data[12:16]
    if chunk == b"VP8X":
        width = 1 + int.from_bytes(data[24:27], "little")
        height = 1 + int.from_bytes(data[27:30], "little")
        return width, height
    if chunk == b"VP8L" and data[20] == 0x2F:
        bits = int.from_bytes(data[21:25], "little")
        return 1 + (bits & 0x3FFF), 1 + ((bits >> 14) & 0x3FFF)
    if chunk == b"VP8 " and data[23:26] == b"\x9d\x01\x2a":
        width, height = struct.unpack("<HH", data[26:30])
        return width & 0x3FFF, height & 0x3FFF
    return None


def _jpeg(data: bytes) -> Optional[tuple[int, int]]:
    if data[:2] != b"\xff\xd8":
        return None
    at = 2
    while at + 9 < len(data):
        if data[at] != 0xFF:
            at += 1
            continue
        marker = data[at + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            at += 2
            continue
        length = int.from_bytes(data[at + 2:at + 4], "big")
        if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            height, width = struct.unpack(">HH", data[at + 5:at + 9])
            return width, height
        at += 2 + length
    return None


def image_dimensions(part: dict) -> Optional[tuple[int, int]]:
    """(width, height) in pixels, or None where the header cannot tell."""
    data = _data_bytes(part)
    if not data:
        return None
    for reader in (_png, _jpeg, _gif, _webp):
        try:
            size = reader(data)
        except (struct.error, IndexError):
            size = None
        if size and size[0] > 0 and size[1] > 0:
            return int(size[0]), int(size[1])
    return None
