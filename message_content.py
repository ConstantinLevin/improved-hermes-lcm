"""Readers over a message's content: its images, found by structure, and the text
the full-text index reads.

Hermes/OpenAI-format messages carry ``content`` as a string or as a list of parts
(text parts, image parts). An image is a part of type ``image_url``, ``image`` or
``input_image``, or sits in the ``_multimodal`` envelope an engine tool returns;
that is read from the message's structure, never from its text (#35). The record
keeps the message verbatim (#3); these readers derive from it and change nothing.
"""

from __future__ import annotations

import json
import re
from typing import Any

_TEXT_PART_TYPES = {"text", "input_text", "output_text"}
IMAGE_PART_TYPES = frozenset({"image_url", "image", "input_image"})

# Base64 inside a string is text the agent saw, not an image. This shape test only
# names it in a log line labelled as a heuristic; nothing depends on it.
_DATA_URI_RE = re.compile(r"data:([\w.+-]+/[\w.+-]+);base64,[A-Za-z0-9+/=_-]{256,}")
_BASE64_RUN_RE = re.compile(r"[A-Za-z0-9+/=_-]{4096,}")


def _extract_text_part_value(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        nested = value.get("value")
        if isinstance(nested, str):
            return nested
        nested = value.get("content")
        if isinstance(nested, str):
            return nested
    return None


def normalize_content_value(content: Any) -> str | None:
    """Return a stable text representation for message content.

    ``None`` remains ``None``. Strings are returned unchanged. Structured content
    is serialized deterministically for token accounting.
    """
    if content is None:
        return None
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return str(content)


def _is_multimodal_envelope(value: Any) -> bool:
    return isinstance(value, dict) and value.get("_multimodal") is True and isinstance(value.get("content"), list)


def _parts(content: Any) -> list | None:
    """The content's list of parts, looking inside the ``_multimodal`` envelope."""
    if _is_multimodal_envelope(content):
        return content["content"]
    if isinstance(content, list):
        return content
    return None


def _is_image_part(part: Any) -> bool:
    return isinstance(part, dict) and part.get("type") in IMAGE_PART_TYPES


def image_parts(content: Any) -> list[dict]:
    """The image parts of a message's content, by structure."""
    parts = _parts(content)
    return [part for part in parts if _is_image_part(part)] if parts else []


def describe_image_part(part: dict) -> str:
    """A log line's description of an image part: its MIME type and size, or that
    it is a remote URL. Never the data."""
    value = part.get("image_url", part.get("image", part.get("url")))
    if isinstance(value, dict):
        value = value.get("url")
    if isinstance(value, str) and value.startswith("data:"):
        mime = value[5:].split(";", 1)[0] or "unknown type"
        return f"{part.get('type')} {mime}, {len(value):,} characters"
    if isinstance(value, str) and value:
        return f"{part.get('type')} remote URL"
    return f"{part.get('type')} of another shape"


def index_text(content: Any) -> str | None:
    """The text the full-text index reads for a message's content.

    A string is itself. A list of parts (or the ``_multimodal`` envelope) gives its
    text parts, and any other part that is not an image as JSON; image parts are
    left out by structure (#35). Base64 inside a string is text and stays in.
    """
    if content is None or isinstance(content, str):
        return content
    parts = _parts(content)
    if parts is None:
        return normalize_content_value(content)
    pieces: list[str] = []
    for part in parts:
        if isinstance(part, str):
            pieces.append(part)
        elif _is_image_part(part):
            continue
        elif isinstance(part, dict) and part.get("type") in _TEXT_PART_TYPES:
            text = _extract_text_part_value(part.get("text"))
            if text is None:
                text = _extract_text_part_value(part.get("content"))
            if text:
                pieces.append(text)
        else:
            pieces.append(normalize_content_value(part) or "")
    return "\n".join(piece for piece in pieces if piece)


def base64_like_strings(message: dict) -> list[str]:
    """Heuristic, by the text's shape: descriptions of strings in a message's
    content and tool-call arguments that look like base64 (a data URI, or a long
    run of the base64 alphabet). Image parts are not looked at here."""
    texts: list[str] = []
    content = message.get("content")
    if isinstance(content, str):
        texts.append(content)
    for part in _parts(content) or ():
        if isinstance(part, str):
            texts.append(part)
        elif isinstance(part, dict) and not _is_image_part(part):
            for value in part.values():
                if isinstance(value, str):
                    texts.append(value)
    for call in message.get("tool_calls") or ():
        function = call.get("function") if isinstance(call, dict) else None
        arguments = function.get("arguments") if isinstance(function, dict) else None
        if isinstance(arguments, str):
            texts.append(arguments)
    found: list[str] = []
    for text in texts:
        for match in _DATA_URI_RE.finditer(text):
            found.append(f"a data URI of type {match.group(1)}, {len(match.group(0)):,} characters")
        if not _DATA_URI_RE.search(text):
            for match in _BASE64_RUN_RE.finditer(text):
                found.append(f"a run of {len(match.group(0)):,} base64-alphabet characters")
    return found
