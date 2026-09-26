"""Readers over a message's content: its images, found by structure, and the text
grep searches.

Hermes/OpenAI-format messages carry ``content`` as a string or as a list of parts
(text parts, image parts). An image is a part of type ``image_url`` (Chat
Completions), ``image`` (Anthropic, with a ``source``) or ``input_image``
(Responses), in the content list or in the ``_multimodal`` envelope an engine tool
returns; that is read from the message's structure, never from its text (#35). The
record keeps the message verbatim (#3); these readers derive from it and change
nothing. Log lines name an image's media type and size, never its data.
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


_LOGGED_FIELD_MAX = 100


def _log_field(value: str) -> str:
    """A value for a log line, cut at 100 characters with the cut stated. This cuts
    a log field, never stored content."""
    if len(value) <= _LOGGED_FIELD_MAX:
        return value
    return f"{value[:_LOGGED_FIELD_MAX]}… (cut at {_LOGGED_FIELD_MAX} of {len(value):,} characters)"


def _describe_data_url(url: str) -> str:
    """A data URL by RFC 2397: ``data:[<mediatype>][;<parameter>]*[;base64],<data>``.
    The media type and parameters end at the first comma; the payload follows it and
    is only counted, never logged."""
    header, comma, payload = url[5:].partition(",")
    if not comma:
        return f"a data URL without a comma, {len(url):,} characters"
    params = header.split(";")
    media_type = params[0] or "no media type (text/plain by default)"
    encoding = "base64" if "base64" in (p.strip().lower() for p in params[1:]) else "percent-encoded"
    return f"{_log_field(media_type)}, {encoding}, {len(payload):,} payload characters"


def describe_image_part(part: dict) -> str:
    """A log line's description of an image part: its media type and size, a remote
    URL, or a file id. Never the image's data."""
    kind = _log_field(str(part.get("type")))
    source = part.get("source")
    if isinstance(source, dict):  # Anthropic: {"type": "image", "source": {...}}
        if source.get("type") == "base64":
            data = source.get("data")
            size = len(data) if isinstance(data, str) else 0
            return f"{kind} {_log_field(str(source.get('media_type') or 'no media type'))}, base64, {size:,} payload characters"
        if source.get("type") == "url":
            return f"{kind} remote URL"
        return f"{kind} with a source of type {_log_field(str(source.get('type')))}"
    value = part.get("image_url", part.get("url"))
    if isinstance(value, dict):  # Chat Completions: {"image_url": {"url": ...}}
        value = value.get("url")
    if isinstance(value, str) and value.startswith("data:"):
        return f"{kind} {_describe_data_url(value)}"
    if isinstance(value, str) and value:
        return f"{kind} remote URL"
    if part.get("file_id"):  # Responses: {"type": "input_image", "file_id": ...}
        return f"{kind} file id {_log_field(str(part.get('file_id')))}"
    return f"{kind} of another shape"


def image_media_type(part: dict) -> str:
    """An image part's media type, read from its structure: the data URL's header or
    the Anthropic source's ``media_type``; "remote URL" or "file id" where the part
    carries no data. Never the data itself."""
    source = part.get("source")
    if isinstance(source, dict):
        if source.get("type") == "base64":
            return str(source.get("media_type") or "no media type")
        return "remote URL" if source.get("type") == "url" else f"source of type {source.get('type')}"
    value = part.get("image_url", part.get("url"))
    if isinstance(value, dict):
        value = value.get("url")
    if isinstance(value, str) and value.startswith("data:"):
        return value[5:].partition(",")[0].split(";")[0] or "no media type"
    if isinstance(value, str) and value:
        return "remote URL"
    if part.get("file_id"):
        return "file id"
    return "unknown shape"


def content_parts(content: Any) -> list | None:
    """The content's list of parts, looking inside the ``_multimodal`` envelope; None
    for a string or no content."""
    return _parts(content)


def is_image_part(part: Any) -> bool:
    return _is_image_part(part)


def _shown_json(value: Any) -> str:
    """A value that is not a string as the tools' pages render it: JSON in its stored key
    order, ``ensure_ascii`` off (``results.final_result``)."""
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def _parts_strings(parts: list) -> list[str]:
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
            pieces.append(_shown_json(part))
    return pieces


def _content_strings(content: Any) -> list[str]:
    """The strings a message's content shows: a string is itself; a list gives each text
    part's text and every other part that is not an image as its JSON; the ``_multimodal``
    envelope gives the same for its list and every other field it carries (``text_summary``,
    ``meta`` …) as a string or its JSON. Structural image parts are left out (#35); base64
    inside a string stays in."""
    if content is None:
        return []
    if isinstance(content, str):
        return [content]
    if _is_multimodal_envelope(content):
        pieces: list[str] = []
        for key, value in content.items():
            if key == "_multimodal":
                continue
            if key == "content":
                pieces.extend(_parts_strings(value))
            elif isinstance(value, str):
                pieces.append(value)
            else:
                pieces.append(_shown_json(value))
        return pieces
    if isinstance(content, list):
        return _parts_strings(content)
    return [_shown_json(content)]


def sent_content(raw: dict) -> Any:
    """A stored message's content as the host sends it: the ``api_content`` sidecar, where
    it is a non-empty string on a user or assistant row, in place of ``content``; else
    ``content``. The host's rule for a row of its history (``build_api_messages``,
    agent/turn_context.py:1227-1252 at Hermes cdcd53c2cd); one rule for expansion
    (``summariser_input._row_before_fill``) and for what grep searches (``grep_text``)."""
    return raw["api_content"] if sidecar_sent(raw) else raw.get("content")


def sidecar_sent(raw: dict) -> bool:
    """Whether the host sends the row's ``api_content`` in place of its ``content``."""
    sidecar = raw.get("api_content")
    return isinstance(sidecar, str) and bool(sidecar) and raw.get("role") in ("user", "assistant")


def readable_reasoning(raw: dict) -> str | None:
    """The readable reasoning of a stored message, the one rule expansion, the summariser
    and grep read (#8, #18): ``reasoning``, else a non-blank ``reasoning_content``. The
    host's ``reasoning`` is its merged readable text (agent/agent_runtime_helpers.py:1362-1395
    at Hermes cdcd53c2cd); a blank ``reasoning_content`` is the host's tool-call pad."""
    for key in ("reasoning", "reasoning_content"):
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


# What separates the strings of ``grep_text``; a term holding it is refused (``grep``), so a
# match never spans two strings. A NUL, where SQLite's text functions stop, is written as
# this too: a term never holds a NUL either.
GREP_SEPARATOR = "\x1f"


def grep_text(raw: dict) -> str:
    """What grep searches in a stored message (#18 D2): the strings the agent's past shows as
    the message's own words and actions, each by itself, joined by ``GREP_SEPARATOR``: its
    content as the host sends it (``sent_content``), then for each tool call its name and
    its arguments (the stored string, or the JSON of another value), then, on an assistant
    message, its readable reasoning (expansion shows it there only). Never an image, an
    encrypted item, a native carrier, a key or the host's bookkeeping. "" for a message
    without strings, never None."""
    strings = list(_content_strings(sent_content(raw)))
    calls = raw.get("tool_calls")
    if isinstance(calls, list):
        for call in calls:
            function = call.get("function") if isinstance(call, dict) else None
            if not isinstance(function, dict):
                continue
            name, arguments = function.get("name"), function.get("arguments")
            if isinstance(name, str):
                strings.append(name)
            if isinstance(arguments, str):
                strings.append(arguments)
            elif arguments is not None:
                strings.append(_shown_json(arguments))
    if raw.get("role") == "assistant":
        reasoning = readable_reasoning(raw)
        if reasoning is not None:
            strings.append(reasoning)
    return GREP_SEPARATOR.join(s.replace("\x00", GREP_SEPARATOR) for s in strings if s)


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
            found.append(f"a data URI of type {_log_field(match.group(1))}, {len(match.group(0)):,} characters")
        if not _DATA_URI_RE.search(text):
            for match in _BASE64_RUN_RE.finditer(text):
                found.append(f"a run of {len(match.group(0)):,} base64-alphabet characters")
    return found
