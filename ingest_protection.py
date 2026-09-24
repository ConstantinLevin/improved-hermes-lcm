"""Storage-boundary protection for payloads that should not live inline in SQLite.

Hermes core may hand LCM messages that already contain inline media/base64
payloads. LCM remains lossless by externalizing those payload strings and
storing compact placeholders in ``messages.content`` / ``messages.tool_calls``.
This avoids duplicating large/binary-ish payloads into SQLite rows, FTS shadow
structures, WAL files, and backups while keeping recovery available through LCM
externalized-payload tools.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import stat
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from .externalize import (
    externalize_ingest_payload,
    find_externalized_payload_for_message,
    load_externalized_payload,
)
from .message_content import normalize_content_value

logger = logging.getLogger(__name__)

_MEDIA_TYPE_HINTS = ("image", "audio", "video")
_MEDIA_VALUE_KEYS = (
    "image_url",
    "input_image",
    "output_image",
    "audio_url",
    "video_url",
    "image",
    "audio",
    "video",
)


def _contains_media_payload(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(_DATA_URI_BASE64_RE.search(value))
    if isinstance(value, list):
        return any(_contains_media_payload(item) for item in value)
    if isinstance(value, dict):
        block_type = str(value.get("type") or "").lower()
        if any(hint in block_type for hint in _MEDIA_TYPE_HINTS):
            return True
        for key, nested in value.items():
            key_text = str(key).lower()
            if key_text in _MEDIA_VALUE_KEYS:
                return True
            if _contains_media_payload(nested):
                return True
    return False


# Any data URI base64 payload, not just image/audio/video. Keep the trailing
# payload alphabet conservative so we do not slurp surrounding JSON/markdown.
# Raw scans can see JSON-escaped slashes before decoding, including both `\/`
# and unicode escapes such as `\u002f` in duplicate-key argument strings.
_JSON_ESCAPED_SLASH_RE = r"(?:/|\\/|\\u002[fF])"
_DATA_URI_BASE64_RE = re.compile(
    rf"data:(?:[A-Za-z0-9.+-]|{_JSON_ESCAPED_SLASH_RE})*"
    rf"(?:;[A-Za-z0-9_.+%-]+=(?:[-A-Za-z0-9_.+%]|{_JSON_ESCAPED_SLASH_RE})*)*"
    rf";base64,(?:[A-Za-z0-9+=]|{_JSON_ESCAPED_SLASH_RE}){{256,}}(?=$|[^A-Za-z0-9+/=])",
    re.IGNORECASE,
)

_BASE64_RUN_RE = re.compile(r"(?<![A-Za-z0-9+/=_-])([A-Za-z0-9+/=_-]{4096,})(?![A-Za-z0-9+/=_-])")
# Line-wrapped base64 (MIME 76 / PEM 64 chars per line) never forms a single
# 4096-char contiguous run, so _BASE64_RUN_RE misses it entirely. Match a block
# of consecutive base64-alphabet lines; looks_like_long_base64 makes the final
# call on the whitespace-compacted block.
_WRAPPED_BASE64_MIN_LINE_CHARS = 40
_WRAPPED_BASE64_MIN_TERMINAL_LINE_CHARS = 16
_BASE64_ALPHABET_RE = re.compile(r"^[A-Za-z0-9+/=_\s-]+$")
_BASE64_LINE_ALPHABET_RE = re.compile(r"^[A-Za-z0-9+/=_-]+$")
_EXTERNALIZED_PLACEHOLDER_PREFIX = "[Externalized LCM ingest payload:"
_QUARANTINED_ASSISTANT_KIND = "quarantined_assistant_output"
_QUARANTINED_ASSISTANT_REASON = "high_repetition"
_QUARANTINED_ASSISTANT_MIN_CHARS = 65_536
_QUARANTINED_ASSISTANT_MIN_TOKENS = 1_000
_WORD_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_REPETITION_SEGMENT_SPLIT_RE = re.compile(r"(?:\n+|(?<=[.!?])\s+)")
_GENERIC_BASE64_MIN_CHARS = 4096
_INGEST_PLACEHOLDER_RE = re.compile(r"\[Externalized LCM ingest payload:.*?;\s*ref=([^;\]\s]+)\]")
_PERSISTED_OUTPUT_TAG = "<persisted-output>"
_PERSISTED_OUTPUT_CLOSING_TAG = "</persisted-output>"
_PERSISTED_OUTPUT_SAVED_TO_RE = re.compile(r"^Full output saved to:\s*(?P<path>.+?)\s*$", re.MULTILINE)
_PERSISTED_OUTPUT_PREVIEW_RE = re.compile(
    r"^Preview \(first \d+ chars\):\s*\r?\n(?P<preview>.*?)\r?\n</persisted-output>\s*$",
    re.MULTILINE | re.DOTALL,
)
_PERSISTED_OUTPUT_CHAR_COUNT_RE = re.compile(r"too large\s*\((?P<count>[\d,]+)\s+characters\b", re.IGNORECASE)
_PERSISTED_OUTPUT_INLINE_PREVIEW_SHA256_RE = re.compile(
    r"\r?\n\[LCM persisted-output marker identity: preview_sha256=(?P<sha256>[0-9a-f]{64})\]"
    r"(?:\r?\n\[LCM persisted-output file generation: size=\d+; mtime_ns=\d+; ctime_ns=\d+\])?"
    r"\r?\n</persisted-output>\s*$"
)
_PERSISTED_OUTPUT_INLINE_GENERATION_RE = re.compile(
    r"\r?\n\[LCM persisted-output file generation: size=(?P<size>\d+); mtime_ns=(?P<mtime_ns>\d+); ctime_ns=(?P<ctime_ns>\d+)\]\r?\n</persisted-output>\s*$"
)
_UNRECOVERABLE_TRUNCATION_RE = re.compile(
    r"\[Truncated:\s*tool response was [\d,]+ chars\.\s*Full output could not be saved to sandbox\.\]",
    re.IGNORECASE,
)
_HERMES_RESULTS_DIRNAME = "hermes-results"
_MAX_RECOVERED_PERSISTED_OUTPUT_BYTES = 64 * 1024 * 1024

def _is_wrapped_base64_line(line: str) -> bool:
    stripped = line.strip("\r\n")
    return (
        len(stripped) >= _WRAPPED_BASE64_MIN_LINE_CHARS
        and _BASE64_LINE_ALPHABET_RE.fullmatch(stripped) is not None
    )


def _is_wrapped_base64_terminal_line(line: str) -> bool:
    stripped = line.strip("\r\n")
    return (
        _WRAPPED_BASE64_MIN_TERMINAL_LINE_CHARS
        <= len(stripped)
        < _WRAPPED_BASE64_MIN_LINE_CHARS
        and len(stripped) % 4 == 0
        and _BASE64_LINE_ALPHABET_RE.fullmatch(stripped) is not None
    )


def _looks_like_hex_hash_inventory(payload: str) -> bool:
    """Return True for newline inventories of hex digests, not base64 payloads."""
    lines = [line.strip() for line in payload.splitlines() if line.strip()]
    if len(lines) < 2:
        return False
    digest_lengths = {40, 56, 64, 96, 128}
    return all(
        len(line) in digest_lengths and re.fullmatch(r"[0-9a-fA-F]+", line) is not None
        for line in lines
    )


def _iter_wrapped_base64_blocks(text: str):
    """Yield (start, end, payload) for line-wrapped base64 blocks.

    Implemented as a line scanner instead of a wide regex so long
    base64-alphabet single lines that are not actually wrapped do not trigger
    repeated failed block matches.
    """
    offset = 0
    block_start: int | None = None
    block_parts: list[str] = []
    block_end = 0

    def finish_block():
        nonlocal block_start, block_parts, block_end
        if block_start is not None and block_parts:
            payload = "".join(block_parts)
            start, end = block_start, block_end
            block_start = None
            block_parts = []
            block_end = 0
            if not _looks_like_hex_hash_inventory(payload) and looks_like_long_base64(payload):
                return (start, end, payload)
        block_start = None
        block_parts = []
        block_end = 0
        return None

    for line in text.splitlines(keepends=True):
        line_start = offset
        offset += len(line)
        if _is_wrapped_base64_line(line) or (
            block_start is not None
            and block_parts
            and _is_wrapped_base64_terminal_line(line)
        ):
            if block_start is None:
                block_start = line_start
            block_parts.append(line)
            block_end = offset
            continue
        block = finish_block()
        if block is not None:
            yield block
    block = finish_block()
    if block is not None:
        yield block


def _replace_wrapped_base64_blocks(text: str, replace) -> str:
    chunks: list[str] = []
    cursor = 0
    changed = False
    for start, end, payload in _iter_wrapped_base64_blocks(text):
        chunks.append(text[cursor:start])
        chunks.append(replace(payload))
        cursor = end
        changed = True
    if not changed:
        return text
    chunks.append(text[cursor:])
    return "".join(chunks)


def is_externalized_ingest_placeholder(text: str) -> bool:
    return isinstance(text, str) and bool(_INGEST_PLACEHOLDER_RE.fullmatch(text.strip()))


def _is_unrecoverable_tool_truncation_marker(text: str | None) -> bool:
    return isinstance(text, str) and bool(_UNRECOVERABLE_TRUNCATION_RE.search(text))


def _expected_persisted_output_chars(text: str | None) -> int | None:
    if not isinstance(text, str):
        return None
    match = _PERSISTED_OUTPUT_CHAR_COUNT_RE.search(text)
    if not match:
        return None
    try:
        return int(match.group("count").replace(",", ""))
    except ValueError:
        return None


def _persisted_output_preview_prefix(text: str | None) -> str | None:
    if not isinstance(text, str):
        return None
    match = _PERSISTED_OUTPUT_PREVIEW_RE.search(text.strip())
    if not match:
        return None
    preview = match.group("preview")
    if preview.endswith("\r\n..."):
        preview = preview[: -len("\r\n...")]
    elif preview.endswith("\n..."):
        preview = preview[: -len("\n...")]
    return preview


def _persisted_output_preview_prefix_digest(text: str | None) -> str | None:
    preview_prefix = _persisted_output_preview_prefix(text)
    if not preview_prefix:
        return None
    return hashlib.sha256(
        preview_prefix.encode("utf-8", errors="surrogatepass")
    ).hexdigest()


def _persisted_output_inline_preview_sha256(text: str | None) -> str | None:
    if not isinstance(text, str):
        return None
    match = _PERSISTED_OUTPUT_INLINE_PREVIEW_SHA256_RE.search(text)
    if not match:
        return None
    return match.group("sha256")


def _inline_persisted_output_generation_metadata(text: str | None) -> dict[str, int] | None:
    if not isinstance(text, str):
        return None
    match = _PERSISTED_OUTPUT_INLINE_GENERATION_RE.search(text)
    if not match:
        return None
    try:
        return {
            "size": int(match.group("size")),
            "mtime_ns": int(match.group("mtime_ns")),
            "ctime_ns": int(match.group("ctime_ns")),
        }
    except (TypeError, ValueError):
        return None


def _has_inline_persisted_output_generation_metadata(text: str | None) -> bool:
    return _inline_persisted_output_generation_metadata(text) is not None


def _persisted_output_marker_identity_digest(text: str | None) -> str | None:
    return _persisted_output_inline_preview_sha256(text) or _persisted_output_preview_prefix_digest(text)


def _persisted_output_saved_path(text: str | None) -> str | None:
    if not isinstance(text, str):
        return None
    match = _PERSISTED_OUTPUT_SAVED_TO_RE.search(text.strip())
    if not match:
        return None
    raw_path = match.group("path").strip()
    if not raw_path or "\x00" in raw_path:
        return None
    return raw_path


def _safe_temp_hermes_results_file(path: Path) -> Path | None:
    if not path.is_absolute() or path.name in {"", ".", ".."}:
        return None
    parent = path.parent
    if parent.name != _HERMES_RESULTS_DIRNAME:
        return None
    try:
        expected_parent = (Path(tempfile.gettempdir()) / _HERMES_RESULTS_DIRNAME).resolve()
        parent_is_valid_dir = parent.exists() and parent.is_dir() and not parent.is_symlink()
        if not parent_is_valid_dir or parent.resolve() != expected_parent:
            return None
        return expected_parent / path.name
    except OSError:
        return None


def _is_hermes_persisted_output_marker(text: str | None) -> bool:
    if not isinstance(text, str):
        return False
    marker = text.strip()
    return (
        marker.startswith(_PERSISTED_OUTPUT_TAG)
        and marker.endswith(_PERSISTED_OUTPUT_CLOSING_TAG)
        and _expected_persisted_output_chars(marker) is not None
        and _PERSISTED_OUTPUT_SAVED_TO_RE.search(marker) is not None
    )


def _stat_generation_metadata(stats: os.stat_result) -> dict[str, int]:
    return {
        "size": int(stats.st_size),
        "mtime_ns": int(stats.st_mtime_ns),
        "ctime_ns": int(stats.st_ctime_ns),
    }


def _read_regular_file_no_symlink(path: Path) -> tuple[str, dict[str, int]] | None:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    fd: int | None = None
    try:
        lstat_result = os.lstat(str(path))
        if not stat.S_ISREG(lstat_result.st_mode):
            return None
        if lstat_result.st_size > _MAX_RECOVERED_PERSISTED_OUTPUT_BYTES:
            return None
        fd = os.open(str(path), flags)
        stats_before = os.fstat(fd)
        if not stat.S_ISREG(stats_before.st_mode):
            return None
        if stats_before.st_size > _MAX_RECOVERED_PERSISTED_OUTPUT_BYTES:
            return None
        with os.fdopen(fd, "rb") as handle:
            fd = None
            raw = handle.read()
            stats_after = os.fstat(handle.fileno())
        if _stat_generation_metadata(stats_before) != _stat_generation_metadata(stats_after):
            return None
        return raw.decode("utf-8"), _stat_generation_metadata(stats_after)
    except (OSError, UnicodeDecodeError):
        return None
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass


def recover_hermes_persisted_output_with_file_stat(text: str | None) -> tuple[str, dict[str, int]] | None:
    """Recover Hermes host `<persisted-output>` content when the backing file is safe.

    Recovery is intentionally conservative: the marker must include Hermes'
    character count, the file path must be an absolute basename under a
    `hermes-results` temp directory, the target must be a regular non-symlink
    file, and the recovered character count must match the marker. If any check
    fails, callers should keep the marker/preview instead of claiming lossless
    recovery from an unsafe or stale file.
    """
    if not isinstance(text, str) or not _is_hermes_persisted_output_marker(text):
        return None
    expected_chars = _expected_persisted_output_chars(text)
    if expected_chars is None:
        return None
    raw_path = _persisted_output_saved_path(text)
    if raw_path is None:
        return None
    path = Path(raw_path)
    safe_path = _safe_temp_hermes_results_file(path)
    if safe_path is None:
        return None
    recovered_with_stat = _read_regular_file_no_symlink(safe_path)
    if recovered_with_stat is None:
        return None
    recovered, file_stat = recovered_with_stat
    if len(recovered) != expected_chars:
        return None
    preview_prefix = _persisted_output_preview_prefix(text)
    if not preview_prefix or not recovered.startswith(preview_prefix):
        return None
    return recovered, file_stat


def recover_hermes_persisted_output(text: str | None) -> str | None:
    recovered_with_stat = recover_hermes_persisted_output_with_file_stat(text)
    if recovered_with_stat is None:
        return None
    recovered, _file_stat = recovered_with_stat
    return recovered


def _add_inline_persisted_output_generation_metadata(text: str, file_stat: dict[str, int] | None) -> str:
    if not file_stat or not isinstance(text, str) or "</persisted-output>" not in text:
        return text
    generation = (
        "[LCM persisted-output file generation: "
        f"size={file_stat['size']}; "
        f"mtime_ns={file_stat['mtime_ns']}; "
        f"ctime_ns={file_stat['ctime_ns']}]"
    )
    if generation in text:
        return text
    return text.replace("</persisted-output>", f"{generation}\n</persisted-output>", 1)


def _add_inline_persisted_output_identity_metadata(text: str, preview_sha256: str | None) -> str:
    if (
        not isinstance(text, str)
        or "</persisted-output>" not in text
        or not isinstance(preview_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", preview_sha256)
    ):
        return text
    if _persisted_output_inline_preview_sha256(text):
        return text
    identity = f"[LCM persisted-output marker identity: preview_sha256={preview_sha256}]"
    return text.replace("</persisted-output>", f"{identity}\n</persisted-output>", 1)


def extract_ingest_externalized_refs(text: str) -> list[str]:
    if not isinstance(text, str) or not text:
        return []
    refs: list[str] = []
    for match in _INGEST_PLACEHOLDER_RE.finditer(text):
        ref = match.group(1).strip()
        if ref and ref not in refs:
            refs.append(ref)
    return refs


def _is_basename_ref(ref: str) -> bool:
    return bool(ref) and ref.endswith(".json") and "/" not in ref and "\\" not in ref and Path(ref).name == ref


def extract_all_externalized_payload_refs(text: str) -> list[str]:
    """Return deduplicated refs from recognized externalized payload placeholders."""
    if not isinstance(text, str) or not text:
        return []
    refs: list[str] = []
    for ref in extract_ingest_externalized_refs(text):
        if _is_basename_ref(ref) and ref not in refs:
            refs.append(ref)
    return refs


def _safe_placeholder_metadata(value: Any) -> str:
    text = str(value or "?")
    safe = re.sub(r"[^A-Za-z0-9_.:/-]+", "-", text).strip("-")
    return (safe or "?")[:120]


def _normalized_repetition_segments(text: str) -> list[str]:
    segments = []
    for segment in _REPETITION_SEGMENT_SPLIT_RE.split(text):
        normalized = re.sub(r"\s+", " ", segment.strip().lower())
        if len(normalized) >= 32:
            segments.append(normalized)
    return segments


def assistant_output_quarantine_reason(text: str) -> str | None:
    """Return a quarantine reason for obviously broken assistant output.

    The gate is intentionally conservative: content must be very large and show
    both low token novelty and repeated sentence/line segments. Long diverse
    reports and code with varied identifiers should stay inline.
    """
    if not isinstance(text, str) or len(text) < _QUARANTINED_ASSISTANT_MIN_CHARS:
        return None

    normalized = re.sub(r"\s+", " ", text.strip().lower())
    tokens = _WORD_TOKEN_RE.findall(normalized)
    if len(tokens) < _QUARANTINED_ASSISTANT_MIN_TOKENS:
        if len(normalized) >= _QUARANTINED_ASSISTANT_MIN_CHARS and len(set(normalized)) <= 12:
            return _QUARANTINED_ASSISTANT_REASON
        return None

    token_counts = Counter(tokens)
    unique_token_ratio = len(token_counts) / max(1, len(tokens))
    top_token_ratio = token_counts.most_common(1)[0][1] / max(1, len(tokens))

    segments = _normalized_repetition_segments(text)
    top_segment_ratio = 0.0
    duplicate_segment_ratio = 0.0
    if len(segments) >= 20:
        segment_counts = Counter(segments)
        top_segment_ratio = segment_counts.most_common(1)[0][1] / len(segments)
        duplicate_segment_ratio = 1.0 - (len(segment_counts) / len(segments))

    if unique_token_ratio <= 0.03 and (
        top_segment_ratio >= 0.10
        or duplicate_segment_ratio >= 0.50
        or top_token_ratio >= 0.08
    ):
        return _QUARANTINED_ASSISTANT_REASON

    # Covers degenerate long loops with little punctuation/newline structure.
    if unique_token_ratio <= 0.015 and len(set(normalized)) <= 64:
        return _QUARANTINED_ASSISTANT_REASON

    return None


def _quarantined_assistant_placeholder(summary: Dict[str, Any], *, reason: str) -> str:
    return (
        "[Externalized LCM ingest payload: assistant output quarantined; "
        f"kind={_safe_placeholder_metadata(summary.get('kind') or _QUARANTINED_ASSISTANT_KIND)}; "
        f"reason={_safe_placeholder_metadata(reason)}; "
        f"field={_safe_placeholder_metadata(summary.get('field_path') or 'content')}; "
        f"chars={summary.get('content_chars', 0)}; bytes={summary.get('content_bytes', 0)}; "
        f"ref={summary.get('ref', '')}]"
    )


def _volatile_quarantined_assistant_placeholder(content: str, *, reason: str) -> str:
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
    return (
        "[LCM active replay placeholder: assistant output quarantined; "
        f"kind={_QUARANTINED_ASSISTANT_KIND}; "
        f"reason={_safe_placeholder_metadata(reason)}; "
        "scope=ignored_message_pattern; field=content; "
        f"chars={len(content)}; bytes={len(content.encode('utf-8'))}; "
        f"sha256={digest}]"
    )


def _externalize_quarantined_assistant_output(
    content: str,
    *,
    role: str,
    session_id: str,
    config,
    hermes_home: str,
    reason: str,
) -> str | None:
    existing = find_externalized_payload_for_message(
        content,
        session_id=session_id,
        kind=_QUARANTINED_ASSISTANT_KIND,
        role=role,
        config=config,
        hermes_home=hermes_home,
    )
    if existing is not None:
        return _quarantined_assistant_placeholder(existing, reason=reason)

    result = externalize_ingest_payload(
        content,
        role=role,
        session_id=session_id,
        field_path="content",
        config=config,
        hermes_home=hermes_home,
        kind=_QUARANTINED_ASSISTANT_KIND,
    )
    if result is None:
        logger.warning(
            "LCM ingest protection could not quarantine repetitive assistant output; preserving inline content for lossless recovery"
        )
        return None

    payload = result.get("payload") or {}
    path = result.get("path")
    summary = {
        "ref": getattr(path, "name", ""),
        "kind": payload.get("kind", _QUARANTINED_ASSISTANT_KIND),
        "role": payload.get("role", role),
        "field_path": payload.get("field_path", "content"),
        "content_chars": payload.get("content_chars", len(content)),
        "content_bytes": payload.get("content_bytes", len(content.encode("utf-8"))),
    }
    return _quarantined_assistant_placeholder(summary, reason=reason)


def _existing_quarantined_assistant_placeholder(
    content: str,
    *,
    role: str,
    session_id: str,
    config,
    hermes_home: str,
    reason: str,
) -> str | None:
    existing = find_externalized_payload_for_message(
        content,
        session_id=session_id,
        kind=_QUARANTINED_ASSISTANT_KIND,
        role=role,
        config=config,
        hermes_home=hermes_home,
    )
    if existing is None:
        return None
    return _quarantined_assistant_placeholder(existing, reason=reason)


def restore_ingest_payload_placeholders(
    text: str,
    *,
    config,
    hermes_home: str = "",
    session_id: str = "",
) -> str:
    """Restore ingest placeholders in a stored identity string for matching only.

    Missing or mismatched payload files leave the placeholder untouched so callers
    never fabricate content or hide a recovery problem.
    """
    if not isinstance(text, str) or _EXTERNALIZED_PLACEHOLDER_PREFIX not in text:
        return text

    def replace(match: re.Match[str]) -> str:
        ref = match.group(1).strip()
        payload = load_externalized_payload(ref, config=config, hermes_home=hermes_home)
        if payload is None or payload.get("kind") != "ingest_payload":
            return match.group(0)
        payload_session_id = payload.get("session_id") or ""
        if session_id and payload_session_id and payload_session_id != session_id:
            return match.group(0)
        content = payload.get("content")
        return content if isinstance(content, str) else match.group(0)

    return _INGEST_PLACEHOLDER_RE.sub(replace, text)


def looks_like_long_base64(text: str, *, min_chars: int = _GENERIC_BASE64_MIN_CHARS) -> bool:
    """Conservative long-base64 heuristic.

    Avoids short hashes/IDs/JWT-ish snippets by requiring a very long run and a
    high base64 alphabet ratio. PEM blocks and ordinary logs contain delimiters
    or whitespace/headers that keep them from matching as one clean run.
    """
    if not isinstance(text, str) or len(text) < min_chars:
        return False
    compact = "".join(text.split())
    if len(compact) < min_chars:
        return False
    if len(compact) % 4 == 1:
        return False
    if not _BASE64_ALPHABET_RE.match(text):
        return False
    # Compute the base64 density over the whitespace-stripped content, not the
    # raw text: otherwise line-ending overhead sinks the ratio and canonical
    # CRLF-wrapped MIME (76/78 = 0.974) and PEM (64/66 = 0.970) blocks fall
    # below 0.98 and are wrongly left inline.
    base64_chars = sum(1 for ch in compact if ch.isalnum() or ch in "+/=_-")
    ratio = base64_chars / max(1, len(compact))
    if ratio < 0.98:
        return False
    # Require at least a bit of mixed alphabet so a long log line of one
    # repeated character is not treated as a binary payload.
    return len(set(compact.rstrip("="))) >= 8


def _placeholder_for_payload(
    payload: str,
    *,
    role: str,
    session_id: str,
    field_path: str,
    config,
    hermes_home: str,
) -> str | None:
    result = externalize_ingest_payload(
        payload,
        role=role,
        session_id=session_id,
        field_path=field_path,
        config=config,
        hermes_home=hermes_home,
    )
    if result is None:
        logger.warning(
            "LCM ingest protection could not externalize payload at %s; preserving inline content for lossless recovery",
            field_path,
        )
        return None
    return result["placeholder"]


def _protect_payload_substrings(
    text: str,
    *,
    role: str,
    session_id: str,
    field_path: str,
    config,
    hermes_home: str,
) -> str:
    if not text or is_externalized_ingest_placeholder(text):
        return text

    def replace_data_uri(match: re.Match[str]) -> str:
        payload = match.group(0)
        return _placeholder_for_payload(
            payload,
            role=role,
            session_id=session_id,
            field_path=field_path,
            config=config,
            hermes_home=hermes_home,
        ) or payload

    protected = _DATA_URI_BASE64_RE.sub(replace_data_uri, text)

    def replace_base64_run(match: re.Match[str]) -> str:
        payload = match.group(1)
        if not looks_like_long_base64(payload):
            return payload
        return _placeholder_for_payload(
            payload,
            role=role,
            session_id=session_id,
            field_path=field_path,
            config=config,
            hermes_home=hermes_home,
        ) or payload

    protected = _BASE64_RUN_RE.sub(replace_base64_run, protected)

    def replace_wrapped_base64(payload: str) -> str:
        return _placeholder_for_payload(
            payload,
            role=role,
            session_id=session_id,
            field_path=field_path,
            config=config,
            hermes_home=hermes_home,
        ) or payload

    # Line-wrapped base64 (MIME/PEM) is not a single contiguous run; externalize
    # it here too so it does not land inline in SQLite/FTS/WAL/backups.
    return _replace_wrapped_base64_blocks(protected, replace_wrapped_base64)


def _maybe_parse_json_string(text: str) -> Any | None:
    stripped = text.strip()
    if not stripped or stripped[0] not in "[{":
        return None
    candidates = [text]
    if '\\"' in stripped:
        candidates.append(stripped.replace('\\"', '"'))
    for candidate in candidates:
        if _json_has_duplicate_object_keys(candidate):
            return None
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, (dict, list)):
            return parsed
    return None


def _json_has_duplicate_object_keys(text: str) -> bool:
    stripped = text.strip() if isinstance(text, str) else ""
    if not stripped or stripped[0] not in "[{":
        return False
    duplicate = False

    def detect_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        nonlocal duplicate
        seen: set[str] = set()
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in seen:
                duplicate = True
            seen.add(key)
            result[key] = value
        return result

    try:
        json.loads(text, object_pairs_hook=detect_pairs)
    except Exception:
        return False
    return duplicate


def _dict_field_path(parent: str, key: Any) -> str:
    component = str(key)
    return f"{parent}.{component}" if parent else component


def _payload_key_field_path(parent: str) -> str:
    return f"{parent}.<key>" if parent else "<key>"


def _protect_value(
    value: Any,
    *,
    role: str,
    session_id: str,
    field_path: str,
    config,
    hermes_home: str,
    parse_json_strings: bool = False,
) -> Any:
    if isinstance(value, dict):
        protected: dict[Any, Any] = {}
        for key, val in value.items():
            protected_key = (
                _protect_payload_substrings(
                    key,
                    role=role,
                    session_id=session_id,
                    field_path=_payload_key_field_path(field_path),
                    config=config,
                    hermes_home=hermes_home,
                )
                if isinstance(key, str)
                else key
            )
            child_path_key = "<key>" if protected_key != key else protected_key
            protected[protected_key] = _protect_value(
                val,
                role=role,
                session_id=session_id,
                field_path=_dict_field_path(field_path, child_path_key),
                config=config,
                hermes_home=hermes_home,
                parse_json_strings=parse_json_strings,
            )
        return protected
    if isinstance(value, list):
        return [
            _protect_value(
                item,
                role=role,
                session_id=session_id,
                field_path=f"{field_path}[{idx}]",
                config=config,
                hermes_home=hermes_home,
                parse_json_strings=parse_json_strings,
            )
            for idx, item in enumerate(value)
        ]
    if not isinstance(value, str):
        return value

    if parse_json_strings:
        if _json_has_duplicate_object_keys(value):
            return _protect_payload_substrings(
                value,
                role=role,
                session_id=session_id,
                field_path=field_path,
                config=config,
                hermes_home=hermes_home,
            )
        parsed = _maybe_parse_json_string(value)
        if parsed is not None:
            canonical = json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
            if canonical != value:
                raw_protected = _protect_payload_substrings(
                    value,
                    role=role,
                    session_id=session_id,
                    field_path=field_path,
                    config=config,
                    hermes_home=hermes_home,
                )
                if raw_protected != value:
                    return raw_protected
            protected = _protect_value(
                parsed,
                role=role,
                session_id=session_id,
                field_path=field_path,
                config=config,
                hermes_home=hermes_home,
                parse_json_strings=True,
            )
            if protected != parsed:
                return json.dumps(protected, ensure_ascii=False, separators=(",", ":"))
            return value

    return _protect_payload_substrings(
        value,
        role=role,
        session_id=session_id,
        field_path=field_path,
        config=config,
        hermes_home=hermes_home,
    )


def protect_inline_payloads_in_text(
    text: str,
    *,
    role: str,
    session_id: str,
    field_path: str,
    config,
    hermes_home: str,
) -> str:
    """Externalize inline media/base64 payloads inside a text scaffold.

    This is used for non-SQLite active-context scaffolds that still must not
    duplicate media-ish payloads into summaries or preserved objective text.
    """
    if not isinstance(text, str):
        return text
    return _protect_payload_substrings(
        text,
        role=role,
        session_id=session_id,
        field_path=field_path,
        config=config,
        hermes_home=hermes_home,
    )


def _protect_tool_calls(tool_calls: Any, *, role: str, session_id: str, config, hermes_home: str) -> Any:
    return _protect_value(
        tool_calls,
        role=role,
        session_id=session_id,
        field_path="tool_calls",
        config=config,
        hermes_home=hermes_home,
        parse_json_strings=True,
    )


def protect_message_for_ingest(
    message: Dict[str, Any],
    config,
    hermes_home: str = "",
    session_id: str = "",
) -> Dict[str, Any]:
    """Return a copy of ``message`` safe to persist in SQLite.

    Inline media/base64-like strings are moved to side files before they hit
    ``messages.content`` or ``messages.tool_calls``.
    """
    msg = dict(message or {})
    role = str(msg.get("role") or "unknown")
    raw_content = msg.get("content")
    raw_normalized_content = normalize_content_value(raw_content)
    original_content = raw_content
    normalized_content = normalize_content_value(original_content)
    recovered_with_stat = recover_hermes_persisted_output_with_file_stat(raw_normalized_content) if role == "tool" else None
    recovered_file_stat = None
    if recovered_with_stat is not None:
        _recovered_persisted_output, recovered_file_stat = recovered_with_stat

    # A host-side truncation marker stays visible inline.
    preserve_truncation_marker_inline = (
        role == "tool"
        and isinstance(normalized_content, str)
        and (
            _is_hermes_persisted_output_marker(normalized_content)
            or _is_unrecoverable_tool_truncation_marker(normalized_content)
        )
    )

    if normalized_content:
        if is_externalized_ingest_placeholder(normalized_content):
            msg["content"] = original_content
        elif preserve_truncation_marker_inline:
            protected_content = _protect_value(
                original_content,
                role=role,
                session_id=session_id,
                field_path="content",
                config=config,
                hermes_home=hermes_home,
                parse_json_strings=False,
            )
            if role == "tool" and _is_hermes_persisted_output_marker(raw_normalized_content):
                protected_content = _add_inline_persisted_output_identity_metadata(
                    normalize_content_value(protected_content) or "",
                    _persisted_output_marker_identity_digest(raw_normalized_content),
                )
            if recovered_with_stat is not None and _is_hermes_persisted_output_marker(normalized_content):
                protected_content = _add_inline_persisted_output_generation_metadata(
                    normalize_content_value(protected_content) or "",
                    recovered_file_stat,
                )
            msg["content"] = protected_content
        else:
            reason = (
                assistant_output_quarantine_reason(normalized_content)
                if role == "assistant"
                else None
            )
            externalized = None
            if reason:
                placeholder = _externalize_quarantined_assistant_output(
                    normalized_content,
                    role=role,
                    session_id=session_id,
                    config=config,
                    hermes_home=hermes_home,
                    reason=reason,
                )
                if placeholder:
                    externalized = {"placeholder": placeholder}
            if externalized:
                msg["content"] = externalized["placeholder"]
            else:
                msg["content"] = _protect_value(
                    original_content,
                    role=role,
                    session_id=session_id,
                    field_path="content",
                    config=config,
                    hermes_home=hermes_home,
                    parse_json_strings=False,
                )
    else:
        msg["content"] = original_content

    if msg.get("tool_calls"):
        msg["tool_calls"] = _protect_tool_calls(
            msg.get("tool_calls"),
            role=role,
            session_id=session_id,
            config=config,
            hermes_home=hermes_home,
        )

    return msg


def quarantine_suspicious_assistant_message(
    message: Dict[str, Any],
    config,
    hermes_home: str = "",
    session_id: str = "",
    *,
    externalize: bool = True,
    prefer_existing_externalized: bool = False,
) -> Dict[str, Any]:
    """Return ``message`` with obviously broken assistant output quarantined.

    Unlike full ingest protection, this only touches suspicious assistant text.
    It is safe for active-context replay because it does not externalize user
    media, tool results, or ordinary long content.
    """
    msg = dict(message or {})
    role = str(msg.get("role") or "unknown")
    if role != "assistant":
        return msg
    normalized_content = normalize_content_value(msg.get("content"))
    reason = assistant_output_quarantine_reason(normalized_content)
    if not reason:
        return msg
    if externalize:
        placeholder = _externalize_quarantined_assistant_output(
            normalized_content,
            role=role,
            session_id=session_id,
            config=config,
            hermes_home=hermes_home,
            reason=reason,
        )
    else:
        placeholder = None
        if prefer_existing_externalized:
            placeholder = _existing_quarantined_assistant_placeholder(
                normalized_content,
                role=role,
                session_id=session_id,
                config=config,
                hermes_home=hermes_home,
                reason=reason,
            )
        if placeholder is None:
            placeholder = _volatile_quarantined_assistant_placeholder(
                normalized_content,
                reason=reason,
            )
    if not placeholder:
        return msg
    msg["content"] = placeholder
    return msg


def quarantine_suspicious_assistant_messages(
    messages: List[Dict[str, Any]],
    config,
    hermes_home: str = "",
    session_id: str = "",
    externalize: List[bool] | None = None,
    prefer_existing_externalized: List[bool] | None = None,
) -> List[Dict[str, Any]]:
    return [
        quarantine_suspicious_assistant_message(
            message,
            config=config,
            hermes_home=hermes_home,
            session_id=session_id,
            externalize=True if externalize is None else externalize[idx],
            prefer_existing_externalized=False
            if prefer_existing_externalized is None
            else prefer_existing_externalized[idx],
        )
        for idx, message in enumerate(messages)
    ]


def protect_messages_for_ingest(
    messages: List[Dict[str, Any]],
    config,
    hermes_home: str = "",
    session_id: str = "",
) -> List[Dict[str, Any]]:
    return [
        protect_message_for_ingest(
            message,
            config=config,
            hermes_home=hermes_home,
            session_id=session_id,
        )
        for message in messages
    ]


