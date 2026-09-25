"""Readers for ingest placeholders, which the tools still call (E's to settle).

The old ingest replaced inline payloads with "[Externalized LCM ingest payload: …;
ref=<file>]" placeholders and wrote the payload to a side file. The record writes
nothing of the kind; what remains here are the readers the tools and the doctor
still use: finding placeholder refs in a text, and comparing the refs stored in
messages with the side files on disk.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .externalize import get_large_output_storage_dir

_QUARANTINED_ASSISTANT_KIND = "quarantined_assistant_output"
_INGEST_PLACEHOLDER_RE = re.compile(r"\[Externalized LCM ingest payload:.*?;\s*ref=([^;\]\s]+)\]")


def is_externalized_ingest_placeholder(text: str) -> bool:
    return isinstance(text, str) and bool(_INGEST_PLACEHOLDER_RE.fullmatch(text.strip()))


def extract_ingest_externalized_refs(text: str) -> list[str]:
    if not isinstance(text, str) or not text:
        return []
    refs: list[str] = []
    for match in _INGEST_PLACEHOLDER_RE.finditer(text):
        ref = match.group(1).strip()
        if ref and ref not in refs:
            refs.append(ref)
    return refs


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


# Side files the old ingest wrote were named "<YYYYmmdd_HHMMSS>_<kind>_...json"
# for these two kinds.
_INGEST_SIDE_FILE_NAME_RE = re.compile(
    r"^\d{8}_\d{6}_(?:ingest_payload|" + re.escape(_QUARANTINED_ASSISTANT_KIND) + r")_.*\.json$"
)


def _is_basename_ref(ref: str) -> bool:
    return bool(ref) and ref.endswith(".json") and "/" not in ref and "\\" not in ref and Path(ref).name == ref


def _append_unique_refs(target: list[str], refs: list[str]) -> None:
    for ref in refs:
        if ref not in target:
            target.append(ref)


def _walk_string_values(value: Any):
    if isinstance(value, str):
        yield value
        parsed = _maybe_parse_json_string(value)
        if parsed is not None and not (isinstance(parsed, str) and parsed == value):
            yield from _walk_string_values(parsed)
    elif isinstance(value, dict):
        for key, nested in value.items():
            if isinstance(key, str):
                yield key
            yield from _walk_string_values(nested)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_string_values(item)


def _walk_tool_call_argument_values(value: Any):
    if isinstance(value, dict):
        for key, nested in value.items():
            if key == "arguments":
                yield nested
            yield from _walk_tool_call_argument_values(nested)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_tool_call_argument_values(item)


def _is_inside_token_quote_span(text: str, start: int, token: str) -> bool:
    in_span = False
    i = 0
    while i < start:
        if text.startswith(token, i):
            in_span = not in_span
            i += len(token)
        else:
            i += 1
    return in_span


def _looks_like_example_quote_context(context: str) -> bool:
    return re.search(r"(?:pytest\s+output|log|example|traceback|failure)\s*:\s*$", context.lower()) is not None


def _has_local_escaped_quote_before(text: str, start: int) -> bool:
    boundary = max(text.rfind(delimiter, 0, start) for delimiter in (",", "{", "["))
    segment = text[boundary + 1:start]
    matches = list(re.finditer(r"\\+[\"']", segment))
    if not matches:
        return False
    quote = matches[-1]
    context = segment[max(0, quote.start() - 80):quote.start()]
    return _looks_like_example_quote_context(context)


def _is_escaped_placeholder_example(text: str, start: int) -> bool:
    prefix = text[max(0, start - 8):start]
    return prefix.endswith("\\") or _has_local_escaped_quote_before(text, start)


def _is_quoted_placeholder_example(text: str, start: int) -> bool:
    for quote_token in ('"', "'"):
        if not _is_inside_token_quote_span(text, start, quote_token):
            continue
        quote = text.rfind(quote_token, 0, start)
        if quote < 0:
            continue
        context = text[max(0, quote - 80):quote]
        if _looks_like_example_quote_context(context):
            return True
    return False


def _looks_like_example_payload_ref(ref: str) -> bool:
    name = Path(ref).name.lower()
    return name.startswith(("example-", "example_", "fake-", "fake_", "dummy-", "dummy_", "placeholder-", "placeholder_"))


def _extract_unescaped_ingest_payload_refs(text: str, *, ignore_quoted_spans: bool = False) -> list[str]:
    refs: list[str] = []
    for match in _INGEST_PLACEHOLDER_RE.finditer(text):
        ref = match.group(1).strip()
        if not _is_basename_ref(ref):
            continue
        if _looks_like_example_payload_ref(ref) and _is_escaped_placeholder_example(text, match.start()):
            continue
        if (
            ignore_quoted_spans
            and _looks_like_example_payload_ref(ref)
            and _is_quoted_placeholder_example(text, match.start())
        ):
            continue
        if ref not in refs:
            refs.append(ref)
    return refs


def _nested_ingest_refs(value: str) -> list[str]:
    stripped = value.strip()
    if is_externalized_ingest_placeholder(stripped):
        return extract_ingest_externalized_refs(stripped)
    return _extract_unescaped_ingest_payload_refs(value, ignore_quoted_spans=True)


def _refs_for_ingest_integrity_scan(value: str, *, role: str, field: str) -> list[str]:
    """Return ingest side-file refs that plausibly came from ingest placeholders.

    Tool outputs and tool-call arguments often contain escaped code snippets,
    pytest failures, or docs that mention placeholder examples; those are not
    counted. Exact placeholders are counted everywhere; embedded unescaped
    placeholders are counted in message content and in tool-call argument
    strings.
    """
    if not isinstance(value, str) or not value:
        return []
    stripped = value.strip()
    if is_externalized_ingest_placeholder(stripped):
        return extract_ingest_externalized_refs(stripped)
    if field == "tool_calls":
        refs = _extract_unescaped_ingest_payload_refs(value, ignore_quoted_spans=True)
        parsed = _maybe_parse_json_string(value)
        if parsed is None:
            return refs
        for argument in _walk_tool_call_argument_values(parsed):
            if isinstance(argument, str):
                _append_unique_refs(refs, _extract_unescaped_ingest_payload_refs(argument, ignore_quoted_spans=True))
                parsed_argument = _maybe_parse_json_string(argument)
                if parsed_argument is not None:
                    for nested in _walk_string_values(parsed_argument):
                        _append_unique_refs(refs, _nested_ingest_refs(nested))
            else:
                for nested in _walk_string_values(argument):
                    _append_unique_refs(refs, _nested_ingest_refs(nested))
        for nested in _walk_string_values(parsed):
            _append_unique_refs(refs, _nested_ingest_refs(nested))
        return refs
    if role == "tool":
        refs = _extract_unescaped_ingest_payload_refs(value)
        parsed = _maybe_parse_json_string(value)
        if parsed is not None:
            for nested in _walk_string_values(parsed):
                _append_unique_refs(refs, _nested_ingest_refs(nested))
        return refs
    return extract_ingest_externalized_refs(value)


def scan_ingest_side_file_integrity(conn, config, *, hermes_home: str = "", limit: int = 5) -> dict[str, Any]:
    """Compare the ingest side-file refs stored in messages with the files on disk.

    Covers the files ingest writes for base64 payloads and quarantined
    assistant output. Read-only and metadata-only: payload files are not
    opened, and row samples never include message content or tool-call
    arguments.
    """
    storage_dir = get_large_output_storage_dir(config, hermes_home=hermes_home, create=False)
    existing_files: set[str] = set()
    if storage_dir.exists() and storage_dir.is_dir():
        existing_files = {
            path.name
            for path in storage_dir.glob("*.json")
            if path.is_file() and _INGEST_SIDE_FILE_NAME_RE.match(path.name)
        }

    referenced_refs: set[str] = set()
    first_location_by_ref: dict[str, dict[str, Any]] = {}
    for store_id, session_id, source, role, content, tool_calls in conn.execute(
        """
        SELECT store_id, session_id, source, role, content, tool_calls
        FROM messages
        WHERE COALESCE(content, '') LIKE '%ref=%]%'
           OR COALESCE(tool_calls, '') LIKE '%ref=%]%'
        ORDER BY store_id ASC
        """
    ).fetchall():
        for field, value in (("content", content), ("tool_calls", tool_calls)):
            if not isinstance(value, str):
                continue
            for ref in _refs_for_ingest_integrity_scan(value, role=str(role or ""), field=field):
                referenced_refs.add(ref)
                first_location_by_ref.setdefault(
                    ref,
                    {
                        "store_id": int(store_id),
                        "session_id": session_id,
                        "source": source,
                        "role": role,
                        "field": field,
                        "externalized_ref": ref,
                    },
                )

    missing_refs = sorted(ref for ref in referenced_refs if ref not in existing_files)
    existing_ref_count = sum(1 for ref in referenced_refs if ref in existing_files)
    unreferenced_files = sorted(ref for ref in existing_files if ref not in referenced_refs)

    return {
        "externalized_payload_refs_total": len(referenced_refs),
        "externalized_payload_refs_existing": existing_ref_count,
        "externalized_payload_refs_missing": len(missing_refs),
        "externalized_payload_files_unreferenced": len(unreferenced_files),
        "missing_externalized_payload_refs": [
            first_location_by_ref[ref] for ref in missing_refs[:limit] if ref in first_location_by_ref
        ],
        "unreferenced_externalized_payload_files": [
            {"externalized_ref": ref} for ref in unreferenced_files[:limit]
        ],
    }
