"""Readers for the side files the old ingest path wrote, which the tools still call
(E's to settle).

The old ingest moved embedded base64 and quarantined assistant output into JSON
side files beside the store and left a placeholder in the message. The record
writes nothing of the kind; these helpers locate and read the side files for the
tools' externalized-payload expansion and the doctor.
"""

from __future__ import annotations

import codecs
import errno
import json
import logging
import os
from pathlib import Path
from typing import Any, BinaryIO, Dict

DEFAULT_LARGE_OUTPUT_DIRNAME = "lcm-large-outputs"
_SESSION_ID_VALUE_SPAN_KEY = "_session_id_value_span"

logger = logging.getLogger(__name__)


_UNSUPPORTED_FILESYSTEM_ERRNOS = {
    value
    for value in (
        getattr(errno, "ENOSYS", None),
        getattr(errno, "ENOTSUP", None),
        getattr(errno, "EOPNOTSUPP", None),
    )
    if value is not None
}


def _is_unsupported_filesystem_capability(
    exc: BaseException,
    *,
    windows_directory_access: bool = False,
) -> bool:
    if isinstance(exc, (TypeError, NotImplementedError)):
        return True
    if not isinstance(exc, OSError):
        return False
    if exc.errno in _UNSUPPORTED_FILESYSTEM_ERRNOS:
        return True
    return windows_directory_access and os.name == "nt" and isinstance(exc, PermissionError)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    try:
        dir_fd = os.open(path, flags)
    except (OSError, TypeError, NotImplementedError) as exc:
        if _is_unsupported_filesystem_capability(
            exc,
            windows_directory_access=True,
        ):
            return
        raise
    try:
        try:
            os.fsync(dir_fd)
        except (OSError, TypeError, NotImplementedError) as exc:
            if not _is_unsupported_filesystem_capability(
                exc,
                windows_directory_access=True,
            ):
                raise
    finally:
        os.close(dir_fd)


def _missing_directory_components(path: Path) -> list[Path]:
    missing: list[Path] = []
    current = path
    while not current.exists():
        missing.append(current)
        parent = current.parent
        if parent == current:
            break
        current = parent
    return list(reversed(missing))


_WARNED_EXTERNALIZATION_PATHS: set[str] = set()


def _warn_externalization_path_outside_base(path: Path, allowed_base: Path) -> None:
    key = str(path)
    if key in _WARNED_EXTERNALIZATION_PATHS:
        return
    _WARNED_EXTERNALIZATION_PATHS.add(key)
    logger.warning(
        "LCM externalized-payload path %s is outside the hermes_home base %s; "
        "set LCM_HERMES_BASE_DIR to enforce strict containment",
        path,
        allowed_base,
    )


def get_large_output_storage_dir(config, hermes_home: str = "", *, create: bool) -> Path:
    configured = getattr(config, "large_output_externalization_path", "") or ""
    if configured:
        path = Path(configured).expanduser().resolve()
        # Check containment for configured paths when LCM_HERMES_BASE_DIR is set
        env_base = os.environ.get("LCM_HERMES_BASE_DIR")
        if env_base:
            allowed_base = Path(env_base).expanduser().resolve()
            try:
                path.relative_to(allowed_base)
            except ValueError:
                raise ValueError(f"Path {path} is not within allowed base {allowed_base}")
        elif hermes_home:
            # No explicit base configured: hermes_home is the natural default
            # containment root. A configured path may legitimately point to
            # another volume, so warn (once) rather than break a running
            # deployment; set LCM_HERMES_BASE_DIR to enforce strictly.
            allowed_base = Path(hermes_home).expanduser().resolve()
            try:
                path.relative_to(allowed_base)
            except ValueError:
                _warn_externalization_path_outside_base(path, allowed_base)
    else:
        base = Path(hermes_home).expanduser().resolve() if hermes_home else Path("~/.hermes").expanduser().resolve()
        path = base / DEFAULT_LARGE_OUTPUT_DIRNAME
        # Check containment within allowed base for default/hermes_home-based paths
        # Only enforced when LCM_HERMES_BASE_DIR is explicitly set
        env_base = os.environ.get("LCM_HERMES_BASE_DIR")
        if env_base:
            allowed_base = Path(env_base).expanduser().resolve()
            try:
                path.relative_to(allowed_base)
            except ValueError:
                raise ValueError(f"Path {path} is not within allowed base {allowed_base}")
    if create:
        missing_dirs = _missing_directory_components(path)
        path.mkdir(parents=True, exist_ok=True)
        for created_dir in missing_dirs:
            _fsync_directory(created_dir.parent)
        try:
            path.chmod(0o700)
        except OSError as exc:
            logger.warning("Could not restrict LCM externalized payload directory permissions for %s: %s", path, exc)
    return path


def _externalized_summary(path: Path, payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "ref": path.name,
        "kind": payload.get("kind", "tool_result"),
        "tool_call_id": payload.get("tool_call_id", ""),
        "role": payload.get("role", ""),
        "session_id": payload.get("session_id", ""),
        "field_path": payload.get("field_path", ""),
        "content_chars": payload.get("content_chars", len(payload.get("content", ""))),
        "content_bytes": payload.get("content_bytes", len((payload.get("content", "") or "").encode("utf-8"))),
        "created_at": payload.get("created_at"),
    }


def load_externalized_payload(ref: str, *, config, hermes_home: str = "") -> Dict[str, Any] | None:
    if not ref or Path(ref).name != ref:
        return None
    storage_dir = get_large_output_storage_dir(config, hermes_home=hermes_home, create=False)
    if not storage_dir.exists() or not storage_dir.is_dir():
        return None
    path = storage_dir / ref
    if not path.exists() or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    summary = _externalized_summary(path, payload)
    summary["content"] = payload.get("content", "")
    return summary


def _parse_top_level_json_string_fields_before_content(text: str) -> tuple[dict[str, Any], int | None]:
    decoder = json.JSONDecoder()
    fields: dict[str, Any] = {}
    length = len(text)
    index = 0

    def skip_json_whitespace(pos: int) -> int:
        while pos < length and text[pos] in " \t\n\r":
            pos += 1
        return pos

    index = skip_json_whitespace(index)
    if index >= length or text[index] != "{":
        return fields, None
    index += 1

    while True:
        index = skip_json_whitespace(index)
        if index >= length or text[index] == "}":
            return fields, None
        if text[index] != '"':
            return fields, None
        try:
            key, index = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            return fields, None
        if not isinstance(key, str):
            return fields, None
        index = skip_json_whitespace(index)
        if index >= length or text[index] != ":":
            return fields, None
        index += 1
        index = skip_json_whitespace(index)
        if key == "content":
            return fields, index if index < length and text[index] == '"' else None
        if index >= length:
            return fields, None
        value_start = index
        try:
            value, index = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            return fields, None
        if isinstance(value, str):
            fields[key] = value
            if key == "session_id":
                fields[_SESSION_ID_VALUE_SPAN_KEY] = (
                    len(text[:value_start].encode("utf-8")),
                    len(text[:index].encode("utf-8")),
                )
        elif key == "kind":
            # Preserve an explicitly invalid kind so oversized readers cannot
            # mistake it for the supported missing-kind legacy shape.
            fields[key] = value
        elif key == "session_id":
            fields.pop(key, None)
            fields[_SESSION_ID_VALUE_SPAN_KEY] = None
        index = skip_json_whitespace(index)
        if index >= length:
            return fields, None
        if text[index] == ",":
            index += 1
            continue
        if text[index] == "}":
            return fields, None
        return fields, None


def _inspect_top_level_json_string_fields_before_content(text: str) -> tuple[dict[str, Any], bool]:
    fields, content_quote_index = _parse_top_level_json_string_fields_before_content(text)
    return fields, content_quote_index is not None


def _read_externalized_payload_metadata_prefix_from_handle(
    handle: BinaryIO,
    *,
    max_read_bytes: int,
) -> tuple[str, dict[str, Any], bool, bool]:
    """Read through the opening quote of the top-level content string."""
    prefix = bytearray()
    text_parts: list[str] = []
    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    prefix_truncated = False
    read_limit = max(1, int(max_read_bytes))
    fields: dict[str, Any] = {}

    while len(prefix) < read_limit:
        chunk = handle.read(min(4096, read_limit - len(prefix)))
        if not chunk:
            break
        prefix.extend(chunk)
        try:
            decoded = decoder.decode(chunk, final=False)
        except UnicodeDecodeError as exc:
            raise ValueError("invalid_payload") from exc
        if decoded:
            text_parts.append(decoded)
        prefix_text = "".join(text_parts)
        fields, content_quote_index = _parse_top_level_json_string_fields_before_content(prefix_text)
        if content_quote_index is not None:
            metadata_prefix_text = prefix_text[: content_quote_index + 1]
            return metadata_prefix_text, fields, True, False

    prefix_truncated = len(prefix) >= read_limit and bool(handle.read(1))
    if not prefix_truncated:
        try:
            final_text = decoder.decode(b"", final=True)
        except UnicodeDecodeError as exc:
            raise ValueError("invalid_payload") from exc
        if final_text:
            text_parts.append(final_text)
    return "".join(text_parts), fields, False, prefix_truncated


def read_externalized_payload_metadata_prefix(
    path: Path,
    *,
    max_read_bytes: int,
) -> tuple[str, bool, bool]:
    with path.open("rb") as handle:
        prefix_text, _fields, content_key_seen, prefix_truncated = (
            _read_externalized_payload_metadata_prefix_from_handle(
                handle,
                max_read_bytes=max_read_bytes,
            )
        )
    return prefix_text, content_key_seen, prefix_truncated

