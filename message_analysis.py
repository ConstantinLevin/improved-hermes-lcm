"""The host's identity of a tool call: its id, which pairs it with its result."""

from __future__ import annotations

from typing import Any


def _tool_call_id(tool_call: Any) -> str:
    if not isinstance(tool_call, dict):
        return ""
    value = tool_call.get("id") or tool_call.get("tool_call_id")
    return str(value).strip() if value else ""
