"""The tail's group boundary: a tool call and its results are never separated (#13).

The tail itself is sized in tokens from the target (``CompactionMixin._tail_plan``,
#31); no message count survives beside it. This module keeps the one rule the cut
shares with it: where a boundary would fall on a tool result, it moves back to the
assistant row that made the call, found by the result's ``tool_call_id``.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

from .message_analysis import _tool_call_id


class ToolPairingError(Exception):
    """A tool row the boundary cannot be placed around: its ``tool_call_id`` is empty,
    so the call it answers cannot be found (#13)."""


def _assistant_group_start(messages: Sequence[Dict[str, Any]], start: int) -> int:
    """Where the group of the row at ``start`` begins.

    For a tool result: the assistant row whose ``tool_calls`` carry the result's
    ``tool_call_id``, searched backwards across tool rows, other assistant rows and
    whatever else stands between (a result can stand behind another assistant's
    calls). Where no row of the list made the call, the result begins its own group.
    An empty ``tool_call_id`` is an error (``ToolPairingError``). Any other row begins
    its own group."""
    if start >= len(messages) or not isinstance(messages[start], dict) or messages[start].get("role") != "tool":
        return start
    result_id = str(messages[start].get("tool_call_id") or "").strip()
    if not result_id:
        raise ToolPairingError(f"the tool row at position {start} carries no tool_call_id")
    for index in range(start - 1, -1, -1):
        message = messages[index]
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        if result_id in {_tool_call_id(call) for call in (message.get("tool_calls") or [])}:
            return index
    return start
