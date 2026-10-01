"""Validate the common pairing before compaction assembles any context.

Calls and results are never invented or dropped to make the list pass. An ambiguity
group preserves its actual members without claiming individual attribution.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

from .pairing import Pairing, pair


class ToolPairingError(Exception):
    """A missing side or a cut that would split a call/result unit (#13)."""


def check_tool_pairing(messages: Sequence[Dict[str, Any]]) -> Pairing:
    """Validate both sides everywhere in the full given order, including the tail.

    A nonempty ambiguity group is kept unattributed; it does not prove a distinct
    execution result for each call. An empty group cannot stand for a missing result.
    """
    found = pair(list(enumerate(messages)))
    for position, (why, result_id) in found.stray.items():
        raise ToolPairingError(
            f"the tool row at position {position} has no call in its assistant/user block "
            f"({why}: {result_id!r})")
    for position, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        calls = message.get("tool_calls")
        if calls is None:
            continue
        if not isinstance(calls, list):
            raise ToolPairingError(f"the assistant row at position {position} has tool_calls that is not a list")
        for slot, call in enumerate(calls):
            key = (position, slot)
            group = found.group.get(key)
            if key not in found.answer and (group is None or not group.results):
                call_id = call.get("id") if isinstance(call, dict) else None
                raise ToolPairingError(
                    f"the call at position {position}, tool_calls[{slot}], has no result in its "
                    f"assistant/user block (id: {call_id!r})")
    return found
