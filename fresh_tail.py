"""The rule the tail and the cut share: a tool call is never separated from its
results (#13), and a result that names no call of the list is an error.

The tail is sized in tokens from the target (``CompactionMixin._tail_plan``, #31) and
placed at the boundaries of the same groups the cut uses (``CompactionMixin._groups``).
Both rest on the pairing this module checks once over the whole list, before any
boundary is chosen: every tool row names, by its ``tool_call_id``, a call an assistant
row before it made.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

from .message_analysis import _tool_call_id


class ToolPairingError(Exception):
    """A tool row no boundary can be placed around: its ``tool_call_id`` is empty, or
    no assistant row before it in the list made that call (#13)."""


def check_tool_pairing(messages: Sequence[Dict[str, Any]]) -> None:
    """Every tool row of the list, wherever it stands (the material, the tail, between
    the summaries), answers a call an assistant row before it made; otherwise
    ``ToolPairingError``. Nothing is guessed: a result without its call cannot be
    grouped, cut or kept in the tail as the model requires."""
    made: set = set()
    for position, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "assistant":
            made |= {_tool_call_id(call) for call in (message.get("tool_calls") or [])} - {""}
        elif role == "tool":
            result_id = str(message.get("tool_call_id") or "").strip()
            if not result_id:
                raise ToolPairingError(f"the tool row at position {position} carries no tool_call_id")
            if result_id not in made:
                raise ToolPairingError(f"the tool row at position {position} answers the call {result_id!r}, "
                                       f"which no assistant row before it in the list made")
