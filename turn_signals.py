"""The host's per-turn signals, per host session id, process-wide (#32 §1, D1; #11).

The hooks arrive on threads of their own, and the host clones the engine per agent
(#23), so the state lives here, keyed by the session id each hook names:

- ``pre_llm_call``: a turn begins. It carries ``turn_id`` and ``conversation_history``,
  a shallow copy of the live list whose dicts are the live ones; the turn's opening
  row is its last user row, whose ``_row_id`` the host stamps in place when it
  persists it (#32, Point 2). The dict is kept for that id.
- ``post_tool_call`` of the same turn: a tool ran, so the list now ends with tool
  results (the host appends every result before the occasion after the round, #32 §3).
- ``post_api_request`` of the same turn: a response arrived after them, so the list no
  longer ends with tool results.
- ``pre_api_request`` of the same turn: the list a request sends, for the fixed prefix F
  measured at each response (R10).
- ``on_turn_complete`` (an engine method): the turn ended. The host skips it on some
  early exits (#32 P1), so a stale "in a turn" is possible; the list's structure then
  disagrees, and the occasion is a gap (D1).

An event whose ``turn_id`` is not the turn ``pre_llm_call`` opened for the session is
not this session's turn and changes nothing: the host's review fork reuses its parent's
session id with its own turns and gets no ``pre_llm_call`` (#20, #32 D1), so neither its
tool calls, its responses nor its ``on_turn_complete`` touch the parent's state (the
C1 ruling on the fork guard).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional

TOOL_RESULT = "tool_result"
API_RESPONSE = "api_response"


@dataclass(frozen=True)
class TurnState:
    in_turn: bool = False
    turn_id: str = ""
    # The opening row as pre_llm_call handed it over (the live dict).
    opening: Optional[Dict[str, Any]] = field(default=None, compare=False)
    # The last of this turn's events since pre_llm_call: "", TOOL_RESULT or API_RESPONSE.
    last_event: str = ""
    # A list the plugin was handed while the hooks say this turn runs ended as a turn
    # start does (a user row the host has not persisted, neither a steer nor flagged):
    # the gap of #32 D1, a turn whose end the host skipped or a nudge the host knows only
    # by its text. The plugin does not guess which: τ applies until the hooks settle it,
    # by the next pre_llm_call (a new turn) or by a tool or a response of this turn (it
    # lives). Only a list can see this; a read with no list before it cannot (A-32.2).
    contested: bool = False

    def opening_row_id(self) -> Optional[int]:
        row_id = self.opening.get("_row_id") if isinstance(self.opening, dict) else None
        return row_id if isinstance(row_id, int) else None

    def running(self) -> bool:
        """By the hooks alone: a turn runs, uncontested, and its list ends with tool
        results."""
        return self.in_turn and not self.contested and self.last_event == TOOL_RESULT

    def raises(self) -> bool:
        """Whether the published threshold is τ′ (#32 D2): a turn runs by the hooks and
        no list has contested it since (D1: at the gap, τ)."""
        return self.in_turn and not self.contested


_LOCK = threading.Lock()
_STATES: Dict[str, TurnState] = {}


def state(session_id: str) -> TurnState:
    with _LOCK:
        return _STATES.get(str(session_id or ""), TurnState())


def _opening_row(history: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(history, list):
        return None
    for message in reversed(history):
        if isinstance(message, dict) and message.get("role") == "user":
            return message
    return None


def turn_began(session_id: str, turn_id: str, conversation_history: Optional[List[Any]]) -> None:
    if not session_id:
        return
    with _LOCK:
        _STATES[str(session_id)] = TurnState(in_turn=True, turn_id=str(turn_id or ""),
                                             opening=_opening_row(conversation_history))
        # A list of an earlier turn (one whose end the host skipped) is no request of this one.
        _REQUEST_LISTS.pop(str(session_id), None)


def _event(session_id: str, turn_id: str, event: str) -> None:
    if not session_id:
        return
    with _LOCK:
        current = _STATES.get(str(session_id))
        if current is None or not current.in_turn or current.turn_id != str(turn_id or ""):
            return
        # An event of this turn shows it lives: a contest from a nudge-shaped list ends.
        _STATES[str(session_id)] = replace(current, last_event=event, contested=False)


def turn_start_seen(session_id: str, turn_id: str) -> bool:
    """A list ended as a turn start does while the hooks say the turn ``turn_id`` runs:
    the turn is contested (#32 D1) until the hooks settle it. Returns whether it was
    marked; a turn another event already replaced is left alone."""
    if not session_id:
        return False
    with _LOCK:
        current = _STATES.get(str(session_id))
        if current is None or not current.in_turn or current.turn_id != str(turn_id or ""):
            return False
        _STATES[str(session_id)] = replace(current, contested=True)
        return True


def tool_ran(session_id: str, turn_id: str) -> None:
    _event(session_id, turn_id, TOOL_RESULT)


def response_arrived(session_id: str, turn_id: str) -> None:
    _event(session_id, turn_id, API_RESPONSE)


def turn_ended(session_id: str, turn_id: Optional[str]) -> bool:
    """The turn ``pre_llm_call`` opened for the session ended. An end naming another
    turn changes nothing; an end naming none (an older host) ends the open turn.
    Returns whether the state changed."""
    if not session_id:
        return False
    with _LOCK:
        current = _STATES.get(str(session_id))
        if current is None or not current.in_turn:
            return False
        if turn_id is not None and str(turn_id) and current.turn_id and str(turn_id) != current.turn_id:
            return False
        _STATES[str(session_id)] = TurnState()
        _REQUEST_LISTS.pop(str(session_id), None)
        return True


def carry(old_session_id: str, new_session_id: str) -> None:
    """A compaction boundary renamed the host session: its turn goes on under the new id.
    The list the last request sent under the old id is dropped: it holds the dicts the
    compaction replaced, and the next request under the new id sends its own."""
    if not old_session_id or not new_session_id or old_session_id == new_session_id:
        return
    with _LOCK:
        current = _STATES.get(str(old_session_id))
        if current is not None:
            _STATES[str(new_session_id)] = current
        _REQUEST_LISTS.pop(str(old_session_id), None)


# R10: the list the newest request of a session sent (``pre_api_request``'s
# ``conversation_history``, a shallow copy whose dicts are the live ones), for the
# fixed prefix measured at its response. At most one per host session id, replaced at
# every request, and dropped once its response is measured (``take_request_list``),
# when the turn ends or a new one begins, when a compaction renames the id
# (``carry``), and when the session leaves the engine (``drop_request_list``: a
# switch, the engine's close). It never outlives the request it describes.
_REQUEST_LISTS: Dict[str, List[Any]] = {}


def request_sent(session_id: str, turn_id: str, conversation_history: Any) -> None:
    """Only a request of the turn ``pre_llm_call`` opened for the session counts: the
    review fork reuses its parent's id with turns of its own (C1)."""
    if not session_id or not isinstance(conversation_history, list):
        return
    with _LOCK:
        current = _STATES.get(str(session_id))
        if current is None or not current.in_turn or current.turn_id != str(turn_id or ""):
            return
        _REQUEST_LISTS[str(session_id)] = conversation_history


def take_request_list(session_id: str) -> Optional[List[Any]]:
    """The list the session's newest request sent, removed: a response measures it once."""
    with _LOCK:
        return _REQUEST_LISTS.pop(str(session_id or ""), None)


def drop_request_list(session_id: str) -> None:
    with _LOCK:
        _REQUEST_LISTS.pop(str(session_id or ""), None)
