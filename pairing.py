"""Which result answers which tool call: read off the host's own passes, never computed
by the plugin (the plan of #71, approved 2026-09-26, §1.6).

The store keeps no pairing. A call's handle names (record, position) and nothing else
(``RecordStore._write_tool_calls``). Which result answers it is what the host sends of the
stored sequence, read off the host's own functions (agent/turn_context.py and
agent/agent_runtime_helpers.py at Hermes 8afaab3703):

- on every api_mode but codex_responses, ``call_id`` and ``response_item_id`` are stripped
  from a message's tool calls before anything else (``build_api_messages``, 1271-1274, by
  ``ReasoningParamsMixin._should_sanitize_tool_calls`` and
  ``_sanitize_tool_calls_for_strict_api``, agent/reasoning_params.py:219-236), which changes
  a call's aliases;
- then the four passes of ``sanitize_api_messages`` (3035-3046) that decide what is sent of
  a call or a result, in the host's order: ``_drop_invalid_roles``,
  ``_drop_results_without_ids``, ``_pair_tool_calls_positionally`` (a declared call without
  a result gets the host's own stand-in) and ``_dedupe_tool_call_ids``.

The other passes of ``sanitize_api_messages`` are not called: ``repair_empty_non_final_messages``
changes only the content of empty messages and, when it fires, feeds the host's heal counter
and a notice to its user (2632-2705); ``_drop_empty_tool_calls_arrays`` touches only empty or
non-list ``tool_calls``; ``_repair_invalid_tool_call_names`` changes only a call's name and
logs a warning into the host's log; ``_realign_tool_result_names`` changes only a result's
name. Every pass pairs by id.

**The block (the pairing's definition).** A record's pairing is what these passes send of
the block that decides it: the nearest assistant or user record at or before it, up to the
next assistant or user record. The positional pass resets at every assistant and user
message and flushes at the end (2877-2903), so for it a block run equals a run over the
whole list. The duplicate pass never resets (2919-2977): a kept call whose alias group no
result of its run consumes stays outstanding into later messages. The host's own requests
held a different past before a stretch at every request (summaries, later records), so
there is no single run of the whole list to reproduce; the block is the definition, and
every handle (a chunk, a message, a tool call) is paired through the same blocks. The
carry-over across blocks is a named residual: it needs a call whose alias equals another
call's id in the same message, which the host's own lists never hold after its repair.

Not modelled: the host's time-dependent ``canonicalize_replay_history`` on later requests
(agent/replay_cleanup.py), which is the host's rewrite of a request, not of the record.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Optional


class PairingUnavailable(Exception):
    """The host's functions cannot be read or run on these records: no pairing is known."""


def host_alias_helpers() -> tuple[Callable[[Any], frozenset], Callable[[Any], frozenset], Callable[[Any], str]]:
    try:
        from agent.message_sanitization import (  # type: ignore
            coalesce_tool_call_id,
            tool_call_id_variants,
            tool_result_id_variants,
        )
    except Exception as exc:
        raise PairingUnavailable(f"the host's tool-call id helpers (agent.message_sanitization) cannot be read "
                                 f"({type(exc).__name__}: {exc}), so which result answers which call is not "
                                 f"known") from None
    return tool_call_id_variants, tool_result_id_variants, coalesce_tool_call_id


def host_sending_passes() -> list[tuple[str, Callable]]:
    """The host's pre-call passes that decide which calls and results are sent, in the
    host's order (``sanitize_api_messages``, agent/agent_runtime_helpers.py:3035-3046)."""
    try:
        from agent.agent_runtime_helpers import (  # type: ignore
            _dedupe_tool_call_ids,
            _drop_invalid_roles,
            _drop_results_without_ids,
            _pair_tool_calls_positionally,
        )
    except Exception as exc:
        raise PairingUnavailable(f"the host's pre-call sanitizer (agent.agent_runtime_helpers) cannot be read "
                                 f"({type(exc).__name__}: {exc}), so which result answers which call is not "
                                 f"known") from None
    return [("_drop_invalid_roles", _drop_invalid_roles), ("_drop_results_without_ids", _drop_results_without_ids),
            ("_pair_tool_calls_positionally", _pair_tool_calls_positionally),
            ("_dedupe_tool_call_ids", _dedupe_tool_call_ids)]


def host_strict_scrub(api_mode: str) -> Optional[Callable[[dict], Any]]:
    """The host's scrub of Codex Responses fields from tool calls, where the host applies it
    for this api_mode (``build_api_messages``, agent/turn_context.py:1271-1274); None where
    it does not."""
    try:
        from agent.reasoning_params import ReasoningParamsMixin  # type: ignore
    except Exception as exc:
        raise PairingUnavailable(f"the host's strict-API tool-call scrub (agent.reasoning_params) cannot be read "
                                 f"({type(exc).__name__}: {exc}), so which result answers which call is not "
                                 f"known") from None
    if not ReasoningParamsMixin._should_sanitize_tool_calls(SimpleNamespace(api_mode=api_mode)):
        return None
    return ReasoningParamsMixin._sanitize_tool_calls_for_strict_api


_RECORD = "_lcm_pairing_record"
_POSITION = "_lcm_pairing_position"


@dataclass
class Pairing:
    """What the host's passes send of a run of blocks."""

    # result record -> (assistant record, position of the call it answers)
    result_of: dict = field(default_factory=dict)
    # (assistant record, position) -> the result record that answers it on the wire
    answer: dict = field(default_factory=dict)
    # (assistant record, position) -> the content of the host's own stand-in for its result
    stand_in: dict = field(default_factory=dict)
    # (assistant record, position) -> the host id of a call the host does not send
    call_not_sent: dict = field(default_factory=dict)
    # result record -> (the pass that does not send it: "no_id", "positional",
    # "duplicate"; its host id)
    result_not_sent: dict = field(default_factory=dict)
    # record -> its role, for a record of a role the host does not send
    role_not_sent: dict = field(default_factory=dict)
    # (assistant record, position) -> (host id, the calls of its message with that id)
    shared: dict = field(default_factory=dict)


def _copy_for_passes(record: str, raw: Any) -> dict:
    """A shallow copy of the stored dict, tagged with its record, its tool calls copied and
    tagged with their positions; content is shared: no pass changes content in place, each
    rebuilds a message it changes (``{**msg, ...}``)."""
    message = dict(raw) if isinstance(raw, dict) else {}
    message[_RECORD] = record
    calls = message.get("tool_calls")
    if isinstance(calls, list):
        copied = []
        for position, call in enumerate(calls):
            if isinstance(call, dict):
                call = dict(call)
                call[_POSITION] = position
            copied.append(call)
        message["tool_calls"] = copied
    return message


def _pair_block(block: list[tuple[str, Any]], found: Pairing, *, api_mode: str, model: str) -> None:
    call_variants, result_variants, coalesce = host_alias_helpers()
    passes = host_sending_passes()
    scrub = host_strict_scrub(api_mode)
    tagged = [_copy_for_passes(record, raw) for record, raw in block]
    if scrub is not None:
        for message in tagged:
            try:
                scrub(message, model=model)
            except Exception as exc:
                raise PairingUnavailable(f"the host's strict-API tool-call scrub cannot be run on these records "
                                         f"({type(exc).__name__}: {exc})") from None

    calls_of: dict[str, list] = {}
    for message in tagged:
        calls = message.get("tool_calls")
        if message.get("role") == "assistant" and isinstance(calls, list):
            calls_of[message[_RECORD]] = [
                (call.get(_POSITION, position) if isinstance(call, dict) else position, call,
                 frozenset(call_variants(call)))
                for position, call in enumerate(calls)]
    # Which calls of one message share an id, by the host's alias helpers, on the calls as
    # the host pairs them (after the scrub).
    for record, calls in calls_of.items():
        for position, call, variants in calls:
            if not variants:
                continue
            sharing = [p for p, _c, v in calls if v & variants]
            if len(sharing) > 1:
                found.shared[(record, position)] = (coalesce(call) or sorted(variants)[0], len(sharing))

    stages: list[list] = []
    current = tagged
    for name, host_pass in passes:
        try:
            current = host_pass(current)
        except Exception as exc:
            raise PairingUnavailable(f"the host's pre-call sanitizer ({name}) cannot be run on these records "
                                     f"({type(exc).__name__}: {exc})") from None
        stages.append(current)

    def records_in(messages: list) -> set:
        return {m.get(_RECORD) for m in messages if isinstance(m, dict) and m.get(_RECORD)}

    after_roles, after_ids, after_positional, sent = (records_in(stage) for stage in stages)
    for message in tagged:
        record, role = message[_RECORD], message.get("role")
        if record not in after_roles:
            found.role_not_sent[record] = role
            continue
        if role != "tool" or record in sent:
            continue
        call_id = message.get("tool_call_id")
        shown_id = str(call_id).strip() if call_id is not None else None
        if record not in after_ids:
            found.result_not_sent[record] = ("no_id", shown_id)
        elif record not in after_positional:
            found.result_not_sent[record] = ("positional", shown_id)
        else:
            found.result_not_sent[record] = ("duplicate", shown_id)

    # The sent output: each assistant with its calls, then what follows it before the next
    # assistant or user message; a call's result is the tool message whose id is one of its
    # aliases (unique once the host's passes ran).
    holder: Optional[str] = None
    sent_calls: list = []
    for message in stages[-1]:
        role = message.get("role") if isinstance(message, dict) else None
        if role in ("assistant", "user"):
            holder = message.get(_RECORD) if role == "assistant" else None
            sent_calls = [(call.get(_POSITION), frozenset(call_variants(call)))
                          for call in (message.get("tool_calls") or []) if isinstance(call, dict)] \
                if role == "assistant" else []
            if holder is not None:
                sent_positions = {p for p, _v in sent_calls}
                for position, call, _variants in calls_of.get(holder, []):
                    if position not in sent_positions:
                        found.call_not_sent[(holder, position)] = coalesce(call) or None
            continue
        if role != "tool" or holder is None:
            continue
        variants = frozenset(result_variants(message.get("tool_call_id")))
        position = next((p for p, v in sent_calls if v & variants), None)
        if position is None:
            continue
        record = message.get(_RECORD)
        if record:
            found.answer[(holder, position)] = record
            found.result_of[record] = (holder, position)
        else:
            content = message.get("content")
            found.stand_in[(holder, position)] = content if isinstance(content, str) else json.dumps(content)


def _opens_block(message: Any) -> bool:
    return isinstance(message, dict) and message.get("role") in ("assistant", "user")


def pair(sequence: list[tuple[str, Any]], *, api_mode: str, model: str = "") -> Pairing:
    """What the host sends of ``sequence`` ((record, the host's dict as stored), in the
    active order), block by block (the module docstring)."""
    found = Pairing()
    block: list = []
    for record, raw in sequence:
        if _opens_block(raw) and block:
            _pair_block(block, found, api_mode=api_mode, model=model)
            block = []
        block.append((record, raw))
    if block:
        _pair_block(block, found, api_mode=api_mode, model=model)
    return found


def blocks_span(order: list[str], roles: Callable[[list[str]], dict], first: int, last: int) -> tuple[int, int]:
    """The indexes [start, end) of ``order`` holding the blocks of the records
    ``order[first:last + 1]``: back to the nearest assistant or user record at or before
    ``first``, and on to the last record before the next assistant or user record after
    ``last`` (or the end). Records of every other role lie inside."""
    start = first
    step = 64
    while start > 0:
        lower = max(0, start - step)
        part = order[lower:start + 1]
        kinds = roles(part)
        hit = next((i for i in range(len(part) - 1, -1, -1) if kinds.get(part[i]) in ("assistant", "user")), None)
        if hit is not None:
            start = lower + hit
            break
        start = lower
    end = last + 1
    while end < len(order):
        part = order[end:end + step]
        kinds = roles(part)
        stop = next((i for i, record in enumerate(part) if kinds.get(record) in ("assistant", "user")), None)
        if stop is not None:
            end += stop
            break
        end += len(part)
    return start, end


def pairing_around(order: list[str], index: dict, records: list[str], roles: Callable[[list[str]], dict],
                   raws: Callable[[list[str]], dict], *, api_mode: str, model: str = "") -> Pairing:
    """The pairing of ``records`` (contiguous in ``order``) through the blocks they lie in."""
    places = [index[r] for r in records if r in index]
    if not places:
        return Pairing()
    start, end = blocks_span(order, roles, min(places), max(places))
    span = order[start:end]
    raw = raws(span)
    return pair([(record, raw.get(record) or {}) for record in span], api_mode=api_mode, model=model)
