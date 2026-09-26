"""Which result answers which tool call: read when it is asked, by the host's own rule
(the orchestrator's ruling on round 5 of #71).

The store keeps no pairing. A call's handle names (record, position) and nothing else
(``RecordStore._write_tool_calls``); which result answers it is read over the active
record's order by the host's list rule, ``_drop_stray_tool_results``
(agent/agent_runtime_helpers.py:496-535 at Hermes d0288be5b3), with the host's alias
helpers ``tool_call_id_variants`` and ``tool_result_id_variants``
(agent/message_sanitization.py:508-515). The rule, as the host runs it:

- an assistant or a user message starts afresh: the ids known so far and the calls
  answered are forgotten; an assistant message's calls become known, each alias of a call
  taken by the first call that has it (``setdefault``);
- a tool result takes the lowest known call among those its aliases name that no earlier
  result took; where its aliases name none, it is a stray (the host drops it from its list);
  a result without an id pairs with nothing.

So of two calls of one message that share an id, only the first is ever answered: a
second result with that id is a stray, and the host's pre-call sanitizer
(``_dedupe_tool_call_ids``, 2914-2980) sends the provider only the first call and result
of one id. Such calls and results carry a note that says so.

Since the rule forgets everything at an assistant or user message, what a stretch pairs
depends only on the records from the nearest such message before it to the end of the
tool results that follow it (``window``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


class PairingUnavailable(Exception):
    """The host's alias helpers cannot be read: no pairing is known."""


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


@dataclass
class Pairing:
    """What the host's rule pairs over a run of records."""

    # result record -> (assistant record, position of the call it answers)
    result_of: dict = field(default_factory=dict)
    # (assistant record, position) -> the result record that answers it
    answer: dict = field(default_factory=dict)
    # result record -> (why it pairs with nothing, its host id): "no_id", "taken"
    # (a known call with its id was already answered), "unknown" (no call before it has it)
    unpaired: dict = field(default_factory=dict)
    # (assistant record, position) -> (host id, calls of the message with it, results with it)
    shared_calls: dict = field(default_factory=dict)
    # result record -> the same, for a result with an id several calls of its message share
    shared_results: dict = field(default_factory=dict)


def pair(sequence: list[tuple[str, dict]]) -> Pairing:
    """The host's list rule over ``sequence`` ((record, the host's dict), in the active
    order), and which calls of one message share an id."""
    call_variants, result_variants, coalesce = host_alias_helpers()
    found = Pairing()
    known: dict[str, int] = {}
    matched: set = set()
    group_of: dict[int, tuple[str, int]] = {}
    next_group = 0
    run_calls: list[tuple[str, int, frozenset, Any]] = []
    run_results: list[tuple[str, frozenset]] = []

    def close_run() -> None:
        for record, position, variants, call in run_calls:
            if not variants:
                continue
            sharing = [c for c in run_calls if c[2] & variants]
            if len(sharing) < 2:
                continue
            results = [r for r, rv in run_results if rv & variants]
            note = (coalesce(call) or sorted(variants)[0], len(sharing), len(results))
            found.shared_calls[(record, position)] = note
            for result in results:
                found.shared_results.setdefault(result, note)
        run_calls.clear()
        run_results.clear()

    in_run = False
    for record, message in sequence:
        role = message.get("role") if isinstance(message, dict) else None
        if role in ("assistant", "user"):
            close_run()
            known = {}
            matched = set()
            in_run = role == "assistant"
            calls = message.get("tool_calls") if role == "assistant" else None
            for position, call in enumerate(calls if isinstance(calls, list) else ()):
                variants = call_variants(call)
                if variants:
                    for alias in variants:
                        known.setdefault(alias, next_group)
                    group_of[next_group] = (record, position)
                    next_group += 1
                run_calls.append((record, position, variants, call))
        elif role == "tool":
            call_id = message.get("tool_call_id")
            variants = result_variants(call_id)
            candidates = {known[alias] for alias in variants if alias in known and known[alias] not in matched}
            if candidates:
                group = min(candidates)
                matched.add(group)
                found.result_of[record] = group_of[group]
                found.answer[group_of[group]] = record
            elif not variants:
                found.unpaired[record] = ("no_id", None)
            else:
                taken = any(alias in known for alias in variants)
                found.unpaired[record] = ("taken" if taken else "unknown",
                                          str(call_id).strip() if isinstance(call_id, str) else str(call_id))
            if in_run:
                run_results.append((record, variants))
    close_run()
    return found


def window(order: list[str], roles: Callable[[list[str]], dict], first: int, last: int) -> tuple[int, int]:
    """The indexes [start, end) of ``order`` whose records decide the pairing of the
    records ``order[first:last + 1]``: back to the nearest assistant or user record at or
    before ``first``, and on through the tool records that follow ``last``."""
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
        stop = next((i for i, record in enumerate(part) if kinds.get(record) != "tool"), None)
        if stop is not None:
            end += stop
            break
        end += len(part)
    return start, end


def pairing_around(order: list[str], index: dict, records: list[str], roles: Callable[[list[str]], dict],
                   raws: Callable[[list[str]], dict]) -> Pairing:
    """The pairing of ``records`` (contiguous in ``order``) with what decides it."""
    places = [index[r] for r in records if r in index]
    if not places:
        return Pairing()
    start, end = window(order, roles, min(places), max(places))
    span = order[start:end]
    raw = raws(span)
    return pair([(record, raw.get(record) or {}) for record in span])
