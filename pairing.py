"""Which result belongs to which tool call: a fact of the store, never a prediction of
what the host sends (the orchestrator's ruling on the re-plan of #71, 2026-09-26).

The store keeps no pairing. A call's handle names (record, position) and nothing else
(``RecordStore._write_tool_calls``). Which stored result belongs to it is read off the
stored records alone:

1. **Within the block**, a result belongs to a call iff the result's ``tool_call_id`` is
   the same id as the call's ``id`` (``same_id``). The block of a record is the nearest
   assistant or user record at or before it, up to the next assistant or user record:
   records of every other role lie inside. The store's write check
   (``fresh_tail.check_tool_pairing``) does not guarantee this: it checks the whole list
   cumulatively, on ``str()``-ed and stripped ids, and takes a call's ``tool_call_id`` where
   its ``id`` is empty (``message_analysis._tool_call_id``); so a stored result can stand
   outside its call's block or carry an alias of its call. Where a call or a result is left
   unpaired, the pairing says why, from the block (#78, A11: ``Pairing.unanswered``,
   ``Pairing.stray``).
2. **A degenerate id group is stated, never resolved.** Where two or more calls of a
   block carry one ``id``, or a call's aliases (its ``call_id``, its ``response_item_id``,
   and each part of a composite ``a|b`` of any of them) include another call's ``id``, those
   calls form one group, with every result of the block whose id is one of the group's
   spellings; so does a single call that two or more results of the block name. Every call
   and every result of the group carries one note saying that the store cannot tell which
   result answered which call; every result of the group is listed under each of its
   calls, and nothing is attributed.
3. A call with no result of its id in the block has none; a result whose id no call of
   the block carries, or that carries no id, belongs to none. Both are store facts.

**Id equality** (#82): ``same_id`` is the only place two ids are compared, and ``id_text``
the only way an id is shown. A stored id can be any JSON value (LEARNINGSFÜRPLÄNE A2 on the
domain axis). Two ids are the same iff both are scalars (``id_state``: a non-blank string, a
number, a boolean) of the same JSON type with the same value: strings exactly, numbers by
value (``1`` is ``1.0``, JSON's own equality of numbers), booleans as booleans, never as
numbers. A null, blank, list or object id is the same as nothing. No id is ever hashed. The
host pairs only non-blank strings, stripped and split at ``|`` (agent/message_sanitization.py
:496-515 at Hermes fbb06142ef); where a stored id is not a string, or carries whitespace, the
store's pairing is the store's, and the notes say what the store holds.

No host function is called here. What the provider received of a stretch depends on the
host's pre-call sanitizer and on the session's route, and those differ from one another
exactly where rule 2 applies (the re-plan of #71, section A); this plugin does not
reproduce them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Group:
    """Calls and results of one block that the store cannot pair (rule 2)."""

    ids: list                                 # the distinct ids of its calls, in order (``same_id``)
    calls: list                               # (assistant record, position), in order
    results: list                             # result records, in order
    spellings: list = field(default_factory=list)   # every spelling that joined it, in order (#78)


@dataclass
class Pairing:
    """The store's pairing of a run of blocks."""

    # result record -> (assistant record, position of the call it belongs to)
    result_of: dict = field(default_factory=dict)
    # (assistant record, position) -> the result record that belongs to it
    answer: dict = field(default_factory=dict)
    # (assistant record, position) or result record -> its Group
    group: dict = field(default_factory=dict)
    # result record -> (why, its id, detail): a result that belongs to no call, and the block's
    # fact that says why: "no_id"; "structured" (its id is a list or an object, ``id_state``);
    # "no_calls" (its block opens with ``detail``, a message holding no calls); "alias" (its
    # id is another spelling of call ``detail`` of the block, not that call's id); "unknown"
    # (no call of the block has its id in any spelling)
    stray: dict = field(default_factory=dict)
    # (assistant record, position) -> (why, detail): a call no result of its block pairs with,
    # and why, from the block: "no_id" (the call carries no id); "empty_id" / "structured_id"
    # (its id, ``detail``, is blank, or a list or an object: ``id_state``); "alias" (result
    # ``detail[0]`` of the block carries the call's alias ``detail[1]``, not its id); "none"
    # (no result of the block carries its id or an alias of it)
    unanswered: dict = field(default_factory=dict)


def _id_of(call: Any) -> Any:
    return call.get("id") if isinstance(call, dict) else None


def id_state(value: Any) -> str:
    """What a stored host id is, over its whole domain (#82, LEARNINGSFÜRPLÄNE A2 on the domain
    axis: the store takes any JSON value for a call's ``id``, and the pairing reads it):
    "none" (absent or null), "empty" (a string of nothing but whitespace), "scalar" (any other
    string, a number or a boolean), "structured" (a list, an object)."""
    if value is None:
        return "none"
    if isinstance(value, str):
        return "scalar" if value.strip() else "empty"
    if isinstance(value, (bool, int, float)):
        return "scalar"
    return "structured"


def _json_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    return "string"


def same_id(a: Any, b: Any) -> bool:
    """Whether two stored ids are the same id (the module docstring, "Id equality"): both
    scalars of the same JSON type with the same value. The only comparison of ids."""
    if id_state(a) != "scalar" or id_state(b) != "scalar" or _json_type(a) != _json_type(b):
        return False
    return a == b


def id_text(value: Any) -> str:
    """An id as a note shows it: its JSON, so "5" and 5, "true" and true, "[]" and [] are
    told apart (#82). The only rendering of an id."""
    return json.dumps(value, ensure_ascii=False)


def _merge(spellings: list, more: list) -> list:
    for value in more:
        if not any(same_id(value, have) for have in spellings):
            spellings.append(value)
    return spellings


def _aliases(call: Any) -> list:
    """A call's own spellings, in order: its ``id`` where it is a scalar, then its string
    ``id``, ``call_id`` and ``response_item_id`` and each part of a composite ``a|b`` of
    any of them; distinct by ``same_id``."""
    found: list = []
    if not isinstance(call, dict):
        return found
    if id_state(call.get("id")) == "scalar":
        _merge(found, [call["id"]])
    for key in ("id", "call_id", "response_item_id"):
        value = call.get(key)
        if isinstance(value, str) and value.strip():
            _merge(found, [value] + [part for part in value.split("|") if part.strip()])
    return found


def _holds(spellings: list, value: Any) -> bool:
    return any(same_id(value, spelling) for spelling in spellings)


def _joined(ci: Any, cj: Any) -> bool:
    """Rule 2: two calls of one block share an ``id``, or one's aliases hold the other's."""
    return _holds(_aliases(cj), _id_of(ci)) or _holds(_aliases(ci), _id_of(cj))


def _pair_block(block: list[tuple[str, Any]], found: Pairing) -> None:
    head_record, head = block[0]
    calls: list = []
    if isinstance(head, dict) and head.get("role") == "assistant" and isinstance(head.get("tool_calls"), list):
        calls = list(enumerate(head["tool_calls"]))
    results = [(record, raw) for record, raw in block if isinstance(raw, dict) and raw.get("role") == "tool"]

    parent = list(range(len(calls)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(calls)):
        for j in range(i + 1, len(calls)):
            if _joined(calls[i][1], calls[j][1]):
                parent[root(j)] = root(i)
    members: dict[int, list] = {}
    for i in range(len(calls)):
        members.setdefault(root(i), []).append(i)

    claimed: list = []
    for indexes in members.values():
        if len(indexes) == 1:
            position, call = calls[indexes[0]]
            call_id = _id_of(call)
            state = id_state(call_id)
            mine = [record for record, raw in results if same_id(raw.get("tool_call_id"), call_id)]
            if len(mine) <= 1:
                if mine:
                    found.answer[(head_record, position)] = mine[0]
                    found.result_of[mine[0]] = (head_record, position)
                    claimed.append(mine[0])
                elif state == "none":
                    found.unanswered[(head_record, position)] = ("no_id", None)
                elif state in ("empty", "structured"):
                    found.unanswered[(head_record, position)] = (f"{state}_id", call_id)
                else:
                    other = [spelling for spelling in _aliases(call) if not same_id(spelling, call_id)]
                    alias = next(((record, raw["tool_call_id"]) for record, raw in results
                                  if _holds(other, raw.get("tool_call_id"))), None)
                    found.unanswered[(head_record, position)] = ("alias", alias) if alias else ("none", None)
                continue
            spellings = _aliases(call)
        else:
            spellings = []
            for i in indexes:
                _merge(spellings, _aliases(calls[i][1]))
            mine = [record for record, raw in results if _holds(spellings, raw.get("tool_call_id"))]
        ids: list = []
        for i in indexes:
            value = _id_of(calls[i][1])
            if value is not None and not any(same_id(value, have) or value is have for have in ids):
                ids.append(value)
        group = Group(ids, [(head_record, calls[i][0]) for i in indexes], mine, spellings)
        for key in group.calls:
            found.group[key] = group
        for record in mine:
            found.group[record] = group
            claimed.append(record)

    for record, raw in results:
        if record in claimed:
            continue
        call_id = raw.get("tool_call_id")
        state = id_state(call_id)
        if state in ("none", "empty"):
            found.stray[record] = ("no_id", None, None)
        elif state == "structured":
            found.stray[record] = ("structured", call_id, None)
        elif not calls:
            found.stray[record] = ("no_calls", call_id, head_record)
        else:
            owner = next((position for position, call in calls if _holds(_aliases(call), call_id)), None)
            found.stray[record] = ("alias", call_id, owner) if owner is not None else ("unknown", call_id, None)


def _opens_block(message: Any) -> bool:
    return isinstance(message, dict) and message.get("role") in ("assistant", "user")


def pair(sequence: list[tuple[str, Any]]) -> Pairing:
    """The store's pairing of ``sequence`` ((record, the host's dict as stored), in the
    active order), block by block (the module docstring)."""
    found = Pairing()
    block: list = []
    for record, raw in sequence:
        if _opens_block(raw) and block:
            _pair_block(block, found)
            block = []
        block.append((record, raw))
    if block:
        _pair_block(block, found)
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
                   raws: Callable[[list[str]], dict]) -> Pairing:
    """The pairing of ``records`` (contiguous in ``order``) through the blocks they lie in."""
    places = [index[r] for r in records if r in index]
    if not places:
        return Pairing()
    start, end = blocks_span(order, roles, min(places), max(places))
    span = order[start:end]
    raw = raws(span)
    return pair([(record, raw.get(record) or {}) for record in span])
