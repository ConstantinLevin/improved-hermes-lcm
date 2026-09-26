"""Which result belongs to which tool call: a fact of the store, never a prediction of
what the host sends (the orchestrator's ruling on the re-plan of #71, 2026-09-26).

The store keeps no pairing. A call's handle names (record, position) and nothing else
(``RecordStore._write_tool_calls``). Which stored result belongs to it is read off the
stored records alone:

1. **Within the block**, a result belongs to a call iff the result's ``tool_call_id``
   equals the call's ``id`` exactly. The block of a record is the nearest assistant or user
   record at or before it, up to the next assistant or user record: records of every other
   role lie inside. This is what the store's own write check guarantees for every result it
   takes (``fresh_tail.check_tool_pairing``: a result names an earlier call's exact id).
2. **A degenerate id group is stated, never resolved.** Where two or more calls of a
   block carry one ``id``, or a call's aliases (its ``call_id``, its ``response_item_id``,
   and each part of a composite ``a|b`` of any of them) include another call's ``id``, those
   calls form one group, with every result of the block whose id is one of the group's
   spellings; so does a single call that two or more results of the block name. Every call
   and every result of the group carries one note saying that the store cannot tell which
   result answered which call; every result of the group is listed under each of its
   calls, and nothing is attributed.
3. A call with no result of its exact id in the block has none; a result whose id no call
   of the block carries, or that carries no id, belongs to none. Both are store facts.

No host function is called here. What the provider received of a stretch depends on the
host's pre-call sanitizer and on the session's route, and those differ from one another
exactly where rule 2 applies (the re-plan of #71, section A); this plugin does not
reproduce them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Group:
    """Calls and results of one block that the store cannot pair (rule 2)."""

    ids: list                                 # the distinct ``id`` values of its calls, in order
    calls: list                               # (assistant record, position), in order
    results: list                             # result records, in order


@dataclass
class Pairing:
    """The store's pairing of a run of blocks."""

    # result record -> (assistant record, position of the call it belongs to)
    result_of: dict = field(default_factory=dict)
    # (assistant record, position) -> the result record that belongs to it
    answer: dict = field(default_factory=dict)
    # (assistant record, position) or result record -> its Group
    group: dict = field(default_factory=dict)
    # result record -> ("unknown", its id) or ("no_id", None): a result that belongs to no call
    stray: dict = field(default_factory=dict)


def _id_of(call: Any) -> Any:
    return call.get("id") if isinstance(call, dict) else None


def _aliases(call: Any) -> set:
    """A call's own spellings: its ``id``, ``call_id`` and ``response_item_id``, and each
    part of a composite ``a|b`` of any of them."""
    found: set = set()
    if not isinstance(call, dict):
        return found
    for key in ("id", "call_id", "response_item_id"):
        value = call.get(key)
        if isinstance(value, str) and value:
            found.add(value)
            found.update(part for part in value.split("|") if part)
    return found


def _joined(ci: Any, cj: Any) -> bool:
    """Rule 2: two calls of one block share an ``id``, or one's aliases hold the other's."""
    id_i, id_j = _id_of(ci), _id_of(cj)
    return ((id_i is not None and id_i == id_j) or (isinstance(id_j, str) and id_j in _aliases(ci))
            or (isinstance(id_i, str) and id_i in _aliases(cj)))


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

    claimed: set = set()
    for indexes in members.values():
        if len(indexes) == 1:
            position, call = calls[indexes[0]]
            call_id = _id_of(call)
            mine = [record for record, raw in results
                    if call_id is not None and raw.get("tool_call_id") == call_id]
            if len(mine) <= 1:
                if mine:
                    found.answer[(head_record, position)] = mine[0]
                    found.result_of[mine[0]] = (head_record, position)
                    claimed.add(mine[0])
                continue
        else:
            spellings: set = set()
            for i in indexes:
                spellings |= _aliases(calls[i][1])
            mine = [record for record, raw in results
                    if isinstance(raw.get("tool_call_id"), str) and raw["tool_call_id"] in spellings]
        ids: list = []
        for i in indexes:
            if _id_of(calls[i][1]) not in ids:
                ids.append(_id_of(calls[i][1]))
        group = Group(ids, [(head_record, calls[i][0]) for i in indexes], mine)
        for key in group.calls:
            found.group[key] = group
        for record in mine:
            found.group[record] = group
            claimed.add(record)

    for record, raw in results:
        if record in claimed:
            continue
        call_id = raw.get("tool_call_id")
        if call_id is None or (isinstance(call_id, str) and not call_id.strip()):
            found.stray[record] = ("no_id", None)
        else:
            found.stray[record] = ("unknown", call_id)


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
