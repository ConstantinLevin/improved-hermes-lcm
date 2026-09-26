"""Looking behind a handle (#18): what expansion returns, and the one page mechanism.

**What a handle opens into** (manifesto, "Looking behind a handle"; #29 W5; #34 D6):

- a summary's handle (``s``): the chunk it stands for; a summary written from the raw of
  several chunks, all of them, with the leaf summary of each named beside it as a
  lookup; a summary written from summaries, one layer: those summaries;
- a chunk's handle (``c``): the stretch itself;
- a tool call's handle (``t``): its result, the one the host's own rule pairs with it on
  the active record (``pairing``);
- a message's handle (``m``): that message.

A stretch comes back in its collapsed form by default: every user and agent message as
stored (its ``content``, never the host's ``api_content`` sidecar, which carries hook
injections and the memory prefetch), the readable reasoning beside the agent's messages
(plain reasoning, as the summariser reads it: ``summariser_input``), and each tool call
as its handle, its name and its arguments, without its result. A result whose call is
not in the same stretch stands as a pointer to that call, and one the host's rule pairs
with no call says why. ``raw`` puts every result inline. Encrypted reasoning stays out,
and nothing stands where it was.

A handle that is not on the active record is refused with the cause the store records
(``RecordStore.inactive_cause``): the record that stands for it now, the compaction the
host rejected or has not confirmed, or the compaction whose list no longer held it; where
the store records none, it says so.

**Pages.** One mechanism for every tool that returns more than one page: an opaque
``next_page`` token names where the next page begins (a record, a field of it and a
character offset in it). The store is append-only and a chunk never changes, so a token
never goes stale; a token of another store is refused. A page is at most what the host
keeps inline: its spill threshold, computed by the host's own function
(``agent.tool_executor._budget_for_agent``) from the engine's window, as the host
computes it for this result, and compared with the exact string the engine returns
(``results.final_result``). Paging is not a cap: nothing is left out, and a record
larger than a page is split over pages by character offset, each piece saying where it
lies. An image is never split: it is returned as an image, in the ``_multimodal``
envelope, counted against the page by the model table's rule (#21); one larger than a
page, or one no rule counts, stands alone on its page. A page carries no more images than
the host's send path leaves room for in the request (``ImageRoom``).
"""

from __future__ import annotations

import base64
import binascii
import copy
import hashlib
import json
import math
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Optional

from . import pairing as host_pairing
from .handles import CHUNK, DERIVATION, MESSAGE, TOOL_CALL
from .message_content import image_media_type, is_image_part
from .record_store import HANDLE_RE, Cover, RecordStore, Resolved
from .results import final_result
# Readable reasoning is what the summariser reads as readable, one rule (#8, #18).
from .summariser_input import _readable_reasoning as readable_reasoning
from .tokens import CHARS_PER_TOKEN

# 2: a call's handle names (record, position), and a page's records are paired at read
# time (round 5 of #71); a token of version 1 is refused with its own text.
TOKEN_VERSION = 2

# What each status of ``RecordStore.resolve`` tells the agent; "inactive" is told by its
# cause (``_inactive_text``).
_UNRESOLVED = {
    "malformed": ("{handle!r} is not a handle. A handle is a kind letter (m a message, t a tool call, c a chunk, "
                  "s a summary) and eight characters, as the summaries in your context and these tools show them."),
    "unknown": "{handle} is unknown in this store: no message, tool call, chunk or summary has this handle here.",
    "other_session": ("{handle} belongs to another session. A handle resolves only in the session that holds it; "
                      "this session cannot reach another's past."),
    "summary_revision": ("{handle} is the host's rewrite of summary {of} in your context (the row the plugin "
                         "returned, as the host changed it, for example with its task list folded in). Expand {of} "
                         "to read behind the summary."),
}

# Notes on calls and results, from the host's pairing rule (``pairing``).
NO_RESULT = "no result follows this call on the active record"
_UNPAIRED = {
    "unknown": "no call of the message before this result has its host id {id}",
    "taken": ("the call the host's list rule gives the host id {id} was already answered by an earlier result; "
              "the rule drops this one as a stray"),
    "no_id": ("this result carries no host id, so no call pairs with it, and the host's pre-call sanitizer does not "
              "send it to the provider"),
}
_SHARED = ("this message carries {k} calls with the host id {id}, and {j} with it {follow} them; the host's "
           "list rule answers the first call with the first of them and drops the later ones as strays, and its "
           "pre-call sanitizer sends the provider only the first call and result of one id, so the id cannot confirm "
           "which call a result answered")


class ExpansionError(Exception):
    """Something the caller is told instead of a page: a wrong argument, a handle that
    does not resolve, a token that is not this store's."""


def unresolved_message(resolved: Resolved) -> str:
    if resolved.status == "inactive":
        return _inactive_text(resolved)
    return _UNRESOLVED[resolved.status].format(handle=resolved.handle, of=resolved.of)


def _inactive_text(resolved: Resolved) -> str:
    """The store's cause of a handle not being on the active record (round 5 of #71): only
    what the store records, never a guess at what the host did."""
    cause = resolved.cause or {"kind": "none"}
    kind = cause.get("kind")
    if resolved.kind == TOOL_CALL:
        prefix = (f"{resolved.handle} is call {(resolved.position or 0) + 1} ({resolved.name or 'no name'}) of "
                  f"message {resolved.record}. ")
        subject = resolved.record
    else:
        prefix, subject = "", (resolved.record or resolved.handle)
    if kind == "revised":
        text = f"{subject} is a message the host rewrote. It stands on the active record as {cause['active']}, " \
               f"recorded by compaction {cause['compaction']}"
        if cause.get("via"):
            text += f", after {', '.join(cause['via'])}"
        if cause.get("together"):
            text += f", together with {', '.join(cause['together'])}"
        text += f". Expand {cause['active']}."
        if resolved.kind == TOOL_CALL:
            text += " A rewritten message's calls have handles of their own."
    elif kind == "rejected":
        text = (f"{subject} was recorded by compaction {cause['compaction']}, which the host rejected; it never stood "
                f"on the active record.")
    elif kind == "unconfirmed":
        text = (f"{subject} was recorded by compaction {cause['compaction']}, which the host has not confirmed; it is "
                f"not on the active record.")
    elif kind == "left":
        text = (f"{subject} stood on the active record at compaction {cause['stood']}; the host's list at compaction "
                f"{cause['gone']} held neither it nor a rewrite of it.")
    else:
        text = f"{subject} is not on the active record, and the store records no cause."
    return prefix + text


# --- The page limit -------------------------------------------------------------------------

def calls_in_current_message(messages: Any) -> Optional[int]:
    """How many tool calls the assistant message now being answered holds: the last
    assistant message with tool calls in the live list the host hands the engine tool
    (``handle_tool_call(..., messages=messages)``, agent/tool_executor.py:1655; the host
    appends that message before running its calls and each result after it). None where
    the list holds none."""
    if not isinstance(messages, list):
        return None
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "assistant":
            calls = message.get("tool_calls")
            return len(calls) if isinstance(calls, list) and calls else None
    return None


def host_guardrail_margin(tool_name: str) -> int:
    """The most text the host can append to this tool's result before it compares the
    result with its spill threshold, from the host's own templates: its
    ``_append_guardrail_observation`` (run_agent.py:1287-1312 at Hermes d0288be5b3), run by
    ``_commit_tool_result`` before the spill (agent/tool_executor.py:1075-1112), appends up
    to two guidance lines (``append_toolguard_guidance``, agent/tool_guardrails.py:567-572:
    the after-call decision and the identical-streak or cycle halt) and one stall notice
    (``_IDENTICAL_CALL_NOTICE`` or ``_IDENTICAL_CYCLE_NOTICE``, 284-296, joined by two
    newlines). Each is measured at its longest: every decision message of
    ``_DECISION_MESSAGES`` and the failure hint, formatted with this tool's name and counts
    of seven digits (a count is per turn; the digits are this plugin's bound, named).
    Where the host's templates cannot be read, no margin is known and the call is
    refused."""
    try:
        from agent import tool_guardrails as guard  # type: ignore
        big = 10 ** 6
        fields = {"tool_name": tool_name, "count": big, "period": big, "cap": big}
        messages = [text.format(**fields) for text in guard._DECISION_MESSAGES.values()]
        messages.append(guard._tool_failure_recovery_hint(tool_name, big))
        codes = list(guard._DECISION_MESSAGES) + ["same_tool_failure_warning"]
        guidance = max(
            len(guard.append_toolguard_guidance("", guard.ToolGuardrailDecision(
                "halt", code, message, tool_name, big, None)))
            for code in codes for message in messages)
        notice = max(len(guard._IDENTICAL_CALL_NOTICE.format(ordinal=guard._ordinal(big), tool_name=tool_name)),
                     len(guard._IDENTICAL_CYCLE_NOTICE.format(count=big, period=big, tool_name=tool_name)))
    except Exception as exc:
        raise ExpansionError(f"the host's guardrail texts cannot be read ({type(exc).__name__}: {exc}), so the "
                             f"room they take beside a page is not known") from None
    return 2 * guidance + len("\n\n") + notice


def host_page_limit(engine: Any, tool_name: str, messages: Any) -> int:
    """The most characters a page may hold so that the host keeps it inline, from host
    values only (orchestrator ruling on the pre-review of 97a8483, finding 1):

    - the host's threshold for one result, ``_budget_for_agent(agent).resolve_threshold(name)``
      (agent/tool_executor.py:112-125, 1100-1112 at Hermes d0288be5b3), from the window
      the host reads there, ``agent.context_compressor.context_length``, this engine's;
    - the host's budget for one assistant message's results, ``turn_budget``, which
      ``enforce_turn_budget`` applies to the last ``len(tool_calls)`` results of the batch
      (tool_executor.py:1183-1187, 1818-1819, 1902-1904; tools/tool_result_storage.py:283-312),
      divided by the calls of the message being answered;
    - less the host's guardrail texts (``host_guardrail_margin``).

    The host computes the budget once per batch and turns an error of it into its default
    budget; this reads it with the host's own function when the page is built. A batch
    whose other tools return more than their share can still push a page out: that is the
    host's aggregate (#24), named. Where any of it cannot be read, the call is refused."""
    try:
        from agent.tool_executor import _budget_for_agent  # type: ignore
        budget = _budget_for_agent(SimpleNamespace(context_compressor=engine))
        threshold = budget.resolve_threshold(tool_name)
        turn_budget = budget.turn_budget
    except Exception as exc:
        raise ExpansionError(f"the host's spill threshold for {tool_name} cannot be read "
                             f"({type(exc).__name__}: {exc}), so no page size is known") from None
    for name, value in (("threshold", threshold), ("turn budget", turn_budget)):
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ExpansionError(f"the host's {name} for {tool_name} is {value!r}, not a size a page can be "
                                 f"measured against")
    calls = calls_in_current_message(messages)
    if calls is None:
        raise ExpansionError("the host handed no message list with the tool calls being answered, so the "
                             "share of the host's per-message budget a page may take is not known")
    limit = min(int(threshold), int(turn_budget) // calls) - host_guardrail_margin(tool_name)
    if limit <= 0:
        raise ExpansionError(f"{calls} tool calls in one message leave a page no room within the host's budget of "
                             f"{int(turn_budget)} characters for the message; call {tool_name} in fewer at once")
    return limit


# --- Tokens ---------------------------------------------------------------------------------

def encode_token(state: dict) -> str:
    text = json.dumps(state, separators=(",", ":"), sort_keys=True)
    return base64.urlsafe_b64encode(text.encode("utf-8")).decode("ascii").rstrip("=")


def decode_token(token: Any) -> dict:
    if not isinstance(token, str) or not token.strip():
        raise ExpansionError("page must be the next_page token of an earlier result")
    text = token.strip()
    try:
        # Strictly: a character outside the URL-safe alphabet is an error, never skipped
        # (``urlsafe_b64decode`` alone discards them). A standard-alphabet character in
        # the token ("+" or "/") is not ours either.
        if "+" in text or "/" in text:
            raise ValueError("not the URL-safe alphabet")
        standard = (text + "=" * (-len(text) % 4)).replace("-", "+").replace("_", "/")
        state = json.loads(base64.b64decode(standard, validate=True).decode("utf-8"))
    except (binascii.Error, UnicodeDecodeError, ValueError):
        raise ExpansionError("page is not a next_page token of these tools") from None
    if not isinstance(state, dict) or not _plain_int(state.get("v")):
        raise ExpansionError("page is not a next_page token of these tools")
    if 1 <= state["v"] < TOKEN_VERSION:
        raise ExpansionError("page is a token of an earlier version of these tools; start again from the handle, "
                             "without page")
    if state["v"] != TOKEN_VERSION:
        raise ExpansionError("page is not a next_page token of these tools")
    # Every field, by type and range: a garbled token is refused with what is wrong in it.
    checks = (
        ("t", lambda v: isinstance(v, str) and bool(v), "a tool name"),
        ("s", lambda v: isinstance(v, str) and bool(v), "a store's uuid"),
        ("h", lambda v: isinstance(v, str) and HANDLE_RE.fullmatch(v) is not None, "a handle"),
        ("m", lambda v: v in ("raw", "collapsed"), "raw or collapsed"),
        ("i", lambda v: _plain_int(v) and v >= 0, "an item number of 0 or more"),
        ("f", lambda v: _plain_int(v) and v >= -1, "a field number of -1 or more"),
        ("o", lambda v: _plain_int(v) and v >= 0, "a character offset of 0 or more"),
        ("n", lambda v: _plain_int(v) and v >= 1, "a page number of 1 or more"),
        ("r", lambda v: isinstance(v, str) and len(v) == 16 and all(c in "0123456789abcdef" for c in v),
         "the identity of the records paging began on"),
    )
    for key, valid, what in checks:
        if key not in state or not valid(state[key]):
            raise ExpansionError(f"page is a garbled next_page token: its field {key!r} is "
                                 f"{state.get(key)!r}, not {what}")
    if state["f"] == -1 and state["o"] != 0:
        raise ExpansionError("page is a garbled next_page token: an offset inside a whole item")
    return state


def _plain_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True)
class Cursor:
    """Where a page begins: item ``item``; ``field`` -1 for the whole item, else the index
    of one of its fields (``fields_of``), from character ``offset``."""

    item: int = 0
    field: int = -1
    offset: int = 0


# --- Items ----------------------------------------------------------------------------------

@dataclass
class Target:
    """What a handle opens into: a header for every page and the items, in order."""

    header: dict
    items: list = field(default_factory=list)


def _call_parts(call: Any) -> tuple[Any, Any]:
    if isinstance(call, dict) and isinstance(call.get("function"), dict):
        return call["function"].get("name"), call["function"].get("arguments")
    return None, None


def _shared_note(note: tuple) -> str:
    call_id, calls, results = note
    return _SHARED.format(k=calls, id=call_id, j=f"{results} result" if results == 1 else f"{results} results",
                          follow="follows" if results == 1 else "follow")


def _unpaired_note(unpaired: tuple) -> str:
    why, call_id = unpaired
    return _UNPAIRED[why].format(id=call_id)


def _message_item(handle: str, raw: dict, calls: dict[int, str], found: host_pairing.Pairing, *, inline: set,
                  with_results: bool) -> dict:
    """One user or agent message: its content as stored, its readable reasoning beside it,
    each tool call as its handle, name and arguments, and what the host's pairing says of
    it: where several calls share an id, a note; in raw form, where its result is when
    this stretch does not hold it, or that none follows it."""
    item: dict = {"handle": handle, "role": raw.get("role")}
    if raw.get("role") == "assistant":
        reasoning = readable_reasoning(raw)
        if reasoning is not None:
            item["reasoning"] = reasoning
    item["content"] = raw.get("content")
    tool_calls = raw.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        shown = []
        for position, call in enumerate(tool_calls):
            name, arguments = _call_parts(call)
            entry: dict = {"handle": calls.get(position)}
            if name is None and arguments is None:
                entry["call"] = call          # a shape other than the host's: as stored
            else:
                entry["name"] = name
                entry["arguments"] = arguments
            notes = []
            if with_results:
                result = found.answer.get((handle, position))
                if result is None:
                    entry["result"] = None
                    notes.append(NO_RESULT)
                elif result not in inline:
                    entry["result_in"] = result
            if (handle, position) in found.shared_calls:
                notes.append(_shared_note(found.shared_calls[(handle, position)]))
            if notes:
                entry["note"] = "; ".join(notes)
            shown.append(entry)
        item["tool_calls"] = shown
    return item


def _result_item(handle: str, raw: dict, call: Optional[str], *, inline: bool, notes: list) -> dict:
    item: dict = {"handle": handle, "role": raw.get("role"), "result_of": call}
    name = raw.get("name") or raw.get("tool_name")
    if name:
        item["name"] = name
    if inline:
        item["content"] = raw.get("content")
    else:
        content = raw.get("content")
        item["content_chars"] = len(content) if isinstance(content, str) else len(json.dumps(content, ensure_ascii=False))
    if notes:
        item["note"] = "; ".join(notes)
    return item


@dataclass
class Order:
    """The active record's order (``RecordStore.active_units``), for pairing a stretch
    with what decides it (``pairing.window``)."""

    records: list
    index: dict

    @classmethod
    def of(cls, store: RecordStore, cover: Cover) -> "Order":
        records = [record for unit in store.active_units(cover) for record in unit]
        return cls(records, {record: i for i, record in enumerate(records)})

    def pairing(self, store: RecordStore, stretch: list[str]) -> host_pairing.Pairing:
        try:
            return host_pairing.pairing_around(self.records, self.index, stretch, store.record_roles,
                                               store.records_raw)
        except host_pairing.PairingUnavailable as exc:
            raise ExpansionError(str(exc)) from None


def _records_items(store: RecordStore, order: Order, records: list[tuple[str, dict]], *, raw: bool) -> list[dict]:
    """The items of a run of records, collapsed or raw, paired by the host's rule over the
    active record (``pairing``)."""
    handles = [handle for handle, _raw in records]
    found = order.pairing(store, handles)
    calls = store.tool_calls_of(handles + [record for record, _position in found.result_of.values()])
    held = set(handles)
    inline = {h for h, r in records if r.get("role") == "tool"} if raw else set()
    items: list[dict] = []
    for handle, message in records:
        if message.get("role") == "tool":
            paired = found.result_of.get(handle)
            call = calls.get(paired[0], {}).get(paired[1]) if paired else None
            notes = []
            if handle in found.unpaired:
                notes.append(_unpaired_note(found.unpaired[handle]))
            if handle in found.shared_results:
                notes.append(_shared_note(found.shared_results[handle]))
            if raw:
                items.append(_result_item(handle, message, call, inline=True, notes=notes))
            elif paired is None or paired[0] not in held:
                if paired is not None:
                    notes.insert(0, "the result of a call this stretch does not hold; expand its own handle to "
                                    "read it")
                items.append(_result_item(handle, message, call, inline=False, notes=notes))
            continue
        items.append(_message_item(handle, message, calls.get(handle, {}), found, inline=inline, with_results=raw))
    return items


def target_for(store: RecordStore, order: Order, resolved: Resolved, *, raw: bool) -> Target:
    """What ``resolved`` opens into (the module docstring), read from the store. Called
    inside the caller's read transaction (``RecordStore.snapshot``)."""
    handle = resolved.handle
    form = "raw" if raw else "collapsed"
    if resolved.kind == CHUNK:
        return Target({"handle": handle, "kind": "chunk", "form": form},
                      _records_items(store, order, store.chunk_records(handle), raw=raw))
    if resolved.kind == DERIVATION:
        sources = store.derivation_sources(handle)
        header: dict = {"handle": handle, "kind": "summary", "form": form}
        if sources and all(chunk for chunk, _derivation in sources):
            chunks = [str(chunk) for chunk, _derivation in sources]
            header["chunks"] = chunks
            if len(chunks) == 1:
                return Target(header, _records_items(store, order, store.chunk_records(chunks[0]), raw=raw))
            # A summary written from the raw of several chunks (#34 D6, "raw"): all of them,
            # each opened by a marker naming its chunk and that chunk's own summary.
            leaves = store.leaf_summaries(chunks, _all_summaries(store, chunks))
            items: list[dict] = []
            for chunk in chunks:
                items.append({"chunk": chunk, "summary": leaves.get(chunk)})
                items.extend(_records_items(store, order, store.chunk_records(chunk), raw=raw))
            return Target(header, items)
        # A summary written from summaries (#34 D6, "summaries"): one layer down.
        header["form"] = "summaries"
        items = []
        for chunk, derivation in sources:
            if derivation:
                items.append({"summary": derivation, "text": store.derivation_text(derivation),
                              "note": "a summary: a description of what happened, not what happened; expand its "
                                      "handle to read behind it"})
            else:
                items.append({"chunk": chunk})
        return Target(header, items)
    if resolved.kind == TOOL_CALL:
        # The call's result is the one the host's rule pairs with it on the active record
        # (round 5 of #71): read now, never stored.
        record, position = resolved.record, int(resolved.position or 0)
        header = {"handle": handle, "kind": "tool_call", "form": "raw", "name": resolved.name, "call_in": record}
        found = order.pairing(store, [record])
        notes = []
        if (record, position) in found.shared_calls:
            notes.append(_shared_note(found.shared_calls[(record, position)]))
        result = found.answer.get((record, position))
        if result is None:
            header["result"] = None
            header["note"] = "; ".join([NO_RESULT] + notes)
            return Target(header, [])
        if notes:
            header["note"] = "; ".join(notes)
        return Target(header, [_result_item(result, store.record_raw(result) or {}, handle, inline=True, notes=[])])
    if resolved.kind == MESSAGE:
        return Target({"handle": handle, "kind": "message", "form": form},
                      _records_items(store, order, [(handle, store.record_raw(handle) or {})], raw=True))
    raise ExpansionError(f"{handle} is not a handle this tool opens")


def _all_summaries(store: RecordStore, chunks: list[str]) -> list[str]:
    """The summaries whose only source is one of these chunks."""
    if not chunks:
        return []
    marks = ",".join("?" * len(chunks))
    return [str(d) for (d,) in store._q(
        f"SELECT s.derivation FROM derivation_sources s WHERE s.chunk IN ({marks}) AND s.ordinal = 0 "
        f"AND NOT EXISTS (SELECT 1 FROM derivation_sources o WHERE o.derivation = s.derivation AND o.ordinal > 0) "
        f"ORDER BY s.rowid", chunks)]


# --- Fields and pieces ------------------------------------------------------------------------

def fields_of(item: dict) -> list[tuple[str, Any]]:
    """The fields an item is split into when it does not fit on a page by itself, in
    order: its reasoning, its content (a string, or each part of a list, or each part and
    key of an envelope), and each tool call's arguments. Deterministic, so a cursor
    naming a field and an offset finds the same place on every call."""
    fields: list[tuple[str, Any]] = []
    if isinstance(item.get("reasoning"), str):
        fields.append(("reasoning", item["reasoning"]))
    if "text" in item and isinstance(item.get("text"), str):
        fields.append(("text", item["text"]))
    if "content" in item:
        content = item["content"]
        if isinstance(content, list):
            fields.extend((f"content[{i}]", part) for i, part in enumerate(content))
        elif isinstance(content, dict) and isinstance(content.get("content"), list):
            for key, value in content.items():
                if key == "content":
                    fields.extend((f"content.content[{i}]", part) for i, part in enumerate(value))
                else:
                    fields.append((f"content.{key}", value))
        else:
            fields.append(("content", content))
    for index, call in enumerate(item.get("tool_calls") or []):
        if "arguments" in call:
            fields.append((f"tool_calls[{index}].arguments", call["arguments"]))
        elif "call" in call:
            fields.append((f"tool_calls[{index}].call", call["call"]))
    return fields


def _piece(item: dict, path: str, *, value: Any = None, text: Optional[str] = None, offset: int = 0,
           total: int = 0, json_text: bool = False) -> dict:
    """One field of an item that does not fit on a page by itself."""
    piece: dict = {key: item[key] for key in ("handle", "role", "result_of", "chunk", "summary") if key in item}
    piece["field"] = path
    if path.startswith("tool_calls["):
        index = int(path[len("tool_calls["):path.index("]")])
        call = (item.get("tool_calls") or [])[index]
        piece["tool_call"] = call.get("handle")
        # Everything the call says beside its arguments (its name, and in a raw stretch
        # where its result is or that none is recorded) goes with each piece.
        for key, said in call.items():
            if key not in ("handle", "arguments", "call"):
                piece[key] = said
    if text is None:
        piece["value"] = value
    else:
        piece["offset"] = offset
        piece["chars"] = total
        piece["text"] = text
        if json_text:
            piece["json"] = True          # the field's value as JSON text, split by offset
    return piece


# --- Pages -----------------------------------------------------------------------------------

@dataclass
class _Rendered:
    text: str
    images: list
    labels: list
    image_tokens: Optional[int]           # None where an image on it has no count
    summary: str


def canonical_image_part(part: dict) -> tuple[Optional[dict], str]:
    """An image part in the host's canonical list-content form, ``{"type": "image_url",
    "image_url": {"url": …}}``, which every converter of the host takes: the Chat
    Completions path as it is, the Anthropic converter by ``_image_block_from_openai_url``
    (agent/anthropic_message_convert.py:170-187 at Hermes d0288be5b3), the Responses
    converter by ``_iter_content_parts``/``_input_image_part`` (agent/codex_responses_adapter.py
    92, 205-247), and the send path's own image test (``_IMAGE_PART_TYPES``,
    agent/vision_message_prep.py:24). The host has no helper for the other direction; the
    conversion is by the shapes' own definitions: an Anthropic ``image`` block's base64
    source becomes a data URL, its URL source that URL; a Responses ``input_image`` its
    ``image_url``. Returns (the part, "") or (None, why) where no URL can be made (a
    Responses ``file_id``, a source of another type): such an image is refused visibly
    (Codex review of 9a787bd, finding 4)."""
    kind = part.get("type")
    detail = None
    url: Any = None
    if kind in ("image_url", "input_image"):
        url = part.get("image_url")
        if isinstance(url, dict):
            detail = url.get("detail")
            url = url.get("url")
        detail = detail or part.get("detail")
        if not isinstance(url, str) or not url:
            return None, (f"stored as a {kind} part without an image URL"
                          + (" (a file id only the provider holds)" if part.get("file_id") else ""))
    elif kind == "image":
        source = part.get("source")
        if isinstance(source, dict) and source.get("type") == "base64" and isinstance(source.get("data"), str):
            url = f"data:{source.get('media_type') or 'image/jpeg'};base64,{source['data']}"
        elif isinstance(source, dict) and source.get("type") == "url" and isinstance(source.get("url"), str):
            url = source["url"]
        else:
            kind_of = source.get("type") if isinstance(source, dict) else type(source).__name__
            return None, f"stored as an image block with a source of type {kind_of!r}, which gives no URL"
    else:
        return None, f"stored as a part of type {kind!r}"
    image_url: dict = {"url": url}
    if detail:
        image_url["detail"] = detail
    return {"type": "image_url", "image_url": image_url}, ""


def is_content_image(path: str, value: Any) -> bool:
    """An image is a part of a message's content (a content list, or the ``_multimodal``
    envelope's list), never a value elsewhere: an object in a tool call's arguments with
    ``"type": "image"`` is arguments (Codex review of ca0969d, finding 3)."""
    return (path.startswith("content[") or path.startswith("content.content[")) and \
        isinstance(value, dict) and is_image_part(value)


def _image_or_mark(part: dict, images: list, describe: list) -> dict:
    canonical, why = canonical_image_part(part)
    if canonical is None:
        # Refused visibly, never dropped: the page says an image stands here and why it is
        # not shown, with the file id where the part has one.
        mark = {"type": "image", "not_shown": f"an image ({image_media_type(part)}) {why}; "
                                              f"the host's converters take no such part"}
        if part.get("file_id"):
            mark["file_id"] = part["file_id"]
        return mark
    images.append(canonical)
    describe.append(image_media_type(part))
    return {"type": "image", "image": len(images)}


def _replace_images(page_items: list, images: list, describe: list) -> list:
    """The page's items with each image part of a message's content replaced by its mark,
    the images collected in order. Only content parts are images (``is_content_image``):
    an item's ``content`` list, the envelope's ``content`` list, and a piece's ``value``
    where the piece is one of those parts. A piece the request has no room for
    (``PageBuilder.build``) carries the reason in ``held`` and is marked, not shown."""
    def parts(content: list) -> list:
        return [_image_or_mark(p, images, describe) if isinstance(p, dict) and is_image_part(p) else p
                for p in content]

    replaced: list = []
    for item in page_items:
        item = dict(item)
        held = item.pop("held", None)
        content = item.get("content")
        if isinstance(content, list):
            item["content"] = parts(content)
        elif isinstance(content, dict) and isinstance(content.get("content"), list):
            item["content"] = dict(content, content=parts(content["content"]))
        if "value" in item and is_content_image(str(item.get("field") or ""), item["value"]):
            if held is not None:
                mark = {"type": "image", "not_shown": f"an image ({image_media_type(item['value'])}): {held}"}
                if item["value"].get("file_id"):
                    mark["file_id"] = item["value"]["file_id"]
                item["value"] = mark
            else:
                item["value"] = _image_or_mark(item["value"], images, describe)
        replaced.append(item)
    return replaced


@dataclass(frozen=True)
class ImageRoom:
    """The room the host's send path leaves this page's images in the request that will
    carry it (the orchestrator's ruling on round 5 of #71, 5(a) as changed).

    The host retires image-bearing tool results on its send path by one policy,
    ``outbound_image_retire_count`` (agent/image_eviction_policy.py:33 at Hermes
    d0288be5b3; ``OUTBOUND_IMAGE_LIMIT`` blocks and ``OUTBOUND_IMAGE_BUDGET_BYTES``,
    26-27), applied to every request whatever its wire (``evict_stale_outbound_tool_images``,
    agent/context_compressor.py:1478-1510) and again by blocks by the Anthropic converter
    (``_evict_old_screenshots``, agent/anthropic_message_convert.py:605). Counted, over the
    live list the host hands the tool and measured by the host's ``_image_payload``
    (context_compressor.py:1452): the uploads the host reserves (images outside tool
    results), the older carriers, the earlier results of this message already in the list,
    and our page. Of the calls of this message still to come, only this tool's are known
    image sources: they share the room equally (``share``, this one included). A page is
    admitted where, with ``share`` carriers like it at the newest end, the host retires no
    more carriers than it retires without them, so never one of them.

    Another tool's later result may carry images this cannot foresee; that retirement is
    the host's, visible in its own placeholder, and where the message has such calls the
    page says so (``note``). Where the host cannot be read, a page carries one image and
    says why (unknown is never unlimited)."""

    retire: Optional[Callable]
    measure: Optional[Callable[[dict], tuple]]
    older: tuple = ()               # (blocks, bytes) of each carrier in the list, newest first
    reserved: tuple = (0, 0)        # (blocks, bytes) the host reserves
    share: int = 1
    note: str = ""
    limit: Optional[int] = None     # the host's OUTBOUND_IMAGE_LIMIT
    budget: Optional[int] = None    # the host's OUTBOUND_IMAGE_BUDGET_BYTES

    def _retired(self, carriers: list) -> int:
        return self.retire([b for b, _s in carriers], self.reserved[0],
                           carrier_bytes_newest_first=[s for _b, s in carriers], reserved_bytes=self.reserved[1])

    def baseline(self) -> int:
        return self._retired(list(self.older))

    def admits(self, images: list) -> bool:
        if not images:
            return True
        if self.retire is None or self.measure is None:
            return len(images) <= 1
        blocks, size = self.measure({"role": "tool", "content": list(images)})
        return self._retired([(blocks, size)] * self.share + list(self.older)) <= self.baseline()

    def why_not(self, image: dict, handle: str, media: str) -> str:
        """Why an image that no page can carry now is held: what the host's own function
        would do with the page added (ruling on the pre-review of 0771477, B)."""
        blocks, size = self.measure({"role": "tool", "content": [image]})
        if self.budget is not None and size > self.budget:
            # The host strips it whatever else the request holds.
            return (f"not shown: this image is {size} bytes as the host measures it, over the host's budget of "
                    f"{self.budget} bytes for the images of one request, so the host's send path would strip it "
                    f"from any page; no page can carry this image in this host. The store still holds it: {media}, "
                    f"{size} bytes, in message {handle}")
        if self._retired([(blocks, size)]) > 0:
            # With no earlier carrier at all the host would still retire the page: the images
            # it reserves (uploads, which it never retires) fill the ceiling.
            count, taken = self.reserved
            if self.limit is not None and count + blocks > self.limit:
                return (f"not shown: the {count} images the host reserves in this request (uploads, which it never "
                        f"retires) fill its ceiling of {self.limit} images, so the host would retire this page; the "
                        f"image shows once those uploads leave the context")
            return (f"not shown: the images the host reserves in this request (uploads, which it never retires) take "
                    f"{taken} bytes of its budget of {self.budget}, which leaves no room for this image's {size} "
                    f"bytes, so the host would retire this page; the image shows once those uploads leave the context")
        carriers = [(blocks, size)] * self.share + list(self.older)
        retired, base = self._retired(carriers), self.baseline()
        # The host retires the oldest carriers first; the list is newest first, so the ones
        # this page adds to the retirement are those just before the ones retired anyway.
        extra = range(len(carriers) - retired, len(carriers) - base)
        if any(i >= self.share for i in extra):
            shared = (f", shared with {self.share - 1} more {'call' if self.share == 2 else 'calls'} of this tool "
                      f"in this message") if self.share > 1 else ""
            reserved, older = self.reserved[0], sum(b for b, _s in self.older)
            return (f"not shown: the host's send path has no room for it in this request without retiring earlier "
                    f"images ({reserved} {'image' if reserved == 1 else 'images'} it reserves and {older} in earlier "
                    f"results stand in it{shared}); it stays in the store under this handle")
        return (f"not shown: the {self.share} calls of this tool in this message share the room left in this "
                f"request, and this image does not fit this call's share; the host would retire one of these "
                f"pages; it stays in the store under this handle")


_OTHER_CALLS_NOTE = ("the host may retire this page's images if other results of this turn carry images; expand "
                     "again later to see them")


def host_image_room(messages: Any, tool_name: str) -> ImageRoom:
    """The ``ImageRoom`` of this call, from the live list and the host's own functions."""
    try:
        from agent.context_compressor import _image_payload  # type: ignore
        from agent.image_eviction_policy import (  # type: ignore
            OUTBOUND_IMAGE_BUDGET_BYTES,
            OUTBOUND_IMAGE_LIMIT,
            outbound_image_retire_count,
        )
        call_variants, result_variants, _coalesce = host_pairing.host_alias_helpers()
        if not isinstance(messages, list):
            raise ValueError("no message list")
        older: list = []
        reserved = [0, 0]
        for message in reversed(messages):
            if not isinstance(message, dict):
                continue
            blocks, size = _image_payload(message)
            if not blocks:
                continue
            if message.get("role") == "tool":
                older.append((int(blocks), int(size)))
            else:
                reserved[0] += int(blocks)
                reserved[1] += int(size)
        # The message being answered, and which of its calls no result in the list answers.
        at = next(i for i in range(len(messages) - 1, -1, -1)
                  if isinstance(messages[i], dict) and messages[i].get("role") == "assistant")
        answered: set = set()
        for message in messages[at + 1:]:
            if isinstance(message, dict) and message.get("role") == "tool":
                answered |= set(result_variants(message.get("tool_call_id")))
        pending = [call for call in (messages[at].get("tool_calls") or [])
                   if not (set(call_variants(call)) & answered)]
        ours = sum(1 for call in pending if _call_parts(call)[0] == tool_name)
        others = len(pending) - ours
        return ImageRoom(outbound_image_retire_count, _image_payload, tuple(older), tuple(reserved),
                         max(1, ours), _OTHER_CALLS_NOTE if others else "", int(OUTBOUND_IMAGE_LIMIT),
                         int(OUTBOUND_IMAGE_BUDGET_BYTES))
    except Exception as exc:
        return ImageRoom(None, None, note=f"the host's send-path image ceiling cannot be read ({type(exc).__name__}: "
                                          f"{exc}), so this page carries at most one image")


class PageBuilder:
    """Fills pages of ``target`` from a cursor, each at most ``limit`` characters as the
    host measures the result it receives."""

    def __init__(self, target: Target, *, limit: int, token_state: dict,
                 image_tokens: Callable[[dict], Optional[int]], image_room: ImageRoom):
        self.target = target
        self.limit = limit
        self.token_state = token_state
        self.image_tokens = image_tokens
        self.image_room = image_room
        self._fields: dict[int, list] = {}

    def _fields_of(self, index: int) -> list:
        if index not in self._fields:
            self._fields[index] = fields_of(self.target.items[index])
        return self._fields[index]

    def _payload(self, page_items: list, page: int, next_page: Optional[str], images: bool) -> dict:
        payload = {**self.target.header, "items_total": len(self.target.items), "page": page,
                   "next_page": next_page}
        if images and self.image_room.note:
            payload["images_note"] = self.image_room.note
        payload["items"] = page_items
        return payload

    def render(self, page_items: list, page: int, next_page: Optional[str]) -> _Rendered:
        images: list = []
        media: list = []
        items = _replace_images(copy.deepcopy(page_items), images, media)
        text = final_result(self._payload(items, page, next_page, bool(images)))
        labels: list = []
        counted: Optional[int] = 0
        notes = []
        for number, part in enumerate(images, start=1):
            tokens = self.image_tokens(part)
            counted = None if tokens is None or counted is None else counted + tokens
            # An image no rule counts is said to be uncounted, and it stands alone (#21, #35).
            count = (f"{tokens} tokens by the model table's rule" if tokens is not None
                     else "not counted: no image rule is known for this model, so it stands alone on its page")
            label = (f"[image {number} of this page ({media[number - 1]}; {count}), which the page above names as "
                     f"image {number}]")
            labels.append(label)
            notes.append(f"[image {number} of this page ({media[number - 1]}) is not shown in this text]")
        summary = text + ("\n" + "\n".join(notes) if notes else "")
        return _Rendered(text, images, labels, counted, summary)

    def fits(self, page_items: list, page: int) -> bool:
        rendered = self.render(page_items, page, self._longest_token(page))
        if not rendered.images:
            return len(rendered.text) <= self.limit
        if rendered.image_tokens is None:
            return False
        if not self.image_room.admits(rendered.images):
            return False
        text_chars = len(rendered.text) + sum(len(label) for label in rendered.labels)
        return (len(rendered.summary) <= self.limit
                and text_chars + rendered.image_tokens * CHARS_PER_TOKEN <= self.limit)

    def _room_alone(self, piece: dict) -> bool:
        images: list = []
        _replace_images([dict(piece)], images, [])
        return self.image_room.admits(images)

    def _longest_token(self, page: int) -> str:
        """A token at least as long as any this page could carry: the page is fitted with
        it, so the real one never makes the page longer than it was measured."""
        big = 10 ** 15
        return encode_token({**self.token_state, "i": big, "f": big, "o": big, "n": page + big})

    def build(self, cursor: Cursor, page: int) -> tuple[Any, Optional[Cursor]]:
        """One page from ``cursor``: the result the engine returns, and the cursor of the
        next page, or None where this page is the last."""
        items = self.target.items
        page_items: list = []
        i, f, o = cursor.item, cursor.field, cursor.offset
        while i < len(items):
            item = items[i]
            if f < 0:
                if self.fits(page_items + [item], page):
                    page_items.append(item)
                    i, f, o = i + 1, -1, 0
                    continue
                if page_items:
                    break
                f, o = 0, 0          # it does not fit alone: its fields, piece by piece
                continue
            fields = self._fields_of(i)
            if f >= len(fields):
                i, f, o = i + 1, -1, 0
                continue
            path, value = fields[f]
            if is_content_image(path, value):
                piece = _piece(item, path, value=value)
                if self.fits(page_items + [piece], page) or not page_items:
                    if not self._room_alone(piece):
                        # The request has no room for it even alone: marked, never sent
                        # to be retired unseen (``ImageRoom``).
                        piece["held"] = self.image_room.why_not(canonical_image_part(value)[0], str(item.get("handle")),
                                                                image_media_type(value))
                        if not self.fits(page_items + [piece], page) and page_items:
                            break
                    # An image larger than a page, or one no rule counts, stands alone.
                    page_items.append(piece)
                    f, o = f + 1, 0
                    if not self.fits(page_items, page):
                        break
                    continue
                break
            json_text = not isinstance(value, str)
            if json_text and o == 0:
                piece = _piece(item, path, value=value)
                if self.fits(page_items + [piece], page):
                    page_items.append(piece)
                    f, o = f + 1, 0
                    continue
                if page_items:
                    break
            text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
            if not text:
                piece = _piece(item, path, text="", offset=0, total=0, json_text=json_text)
                if not self.fits(page_items + [piece], page) and page_items:
                    break
                page_items.append(piece)
                f, o = f + 1, 0
                continue
            if o >= len(text):
                f, o = f + 1, 0
                continue
            length = self._longest_slice(page_items, item, path, text, o, page, json_text)
            if length == 0:
                if page_items:
                    break
                raise ExpansionError(f"a page of {self.limit} characters cannot hold one character of {path} "
                                     f"beside the page's own fields")
            page_items.append(_piece(item, path, text=text[o:o + length], offset=o, total=len(text),
                                     json_text=json_text))
            o += length
            if o >= len(text):
                f, o = f + 1, 0
                continue
            break
        if i < len(items) and f >= 0 and f >= len(self._fields_of(i)):
            # A page that ends with an item's last field (an image standing alone, for
            # one) continues at the next item: a token never names a field past the end.
            i, f, o = i + 1, -1, 0
        next_cursor = Cursor(i, f, o) if i < len(items) else None
        next_page = encode_token({**self.token_state, "i": i, "f": f, "o": o, "n": page + 1}) \
            if next_cursor is not None else None
        rendered = self.render(page_items, page, next_page)
        if not rendered.images:
            return rendered.text, next_cursor
        content: list = [{"type": "text", "text": rendered.text}]
        for label, part in zip(rendered.labels, rendered.images):
            content.append({"type": "text", "text": label})
            content.append(part)
        return {"_multimodal": True, "content": content, "text_summary": rendered.summary}, next_cursor

    def _longest_slice(self, page_items: list, item: dict, path: str, text: str, offset: int, page: int,
                       json_text: bool) -> int:
        low, high, best = 1, len(text) - offset, 0
        while low <= high:
            middle = (low + high) // 2
            piece = _piece(item, path, text=text[offset:offset + middle], offset=offset, total=len(text),
                           json_text=json_text)
            if self.fits(page_items + [piece], page):
                best, low = middle, middle + 1
            else:
                high = middle - 1
        return best


def records_identity(target: Target) -> str:
    """The identity of what a target's items stand on, in order: each item's record handle
    (or the chunk or summary a marker names). Sixteen hex characters of its SHA-256."""
    names = [item.get("handle") or item.get("chunk") or item.get("summary") for item in target.items]
    return hashlib.sha256(json.dumps(names).encode("utf-8")).hexdigest()[:16]


def expand(engine: Any, args: dict, *, messages: Any = None, tool_name: str = "lcm_expand") -> Any:
    """The ``lcm_expand`` tool: one page of what a handle opens into. ``messages`` is the
    live list the host hands the engine tool; the page's size depends on it
    (``host_page_limit``)."""
    session = engine.current_session_id          # the caller's session, read once (#20)
    if not session:
        raise ExpansionError("this engine copy is bound to no session of the plugin, so no handle resolves")
    raw = args.get("raw", False)
    if not isinstance(raw, bool):
        raise ExpansionError("raw must be true or false")
    handle = args.get("handle")
    page_arg = args.get("page")
    state = decode_token(page_arg) if page_arg is not None else None
    if state is not None:
        if state.get("t") != tool_name:
            raise ExpansionError("page is a token of another tool")
        if handle is not None and str(handle).strip() != state.get("h"):
            raise ExpansionError(f"page is a token of {state.get('h')}, not of {handle}")
        # The token decides the form. raw=false is the schema's default, sent by some
        # callers with every call; only raw=true beside a collapsed token asks for another
        # form than the token's (finding 5 of the pre-review of 97a8483).
        if raw is True and state["m"] != "raw":
            raise ExpansionError("page is a token of the collapsed form; raw=true starts a raw expansion without "
                                 "page, from its first page")
        handle = state["h"]
        raw = state["m"] == "raw"
    if handle is None:
        raise ExpansionError("handle is required: the handle of a summary, chunk, tool call or message")
    limit = host_page_limit(engine, tool_name, messages)
    records: RecordStore = engine._records
    with records.snapshot():
        store_uuid = records.identity().get("store_uuid")
        if state is not None and state.get("s") != store_uuid:
            raise ExpansionError("page is a token of another store: the store it was issued by is not this one")
        cover = records.cover(session)
        resolved = records.resolve(str(handle), session, cover)
        if resolved.status != "ok":
            raise ExpansionError(unresolved_message(resolved))
        target = target_for(records, Order.of(records, cover), resolved, raw=bool(raw))
    # The records paging began on (Codex review of ca0969d, finding 2): a token carries
    # their identity, and a later page on other records (the active form of a message
    # changed, a result the host rewrote since) is refused, never spliced.
    identity = records_identity(target)
    if state is not None and state["r"] != identity:
        raise ExpansionError("the content changed since page 1 (the host rewrote a record it holds, or a later "
                             "compaction replaced it); start again from the handle, without page")
    token_state = {"v": TOKEN_VERSION, "t": tool_name, "s": store_uuid, "h": resolved.handle,
                   "m": "raw" if raw else "collapsed", "r": identity}
    cursor = Cursor(state["i"], state["f"], state["o"]) if state else Cursor()
    page = state["n"] if state else 1
    if cursor.item >= len(target.items) and not (cursor.item == 0 and not target.items):
        raise ExpansionError("page is a garbled next_page token: it points past the end of what this handle "
                             "opens into")
    if cursor.field >= 0:
        fields = fields_of(target.items[cursor.item])
        if cursor.field == len(fields) and cursor.offset == 0:
            # Just past an item's last field (a token of 3a4e019 after an image that stood
            # alone): the next item (finding 1 of the Codex review of 3a4e019).
            cursor = Cursor(cursor.item + 1, -1, 0)
        elif cursor.field >= len(fields):
            raise ExpansionError("page is a garbled next_page token: it names a field this item does not have")
        else:
            path, value = fields[cursor.field]
            length = len(value) if isinstance(value, str) else len(json.dumps(value, ensure_ascii=False))
            if cursor.offset > length or (cursor.offset and is_content_image(path, value)):
                raise ExpansionError("page is a garbled next_page token: its offset lies outside the field it "
                                     "names")
    estimator = engine._estimator()
    builder = PageBuilder(target, limit=limit, token_state=token_state, image_tokens=estimator.image,
                          image_room=host_image_room(messages, tool_name))
    result, _next = builder.build(cursor, page)
    return result
