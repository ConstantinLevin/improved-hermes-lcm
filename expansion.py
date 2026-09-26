"""Looking behind a handle (#18): what expansion returns, and the one page mechanism.

**What a handle opens into** (manifesto, "Looking behind a handle"; #29 W5; #34 D6):

- a summary's handle (``s``): the chunk it stands for; a summary written from the raw of
  several chunks, all of them, with the leaf summary of each named beside it as a
  lookup; a summary written from summaries, one layer: those summaries;
- a chunk's handle (``c``): the stretch itself;
- a tool call's handle (``t``): its result, the stored result of its exact id in its
  block (``pairing``); where the store cannot tell which result answered which call (calls
  of one block sharing an id or an alias), every result of that group, attributed to none;
- a message's handle (``m``): that message.

A stretch comes back in its collapsed form by default: every user and agent message by
the host's per-row rules (``summariser_input.host_row_before_fill``, then only the plugin's
own ``item_message``: the host's sidecar in place of ``content`` where the host sends it, its
bookkeeping popped, the encrypted items withheld), less the host's ``_``-prefixed in-process
markers, every other key carried; the
readable reasoning beside the agent's messages; and each tool call as its handle, its name
and its arguments and every other key but the host's ids, without its result. The host's
fill of an empty message is quoted in a note, never shown as content. A result whose call
is not in the same stretch stands as a pointer to that call; a call without a result, a
result without a call and a group the store cannot pair say so, as facts of the store.
What the provider received of calls and results is not reproduced (``pairing``). ``raw`` puts
every result inline and adds the host's native carriers of a message. Encrypted reasoning
stays out, and nothing stands where it was. Every image part of a message's content stands
as a mark: its media type, its size as the host measures it and the message's handle, with
the text ``IMAGE_MARK``; image delivery is a later PR. The copies of repeated injections in the host's
sidecar are not left out: the host records no producer (#36).

A handle that is not on the active record is refused with the cause the store records
(``RecordStore.inactive_cause``): the record that stands for it now, the compaction the
host rejected or has not confirmed, or the compaction whose list no longer held it; where
the store records none, it says so.

**Pages.** One mechanism for every tool that returns more than one page: an opaque
``next_page`` token names where the next page begins (a record, a field of it and a
character offset in it) and the identity of everything the handle opens into; a later
page on anything else is refused, and so is a token of another store. A page is at most what the host
keeps inline: its spill threshold, computed by the host's own function
(``agent.tool_executor._budget_for_agent``) from the engine's window, as the host
computes it for this result, and compared with the exact string the engine returns
(``results.final_result``). Paging is not a cap: nothing is left out, and a record
larger than a page is split over pages by character offset, each piece saying where it
lies. Every page is measured as it will be returned, with the real ``next_page`` of the
place it ends at, or null (``PageBuilder.measure``). What cannot be split (an item without
fields, an image's mark, an empty field, the annotations every piece carries, the header
every page carries) stands alone on the next page if it fits there; where it does not fit
even alone, the page is refused, naming it (or the header, where the header alone does not
fit), its size and the page's limit. Nothing is skipped.
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
from .message_content import image_media_type
from .record_store import HANDLE_RE, Cover, RecordStore, Resolved
from .results import final_result
# Readable reasoning is what the summariser reads as readable, one rule (#8, #18).
from .summariser_input import _readable_reasoning as readable_reasoning
from .summariser_input import (
    HostUnavailable,
    host_fill_text,
    host_reasoning_pad,
    host_row_before_fill,
    item_message,
)

# 3: an item is the message as the host sent it and a piece carries all of its item's
# annotations (the plan of #71, §5), so fields and offsets moved; a token of an earlier
# version is refused with its own text.
TOKEN_VERSION = 3

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

# Notes on calls and results: facts of the store (``pairing``), never what the host sends.
NO_RESULT = "no result follows this call on the active record"
_STRAY = {
    "unknown": "no call of the message before this result has its host id {id}",
    "no_id": "this result carries no host id, so no call pairs with it",
}
_GROUP = ("{k} calls and {j} results in this stretch carry the host id {id}; the store cannot tell which result "
          "answered which call, and what the provider received of them depends on the host's sanitizer and on the "
          "session's route, which this plugin does not reproduce")
_FILL = "the host sends this empty message to the provider with its own stand-in as content: {content}"
_FILL_UNSURE = ("whether the host sends this empty message with its own stand-in {content} depends on its "
                "reasoning-echo setting (model.reasoning_echo), which the plugin cannot read")



def _group_note(group: host_pairing.Group) -> str:
    """Rule 2's note: one id, or the overlapping ids of the group's calls."""
    ids = [str(i) for i in group.ids]
    said = ids[0] if len(ids) == 1 else ", ".join(ids[:-1]) + " or " + ids[-1] + " (overlapping)"
    return _GROUP.format(k=len(group.calls), j=len(group.results), id=said)


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


@dataclass(frozen=True)
class PageLimit:
    """A page's limit and where it comes from (``host_page_limit``), for a refusal to name."""

    limit: int
    threshold: int
    turn_budget: int
    calls: int
    margin: int

    @property
    def alone(self) -> int:
        """The limit when the call stands alone in its message."""
        return min(self.threshold, self.turn_budget) - self.margin

    def origin(self) -> str:
        if self.turn_budget // self.calls < self.threshold:
            return (f"the host's per-message budget divided among the {self.calls} calls of this message, less its "
                    f"margin")
        return "the host's threshold for one result, less its margin"


def host_page_limit(engine: Any, tool_name: str, messages: Any) -> int:
    """The limit of ``host_page_limits``."""
    return host_page_limits(engine, tool_name, messages).limit


def host_page_limits(engine: Any, tool_name: str, messages: Any) -> PageLimit:
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
    margin = host_guardrail_margin(tool_name)
    limit = min(int(threshold), int(turn_budget) // calls) - margin
    if limit <= 0:
        raise ExpansionError(f"{calls} tool calls in one message leave a page no room within the host's budget of "
                             f"{int(turn_budget)} characters for the message; call {tool_name} in fewer at once")
    return PageLimit(limit, int(threshold), int(turn_budget), calls, margin)


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


@dataclass
class Item:
    """One item of what a handle opens into, as a pair (the plan of #71, §5.2): its
    ``annotations``, what the plugin says of it (``handle``, ``role``, ``result_of``,
    ``note``, ``content_chars``, ``chunk``, ``summary``, ``summaries``), which every piece of it
    carries; and
    its ``fields``, the message's own keys (§5.1), each of which becomes a piece when the
    item does not fit on a page by itself. A piece is its item by complement, never a
    hand-picked list."""

    annotations: dict
    fields: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {**self.annotations, **self.fields}


# Host keys an item leaves out, with the reason (the plan of #71, §5.1). What is not named
# here is carried, a key the plugin does not know included.
_ID_KEYS = ("id", "call_id", "response_item_id")          # the host's ids: not identities; the handle replaces them
_NATIVE_CARRIERS = ("anthropic_content_blocks", "bedrock_content_blocks",
                    "codex_message_items")                  # re-encodings of the message: carried in raw form only


def _call_entry(handle: Optional[str], call: Any) -> dict:
    """A tool call as an item shows it: its handle, its name and arguments, and every other
    key of the call except the host's ids and a withheld signature (the provider's thought
    signature, encrypted, as the Reasoning paragraph says)."""
    entry: dict = {"handle": handle}
    if not isinstance(call, dict):
        entry["call"] = call                                  # a shape other than the host's: as stored
        return entry
    for key, value in call.items():
        if key in _ID_KEYS:
            continue
        if key == "function" and isinstance(value, dict):
            entry["name"] = value.get("name")
            entry["arguments"] = value.get("arguments")
            for other, said in value.items():
                if other not in ("name", "arguments"):
                    entry[f"function.{other}"] = said
            continue
        if key == "extra_content" and isinstance(value, dict):
            # The host writes {"google": {"thought_signature"}} (agent/gemini_native_adapter.py:557)
            # and reads the signature there or at the top, as thought_signature or
            # thoughtSignature (306-308; agent/transports/chat_completions.py:284-287): every
            # one of those is withheld.
            value = copy.deepcopy(value)
            for spelling in ("thought_signature", "thoughtSignature"):
                value.pop(spelling, None)
            google = value.get("google")
            if isinstance(google, dict):
                for spelling in ("thought_signature", "thoughtSignature"):
                    google.pop(spelling, None)
                if not google:
                    value.pop("google", None)
            if not value:
                continue
        entry[key] = value
    return entry


def _as_sent(raw: dict, route: "Route", *, raw_form: bool) -> tuple[dict, Optional[str]]:
    """The message by the host's per-row rules and what the host does with it when it is
    empty (the re-plan of #71, C2; LEARNINGSFÜRPLÄNE A10). First the host's own row, up to
    its fill, with the host's reasoning pad for the route (``host_row_before_fill``); the
    host's fill is asked on that row, before anything of the plugin's. Where the pad is not
    the host's for certain (the agent's opt-in cannot be read), the fill is asked both ways,
    and where the answers differ the note says so. Then, on a copy, only the plugin's own
    transformations (``item_message``), the host's ``_``-prefixed in-process markers left out
    (its chat transport strips them as scaffolding, agent/transports/chat_completions.py:375-)
    and, in the collapsed form, the native carriers. Returns the message and its note."""
    try:
        row = host_row_before_fill(raw, pad=route.pad)
        fill = host_fill_text(row)
        note = _FILL.format(content=json.dumps(fill, ensure_ascii=False)) if fill is not None else None
        if not route.pad:
            other = host_fill_text(host_row_before_fill(raw, pad=True))
            if other != fill:
                note = _FILL_UNSURE.format(content=json.dumps(fill if fill is not None else other,
                                                                ensure_ascii=False))
        message = item_message(row)
    except HostUnavailable as exc:
        raise ExpansionError(str(exc)) from None
    for key in [key for key in message if isinstance(key, str) and key.startswith("_")]:
        message.pop(key, None)
    if not raw_form:
        for key in _NATIVE_CARRIERS:
            message.pop(key, None)
    return message, note


def _call_notes(found: host_pairing.Pairing, handle: str, position: int) -> list:
    """The note of one call: its group's (rule 2), or that no result of its id follows it
    (rule 3), or none."""
    key = (handle, position)
    if key in found.group:
        return [_group_note(found.group[key])]
    if key not in found.answer:
        return [NO_RESULT]
    return []


def _result_notes(found: host_pairing.Pairing, handle: str) -> list:
    """The note of one result: its group's, or that it belongs to no call; or none."""
    if handle in found.group:
        return [_group_note(found.group[handle])]
    if handle in found.stray:
        why, call_id = found.stray[handle]
        return [_STRAY[why].format(id=call_id)]
    return []


def _message_item(handle: str, raw: dict, calls: dict[int, str], found: host_pairing.Pairing, route: "Route", *,
                  inline: set, raw_form: bool) -> Item:
    """One user or agent message by the host's rules (``_as_sent``), its readable
    reasoning first, each tool call as ``_call_entry`` with what the store's pairing says of
    it; in raw form also where its result is when this stretch does not hold it, and for a
    call of a group the store cannot pair, every result of the group."""
    message, fill_note = _as_sent(raw, route, raw_form=raw_form)
    _mark_images(message, handle)
    annotations = {"handle": handle, "role": message.pop("role", raw.get("role"))}
    if fill_note is not None:
        annotations["note"] = fill_note
    fields: dict = {}
    if raw.get("role") == "assistant":
        reasoning = readable_reasoning(raw)
        if reasoning is not None:
            fields["reasoning"] = reasoning
    fields["content"] = message.pop("content", None)
    tool_calls = message.pop("tool_calls", None)
    if isinstance(tool_calls, list) and tool_calls:
        shown = []
        for position, call in enumerate(tool_calls):
            entry = _call_entry(calls.get(position), call)
            notes = _call_notes(found, handle, position)
            key = (handle, position)
            if key in found.group:
                entry["result_in"] = list(found.group[key].results)     # listed, attributed to none
            elif raw_form:
                if key not in found.answer:
                    entry["result"] = None
                elif found.answer[key] not in inline:
                    entry["result_in"] = found.answer[key]
            if notes:
                entry["note"] = "; ".join(notes)
            shown.append(entry)
        fields["tool_calls"] = shown
    message.pop("tool_call_id", None)
    fields.update(message)
    return Item(annotations, fields)


def _result_item(handle: str, raw: dict, call: Optional[str], route: "Route", *, inline: bool, notes: list,
                 raw_form: bool) -> Item:
    """A tool result by the host's rules, answering ``call``; where it stands as a pointer
    (``inline`` false) its content is replaced by its length and every other key stays."""
    message, _fill = _as_sent(raw, route, raw_form=raw_form)
    _mark_images(message, handle)
    annotations: dict = {"handle": handle, "role": message.pop("role", raw.get("role")), "result_of": call}
    message.pop("tool_call_id", None)
    content = message.pop("content", None)
    if inline:
        fields = {"content": content, **message}
    else:
        annotations["content_chars"] = (len(content) if isinstance(content, str)
                                        else len(json.dumps(content, ensure_ascii=False)))
        fields = dict(message)
    if notes:
        annotations["note"] = "; ".join(notes)
    return Item(annotations, fields)


@dataclass
class Order:
    """The active record's order (``RecordStore.active_units``), for pairing a stretch
    through the blocks it lies in (``pairing``); the session's route as read at the call's
    entry, for the items' host rows (the fill note)."""

    records: list
    index: dict
    route: "Route"

    @classmethod
    def of(cls, store: RecordStore, cover: Cover, route: "Route") -> "Order":
        records = [record for unit in store.active_units(cover) for record in unit]
        return cls(records, {record: i for i, record in enumerate(records)}, route)

    def pairing(self, store: RecordStore, stretch: list[str]) -> host_pairing.Pairing:
        return host_pairing.pairing_around(self.records, self.index, stretch, store.record_roles, store.records_raw)


def _records_items(store: RecordStore, order: Order, records: list[tuple[str, dict]], *, raw: bool) -> list[dict]:
    """The items of a run of records, collapsed or raw, paired by the store's rule over the
    active record (``pairing``)."""
    handles = [handle for handle, _raw in records]
    found = order.pairing(store, handles)
    calls = store.tool_calls_of(handles + [record for record, _position in found.result_of.values()])
    held = set(handles)
    inline = {h for h, r in records if r.get("role") == "tool"} if raw else set()
    items: list[Item] = []
    for handle, message in records:
        if message.get("role") == "tool":
            paired = found.result_of.get(handle)
            call = calls.get(paired[0], {}).get(paired[1]) if paired else None
            notes = _result_notes(found, handle)
            if raw:
                items.append(_result_item(handle, message, call, order.route, inline=True, notes=notes,
                                          raw_form=True))
            elif paired is None or paired[0] not in held:
                if paired is not None:
                    notes.insert(0, "the result of a call this stretch does not hold; expand its own handle to "
                                    "read it")
                items.append(_result_item(handle, message, call, order.route, inline=False, notes=notes,
                                          raw_form=False))
            continue
        items.append(_message_item(handle, message, calls.get(handle, {}), found, order.route, inline=inline,
                                   raw_form=raw))
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
            # each opened by a marker naming its chunk and every leaf summary of that chunk, in
            # write order, each with the state the store records for it from the compaction that
            # wrote it (effective, unconfirmed, rejected). No choice is made among them.
            leaves: dict[str, list] = {}
            for derivation, chunk in _all_summaries(store, chunks):
                leaves.setdefault(chunk, []).append({"summary": derivation,
                                                     "state": store.derivation_state(derivation)})
            items: list[Item] = []
            for chunk in chunks:
                items.append(Item({"chunk": chunk, "summaries": leaves.get(chunk, [])}))
                items.extend(_records_items(store, order, store.chunk_records(chunk), raw=raw))
            return Target(header, items)
        # A summary written from summaries (#34 D6, "summaries"): one layer down.
        header["form"] = "summaries"
        items = []
        for chunk, derivation in sources:
            if derivation:
                items.append(Item({"summary": derivation,
                                   "note": "a summary: a description of what happened, not what happened; expand "
                                           "its handle to read behind it"},
                                  {"text": store.derivation_text(derivation)}))
            else:
                items.append(Item({"chunk": chunk}))
        return Target(header, items)
    if resolved.kind == TOOL_CALL:
        # The call's result is the stored result of its exact id in its block (``pairing``):
        # read now, never stored. For a call of a group the store cannot pair, every result
        # of the group, attributed to none.
        record, position = resolved.record, int(resolved.position or 0)
        header = {"handle": handle, "kind": "tool_call", "form": "raw", "name": resolved.name, "call_in": record}
        found = order.pairing(store, [record])
        notes = _call_notes(found, record, position)
        if notes:
            header["note"] = "; ".join(notes)
        group = found.group.get((record, position))
        if group is not None:
            results, of = list(group.results), None
        elif (record, position) in found.answer:
            results, of = [found.answer[(record, position)]], handle
        else:
            results, of = [], None
        if not results:
            header["result"] = None
            return Target(header, [])
        raws = store.records_raw(results)
        return Target(header, [_result_item(result, raws.get(result) or {}, of, order.route, inline=True,
                                            notes=_result_notes(found, result) if group is not None else [],
                                            raw_form=True) for result in results])
    if resolved.kind == MESSAGE:
        return Target({"handle": handle, "kind": "message", "form": form},
                      _records_items(store, order, [(handle, store.record_raw(handle) or {})], raw=True))
    raise ExpansionError(f"{handle} is not a handle this tool opens")


def _all_summaries(store: RecordStore, chunks: list[str]) -> list[tuple[str, str]]:
    """(summary, its chunk) of every summary whose only source is one of these chunks, in
    write order."""
    if not chunks:
        return []
    marks = ",".join("?" * len(chunks))
    return [(str(d), str(c)) for d, c in store._q(
        f"SELECT s.derivation, s.chunk FROM derivation_sources s JOIN derivations d ON d.handle = s.derivation "
        f"WHERE s.chunk IN ({marks}) AND s.ordinal = 0 "
        f"AND NOT EXISTS (SELECT 1 FROM derivation_sources o WHERE o.derivation = s.derivation AND o.ordinal > 0) "
        f"ORDER BY d.derivation_id", chunks)]


# --- Fields and pieces ------------------------------------------------------------------------

def fields_of(item: Item) -> list[tuple[str, Any]]:
    """The fields an item is split into when it does not fit on a page by itself: every
    key of its ``fields``, in order; content by part (a list, or each part and key of an
    envelope), and each tool call by its arguments. Deterministic, so a cursor naming a
    field and an offset finds the same place on every call."""
    fields: list[tuple[str, Any]] = []
    for key, value in item.fields.items():
        if key == "content" and isinstance(value, list):
            fields.extend((f"content[{i}]", part) for i, part in enumerate(value))
        elif key == "content" and isinstance(value, dict) and isinstance(value.get("content"), list):
            for inner, said in value.items():
                if inner == "content":
                    fields.extend((f"content.content[{i}]", part) for i, part in enumerate(said))
                else:
                    fields.append((f"content.{inner}", said))
        elif key == "tool_calls" and isinstance(value, list):
            for index, call in enumerate(value):
                if isinstance(call, dict) and "arguments" in call:
                    fields.append((f"tool_calls[{index}].arguments", call["arguments"]))
                elif isinstance(call, dict) and "call" in call:
                    fields.append((f"tool_calls[{index}].call", call["call"]))
                else:
                    fields.append((f"tool_calls[{index}]", call))
        else:
            fields.append((key, value))
    return fields


def _piece(item: Item, path: str, *, value: Any = None, text: Optional[str] = None, offset: int = 0,
           total: int = 0, json_text: bool = False) -> dict:
    """One field of an item that does not fit on a page by itself: every annotation of
    the item, by complement, then the field and its slice. A tool call's piece also carries
    every key of the call's entry but the one it slices."""
    piece: dict = dict(item.annotations)
    piece["field"] = path
    if path.startswith("tool_calls[") and path.endswith((".arguments", ".call")):
        index = int(path[len("tool_calls["):path.index("]")])
        call = (item.fields.get("tool_calls") or [])[index]
        sliced = path.rsplit(".", 1)[1]
        piece["tool_call"] = {key: said for key, said in call.items() if key != sliced}
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

# --- Images (PR A): a mark, never the image ----------------------------------------------

# Image delivery is PR B (the orchestrator's cut of #71, 2026-09-26): here every image part of
# a message's content stands as a mark with its media type, its size as the host measures
# it and the message's handle. The loss is visible; nothing is sent as an image.
IMAGE_MARK = "images are not delivered by expansion yet; the store holds this image"


def _host_image_helpers() -> tuple[Callable[[Any], bool], Callable[[dict], tuple]]:
    """The host's own image test and size (agent/context_compressor.py:1452-1475, 1567-1572
    at Hermes 8afaab3703): ``_is_image_part`` (``image_url``, ``input_image``, ``image``) and
    ``_image_payload`` (the data URL's or base64 source's length, summed per part)."""
    try:
        from agent.context_compressor import _image_payload, _is_image_part  # type: ignore
    except Exception as exc:
        raise ExpansionError(f"the host's image test (agent.context_compressor) cannot be read "
                             f"({type(exc).__name__}: {exc}), so which content parts are images is not known") from None
    return _is_image_part, _image_payload


def _image_mark(part: dict, handle: str, image_size: Callable[[dict], tuple]) -> dict:
    return {"type": "image", "media_type": image_media_type(part),
            "size": int(image_size({"role": "user", "content": [part]})[1]), "in": handle,
            "not_shown": IMAGE_MARK}


def _mark_images(message: dict, handle: str) -> None:
    """Every image part of the message's content, a list or the ``_multimodal`` envelope's
    list, replaced in place by its mark (``IMAGE_MARK``)."""
    content = message.get("content")
    parts = content.get("content") if isinstance(content, dict) and content.get("_multimodal") else content
    if not isinstance(parts, list):
        return
    is_image, image_size = _host_image_helpers()
    marked = [_image_mark(p, handle, image_size) if is_image(p) else p for p in parts]
    if isinstance(content, dict):
        message["content"] = dict(content, content=marked)
    else:
        message["content"] = marked


def is_mark(path: str, value: Any) -> bool:
    """A content part that is an image's mark: it is never split."""
    return (path.startswith("content[") or path.startswith("content.content[")) and \
        isinstance(value, dict) and value.get("not_shown") == IMAGE_MARK


@dataclass(frozen=True)
class Route:
    """The session's route as the engine holds it (``update_model``, engine.py), read once
    at a call's entry, for the host's rows of the items (its reasoning pad decides the fill
    note). Two host paths change the agent's route without telling the engine
    (agent/turn_recovery.py:305-313, 596 at Hermes 8afaab3703): named, asked of Hermes."""

    api_mode: str = ""
    model: str = ""
    base_url: str = ""
    provider: str = ""
    # The host's own reasoning-echo decision for this route (``host_reasoning_pad``); its
    # opt-in, the agent's, is not visible here (the re-plan of #71, C2).
    pad: bool = False

    @classmethod
    def of(cls, engine: Any) -> "Route":
        api_mode, model = str(getattr(engine, "api_mode", "") or ""), str(getattr(engine, "model", "") or "")
        base_url, provider = str(getattr(engine, "base_url", "") or ""), str(getattr(engine, "provider", "") or "")
        try:
            pad = host_reasoning_pad(provider, model, base_url)
        except HostUnavailable as exc:
            raise ExpansionError(str(exc)) from None
        return cls(api_mode, model, base_url, provider, pad)


# --- Pages -----------------------------------------------------------------------------------

class PageBuilder:
    """Fills pages of ``target`` from a cursor, each at most ``limit`` characters of the
    exact string the engine returns. One measure (``measure``): a candidate page as it will
    be returned, with the real ``next_page`` of the place it ends at, or null where nothing
    is left (the orchestrator's cut of #71, B; LEARNINGSFÜRPLÄNE A9). Every site uses it:
    a whole item, the first piece, every slice (the rest of a field is tried first, ending
    where it ends), a unit standing alone, the header."""

    def __init__(self, target: Target, *, limit: Any, token_state: dict):
        self.target = target
        # ``limit`` is a ``PageLimit``, or a bare number where no origin is known.
        self.origin = limit if isinstance(limit, PageLimit) else None
        self.limit = limit.limit if isinstance(limit, PageLimit) else int(limit)
        self.token_state = token_state
        self._fields: dict[int, list] = {}

    def _fields_of(self, index: int) -> list:
        if index not in self._fields:
            self._fields[index] = fields_of(self.target.items[index])
        return self._fields[index]

    def _normal(self, end: tuple) -> Optional[tuple]:
        """The cursor ``end`` names, past an item's last field moved to the next item; None
        where nothing is left."""
        i, f, o = end
        items = self.target.items
        if i < len(items) and f >= 0 and f >= len(self._fields_of(i)):
            i, f, o = i + 1, -1, 0
        return None if i >= len(items) else (i, f, o)

    def token_at(self, page: int, end: tuple) -> Optional[str]:
        """The ``next_page`` of page ``page`` when it ends at ``end``."""
        at = self._normal(end)
        if at is None:
            return None
        return encode_token({**self.token_state, "i": at[0], "f": at[1], "o": at[2], "n": page + 1})

    def render(self, page_items: list, page: int, next_page: Optional[str]) -> str:
        items = [item.as_dict() if isinstance(item, Item) else dict(item) for item in page_items]
        return final_result({**self.target.header, "items_total": len(self.target.items), "page": page,
                             "next_page": next_page, "items": items})

    def measure(self, page_items: list, page: int, end: tuple) -> int:
        """The characters of the page ``page_items`` as it will be returned, ending at ``end``."""
        return len(self.render(page_items, page, self.token_at(page, end)))

    def fits(self, page_items: list, page: int, end: tuple) -> bool:
        return self.measure(page_items, page, end) <= self.limit

    def _refuse(self, page: int, index: Optional[int], unit: Any, end: tuple) -> None:
        """Nothing is skipped (the orchestrator's ruling on the re-plan of #71, B): an
        unsplittable unit that does not fit even on a page by itself refuses the page,
        naming it (or the header, where the header alone does not fit) and its size."""
        total = len(self.target.items)
        if unit is not None and self.measure([], page, end) > self.limit:
            index, unit = None, None
        size = self.measure([] if unit is None else [unit], page, end)
        if index is None:
            what = "its header, which every page carries,"
        else:
            what = f"item {index + 1} of {total} ({_identity(self.target.items[index], unit)})"
        origin = f" ({self.origin.origin()})" if self.origin else ""
        alone = (f"; alone in a message it would hold {self.origin.alone}"
                 if self.origin and self.origin.alone > self.limit else "")
        token = "; this page's token stays valid" if page > 1 else ""
        raise ExpansionError(f"page {page} of {self.target.header.get('handle')} cannot be built without leaving "
                             f"something out: {what} needs {size} characters on a page by itself; a page here holds "
                             f"{self.limit}{origin}{alone}. Nothing was skipped{token}.")

    def _rest(self, i: int, f: int, o: int) -> list:
        """Everything from (i, f, o) to the end, laid out as ``build`` lays out what fits: the
        rest of a split item piece by piece (a mark or a JSON field's whole value, a text
        field's rest), every later item whole."""
        items = self.target.items
        if i >= len(items):
            return []
        out: list = []
        if f >= 0:
            item = items[i]
            for g, (path, value) in enumerate(self._fields_of(i)):
                if g < f:
                    continue
                start = o if g == f else 0
                if is_mark(path, value) or (not isinstance(value, str) and start == 0):
                    out.append(_piece(item, path, value=value))
                    continue
                text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
                out.append(_piece(item, path, text=text[start:], offset=start, total=len(text),
                                  json_text=not isinstance(value, str)))
            i += 1
        return out + list(items[i:])

    def _unit(self, page_items: list, page: int, index: int, unit: Any, end: tuple) -> bool:
        """An unsplittable unit: on this page if it fits there (True), else the page ends
        before it (False); on a page by itself that it does not fit, refused."""
        if self.fits(page_items + [unit], page, end):
            page_items.append(unit)
            return True
        if page_items:
            return False
        self._refuse(page, index, unit, end)
        return False

    def build(self, cursor: Cursor, page: int) -> tuple[str, Optional[Cursor]]:
        """One page from ``cursor``: the string the engine returns, and the cursor of the
        next page, or None where this page is the last."""
        items = self.target.items
        page_items: list = []
        i, f, o = cursor.item, cursor.field, cursor.offset
        if not items and not self.fits([], page, (0, -1, 0)):
            self._refuse(page, None, None, (0, -1, 0))
        # Everything that is left, as the rest would be laid out, on this page with no
        # next_page: a page that continues carries a token, which can be longer than the rest
        # itself, so the last page is measured as the last page before anything is cut.
        rest = self._rest(i, f, o)
        if rest and self.fits(rest, page, (len(items), -1, 0)):
            return self.render(rest, page, None), None
        while i < len(items):
            item = items[i]
            if f < 0:
                after = (i + 1, -1, 0)
                if self.fits(page_items + [item], page, after):
                    page_items.append(item)
                    i, f, o = after
                    continue
                if page_items:
                    break
                if not self._fields_of(i):
                    self._refuse(page, i, item, after)      # no fields to split it by
                f, o = 0, 0          # it does not fit alone: its fields, piece by piece
                continue
            fields = self._fields_of(i)
            if f >= len(fields):
                i, f, o = i + 1, -1, 0
                continue
            path, value = fields[f]
            after = (i, f + 1, 0)
            if is_mark(path, value):
                if not self._unit(page_items, page, i, _piece(item, path, value=value), after):
                    break
                f, o = f + 1, 0
                continue
            json_text = not isinstance(value, str)
            if json_text and o == 0:
                piece = _piece(item, path, value=value)
                if self.fits(page_items + [piece], page, after):
                    page_items.append(piece)
                    f, o = f + 1, 0
                    continue
                if page_items:
                    break
            text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
            if not text:
                if not self._unit(page_items, page, i, _piece(item, path, text="", offset=0, total=0,
                                                              json_text=json_text), after):
                    break
                f, o = f + 1, 0
                continue
            if o >= len(text):
                f, o = f + 1, 0
                continue
            rest = _piece(item, path, text=text[o:], offset=o, total=len(text), json_text=json_text)
            if self.fits(page_items + [rest], page, after):
                page_items.append(rest)
                f, o = f + 1, 0
                continue
            length = self._longest_slice(page_items, item, path, text, o, page, json_text, i, f)
            if length == 0:
                if page_items:
                    break
                self._refuse(page, i, _piece(item, path, text=text[o:o + 1], offset=o, total=len(text),
                                             json_text=json_text), (i, f, o + 1))
            page_items.append(_piece(item, path, text=text[o:o + length], offset=o, total=len(text),
                                     json_text=json_text))
            o += length
            break
        end = self._normal((i, f, o))
        next_page = self.token_at(page, (i, f, o))
        return self.render(page_items, page, next_page), (Cursor(*end) if end is not None else None)

    def _longest_slice(self, page_items: list, item: Item, path: str, text: str, offset: int, page: int,
                       json_text: bool, i: int, f: int) -> int:
        """The longest slice short of the field's end that fits, each measured ending where it
        ends (the whole rest was tried first)."""
        low, high, best = 1, len(text) - offset - 1, 0
        while low <= high:
            middle = (low + high) // 2
            piece = _piece(item, path, text=text[offset:offset + middle], offset=offset, total=len(text),
                           json_text=json_text)
            if self.fits(page_items + [piece], page, (i, f, offset + middle)):
                best, low = middle, middle + 1
            else:
                high = middle - 1
        return best



def _identity(item: Item, piece: Any) -> str:
    """What a refusal names: the item, and the field of a piece."""
    said = item.annotations
    if "summaries" in said:
        name = f"the marker of chunk {said.get('chunk')}"
    elif "summary" in said:
        name = f"summary {said.get('summary')}"
    elif "chunk" in said and "handle" not in said:
        name = f"chunk {said.get('chunk')}"
    elif said.get("role") == "tool":
        name = f"result {said.get('handle')}" + (" as a pointer" if "content_chars" in said else "")
    else:
        name = f"message {said.get('handle')}"
    path = piece.get("field") if isinstance(piece, dict) and piece is not item else None
    if path is None:
        return name
    return f"{path} of {name}" + (", an image's mark" if is_mark(path, piece.get("value")) else "")


def target_identity(target: Target) -> str:
    """The identity of everything a handle opens into, as it renders: its header and every
    item after every transformation between the store and the page (the host's row, the
    withholding, the marks, the notes), as JSON; pages are cut from exactly these items, so
    anything that changes what a page shows changes it (the orchestrator's cut of #71, C).
    Sixteen hex characters of its SHA-256. A record the host rewrote, a note that changed (a
    fill note after a route change), or a projection that changed (a reload onto other
    code) changes it; a compaction that only records new material after the stretch does
    not."""
    payload = [target.header, [item.as_dict() for item in target.items]]
    text = json.dumps(payload, ensure_ascii=False, sort_keys=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def expand(engine: Any, args: dict, *, messages: Any = None, tool_name: str = "lcm_expand") -> Any:
    """The ``lcm_expand`` tool: one page of what a handle opens into. ``messages`` is the
    live list the host hands the engine tool; the page's size depends on it
    (``host_page_limit``)."""
    session = engine.current_session_id          # the caller's session, read once (#20)
    route = Route.of(engine)                     # the session's route, read once (§1.4)
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
    limit = host_page_limits(engine, tool_name, messages)
    records: RecordStore = engine._records
    with records.snapshot():
        store_uuid = records.identity().get("store_uuid")
        if state is not None and state.get("s") != store_uuid:
            raise ExpansionError("page is a token of another store: the store it was issued by is not this one")
        cover = records.cover(session)
        resolved = records.resolve(str(handle), session, cover)
        if resolved.status != "ok":
            raise ExpansionError(unresolved_message(resolved))
        target = target_for(records, Order.of(records, cover, route), resolved, raw=bool(raw))
    # What paging began on (Codex review of ca0969d, finding 2; the plan of #71, §5.3): a
    # token carries the identity of the whole target, and a later page on anything else is
    # refused, never spliced.
    identity = target_identity(target)
    if state is not None and state["r"] != identity:
        raise ExpansionError("the content changed since page 1 (a record the host rewrote, a pairing or note that "
                             "changed, or code that changed); start again from the handle, without page")
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
            # Just past an item's last field (a token of 3a4e019 after a unit that stood
            # alone): the next item (finding 1 of the Codex review of 3a4e019).
            cursor = Cursor(cursor.item + 1, -1, 0)
        elif cursor.field >= len(fields):
            raise ExpansionError("page is a garbled next_page token: it names a field this item does not have")
        else:
            path, value = fields[cursor.field]
            length = len(value) if isinstance(value, str) else len(json.dumps(value, ensure_ascii=False))
            if cursor.offset > length or (cursor.offset and is_mark(path, value)):
                raise ExpansionError("page is a garbled next_page token: its offset lies outside the field it "
                                     "names")
    builder = PageBuilder(target, limit=limit, token_state=token_state)
    result, _next = builder.build(cursor, page)
    return result
