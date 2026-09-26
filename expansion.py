"""Looking behind a handle (#18): what expansion returns, and the one page mechanism.

**What a handle opens into** (manifesto, "Looking behind a handle"; #29 W5; #34 D6):

- a summary's handle (``s``): the chunk it stands for; a summary written from the raw of
  several chunks, all of them, with the leaf summary of each named beside it as a
  lookup; a summary written from summaries, one layer: those summaries;
- a chunk's handle (``c``): the stretch itself;
- a tool call's handle (``t``): its result, the one the host's own pre-call passes send
  with it, block by block (``pairing``);
- a message's handle (``m``): that message.

A stretch comes back in its collapsed form by default: every user and agent message by
the host's per-row rules (``summariser_input.message_as_sent``: the host's sidecar in place
of ``content`` where the host sends it, its bookkeeping popped, the encrypted items
withheld), less the host's ``_``-prefixed in-process markers, every other key carried; the
readable reasoning beside the agent's messages; and each tool call as its handle, its name
and its arguments and every other key but the host's ids, without its result. The host's
stand-ins (its fill of an empty message, its stand-in for a missing result) are quoted in
notes, never shown as content. A result whose call is not in the same stretch stands as a
pointer to that call, and a call or result the host does not send says why. ``raw`` puts
every result inline and adds the host's native carriers of a message. Encrypted reasoning
stays out, and nothing stands where it was. The copies of repeated injections in the host's
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
lies. An image is never split: it is returned as an image, in the ``_multimodal``
envelope, counted against the page by the model table's rule (#21); one larger than a
page, or one no rule counts, stands alone on its page. A page carries no image the host's
converter for the session's route does not carry as an image, and no more images than the
host's send path leaves room for in the request (``ImageRoom``).
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
from .summariser_input import HostUnavailable, host_fill_text, message_as_sent
from .tokens import CHARS_PER_TOKEN

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

# Notes on calls and results, from what the host's pre-call sanitizer sends (``pairing``).
NO_RESULT = "no result follows this call on the active record"
_STAND_IN = "the host's pre-call sanitizer sends the provider its own stand-in for this call's result: {content}"
_CALL_NOT_SENT = ("the host's pre-call sanitizer does not send this call to the provider: an earlier call of this "
                  "message carries its host id {id}")
_RESULT_NOT_SENT = {
    "no_id": "the host's pre-call sanitizer does not send this result to the provider: it carries no host id",
    "positional": ("the host's pre-call sanitizer does not send this result to the provider: no call before it, since "
                   "the last assistant or user message, that is still waiting for a result has its host id {id}"),
    "duplicate": ("the host's pre-call sanitizer does not send this result to the provider: it answers no call still "
                  "waiting for a result"),
}
_ROLE_NOT_SENT = "the host's pre-call sanitizer does not send a message of role {role} to the provider"
_FILL = "the host sends this empty message to the provider with its own stand-in as content: {content}"
_SHARED = ("{k} calls of this message carry the host id {id}; the pairing shown is the host's, which the id cannot "
           "confirm")


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


@dataclass
class Item:
    """One item of what a handle opens into, as a pair (the plan of #71, §5.2): its
    ``annotations``, what the plugin says of it (``handle``, ``role``, ``result_of``,
    ``note``, ``content_chars``, ``chunk``, ``summary``), which every piece of it carries; and
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
            value = copy.deepcopy(value)
            google = value.get("google")
            if isinstance(google, dict):
                google.pop("thought_signature", None)
                if not google:
                    value.pop("google", None)
            if not value:
                continue
        entry[key] = value
    return entry


def _as_sent(raw: dict, *, raw_form: bool) -> tuple[dict, Optional[str]]:
    """The message by the host's per-row rules, strictly (``summariser_input.message_as_sent``:
    the sidecar as content where the host sends it, its bookkeeping popped, the encrypted
    items withheld), minus the host's ``_``-prefixed in-process markers, which its own chat
    transport strips as scaffolding (agent/transports/chat_completions.py:375-), and, in the
    collapsed form, minus the native carriers. With it the host's own stand-in where it
    would fill the message for being empty (a note, never content)."""
    try:
        message = message_as_sent(raw)
        fill = host_fill_text(message) if raw.get("role") in ("user", "assistant") else None
    except HostUnavailable as exc:
        raise ExpansionError(str(exc)) from None
    for key in [key for key in message if isinstance(key, str) and key.startswith("_")]:
        message.pop(key, None)
    if not raw_form:
        for key in _NATIVE_CARRIERS:
            message.pop(key, None)
    return message, fill


def _call_notes(found: host_pairing.Pairing, handle: str, position: int) -> tuple[list, bool]:
    """The notes of one call, and whether a result answers it on the wire."""
    notes = []
    key = (handle, position)
    answered = key in found.answer
    if key in found.call_not_sent:
        notes.append(_CALL_NOT_SENT.format(id=found.call_not_sent[key]))
    elif not answered:
        notes.append(NO_RESULT)
        if key in found.stand_in:
            notes.append(_STAND_IN.format(content=json.dumps(found.stand_in[key], ensure_ascii=False)))
    if key in found.shared:
        call_id, calls = found.shared[key]
        notes.append(_SHARED.format(k=calls, id=call_id))
    return notes, answered


def _record_notes(found: host_pairing.Pairing, handle: str, fill: Optional[str]) -> list:
    """What the host does with a record as a whole: a role it does not send; its own
    stand-in as the content of an empty message."""
    notes = []
    if handle in found.role_not_sent:
        notes.append(_ROLE_NOT_SENT.format(role=json.dumps(found.role_not_sent[handle], ensure_ascii=False)))
    if fill is not None:
        notes.append(_FILL.format(content=json.dumps(fill, ensure_ascii=False)))
    return notes


def _message_item(handle: str, raw: dict, calls: dict[int, str], found: host_pairing.Pairing, *, inline: set,
                  raw_form: bool) -> Item:
    """One user or agent message by the host's rules (``_as_sent``), its readable
    reasoning first, each tool call as ``_call_entry`` with what the host's pairing says of
    it; in raw form also where its result is when this stretch does not hold it."""
    message, fill = _as_sent(raw, raw_form=raw_form)
    annotations = {"handle": handle, "role": message.pop("role", raw.get("role"))}
    notes = _record_notes(found, handle, fill)
    if notes:
        annotations["note"] = "; ".join(notes)
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
            notes, answered = _call_notes(found, handle, position)
            if raw_form:
                if not answered:
                    entry["result"] = None
                elif found.answer[(handle, position)] not in inline:
                    entry["result_in"] = found.answer[(handle, position)]
            if notes:
                entry["note"] = "; ".join(notes)
            shown.append(entry)
        fields["tool_calls"] = shown
    message.pop("tool_call_id", None)
    fields.update(message)
    return Item(annotations, fields)


def _result_item(handle: str, raw: dict, call: Optional[str], *, inline: bool, notes: list, raw_form: bool) -> Item:
    """A tool result by the host's rules, answering ``call``; where it stands as a pointer
    (``inline`` false) its content is replaced by its length and every other key stays."""
    message, _fill = _as_sent(raw, raw_form=raw_form)
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
    through the blocks it lies in (``pairing``), on the session's route as read at the
    call's entry."""

    records: list
    index: dict
    route: "Route"

    @classmethod
    def of(cls, store: RecordStore, cover: Cover, route: "Route") -> "Order":
        records = [record for unit in store.active_units(cover) for record in unit]
        return cls(records, {record: i for i, record in enumerate(records)}, route)

    def pairing(self, store: RecordStore, stretch: list[str]) -> host_pairing.Pairing:
        try:
            return host_pairing.pairing_around(self.records, self.index, stretch, store.record_roles,
                                               store.records_raw, api_mode=self.route.api_mode,
                                               model=self.route.model)
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
    items: list[Item] = []
    for handle, message in records:
        if message.get("role") == "tool":
            paired = found.result_of.get(handle)
            call = calls.get(paired[0], {}).get(paired[1]) if paired else None
            notes = []
            if handle in found.result_not_sent:
                why, call_id = found.result_not_sent[handle]
                notes.append(_RESULT_NOT_SENT[why].format(id=call_id))
            if paired is not None and paired in found.shared:
                call_id, count = found.shared[paired]
                notes.append(_SHARED.format(k=count, id=call_id))
            if raw:
                items.append(_result_item(handle, message, call, inline=True, notes=notes, raw_form=True))
            elif paired is None or paired[0] not in held:
                if paired is not None:
                    notes.insert(0, "the result of a call this stretch does not hold; expand its own handle to "
                                    "read it")
                items.append(_result_item(handle, message, call, inline=False, notes=notes, raw_form=False))
            continue
        items.append(_message_item(handle, message, calls.get(handle, {}), found, inline=inline, raw_form=raw))
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
            items: list[Item] = []
            for chunk in chunks:
                items.append(Item({"chunk": chunk, "summary": leaves.get(chunk)}))
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
        # The call's result is the one the host's rule pairs with it on the active record
        # (round 5 of #71): read now, never stored.
        record, position = resolved.record, int(resolved.position or 0)
        header = {"handle": handle, "kind": "tool_call", "form": "raw", "name": resolved.name, "call_in": record}
        found = order.pairing(store, [record])
        notes, answered = _call_notes(found, record, position)
        if notes:
            header["note"] = "; ".join(notes)
        if not answered:
            header["result"] = None
            return Target(header, [])
        result = found.answer[(record, position)]
        return Target(header, [_result_item(result, store.record_raw(result) or {}, handle, inline=True, notes=[],
                                            raw_form=True)])
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
            media = source.get("media_type")
            if not isinstance(media, str) or not media:
                # Known, or nothing: no media type is guessed (defe17e guessed image/jpeg).
                return None, "stored as an Anthropic image block without a media type, which gives no data URL"
            url = f"data:{media};base64,{source['data']}"
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
        item = item.as_dict() if isinstance(item, Item) else dict(item)
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

    First, whether the session's route carries the image to the model as an image at all:
    the route's own converter is run on it (``route_image_check``, the plan of #71, §1.8).

    Then the room. The host retires image-bearing tool results on its send path by one
    policy, ``outbound_image_retire_count`` (agent/image_eviction_policy.py:33-96 at Hermes
    8afaab3703; 20 blocks and 24,000,000 bytes, 26-27), in ``evict_stale_outbound_tool_images``
    (agent/context_compressor.py:1478-1517), which is run here on a skeleton of the live list
    the host hands the tool, its image-bearing messages with their own parts, plus the page,
    so the uploads it reserves, the older carriers and the earlier results of this message
    count exactly as the host counts them. The wire holds a subset of the list's images (a
    historical row whose sidecar string replaces list content, a stray result the sanitizer
    drops), so this can hold a page the host would have carried, never admit one it retires.
    The Anthropic converter's own block pass (agent/anthropic_message_convert.py:605-633)
    retires nothing more after this pass: it counts the same carriers and no more reserved
    images (the plan, §1.7). Of the calls of this message still to come, only this tool's are
    known image sources: they share the room equally (``share``, this one included; the host
    runs them one after another, so this is conservative). A page is admitted where, with
    ``share`` carriers like it at the newest end, the pass retires exactly what it retires
    without them, and none of them.

    Another tool's later result may carry images this cannot foresee; that retirement is
    the host's, visible in its own placeholder, and where the message has such calls the
    page says so (``note``). Where the host cannot be read, a page carries one image and
    says why (unknown is never unlimited)."""

    evict: Optional[Callable]               # the host's evict_stale_outbound_tool_images
    measure: Optional[Callable[[dict], tuple]]  # the host's _image_payload
    skeleton: tuple = ()                    # the live list's image-bearing messages, {"role", "content"}
    share: int = 1
    note: str = ""
    limit: Optional[int] = None             # the host's OUTBOUND_IMAGE_LIMIT
    budget: Optional[int] = None            # the host's OUTBOUND_IMAGE_BUDGET_BYTES
    part_hold: Optional[Callable[[dict], str]] = None  # why the route does not carry one image, or ""

    def _run(self, base: list, images: list, copies: int) -> tuple[set, bool]:
        """The host's own send-path pass over ``base`` plus ``copies`` carriers holding
        ``images``: which messages of ``base`` it retires, and whether it retires one of the
        carriers. The pass replaces retired messages by new dicts and never mutates a part
        (context_compressor.py:1396-1427)."""
        request = list(base) + [{"role": "tool", "content": list(images)} for _ in range(copies)]
        before = list(request)
        self.evict(request)
        retired = {i for i in range(len(base)) if request[i] is not before[i]}
        return retired, any(request[i] is not before[i] for i in range(len(base), len(request)))

    def admits(self, images: list) -> bool:
        if not images:
            return True
        if self.part_hold is not None and any(self.part_hold(p) for p in images):
            return False
        if self.evict is None or self.measure is None:
            return len(images) <= 1
        baseline, _ = self._run(list(self.skeleton), [], 0)
        retired, ours = self._run(list(self.skeleton), images, self.share)
        return not ours and retired == baseline

    def why_not(self, image: dict, handle: str, media: str) -> str:
        """Why an image no page can carry now is held: the route's converter, or what the
        host's own send-path pass would do with the page added, found by running it on
        counterfactual lists (the page alone; the uploads and the page; the whole list)."""
        size = self.measure({"role": "tool", "content": [image]})[1] if self.measure is not None else len(str(image))
        held = f"; the store still holds it: {media}, {size} bytes, in message {handle}"
        if self.part_hold is not None and self.part_hold(image):
            return f"not shown: {self.part_hold(image)}{held}"
        if self.evict is None or self.measure is None:
            return f"not shown: {self.note}{held}"
        _retired, alone = self._run([], [image], 1)
        if alone:
            return (f"not shown: this image is {size} bytes as the host measures it, over the host's budget of "
                    f"{self.budget} bytes for the images of one request, so the host's send path would strip it "
                    f"from any page; no page can carry this image in this host. The store still holds it: {media}, "
                    f"{size} bytes, in message {handle}")
        uploads = [m for m in self.skeleton if m.get("role") != "tool"]
        count = sum(self.measure(m)[0] for m in uploads)
        taken = sum(self.measure(m)[1] for m in uploads)
        _retired, with_uploads = self._run(uploads, [image], 1)
        if with_uploads:
            if self.limit is not None and count + 1 > self.limit:
                return (f"not shown: the {count} images the host reserves in this request (uploads, which it never "
                        f"retires) fill its ceiling of {self.limit} images, so the host would retire this page; the "
                        f"image shows once those uploads leave the context")
            return (f"not shown: the images the host reserves in this request (uploads, which it never retires) take "
                    f"{taken} bytes of its budget of {self.budget}, which leaves no room for this image's {size} "
                    f"bytes, so the host would retire this page; the image shows once those uploads leave the context")
        baseline, _ = self._run(list(self.skeleton), [], 0)
        retired, ours = self._run(list(self.skeleton), [image], self.share)
        if retired - baseline:
            shared = (f", shared with {self.share - 1} more {'call' if self.share == 2 else 'calls'} of this tool "
                      f"in this message") if self.share > 1 else ""
            older = sum(self.measure(m)[0] for m in self.skeleton if m.get("role") == "tool")
            return (f"not shown: the host's send path has no room for it in this request without retiring earlier "
                    f"images ({count} {'image' if count == 1 else 'images'} it reserves and {older} in earlier "
                    f"results stand in it{shared}); it stays in the store under this handle")
        return (f"not shown: the {self.share} calls of this tool in this message share the room left in this "
                f"request, and this image does not fit this call's share; the host would retire one of these "
                f"pages; it stays in the store under this handle")


_OTHER_CALLS_NOTE = ("the host may retire this page's images if other results of this turn carry images; expand "
                     "again later to see them")


@dataclass(frozen=True)
class Route:
    """The session's route as the engine holds it (``update_model``, engine.py), read once
    at a call's entry. Two host paths change the agent's route without telling the engine
    (agent/turn_recovery.py:305-313, 596 at Hermes 8afaab3703): named, asked of Hermes."""

    api_mode: str = ""
    model: str = ""
    base_url: str = ""

    @classmethod
    def of(cls, engine: Any) -> "Route":
        return cls(str(getattr(engine, "api_mode", "") or ""), str(getattr(engine, "model", "") or ""),
                   str(getattr(engine, "base_url", "") or ""))


def _probe_messages(tool_name: str, part: dict) -> list:
    call_id = "call_lcm_route_check"
    return [{"role": "user", "content": "."},
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": call_id, "type": "function", "function": {"name": tool_name, "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": call_id, "name": tool_name,
             "content": [{"type": "text", "text": "."}, part]}]


def _blocks(value: Any) -> list:
    return value if isinstance(value, list) else []


def _route_images(route: Route, tool_name: str, part: dict) -> tuple[str, int]:
    """(the route's name, the images the host's converter for it puts into the wire request
    for a tool result holding ``part``). Counted in each wire's own format; raises
    ``LookupError`` for a route this plugin does not know how to read."""
    from agent.transports import get_transport  # type: ignore
    messages = _probe_messages(tool_name, part)
    mode = route.api_mode
    if mode == "chat_completions":
        from agent.gemini_native_adapter import build_gemini_request, is_native_gemini_base_url  # type: ignore
        if is_native_gemini_base_url(route.base_url):
            request = build_gemini_request(messages=messages, model=route.model)
            return "chat_completions, Gemini's native API", sum(
                1 for content in request.get("contents") or [] for p in _blocks(content.get("parts"))
                if isinstance(p, dict) and "functionResponse" in p
                for inner in _blocks(p["functionResponse"].get("parts")) if isinstance(inner, dict) and "inlineData" in inner)
        transport = get_transport(mode)
        if transport is None:
            raise LookupError(mode)
        out = transport.convert_messages(messages, model=route.model, base_url=route.base_url)
        return mode, sum(1 for m in out if isinstance(m, dict) and m.get("role") == "tool"
                         for p in _blocks(m.get("content")) if isinstance(p, dict) and p.get("type") == "image_url")
    if mode == "anthropic_messages":
        transport = get_transport(mode)
        if transport is None:
            raise LookupError(mode)
        _system, out = transport.convert_messages(messages, base_url=route.base_url)
        return mode, sum(1 for m in out if isinstance(m, dict) for block in _blocks(m.get("content"))
                         if isinstance(block, dict) and block.get("type") == "tool_result"
                         for inner in _blocks(block.get("content")) if isinstance(inner, dict)
                         and inner.get("type") == "image")
    if mode == "codex_responses":
        transport = get_transport(mode)
        if transport is None:
            raise LookupError(mode)
        out = transport.convert_messages(messages, model=route.model)
        return mode, sum(1 for item in out if isinstance(item, dict) and item.get("type") == "function_call_output"
                         for inner in _blocks(item.get("output")) if isinstance(inner, dict)
                         and inner.get("type") == "input_image")
    if mode == "bedrock_converse":
        transport = get_transport(mode)
        if transport is None:
            raise LookupError(mode)
        _system, out = transport.convert_messages(messages)
        return mode, sum(1 for m in out if isinstance(m, dict) for block in _blocks(m.get("content"))
                         if isinstance(block, dict) and isinstance(block.get("toolResult"), dict)
                         for inner in _blocks(block["toolResult"].get("content")) if isinstance(inner, dict)
                         and "image" in inner)
    raise LookupError(mode)


def route_image_check(route: Route, tool_name: str) -> Callable[[dict], str]:
    """Whether the session's route carries an image to the model as an image, by the
    route's own converter (the plan of #71, §1.8): "" where it does, else why not. Once per
    image per call. No prediction of the host: where the plugin cannot read how a route
    carries images, or its converter raises, the image is held and the page says so."""
    seen: dict[int, str] = {}

    def check(part: dict) -> str:
        key = id(part)
        if key not in seen:
            try:
                name, count = _route_images(route, tool_name, part)
                seen[key] = "" if count >= 1 else (
                    f"the host's converter for this session's route ({name}) does not carry this image to the model "
                    f"as an image")
            except LookupError:
                seen[key] = (f"how the host carries a tool result's images on this session's route "
                             f"({route.api_mode or 'none'}) is not known to this plugin")
            except Exception as exc:
                seen[key] = (f"the host's converter for this session's route ({route.api_mode}) could not be run on "
                             f"this image ({type(exc).__name__}: {exc})")
        return seen[key]

    return check


def host_image_room(messages: Any, tool_name: str, route: Route) -> ImageRoom:
    """The ``ImageRoom`` of this call, from the live list and the host's own functions."""
    part_hold = route_image_check(route, tool_name)
    try:
        from agent.context_compressor import _image_payload, evict_stale_outbound_tool_images  # type: ignore
        from agent.image_eviction_policy import (  # type: ignore
            OUTBOUND_IMAGE_BUDGET_BYTES,
            OUTBOUND_IMAGE_LIMIT,
        )
        call_variants, result_variants, _coalesce = host_pairing.host_alias_helpers()
        if not isinstance(messages, list):
            raise ValueError("no message list")
        skeleton = tuple({"role": m.get("role"), "content": m.get("content")} for m in messages
                         if isinstance(m, dict) and _image_payload(m)[0])
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
        return ImageRoom(evict_stale_outbound_tool_images, _image_payload, skeleton,
                         max(1, ours), _OTHER_CALLS_NOTE if others else "", int(OUTBOUND_IMAGE_LIMIT),
                         int(OUTBOUND_IMAGE_BUDGET_BYTES), part_hold)
    except Exception as exc:
        return ImageRoom(None, None, note=f"the host's send-path image ceiling cannot be read "
                                          f"({type(exc).__name__}: {exc}), so this page carries at most one image",
                         part_hold=part_hold)


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
                        piece["held"] = self.image_room.why_not(canonical_image_part(value)[0],
                                                                str(item.annotations.get("handle")),
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


def target_identity(target: Target) -> str:
    """The identity of everything a handle opens into: its header and every item, as JSON.
    Sixteen hex characters of its SHA-256 (the plan of #71, §5.3). A record the host
    rewrote, a pairing or note that changed (a route change that changes a call's aliases),
    or a projection that changed (a reload onto other code) changes it; a compaction that
    only records new material after the stretch does not."""
    payload = [target.header, [item.as_dict() for item in target.items]]
    text = json.dumps(payload, ensure_ascii=False, sort_keys=False, default=repr)
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
                          image_room=host_image_room(messages, tool_name, route))
    result, _next = builder.build(cursor, page)
    return result
