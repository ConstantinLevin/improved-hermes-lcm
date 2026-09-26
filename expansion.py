"""Looking behind a handle (#18): what expansion returns, and the one page mechanism.

**What a handle opens into** (manifesto, "Looking behind a handle"; #29 W5; #34 D6):

- a summary's handle (``s``): the chunk it stands for; a summary written from the raw of
  several chunks, all of them, with the leaf summary of each named beside it as a
  lookup; a summary written from summaries, one layer: those summaries;
- a chunk's handle (``c``): the stretch itself;
- a tool call's handle (``t``): its result;
- a message's handle (``m``): that message.

A stretch comes back in its collapsed form by default: every user and agent message as
stored (its ``content``, never the host's ``api_content`` sidecar, which carries hook
injections and the memory prefetch), the readable reasoning beside the agent's messages
(plain reasoning, as the summariser reads it: ``summariser_input``), and each tool call
as its handle, its name and its arguments, without its result. A result whose call is
not in the same stretch stands as a pointer to that call. ``raw`` puts every result
inline. Encrypted reasoning stays out, and nothing stands where it was.

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
page, or one no rule counts, stands alone on its page.
"""

from __future__ import annotations

import base64
import binascii
import copy
import json
import math
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Optional

from .handles import CHUNK, DERIVATION, MESSAGE, TOOL_CALL
from .message_content import image_media_type, is_image_part
from .record_store import HANDLE_RE, Cover, RecordStore, Resolved
from .results import final_result
# Readable reasoning is what the summariser reads as readable, one rule (#8, #18).
from .summariser_input import _readable_reasoning as readable_reasoning
from .tokens import CHARS_PER_TOKEN

TOKEN_VERSION = 1

# What each status of ``RecordStore.resolve`` tells the agent.
_UNRESOLVED = {
    "malformed": ("{handle!r} is not a handle. A handle is a kind letter (m a message, t a tool call, c a chunk, "
                  "s a summary) and eight characters, as the summaries in your context and these tools show them."),
    "unknown": "{handle} is unknown in this store: no message, tool call, chunk or summary has this handle here.",
    "other_session": ("{handle} belongs to another session. A handle resolves only in the session that holds it; "
                      "this session cannot reach another's past."),
    "inactive": ("{handle} is this session's, but not on its active record: it lies on a branch an undo or retry "
                 "left behind, or in a compaction that never took effect."),
    "summary_revision": ("{handle} is the host's rewrite of summary {of} in your context (the row the plugin "
                         "returned, as the host changed it, for example with its task list folded in). Expand {of} "
                         "to read behind the summary."),
}


class ExpansionError(Exception):
    """Something the caller is told instead of a page: a wrong argument, a handle that
    does not resolve, a token that is not this store's."""


def unresolved_message(resolved: Resolved) -> str:
    return _UNRESOLVED[resolved.status].format(handle=resolved.handle, of=resolved.of)


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
        state = json.loads(base64.urlsafe_b64decode(text + "=" * (-len(text) % 4)).decode("utf-8"))
    except (binascii.Error, UnicodeDecodeError, ValueError):
        raise ExpansionError("page is not a next_page token of these tools") from None
    if not isinstance(state, dict) or state.get("v") != TOKEN_VERSION:
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


def _message_item(handle: str, raw: dict, calls: dict[int, str], by_call_id: dict[str, str]) -> dict:
    """One user or agent message: its content as stored, its readable reasoning beside it,
    each tool call as its handle, name and arguments."""
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
            call_id = str(call.get("id") or "") if isinstance(call, dict) else ""
            entry = {"handle": calls.get(position) or by_call_id.get(call_id)}
            if name is None and arguments is None:
                entry["call"] = call          # a shape other than the host's: as stored
            else:
                entry["name"] = name
                entry["arguments"] = arguments
            shown.append(entry)
        item["tool_calls"] = shown
    return item


def _result_item(handle: str, raw: dict, call: Optional[str], *, inline: bool) -> dict:
    item: dict = {"handle": handle, "role": raw.get("role"), "result_of": call}
    name = raw.get("name") or raw.get("tool_name")
    if name:
        item["name"] = name
    if inline:
        item["content"] = raw.get("content")
    else:
        content = raw.get("content")
        item["content_chars"] = len(content) if isinstance(content, str) else len(json.dumps(content, ensure_ascii=False))
        item["note"] = "the result of a call this stretch does not hold; expand its own handle to read it"
    return item


def _records_items(store: RecordStore, records: list[tuple[str, dict]], *, raw: bool) -> list[dict]:
    """The items of a run of records, collapsed or raw."""
    handles = [handle for handle, _raw in records]
    calls = store.tool_calls_of(handles)
    call_of = store.call_of_result([h for h, r in records if r.get("role") == "tool"])
    shown_calls: set = set()
    by_record_call_id: dict[str, dict[str, str]] = {}
    for handle, message in records:
        if message.get("role") == "assistant" and isinstance(message.get("tool_calls"), list):
            by_call_id = store.revision_call_handles(handle) if store.record_kind(handle) == "revision" else {}
            by_record_call_id[handle] = by_call_id
            shown_calls.update(h for h in calls.get(handle, {}).values())
            shown_calls.update(by_call_id.values())
    items: list[dict] = []
    for handle, message in records:
        if message.get("role") == "tool":
            call = call_of.get(handle)
            if raw:
                items.append(_result_item(handle, message, call, inline=True))
            elif call is None or call not in shown_calls:
                items.append(_result_item(handle, message, call, inline=False))
            continue
        items.append(_message_item(handle, message, calls.get(handle, {}), by_record_call_id.get(handle, {})))
    if raw:
        # Raw puts every result inline; a call whose result is not in this stretch says
        # where it is, or that none is recorded, as a tool call's own expansion does.
        inline = {h for h, r in records if r.get("role") == "tool"}
        for item in items:
            for entry in item.get("tool_calls") or ():
                if not entry.get("handle"):
                    continue
                found = store.tool_call(entry["handle"])
                result = found[2] if found else None
                if result is None:
                    entry["result"] = None
                    entry["note"] = "no result of this call is recorded in the store"
                elif result not in inline:
                    entry["result_in"] = result
    return items


def target_for(store: RecordStore, cover: Cover, resolved: Resolved, *, raw: bool) -> Target:
    """What ``resolved`` opens into (the module docstring), read from the store. Called
    inside the caller's read transaction (``RecordStore.snapshot``)."""
    handle = resolved.handle
    form = "raw" if raw else "collapsed"
    if resolved.kind == CHUNK:
        return Target({"handle": handle, "kind": "chunk", "form": form},
                      _records_items(store, store.chunk_records(handle), raw=raw))
    if resolved.kind == DERIVATION:
        sources = store.derivation_sources(handle)
        header: dict = {"handle": handle, "kind": "summary", "form": form}
        if sources and all(chunk for chunk, _derivation in sources):
            chunks = [str(chunk) for chunk, _derivation in sources]
            header["chunks"] = chunks
            if len(chunks) == 1:
                return Target(header, _records_items(store, store.chunk_records(chunks[0]), raw=raw))
            # A summary written from the raw of several chunks (#34 D6, "raw"): all of them,
            # each opened by a marker naming its chunk and that chunk's own summary.
            leaves = store.leaf_summaries(chunks, _all_summaries(store, chunks))
            items: list[dict] = []
            for chunk in chunks:
                items.append({"chunk": chunk, "summary": leaves.get(chunk)})
                items.extend(_records_items(store, store.chunk_records(chunk), raw=raw))
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
        found = store.tool_call(handle)
        if found is None:
            raise ExpansionError(f"{handle} is unknown in this store")
        record, position, result = found
        assistant = store.record_raw(record) or {}
        calls = assistant.get("tool_calls") if isinstance(assistant.get("tool_calls"), list) else []
        name, _arguments = _call_parts(calls[position]) if position < len(calls) else (None, None)
        header = {"handle": handle, "kind": "tool_call", "form": "raw", "name": name, "call_in": record}
        if result is None:
            header["result"] = None
            header["note"] = "no result of this call is recorded in the store"
            return Target(header, [])
        return Target(header, [_result_item(result, store.record_raw(result) or {}, handle, inline=True)])
    if resolved.kind == MESSAGE:
        return Target({"handle": handle, "kind": "message", "form": form},
                      _records_items(store, [(handle, store.record_raw(handle) or {})], raw=True))
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
        if "name" in call:
            piece["name"] = call["name"]
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


def _replace_images(value: Any, images: list, describe: list) -> Any:
    if isinstance(value, dict):
        if is_image_part(value):
            images.append(value)
            describe.append(image_media_type(value))
            return {"type": "image", "image": len(images)}
        return {key: _replace_images(item, images, describe) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_images(item, images, describe) for item in value]
    return value


class PageBuilder:
    """Fills pages of ``target`` from a cursor, each at most ``limit`` characters as the
    host measures the result it receives."""

    def __init__(self, target: Target, *, limit: int, token_state: dict,
                 image_tokens: Callable[[dict], Optional[int]]):
        self.target = target
        self.limit = limit
        self.token_state = token_state
        self.image_tokens = image_tokens
        self._fields: dict[int, list] = {}

    def _fields_of(self, index: int) -> list:
        if index not in self._fields:
            self._fields[index] = fields_of(self.target.items[index])
        return self._fields[index]

    def _payload(self, page_items: list, page: int, next_page: Optional[str]) -> dict:
        return {**self.target.header, "items_total": len(self.target.items), "page": page,
                "next_page": next_page, "items": page_items}

    def render(self, page_items: list, page: int, next_page: Optional[str]) -> _Rendered:
        images: list = []
        media: list = []
        items = _replace_images(copy.deepcopy(page_items), images, media)
        text = final_result(self._payload(items, page, next_page))
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
        text_chars = len(rendered.text) + sum(len(label) for label in rendered.labels)
        return (len(rendered.summary) <= self.limit
                and text_chars + rendered.image_tokens * CHARS_PER_TOKEN <= self.limit)

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
            if isinstance(value, dict) and is_image_part(value):
                piece = _piece(item, path, value=value)
                if self.fits(page_items + [piece], page) or not page_items:
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
        target = target_for(records, cover, resolved, raw=bool(raw))
    token_state = {"v": TOKEN_VERSION, "t": tool_name, "s": store_uuid, "h": resolved.handle,
                   "m": "raw" if raw else "collapsed"}
    cursor = Cursor(state["i"], state["f"], state["o"]) if state else Cursor()
    page = state["n"] if state else 1
    if cursor.item >= len(target.items) and not (cursor.item == 0 and not target.items):
        raise ExpansionError("page is a garbled next_page token: it points past the end of what this handle "
                             "opens into")
    if cursor.field >= 0:
        fields = fields_of(target.items[cursor.item])
        if cursor.field >= len(fields):
            raise ExpansionError("page is a garbled next_page token: it names a field this item does not have")
        value = fields[cursor.field][1]
        length = len(value) if isinstance(value, str) else len(json.dumps(value, ensure_ascii=False))
        if cursor.offset > length or (cursor.offset and isinstance(value, dict) and is_image_part(value)):
            raise ExpansionError("page is a garbled next_page token: its offset lies outside the field it names")
    estimator = engine._estimator()
    builder = PageBuilder(target, limit=limit, token_state=token_state, image_tokens=estimator.image)
    result, _next = builder.build(cursor, page)
    return result
