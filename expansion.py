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
stays out, and nothing stands where it was. An image comes back as an image, in the host's
``_multimodal`` envelope, where the session's route carries it (the route's converter on the
image alone, ``route_image_check``) and its payload is within the host's byte budget; else it
stands as a held mark with its media type, its size as the host measures it, the message's
handle and why. Both are decided when the item is built, so the token's hash covers them. A
page holds at most the host's block ceiling of images and its byte budget of payload (the
host's constants); where the host's tool-loop guard has halted the turn, none (a read of the
host's state). Whether the host's next request carries the page is not predicted: that
request cannot be built from readable state (``image_room``); the page says what the host
does then, and the store keeps every image. The copies of repeated injections in the host's
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
lies (its structural path, a JSON array of steps). An item's plugin side (``lcm``) and
its stored message (``message``) are two objects (``Item``), so no stored key can shadow
the plugin's and none of the plugin's can shadow a stored key; the pieces of an item over
all its pages reassemble to exactly the whole item (``fields_of``), each delivered image
at its path. Every page is measured as it will be returned, with the real ``next_page`` of
the place it ends at, or null (``PageBuilder.measure``), with its images counted by the
model table's rule (#21). What cannot be split (an item without fields, an image, an
image's held mark, an empty field, the annotations every piece carries, the header every
page carries) stands alone on the next page if it fits there; where it does not fit
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
from .image_room import (
    ImageRoom,
    RoomUnavailable,
    convert_request,
    host_image_room,
    route_name,
    route_of,
    static_limits,
    wire_result,
)
from .message_content import image_media_type
from .record_store import HANDLE_RE, Cover, RecordStore, Resolved
from .results import final_result, is_envelope
from .tokens import CHARS_PER_TOKEN
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
# 4: an item's plugin side and host side are two objects (``lcm``, ``message``), a path is a
# tuple of structural steps (PR P): the fields and the pages moved.
TOKEN_VERSION = 4

# What each status of ``RecordStore.resolve`` tells the agent; "inactive" is told by its
# cause (``_inactive_text``).
_UNRESOLVED = {
    "malformed": ("{handle!r} is not a handle. A handle is a kind letter (m a message, t a tool call, c a chunk, "
                  "s a summary) and eight characters, as the summaries in your context and these tools show them."),
    "summary_revision": ("{handle} is the host's rewrite of summary {of} (the row the plugin returned, as the host "
                         "changed it). Expand {of} to read behind the summary."),
}
# "unknown" and "other_session" by the store fact that decided them (``Resolved.why``, #78).
_KIND_NAME = {"m": "message", "t": "tool call", "c": "chunk", "s": "summary"}
_UNREACHED = {
    "no_row": "{handle} is unknown in this store: no {kind} has this handle here.",
    # A summary resolves in the one session of the chunks it was written from (``resolve``):
    # said as that rule, never as the summary's own session, which is not read (#82).
    "no_chunk": ("{handle} is a summary in this store, but the store records no chunk under it, through its sources "
                 "or theirs; a summary resolves only in the session of its chunks, so it resolves in none."),
    "session": ("{handle} belongs to another session of this store. A handle resolves only in the session that holds "
                "it, so it does not resolve here."),
    "sessions": ("{handle} is a summary whose chunks the store records under {sessions} sessions; a summary resolves "
                 "only in the one session of its chunks, so it resolves in none."),
    "unrecorded_chunks": ("{handle} is a summary whose chunks the store holds no row of; a summary resolves only in "
                          "the session of its chunks, so it resolves in none."),
}

# Notes on calls and results: facts of the store (``pairing``), never what the host sends.
# Each cause as the pairing states it (``Pairing.unanswered``, ``stray``; #78, #82); every id as
# its JSON (``pairing.id_text``).
NO_RESULT = ("no result with this call's host id stands in its block; the store pairs a result only within its "
             "call's block")
_UNANSWERED = {
    "not_object": "this call is {kind}, not an object, so no result pairs with it",
    "no_id": 'this call has no "id", so no result pairs with it',
    "unpairable": "this call's host id is {state}, so no result pairs with it",
    "alias": ("result {result} of this call's block carries {alias}, another spelling of this call's host id, not the "
              "id itself; the store pairs by the id and does not pair them"),
}
# A call on a stored message whose role is not "assistant" (the only such calls a page shows:
# an assistant message shows calls only for a non-empty tool_calls list, which the pairing
# reads whole): its role as stored, and the store's rule.
NOT_ASSISTANT = ("this call's message {role}, not \"assistant\"; the store pairs results only with the calls of an "
                 "assistant message, so no result is paired with it")
_ALIAS_STATE = "this call's {key} is {state}"
_STRAY = {
    "unknown": "no call of assistant message {head}, which opens this result's block, has its host id {id} in any "
               "spelling",
    "no_id": "this result carries no host id, so no call pairs with it",
    "unpairable": "this result's host id is {state}, so no call pairs with it",
    # the block's opening record as its check found it (``pairing._head_fact``), and no more
    "no_calls": {
        "user": "this result's block opens with user message {head}, so no assistant call precedes it in its block",
        "no_key": ("this result's block opens with assistant message {head}, which has no tool_calls, so no "
                   "assistant call precedes it in its block"),
        "not_list": ("this result's block opens with assistant message {head}, whose tool_calls is {kind}, not a list, "
                     "so no assistant call precedes it in its block"),
        "empty": ("this result's block opens with assistant message {head}, whose tool_calls list is empty, so no "
                  "assistant call precedes it in its block"),
        "none": ("no user or assistant message precedes this result on the active record, so no assistant call "
                 "precedes it"),
    },
    "alias": ("this result's host id {id} is another spelling of call {position} of assistant message {head}, which "
              "opens its block, not that call's id; the store pairs by the id and does not pair them"),
}
# Rule 2, as ``_pair_block`` found it: one call whose exact id two or more results of its block
# carry; or calls joined by an id or another spelling of one, and the results of the block
# carrying any of their spellings.
_GROUP_ONE = ("call {n} of message {head} and {j} results of its block carry the host id {id} ({members}); the store "
              "cannot tell which of the results answered the call")
_GROUP = ("{k} calls of message {head} share an id or another spelling of one; their ids and other spellings are "
          "{spellings} ({members}); {j} results of its block carry one of these; the store cannot tell which result "
          "answered which call")
_FILL = "the host sends this empty message to the provider with its own stand-in as content: {content}"
# A later page whose target's identity differs from page 1's (``target_identity``): only that is
# known, never why.
CHANGED = "{what} renders differently now than at page 1; {again}"
_FILL_UNREAD = ("whether the host sends its stand-in {content} for this empty message depends on its reasoning-echo "
                "setting, which could not be read ({why})")



def _state(value: Any) -> str:
    """A stored id as a note says it (#82): its JSON (``pairing.id_text``), and where it never
    pairs (``pairing.unpairable``), which state it is and that it cannot pair. The item's
    ``message`` shows no host id, so this is the only place the stored value shows."""
    text = host_pairing.id_text(value)
    state = host_pairing.id_state(value)
    if state == "none":
        return "null, which cannot pair"
    if state == "empty":
        return f"blank ({text}), which cannot pair"
    if state == "nonfinite":
        return f"{text}, which is not a JSON value and cannot pair"
    if state == "structured":
        return f"{text}, which is not a string, a number or a boolean and cannot pair"
    return text


def _member_said(position: int, call: Any) -> str:
    """One call of a group as its note lists it: its host id in whatever state it is stored,
    and each alias of it that never pairs. A member is always an object: a call that is not
    one has no spelling (``pairing._aliases``) and joins no group."""
    said = [f"host id {_state(call['id'])}" if "id" in call else 'no "id"']
    said += [f"{key} {_state(call[key])}" for key in host_pairing.ID_KEYS[1:]
             if key in call and host_pairing.unpairable(call[key])]
    return f"call {position + 1}: " + ", ".join(said)


def _group_note(group: host_pairing.Group) -> str:
    """Rule 2's note, as ``_pair_block`` found the group (#78; #82, the review of c068d88: the
    fact, never more): the block (its opening message); for one call, the id its results
    carry; for joined calls, every id and other spelling of theirs, which the results matched;
    and every member call's host id in whatever state it is stored, each as its JSON."""
    members = "; ".join(_member_said(position, call) for (_record, position), call in zip(group.calls, group.members))
    if len(group.calls) == 1:
        return _GROUP_ONE.format(n=group.calls[0][1] + 1, head=group.calls[0][0], j=len(group.results),
                                 id=host_pairing.id_text(group.members[0].get("id")), members=members)
    return _GROUP.format(k=len(group.calls), j=len(group.results), head=group.calls[0][0], members=members,
                         spellings=", ".join(host_pairing.id_text(s) for s in group.spellings))


def _aliases_said(call: Any) -> list:
    """Each alias of a call outside a group (its ``call_id``, ``response_item_id``) that never
    pairs: not a scalar (``pairing.id_state``, #82). Its ``id`` is said by the cause."""
    if not isinstance(call, dict):
        return []
    return [_ALIAS_STATE.format(key=key, state=_state(call[key])) for key in host_pairing.ID_KEYS[1:]
            if key in call and host_pairing.unpairable(call[key])]


class ExpansionError(Exception):
    """Something the caller is told instead of a page: a wrong argument, a handle that
    does not resolve, a token that is not this store's."""


def unresolved_message(resolved: Resolved) -> str:
    if resolved.status == "inactive":
        return _inactive_text(resolved)
    if resolved.status in ("unknown", "other_session"):
        return _UNREACHED[resolved.why].format(handle=resolved.handle, kind=_KIND_NAME.get(resolved.kind, "row"),
                                               sessions=resolved.sessions)
    return _UNRESOLVED[resolved.status].format(handle=resolved.handle, of=resolved.of)


def call_named(resolved: Resolved) -> str:
    """A call as a text names it: its tool name, or the step at which its stored call gives
    none (``Resolved.name_why``), never "no name" (#78)."""
    return resolved.name if resolved.name is not None else f"no string tool name: {resolved.name_why}"


def _compaction_said(subject: str, cause: dict) -> str:
    """What the store records of the compaction that wrote ``subject`` and did not take
    effect (``RecordStore._compaction_cause``), each state as recorded (#78)."""
    compaction = cause["compaction"]
    if cause["kind"] == "rejected":
        text = (f"{subject} was recorded by compaction {compaction}, which the store records as rejected "
                f"({cause.get('how')})")
    elif cause.get("returned"):
        text = (f"{subject} was recorded by compaction {compaction}, whose return the store wrote; the store records no "
                f"confirmation, adoption or rejection of it")
    else:
        events = cause.get("events") or []
        text = (f"{subject} was recorded by compaction {compaction}, for which the store holds no return written"
                + (f" (its events: {', '.join(events)})" if events else " and no event"))
    if cause.get("never_held"):
        text += "; no effective compaction of this session held it in its list"
    return text + "; it is not on the active record."


def _inactive_text(resolved: Resolved) -> str:
    """The store's cause of a handle not being on the active record (round 5 of #71): only
    what the store records, never a guess at what the host did."""
    cause = resolved.cause or {"kind": "none"}
    kind = cause.get("kind")
    if resolved.kind == TOOL_CALL:
        prefix = (f"{resolved.handle} is call {(resolved.position or 0) + 1} ({call_named(resolved)}) of message "
                  f"{resolved.record}. ")
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
    elif kind in ("rejected", "unconfirmed"):
        text = _compaction_said(subject, cause)
    elif kind == "left":
        text = (f"{subject} was held by the host's list at compaction {cause['stood']}, which took effect; the host's "
                f"list at compaction {cause['gone']}, which took effect, held neither it nor a rewrite of it.")
    elif kind == "held_unreached":
        text = (f"{subject} was held by the host's list at compaction {cause['stood']}, which took effect, and every "
                f"later effective list held it or a rewrite of it; yet the session's latest effective return does not "
                f"reach it.")
    elif kind == "effective_unheld":
        text = (f"{subject} was recorded by compaction {cause['compaction']}, which took effect, but no effective "
                f"compaction of this session held it in its list; it is not on the active record.")
    elif kind == "effective_unreached" and resolved.kind == DERIVATION:
        # ``resolve`` checked that not every chunk the summary reaches lies under the cover.
        text = (f"{subject} was recorded by compaction {cause['compaction']}, which took effect, yet not every chunk "
                f"under it lies under a summary of the session's latest effective return.")
    elif kind == "effective_unreached":
        text = (f"{subject} was recorded by compaction {cause['compaction']}, which took effect, yet the session's "
                f"latest effective return does not reach it.")
    elif kind == "no_writer":
        text = f"{subject} is a summary the store records no compaction for; it is not on the active record."
    else:
        text = f"{subject} is not on the active record, and the store holds no record row of it."
    return prefix + text


# --- The page limit -------------------------------------------------------------------------

def calls_in_current_message(messages: Any) -> tuple[Optional[int], str]:
    """How many tool calls the assistant message now being answered holds, and "": the last
    assistant message in the live list the host hands the engine tool
    (``handle_tool_call(..., messages=messages)``, agent/tool_executor.py:1655; the host
    appends that message before running its calls and each result after it). Else None and
    why, each asked on its own (#78, A11): no list; no assistant message in it; the last
    assistant message has no ``tool_calls``, or not a list, or an empty one."""
    if not isinstance(messages, list):
        return None, (f"the messages argument of this call of the tool is not a list but {type(messages).__name__} "
                      f"(NoneType where the call carried no messages argument or carried None)")
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "assistant":
            if "tool_calls" not in message:
                return None, "the last assistant message of the list the host handed holds no tool_calls"
            calls = message["tool_calls"]
            if not isinstance(calls, list):
                return None, (f"the last assistant message of the list the host handed holds tool_calls that are "
                              f"not a list but {type(calls).__name__}")
            if not calls:
                return None, "the last assistant message of the list the host handed holds an empty tool_calls list"
            return len(calls), ""
    return None, f"the list the host handed ({len(messages)} entries) holds no object with role \"assistant\""


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
        # Everything in the try is the host's (its module, templates and functions), measured
        # with this tool's name: the text names that and the exception, nothing else (#78).
        raise ExpansionError(f"measuring the host's guardrail texts (agent.tool_guardrails) for {tool_name} raised "
                             f"{type(exc).__name__} ({exc}), so the room they take beside a page is not known") from None
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
        raise ExpansionError(f"importing or calling the host's _budget_for_agent (agent.tool_executor) for "
                             f"{tool_name}, or reading its threshold or turn budget, raised {type(exc).__name__} "
                             f"({exc}), so no page size is known") from None
    for name, value in (("threshold", threshold), ("turn budget", turn_budget)):
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ExpansionError(f"the host's {name} for {tool_name} is {value!r}, not a size a page can be "
                                 f"measured against")
    calls, why = calls_in_current_message(messages)
    if calls is None:
        raise ExpansionError(f"{why}, so the share of the host's per-message budget a page may take is not known")
    margin = host_guardrail_margin(tool_name)
    limit = min(int(threshold), int(turn_budget) // calls) - margin
    if limit <= 0:
        # What was computed, each term by itself; which one to change follows from them.
        raise ExpansionError(f"a page has no room: the smaller of the host's threshold for one {tool_name} result "
                             f"({int(threshold)} characters) and its budget of {int(turn_budget)} characters for the "
                             f"message's results divided among its {calls} tool calls ({int(turn_budget) // calls}), "
                             f"less the {margin} characters of the host's guardrail texts, is {limit}")
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
    # The fields every token has, then the fields of its tool: lcm_expand's handle, and
    # lcm_grep's term, scope and all (#18 D2).
    checks = (
        ("t", lambda v: isinstance(v, str) and bool(v), "a tool name"),
        ("s", lambda v: isinstance(v, str) and bool(v), "a store's uuid"),
        *(_TOOL_TOKEN_FIELDS.get(state.get("t"), _TOOL_TOKEN_FIELDS["lcm_expand"])),
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


# The fields a tool's token carries beside the common ones (``decode_token``). A token of a
# tool not named here is checked as lcm_expand's, and refused by the tool that reads it.
_TOOL_TOKEN_FIELDS = {
    "lcm_expand": (("h", lambda v: isinstance(v, str) and HANDLE_RE.fullmatch(v) is not None, "a handle"),),
    "lcm_grep": (
        ("q", lambda v: isinstance(v, str) and bool(v), "a search term"),
        ("p", lambda v: v == "" or (isinstance(v, str) and HANDLE_RE.fullmatch(v) is not None),
         "the session (\"\") or a handle"),
        ("a", lambda v: isinstance(v, bool), "true or false"),
    ),
}


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
    """One item of what a handle opens into (the plan of #71, §5.2), in two namespaces that
    are disjoint by construction (PR P):

    - the plugin's side, rendered under the one key ``lcm``: ``annotations``, what the plugin
      says of the item (``handle``, ``role``, ``result_of``, ``note``, ``content_chars``,
      ``chunk``, ``under``, ``summary``, ``summaries``, ``tail``, ``compaction_began_at``,
      ``images`` and the call's provenance statements ``route_note``, ``images_note``,
      ``guard_note``, ``commit_note``, ``_provenance``), which every piece of it carries; and
      ``plugin``, the plugin's own fields that can be split over
      pages (the readable ``reasoning``, each tool call's handle and notes by position under
      ``calls``, a summary's ``text``, grep's ``results_holding_term`` and ``messages``); a
      piece of a tool call carries that call's entry of ``calls`` as ``call``;
    - the host's side, rendered under the one key ``message``: ``fields``, the stored
      message's own keys as the host's rules leave them (§5.1), never a key of the plugin's.

    No stored key can shadow an annotation and no annotation can shadow a stored key: each
    lives in its own object. Each field becomes a piece when the item does not fit on a page
    by itself; a piece is its item by complement, never a hand-picked list. The pieces of an
    item over all its pages reassemble to exactly ``as_dict()``, every key and value, empty
    containers and empty strings included (``fields_of``); so the plugin's two dicts and
    the keys a piece adds (``PIECE_KEYS``) never share a name, which is checked here."""

    annotations: dict
    fields: dict = field(default_factory=dict)
    plugin: dict = field(default_factory=dict)
    # The plugin's record of the content paths that hold an image it placed (``_images_of``),
    # by structural path: "shown" (a canonical part to deliver) or "held:<why>" (its held
    # mark). The page reads images only from here, never by a part's shape or a key's name in
    # ``fields``, so a stored part that looks like one is the host's and untouched. It is
    # rendered once, as the annotation ``images`` (each path and whether an image or a held
    # mark stands there), so a reader tells the plugin's parts from the host's too.
    images: dict = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if self.images:
            self.annotations["images"] = [{"field": list(path), "shown": kind == "shown"}
                                          for path, kind in self.images.items()]
        shared = (set(self.annotations) & set(self.plugin)) | (set(self.annotations) & set(PIECE_KEYS))
        if shared:
            raise ValueError(f"an item's plugin side names a key twice: {sorted(shared)}")

    def as_dict(self) -> dict:
        out: dict = {"lcm": {**self.annotations, **self.plugin}}
        if self.fields:
            out["message"] = dict(self.fields)
        return out


# Host keys an item leaves out, with the reason (the plan of #71, §5.1). What is not named
# here is carried, a key the plugin does not know included.
_ID_KEYS = ("id", "call_id", "response_item_id")          # the host's ids: not identities; the handle replaces them
_NATIVE_CARRIERS = ("anthropic_content_blocks", "bedrock_content_blocks",
                    "codex_message_items")                  # re-encodings of the message: carried in raw form only


def _call_shown(call: Any) -> Any:
    """A tool call as the item's ``message`` shows it: the host's own call, every key of it
    in the host's own shape, except the host's ids and a withheld signature (the provider's
    thought signature, encrypted, as the Reasoning paragraph says). What the plugin says of
    the call (its handle, where its result is, its notes) is on the plugin's side
    (``lcm.calls``), never a key of the call: a host call key named like any plugin key is
    the host's and survives (PR P)."""
    if not isinstance(call, dict):
        return call                                            # a shape other than the host's: as stored
    entry: dict = {}
    for key, value in call.items():
        if key in _ID_KEYS:
            continue
        if key == "extra_content" and isinstance(value, dict):
            # The host writes {"google": {"thought_signature"}} (agent/gemini_native_adapter.py:557)
            # and reads the signature there or at the top, as thought_signature or
            # thoughtSignature (306-308; agent/transports/chat_completions.py:284-287): every
            # one of those is withheld.
            # Only what held a signature is left out: a container the host stored empty stays.
            value = copy.deepcopy(value)
            withheld = [value.pop(spelling) for spelling in ("thought_signature", "thoughtSignature")
                        if spelling in value]
            google = value.get("google")
            if isinstance(google, dict):
                inner = [google.pop(spelling) for spelling in ("thought_signature", "thoughtSignature")
                         if spelling in google]
                withheld += inner
                if inner and not google:
                    value.pop("google", None)
            if withheld and not value:
                continue
        entry[key] = value
    return entry


def _as_sent(raw: dict, route: "Route", *, raw_form: bool) -> tuple[dict, Optional[str]]:
    """The message by the host's per-row rules and what the host does with it when it is
    empty (the re-plan of #71, C2; LEARNINGSFÜRPLÄNE A10). First the host's own row, up to
    its fill, with the host's reasoning pad for the route (``host_row_before_fill``); the
    host's fill is asked on that row, before anything of the plugin's. The pad is the host's
    own decision (``Route.pad``): with a bound turn, the agent's ``_needs_thinking_reasoning_pad``,
    its reasoning-echo setting included, taken once; without one, the route family's decision,
    and where that is off, the host's fill under both values of the unread boolean setting
    (exhaustive, not an approximation): where they agree the fill is known and said plainly;
    where they differ the note says the fill depends on the setting, which could not be read.
    Then, on a copy, only the plugin's own transformations (``item_message``), the host's
    ``_``-prefixed in-process markers left out (its chat transport strips them as scaffolding,
    agent/transports/chat_completions.py:375-) and, in the collapsed form, the native carriers.
    Returns the message and its note."""
    try:
        row = host_row_before_fill(raw, pad=route.pad)
        fill = host_fill_text(row)
        note = _FILL.format(content=json.dumps(fill, ensure_ascii=False)) if fill is not None else None
        if route.pad_unread and not route.pad:
            # No bound turn: the agent's reasoning-echo setting, a boolean that can turn the pad
            # on, could not be read. The host's own fill under both of its values is exhaustive
            # over it: where they agree the fill is known; where they differ the setting decides
            # this item, and the note says so.
            other = host_fill_text(host_row_before_fill(raw, pad=True))
            if other != fill:
                note = _FILL_UNREAD.format(content=json.dumps(fill if fill is not None else other,
                                                                ensure_ascii=False), why=route.pad_unread)
        message = item_message(row)
    except HostUnavailable as exc:
        raise ExpansionError(str(exc)) from None
    for key in [key for key in message if isinstance(key, str) and key.startswith("_")]:
        message.pop(key, None)
    if not raw_form:
        for key in _NATIVE_CARRIERS:
            message.pop(key, None)
    return message, note


def _call_notes(found: host_pairing.Pairing, handle: str, position: int, raw: Any) -> list:
    """The note of one call of the stored message ``raw``: where its role is not "assistant",
    that; else its group's (rule 2), or why no result of its block pairs with it, each cause as
    the pairing states it (#78, #82): not an object; no id; an id that cannot pair; an alias
    of it on a result of its block; none of its block carries its id; and every alias of it
    that never pairs (``_aliases_said``), for a paired call too. An assistant message whose
    calls a page shows has a non-empty ``tool_calls`` list and opens its block, so the pairing
    holds every one of its calls."""
    if not (isinstance(raw, dict) and raw.get("role") == "assistant"):
        role = (f"has role {host_pairing.id_text(raw['role'])}" if isinstance(raw, dict) and "role" in raw
                else "has no role")
        return [NOT_ASSISTANT.format(role=role)]
    key = (handle, position)
    if key in found.group:
        return [_group_note(found.group[key])]
    also = _aliases_said(found.calls[key])
    if key in found.answer:
        return also
    why, detail = found.unanswered[key]
    if why == "not_object":
        return [_UNANSWERED["not_object"].format(kind=detail)]
    if why == "no_id":
        return [_UNANSWERED["no_id"]] + also
    if why == "unpairable":
        return [_UNANSWERED["unpairable"].format(state=_state(detail))] + also
    if why == "alias":
        return [_UNANSWERED["alias"].format(result=detail[0], alias=host_pairing.id_text(detail[1]))] + also
    return [NO_RESULT] + also


def _result_notes(found: host_pairing.Pairing, handle: str) -> list:
    """The note of one result: its group's, or why it belongs to no call; or none."""
    if handle in found.group:
        return [_group_note(found.group[handle])]
    if handle in found.stray:
        why, call_id, detail = found.stray[handle]
        if why == "no_calls":
            head, fact, kind = detail
            return [_STRAY["no_calls"][fact].format(head=head, kind=kind)]
        if why == "alias":
            head, owner = detail
            return [_STRAY["alias"].format(id=host_pairing.id_text(call_id), head=head, position=owner + 1)]
        return [_STRAY[why].format(id=host_pairing.id_text(call_id), state=_state(call_id), head=detail)]
    return []


def _message_item(handle: str, raw: dict, calls: dict[int, str], found: host_pairing.Pairing, route: "Route", *,
                  inline: set, raw_form: bool) -> Item:
    """One user or agent message by the host's rules (``_as_sent``). The plugin's side: its
    readable reasoning, and per tool call (``lcm.calls``, by position) its handle and what the
    store's pairing says of it; in raw form also where its result is when this stretch does
    not hold it, and for a call of a group the store cannot pair, every result of the group.
    The host's side: the message's own keys, each tool call in the host's shape
    (``_call_shown``)."""
    message, fill_note = _as_sent(raw, route, raw_form=raw_form)
    images = _images_of(message, handle, route)
    annotations = {"handle": handle, "role": message.pop("role", raw.get("role"))}
    if fill_note is not None:
        annotations["note"] = fill_note
    plugin: dict = {}
    if raw.get("role") == "assistant":
        reasoning = readable_reasoning(raw)
        if reasoning is not None:
            plugin["reasoning"] = reasoning
    fields: dict = {"content": message.pop("content")} if "content" in message else {}   # never invented
    stored_calls = "tool_calls" in message
    tool_calls = message.pop("tool_calls", None)
    if isinstance(tool_calls, list) and tool_calls:
        said = []
        for position, call in enumerate(tool_calls):
            entry: dict = {"handle": calls.get(position)}
            notes = _call_notes(found, handle, position, raw)
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
            said.append(entry)
        plugin["calls"] = said
        fields["tool_calls"] = [_call_shown(call) for call in tool_calls]
    elif stored_calls:
        # Any other value (``[]``, ``null``, a string, …) is the host's own, as stored (PR P;
        # staging dropped it).
        fields["tool_calls"] = tool_calls
    message.pop("tool_call_id", None)
    fields.update(message)
    _provenance(annotations, images, route, fill=fill_note is not None)
    return Item(annotations, fields, plugin, images)


def _result_item(handle: str, raw: dict, call: Optional[str], route: "Route", *, inline: bool, notes: list,
                 raw_form: bool) -> Item:
    """A tool result by the host's rules, answering ``call``; where it stands as a pointer
    (``inline`` false) its content is replaced by its length and every other key stays."""
    message, _fill = _as_sent(raw, route, raw_form=raw_form)
    images = _images_of(message, handle, route)
    annotations: dict = {"handle": handle, "role": message.pop("role", raw.get("role")), "result_of": call}
    message.pop("tool_call_id", None)
    if "content" not in message or inline:          # a content the host did not store is never invented
        fields = {"content": message.pop("content")} if "content" in message else {}
        fields.update(message)
    else:
        content = message.pop("content")
        annotations["content_chars"] = (len(content) if isinstance(content, str)
                                        else len(json.dumps(content, ensure_ascii=False)))
        fields = dict(message)
        images = {}                     # a pointer shows no content
    if notes:
        annotations["note"] = "; ".join(notes)
    _provenance(annotations, images, route)
    return Item(annotations, fields, images=images)


def _provenance(annotations: dict, images: dict, route: "Route", *, fill: bool = False) -> None:
    """The call's statements about where a value came from, on the item that value decided
    (LEARNINGSFÜRPLÄNE A15: provenance belongs to the value), as annotations, so every piece
    of the item carries them and the token's hash covers them: ``route_note`` where the
    route was read from the engine and decided something here (an image it was asked about,
    or the host's fill of an empty message, its reasoning pad); ``images_note`` (the host's
    own retirement) and ``guard_note`` (the guard could not be read) where an image is
    delivered; ``commit_note`` where an image is delivered and the host's commit was asked
    without the running agent (its learned set unread). Each is a constant of the call
    (``Route.room``)."""
    room = route.room
    if room is None:
        return
    # The images whose fate the route decided: the host's commit asked with its provider and
    # model, the converter asked, or shown after both (``_images_of``).
    asked = any(kind in ("shown", "held:commit", "held:route") for kind in images.values())
    if room.route_note and (asked or fill):
        annotations["route_note"] = room.route_note
    if "shown" in images.values():
        annotations["images_note"] = room.note
        if room.guard_unread:
            annotations["guard_note"] = room.guard_unread
        if room.commit_unread:
            annotations["commit_note"] = room.commit_unread


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
        """The store's pairing of ``stretch`` through the blocks it lies in: block-scoped, as
        the store's rule is (``pairing``); no lookup outside the blocks (#82)."""
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
            # write order, each with the compaction that wrote it and the state the store records
            # of that compaction (``RecordStore.compaction_state``). No choice is made among them.
            leaves: dict[str, list] = {}
            for derivation, chunk in _all_summaries(store, chunks):
                written_by, state = store.compaction_state(derivation)
                leaves.setdefault(chunk, []).append({"summary": derivation, "compaction": written_by,
                                                     "compaction_state": state})
            items: list[Item] = []
            for chunk in chunks:
                items.append(Item({"chunk": chunk, "summaries": leaves.get(chunk, [])}))
                items.extend(_records_items(store, order, store.chunk_records(chunk), raw=raw))
            return Target(header, items)
        # A summary written from summaries (#34 D6, "summaries"): one layer down. Where its
        # sources mix summaries and chunks the form says so, and a chunk source says what it
        # is (#78: the store's sources as they are, never one form for two).
        header["form"] = "summaries" if all(derivation for _chunk, derivation in sources) else "summaries and chunks"
        items = []
        for chunk, derivation in sources:
            if derivation:
                items.append(Item({"summary": derivation,
                                   "note": "a summary: a description of what happened, not what happened; expand "
                                           "its handle to read behind it"},
                                  plugin={"text": store.derivation_text(derivation)}))
            else:
                items.append(Item({"chunk": chunk, "note": "a chunk this summary was written from; expand its "
                                                           "handle to read its messages"}))
        return Target(header, items)
    if resolved.kind == TOOL_CALL:
        # The call's result is the stored result of its exact id in its block (``pairing``):
        # read now, never stored. For a call of a group the store cannot pair, every result
        # of the group, attributed to none.
        record, position = resolved.record, int(resolved.position or 0)
        header = {"handle": handle, "kind": "tool_call", "form": "raw", "name": resolved.name, "call_in": record}
        if resolved.name is None:
            header["name_note"] = call_named(resolved)     # the step, never "no name" (#78)
        found = order.pairing(store, [record])
        notes = _call_notes(found, record, position, store.records_raw([record]).get(record))
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
        # A message is shown in raw form whatever was asked, and its header says so (#78: the
        # header's form is the form of its items).
        return Target({"handle": handle, "kind": "message", "form": "raw"},
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

# A field's path is a tuple of structural steps into the item, never a string (PR P): its
# first step names the side ("lcm", the plugin's fields; "message", the host's), then each
# step is a dict key taken literally at its own level (a ``str``) or a list index (an
# ``int``). A host key is a literal at its level, so a key named "content[0]" is the path
# ("message", "content[0]"), which can never equal the structural ("message", "content", 0).
# A piece carries its path as it is, a JSON array of those steps (``lcm.field``): a string is
# a key, a number an index, so a reader places a piece without parsing anything; a string
# form such as a JSON Pointer could not tell the index 0 from a key "0".
Path = tuple


def fields_of(item: Item) -> list[tuple[Path, Any]]:
    """The fields an item is split into when it does not fit on a page by itself: every
    field of the plugin's side, in order, then every key of the host's side, in order;
    content by part (a list, or each part and key of an envelope), and each tool call by
    its arguments. The pieces of an item over all its pages reassemble to exactly
    ``Item.as_dict()`` (LEARNINGSFÜRPLÄNE A12, whole × split): every key yields at least one
    field, and a list with no element to split by is one field holding the empty list.
    Deterministic, so a cursor naming a field and an offset finds the same place on every
    call."""
    out: list[tuple[Path, Any]] = [(("lcm", key), value) for key, value in item.plugin.items()]
    for key, value in item.fields.items():
        if key == "content" and isinstance(value, list) and value:
            out.extend((("message", "content", i), part) for i, part in enumerate(value))
        elif key == "content" and isinstance(value, dict) and isinstance(value.get("content"), list):
            for inner, said in value.items():
                if inner == "content" and said:
                    out.extend((("message", "content", "content", i), part) for i, part in enumerate(said))
                else:
                    out.append((("message", "content", inner), said))
        elif key == "tool_calls" and isinstance(value, list) and value:
            for index, call in enumerate(value):
                function = call.get("function") if isinstance(call, dict) else None
                if isinstance(function, dict) and "arguments" in function:
                    out.append((("message", "tool_calls", index, "function", "arguments"), function["arguments"]))
                else:
                    out.append((("message", "tool_calls", index), call))
        else:
            out.append((("message", key), value))
    return out


# The keys a piece adds on its item's plugin side: an annotation never carries one (``Item``).
PIECE_KEYS = ("field", "offset", "chars", "json", "call")


class Piece(dict):
    """One piece of an item as it is rendered (a plain dict to JSON), with its field's
    structural ``path`` and the plugin's record of an image there (``image``, from
    ``Item.images``) beside it, never keys of it."""

    path: Path = ()
    image: str = ""


def _piece(item: Item, path: Path, *, value: Any = None, text: Optional[str] = None, offset: int = 0,
           total: int = 0, json_text: bool = False) -> Piece:
    """One field of an item that does not fit on a page by itself: on the plugin's side
    (``lcm``) every annotation of the item, by complement, the field's path (a JSON array of
    its steps) and the slice's place; beside it the host's value or text. A tool call's piece
    also carries what the plugin says of the call (``lcm.call``) and, for its arguments, the
    rest of the host's call (``tool_call``)."""
    said: dict = dict(item.annotations)
    said["field"] = list(path)
    piece = Piece()
    if len(path) >= 3 and path[:2] == ("message", "tool_calls") and isinstance(path[2], int):
        calls = item.plugin.get("calls") or []
        if 0 <= path[2] < len(calls):
            said["call"] = calls[path[2]]
        if path[3:] == ("function", "arguments"):
            call = item.fields["tool_calls"][path[2]]
            piece["tool_call"] = {**call, "function": {k: v for k, v in call["function"].items() if k != "arguments"}}
    if text is not None:
        said["offset"] = offset
        said["chars"] = total
        if json_text:
            said["json"] = True          # the field's value as JSON text, split by offset
    piece["lcm"] = said
    if text is None:
        piece["value"] = value
    else:
        piece["text"] = text
    piece.path = path
    # The plugin's record of the image at this path (``Item.images``), beside the piece and
    # never a key of it: the page reads it (``_replace_images``), never the value's shape.
    piece.image = item.images.get(path, "")
    return piece


# --- Pages -----------------------------------------------------------------------------------

# --- Images: identity, the route's verdict, the held mark ---------------------------------

# An image is identified by its content (LEARNINGSFÜRPLÄNE A9): the canonical part as JSON
# with sorted keys. Whether the session's route carries it is decided once per call and per
# identity, when the item is built, so the token's hash covers it (the plan of PR B, §2). So
# is whether a request of this call has room for any image (the guard, the host's limits:
# ``Route.room``); how many images a page holds is the page's (``PageBuilder.fits``).


def _host_image_helpers() -> tuple[Callable[[Any], bool], Callable[[dict], tuple]]:
    """The host's own image test and size (agent/context_compressor.py:1464-1487, 1579-1584
    at Hermes 1c535d9689): ``_is_image_part`` (``image_url``, ``input_image``, ``image``) and
    ``_image_payload`` (the data URL's or base64 source's length, summed per part)."""
    try:
        from agent.context_compressor import _image_payload, _is_image_part  # type: ignore
    except Exception as exc:
        raise ExpansionError(f"the host's image test (agent.context_compressor) cannot be read "
                             f"({type(exc).__name__}: {exc}), so which content parts are images is not known") from None
    return _is_image_part, _image_payload


def canonical_image_part(part: dict) -> tuple[Optional[dict], str]:
    """An image part in the host's canonical list-content form, ``{"type": "image_url",
    "image_url": {"url": …}}``, which every converter of the host takes (the Chat Completions
    path as it is; the Anthropic converter by ``_image_block_from_openai_url``,
    agent/anthropic_message_convert.py:170-187; the Responses converter by
    ``_input_image_part``, agent/codex_responses_adapter.py:218-246). An Anthropic ``image``
    block's base64 source becomes a data URL, its URL source that URL; a Responses
    ``input_image`` its ``image_url``. Returns (the part, "") or (None, why) where no URL can
    be made: such an image is held, with why."""
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
                          + (" (it carries a file id)" if part.get("file_id") else ""))
    elif kind == "image":
        source = part.get("source")
        if isinstance(source, dict) and source.get("type") == "base64" and isinstance(source.get("data"), str):
            media = source.get("media_type")
            if not isinstance(media, str) or not media:
                return None, "stored as an Anthropic image block without a media type, which gives no data URL"
            url = f"data:{media};base64,{source['data']}"
        elif isinstance(source, dict) and source.get("type") == "url" and isinstance(source.get("url"), str):
            url = source["url"]
        else:
            # A stored value: its kind in JSON's words, its value as JSON (#82, the type-word ruling).
            if "source" not in part:
                said = "without a source"
            elif not isinstance(source, dict):
                said = f"whose source is {host_pairing.json_kind(source)}, not an object"
            elif "type" not in source:
                said = "whose source object has no type"
            else:
                said = f"whose source has type {host_pairing.id_text(source['type'])}"
            return None, f"stored as an image block {said}, which gives no URL"
    else:
        return None, f"stored as a part of type {kind!r}"
    image_url: dict = {"url": url}
    if detail:
        image_url["detail"] = detail
    return {"type": "image_url", "image_url": image_url}, ""


def _held_mark(part: dict, handle: str, cause: str) -> dict:
    """An image not shown: its media type, its size as the host measures it, the message's
    handle and why (PR A's mark, with the cause in ``not_shown``)."""
    _is_image, image_size = _host_image_helpers()
    mark = {"type": "image", "media_type": image_media_type(part),
            "size": int(image_size({"role": "user", "content": [part]})[1]), "in": handle, "not_shown": cause}
    if part.get("file_id"):
        mark["file_id"] = part["file_id"]
    return mark


def _probe_messages(tool_name: str, part: dict) -> list:
    call_id = "call_lcm_route_check"
    return [{"role": "user", "content": "."},
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": call_id, "type": "function", "function": {"name": tool_name, "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": call_id, "name": tool_name,
             "content": [{"type": "text", "text": "."}, part]}]


def route_image_check(route: "Route", tool_name: str = "lcm_expand") -> Callable[[dict], str]:
    """Whether the session's route carries an image to the model as an image: the route's
    own converter (``image_room.convert_request``) on a request whose one tool result holds
    only this image; carried where that result, as the converter paired it with the call,
    holds an image block (one image, one slot: by position, never by bytes, which a converter
    may re-encode). "" where it does, else why not. It depends on the image and the route
    only, so every page of a target shares it and the token's hash covers it. Decided once per
    image per call, keyed by the canonical part (build
    decision A: our own data, not a seam with the host)."""
    seen: dict[str, str] = {}

    def check(part: dict) -> str:
        key = json.dumps(part, sort_keys=True, ensure_ascii=False)
        if key not in seen:
            probe = _probe_messages(tool_name, part)
            # Only ``image_room.convert_request`` is inside the try (#78, A11): a failure of the
            # plugin's own reading of its output after it is the tool's failure. It holds the
            # host's transport, provider profile and converter calls and the plugin's reading of
            # their output, so its failure is said as that (#82), never as the converter's alone.
            try:
                wire = convert_request(route, probe)
            except RoomUnavailable as exc:
                seen[key] = str(exc)
                return seen[key]
            except Exception as exc:
                seen[key] = (f"converting a probe request holding this image for this session's route "
                             f"({route_name(route)}) with the host's transport, provider profile and converter, and "
                             f"this plugin's reading of their output, raised {type(exc).__name__} ({exc})")
                return seen[key]
            try:
                found, broken = wire_result(wire, probe, 0)
            except RoomUnavailable as exc:
                seen[key] = str(exc)
                return seen[key]
            # Each cause named where it arises: the pairing breaks at a step ``wire_result``
            # names, or the paired result holds no image block.
            if found is None:
                seen[key] = (f"the host's converter for this session's route ({route_name(route)}) does not carry "
                             f"this image to the model: {broken}")
            elif found < 1:
                seen[key] = (f"the host's converter for this session's route ({route_name(route)}) does not carry "
                             f"this image to the model as an image: the tool result it builds holds no image block")
            else:
                seen[key] = ""
        return seen[key]

    return check


def _over_budget(part: dict) -> str:
    """Why an image can never be carried by its size: its payload alone, as the host measures
    it, over the host's byte budget for one request (a constant, image_eviction_policy.py:27);
    "" where it is within it."""
    try:
        _limit, budget, measure = static_limits()
    except RoomUnavailable as exc:
        return f"{exc}, so the host's byte budget for images is not known"
    size = int(measure({"role": "tool", "content": [part]})[1])
    if size > budget:
        return (f"this image is {size} bytes as the host measures it, over the host's budget of {budget} bytes for the "
                f"images of one request, so the host strips it from any request")
    return ""


def _images_of(message: dict, handle: str, route: "Route") -> dict:
    """Every image part of the message's content (a list, or the ``_multimodal`` envelope's
    list), in place: the canonical part where the route carries it, else its held mark with
    why (the route cannot carry it; the plugin cannot read the route or its converter; no
    URL can be made of it; the request of this call has no room for any image: the host's
    tool-loop guard has halted the turn, or the host's limits cannot be read, both constants
    of the call, ``Route.room``). Returns which content paths of the item (structural,
    ``Path``) now hold such a part, "shown" or "held:<why>" (``Item.images``): the plugin's
    own record of what it placed. Every image is decided here, when the item is built, so the
    token's hash covers it and the pieces of an item reassemble to it on every page."""
    content = message.get("content")
    # The host's own envelope test (``results.is_envelope``: ``_multimodal is True`` and a list
    # ``content``), never a truthiness of our own (#78: a marker that is not True is stored
    # content, and no image of it is decided).
    envelope = is_envelope(content)
    parts = content.get("content") if envelope else content
    if not isinstance(parts, list):
        return {}
    prefix: Path = ("message", "content", "content") if envelope else ("message", "content")
    is_image, _size = _host_image_helpers()
    out, placed = [], {}
    for index, part in enumerate(parts):
        if not is_image(part):
            out.append(part)
            continue
        canonical, why = canonical_image_part(part)
        if canonical is None:
            # The function's own cause only; nothing about the converters is claimed here.
            out.append(_held_mark(part, handle, f"not shown: an image {why}, so no image part can be made of it; the "
                                                f"store still holds it"))
            placed[(*prefix, index)] = "held:part"            # the part itself: no route asked
            continue
        # In the order of the host's chain from this result to the model, the first step that
        # drops the image names why: the commit (text summary, the guard; ``Route.room``), the
        # next request's byte budget, the route's converter.
        room = route.room
        verdict, by = "", ""
        if room is not None and room.commit:
            verdict, by = room.commit, "held:commit"
        elif room is not None and room.blocked():
            verdict, by = room.blocked(), "held:room"
        if not verdict:
            verdict, by = _over_budget(canonical), "held:budget"
        if not verdict:
            verdict = route.check(canonical) if route.check is not None else "the session's route was not read"
            by = "held:route"
        out.append(canonical if not verdict else _held_mark(part, handle, f"not shown: {verdict}; the store still "
                                                                         f"holds it"))
        placed[(*prefix, index)] = by if verdict else "shown"
    if isinstance(content, dict):
        message["content"] = dict(content, content=out)
    else:
        message["content"] = out
    return placed


def is_mark(item: Item, path: Path) -> bool:
    """A held mark the plugin placed at ``path`` (``Item.images``): it is never split. Read
    from the plugin's record, never from a part's shape."""
    return str(item.images.get(path, "")).startswith("held")


def is_content_image(item: Item, path: Path) -> bool:
    """An image the plugin delivers at ``path`` (``Item.images``), never a value elsewhere or
    a stored part that merely looks like one."""
    return item.images.get(path) == "shown"


@dataclass(frozen=True)
class Route:
    """The session's route, the one snapshot of a tool call (``image_room.route_of``), read
    once at the call's entry: the attributes of the agent the host bound for this turn, where
    its engine is this engine; else the engine's last ``update_model`` (two host paths change
    the agent's route without telling the engine: the welcome-tier model switch and the Bedrock
    stream fallback, agent/turn_recovery.py:305-313, 596), and ``source`` says which. Every
    reader of the route in the call takes it from here: the host's rows of the items (its
    reasoning pad decides the fill note), the route's verdict on each image (``check``, which
    the token's hash covers), the page's image counts by the model table (``estimator``), the
    note saying where the route was read, and the guard's read of the bound agent (``agent``).
    The window (``context_length``) has one source too, the engine: the host's own spill
    threshold reads it there (``agent.context_compressor.context_length``,
    agent/tool_executor.py:112-126), and so does ``host_page_limits``."""

    api_mode: str = ""
    model: str = ""
    base_url: str = ""
    provider: str = ""
    # The host's own reasoning-pad decision: with a bound turn, the agent's own
    # ``_needs_thinking_reasoning_pad`` (its reasoning-echo setting, ``_reasoning_echo_flag``,
    # included; agent/reasoning_params.py:163-179); else the route family's decision
    # (``host_reasoning_pad``), and ``pad_unread`` says why the setting could not be read.
    pad: bool = False
    estimator: Any = None
    check: Optional[Callable[[dict], str]] = None
    source: str = "engine"
    agent: Any = field(default=None, compare=False)
    pad_unread: str = "no running turn was bound"
    # Why the agent's values were not read, where the route came from the engine (``route_of``).
    why: str = ""
    # What a request of this call has room for (``image_room.host_image_room``): the host's
    # commit (content list or text summary), the guard's halt of the bound agent, the host's
    # static limits and the call's provenance statements, read once with the rest of the
    # snapshot, before any item is built.
    room: Any = field(default=None, compare=False)

    @classmethod
    def of(cls, engine: Any, tool_name: str = "lcm_expand") -> "Route":
        values, source, agent, why = route_of(engine)
        api_mode, model, base_url, provider = (values["api_mode"], values["model"], values["base_url"],
                                               values["provider"])
        pad, pad_unread = None, why or "no running turn was bound"
        if agent is not None and source == "agent":
            try:
                # The host's own method, on a shallow copy of the agent: it caches its answer on
                # the object it runs on, and the plugin writes nothing into the agent.
                pad, pad_unread = bool(copy.copy(agent)._needs_thinking_reasoning_pad()), ""
            except AttributeError as exc:
                # Raised by the host's method or on its way; the text names what was asked and
                # what came back, never which attribute is "missing".
                pad_unread = f"the bound agent's _needs_thinking_reasoning_pad raised AttributeError ({exc})"
        if pad is None:
            try:
                pad = host_reasoning_pad(provider, model, base_url)
            except HostUnavailable as exc:
                raise ExpansionError(str(exc)) from None
        bare = cls(api_mode, model, base_url, provider, pad)
        return cls(api_mode, model, base_url, provider, pad, engine._estimator(values),
                   route_image_check(bare, tool_name), source, agent, pad_unread, why,
                   host_image_room(source, agent, why, values, tool_name))


# --- Pages -----------------------------------------------------------------------------------

@dataclass
class _Rendered:
    text: str
    images: list
    labels: list
    image_tokens: Optional[int]           # None where an image on it has no count
    summary: str
    places: tuple = ()                    # each image's place in the target (``_replace_images``)


def _replace_images(page_items: list, images: list, describe: list, places: Optional[list] = None) -> list:
    """The page's items with each image to deliver replaced by its number on the page, the
    images collected in order, and with ``places`` each image's place in the target (its
    message's handle and its path), which names it to the room, never its bytes. Only the
    images the plugin placed are delivered (``Item.images``; a piece's ``image``), never a
    part found by its shape. A whole item's rendered form changes only at the plugin's own
    image paths, followed step by step (``Path``); the host's dict loses nothing."""
    places = places if places is not None else []

    def take(part: dict, place: tuple) -> dict:
        images.append(part)
        describe.append(image_media_type(part))
        places.append(place)
        return {"type": "image", "image": len(images)}

    replaced: list = []
    for item in page_items:
        if isinstance(item, Item):
            out = item.as_dict()
            handle = item.annotations.get("handle")
            for path, kind in item.images.items():
                if kind != "shown":
                    continue
                node: Any = out
                for step in path[:-1]:
                    node = node[step]
                node[path[-1]] = take(node[path[-1]], (handle, path))
            replaced.append(out)
            continue
        out = dict(item)
        if isinstance(item, Piece) and item.image == "shown" and "value" in out:
            out["value"] = take(out["value"], (out["lcm"].get("handle"), item.path))
        replaced.append(out)
    return replaced


class PageBuilder:
    """Fills pages of ``target`` from a cursor, each at most ``limit`` characters of the
    exact string the engine returns. One measure (``measure``): a candidate page as it will
    be returned, with the real ``next_page`` of the place it ends at, or null where nothing
    is left (the orchestrator's cut of #71, B; LEARNINGSFÜRPLÄNE A9). Every site uses it:
    a whole item, the first piece, every slice (the rest of a field is tried first, ending
    where it ends), a unit standing alone, the header."""

    def __init__(self, target: Target, *, limit: Any, token_state: dict,
                 image_tokens: Optional[Callable[[dict], tuple]] = None,
                 image_room: Optional[ImageRoom] = None):
        self.target = target
        # ``limit`` is a ``PageLimit``, or a bare number where no origin is known.
        self.origin = limit if isinstance(limit, PageLimit) else None
        self.limit = limit.limit if isinstance(limit, PageLimit) else int(limit)
        self.token_state = token_state
        # (tokens and the rule's basis, or None and why uncounted): ``Estimator.image_count``
        # of the call's route.
        self.image_tokens = image_tokens or (lambda _part: (None, "no route was handed to the page to count it by"))
        self.image_room = image_room or ImageRoom()
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

    def render(self, page_items: list, page: int, next_page: Optional[str]) -> "_Rendered":
        """The page as it will be returned: its text part, and where it holds images, the
        images in order, a label per image, their counts by the model table's rule (#21) and
        the ``text_summary`` the host sends a model that reads no images."""
        images: list = []
        media: list = []
        places: list = []
        items = _replace_images(copy.deepcopy(page_items), images, media, places)
        # The statements about where a value came from are on the items that value decided,
        # under ``lcm`` (``_provenance``), never on the page's header.
        payload = {**self.target.header, "items_total": len(self.target.items), "page": page, "next_page": next_page,
                   "items": items}
        text = final_result(payload)
        labels: list = []
        counted: Optional[int] = 0
        notes = []
        for number, part in enumerate(images, start=1):
            tokens, said = self.image_tokens(part)
            counted = None if tokens is None or counted is None else counted + tokens
            # The count's own basis, or its own cause (``Estimator.image_count``), each as the
            # function states it: a label never claims a rule it did not get for this route.
            count = (f"{tokens} tokens by the image rule in {said}" if tokens is not None
                     else f"not counted: {said}; so it stands alone on its page")
            labels.append(f"[image {number} of this page ({media[number - 1]}; {count}), which the page above names "
                          f"as image {number}]")
            notes.append(f"[image {number} of this page ({media[number - 1]}) is not shown in this text]")
        summary = text + ("\n" + "\n".join(notes) if notes else "")
        return _Rendered(text, images, labels, counted, summary, tuple(places))

    def measure(self, page_items: list, page: int, end: tuple) -> int:
        """The characters of the page ``page_items`` as it will be returned, ending at ``end``:
        its text, or where it holds images its ``text_summary`` (the longest string the host
        measures of it; every text part is shorter)."""
        rendered = self.render(page_items, page, self.token_at(page, end))
        return len(rendered.summary) if rendered.images else len(rendered.text)

    def fits(self, page_items: list, page: int, end: tuple) -> bool:
        """Whether the page fits as it will be returned: every string the host measures within
        the limit; with images, their counts by the model table's rule beside its text within
        the limit too (#21; an image no rule counts never fits beside anything: it stands
        alone), and the request's room for them (``ImageRoom``)."""
        rendered = self.render(page_items, page, self.token_at(page, end))
        if not rendered.images:
            return len(rendered.text) <= self.limit
        if rendered.image_tokens is None or not self.image_room.admits(rendered.images, rendered.places):
            return False
        text_chars = len(rendered.text) + sum(len(label) for label in rendered.labels)
        return (len(rendered.summary) <= self.limit
                and text_chars + rendered.image_tokens * CHARS_PER_TOKEN <= self.limit)

    def _room_alone(self, piece: dict) -> bool:
        images: list = []
        places: list = []
        _replace_images([dict(piece)], images, [], places)
        return self.image_room.admits(images, tuple(places))

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
                if is_mark(item, path) or (not isinstance(value, str) and start == 0):
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
            return self._output(self.render(rest, page, None)), None
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
            if is_content_image(item, path):
                piece = _piece(item, path, value=value)
                if self.fits(page_items + [piece], page, after):
                    page_items.append(piece)
                    f, o = f + 1, 0
                    continue
                if page_items:
                    break
                if not self._room_alone(piece):
                    # What a request has room for is decided when the item is built
                    # (``_images_of``, ``Route.room``); an image shown there that no request
                    # admits alone is refused visibly, never held here or sent unseen.
                    raise ExpansionError(f"page {page} of {self.target.header.get('handle')} cannot be built: "
                                         f"{_identity(item, piece)} is an image the host's request limits do not "
                                         f"admit even alone ({self.image_room.why_not(value)}). Nothing was "
                                         f"skipped.")
                # An image no rule counts, or larger than a page, stands alone; its page's text
                # must fit.
                if self.measure([piece], page, after) > self.limit:
                    self._refuse(page, i, piece, after)
                page_items.append(piece)
                i, f, o = after
                break
            if is_mark(item, path):
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
        return self._output(self.render(page_items, page, next_page)), (Cursor(*end) if end is not None else None)

    @staticmethod
    def _output(rendered: "_Rendered") -> Any:
        """The result the engine returns: the text, or where the page holds images the host's
        ``_multimodal`` envelope (agent/tool_dispatch_helpers.py:305-307): the text, then a
        label and the image for each, and the ``text_summary``."""
        if not rendered.images:
            return rendered.text
        content: list = [{"type": "text", "text": rendered.text}]
        for label, part in zip(rendered.labels, rendered.images):
            content.append({"type": "text", "text": label})
            content.append(part)
        return {"_multimodal": True, "content": content, "text_summary": rendered.summary}

    def _longest_slice(self, page_items: list, item: Item, path: Path, text: str, offset: int, page: int,
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
    if not isinstance(piece, Piece):
        return name
    where = f"field {json.dumps(list(piece.path), ensure_ascii=False)} of {name}"
    if is_mark(item, piece.path):
        return f"{where}, an image's held mark"
    return where + (", an image" if is_content_image(item, piece.path) else "")


def target_identity(target: Target) -> str:
    """The identity of everything a handle opens into, as it renders: its header and every
    item after every transformation between the store and the page (the host's row, the
    withholding, the marks, the notes), as JSON; pages are cut from exactly these items, so
    anything that changes what a page shows changes it (the orchestrator's cut of #71, C).
    Sixteen hex characters of its SHA-256. Only that it differs is known when a later page
    compares it, never why, so the refusal says only that (LEARNINGSFÜRPLÄNE A11)."""
    payload = [target.header, [item.as_dict() for item in target.items]]
    text = json.dumps(payload, ensure_ascii=False, sort_keys=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def expand(engine: Any, args: dict, *, messages: Any = None, tool_name: str = "lcm_expand") -> Any:
    """The ``lcm_expand`` tool: one page of what a handle opens into. ``messages`` is the
    live list the host hands the engine tool; the page's size depends on it
    (``host_page_limit``)."""
    session = engine.current_session_id          # the caller's session, read once (#20)
    route = Route.of(engine, tool_name)          # the session's route, read once (§1.4)
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
        raise ExpansionError(CHANGED.format(what="what this handle opens into", again="start again from the handle, "
                                                                                          "without page"))
    token_state = {"v": TOKEN_VERSION, "t": tool_name, "s": store_uuid, "h": resolved.handle,
                   "m": "raw" if raw else "collapsed", "r": identity}
    return serve_page(target, state, token_state, limit, found="what this handle opens into", route=route)


def _holds_images(target: Target) -> bool:
    return any("shown" in item.images.values() for item in target.items)


def serve_page(target: Target, state: Optional[dict], token_state: dict, limit: Any, *, found: str,
               route: Optional["Route"] = None) -> Any:
    """One page of ``target`` from the token's cursor (``state``; None for page 1): the one
    page mechanism every tool that returns pages uses (``lcm_expand``, ``lcm_grep``). A token
    whose cursor lies outside ``target`` is refused as garbled, ``found`` naming what the call
    found. Where ``target`` holds images to deliver, the page counts them by the route's
    model-table rule and keeps within the host's image limits (``Route.room``), all from
    ``route``, the call's one snapshot (``Route.of``)."""
    cursor = Cursor(state["i"], state["f"], state["o"]) if state else Cursor()
    page = state["n"] if state else 1
    if cursor.item >= len(target.items) and not (cursor.item == 0 and not target.items):
        raise ExpansionError(f"page is a garbled next_page token: it points past the end of {found}")
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
            if cursor.offset > length or (cursor.offset and path in target.items[cursor.item].images):
                raise ExpansionError("page is a garbled next_page token: its offset lies outside the field it "
                                     "names")
    image_tokens = image_room = None
    if route is not None:
        image_tokens, image_room = route.estimator.image_count, route.room
    elif _holds_images(target):
        # Images reach a target only through a route (``_images_of``): a caller that built one
        # passes that snapshot here; nothing reads the route a second time.
        raise ExpansionError("the page holds images but the call's route was not handed to the page")
    builder = PageBuilder(target, limit=limit, token_state=token_state, image_tokens=image_tokens,
                          image_room=image_room)
    result, _next = builder.build(cursor, page)
    return result
