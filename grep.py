"""grep (#18 D2): an expansion whose handle the term chose.

It takes a term and a scope: the caller's plugin session as the store holds it (every chunk
under the summaries of its latest effective return, and the fresh tail stored there), or
what stands behind one summary or chunk handle (#34 D6: the union of the chunks under it).
It finds every record of the scope whose searchable text contains the term, and returns the
chunks they lie in, whole, in expansion's form: collapsed by default, raw on request, in the
order of the timeline. At ``GREP_COUNT_ONLY_AT`` or more matching chunks it returns only the
count, unless all of them are asked for (``all``).

**What is searched** is ``records.text`` (``message_content.grep_text``): the strings the
agent's past shows as the message's own words and actions, each by itself: its content as
the host sends it, its tool calls' names and arguments, its readable reasoning. Never a
summary, an image, an encrypted item. A hit means "this shown string contains the term",
exactly: codepoints as stored, case-sensitive, no normalisation, no folding table. Every
record of the scope is read and tested with Python's ``in``, in one snapshot of the store.
There is no index: the column read is the one the hit is decided by, so no second copy of it
exists that could disagree (measured on #76: about 0.1 s for a scope of 39 M characters, 0.6 s
for 230 M). What came after the last compaction is not in the store: the result says what
it searched and when that was stored.

**What a result holds**, never the term itself (the host classifies a result as failed when
its first 500 characters hold the JSON string "error" or "failed", agent/display.py:975-1017
at Hermes cdcd53c2cd; the page token carries the term):

- a header: the scope, the form, what was searched, how many chunks match;
- per matching chunk (full mode), a marker naming the chunk and the summary of the context
  that covers it, and, in the collapsed form, the handles of hit records the form does not
  show (tool results): an address, never the match; then the chunk's items as expansion
  gives them;
- where records of the stored fresh tail hold the term, a note with their handles and the
  time they were stored (the tail is no chunk; whether the context still holds them the
  store learns only at the next compaction), in count-only mode too.

A search that fails is an error, never "no hits": whatever reading the store raises (a
damaged page, a lock held past the busy timeout) is the tool's error.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from . import expansion
from .expansion import ExpansionError, Item, Target
from .handles import CHUNK, DERIVATION
from .message_content import GREP_SEPARATOR
from .record_store import RecordStore

TOOL = "lcm_grep"

# The count at which the chunks are withheld: the manifesto's three ("If three or more chunks
# match, a count that is a value like the others"). Its slot is in #22's table of values; it
# is about how ambiguous a term is, not about the window, and is not scaled.
GREP_COUNT_ONLY_AT = 3

_ARGUMENTS = ("term", "scope", "all", "raw", "page")
_REMOVED = ("query", "limit", "sort", "role", "time_from", "time_to", "mode", "session_scope", "session_id",
            "source", "conversation_id", "content_scope", "externalized_refs")

NOTHING_STORED = "nothing of this session is stored yet: the store is filled at a compaction"
COUNT_ONLY = ("the term lies in this many chunks, too many to return; narrow the term or the scope, or call again "
              "with all=true for every one of them")
TAIL_NOTE = "records of the fresh tail stored at the last compaction hold the term"


@dataclass
class _Found:
    store_uuid: str
    stored_at: Optional[str] = None
    chunks: list = field(default_factory=list)          # the chunks of the scope, in cover order
    tail: list = field(default_factory=list)            # the tail records of the scope (session only)
    members: dict = field(default_factory=dict)
    under: dict = field(default_factory=dict)            # chunk -> the summary of the cover over it
    hits: set = field(default_factory=set)
    cover: Any = None
    session_scope: bool = True


def check_term(term: Any) -> str:
    """The term, or a refusal naming what is wrong with it. NUL and U+001F are refused: the
    stored text writes a NUL as U+001F and separates its strings by it, so a term holding
    either would be answered wrongly. A lone surrogate is not refused: stored text cannot
    hold one, so the true answer, no chunk, is what the scan gives."""
    if not isinstance(term, str):
        raise ExpansionError("term must be a string: the text to search for")
    if not term:
        raise ExpansionError("term is empty: give the text to search for")
    if "\x00" in term:
        raise ExpansionError("term holds a NUL character, which the store cannot search for")
    if GREP_SEPARATOR in term:
        raise ExpansionError("term holds U+001F (unit separator), which separates the searched strings and is "
                             "never searched for")
    return term


def _stored_at(records: RecordStore, compaction: int) -> Optional[str]:
    began = records.compaction_began_at(compaction)
    return datetime.fromtimestamp(began, timezone.utc).isoformat(timespec="seconds") if began is not None else None


def _search(records: RecordStore, session: str, term: str, scope: str) -> _Found:
    """The scope, its records and the hits, read in one snapshot of the store."""
    with records.snapshot():
        found = _Found(store_uuid=str(records.identity().get("store_uuid") or ""))
        cover = records.cover(session)
        if cover is None:
            if scope:
                resolved = records.resolve(scope, session, None)
                raise ExpansionError(expansion.unresolved_message(resolved))
            return found
        found.cover = cover
        found.stored_at = _stored_at(records, cover.compaction)
        for summary in cover.summaries:
            for chunk in cover.reaches.get(summary, []):
                found.under[chunk] = summary
        if scope:
            resolved = records.resolve(scope, session, cover)
            if resolved.status != "ok":
                raise ExpansionError(expansion.unresolved_message(resolved))
            if resolved.kind == DERIVATION:
                found.chunks = records._chunks_of(resolved.handle)
            elif resolved.kind == CHUNK:
                found.chunks = [resolved.handle]
            else:
                raise ExpansionError(f"{resolved.handle} is not a summary's or a chunk's handle; grep's scope is the "
                                     f"session, or what stands behind one summary or chunk")
            found.session_scope = False
        else:
            found.chunks = list(cover.chunks)
            found.tail = list(cover.tail)
        found.members = records.chunk_member_lists(found.chunks)
        scope_records = [r for chunk in found.chunks for r in found.members.get(chunk, [])] + found.tail
        found.hits = records.grep_hits(session, scope_records, term, whole_session=found.session_scope)
    return found


def _target(records: RecordStore, engine: Any, found: _Found, *, scope: str, everything: bool,
            raw: bool) -> tuple[Target, Any]:
    """What the search found, and the call's route snapshot (``expansion.Route.of``, read once,
    where chunks are shown), which the page takes too; None where no chunk is shown."""
    matching = [c for c in found.chunks if any(r in found.hits for r in found.members.get(c, []))]
    tail_hits = [r for r in found.tail if r in found.hits]
    searched: dict = {"chunks": len(found.chunks)}
    if found.session_scope:
        searched["tail_messages"] = len(found.tail)
    searched["stored_at"] = found.stored_at
    header: dict = {"kind": "grep", "scope": scope or "session", "form": "raw" if raw else "collapsed",
                    "searched": searched, "chunks_matching": len(matching)}
    if found.cover is None:
        header["note"] = NOTHING_STORED
        return Target(header, []), None
    items: list = []
    route = None
    count_only = len(matching) >= GREP_COUNT_ONLY_AT and not everything
    if count_only:
        header["count_only"] = True
        header["note"] = COUNT_ONLY
    else:
        # Built outside the snapshot: every row read here is of an insert-only table, and the
        # chunks were chosen in the snapshot (B7).
        route = expansion.Route.of(engine, TOOL) if matching else None
        order = expansion.Order.of(records, found.cover, route) if matching else None
        hidden: dict = {}
        if not raw and matching:
            roles = records.record_roles(sorted(found.hits))
            for chunk in matching:
                hidden[chunk] = [r for r in found.members.get(chunk, []) if r in found.hits and roles.get(r) == "tool"]
        for chunk in matching:
            # The plugin's own fields (the item's ``lcm`` side, PR P): no host dict here.
            said = {"results_holding_term": hidden[chunk]} if hidden.get(chunk) else {}
            items.append(Item({"chunk": chunk, "under": found.under.get(chunk)}, plugin=said))
            items.extend(expansion._records_items(records, order, records.chunk_records(chunk), raw=raw))
    if tail_hits:
        items.append(Item({"tail": TAIL_NOTE, "stored_at": found.stored_at}, plugin={"messages": tail_hits}))
    return Target(header, items), route


def grep(engine: Any, args: dict, *, messages: Any = None) -> Any:
    """The ``lcm_grep`` tool: one page of what the term chose. ``messages`` is the live list
    the host hands the engine tool; the page's size depends on it (``host_page_limits``)."""
    removed = [name for name in _REMOVED if name in args]
    if removed:
        raise ExpansionError("lcm_grep no longer accepts " + ", ".join(removed) + "; it takes term, and optionally "
                             "scope, all, raw and page")
    unknown = [name for name in args if name not in _ARGUMENTS]
    if unknown:
        raise ExpansionError("lcm_grep takes term, scope, all, raw and page; not " + ", ".join(unknown))
    session = engine.current_session_id
    if not session:
        raise ExpansionError("this engine copy is bound to no session of the plugin, so there is nothing to search")
    for name in ("all", "raw"):
        if name in args and not isinstance(args[name], bool):
            raise ExpansionError(f"{name} must be true or false")
    scope_arg = args.get("scope")
    if scope_arg is not None and not isinstance(scope_arg, str):
        raise ExpansionError("scope must be a handle: a summary's (s…) or a chunk's (c…)")
    page_arg = args.get("page")
    state = expansion.decode_token(page_arg) if page_arg is not None else None
    if state is not None:
        if state.get("t") != TOOL:
            raise ExpansionError("page is a token of another tool")
        given = {"term": args.get("term"), "scope": scope_arg, "all": args.get("all"),
                 "raw": args.get("raw")}
        held = {"term": state["q"], "scope": state["p"] or None, "all": state["a"], "raw": state["m"] == "raw"}
        for name, value in given.items():
            # raw=false and all=false are the schema's defaults, which some callers send with every call.
            if value is None or (name in ("all", "raw") and value is False and held[name] is False):
                continue
            if (value.strip() if name == "scope" else value) != held[name]:
                raise ExpansionError(f"page is a token of another search: its {name} differs from the one given; "
                                     f"call again without page to start this search")
        term, scope, everything, raw = state["q"], state["p"], state["a"], state["m"] == "raw"
    else:
        if "term" not in args:
            raise ExpansionError("term is required: the text to search for")
        term, scope = args["term"], (scope_arg or "").strip()
        everything, raw = bool(args.get("all", False)), bool(args.get("raw", False))
    check_term(term)
    limit = expansion.host_page_limits(engine, TOOL, messages)
    records: RecordStore = engine._records
    found = _search(records, session, term, scope)
    target, route = _target(records, engine, found, scope=scope, everything=everything, raw=raw)
    identity = expansion.target_identity(target)
    if state is not None:
        if state["s"] != found.store_uuid:
            raise ExpansionError("page is a token of another store: the store it was issued by is not this one")
        if state["r"] != identity:
            raise ExpansionError(expansion.CHANGED.format(what="what this search returns",
                                                          again="call again without page to start it"))
    token_state = {"v": expansion.TOKEN_VERSION, "t": TOOL, "s": found.store_uuid, "q": term, "p": scope,
                   "a": everything, "m": "raw" if raw else "collapsed", "r": identity}
    # The one page mechanism of expansion (B10: the unification of grep._serve and expand's tail).
    return expansion.serve_page(target, state, token_state, limit, found="what this search found", route=route)
