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
exactly: codepoints as stored, case-sensitive, no normalisation, no folding table. The
trigram index (``grep_index``) only chooses candidates, the records holding every trigram of
the term (a sample of at most ``TRIGRAM_LIMIT`` of them); Python's ``in`` on each candidate
decides. A term shorter than three characters has no trigram: every record of the scope is
a candidate. What came after the last compaction is not in the store: the result says what
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

A search that fails is an error, never "no hits". The index is checked on every call for
records of the scope it does not hold, and for three of them (the first, middle and last)
that it must return for a trigram of their own text (an index whose structure was lost
raises nothing and answers nothing); SQLite's corruption errors on the query are caught.
Any of these records an event, rebuilds the index under the store's write lock and retries
the search once. Damage to other records' postings that raises nothing is found by the
doctor's deep check, which rebuilds too.
"""

from __future__ import annotations

import json
import sqlite3
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

# At most this many distinct trigrams of the term go to the index; any subset of a term's
# trigrams is held by every record that holds the term, so the number moves only how many
# candidates are verified, never the result.
TRIGRAM_LIMIT = 64

_ARGUMENTS = ("term", "scope", "all", "raw", "page")
_REMOVED = ("query", "limit", "sort", "role", "time_from", "time_to", "mode", "session_scope", "session_id",
            "source", "conversation_id", "content_scope", "externalized_refs")
_CORRUPTION = ("malformed", "disk image", "not a database", "corrupt", "checksum mismatch")

NOTHING_STORED = ("nothing of this session is stored yet: the store is filled at a compaction, and until the first "
                  "one everything of the session is in your context")
COUNT_ONLY = ("the term lies in this many chunks, too many to return; narrow the term or the scope, or call again "
              "with all=true for every one of them")
TAIL_NOTE = ("records of the fresh tail stored at the last compaction hold the term; your context held them "
             "verbatim then")


class _IndexDamaged(Exception):
    def __init__(self, found: str, detail: Any):
        super().__init__(found)
        self.found = found
        self.detail = detail


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
    """The term, or a refusal naming what is wrong with it."""
    if not isinstance(term, str):
        raise ExpansionError("term must be a string: the text to search for")
    if not term:
        raise ExpansionError("term is empty: give the text to search for")
    if "\x00" in term:
        raise ExpansionError("term holds a NUL character, which the store cannot search for")
    if GREP_SEPARATOR in term:
        raise ExpansionError("term holds U+001F (unit separator), which separates the searched strings and is "
                             "never searched for")
    try:
        term.encode("utf-8")
    except UnicodeEncodeError:
        raise ExpansionError("term holds a lone surrogate, which is not text and cannot be searched for") from None
    return term


def trigram_match(term: str) -> Optional[str]:
    """The MATCH expression for grep's index: the AND of at most ``TRIGRAM_LIMIT`` distinct
    trigrams of the term, sampled evenly in the order first seen, each quoted as a string
    (every character literal); None for a term shorter than three characters."""
    grams = list(dict.fromkeys(term[i:i + 3] for i in range(len(term) - 2)))
    if not grams:
        return None
    if len(grams) > TRIGRAM_LIMIT:
        step = len(grams) / TRIGRAM_LIMIT
        grams = [grams[int(k * step)] for k in range(TRIGRAM_LIMIT)]
    return " AND ".join('"' + gram.replace('"', '""') + '"' for gram in grams)


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
        unindexed = records.grep_unindexed(scope_records)
        if unindexed:
            raise _IndexDamaged("coverage", {"records_not_indexed": unindexed})
        try:
            unanswered = records.grep_unanswered(scope_records)
            if unanswered:
                raise _IndexDamaged("unanswered", {"records_the_index_does_not_return": unanswered})
            found.hits = records.grep_hits(session, scope_records, term, trigram_match(term))
        except sqlite3.DatabaseError as exc:
            if any(signature in str(exc).lower() for signature in _CORRUPTION):
                raise _IndexDamaged("query_error", f"{type(exc).__name__}: {exc}") from exc
            raise
    return found


def _search_with_repair(records: RecordStore, session: str, term: str, scope: str) -> _Found:
    """``_search``; where the index is damaged, an event, a rebuild under the write lock, and
    one more search. Damage that shows again, or a rebuild that fails, is an error."""
    try:
        return _search(records, session, term, scope)
    except _IndexDamaged as damage:
        records.event("grep_index_damaged", session=session, detail={"found": damage.found, "detail": damage.detail})
        try:
            records.rebuild_grep_index(session=session, found=damage.found)
        except Exception as exc:
            raise ExpansionError(f"grep's index is damaged ({damage.found}) and could not be rebuilt "
                                 f"({type(exc).__name__}: {exc}); the store events grep_index_damaged and "
                                 f"grep_index_rebuild_failed record it. Nothing was searched.") from None
    try:
        return _search(records, session, term, scope)
    except _IndexDamaged as again:
        records.event("grep_index_damaged", session=session,
                      detail={"found": again.found, "detail": again.detail, "after_rebuild": True})
        raise ExpansionError(f"grep's index is still damaged ({again.found}) after it was rebuilt; the store event "
                             f"grep_index_damaged records it. Nothing was searched.") from None


def _target(records: RecordStore, engine: Any, found: _Found, *, scope: str, everything: bool, raw: bool) -> Target:
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
        return Target(header, [])
    items: list = []
    count_only = len(matching) >= GREP_COUNT_ONLY_AT and not everything
    if count_only:
        header["count_only"] = True
        header["note"] = COUNT_ONLY
    else:
        # Built outside the snapshot: every row read here is of an insert-only table, and the
        # chunks were chosen in the snapshot (B7).
        route = expansion.Route.of(engine) if matching else None
        order = expansion.Order.of(records, found.cover, route) if matching else None
        hidden: dict = {}
        if not raw and matching:
            roles = records.record_roles(sorted(found.hits))
            for chunk in matching:
                hidden[chunk] = [r for r in found.members.get(chunk, []) if r in found.hits and roles.get(r) == "tool"]
        for chunk in matching:
            fields = {"results_holding_term": hidden[chunk]} if hidden.get(chunk) else {}
            items.append(Item({"chunk": chunk, "under": found.under.get(chunk)}, fields))
            items.extend(expansion._records_items(records, order, records.chunk_records(chunk), raw=raw))
    if tail_hits:
        items.append(Item({"tail": TAIL_NOTE, "stored_at": found.stored_at}, {"messages": tail_hits}))
    return Target(header, items)


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
    found = _search_with_repair(records, session, term, scope)
    target = _target(records, engine, found, scope=scope, everything=everything, raw=raw)
    identity = expansion.target_identity(target)
    if state is not None:
        if state["s"] != found.store_uuid:
            raise ExpansionError("page is a token of another store: the store it was issued by is not this one")
        if state["r"] != identity:
            raise ExpansionError("what this search finds changed since page 1 (a compaction since, a record the host "
                                 "rewrote, or code that changed); call again without page to start it")
    token_state = {"v": expansion.TOKEN_VERSION, "t": TOOL, "s": found.store_uuid, "q": term, "p": scope,
                   "a": everything, "m": "raw" if raw else "collapsed", "r": identity}
    return _serve(target, state, token_state, limit)


def _serve(target: Target, state: Optional[dict], token_state: dict, limit: Any) -> Any:
    """One page of ``target`` from the token's cursor, by expansion's one page mechanism.
    The same steps as the end of ``expansion.expand``; they become one shared function when
    image delivery has merged (B10)."""
    cursor = expansion.Cursor(state["i"], state["f"], state["o"]) if state else expansion.Cursor()
    page = state["n"] if state else 1
    if cursor.item >= len(target.items) and not (cursor.item == 0 and not target.items):
        raise ExpansionError("page is a garbled next_page token: it points past the end of what this search found")
    if cursor.field >= 0:
        fields = expansion.fields_of(target.items[cursor.item])
        if cursor.field == len(fields) and cursor.offset == 0:
            cursor = expansion.Cursor(cursor.item + 1, -1, 0)
        elif cursor.field >= len(fields):
            raise ExpansionError("page is a garbled next_page token: it names a field this item does not have")
        else:
            path, value = fields[cursor.field]
            length = len(value) if isinstance(value, str) else len(json.dumps(value, ensure_ascii=False))
            if cursor.offset > length or (cursor.offset and expansion.is_mark(path, value)):
                raise ExpansionError("page is a garbled next_page token: its offset lies outside the field it names")
    builder = expansion.PageBuilder(target, limit=limit, token_state=token_state)
    result, _next = builder.build(cursor, page)
    return result
