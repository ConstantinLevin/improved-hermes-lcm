"""The plugin's record of what happened at each compaction (#29, W1-W3; #1, #2, #3).

It is the only store: the summariser reads from it, the return is emitted from it,
and the tools read views over it. Every table here is insert-only. A compaction is
written in short transactions, none held open across a summariser call:

1. the compaction, its inputs, a record for every input entry the store does not
   hold yet (the fresh tail included), their tool calls, and every chunk with its
   members, all before the first summariser call;
2. each summary, as a derivation of its chunk, when it arrives;
3. what the plugin returned.

The host confirms a compaction with ``on_session_start(boundary_reason="compression")``
(the record line) or rejects it with ``record_rejected_compaction()``; the host rows
the returned entries were committed as are written as bindings.

``raw`` is the host's dict as JSON: every key the host handed over, in its order,
``ensure_ascii`` off, no fallback for a value JSON cannot hold (an error), and without
the plugin's own in-memory key only.
"""

from __future__ import annotations

import contextlib
import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from .db_bootstrap import open_store
from .handles import CHUNK, DERIVATION, MESSAGE, TOOL_CALL, new_handle
from .message_content import base64_like_strings, describe_image_part, image_parts, index_text
from .tokens import count_message_tokens

logger = logging.getLogger(__name__)

# The plugin's in-memory key on the dicts it returns: "<compaction>:<position>". It
# bridges a returned entry to the host row it is committed as; it is never identity
# and never written into ``raw``.
RET_KEY = "_lcm_ret"

_HANDLE_DRAWS = 8


def raw_json(message: dict) -> str:
    """The host's dict as JSON, verbatim; a value JSON cannot hold raises."""
    return json.dumps(
        {key: value for key, value in message.items() if key != RET_KEY},
        ensure_ascii=False,
        allow_nan=False,
    )


def _warn_media(handle: str, message: dict) -> None:
    """A message carrying images or base64 is stored as it came, and a warning is
    logged (#3, #35). Images are found by structure; base64 inside a string only by
    its shape, in a separate line that says so."""
    try:
        images = image_parts(message.get("content"))
        if images:
            logger.warning(
                "LCM stored a %s message with %d image part(s) as it came, in record %s: %s",
                message.get("role") or "?",
                len(images),
                handle,
                "; ".join(describe_image_part(part) for part in images),
            )
        suspects = base64_like_strings(message)
        if suspects:
            logger.warning(
                "LCM heuristic (by the text's shape, not a detection): the %s message in record %s "
                "may carry base64 inside its text, stored as it came: %s",
                message.get("role") or "?",
                handle,
                "; ".join(suspects),
            )
    except Exception:
        logger.warning("LCM could not check record %s for media", handle, exc_info=True)


def parse_ret_key(value: Any) -> Optional[tuple[int, int]]:
    if not isinstance(value, str) or ":" not in value:
        return None
    left, _, right = value.partition(":")
    try:
        return int(left), int(right)
    except ValueError:
        return None


@dataclass
class InputEntry:
    """One entry of the list the host handed to ``compress()``, classified (W3, W4)."""

    position: int
    host_row_id: Optional[int]
    # "system": the host's system row, not recorded (ruling 12)
    # "bound": a returned record of the effective compaction, known by binding or key
    # "bound_summary": a returned summary of it; the mechanism's layer, no record
    # "reused": a row an unconfirmed attempt already recorded, known by its _row_id
    # "host_insertion": the host's insertion, known by binding or by its place
    # "transcript": unbound and after the last bound entry
    # "revision": a known row the host rewrote or merged; a new record
    klass: str
    record: Optional[str] = None
    message: Optional[dict] = None
    # For a new record: its predecessor, ("entry", position of an earlier entry of this
    # list), ("record", handle) or None; for a revision: what it revises, each
    # ("record", handle) or ("return", compaction, position).
    pred: Optional[tuple] = None
    sources: list = field(default_factory=list)


class RecordStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self._lock = threading.RLock()
        self._pending_events: list[tuple] = []
        self._conn: Optional[sqlite3.Connection] = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,
            check_same_thread=False,
            isolation_level=None,
        )
        try:
            open_store(self._conn, self.db_path)
        except BaseException:
            self._conn.close()
            self._conn = None
            raise

    def close(self) -> None:
        conn = self._conn
        if conn is not None:
            conn.close()
            self._conn = None

    @contextlib.contextmanager
    def _tx(self):
        """One short write transaction. It never stays open: when the body or the
        COMMIT fails (in rollback-journal mode a COMMIT can fail on the busy timeout
        while a reader holds its shared lock), the transaction is rolled back before
        the error is raised, so that no lock of the shadow ever blocks a live write."""
        with self._lock:
            conn = self._conn
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
                conn.execute("COMMIT")
            except BaseException:
                self._rollback(conn)
                raise
            self._flush_events()

    def _rollback(self, conn: sqlite3.Connection) -> None:
        if not conn.in_transaction:
            return
        try:
            conn.execute("ROLLBACK")
        except Exception:
            logger.error("LCM could not roll back a shadow transaction in %s", self.db_path, exc_info=True)

    def _q(self, sql: str, args: Sequence[Any] = ()) -> list:
        with self._lock:
            return self._conn.execute(sql, args).fetchall()

    # --- Events -------------------------------------------------------------------

    def event(
        self,
        kind: str,
        *,
        session: Optional[str] = None,
        compaction: Optional[int] = None,
        detail: Any = None,
    ) -> None:
        """Record something the plugin could not do, or a fact about its health.

        It never raises. An event that cannot be written now (the store is locked by
        the same cause as the failure it reports) is logged at ERROR, kept in this
        process, and written with the next shadow transaction that commits.
        """
        text = detail if isinstance(detail, str) or detail is None else json.dumps(detail, default=repr)
        logger.warning("LCM store event %s (session=%s compaction=%s): %s", kind, session, compaction, text)
        with self._lock:
            self._pending_events.append((time.time(), kind, session, compaction, text))
            self._flush_events()

    def _flush_events(self) -> None:
        """Write the events not written yet, in their own short transaction."""
        with self._lock:
            if not self._pending_events or self._conn is None:
                return
            conn = self._conn
            try:
                conn.execute("BEGIN IMMEDIATE")
                try:
                    conn.executemany(
                        "INSERT INTO store_events(at, kind, session, compaction, detail) VALUES (?, ?, ?, ?, ?)",
                        self._pending_events,
                    )
                    conn.execute("COMMIT")
                except BaseException:
                    self._rollback(conn)
                    raise
            except Exception:
                logger.error(
                    "LCM could not record %d store event(s) in %s yet: %s",
                    len(self._pending_events),
                    self.db_path,
                    ", ".join(event[1] for event in self._pending_events),
                    exc_info=True,
                )
                return
            self._pending_events = []

    # --- Reading ------------------------------------------------------------------

    def effective_compaction(self, session: str) -> Optional[int]:
        """The session's latest compaction that took effect and was not rejected: the
        host confirmed it, or its return was found adopted without a confirmation."""
        rows = self._q(
            "SELECT compaction_id FROM effective_compactions WHERE session = ? "
            "ORDER BY compaction_id DESC LIMIT 1",
            (session,),
        )
        return int(rows[0][0]) if rows else None

    def effective_count(self, session: str) -> int:
        """How many of the session's compactions took effect."""
        rows = self._q("SELECT COUNT(*) FROM effective_compactions WHERE session = ?", (session,))
        return int(rows[0][0]) if rows else 0

    def derivations(self, handles: Iterable[str]) -> dict[str, tuple[int, str, Optional[str]]]:
        """handle -> (derivation id, text, expand hint)."""
        wanted = [h for h in set(handles) if h]
        found: dict[str, tuple[int, str, Optional[str]]] = {}
        for start in range(0, len(wanted), 500):
            chunk = wanted[start:start + 500]
            rows = self._q(
                "SELECT handle, derivation_id, text, expand_hint FROM derivations "
                f"WHERE handle IN ({','.join('?' * len(chunk))})",
                chunk,
            )
            found.update({str(h): (int(i), str(t), e) for h, i, t, e in rows})
        return found

    def is_settled(self, compaction: int) -> bool:
        """Confirmed, adopted or rejected: the host's handling of it is known."""
        return bool(self._q(
            "SELECT 1 WHERE EXISTS (SELECT 1 FROM confirmations WHERE compaction = ?) "
            "OR EXISTS (SELECT 1 FROM adoptions WHERE compaction = ?) "
            "OR EXISTS (SELECT 1 FROM rejections WHERE compaction = ?)",
            (compaction, compaction, compaction),
        ))

    def return_entries(self, compaction: int) -> dict[int, tuple[str, Optional[str], Optional[str], Optional[str]]]:
        """position -> (kind, record, derivation, raw of a returned summary)."""
        return {
            int(pos): (str(kind), record, derivation, raw)
            for pos, kind, record, derivation, raw in self._q(
                "SELECT position, kind, record, derivation, raw FROM compaction_returns WHERE compaction = ?",
                (compaction,),
            )
        }

    def record_facts(self, handles: Iterable[str]) -> dict[str, tuple[Optional[str], str, str, int]]:
        """handle -> (predecessor, raw, kind, record id); the id is also the order written."""
        wanted = [h for h in set(handles) if h]
        facts: dict[str, tuple[Optional[str], str, str, int]] = {}
        for start in range(0, len(wanted), 500):
            chunk = wanted[start:start + 500]
            rows = self._q(
                "SELECT handle, predecessor, raw, kind, rowid FROM records "
                f"WHERE handle IN ({','.join('?' * len(chunk))})",
                chunk,
            )
            facts.update({str(h): (p, str(r), str(k), int(o)) for h, p, r, k, o in rows})
        return facts

    def beside(self, handles: Iterable[str]) -> dict[str, bool]:
        """Whether each record stands beside the chain (F8): a host insertion, or a
        revision none of whose record sources is on the chain (a rewritten summary, a
        rewritten host insertion)."""
        result: dict[str, bool] = {}

        def visit(handle: str, depth: int = 0) -> bool:
            if handle in result:
                return result[handle]
            kind_rows = self._q("SELECT kind FROM records WHERE handle = ?", (handle,))
            kind = str(kind_rows[0][0]) if kind_rows else "transcript"
            if kind == "host_insertion":
                value = True
            elif kind == "revision" and depth < 32:
                sources = [s for (s,) in self._q(
                    "SELECT source_record FROM revision_sources WHERE revision = ? AND source_record IS NOT NULL",
                    (handle,),
                )]
                value = all(visit(str(s), depth + 1) for s in sources)
            else:
                value = False
            result[handle] = value
            return value

        for handle in set(handles):
            if handle:
                visit(handle)
        return result

    def chain_end(self, compaction: Optional[int]) -> Optional[str]:
        """The record of the last input entry of a compaction that is on the chain."""
        if compaction is None:
            return None
        rows = [str(r) for (r,) in self._q(
            "SELECT record FROM compaction_inputs WHERE compaction = ? AND record IS NOT NULL "
            "ORDER BY position DESC",
            (compaction,),
        )]
        side = self.beside(rows)
        return next((r for r in rows if not side.get(r)), None)

    def bound_rows(self, compaction: int) -> dict[int, int]:
        """host_row_id -> returned position, for the returned entries."""
        return {
            int(row_id): int(pos)
            for pos, row_id in self._q(
                "SELECT position, host_row_id FROM bindings WHERE compaction = ? AND kind = 'return'",
                (compaction,),
            )
        }

    def bound_insertions(self, compaction: int) -> set[int]:
        """host_row_ids the host inserted into the committed list of a compaction."""
        return {
            int(row_id)
            for (row_id,) in self._q(
                "SELECT host_row_id FROM bindings WHERE compaction = ? AND kind = 'host_insertion'",
                (compaction,),
            )
        }

    def unconfirmed_inputs(self, session: str, after: Optional[int]) -> dict[int, str]:
        """host_row_id -> record, from inputs of attempts after the effective one that
        the host neither confirmed nor rejected. Ids hold until a commit, so an entry
        with the same _row_id is the same host row (W2 step 8)."""
        rows = self._q(
            "SELECT i.host_row_id, i.record FROM compaction_inputs i JOIN compactions c "
            "ON c.compaction_id = i.compaction WHERE c.session = ? AND c.compaction_id > ? "
            "AND i.host_row_id IS NOT NULL AND i.record IS NOT NULL "
            "AND NOT EXISTS (SELECT 1 FROM confirmations f WHERE f.compaction = c.compaction_id) "
            "AND NOT EXISTS (SELECT 1 FROM adoptions a WHERE a.compaction = c.compaction_id) "
            "AND NOT EXISTS (SELECT 1 FROM rejections r WHERE r.compaction = c.compaction_id) "
            "ORDER BY c.compaction_id ASC",
            (session, after or 0),
        )
        return {int(row_id): str(record) for row_id, record in rows}

    # --- Writing ------------------------------------------------------------------

    @staticmethod
    def _insert_with_handle(conn: sqlite3.Connection, kind: str, sql: str, args_after_handle: tuple) -> str:
        for _ in range(_HANDLE_DRAWS):
            handle = new_handle(kind)
            try:
                conn.execute(sql, (handle,) + args_after_handle)
                return handle
            except sqlite3.IntegrityError as exc:
                if "UNIQUE constraint failed" in str(exc) and ".handle" in str(exc):
                    continue
                raise
        raise RuntimeError(f"LCM drew {_HANDLE_DRAWS} handles of kind {kind!r} that were all taken")

    def begin_compaction(
        self,
        *,
        session: str,
        kind: str,
        host_session_before: Optional[str],
        attempt_generation: Optional[int],
        entries: Sequence[InputEntry],
        chunks: Sequence[Sequence[int]] = (),
    ) -> tuple[int, dict[int, str], list[str]]:
        """Transaction 1: the compaction, its inputs, the new records, their tool calls
        and the chunks.

        Returns the compaction id, the record of every input position that has one,
        and the chunk handles in the order given. Each new record takes the
        predecessor its entry names (the classification decides it, #29 W3/W4); a
        revision also gets its ``revision_sources``. Tool calls are written for new
        transcript and host insertions; a revision keeps its original's calls. A chunk
        is given as the input positions of its members, in order; every member must
        have a record.
        """
        records: dict[int, str] = {}
        chunk_handles: list[str] = []
        with self._tx() as conn:
            cid = conn.execute(
                "INSERT INTO compactions(session, kind, host_session_before, attempt_generation, began_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (session, kind, host_session_before, attempt_generation, time.time()),
            ).lastrowid
            written: list[tuple[str, dict]] = []
            for entry in entries:
                if entry.klass in ("bound", "reused"):
                    if entry.record is not None:
                        records[entry.position] = entry.record
                elif entry.klass in ("transcript", "host_insertion", "revision"):
                    message = entry.message or {}
                    predecessor = None
                    if entry.pred is not None and entry.pred[0] == "entry":
                        predecessor = records[entry.pred[1]]
                    elif entry.pred is not None:
                        predecessor = entry.pred[1]
                    handle = self._insert_with_handle(
                        conn,
                        MESSAGE,
                        "INSERT INTO records(handle, session, predecessor, compaction, kind, raw, "
                        "role, tool_call_id, text, est_tokens) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            session,
                            predecessor,
                            cid,
                            entry.klass,
                            raw_json(message),
                            message.get("role"),
                            message.get("tool_call_id"),
                            index_text(message.get("content")),
                            count_message_tokens(message),
                        ),
                    )
                    records[entry.position] = handle
                    if entry.klass == "revision":
                        for ordinal, source in enumerate(entry.sources):
                            if source[0] == "record":
                                conn.execute(
                                    "INSERT INTO revision_sources(revision, ordinal, source_record) VALUES (?, ?, ?)",
                                    (handle, ordinal, source[1]),
                                )
                            else:
                                conn.execute(
                                    "INSERT INTO revision_sources(revision, ordinal, source_compaction, "
                                    "source_position) VALUES (?, ?, ?, ?)",
                                    (handle, ordinal, source[1], source[2]),
                                )
                        # Calls the sources already hold keep their handles; only the
                        # revision's new calls are written, as for a new record.
                        source_records = [s[1] for s in entry.sources if s[0] == "record"]
                        known = {
                            str(call_id)
                            for (call_id,) in conn.execute(
                                "SELECT tool_call_id FROM tool_calls WHERE tool_call_id IS NOT NULL AND record IN "
                                f"({','.join('?' * len(source_records))})",
                                source_records,
                            ).fetchall()
                        } if source_records else set()
                        written.append((handle, message, known))
                    else:
                        written.append((handle, message, set()))
                conn.execute(
                    "INSERT INTO compaction_inputs(compaction, position, host_row_id, record) VALUES (?, ?, ?, ?)",
                    (cid, entry.position, entry.host_row_id, records.get(entry.position)),
                )
            self._write_tool_calls(conn, session, cid, written)
            for positions in chunks:
                members = [records[position] for position in positions]
                handle = self._insert_with_handle(
                    conn,
                    CHUNK,
                    "INSERT INTO chunks(handle, session, compaction) VALUES (?, ?, ?)",
                    (session, cid),
                )
                conn.executemany(
                    "INSERT INTO chunk_members(chunk, ordinal, record) VALUES (?, ?, ?)",
                    [(handle, ordinal, record) for ordinal, record in enumerate(members)],
                )
                chunk_handles.append(handle)
        for handle, message, _known in written:
            _warn_media(handle, message)
        return int(cid), records, chunk_handles

    def _write_tool_calls(
        self,
        conn: sqlite3.Connection,
        session: str,
        cid: int,
        written: list[tuple[str, dict, set]],
    ) -> None:
        """One row per tool call, pointing into its assistant record by position; a
        result recorded in the same compaction is its ``result_record``, one recorded
        by a later compaction is linked through ``tool_results``. Each entry carries
        the call ids that already have a handle (a revision's sources) and are skipped."""
        results_by_call_id: dict[str, list[str]] = {}
        for handle, message, _known in written:
            if message.get("role") == "tool" and message.get("tool_call_id"):
                results_by_call_id.setdefault(str(message["tool_call_id"]), []).append(handle)
        claimed: set[str] = set()
        for handle, message, known in written:
            calls = message.get("tool_calls") or []
            if message.get("role") != "assistant" or not isinstance(calls, list):
                continue
            for index, call in enumerate(calls):
                call_id = str(call.get("id") or "") if isinstance(call, dict) else ""
                if call_id and call_id in known:
                    continue
                result = None
                for candidate in results_by_call_id.get(call_id, []):
                    if candidate not in claimed:
                        result = candidate
                        claimed.add(candidate)
                        break
                self._insert_with_handle(
                    conn,
                    TOOL_CALL,
                    "INSERT INTO tool_calls(handle, record, position, tool_call_id, result_record) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (handle, index, call_id or None, result),
                )
        for call_id, handles in results_by_call_id.items():
            for result in handles:
                if result in claimed:
                    continue
                row = conn.execute(
                    "SELECT t.handle FROM tool_calls t JOIN records r ON r.handle = t.record "
                    "WHERE r.session = ? AND t.tool_call_id = ? AND t.result_record IS NULL "
                    "AND NOT EXISTS (SELECT 1 FROM tool_results x WHERE x.tool_call = t.handle) "
                    "ORDER BY r.rowid DESC, t.position DESC LIMIT 1",
                    (session, call_id),
                ).fetchone()
                if row is not None:
                    conn.execute(
                        "INSERT INTO tool_results(tool_call, result_record, compaction) VALUES (?, ?, ?)",
                        (row[0], result, cid),
                    )

    def write_derivation(
        self,
        *,
        compaction: int,
        chunk: str,
        text: str,
        model: Optional[str],
        provider: Optional[str],
        level: Optional[int],
        budget: Optional[int],
        est_tokens: Optional[int],
        expand_hint: Optional[str] = None,
    ) -> str:
        with self._tx() as conn:
            handle = self._insert_with_handle(
                conn,
                DERIVATION,
                "INSERT INTO derivations(handle, kind, text, compaction, model, provider, effort, prompt, "
                "budget, finish_reason, level, est_tokens, expand_hint, created_at) "
                "VALUES (?, 'summary', ?, ?, ?, ?, NULL, NULL, ?, NULL, ?, ?, ?, ?)",
                (text, compaction, model or None, provider or None, budget, level, est_tokens,
                 expand_hint, time.time()),
            )
            conn.execute(
                "INSERT INTO derivation_sources(derivation, ordinal, chunk, source_derivation) VALUES (?, 0, ?, NULL)",
                (handle, chunk),
            )
        return handle

    def write_returns(
        self,
        compaction: int,
        entries: Iterable[tuple[int, str, Optional[str], Optional[str], Optional[str]]],
    ) -> None:
        """(position, kind, record, derivation, raw): a returned summary names the
        derivation it was emitted from and keeps its dict as returned, verbatim, since
        it is what the agent's context held."""
        with self._tx() as conn:
            conn.executemany(
                "INSERT INTO compaction_returns(compaction, position, kind, record, derivation, raw) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                [(compaction, pos, kind, record, derivation, raw) for pos, kind, record, derivation, raw in entries],
            )

    def confirm(
        self,
        compaction: int,
        *,
        host_session_before: Optional[str],
        host_session_after: str,
        at: Optional[float] = None,
    ) -> None:
        """The record line: the host's identifiers before and after, at the time the
        host's signal arrived (``at``, when it is written once the next list has named
        the compaction the signal was for)."""
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO confirmations(compaction, host_session_before, host_session_after, at) VALUES (?, ?, ?, ?)",
                (compaction, host_session_before or None, host_session_after, at if at is not None else time.time()),
            )

    def reject(self, compaction: int, *, how: str) -> None:
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO rejections(compaction, how, at) VALUES (?, ?, ?)",
                (compaction, how, time.time()),
            )

    def adopt(self, compaction: int, *, evidence: str) -> None:
        """The return took effect without a confirmation (a host without a session
        database uses the list as returned). Its own kind, never a confirmation."""
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO adoptions(compaction, evidence, at) VALUES (?, ?, ?)",
                (compaction, evidence, time.time()),
            )

    def bind(
        self,
        compaction: int,
        returns: Iterable[tuple[int, int, Optional[int]]] = (),
        insertions: Iterable[tuple[int, Optional[int]]] = (),
    ) -> list[int]:
        """Write bindings not written yet: returned entries as (position, host_row_id,
        committed_index), host insertions as (host_row_id, committed_index). A position
        or a row id already bound is left as it is. Returns the positions written."""
        written: list[int] = []
        with self._tx() as conn:
            for position, row_id, index in returns:
                exists = conn.execute(
                    "SELECT 1 FROM bindings WHERE compaction = ? AND (position = ? OR host_row_id = ?)",
                    (compaction, position, int(row_id)),
                ).fetchone()
                if exists:
                    continue
                conn.execute(
                    "INSERT INTO bindings(compaction, host_row_id, kind, position, committed_index, at) "
                    "VALUES (?, ?, 'return', ?, ?, ?)",
                    (compaction, int(row_id), position, index, time.time()),
                )
                written.append(position)
            for row_id, index in insertions:
                exists = conn.execute(
                    "SELECT 1 FROM bindings WHERE compaction = ? AND host_row_id = ?",
                    (compaction, int(row_id)),
                ).fetchone()
                if exists:
                    continue
                conn.execute(
                    "INSERT INTO bindings(compaction, host_row_id, kind, position, committed_index, at) "
                    "VALUES (?, ?, 'host_insertion', NULL, ?, ?)",
                    (compaction, int(row_id), index, time.time()),
                )
        return written
