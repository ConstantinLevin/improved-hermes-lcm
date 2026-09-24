"""The plugin's record of what happened at each compaction (#29, W1-W3; #1, #2, #3).

Every table here is insert-only. A compaction is written in short transactions, none
held open across a summariser call:

1. the compaction, its inputs, a record for every input entry the store does not
   hold yet (the fresh tail included), and their tool calls;
2. each chunk with its members, before its summariser call;
3. each summary, as a derivation of its chunk, when it arrives;
4. what the plugin returned.

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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from .db_bootstrap import open_store
from .handles import CHUNK, DERIVATION, MESSAGE, TOOL_CALL, new_handle
from .message_content import text_content_for_pattern_matching

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
    """One entry of the list the host handed to ``compress()``, classified (W3)."""

    position: int
    host_row_id: Optional[int]
    # "system": the host's system row, not recorded (ruling 12)
    # "bound": the return of the previous effective compaction, known by binding or key
    # "reused": a row an unconfirmed attempt already recorded, known by its _row_id
    # "host_insertion": unbound and before the last bound entry
    # "transcript": unbound and after the last bound entry
    klass: str
    record: Optional[str] = None
    message: Optional[dict] = None


class RecordStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self._lock = threading.RLock()
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
        with self._lock:
            conn = self._conn
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            conn.execute("COMMIT")

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

        It never raises: a failed event write is logged at ERROR instead.
        """
        text = detail if isinstance(detail, str) or detail is None else json.dumps(detail, default=repr)
        logger.warning("LCM store event %s (session=%s compaction=%s): %s", kind, session, compaction, text)
        try:
            with self._tx() as conn:
                conn.execute(
                    "INSERT INTO store_events(at, kind, session, compaction, detail) VALUES (?, ?, ?, ?, ?)",
                    (time.time(), kind, session, compaction, text),
                )
        except Exception:
            logger.error("LCM could not record the store event %s in %s", kind, self.db_path, exc_info=True)

    # --- Reading ------------------------------------------------------------------

    def effective_compaction(self, session: str) -> Optional[int]:
        """The session's latest compaction the host confirmed and did not reject."""
        rows = self._q(
            "SELECT c.compaction_id FROM compactions c JOIN confirmations f ON f.compaction = c.compaction_id "
            "WHERE c.session = ? AND NOT EXISTS (SELECT 1 FROM rejections r WHERE r.compaction = c.compaction_id) "
            "ORDER BY c.compaction_id DESC LIMIT 1",
            (session,),
        )
        return int(rows[0][0]) if rows else None

    def compaction_session(self, compaction: int) -> Optional[str]:
        rows = self._q("SELECT session FROM compactions WHERE compaction_id = ?", (compaction,))
        return str(rows[0][0]) if rows else None

    def return_entries(self, compaction: int) -> dict[int, tuple[str, Optional[str], Optional[str]]]:
        return {
            int(pos): (str(kind), record, derivation)
            for pos, kind, record, derivation in self._q(
                "SELECT position, kind, record, derivation FROM compaction_returns WHERE compaction = ?",
                (compaction,),
            )
        }

    def bound_rows(self, compaction: int) -> dict[int, int]:
        """host_row_id -> returned position."""
        return {
            int(row_id): int(pos)
            for pos, row_id in self._q(
                "SELECT position, host_row_id FROM bindings WHERE compaction = ?",
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
            "AND NOT EXISTS (SELECT 1 FROM rejections r WHERE r.compaction = c.compaction_id) "
            "ORDER BY c.compaction_id ASC",
            (session, after or 0),
        )
        return {int(row_id): str(record) for row_id, record in rows}

    def head(self, compaction: Optional[int]) -> Optional[str]:
        """The last record entry in a compaction's return."""
        if compaction is None:
            return None
        rows = self._q(
            "SELECT record FROM compaction_returns WHERE compaction = ? AND kind = 'record' "
            "AND record IS NOT NULL ORDER BY position DESC LIMIT 1",
            (compaction,),
        )
        return str(rows[0][0]) if rows else None

    def latest_pending(self, session: str) -> Optional[int]:
        """The session's latest compaction with a return and no confirmation or rejection."""
        rows = self._q(
            "SELECT c.compaction_id FROM compactions c WHERE c.session = ? "
            "AND EXISTS (SELECT 1 FROM compaction_returns r WHERE r.compaction = c.compaction_id) "
            "AND NOT EXISTS (SELECT 1 FROM confirmations f WHERE f.compaction = c.compaction_id) "
            "AND NOT EXISTS (SELECT 1 FROM rejections j WHERE j.compaction = c.compaction_id) "
            "ORDER BY c.compaction_id DESC LIMIT 1",
            (session,),
        )
        return int(rows[0][0]) if rows else None

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
        head: Optional[str],
    ) -> tuple[int, dict[int, str]]:
        """Transaction 1: the compaction, its inputs, the new records and their tool calls.

        Returns the compaction id and the record of every input position that has one.
        A record's predecessor is the record of the nearest entry before it in the list
        that has one, else the session's head, else none (the session's first record).
        """
        records: dict[int, str] = {}
        with self._tx() as conn:
            cid = conn.execute(
                "INSERT INTO compactions(session, kind, host_session_before, attempt_generation, began_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (session, kind, host_session_before, attempt_generation, time.time()),
            ).lastrowid
            previous = head
            written: list[tuple[str, dict]] = []
            for entry in entries:
                if entry.klass in ("bound", "reused"):
                    if entry.record is not None:
                        records[entry.position] = entry.record
                        previous = entry.record
                elif entry.klass in ("transcript", "host_insertion"):
                    message = entry.message or {}
                    handle = self._insert_with_handle(
                        conn,
                        MESSAGE,
                        "INSERT INTO records(handle, session, predecessor, compaction, kind, revises, raw, "
                        "role, tool_call_id, text) VALUES (?, ?, ?, ?, ?, NULL, ?, ?, ?, ?)",
                        (
                            session,
                            previous,
                            cid,
                            entry.klass,
                            raw_json(message),
                            message.get("role"),
                            message.get("tool_call_id"),
                            text_content_for_pattern_matching(message.get("content")),
                        ),
                    )
                    records[entry.position] = handle
                    previous = handle
                    written.append((handle, message))
                conn.execute(
                    "INSERT INTO compaction_inputs(compaction, position, host_row_id, record) VALUES (?, ?, ?, ?)",
                    (cid, entry.position, entry.host_row_id, records.get(entry.position)),
                )
            self._write_tool_calls(conn, session, cid, written)
        return int(cid), records

    def _write_tool_calls(self, conn: sqlite3.Connection, session: str, cid: int, written: list[tuple[str, dict]]) -> None:
        """One row per tool call, pointing into its assistant record by position; a
        result recorded in the same compaction is its ``result_record``, one recorded
        by a later compaction is linked through ``tool_results``."""
        results_by_call_id: dict[str, list[str]] = {}
        for handle, message in written:
            if message.get("role") == "tool" and message.get("tool_call_id"):
                results_by_call_id.setdefault(str(message["tool_call_id"]), []).append(handle)
        claimed: set[str] = set()
        for handle, message in written:
            calls = message.get("tool_calls") or []
            if message.get("role") != "assistant" or not isinstance(calls, list):
                continue
            for index, call in enumerate(calls):
                call_id = str(call.get("id") or "") if isinstance(call, dict) else ""
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

    def write_chunk(self, *, session: str, compaction: int, members: Sequence[str]) -> str:
        with self._tx() as conn:
            handle = self._insert_with_handle(
                conn,
                CHUNK,
                "INSERT INTO chunks(handle, session, compaction) VALUES (?, ?, ?)",
                (session, compaction),
            )
            conn.executemany(
                "INSERT INTO chunk_members(chunk, ordinal, record) VALUES (?, ?, ?)",
                [(handle, ordinal, record) for ordinal, record in enumerate(members)],
            )
        return handle

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
    ) -> str:
        with self._tx() as conn:
            handle = self._insert_with_handle(
                conn,
                DERIVATION,
                "INSERT INTO derivations(handle, kind, text, compaction, model, provider, effort, prompt, "
                "budget, finish_reason, level, est_tokens, created_at) "
                "VALUES (?, 'summary', ?, ?, ?, ?, NULL, NULL, ?, NULL, ?, ?, ?)",
                (text, compaction, model or None, provider or None, budget, level, est_tokens, time.time()),
            )
            conn.execute(
                "INSERT INTO derivation_sources(derivation, ordinal, chunk, source_derivation) VALUES (?, 0, ?, NULL)",
                (handle, chunk),
            )
        return handle

    def write_returns(self, compaction: int, entries: Iterable[tuple[int, str, Optional[str], Optional[str]]]) -> None:
        with self._tx() as conn:
            conn.executemany(
                "INSERT INTO compaction_returns(compaction, position, kind, record, derivation) VALUES (?, ?, ?, ?, ?)",
                [(compaction, pos, kind, record, derivation) for pos, kind, record, derivation in entries],
            )

    def confirm(self, compaction: int, *, host_session_before: Optional[str], host_session_after: str) -> None:
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO confirmations(compaction, host_session_before, host_session_after, at) VALUES (?, ?, ?, ?)",
                (compaction, host_session_before or None, host_session_after, time.time()),
            )

    def reject(self, compaction: int, *, how: str) -> None:
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO rejections(compaction, how, at) VALUES (?, ?, ?)",
                (compaction, how, time.time()),
            )

    def bind(self, compaction: int, pairs: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
        """Write (position, host_row_id) bindings not written yet; return those written."""
        written: list[tuple[int, int]] = []
        with self._tx() as conn:
            for position, row_id in pairs:
                exists = conn.execute(
                    "SELECT 1 FROM bindings WHERE compaction = ? AND position = ?",
                    (compaction, position),
                ).fetchone()
                if exists:
                    continue
                conn.execute(
                    "INSERT INTO bindings(compaction, position, host_row_id, at) VALUES (?, ?, ?, ?)",
                    (compaction, position, int(row_id), time.time()),
                )
                written.append((position, int(row_id)))
        return written
