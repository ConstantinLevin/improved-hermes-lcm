"""The plugin's record of what happened at each compaction (#29, W1-W3; #1, #2, #3).

It is the only store: the summariser reads from it, the return is emitted from it,
and the tools read views over it. Every table here is insert-only. A compaction is
written in short transactions, none held open across a summariser call:

1. the compaction, its inputs, a record for every input entry the store does not
   hold yet (the fresh tail included), their tool calls, and every chunk with its
   members, all before the first summariser call, in the planning transaction that
   also holds every read the cut depends on (``planning``; #33 D14 as revised);
2. each summary, as a derivation of its chunk, when it arrives; each failure of a
   chunk's call with its kind, when it happens (#7);
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
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

from .db_bootstrap import close_connection, open_store
from .handles import CHUNK, DERIVATION, MESSAGE, TOOL_CALL, new_handle
from .inflight import ChunkSummary
from .message_content import base64_like_strings, describe_image_part, image_parts, index_text
from .tokens import Estimator

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


HANDLE_RE = re.compile(r"[mtcs][a-z2-7]{8}")


@dataclass(frozen=True)
class Cover:
    """A session's latest effective return as the tools read it (``RecordStore.cover``)."""

    session: str
    compaction: int
    summaries: list        # the summary entries' derivation handles, in return order
    reaches: dict          # derivation -> the chunks it reaches, in order
    chunks: list           # every chunk under the summaries, in cover order
    tail: list             # the record entries (the stored fresh tail), in order


@dataclass(frozen=True)
class Resolved:
    """What a handle names for a caller (#29 W5). ``status``: "ok"; "malformed" (not a
    handle); "unknown" (not in this store); "other_session"; "inactive" (this session's,
    but not on its active record: a reverted branch, an attempt that never took effect,
    or a session with no effective compaction yet)."""

    status: str
    kind: str
    handle: str


def parse_ret_key(value: Any) -> Optional[tuple[int, int]]:
    if not isinstance(value, str) or ":" not in value:
        return None
    left, _, right = value.partition(":")
    try:
        return int(left), int(right)
    except ValueError:
        return None


@dataclass(frozen=True)
class FrozenChunk:
    """A chunk of an unconfirmed attempt that a retry keeps (#33 D14): its members as
    (record, the host's ``_row_id`` when it was cut), in order, and its state,
    "summarised" (its summary is reused) or "cut" (retried as the same chunk)."""

    chunk: str
    compaction: int
    members: tuple
    state: str

    @property
    def records(self) -> list[str]:
        return [record for record, _row_id in self.members]


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
        # How deep the thread that holds ``_lock`` is inside ``_tx``: an inner ``_tx``
        # joins the outer transaction (the planning transaction, ``planning``).
        self._tx_depth = 0
        # How deep the thread that holds ``_lock`` is inside ``snapshot``: a read
        # transaction, in which no event is flushed (that would need the write lock).
        self._read_depth = 0
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

    def close(self, reason: str = "closed") -> None:
        """Close the connection once any statement or transaction of this helper on
        another thread has finished (the helper's lock). Later use raises."""
        with self._lock:
            self._conn = close_connection(self._conn, db_path=self.db_path, reason=reason, owner="the record")

    @contextlib.contextmanager
    def _tx(self):
        """One short write transaction. It never stays open: when the body or the
        COMMIT fails (in rollback-journal mode a COMMIT can fail on the busy timeout
        while a reader holds its shared lock), the transaction is rolled back before
        the error is raised, so that no lock of the shadow ever blocks a live write.
        Inside an open transaction of this helper on the same thread it is a savepoint
        of that transaction: a failed inner block is undone whole, and the outer one
        commits or rolls back."""
        with self._lock:
            if self._tx_depth:
                conn = self._conn
                name = f"lcm_inner_{self._tx_depth}"
                conn.execute(f"SAVEPOINT {name}")
                self._tx_depth += 1
                try:
                    yield conn
                    conn.execute(f"RELEASE {name}")
                except BaseException:
                    try:
                        conn.execute(f"ROLLBACK TO {name}")
                        conn.execute(f"RELEASE {name}")
                    except Exception:
                        logger.error("LCM could not undo a failed step inside a transaction in %s", self.db_path,
                                     exc_info=True)
                    raise
                finally:
                    self._tx_depth -= 1
                return
            conn = self._conn
            conn.execute("BEGIN IMMEDIATE")
            self._tx_depth = 1
            try:
                yield conn
                conn.execute("COMMIT")
            except BaseException:
                self._rollback(conn)
                raise
            finally:
                self._tx_depth = 0
            self._flush_events()

    def transaction(self):
        """One write transaction, or a savepoint inside an open one (``_tx``)."""
        return self._tx()

    def planning(self):
        """The planning transaction of a compaction (#33 D14, revised 2026-09-25): one
        ``BEGIN IMMEDIATE`` around every store read the cut depends on and the write of
        the cut, so that a planner in another process waits for an earlier attempt's
        commit and sees its chunks. Every write inside it joins it. Nothing inside it
        calls a model or does other I/O; it holds the write lock for as long as the
        plan and the write take."""
        return self._tx()

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
        """Write the events not written yet, in their own short transaction; inside an
        open transaction they wait for its end (``_tx`` flushes after its commit)."""
        with self._lock:
            if not self._pending_events or self._conn is None or self._tx_depth or self._read_depth:
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

    def derivations(self, handles: Iterable[str]) -> dict[str, tuple[int, str]]:
        """handle -> (derivation id, text)."""
        wanted = [h for h in set(handles) if h]
        found: dict[str, tuple[int, str]] = {}
        for start in range(0, len(wanted), 500):
            chunk = wanted[start:start + 500]
            rows = self._q(
                "SELECT handle, derivation_id, text FROM derivations "
                f"WHERE handle IN ({','.join('?' * len(chunk))})",
                chunk,
            )
            found.update({str(h): (int(i), str(t)) for h, i, t in rows})
        return found

    def summary_of_records(
        self,
        session: str,
        records: Sequence[str],
        *,
        exclude_chunk: Optional[str],
        model: Optional[str],
        provider: Optional[str],
        effort: Optional[str],
        any_route: bool = False,
    ) -> Optional[ChunkSummary]:
        """The latest summary already written of a chunk of this session with exactly
        these member records, in this order, by the same summariser route and effort, or
        None (#33 D12: a second attempt reuses a summary already written). Records are
        matched by handle, never by content. With ``any_route`` the route and effort do
        not matter: a summarised chunk the frozen cut keeps is never summarised again
        (#33 D14, orchestrator ruling on #61)."""
        if not records:
            return None
        for chunk, _compaction in self._chunks_with_members(session, records):
            if chunk == exclude_chunk:
                continue
            route_clause = "" if any_route else "AND d.model IS ? AND d.provider IS ? AND d.effort IS ? "
            args = (chunk,) if any_route else (chunk, model or None, provider or None, effort or None)
            rows = self._q(
                "SELECT d.text, d.level, d.budget, d.finish_reason, d.model, d.provider, d.effort, "
                "d.withheld_reasoning "
                "FROM derivation_sources s JOIN derivations d ON d.handle = s.derivation "
                "WHERE s.chunk = ? AND s.ordinal = 0 AND d.kind = 'summary' "
                + route_clause + "ORDER BY d.derivation_id DESC LIMIT 1",
                args,
            )
            if rows:
                text, level, budget, finish_reason, model_, provider_, effort_, withheld = rows[0]
                return ChunkSummary(text=str(text), level=level, budget=budget, finish_reason=finish_reason,
                                    model=model_, provider=provider_, effort=effort_, withheld=withheld)
        return None

    def _chunks_with_members(self, session: str, records: Sequence[str]) -> list[tuple[str, int]]:
        """(chunk, compaction) of every chunk of the session with exactly these member
        records, in this order, newest compaction first. By handle, never by content."""
        wanted = [str(r) for r in records]
        if not wanted:
            return []
        found: list[tuple[str, int]] = []
        for chunk, compaction in self._q(
            "SELECT m.chunk, ch.compaction FROM chunk_members m JOIN chunks ch ON ch.handle = m.chunk "
            "WHERE ch.session = ? AND m.ordinal = 0 AND m.record = ? ORDER BY ch.compaction DESC, ch.rowid DESC",
            (session, wanted[0]),
        ):
            members = [str(r) for (r,) in self._q(
                "SELECT record FROM chunk_members WHERE chunk = ? ORDER BY ordinal", (chunk,))]
            if members == wanted:
                found.append((str(chunk), int(compaction)))
        return found

    def _summarised(self, chunk: str) -> bool:
        return bool(self._q(
            "SELECT 1 FROM derivation_sources s JOIN derivations d ON d.handle = s.derivation "
            "WHERE s.chunk = ? AND d.kind = 'summary' LIMIT 1", (chunk,)))

    def _failed_by_itself(self, chunk: str) -> bool:
        """A failure of the chunk's own kind: its reply rejected, or its request
        rejected by the provider (ruling on #61, 2)."""
        return bool(self._q("SELECT 1 FROM chunk_failures WHERE chunk = ? AND kind IN ('reply', 'request') "
                            "LIMIT 1", (chunk,)))

    def chunk_state(self, session: str, records: Sequence[str]) -> str:
        """The state of a set of members over every chunk of the session that held
        exactly them (#33 D14, revised): "summarised" where one of them has a summary,
        which a retry reuses, else "cut": a recorded chunk, retried as the same chunk
        whether or not its call was ever sent. Only chunks that have a summary and the
        same first member are compared member by member, so the cost does not grow with
        the attempts that cut the same chunk."""
        wanted = [str(r) for r in records]
        if not wanted:
            return "cut"
        for (chunk,) in self._q(
            "SELECT DISTINCT m.chunk FROM chunk_members m JOIN chunks ch ON ch.handle = m.chunk "
            "JOIN derivation_sources s ON s.chunk = m.chunk JOIN derivations d ON d.handle = s.derivation "
            "WHERE ch.session = ? AND m.ordinal = 0 AND m.record = ? AND d.kind = 'summary'",
            (session, wanted[0]),
        ):
            members = [str(r) for (r,) in self._q(
                "SELECT record FROM chunk_members WHERE chunk = ? ORDER BY ordinal", (chunk,))]
            if members == wanted:
                return "summarised"
        return "cut"

    def frozen_chunks(self, session: str, after: Optional[int]
                      ) -> tuple[list[FrozenChunk], list[tuple[str, int, tuple[str, ...]]]]:
        """The chunks a retry keeps (#33 D14 as revised 2026-09-25, #31): every chunk
        recorded by the session's attempts after its effective compaction that the host
        neither confirmed, adopted nor rejected, whether or not its call was ever sent.
        A chunk is recorded with its members before any call of its attempt starts,
        in the planning transaction (``planning``), and is frozen from then on. Read
        inside the next attempt's planning transaction, so a planner in another
        process waits for an earlier attempt's commit and sees its chunks.

        The newest attempt's cut comes first, and a chunk that shares a record with
        one already taken is left out: a safety net only, since the planning
        transaction orders the attempts. Each member carries the ``_row_id`` its row
        had in that attempt's list: ids hold until a commit (#29 W2 step 8), so the
        retry recognises the chunk by identity.

        A chunk with a member whose row came without a host identity (``_row_id``) can
        never be found again by identity: the gateway's replayed history carries none,
        and host scaffolding the host never persists has none on the CLI either (ruling
        3 on #61; the ask to Hermes is A1). Such chunks are returned apart, as (chunk,
        compaction, the member records without identity), and none is kept."""
        compactions = [int(c) for (c,) in self._q(
            "SELECT c.compaction_id FROM compactions c WHERE c.session = ? AND c.compaction_id > ? "
            "AND NOT EXISTS (SELECT 1 FROM confirmations f WHERE f.compaction = c.compaction_id) "
            "AND NOT EXISTS (SELECT 1 FROM adoptions a WHERE a.compaction = c.compaction_id) "
            "AND NOT EXISTS (SELECT 1 FROM rejections r WHERE r.compaction = c.compaction_id) "
            "ORDER BY c.compaction_id DESC",
            (session, after or 0),
        )]
        taken: set = set()
        frozen: list[FrozenChunk] = []
        unidentified: list[tuple[str, int, tuple[str, ...]]] = []
        # The records already taken, as a TEMP table of this connection, so that an
        # older attempt's chunk that overlaps one is left out in SQL, at its first
        # overlapping member, without reading its members (the cost of the read no
        # longer grows with the members of every unsettled attempt). TEMP touches no
        # other connection and takes no lock of the store; it is emptied again at once.
        with self._lock:
            conn = self._conn
            conn.execute("CREATE TEMP TABLE IF NOT EXISTS lcm_taken (record TEXT PRIMARY KEY)")
            conn.execute("DELETE FROM temp.lcm_taken")
            try:
                for compaction in compactions:
                    for (chunk,) in conn.execute(
                        "SELECT ch.handle FROM chunks ch WHERE ch.session = ? AND ch.compaction = ? "
                        "AND NOT EXISTS (SELECT 1 FROM chunk_members m JOIN temp.lcm_taken t ON t.record = m.record "
                        "WHERE m.chunk = ch.handle) ORDER BY ch.rowid",
                        (session, compaction),
                    ).fetchall():
                        # A member's host id as that attempt's list carried it
                        # (idx_compaction_inputs_member).
                        members = tuple(
                            (str(record), int(row_id) if row_id is not None else None)
                            for record, row_id in conn.execute(
                                "SELECT m.record, (SELECT i.host_row_id FROM compaction_inputs i WHERE "
                                "i.compaction = ? AND i.record = m.record ORDER BY i.position LIMIT 1) "
                                "FROM chunk_members m WHERE m.chunk = ? ORDER BY m.ordinal",
                                (compaction, chunk),
                            ).fetchall()
                        )
                        records = [record for record, _ in members]
                        if not records or taken.intersection(records):
                            continue
                        taken.update(records)
                        conn.executemany("INSERT OR IGNORE INTO temp.lcm_taken(record) VALUES (?)",
                                         [(record,) for record in records])
                        without = tuple(record for record, row_id in members if row_id is None)
                        if without:
                            unidentified.append((str(chunk), compaction, without))
                            continue
                        frozen.append(FrozenChunk(str(chunk), compaction, members,
                                                  self.chunk_state(session, records)))
            finally:
                conn.execute("DELETE FROM temp.lcm_taken")
        return frozen, unidentified

    def failure_streak(self, session: str, records: Sequence[str]) -> int:
        """In how many consecutive attempts, the newest first, a chunk of exactly these
        members failed by its own fault (#7, #33; ruling on #61, 2): a summary ends the
        streak; an attempt that reached no outcome for it (cancelled, abandoned), or
        whose failure was the route's, the endpoint's or another's, neither counts nor
        ends it."""
        streak = 0
        for chunk, _compaction in self._chunks_with_members(session, records):
            if self._summarised(chunk):
                break
            if self._failed_by_itself(chunk):
                streak += 1
        return streak

    def chunk_failed(self, chunk: str, error: str, *, kind: str, session: str, records: Sequence[str]) -> int:
        """Record a failure of ``chunk`` with its kind, and return the streak of its
        members, read inside the same transaction (ruling on #61, 4). The transaction
        begins IMMEDIATE, so two threads or processes recording failures of the same
        members read distinct streaks."""
        with self._tx() as conn:
            conn.execute("INSERT INTO chunk_failures(chunk, at, kind, error) VALUES (?, ?, ?, ?)",
                         (chunk, time.time(), kind, str(error)))
            return self.failure_streak(session, records)

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

    # --- Reading for the tools (#18; #29 W5) -----------------------------------------

    @contextlib.contextmanager
    def snapshot(self):
        """One read transaction around a tool's reads, so that a compaction committing
        meanwhile, in this process or another, is not half seen. It holds the helper's
        lock and, in rollback-journal mode, a shared lock on the file: a writer's commit
        waits for it (within its busy timeout), so nothing slow runs inside it, and never
        a model call. Inside an open transaction of this helper it joins that one."""
        with self._lock:
            conn = self._conn
            if self._tx_depth or self._read_depth:
                self._read_depth += 1
                try:
                    yield
                finally:
                    self._read_depth -= 1
                return
            conn.execute("BEGIN")
            self._read_depth = 1
            try:
                yield
            except BaseException:
                self._rollback(conn)
                raise
            else:
                conn.execute("COMMIT")
            finally:
                self._read_depth = 0
            self._flush_events()

    def cover(self, session: str) -> Optional["Cover"]:
        """What the session's latest effective compaction returned, as the tools see it:
        its summary entries in order, the chunks each reaches, and its record entries
        (the stored fresh tail). None where the session has no effective compaction."""
        compaction = self.effective_compaction(session)
        if compaction is None:
            return None
        entries = self.return_entries(compaction)
        summaries: list[str] = []
        tail: list[str] = []
        for position in sorted(entries):
            kind, record, derivation, _raw = entries[position]
            if kind == "summary" and derivation:
                summaries.append(str(derivation))
            elif kind == "record" and record:
                tail.append(str(record))
        reaches = {derivation: self._chunks_of(derivation) for derivation in summaries}
        chunks = [chunk for derivation in summaries for chunk in reaches[derivation]]
        return Cover(session=session, compaction=compaction, summaries=summaries, reaches=reaches,
                     chunks=chunks, tail=tail)

    def resolve(self, handle: str, session: str, cover: Optional["Cover"]) -> "Resolved":
        """What a handle names, for the tools of ``session`` (#29 W5): a handle resolves
        only in this store and only in the caller's plugin session, and there only where
        the session's active record holds it (its latest effective return: the stored tail,
        and the chunks under its summaries). Never against a host identifier."""
        text = str(handle or "").strip()
        if not HANDLE_RE.fullmatch(text):
            return Resolved("malformed", "", text)
        kind = text[0]
        home: Optional[str] = None
        if kind == MESSAGE:
            rows = self._q("SELECT session FROM records WHERE handle = ?", (text,))
            home = str(rows[0][0]) if rows else None
        elif kind == TOOL_CALL:
            rows = self._q("SELECT r.session, r.handle FROM tool_calls t JOIN records r ON r.handle = t.record "
                           "WHERE t.handle = ?", (text,))
            home = str(rows[0][0]) if rows else None
        elif kind == CHUNK:
            rows = self._q("SELECT session FROM chunks WHERE handle = ?", (text,))
            home = str(rows[0][0]) if rows else None
        elif kind == DERIVATION:
            reached = self._chunks_of(text) if self._q("SELECT 1 FROM derivations WHERE handle = ?", (text,)) else []
            if reached:
                sessions = {str(s) for (s,) in self._q(
                    f"SELECT DISTINCT session FROM chunks WHERE handle IN ({','.join('?' * len(reached))})", reached)}
                # A derivation reaches the chunks of one session; more than one is no session's.
                home = next(iter(sessions)) if len(sessions) == 1 else ""
        if home is None:
            return Resolved("unknown", kind, text)
        if home != session:
            return Resolved("other_session", kind, text)
        if cover is None:
            return Resolved("inactive", kind, text)
        chunks = set(cover.chunks)
        if kind == CHUNK:
            active = text in chunks
        elif kind == DERIVATION:
            active = all(chunk in chunks for chunk in reached)
        else:
            record = text if kind == MESSAGE else str(rows[0][1])
            active = self._record_active(record, cover)
        return Resolved("ok" if active else "inactive", kind, text)

    def _record_active(self, record: str, cover: "Cover") -> bool:
        if record in cover.tail:
            return True
        if not cover.chunks:
            return False
        return bool(self._q(
            f"SELECT 1 FROM chunk_members WHERE record = ? AND chunk IN ({','.join('?' * len(cover.chunks))}) "
            f"LIMIT 1", (record, *cover.chunks)))

    def chunk_records(self, chunk: str) -> list[tuple[str, dict]]:
        """(record handle, the host's dict as stored) of a chunk's members, in order."""
        return [(str(handle), json.loads(raw)) for handle, raw in self._q(
            "SELECT r.handle, r.raw FROM chunk_members m JOIN records r ON r.handle = m.record "
            "WHERE m.chunk = ? ORDER BY m.ordinal", (chunk,))]

    def record_raw(self, record: str) -> Optional[dict]:
        rows = self._q("SELECT raw FROM records WHERE handle = ?", (record,))
        return json.loads(rows[0][0]) if rows else None

    def derivation_sources(self, derivation: str) -> list[tuple[Optional[str], Optional[str]]]:
        """(chunk, source derivation) of a derivation, in order: exactly one is set."""
        return [(c, d) for c, d in self._q(
            "SELECT chunk, source_derivation FROM derivation_sources WHERE derivation = ? ORDER BY ordinal",
            (derivation,))]

    def derivation_text(self, derivation: str) -> Optional[str]:
        rows = self._q("SELECT text FROM derivations WHERE handle = ?", (derivation,))
        return str(rows[0][0]) if rows else None

    def leaf_summaries(self, chunks: Sequence[str], among: Sequence[str]) -> dict[str, str]:
        """chunk -> the summary among ``among`` whose only source is that chunk."""
        found: dict[str, str] = {}
        for derivation in among:
            sources = self.derivation_sources(derivation)
            if len(sources) == 1 and sources[0][0] in chunks:
                found[str(sources[0][0])] = derivation
        return found

    def tool_calls_of(self, records: Sequence[str]) -> dict[str, dict[int, str]]:
        """record -> {position in its ``tool_calls``: the call's handle}. A revision's
        calls keep the handles its sources' calls have (``begin_compaction``); those are
        given under the revision, by position where the call ids agree."""
        wanted = [r for r in dict.fromkeys(records) if r]
        found: dict[str, dict[int, str]] = {}
        for start in range(0, len(wanted), 500):
            part = wanted[start:start + 500]
            for record, position, handle in self._q(
                    f"SELECT record, position, handle FROM tool_calls WHERE record IN ({','.join('?' * len(part))})",
                    part):
                found.setdefault(str(record), {})[int(position)] = str(handle)
        return found

    def revision_call_handles(self, revision: str) -> dict[str, str]:
        """tool_call_id -> the call handle its sources hold, for a revision record."""
        sources = [str(s) for (s,) in self._q(
            "SELECT source_record FROM revision_sources WHERE revision = ? AND source_record IS NOT NULL",
            (revision,))]
        if not sources:
            return {}
        return {str(call_id): str(handle) for call_id, handle in self._q(
            f"SELECT tool_call_id, handle FROM tool_calls WHERE tool_call_id IS NOT NULL AND record IN "
            f"({','.join('?' * len(sources))})", sources)}

    def record_kind(self, record: str) -> Optional[str]:
        rows = self._q("SELECT kind FROM records WHERE handle = ?", (record,))
        return str(rows[0][0]) if rows else None

    def call_of_result(self, records: Sequence[str]) -> dict[str, str]:
        """result record -> the handle of the tool call it answers, where one is recorded."""
        wanted = [r for r in dict.fromkeys(records) if r]
        found: dict[str, str] = {}
        for start in range(0, len(wanted), 500):
            part = wanted[start:start + 500]
            marks = ",".join("?" * len(part))
            for handle, result in self._q(
                    f"SELECT handle, result_record FROM tool_calls WHERE result_record IN ({marks}) "
                    f"UNION ALL SELECT tool_call, result_record FROM tool_results WHERE result_record IN ({marks})",
                    part + part):
                found.setdefault(str(result), str(handle))
        return found

    def tool_call(self, handle: str) -> Optional[tuple[str, int, Optional[str]]]:
        """(the assistant record, the call's position in it, its result record) of a call
        handle; the result from ``tool_results`` where a later compaction recorded it."""
        rows = self._q("SELECT record, position, result_record FROM tool_calls WHERE handle = ?", (handle,))
        if not rows:
            return None
        record, position, result = rows[0]
        if result is None:
            linked = self._q("SELECT result_record FROM tool_results WHERE tool_call = ? LIMIT 1", (handle,))
            result = linked[0][0] if linked else None
        return str(record), int(position), (str(result) if result else None)

    def handles_of_records(self, record_ids: Iterable[int]) -> dict[int, str]:
        """record id (the views' store_id) -> its handle."""
        wanted = [int(i) for i in dict.fromkeys(record_ids) if i is not None]
        found: dict[int, str] = {}
        for start in range(0, len(wanted), 500):
            part = wanted[start:start + 500]
            found.update({int(i): str(h) for i, h in self._q(
                f"SELECT record_id, handle FROM records WHERE record_id IN ({','.join('?' * len(part))})", part)})
        return found

    def handles_of_derivations(self, derivation_ids: Iterable[int]) -> dict[int, str]:
        """derivation id (the views' node_id) -> its handle."""
        wanted = [int(i) for i in dict.fromkeys(derivation_ids) if i is not None]
        found: dict[int, str] = {}
        for start in range(0, len(wanted), 500):
            part = wanted[start:start + 500]
            found.update({int(i): str(h) for i, h in self._q(
                f"SELECT derivation_id, handle FROM derivations WHERE derivation_id IN ({','.join('?' * len(part))})",
                part)})
        return found

    # --- The invariant (#29 W7, #34 D5) --------------------------------------------

    def identity(self) -> dict[str, Any]:
        rows = self._q("SELECT format, store_uuid, created_at FROM store_identity")
        return {"format": rows[0][0], "store_uuid": rows[0][1], "created_at": rows[0][2]} if rows else {}

    def recent_events(self, limit: int = 20) -> list[dict[str, Any]]:
        return [
            {"at": at, "kind": kind, "session": session, "compaction": compaction, "detail": detail}
            for at, kind, session, compaction, detail in self._q(
                "SELECT at, kind, session, compaction, detail FROM store_events ORDER BY event_id DESC LIMIT ?",
                (int(limit),),
            )
        ]

    def check_invariant(self, max_problems: int = 20) -> list[dict[str, Any]]:
        """W7, in the store, for each session's latest effective compaction.

        - Every record on the branch, from the head (the last record entry of that
          return) back along the predecessors to the session's first record, is a
          record entry of the return or a member of a chunk the return's summary
          entries reach, directly or through ``derivation_sources``; not both. A
          record is insert-only and keeps the predecessor it was written with, so the
          branch reads a revised record through its revision (C2b): the revision an
          effective compaction's input list held stands in its place, and the walk
          goes on from the revision's predecessor.
        - No two chunks the return reaches share a record.
        - Every chunk a summary entry reaches is on this session's active branch: a
          chunk of the session, written by one of its effective compactions, whose
          members are on the branch or stand beside it (a host insertion or a
          revision chained off a branch record).
        - The summary entries form a cover (#34 D5): complete (every chunk an
          effective compaction of the session wrote is covered: nothing of a return
          can leave a later one, and the host offers no revert past a compaction),
          disjoint (by exactly one summary entry), contiguous (each entry covers a
          contiguous run of chunks, and the entries stand in the order of their runs'
          first chunks). The order is the active branch's own: a chunk stands where
          its first member stands on the branch. A chunk of records beside the chain
          only (host insertions) has no place on the branch and takes no part in the
          order. Every summary entry reaches at least one chunk.

        The whole check reads one snapshot: it runs inside one read transaction, so
        a compaction committing meanwhile is not half seen. In rollback-journal mode
        that transaction holds a shared lock, and a writer's commit waits for it
        (within its busy timeout) until the check ends. Every read is scoped to the
        session by an index, so the check stays short on a large store.

        Returns one report per session: the compaction checked, counts, and the
        problems found (at most ``max_problems`` listed, all counted).
        """
        with self._lock:
            conn = self._conn
            conn.execute("BEGIN")
            try:
                reports = [
                    self._check_session(str(session), int(compaction), max_problems)
                    for session, compaction in self._q(
                        "SELECT session, MAX(compaction_id) FROM effective_compactions "
                        "GROUP BY session ORDER BY session"
                    )
                ]
            except BaseException:
                self._rollback(conn)
                raise
            conn.execute("COMMIT")
        return reports

    def _chunks_of(self, derivation: str) -> list[str]:
        """The chunks a derivation covers: its sources when they are chunks, and the
        chunks of its sources, recursively, when they are derivations. The visited
        set makes the walk finite; there is no depth limit."""
        chunks: list[str] = []
        seen: set[str] = set()
        stack = [derivation]
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            sources = self._q(
                "SELECT chunk, source_derivation FROM derivation_sources WHERE derivation = ? ORDER BY ordinal",
                (current,),
            )
            for chunk, source in reversed(sources):
                if chunk:
                    chunks.append(str(chunk))
                elif source:
                    stack.append(str(source))
        return list(dict.fromkeys(chunks))

    def _check_session(self, session: str, compaction: int, max_problems: int) -> dict[str, Any]:
        problems: list[str] = []
        problem_count = 0

        def problem(text: str) -> None:
            nonlocal problem_count
            problem_count += 1
            if len(problems) < max_problems:
                problems.append(text)

        returns = self._q(
            "SELECT position, kind, record, derivation FROM compaction_returns WHERE compaction = ? ORDER BY position",
            (compaction,),
        )
        record_entries = [str(r) for _p, kind, r, _d in returns if kind == "record" and r]
        summary_entries = [(int(p), str(d)) for p, kind, _r, d in returns if kind == "summary" and d]

        effective = {int(k) for (k,) in self._q(
            "SELECT compaction_id FROM effective_compactions WHERE session = ?", (session,))}

        # The chunks each summary entry reaches, and where each chunk belongs.
        covered_by: dict[str, list[int]] = {}
        entry_chunks: dict[int, list[str]] = {}
        for position, derivation in summary_entries:
            chunks = self._chunks_of(derivation)
            entry_chunks[position] = chunks
            if not chunks:
                problem(f"summary entry {position} reaches no chunk")
            for chunk in chunks:
                covered_by.setdefault(chunk, []).append(position)
        chunk_home = {
            chunk: rows[0] if rows else None
            for chunk in covered_by
            for rows in [self._q("SELECT session, compaction FROM chunks WHERE handle = ?", (chunk,))]
        }
        effective_chunks = [(str(c), int(k)) for c, k in self._q(
            "SELECT handle, compaction FROM chunks WHERE session = ? ORDER BY compaction, rowid", (session,))
            if int(k) in effective]
        members: dict[str, list[str]] = {}
        for chunk, record in self._q(
            "SELECT m.chunk, m.record FROM chunks ch JOIN chunk_members m ON m.chunk = ch.handle "
            "WHERE ch.session = ? ORDER BY m.chunk, m.ordinal",
            (session,),
        ):
            members.setdefault(str(chunk), []).append(str(record))

        # A revised record is read through the revision an effective compaction's
        # input held (the latest such), transitively.
        revised: dict[str, tuple[int, str]] = {}
        for source, revision, at in self._q(
            "SELECT s.source_record, s.revision, i.compaction FROM records r "
            "JOIN revision_sources s ON s.revision = r.handle "
            "JOIN compaction_inputs i ON i.record = r.handle "
            "WHERE r.session = ? AND r.kind = 'revision' AND s.source_record IS NOT NULL",
            (session,),
        ):
            if int(at) in effective and (str(source) not in revised or int(at) > revised[str(source)][0]):
                revised[str(source)] = (int(at), str(revision))

        def resolve(record: str) -> str:
            seen_revisions = {record}
            while record in revised and revised[record][1] not in seen_revisions:
                record = revised[record][1]
                seen_revisions.add(record)
            return record

        # The branch: from the head back along the predecessors.
        predecessor: dict[str, Optional[str]] = {}
        kind_of: dict[str, str] = {}
        for handle, previous_handle, kind in self._q(
            "SELECT handle, predecessor, kind FROM records WHERE session = ?", (session,)
        ):
            predecessor[str(handle)] = str(previous_handle) if previous_handle else None
            kind_of[str(handle)] = str(kind)
        branch: list[str] = []
        if not record_entries:
            problem("the return has no record entry, so the branch has no head")
        else:
            seen: set[str] = set()
            cursor: Optional[str] = resolve(record_entries[-1])
            while cursor is not None:
                if cursor in seen:
                    problem(f"the predecessors form a cycle at {cursor}")
                    break
                seen.add(cursor)
                branch.append(cursor)
                previous = predecessor.get(cursor)
                cursor = resolve(previous) if previous else None
            branch.reverse()
        on_branch = {record: index for index, record in enumerate(branch)}

        def beside(record: str) -> bool:
            """A host insertion or a revision chained off a branch record, not on it."""
            if record in on_branch or kind_of.get(record, "transcript") == "transcript":
                return False
            previous = predecessor.get(record)
            return previous is not None and resolve(previous) in on_branch

        # Every chunk a summary entry reaches is on this session's active branch.
        for position, _derivation in summary_entries:
            for chunk in entry_chunks.get(position, []):
                home = chunk_home.get(chunk)
                if home is None:
                    problem(f"summary entry {position} reaches chunk {chunk}, which the store does not hold")
                elif str(home[0]) != session:
                    problem(f"summary entry {position} reaches chunk {chunk} of session {home[0]}, "
                            f"not of this session's active branch")
                elif int(home[1]) not in effective:
                    problem(f"summary entry {position} reaches chunk {chunk} of compaction {home[1]}, "
                            f"which is not effective, so not on the active branch")
                else:
                    off = [r for r in members.get(chunk, []) if r not in on_branch and not beside(r)]
                    if off:
                        problem(f"summary entry {position} reaches chunk {chunk}, whose record {off[0]} "
                                f"is not on the active branch")

        # Every record on the branch: a record entry, or under a reached chunk; not both.
        in_tail = set(record_entries)
        under_summary: dict[str, list[str]] = {}
        for chunk in covered_by:
            for record in members.get(chunk, []):
                under_summary.setdefault(record, []).append(chunk)
        for record in branch:
            if record not in in_tail and record not in under_summary:
                problem(f"record {record} on the branch is neither in the tail nor under a summary")
            elif record in in_tail and record in under_summary:
                problem(f"record {record} is both in the tail and under a summary")
        for record, chunks in under_summary.items():
            if len(chunks) > 1:
                problem(f"record {record} is in {len(chunks)} chunks the return reaches: {', '.join(chunks)}")

        # The cover: complete, disjoint, contiguous.
        for chunk, chunk_compaction in effective_chunks:
            if chunk not in covered_by:
                problem(f"chunk {chunk} of compaction {chunk_compaction} is covered by no summary entry")
        for chunk, positions in covered_by.items():
            if len(positions) > 1:
                problem(f"chunk {chunk} is covered by {len(positions)} summary entries: {positions}")

        # Contiguity in the active branch's own order: a chunk stands where its first
        # member on the branch stands. Chunks with no member on the branch (host
        # insertions only, beside the chain) have no place in it.
        place = {
            chunk: min(on_branch[r] for r in members.get(chunk, []) if r in on_branch)
            for chunk in covered_by
            if any(r in on_branch for r in members.get(chunk, []))
        }
        order = sorted(place, key=lambda c: place[c])
        rank = {chunk: index for index, chunk in enumerate(order)}
        previous_first = -1
        for position, _derivation in summary_entries:
            ranks = sorted(rank[c] for c in entry_chunks.get(position, []) if c in rank)
            if not ranks:
                continue
            if ranks != list(range(ranks[0], ranks[0] + len(ranks))):
                problem(f"summary entry {position} covers chunks that are not a contiguous run")
            if ranks[0] < previous_first:
                problem(f"summary entry {position} stands before an entry whose run begins earlier")
            previous_first = ranks[0]

        return {
            "session": session,
            "compaction": compaction,
            "records_on_branch": len(branch),
            "tail_records": len(record_entries),
            "summary_entries": len(summary_entries),
            "chunks_reached": len(covered_by),
            "status": "pass" if problem_count == 0 else "fail",
            "problem_count": problem_count,
            "problems": problems,
        }

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
        estimator: Optional[Estimator] = None,
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
        uncounted: dict[str, int] = {}
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
                    estimate = (estimator or Estimator()).message(message)
                    handle = self._insert_with_handle(
                        conn,
                        MESSAGE,
                        "INSERT INTO records(handle, session, predecessor, compaction, kind, raw, "
                        "role, tool_call_id, text, est_tokens, est_uncounted_images) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            session,
                            predecessor,
                            cid,
                            entry.klass,
                            raw_json(message),
                            message.get("role"),
                            message.get("tool_call_id"),
                            index_text(message.get("content")),
                            estimate.tokens,
                            estimate.uncounted_images,
                        ),
                    )
                    records[entry.position] = handle
                    if estimate.uncounted_images:
                        uncounted[handle] = estimate.uncounted_images
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
        if uncounted:
            # "Known, or nothing" (#35): the estimate of these records leaves images
            # out, so every count over them reads lower than the context by those images.
            logger.warning(
                "LCM estimate leaves %d images uncounted (no image rule for model %r, or no readable size) "
                "in records %s",
                sum(uncounted.values()), (estimator or Estimator()).image_model,
                ", ".join(f"{handle} ({count})" for handle, count in uncounted.items()),
            )
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
        finish_reason: Optional[str] = None,
        effort: Optional[str] = None,
        withheld_reasoning: Optional[str] = None,
    ) -> str:
        with self._tx() as conn:
            handle = self._insert_with_handle(
                conn,
                DERIVATION,
                "INSERT INTO derivations(handle, kind, text, compaction, model, provider, effort, prompt, "
                "budget, finish_reason, level, est_tokens, withheld_reasoning, created_at) "
                "VALUES (?, 'summary', ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?, ?, ?)",
                (text, compaction, model or None, provider or None, effort or None, budget, finish_reason or None,
                 level, est_tokens, withheld_reasoning, time.time()),
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
        *,
        fence: Optional[Callable[[], None]] = None,
    ) -> None:
        """(position, kind, record, derivation, raw): a returned summary names the
        derivation it was emitted from and keeps its dict as returned, verbatim, since
        it is what the agent's context held.

        ``fence`` is asked inside the transaction, once the write lock is granted (the
        wait for it can last the busy timeout) and again just before COMMIT; where it
        raises, the transaction is rolled back and nothing of the return is written
        (#7, #33 D12). What it cannot close is a cancellation during COMMIT itself."""
        rows = [(compaction, pos, kind, record, derivation, raw) for pos, kind, record, derivation, raw in entries]
        with self._tx() as conn:
            if fence is not None:
                fence()
            conn.executemany(
                "INSERT INTO compaction_returns(compaction, position, kind, record, derivation, raw) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                rows,
            )
            if fence is not None:
                fence()

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
