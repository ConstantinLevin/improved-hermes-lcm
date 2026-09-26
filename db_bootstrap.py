"""The plugin's SQLite store: identity, creation, connection settings, and FTS health.

A store is a database this plugin created. It carries one ``store_identity`` row
written at creation. A database without that row, or with another format, was not
written by this plugin and is refused: nothing in it is read or changed. There are no
migrations; a change of format means the store is begun again.

The whole schema is created once per database, in one transaction, when the plugin
creates the store. Opening an existing store runs no DDL; it only checks the identity
and the health of the full-text indexes, and repairs an index only when it is damaged.
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Sequence

logger = logging.getLogger(__name__)


# The layout this build writes. A store with any other format is refused. Format 4
# makes the record the only store: the old path's tables are gone, and what the
# tools read are views over the record (#29, #1). Format 5 adds the indexes the
# doctor's invariant check reads by (chunks by session, rejections by compaction),
# so that a store without them is never opened. Format 6 stores beside a record's
# estimate how many of its images the estimate left uncounted
# (``records.est_uncounted_images``, #21, #35), and the views show it. Format 10
# records each failure of a chunk's call with its kind (``chunk_failures``,
# insert-only), for the chunk that keeps failing (#7); a retry keeps every recorded
# chunk of an unsettled attempt, so nothing about dispatch is stored (#33 D14 as
# revised); and it indexes ``compaction_inputs`` by (compaction, record), which the
# frozen cut reads inside the planning transaction. Formats 7 to 9 were written only
# by unmerged commits of #61 (7 and 8 with a ``chunk_dispatches`` table, 9 without
# that index); opening runs no DDL, so each layout has its own name and is refused.
# A store of an earlier format is refused and begun again (#29 W8: a change of format
# wipes the store). Format 11 adds ``derivations.withheld_reasoning``: the encrypted
# reasoning withheld from the summariser's input, named in the summary's provenance (#8);
# adds ``derivations.effort_sent``: the effort field the call actually carried, or why
# none was sent (#9: a configured route sends the effort only where the model table
# documents its field); and drops ``derivations.expand_hint``, a text a pattern took from
# the summary (#9).
STORE_FORMAT = "ihl-store/11"
# The default file name under the host-given Hermes home.
STORE_FILENAME = "lcm-record.db"
SQLITE_BUSY_TIMEOUT_MS = 30_000
REQUIRED_CORE_TABLES = (
    "store_identity",
    "sessions",
    "session_facts",
    "compactions",
    "records",
    "chunks",
    "chunk_members",
    "derivations",
    "derivation_sources",
    "compaction_returns",
    "messages_fts",
    "nodes_fts",
)


class StoreRefusedError(RuntimeError):
    """The database at a path is not a store this plugin can open.

    Raised before anything in the database is read beyond its table names, or
    before a store is created where it cannot be kept safely. The message names the
    path and the reason.
    """


class StoreClosedError(RuntimeError):
    """A connection to the store was used after it was closed. A closed handle is
    never reused or reopened in place; the message names the store and why it was
    closed."""


class ClosedConnection:
    """What a store helper holds once its connection is closed. It is false, and every
    use raises :class:`StoreClosedError`, so a closed handle is never reused."""

    def __init__(self, db_path: str | Path, reason: str):
        self._db_path = str(db_path)
        self._reason = reason

    def __bool__(self) -> bool:
        return False

    def __getattr__(self, name: str):
        raise StoreClosedError(
            f"LCM's connection to the store at {self._db_path} was closed ({self._reason}); "
            f"a closed handle is never reused"
        )


class FetchedCursor:
    """The rows of one statement, fetched in full while the helper's lock was held."""

    def __init__(self, rows: list, description, rowcount: int, lastrowid):
        self._rows = rows
        self._next = 0
        self.description = description
        self.rowcount = rowcount
        self.lastrowid = lastrowid

    def fetchone(self):
        if self._next >= len(self._rows):
            return None
        row = self._rows[self._next]
        self._next += 1
        return row

    def fetchmany(self, size: int = 1) -> list:
        rows = self._rows[self._next:self._next + size]
        self._next += len(rows)
        return rows

    def fetchall(self) -> list:
        rows = self._rows[self._next:]
        self._next = len(self._rows)
        return rows

    def __iter__(self):
        while (row := self.fetchone()) is not None:
            yield row


class LockedConnection:
    """A store helper's connection as its readers use it.

    Every call runs under the lock the helper's ``close()`` takes, and ``execute``
    returns its rows fetched in full under that lock. So a close never lands between
    a statement and its rows: it waits for the read, or the read comes after it and
    raises :class:`StoreClosedError` (the closed connection's own error).
    """

    def __init__(self, get_conn, lock):
        self._get_conn = get_conn
        self._lock = lock

    def __bool__(self) -> bool:
        return bool(self._get_conn())

    def execute(self, sql: str, parameters=()) -> FetchedCursor:
        with self._lock:
            cursor = self._get_conn().execute(sql, parameters)
            rows = cursor.fetchall()
            return FetchedCursor(rows, cursor.description, cursor.rowcount, cursor.lastrowid)

    def __getattr__(self, name: str):
        with self._lock:
            value = getattr(self._get_conn(), name)
        if not callable(value):
            return value

        def locked_call(*args, **kwargs):
            with self._lock:
                return getattr(self._get_conn(), name)(*args, **kwargs)
        return locked_call


def close_connection(conn, *, db_path: str | Path, reason: str, owner: str) -> ClosedConnection:
    """Close one store connection and return what stands in its place.

    A transaction still open on it is rolled back, and that is logged at WARNING:
    nothing is rolled back silently. Closing an already closed connection changes
    nothing.
    """
    if isinstance(conn, ClosedConnection):
        return conn
    if conn is not None:
        try:
            if conn.in_transaction:
                logger.warning(
                    "LCM rolled back an open transaction of %s on the store at %s while closing it (%s)",
                    owner, db_path, reason,
                )
                conn.execute("ROLLBACK")
        finally:
            conn.close()
    return ClosedConnection(db_path, reason)


def _refuse(path: str | Path, reason: str) -> StoreRefusedError:
    message = (
        f"LCM refuses the database at {path}: {reason}. Nothing in it was read or "
        f"changed. Point LCM_DATABASE_PATH at a new file, or move this one away."
    )
    logger.error(message)
    return StoreRefusedError(message)


class ExternalContentFtsSpec:
    def __init__(
        self,
        *,
        table_name: str,
        content_table: str,
        content_rowid: str,
        indexed_column: str,
        trigger_sqls: Sequence[str],
    ) -> None:
        self.table_name = table_name
        self.content_table = content_table
        self.content_rowid = content_rowid
        self.indexed_column = indexed_column
        self.trigger_sqls = tuple(trigger_sqls)


# The full-text indexes are derived from the record and only grow with it (#29 W1):
# one over the records' text, one over the derivations' text. Both tables are
# insert-only, so an insert trigger is all an index needs. The tools join them to
# the ``messages`` and ``summary_nodes`` views by id, which scope what is visible.
MESSAGES_FTS_SPEC = ExternalContentFtsSpec(
    table_name="messages_fts",
    content_table="records",
    content_rowid="record_id",
    indexed_column="text",
    trigger_sqls=(
        """
        CREATE TRIGGER IF NOT EXISTS records_fts_insert
            AFTER INSERT ON records BEGIN
            INSERT INTO messages_fts(rowid, text)
                VALUES (new.record_id, new.text);
        END;
        """,
    ),
)

NODES_FTS_SPEC = ExternalContentFtsSpec(
    table_name="nodes_fts",
    content_table="derivations",
    content_rowid="derivation_id",
    indexed_column="text",
    trigger_sqls=(
        """
        CREATE TRIGGER IF NOT EXISTS derivations_fts_insert
            AFTER INSERT ON derivations BEGIN
            INSERT INTO nodes_fts(rowid, text)
                VALUES (new.derivation_id, new.text);
        END;
        """,
    ),
)

FTS_SPECS = (MESSAGES_FTS_SPEC, NODES_FTS_SPEC)


# The tables of the plugin's own record. Each is insert-only: triggers raise on any
# UPDATE or DELETE, so that append-only is a property of the database (#29, W1).
INSERT_ONLY_TABLES = (
    "store_identity",
    "sessions",
    "session_facts",
    "compactions",
    "records",
    "tool_calls",
    "tool_results",
    "compaction_inputs",
    "chunks",
    "chunk_members",
    "chunk_failures",
    "derivations",
    "derivation_sources",
    "compaction_returns",
    "revision_sources",
    "confirmations",
    "rejections",
    "adoptions",
    "bindings",
    "store_events",
)


def _insert_only_triggers_sql(tables: Sequence[str]) -> str:
    return "\n".join(
        f"CREATE TRIGGER {table}_no_{verb} BEFORE {verb.upper()} ON {table}\n"
        f"    BEGIN SELECT RAISE(ABORT, '{table} is insert-only'); END;"
        for table in tables
        for verb in ("update", "delete")
    )


_RECORD_SQL = """
CREATE TABLE store_identity (
    format TEXT NOT NULL,
    store_uuid TEXT NOT NULL,
    created_at REAL NOT NULL
);

CREATE TABLE sessions (
    handle TEXT PRIMARY KEY,
    began_at REAL NOT NULL,
    signal TEXT NOT NULL,
    kind TEXT,
    host_session_id TEXT NOT NULL UNIQUE
);

CREATE TABLE session_facts (
    fact_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session TEXT NOT NULL REFERENCES sessions(handle),
    kind TEXT NOT NULL,
    value TEXT NOT NULL,
    signal TEXT NOT NULL,
    host_session_id TEXT,
    at REAL NOT NULL
);
CREATE INDEX idx_session_facts_session ON session_facts(session, fact_id);

CREATE TABLE compactions (
    compaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session TEXT NOT NULL REFERENCES sessions(handle),
    kind TEXT NOT NULL,
    host_session_before TEXT,
    attempt_generation INTEGER,
    began_at REAL NOT NULL
);
CREATE INDEX idx_compactions_session ON compactions(session, compaction_id);

CREATE TABLE records (
    record_id INTEGER PRIMARY KEY AUTOINCREMENT,
    handle TEXT NOT NULL UNIQUE,
    session TEXT NOT NULL REFERENCES sessions(handle),
    predecessor TEXT REFERENCES records(handle),
    compaction INTEGER NOT NULL REFERENCES compactions(compaction_id),
    kind TEXT NOT NULL CHECK (kind IN ('transcript', 'host_insertion', 'revision')),
    raw TEXT NOT NULL,
    role TEXT,
    tool_call_id TEXT,
    text TEXT,
    est_tokens INTEGER,
    est_uncounted_images INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX idx_records_session ON records(session, record_id);

CREATE TABLE tool_calls (
    handle TEXT PRIMARY KEY,
    record TEXT NOT NULL REFERENCES records(handle),
    position INTEGER NOT NULL,
    tool_call_id TEXT,
    result_record TEXT REFERENCES records(handle),
    UNIQUE (record, position)
);
CREATE INDEX idx_tool_calls_id ON tool_calls(tool_call_id);

CREATE TABLE tool_results (
    tool_call TEXT NOT NULL REFERENCES tool_calls(handle),
    result_record TEXT NOT NULL REFERENCES records(handle),
    compaction INTEGER NOT NULL REFERENCES compactions(compaction_id),
    PRIMARY KEY (tool_call, result_record)
);

CREATE TABLE compaction_inputs (
    compaction INTEGER NOT NULL REFERENCES compactions(compaction_id),
    position INTEGER NOT NULL,
    host_row_id INTEGER,
    record TEXT REFERENCES records(handle),
    PRIMARY KEY (compaction, position)
);
CREATE INDEX idx_compaction_inputs_row ON compaction_inputs(host_row_id);
CREATE INDEX idx_compaction_inputs_record ON compaction_inputs(record);
-- A member's host id as its attempt's list carried it, looked up per (compaction,
-- record) by the frozen cut inside the planning transaction (#33 D14).
CREATE INDEX idx_compaction_inputs_member ON compaction_inputs(compaction, record, position);

CREATE TABLE chunks (
    handle TEXT PRIMARY KEY,
    session TEXT NOT NULL REFERENCES sessions(handle),
    compaction INTEGER NOT NULL REFERENCES compactions(compaction_id)
);
CREATE INDEX idx_chunks_session ON chunks(session, compaction);

CREATE TABLE chunk_members (
    chunk TEXT NOT NULL REFERENCES chunks(handle),
    ordinal INTEGER NOT NULL,
    record TEXT NOT NULL REFERENCES records(handle),
    PRIMARY KEY (chunk, ordinal),
    UNIQUE (chunk, record)
);
CREATE INDEX idx_chunk_members_record ON chunk_members(record, ordinal);

-- A chunk's call failed, with the failure's kind (#7; ruling on #61, 2): the chunk's
-- own failures (reply, request) in three consecutive attempts of the same members
-- make the chunk that keeps failing; route, endpoint and other failures do not count.
CREATE TABLE chunk_failures (
    chunk TEXT NOT NULL REFERENCES chunks(handle),
    at REAL NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('reply', 'request', 'route', 'endpoint', 'other')),
    error TEXT NOT NULL
);
CREATE INDEX idx_chunk_failures_chunk ON chunk_failures(chunk);

CREATE TABLE derivations (
    derivation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    handle TEXT NOT NULL UNIQUE,
    kind TEXT NOT NULL,
    text TEXT NOT NULL,
    compaction INTEGER REFERENCES compactions(compaction_id),
    model TEXT,
    provider TEXT,
    effort TEXT,
    effort_sent TEXT,
    prompt TEXT,
    budget INTEGER,
    finish_reason TEXT,
    level INTEGER,
    est_tokens INTEGER,
    withheld_reasoning TEXT,
    created_at REAL NOT NULL
);

CREATE TABLE derivation_sources (
    derivation TEXT NOT NULL REFERENCES derivations(handle),
    ordinal INTEGER NOT NULL,
    chunk TEXT REFERENCES chunks(handle),
    source_derivation TEXT REFERENCES derivations(handle),
    PRIMARY KEY (derivation, ordinal),
    CHECK ((chunk IS NULL) != (source_derivation IS NULL))
);

-- A summary entry names the derivation it was emitted from and keeps its dict as
-- returned; a record entry names its record. The kind for #14's in-turn
-- re-insertion is added with #14.
CREATE TABLE compaction_returns (
    compaction INTEGER NOT NULL REFERENCES compactions(compaction_id),
    position INTEGER NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('summary', 'record')),
    record TEXT REFERENCES records(handle),
    derivation TEXT REFERENCES derivations(handle),
    raw TEXT,
    PRIMARY KEY (compaction, position),
    CHECK ((kind = 'summary' AND derivation IS NOT NULL AND raw IS NOT NULL AND record IS NULL)
           OR (kind = 'record' AND record IS NOT NULL AND derivation IS NULL AND raw IS NULL))
);
CREATE INDEX idx_compaction_returns_derivation ON compaction_returns(derivation);

CREATE TABLE revision_sources (
    revision TEXT NOT NULL REFERENCES records(handle),
    ordinal INTEGER NOT NULL,
    source_record TEXT REFERENCES records(handle),
    source_compaction INTEGER,
    source_position INTEGER,
    PRIMARY KEY (revision, ordinal),
    FOREIGN KEY (source_compaction, source_position) REFERENCES compaction_returns(compaction, position),
    CHECK ((source_record IS NOT NULL)
           != (source_compaction IS NOT NULL AND source_position IS NOT NULL)),
    CHECK ((source_compaction IS NULL) = (source_position IS NULL))
);
CREATE INDEX idx_revision_sources_record ON revision_sources(source_record);

CREATE TABLE confirmations (
    compaction INTEGER NOT NULL UNIQUE REFERENCES compactions(compaction_id),
    host_session_before TEXT,
    host_session_after TEXT NOT NULL,
    at REAL NOT NULL
);
CREATE INDEX idx_confirmations_after ON confirmations(host_session_after);

CREATE TABLE rejections (
    compaction INTEGER NOT NULL REFERENCES compactions(compaction_id),
    how TEXT NOT NULL,
    at REAL NOT NULL
);
CREATE INDEX idx_rejections_compaction ON rejections(compaction);

CREATE TABLE adoptions (
    compaction INTEGER NOT NULL UNIQUE REFERENCES compactions(compaction_id),
    evidence TEXT NOT NULL,
    at REAL NOT NULL
);

CREATE TABLE bindings (
    compaction INTEGER NOT NULL REFERENCES compactions(compaction_id),
    host_row_id INTEGER NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('return', 'host_insertion')),
    position INTEGER,
    committed_index INTEGER,
    at REAL NOT NULL,
    PRIMARY KEY (compaction, host_row_id),
    CHECK ((kind = 'return') = (position IS NOT NULL))
);
CREATE UNIQUE INDEX idx_bindings_position ON bindings(compaction, position) WHERE position IS NOT NULL;
CREATE INDEX idx_bindings_row ON bindings(host_row_id);

CREATE TABLE store_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    at REAL NOT NULL,
    kind TEXT NOT NULL,
    session TEXT,
    compaction INTEGER,
    detail TEXT
);
"""

# What the tools read, as read-only views over the record, so that they run
# unchanged (integer ids until #18). Only the session's active branch is visible,
# as W7 defines it (#29), at its latest effective compaction, and what stood beside it:
# - a compaction took effect when the host confirmed it, or its return was found
#   adopted without a confirmation, and the host did not reject it;
# - a record is visible when it is a record entry of that compaction's return (the
#   tail) or a member of a chunk under one of its summaries. A reverted record, a
#   record a revision replaced and an unconfirmed attempt's records are none of these,
#   so they stay invisible;
# - a summary row the host rewrote (a summary revision) is visible beside the chain,
#   when an effective compaction recorded it, naming the summary it revises: what the
#   agent's context held never becomes invisible. It is never chunked, since it holds
#   a summary, and the cover re-emits the summary from its derivation;
# - a summary is visible when that return holds it: the cover.
# ``seq`` is the transcript order: a record entry stands at its return position, a
# chunk member at its summary's position and its ordinal in the chunk, a summary
# revision just before the summary it revises, and a summary just after its chunk.
# Ids are the order of writing, never the transcript's: a host insertion or a revision
# is written after rows it stands before.
# A record's ``timestamp`` is the time of the compaction that stored it; the host's
# own time is ``observed_at``, where the host gave one. ``source`` is the platform the
# session had last stated when the record was written.
_SEQ_COMPACTION = 1_000_000_000_000
_SEQ_POSITION = 1_000_000
_VIEWS_SQL = f"""
CREATE VIEW effective_compactions AS
SELECT c.compaction_id AS compaction_id, c.session AS session, c.began_at AS began_at
FROM compactions c
WHERE (EXISTS (SELECT 1 FROM confirmations f WHERE f.compaction = c.compaction_id)
       OR EXISTS (SELECT 1 FROM adoptions a WHERE a.compaction = c.compaction_id))
  AND NOT EXISTS (SELECT 1 FROM rejections j WHERE j.compaction = c.compaction_id);

CREATE VIEW latest_returns AS
SELECT cr.compaction AS compaction, cr.position AS position, cr.kind AS kind,
       cr.record AS record, cr.derivation AS derivation,
       cr.compaction * {_SEQ_COMPACTION} + cr.position * {_SEQ_POSITION} AS seq
FROM compaction_returns cr
WHERE cr.compaction IN (SELECT MAX(compaction_id) FROM effective_compactions GROUP BY session);

CREATE VIEW summary_revisions AS
WITH RECURSIVE revised(revision, derivation) AS (
    SELECT s.revision, cr.derivation
    FROM revision_sources s
    JOIN compaction_returns cr
      ON cr.compaction = s.source_compaction AND cr.position = s.source_position
    WHERE cr.kind = 'summary'
    UNION
    SELECT s.revision, revised.derivation
    FROM revision_sources s JOIN revised ON s.source_record = revised.revision
)
SELECT revision, derivation FROM revised;

CREATE VIEW record_positions AS
SELECT record, MIN(seq) AS seq FROM (
    SELECT l.record AS record, l.seq AS seq
    FROM latest_returns l WHERE l.kind = 'record'
    UNION ALL
    SELECT m.record, l.seq + 1 + m.ordinal
    FROM latest_returns l
    JOIN derivation_sources s ON s.derivation = l.derivation
    JOIN chunk_members m ON m.chunk = s.chunk
    WHERE l.kind = 'summary'
    UNION ALL
    SELECT x.revision, l.seq + {_SEQ_POSITION - 2}
    FROM summary_revisions x
    JOIN latest_returns l ON l.kind = 'summary' AND l.derivation = x.derivation
    WHERE EXISTS (SELECT 1 FROM compaction_inputs i
                  JOIN effective_compactions e ON e.compaction_id = i.compaction
                  WHERE i.record = x.revision)
)
GROUP BY record;

CREATE VIEW messages AS
SELECT r.record_id AS store_id,
       r.session AS session_id,
       COALESCE((SELECT f.value FROM session_facts f
                 WHERE f.session = r.session AND f.kind = 'platform' AND f.at <= c.began_at
                 ORDER BY f.fact_id DESC LIMIT 1), '') AS source,
       '' AS conversation_id,
       r.role AS role,
       json_extract(r.raw, '$.content') AS content,
       r.tool_call_id AS tool_call_id,
       json_extract(r.raw, '$.tool_calls') AS tool_calls,
       json_extract(r.raw, '$.tool_name') AS tool_name,
       c.began_at AS timestamp,
       r.est_tokens AS token_estimate,
       r.est_uncounted_images AS uncounted_images,
       json_type(r.raw, '$.content') AS content_type,
       0 AS pinned,
       c.began_at AS ingested_at,
       CASE WHEN json_type(r.raw, '$.timestamp') IN ('integer', 'real')
            THEN json_extract(r.raw, '$.timestamp') END AS observed_at,
       CASE WHEN json_type(r.raw, '$.timestamp') IN ('integer', 'real')
            THEN 'host_message_timestamp' END AS observed_at_source,
       p.seq AS seq,
       (SELECT d.derivation_id FROM summary_revisions x JOIN derivations d ON d.handle = x.derivation
        WHERE x.revision = r.handle LIMIT 1) AS revises_node_id
FROM record_positions p
JOIN records r ON r.handle = p.record
JOIN compactions c ON c.compaction_id = r.compaction;

CREATE VIEW summary_nodes AS
SELECT d.derivation_id AS node_id,
       ch.session AS session_id,
       0 AS depth,
       d.text AS summary,
       d.est_tokens AS token_count,
       (SELECT SUM(r.est_tokens) FROM chunk_members m JOIN records r ON r.handle = m.record
        WHERE m.chunk = ch.handle) AS source_token_count,
       (SELECT COALESCE(SUM(r.est_uncounted_images), 0) FROM chunk_members m
        JOIN records r ON r.handle = m.record
        WHERE m.chunk = ch.handle) AS source_uncounted_images,
       (SELECT json_group_array(r.record_id ORDER BY m.ordinal)
        FROM chunk_members m JOIN records r ON r.handle = m.record
        WHERE m.chunk = ch.handle) AS source_ids,
       'messages' AS source_type,
       d.created_at AS created_at,
       (SELECT MIN(CASE WHEN json_type(r.raw, '$.timestamp') IN ('integer', 'real')
                        THEN json_extract(r.raw, '$.timestamp') END)
        FROM chunk_members m JOIN records r ON r.handle = m.record
        WHERE m.chunk = ch.handle) AS earliest_at,
       (SELECT MAX(CASE WHEN json_type(r.raw, '$.timestamp') IN ('integer', 'real')
                        THEN json_extract(r.raw, '$.timestamp') END)
        FROM chunk_members m JOIN records r ON r.handle = m.record
        WHERE m.chunk = ch.handle) AS latest_at,
       l.seq + {_SEQ_POSITION - 1} AS seq
FROM derivations d
JOIN latest_returns l ON l.kind = 'summary' AND l.derivation = d.handle
JOIN derivation_sources s ON s.derivation = d.handle AND s.ordinal = 0
JOIN chunks ch ON ch.handle = s.chunk;
"""

_SCHEMA_SQL = _RECORD_SQL + _insert_only_triggers_sql(INSERT_ONLY_TABLES) + "\n" + _VIEWS_SQL


def _is_sqlite_lock_error(exc: BaseException) -> bool:
    """Return True when an exception chain represents SQLite lock contention."""
    lock_codes = {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}
    lock_messages = (
        "database is locked",
        "database table is locked",
        "database schema is locked",
        "database is busy",
        "database table is busy",
        "database schema is busy",
    )
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, sqlite3.Error):
            error_code = getattr(current, "sqlite_errorcode", None)
            if isinstance(error_code, int) and (error_code & 0xFF) in lock_codes:
                return True
            detail = str(current).lower()
            if any(message in detail for message in lock_messages):
                return True
        current = current.__cause__ or current.__context__
    return False


# --- Journal mode -------------------------------------------------------------
#
# The store runs with a rollback journal (journal_mode=DELETE), in every process
# and with every SQLite version, so that no process ever switches the mode for
# the others. It is the plugin's own choice, never taken from the host's
# settings. WAL is not used: every SQLite from 3.7.0 through 3.51.2 (fixed in
# 3.51.3, backported to 3.50.7 and 3.44.6) can corrupt a WAL database when two or
# more connections in separate threads or processes checkpoint and commit
# concurrently (sqlite.org/wal.html#walresetbug), which is exactly how the Hermes
# processes of one home share this store. The store is written rarely, so it
# does not need WAL's concurrency.
#
# A rollback journal depends on POSIX advisory locks being honoured by every
# process that opens the file; SQLite names filesystems whose locking is broken
# as a cause of corruption (sqlite.org/howtocorrupt.html, 2.1). A store whose
# directory is on a filesystem shared across a VM boundary (virtiofs, 9p) is
# refused, because the plugin cannot establish that locks taken on one side of
# that boundary are seen on the other. The filesystem type is read from
# /proc/self/mountinfo, the way the host detects it.

_CROSS_VM_FSTYPES = frozenset({"virtiofs", "fuse.virtiofs", "9p", "9p2000", "9p2000.l", "9p2000.u"})
_MOUNTINFO_PATH = "/proc/self/mountinfo"


def _mountinfo_fstype(directory: str) -> str:
    """fstype of the longest mount point containing ``directory`` ("" if unknown)."""
    try:
        with open(_MOUNTINFO_PATH, "r", encoding="utf-8", errors="replace") as handle:
            lines = handle.read().splitlines()
    except OSError:
        return ""
    best_len, best_fstype = -1, ""
    for line in lines:
        fields, _, tail = line.partition(" - ")
        parts = fields.split()
        if len(parts) < 5 or not tail:
            continue
        mount_point = parts[4]
        if "\\" in mount_point:
            mount_point = mount_point.encode("latin-1", "ignore").decode("unicode_escape")
        if directory == mount_point or directory.startswith(mount_point.rstrip("/") + "/"):
            if len(mount_point) > best_len:
                best_len, best_fstype = len(mount_point), tail.split()[0]
    return best_fstype.lower()


def refuse_cross_vm_filesystem(db_path: str | Path) -> None:
    """Refuse a store whose directory lies on a virtiofs/9p mount."""
    if sys.platform != "linux":
        return
    directory = os.path.dirname(os.path.realpath(str(db_path))) or "/"
    fstype = _mountinfo_fstype(directory)
    if fstype in _CROSS_VM_FSTYPES:
        raise _refuse(
            db_path,
            f"its directory is on a {fstype} filesystem shared across a VM boundary, "
            f"where it is not established that SQLite's POSIX advisory locks are "
            f"honoured across processes on both sides, and a rollback journal "
            f"corrupts when they are not",
        )


def configure_connection(conn: sqlite3.Connection, db_path: str | Path) -> None:
    """Configure one store connection: rollback journal, verified, durable commits.

    - journal_mode=DELETE, and the mode SQLite reports is checked: a database
      that stays in another mode (for example WAL held by another connection) is
      refused, never used as if it were in rollback mode.
    - synchronous=FULL: fsync the database and the journal at every commit.
    - busy_timeout=30 s: writers wait for each other and for readers.
    - foreign_keys=ON: a reference in the record names a row that exists.
    - No memory-mapped I/O: SQLite documents that it needs a correct unified
      buffer cache when several processes share the file, and that an I/O error
      on a mapped file crashes the process instead of raising (sqlite.org/mmap.html).
    """
    conn.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}")
    refuse_cross_vm_filesystem(db_path)
    row = conn.execute("PRAGMA journal_mode=DELETE").fetchone()
    mode = str(row[0]).lower() if row and row[0] is not None else ""
    if mode != "delete":
        raise _refuse(db_path, f"SQLite kept journal_mode={mode or 'unknown'} when DELETE was requested")
    conn.execute("PRAGMA synchronous=FULL")
    conn.execute("PRAGMA foreign_keys=ON")


# --- Identity and creation ----------------------------------------------------

def _schema_object_names(conn: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE name NOT LIKE 'sqlite\\_%' ESCAPE '\\'"
        ).fetchall()
    }


def _identity_state(conn: sqlite3.Connection, db_path: str | Path) -> str:
    """Return ``"store"`` for this plugin's store, ``"empty"`` for an empty database.

    Anything else is refused before any of its rows is read.
    """
    names = _schema_object_names(conn)
    if not names:
        return "empty"
    if "store_identity" not in names:
        raise _refuse(db_path, "it has no store_identity row, so this plugin did not write it")
    rows = conn.execute("SELECT format FROM store_identity").fetchall()
    if len(rows) != 1:
        raise _refuse(db_path, f"it has {len(rows)} store_identity rows instead of one")
    found = str(rows[0][0] or "")
    if found != STORE_FORMAT:
        raise _refuse(db_path, f"its store format is {found!r}, this build writes {STORE_FORMAT!r}")
    return "store"


def _split_sql(script: str) -> list[str]:
    """Split the schema script into statements, keeping trigger bodies whole."""
    statements: list[str] = []
    current: list[str] = []
    for line in script.splitlines():
        if not line.strip():
            continue
        current.append(line)
        candidate = "\n".join(current)
        if sqlite3.complete_statement(candidate):
            statements.append(candidate)
            current = []
    if current:
        raise ValueError("incomplete statement in the schema script")
    return statements


def _create_store(conn: sqlite3.Connection, db_path: str | Path) -> None:
    """Create the whole schema and the identity row in one transaction.

    Several processes and engine copies can reach an empty database at once; the
    identity is checked again under the write lock, and only one creates.
    """
    conn.execute("BEGIN IMMEDIATE")
    try:
        if _identity_state(conn, db_path) == "store":
            conn.execute("COMMIT")
            return
        for statement in _split_sql(_SCHEMA_SQL):
            conn.execute(statement)
        for spec in FTS_SPECS:
            _create_fts_table(conn, spec)
            for trigger_sql in spec.trigger_sqls:
                conn.execute(trigger_sql)
        conn.execute(
            "INSERT INTO store_identity(format, store_uuid, created_at) VALUES (?, ?, ?)",
            (STORE_FORMAT, uuid.uuid4().hex, time.time()),
        )
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    logger.info("LCM created a new store at %s (format %s)", db_path, STORE_FORMAT)


def open_store(conn: sqlite3.Connection, db_path: str | Path, *, check_fts: bool = False) -> None:
    """Bind a fresh connection to the store at ``db_path``.

    Refuses a database this plugin did not write; creates the store in an empty
    database; configures the connection. With ``check_fts`` the full-text indexes
    are checked and repaired only when damaged.
    """
    conn.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}")
    state = _identity_state(conn, db_path)
    if state == "empty":
        _refuse_foreign_empty_file(db_path)
    configure_connection(conn, db_path)
    if state == "empty":
        _create_store(conn, db_path)
    if check_fts:
        for spec in FTS_SPECS:
            repaired = ensure_fts_intact(conn, spec)
            if repaired["rebuilt"] or repaired["triggers_recreated"]:
                logger.warning(
                    "LCM repaired the full-text index %s at %s: triggers_recreated=%s; "
                    "the index was rebuilt from the stored rows",
                    spec.table_name,
                    db_path,
                    repaired["triggers_recreated"],
                )


def _refuse_foreign_empty_file(db_path: str | Path) -> None:
    """Refuse an empty file that the plugin did not create.

    The plugin creates every store file with mode 0600. An empty file whose mode
    gives group or others any permission was made by someone else, so the plugin
    does not adopt it.
    """
    try:
        mode = os.stat(str(db_path)).st_mode
    except OSError as exc:
        raise _refuse(db_path, f"its mode cannot be read ({exc})") from exc
    if mode & 0o077:
        raise _refuse(
            db_path,
            f"it is an empty file with mode {mode & 0o777:04o}; the plugin creates its "
            f"store files with mode 0600, so this file is someone else's",
        )


# --- Diagnostics --------------------------------------------------------------

def get_existing_table_names(conn: sqlite3.Connection, names: Iterable[str]) -> set[str]:
    existing: set[str] = set()
    for name in names:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name = ?",
            (name,),
        ).fetchone()
        if row and row[0]:
            existing.add(row[0])
    return existing


def _database_path_for_connection(conn: sqlite3.Connection | None, fallback: str = "") -> str:
    if conn is None:
        return fallback
    try:
        rows = conn.execute("PRAGMA database_list").fetchall()
    except sqlite3.DatabaseError:
        return fallback
    for row in rows:
        if len(row) >= 3 and row[1] == "main" and row[2]:
            return str(row[2])
    return fallback


def inspect_lcm_schema_health(
    conn: sqlite3.Connection | None,
    *,
    database_path: str = "",
    required_tables: Iterable[str] = REQUIRED_CORE_TABLES,
) -> dict[str, object]:
    """Return read-only health metadata for the core hermes-lcm SQLite schema."""
    required = tuple(required_tables)
    resolved_path = _database_path_for_connection(conn, database_path)
    detail: dict[str, object] = {
        "database_path": resolved_path,
        "required_tables": list(required),
        "existing_tables": [],
        "missing_tables": [],
    }
    if conn is None:
        detail["error"] = "LCM store connection is not initialized"
        return detail

    try:
        rows = conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type='table'
            ORDER BY name
            """
        ).fetchall()
    except sqlite3.DatabaseError as exc:
        detail["error"] = str(exc)
        return detail

    existing = sorted(str(row[0]) for row in rows if row and row[0])
    existing_set = set(existing)
    missing = [name for name in required if name not in existing_set]
    detail["existing_tables"] = existing
    detail["missing_tables"] = missing
    return detail


# --- Full-text indexes --------------------------------------------------------

def get_fts_shadow_table_names(table_name: str) -> list[str]:
    return [
        f"{table_name}_data",
        f"{table_name}_idx",
        f"{table_name}_docsize",
        f"{table_name}_config",
    ]


def quote_sql_identifier(identifier: str) -> str:
    if not identifier or not identifier.replace("_", "a").isalnum() or identifier[0].isdigit():
        raise ValueError(f"invalid SQL identifier: {identifier}")
    return f'"{identifier}"'


def _create_fts_table(conn: sqlite3.Connection, spec: ExternalContentFtsSpec) -> None:
    conn.execute(
        f"""
        CREATE VIRTUAL TABLE {quote_sql_identifier(spec.table_name)} USING fts5(
            {quote_sql_identifier(spec.indexed_column)},
            content={quote_sql_identifier(spec.content_table)},
            content_rowid={quote_sql_identifier(spec.content_rowid)}
        )
        """
    )


def _fts_needs_rebuild_structural(conn: sqlite3.Connection, spec: ExternalContentFtsSpec) -> bool:
    shadow_tables = get_fts_shadow_table_names(spec.table_name)
    existing_tables = get_existing_table_names(conn, [spec.table_name, *shadow_tables])
    if spec.table_name not in existing_tables:
        return True
    if any(name not in existing_tables for name in shadow_tables):
        return True

    try:
        info = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name = ?",
            (spec.table_name,),
        ).fetchone()
        sql = (info[0] if info else "") or ""
        normalized = sql.lower()
        if "virtual table" not in normalized or "using fts5" not in normalized:
            return True

        columns = conn.execute(
            f"PRAGMA table_info({quote_sql_identifier(spec.table_name)})"
        ).fetchall()
        column_names = {row[1] for row in columns if len(row) > 1}
        if spec.indexed_column not in column_names:
            return True

        content_count = conn.execute(
            f"SELECT COUNT(*) FROM {quote_sql_identifier(spec.content_table)}"
        ).fetchone()[0]
        # For an external-content FTS5 table, ``COUNT(*) FROM <fts>`` reads
        # through to the content table; the ``<fts>_docsize`` shadow table holds
        # the true indexed-document count.
        docsize_table = f"{spec.table_name}_docsize"
        fts_count = conn.execute(
            f"SELECT COUNT(*) FROM {quote_sql_identifier(docsize_table)}"
        ).fetchone()[0]
        if int(content_count or 0) != int(fts_count or 0):
            return True
    except sqlite3.DatabaseError as exc:
        # A busy/locked snapshot is an availability problem, not FTS corruption.
        if _is_sqlite_lock_error(exc):
            raise
        return True

    return False


# SQLite/FTS5 error substrings that denote genuine corruption or index drift.
# Everything else a writable integrity-check can raise (locks, timeouts) is
# transient and classifies as ``unchecked``, never ``fail``.
_FTS_CORRUPTION_SIGNATURES = (
    "malformed",
    "disk image",
    "not a database",
    "corrupt",
    "checksum mismatch",
)


def _is_fts_corruption_error(detail: str) -> bool:
    lowered = detail.lower()
    return any(signature in lowered for signature in _FTS_CORRUPTION_SIGNATURES)


def check_external_content_fts_integrity(
    conn: sqlite3.Connection,
    spec: ExternalContentFtsSpec,
) -> dict[str, str]:
    """Run SQLite's FTS5 integrity-check for an external-content table.

    FTS5 exposes this as a special INSERT command. It is wrapped in a savepoint
    and rolled back so the check leaves no state behind on the connection.
    """

    if _fts_needs_rebuild_structural(conn, spec):
        return {"status": "fail", "detail": "structural repair needed"}

    savepoint = f"lcm_fts_integrity_{spec.table_name}"
    savepoint_sql = quote_sql_identifier(savepoint)
    try:
        conn.execute(f"SAVEPOINT {savepoint_sql}")
        conn.execute(
            f"INSERT INTO {quote_sql_identifier(spec.table_name)}({quote_sql_identifier(spec.table_name)}, rank) VALUES('integrity-check', 1)"
        )
    except sqlite3.DatabaseError as exc:
        try:
            conn.execute(f"ROLLBACK TO {savepoint_sql}")
            conn.execute(f"RELEASE {savepoint_sql}")
        except sqlite3.DatabaseError:
            pass
        detail = str(exc)
        lowered = detail.lower()
        if "readonly" in lowered or "read-only" in lowered:
            return {"status": "unchecked", "detail": detail}
        if _is_fts_corruption_error(detail):
            return {"status": "fail", "detail": detail}
        return {"status": "unchecked", "detail": detail}

    try:
        conn.execute(f"ROLLBACK TO {savepoint_sql}")
        conn.execute(f"RELEASE {savepoint_sql}")
    except sqlite3.DatabaseError as exc:
        return {"status": "fail", "detail": str(exc)}

    return {"status": "pass", "detail": "ok"}


def _drop_fts_table(conn: sqlite3.Connection, table_name: str) -> None:
    conn.execute(f"DROP TABLE IF EXISTS {quote_sql_identifier(table_name)}")
    for shadow_name in get_fts_shadow_table_names(table_name):
        conn.execute(f"DROP TABLE IF EXISTS {quote_sql_identifier(shadow_name)}")


def _extract_trigger_name(trigger_sql: str) -> str | None:
    match = re.search(
        r"CREATE\s+TRIGGER\s+(?:IF\s+NOT\s+EXISTS\s+)?(?:\"([^\"]+)\"|([A-Za-z_][A-Za-z0-9_]*))",
        trigger_sql,
        re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return None
    return match.group(1) or match.group(2)


def _normalize_trigger_sql(sql: str) -> str:
    """Normalise trigger SQL for comparison with what SQLite keeps in sqlite_master.

    Case, whitespace, ``IF NOT EXISTS`` and a trailing semicolon are not
    significant.
    """
    text = re.sub(r"\bif\s+not\s+exists\b", " ", sql, flags=re.IGNORECASE)
    return " ".join(text.split()).strip().rstrip(";").strip().lower()


def _fts_stale_triggers(conn: sqlite3.Connection, spec: ExternalContentFtsSpec) -> list[str]:
    """Names of the spec's triggers that are missing or whose SQL differs from the spec."""
    stale: list[str] = []
    for trigger_sql in spec.trigger_sqls:
        name = _extract_trigger_name(trigger_sql)
        if not name:
            continue
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='trigger' AND name = ?",
            (name,),
        ).fetchone()
        if not row or _normalize_trigger_sql(row[0] or "") != _normalize_trigger_sql(trigger_sql):
            stale.append(name)
    return stale


def external_content_fts_needs_repair(conn: sqlite3.Connection, spec: ExternalContentFtsSpec) -> bool:
    return _fts_needs_rebuild_structural(conn, spec) or bool(_fts_stale_triggers(conn, spec))


@contextmanager
def _fts_repair_ownership(conn: sqlite3.Connection):
    """Own the short FTS repair transaction.

    ``BEGIN IMMEDIATE`` serialises the recheck and the repair across processes.
    A caller that already owns a transaction gets a savepoint instead.
    """
    if conn.in_transaction:
        savepoint = quote_sql_identifier("lcm_fts_repair_ownership")
        conn.execute(f"SAVEPOINT {savepoint}")
        try:
            yield
        except BaseException:
            try:
                conn.execute(f"ROLLBACK TO {savepoint}")
            finally:
                conn.execute(f"RELEASE {savepoint}")
            raise
        else:
            conn.execute(f"RELEASE {savepoint}")
        return

    conn.execute("BEGIN IMMEDIATE")
    try:
        yield
    except BaseException:
        conn.rollback()
        raise
    else:
        conn.commit()


def _repair_fts(
    conn: sqlite3.Connection,
    spec: ExternalContentFtsSpec,
    *,
    force_rebuild: bool = False,
) -> dict[str, bool]:
    """Recreate stale triggers and rebuild the index, under the write lock.

    A trigger that drifted may already have indexed text other than the stored
    rows, so recreating triggers always rebuilds the index from the content table
    in the same transaction. A structurally damaged index is dropped and created
    again first. Another process may have repaired it while this one waited, so
    the state is checked again once the write lock is held.
    """
    rebuilt = False
    table = quote_sql_identifier(spec.table_name)
    with _fts_repair_ownership(conn):
        structural = force_rebuild or _fts_needs_rebuild_structural(conn, spec)
        stale = _fts_stale_triggers(conn, spec)
        if structural:
            _drop_fts_table(conn, spec.table_name)
            _create_fts_table(conn, spec)
        for trigger_sql in spec.trigger_sqls:
            name = _extract_trigger_name(trigger_sql)
            if name in stale:
                conn.execute(f"DROP TRIGGER IF EXISTS {quote_sql_identifier(name)}")
                conn.execute(trigger_sql)
        if structural or stale:
            conn.execute(f"INSERT INTO {table}({table}) VALUES('rebuild')")
            rebuilt = True
    return {"rebuilt": rebuilt, "degraded": False, "triggers_recreated": bool(stale)}


def ensure_fts_intact(conn: sqlite3.Connection, spec: ExternalContentFtsSpec) -> dict[str, bool]:
    """Cheap startup check: repair only a structurally damaged index or stale triggers."""
    if not external_content_fts_needs_repair(conn, spec):
        return {"rebuilt": False, "degraded": False, "triggers_recreated": False}
    return _repair_fts(conn, spec)


def repair_external_content_fts(conn: sqlite3.Connection, spec: ExternalContentFtsSpec) -> dict[str, bool]:
    """Explicit repair: also rebuild when the deep FTS5 integrity-check fails."""
    deep = check_external_content_fts_integrity(conn, spec)
    return _repair_fts(conn, spec, force_rebuild=deep.get("status") == "fail")
