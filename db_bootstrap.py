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


# The layout this build writes. A store with any other format is refused.
STORE_FORMAT = "ihl-store/1"
# The default file name under the host-given Hermes home.
STORE_FILENAME = "lcm-record.db"
SQLITE_BUSY_TIMEOUT_MS = 30_000
REQUIRED_CORE_TABLES = (
    "store_identity",
    "sessions",
    "session_facts",
    "messages",
    "metadata",
    "summary_nodes",
    "lcm_lifecycle_state",
    "messages_fts",
    "nodes_fts",
)


class StoreRefusedError(RuntimeError):
    """The database at a path is not a store this plugin can open.

    Raised before anything in the database is read beyond its table names, or
    before a store is created where it cannot be kept safely. The message names the
    path and the reason.
    """


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


MESSAGES_FTS_SPEC = ExternalContentFtsSpec(
    table_name="messages_fts",
    content_table="messages",
    content_rowid="store_id",
    indexed_column="content",
    trigger_sqls=(
        """
        CREATE TRIGGER IF NOT EXISTS msg_fts_insert
            AFTER INSERT ON messages BEGIN
            INSERT INTO messages_fts(rowid, content)
                VALUES (new.store_id, new.content);
        END;
        """,
        """
        CREATE TRIGGER IF NOT EXISTS msg_fts_delete
            AFTER DELETE ON messages BEGIN
            INSERT INTO messages_fts(messages_fts, rowid, content)
                VALUES('delete', old.store_id, old.content);
        END;
        """,
        """
        CREATE TRIGGER IF NOT EXISTS msg_fts_update
            AFTER UPDATE OF content ON messages BEGIN
            INSERT INTO messages_fts(messages_fts, rowid, content)
                VALUES('delete', old.store_id, old.content);
            INSERT INTO messages_fts(rowid, content)
                VALUES (new.store_id, new.content);
        END;
        """,
    ),
)

NODES_FTS_SPEC = ExternalContentFtsSpec(
    table_name="nodes_fts",
    content_table="summary_nodes",
    content_rowid="node_id",
    indexed_column="summary",
    trigger_sqls=(
        """
        CREATE TRIGGER IF NOT EXISTS nodes_fts_insert
            AFTER INSERT ON summary_nodes BEGIN
            INSERT INTO nodes_fts(rowid, summary)
                VALUES (new.node_id, new.summary);
        END;
        """,
        """
        CREATE TRIGGER IF NOT EXISTS nodes_fts_delete
            AFTER DELETE ON summary_nodes BEGIN
            INSERT INTO nodes_fts(nodes_fts, rowid, summary)
                VALUES('delete', old.node_id, old.summary);
        END;
        """,
    ),
)

FTS_SPECS = (MESSAGES_FTS_SPEC, NODES_FTS_SPEC)


# The tables of the plugin's own record. Each is insert-only: triggers raise on any
# UPDATE or DELETE, so that append-only is a property of the database (#29, W1).
INSERT_ONLY_TABLES = ("store_identity", "sessions", "session_facts")


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
"""

_SCHEMA_SQL = _RECORD_SQL + _insert_only_triggers_sql(INSERT_ONLY_TABLES) + """

CREATE TABLE messages (
    store_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    source TEXT DEFAULT '',
    conversation_id TEXT DEFAULT '',
    role TEXT NOT NULL,
    content TEXT,
    tool_call_id TEXT,
    tool_calls TEXT,
    tool_name TEXT,
    timestamp REAL NOT NULL,
    token_estimate INTEGER DEFAULT 0,
    pinned INTEGER DEFAULT 0,
    ingested_at REAL,
    observed_at REAL,
    observed_at_source TEXT
);
CREATE INDEX idx_msg_session ON messages(session_id, store_id);
CREATE INDEX idx_msg_session_ts ON messages(session_id, timestamp);
CREATE INDEX idx_msg_source_session ON messages(source, session_id, store_id);
CREATE INDEX idx_msg_conversation_session ON messages(conversation_id, session_id, store_id);

CREATE TABLE metadata (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE summary_nodes (
    node_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    depth INTEGER NOT NULL DEFAULT 0,
    summary TEXT NOT NULL,
    token_count INTEGER DEFAULT 0,
    source_token_count INTEGER DEFAULT 0,
    source_ids TEXT NOT NULL DEFAULT '[]',
    source_type TEXT NOT NULL DEFAULT 'messages',
    created_at REAL NOT NULL,
    earliest_at REAL,
    latest_at REAL,
    expand_hint TEXT DEFAULT ''
);
CREATE INDEX idx_nodes_session_depth ON summary_nodes(session_id, depth, created_at);
CREATE INDEX idx_nodes_session_node ON summary_nodes(session_id, node_id);
CREATE INDEX idx_nodes_session_depth_node ON summary_nodes(session_id, depth, node_id);
CREATE INDEX idx_nodes_session_latest ON summary_nodes(session_id, latest_at, created_at);

CREATE TABLE lcm_lifecycle_state (
    conversation_id TEXT PRIMARY KEY,
    current_session_id TEXT,
    last_finalized_session_id TEXT,
    current_frontier_store_id INTEGER NOT NULL DEFAULT 0,
    last_finalized_frontier_store_id INTEGER NOT NULL DEFAULT 0,
    debt_kind TEXT,
    debt_size_estimate INTEGER NOT NULL DEFAULT 0,
    current_bound_at REAL,
    last_finalized_at REAL,
    debt_updated_at REAL,
    last_maintenance_attempt_at REAL,
    last_rollover_at REAL,
    last_reset_at REAL,
    updated_at REAL NOT NULL DEFAULT (strftime('%s','now'))
);
CREATE INDEX idx_lcm_lifecycle_current_session ON lcm_lifecycle_state(current_session_id);
CREATE INDEX idx_lcm_lifecycle_last_finalized_session ON lcm_lifecycle_state(last_finalized_session_id);
"""


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
