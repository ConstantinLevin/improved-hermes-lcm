"""Shared SQLite bootstrap helpers for hermes-lcm.

This module keeps startup DB initialization in one place so store/DAG use the
same schema-version marker, PRAGMA settings, and FTS repair behavior.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import shutil
import sqlite3
import threading
import time
from contextlib import contextmanager
from typing import Iterable, Sequence

logger = logging.getLogger(__name__)


class SchemaVersionTooNewError(RuntimeError):
    """Raised when a database was written by a newer LCM schema than this build.

    Opening and migrating such a database with older code risks silently
    corrupting data written under semantics this build does not understand, so
    we refuse rather than degrade.
    """


# The core schema ladder stops at 5.
SCHEMA_VERSION = 5
SQLITE_BUSY_TIMEOUT_MS = 30_000
_MIN_DISK_SPACE_BYTES = 50 * 1024 * 1024
REQUIRED_CORE_TABLES = (
    "messages",
    "metadata",
    "summary_nodes",
    "lcm_lifecycle_state",
    "lcm_migration_state",
    "messages_fts",
    "nodes_fts",
)


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


def configure_connection(conn: sqlite3.Connection) -> None:
    """Configure SQLite connection for WAL durability and hygiene.

    In a multi-agent deployment (gateway process + CLI sessions + sub-agents),
    every process opens its own sqlite3.Connection pointing at the same
    lcm.db file.  These settings improve committed-write durability and WAL
    hygiene, but do NOT make sibling processes safe from an unexpected process
    death.  Abnormal exit still depends on normal SQLite WAL recovery;
    application-level checkpoints only run during graceful shutdown (see
    ``MessageStore.close()`` etc.).

    Key design decisions:
    - journal_mode=WAL  : writes go to a separate log; readers never block.
    - synchronous=FULL  : fsync both the WAL and the WAL index before every
                          write transaction commit.  WAL + FULL is the only
                          combination SQLite guarantees survives power loss
                          without data loss (NORMAL may lose the WAL index).
    - wal_autocheckpoint=500 : after 500 WAL pages (~2 MB) SQLite will try
                               an automatic passive checkpoint.  This is a
                               best-effort hint — it is silently skipped when
                               another connection holds a read transaction.
                               Under checkpoint starvation WAL can grow well
                               beyond this trigger.
    - journal_size_limit=67108864 (64 MiB) : limits the WAL file size after
                                             a successful checkpoint or reset.
                                             It does NOT force a checkpoint
                                             or cap growth while another
                                             connection holds an old WAL
                                             end mark.
    - mmap_size=268435456 (256 MiB)        : memory-map reads so concurrent
                                              readers cache WAL pages in RAM.
    """
    conn.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}")
    _execute_wal_conversion_with_lock_retry(conn)
    conn.execute("PRAGMA synchronous=FULL")
    conn.execute("PRAGMA wal_autocheckpoint=500")
    conn.execute("PRAGMA journal_size_limit=67108864")
    conn.execute("PRAGMA mmap_size=268435456")


def _execute_wal_conversion_with_lock_retry(
    conn: sqlite3.Connection,
    *,
    budget_ms: int = SQLITE_BUSY_TIMEOUT_MS,
) -> None:
    """Run ``PRAGMA journal_mode=WAL`` with a bounded lock-contention retry.

    Converting a rollback-journal database to WAL needs the exclusive lock,
    and SQLite can return ``SQLITE_BUSY`` for that upgrade without consulting
    the busy handler when other connections are mid-setup on the same file.
    Concurrent process startup (gateway + CLI + sub-agents) on a not-yet-WAL
    database therefore crashed sporadically with ``database is locked``.
    Once the database is in WAL mode the pragma is a plain read and never
    takes this path, so the retry only matters on first boot after an
    install/upgrade or on a rollback-journal restore.
    """
    deadline = time.monotonic() + budget_ms / 1000.0
    delay_seconds = 0.005
    while True:
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            return
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc).lower() or time.monotonic() >= deadline:
                raise
        time.sleep(delay_seconds)
        delay_seconds = min(delay_seconds * 2, 0.25)


def add_column_if_missing(
    conn: sqlite3.Connection,
    existing_columns: set[str],
    column: str,
    alter_sql: str,
) -> None:
    """Idempotently add a column, tolerating a concurrent process that won the race.

    In the multi-agent deployment (gateway + CLI sessions + sub-agents) every
    process opens its own connection to the same ``lcm.db`` and runs startup
    migrations concurrently.  A plain check-``PRAGMA table_info``-then-``ALTER``
    races: two processes both observe the column as absent (each within its own
    connection snapshot) and both issue ``ALTER TABLE ... ADD COLUMN``.  The loser
    then raised ``sqlite3.OperationalError: duplicate column name``, which
    propagated out of ``_init_db`` and crashed store construction.  Swallowing
    exactly that error makes the migration idempotent under concurrency; any other
    OperationalError still propagates.
    """
    if column in existing_columns:
        return
    try:
        conn.execute(alter_sql)
    except sqlite3.OperationalError as exc:
        if "duplicate column name" not in str(exc).lower():
            raise


def ensure_metadata_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        )
        """
    )


def get_schema_version(conn: sqlite3.Connection) -> int:
    ensure_metadata_table(conn)
    row = conn.execute(
        "SELECT value FROM metadata WHERE key = 'schema_version'"
    ).fetchone()
    if not row or row[0] is None:
        return 0
    try:
        return int(str(row[0]))
    except (TypeError, ValueError):
        return 0


def read_existing_schema_version(conn: sqlite3.Connection) -> int:
    """Return schema_version without creating or modifying schema objects."""
    metadata_exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='metadata'"
    ).fetchone()
    if not metadata_exists:
        return 0
    row = conn.execute(
        "SELECT value FROM metadata WHERE key = 'schema_version'"
    ).fetchone()
    if not row or row[0] is None:
        return 0
    try:
        return int(str(row[0]))
    except (TypeError, ValueError):
        return 0


def refuse_schema_version_too_new(conn: sqlite3.Connection) -> None:
    """Raise before any startup DDL when a newer build owns the DB."""
    current_version = read_existing_schema_version(conn)
    if current_version <= SCHEMA_VERSION:
        return
    raise SchemaVersionTooNewError(
        f"LCM database schema version {current_version} is newer than this "
        f"build supports (v{SCHEMA_VERSION}). Refusing to open to avoid "
        f"corrupting data written by a newer hermes-lcm. Upgrade the plugin "
        f"or restore a pre-upgrade backup (.db/-wal/-shm)."
    )


def ensure_migration_state_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS lcm_migration_state (
            step_name TEXT PRIMARY KEY,
            completed_at REAL NOT NULL
        )
        """
    )


def ensure_lifecycle_state_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS lcm_lifecycle_state (
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
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_lcm_lifecycle_current_session ON lcm_lifecycle_state(current_session_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_lcm_lifecycle_last_finalized_session ON lcm_lifecycle_state(last_finalized_session_id)"
    )


def ensure_lifecycle_state_columns(conn: sqlite3.Connection) -> None:
    ensure_lifecycle_state_table(conn)
    columns = {
        row[1] for row in conn.execute("PRAGMA table_info(lcm_lifecycle_state)").fetchall()
    }
    add_column_if_missing(
        conn, columns, "debt_kind",
        "ALTER TABLE lcm_lifecycle_state ADD COLUMN debt_kind TEXT",
    )
    add_column_if_missing(
        conn, columns, "debt_size_estimate",
        "ALTER TABLE lcm_lifecycle_state ADD COLUMN debt_size_estimate INTEGER NOT NULL DEFAULT 0",
    )
    add_column_if_missing(
        conn, columns, "debt_updated_at",
        "ALTER TABLE lcm_lifecycle_state ADD COLUMN debt_updated_at REAL",
    )
    add_column_if_missing(
        conn, columns, "last_maintenance_attempt_at",
        "ALTER TABLE lcm_lifecycle_state ADD COLUMN last_maintenance_attempt_at REAL",
    )
    add_column_if_missing(
        conn, columns, "last_rollover_at",
        "ALTER TABLE lcm_lifecycle_state ADD COLUMN last_rollover_at REAL",
    )
    add_column_if_missing(
        conn, columns, "last_reset_at",
        "ALTER TABLE lcm_lifecycle_state ADD COLUMN last_reset_at REAL",
    )


def ensure_message_origin_columns(conn: sqlite3.Connection) -> None:
    table_row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='messages'"
    ).fetchone()
    if not table_row:
        return
    columns = {
        row[1] for row in conn.execute("PRAGMA table_info(messages)").fetchall()
    }
    add_column_if_missing(
        conn, columns, "conversation_id",
        "ALTER TABLE messages ADD COLUMN conversation_id TEXT DEFAULT ''",
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_msg_conversation_session ON messages(conversation_id, session_id, store_id)"
    )


def mark_migration_step_complete(conn: sqlite3.Connection, step_name: str) -> None:
    ensure_migration_state_table(conn)
    conn.execute(
        """
        INSERT INTO lcm_migration_state(step_name, completed_at)
        VALUES(?, strftime('%s','now'))
        ON CONFLICT(step_name) DO UPDATE SET completed_at = excluded.completed_at
        """,
        (step_name,),
    )


def set_schema_version(conn: sqlite3.Connection, version: int = SCHEMA_VERSION) -> None:
    ensure_metadata_table(conn)
    conn.execute(
        """
        INSERT INTO metadata(key, value)
        VALUES('schema_version', ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (str(version),),
    )


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
        # through to the content table (so it can never reveal a lagging index)
        # and is O(index size). The ``<fts>_docsize`` shadow table holds the
        # true indexed-document count and is a cheap ordinary-table count. Its
        # existence is already guaranteed by the shadow-table check above.
        docsize_table = f"{spec.table_name}_docsize"
        fts_count = conn.execute(
            f"SELECT COUNT(*) FROM {quote_sql_identifier(docsize_table)}"
        ).fetchone()[0]
        if int(content_count or 0) != int(fts_count or 0):
            return True
    except sqlite3.DatabaseError as exc:
        # A busy/locked snapshot is an availability problem, not FTS
        # corruption. Let the bounded caller transaction report it instead of
        # turning lock exhaustion into destructive repair.
        if _is_sqlite_lock_error(exc):
            raise
        return True

    return False


INTEGRITY_CHECK_INTERVAL_ENV = "LCM_FTS_INTEGRITY_CHECK_INTERVAL_HOURS"
DEFAULT_INTEGRITY_CHECK_INTERVAL_HOURS = 24.0


def _integrity_check_interval_hours() -> float:
    """Hours between startup FTS deep integrity-checks.

    ``0`` checks on every startup (previous behavior); a negative value never
    checks on startup (relies on structural checks + LIKE fallback + doctor).
    """
    raw = os.environ.get(INTEGRITY_CHECK_INTERVAL_ENV)
    if raw is None:
        return DEFAULT_INTEGRITY_CHECK_INTERVAL_HOURS
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_INTEGRITY_CHECK_INTERVAL_HOURS
    if not math.isfinite(value):
        # nan/inf would suppress startup checks indefinitely once a marker
        # exists; treat non-finite values as invalid.
        return DEFAULT_INTEGRITY_CHECK_INTERVAL_HOURS
    return value


def _integrity_marker_key(spec: ExternalContentFtsSpec) -> str:
    return f"fts_integrity_checked_at:{spec.table_name}"


def _load_integrity_checked_at(
    conn: sqlite3.Connection, spec: ExternalContentFtsSpec
) -> float | None:
    ensure_metadata_table(conn)
    row = conn.execute(
        "SELECT value FROM metadata WHERE key = ?",
        (_integrity_marker_key(spec),),
    ).fetchone()
    if not row or row[0] is None:
        return None
    try:
        return float(row[0])
    except (TypeError, ValueError):
        return None


def _record_integrity_checked(
    conn: sqlite3.Connection, spec: ExternalContentFtsSpec, *, now: float | None = None
) -> None:
    ensure_metadata_table(conn)
    current = time.time() if now is None else now
    conn.execute(
        """
        INSERT INTO metadata(key, value)
        VALUES(?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (_integrity_marker_key(spec), str(current)),
    )


def _should_run_integrity_check(
    conn: sqlite3.Connection, spec: ExternalContentFtsSpec, *, now: float | None = None
) -> bool:
    hours = _integrity_check_interval_hours()
    if hours == 0:
        return True
    if hours < 0:
        return False
    last = _load_integrity_checked_at(conn, spec)
    if last is None:
        return True
    current = time.time() if now is None else now
    return (current - last) >= hours * 3600.0


# --- Non-blocking startup integrity scan (issue #6 / #235) -----------------
#
# Even throttled, the O(index-size) FTS5 deep integrity-check still blocked the
# bind on every cache-miss (first bind + each interval expiry): ~2min on a cold
# production DB. When a deep check is due on the startup path we now run only the
# cheap structural check synchronously and dispatch the deep scan to a daemon
# thread that opens its OWN sqlite connection (never the store's — that
# connection is not safe to drive from another thread). The background scan does
# NOT rebuild: on corruption it records a ``fts_integrity_failed:<table>`` marker
# that ``/lcm doctor`` surfaces, pointing operators at the explicit repair path.

BACKGROUND_INTEGRITY_ENV = "LCM_FTS_INTEGRITY_BACKGROUND"

# A ``fts_integrity_scan_started_at`` metadata stamp older than this (seconds) is
# treated as a crashed scan, so a later bind re-dispatches instead of wedging
# forever behind a stamp no live thread will ever clear.
INTEGRITY_SCAN_STALE_SECONDS = 15 * 60.0

# Guards the in-process registry and the one-scan-at-a-time decision below.
_integrity_scan_lock = threading.Lock()
# (db_path, table_name) -> daemon Thread. Exposed so tests can join a dispatched
# scan deterministically; entries are removed when the scan thread exits.
_integrity_scan_threads: dict[tuple[str, str], threading.Thread] = {}


def _background_integrity_enabled() -> bool:
    """Kill-switch: ``LCM_FTS_INTEGRITY_BACKGROUND=false`` restores the exact old
    synchronous integrity-check behavior on the startup path."""
    raw = os.environ.get(BACKGROUND_INTEGRITY_ENV)
    if raw is None:
        return True
    return raw.strip().lower() not in ("0", "false", "no", "off")


def _integrity_failed_key(spec: ExternalContentFtsSpec) -> str:
    return f"fts_integrity_failed:{spec.table_name}"


def _integrity_scan_started_key(spec: ExternalContentFtsSpec) -> str:
    return f"fts_integrity_scan_started_at:{spec.table_name}"


def _record_integrity_failed(
    conn: sqlite3.Connection,
    spec: ExternalContentFtsSpec,
    *,
    detail: str,
    now: float | None = None,
) -> None:
    ensure_metadata_table(conn)
    current = time.time() if now is None else now
    payload = json.dumps({"at": current, "detail": str(detail)[:2000]})
    conn.execute(
        """
        INSERT INTO metadata(key, value)
        VALUES(?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (_integrity_failed_key(spec), payload),
    )


def _clear_integrity_failed(conn: sqlite3.Connection, spec: ExternalContentFtsSpec) -> None:
    ensure_metadata_table(conn)
    conn.execute("DELETE FROM metadata WHERE key = ?", (_integrity_failed_key(spec),))


def load_integrity_failed(
    conn: sqlite3.Connection, spec: ExternalContentFtsSpec
) -> dict[str, object] | None:
    """Return ``{'at': float, 'detail': str}`` when a background scan flagged the
    index as corrupt, else ``None``. Used by ``/lcm doctor`` to surface the flag."""
    ensure_metadata_table(conn)
    row = conn.execute(
        "SELECT value FROM metadata WHERE key = ?",
        (_integrity_failed_key(spec),),
    ).fetchone()
    if not row or row[0] is None:
        return None
    try:
        data = json.loads(row[0])
        if isinstance(data, dict):
            return {"at": float(data.get("at") or 0.0), "detail": str(data.get("detail") or "")}
    except (TypeError, ValueError):
        pass
    return {"at": 0.0, "detail": str(row[0])}


def _load_scan_started_at(
    conn: sqlite3.Connection, spec: ExternalContentFtsSpec
) -> float | None:
    ensure_metadata_table(conn)
    row = conn.execute(
        "SELECT value FROM metadata WHERE key = ?",
        (_integrity_scan_started_key(spec),),
    ).fetchone()
    if not row or row[0] is None:
        return None
    try:
        return float(row[0])
    except (TypeError, ValueError):
        return None


def _record_scan_started(
    conn: sqlite3.Connection, spec: ExternalContentFtsSpec, *, now: float
) -> None:
    ensure_metadata_table(conn)
    conn.execute(
        """
        INSERT INTO metadata(key, value)
        VALUES(?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (_integrity_scan_started_key(spec), str(now)),
    )


def _clear_scan_started(
    conn: sqlite3.Connection,
    spec: ExternalContentFtsSpec,
    *,
    expected: float | None = None,
) -> None:
    ensure_metadata_table(conn)
    if expected is None:
        conn.execute(
            "DELETE FROM metadata WHERE key = ?", (_integrity_scan_started_key(spec),)
        )
    else:
        # Only clear our own stamp so a newer scan's stamp survives.
        conn.execute(
            "DELETE FROM metadata WHERE key = ? AND value = ?",
            (_integrity_scan_started_key(spec), str(expected)),
        )


def _run_background_integrity_scan(
    db_path: str, spec: ExternalContentFtsSpec, started_at: float
) -> None:
    """Daemon-thread body: deep-check ``spec`` on a private connection.

    Opens its own read/write connection for the scan (the FTS5 integrity-check is
    issued as an INSERT command that rolls back inside a savepoint, so it never
    mutates data, but it does require a writable handle) and a separate brief
    connection to stamp the result. On corruption it flags rather than rebuilds.
    """
    key = (db_path, spec.table_name)
    timeout = SQLITE_BUSY_TIMEOUT_MS / 1000.0
    try:
        scan_conn = sqlite3.connect(db_path, timeout=timeout, check_same_thread=False)
        try:
            scan_conn.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}")
            # Persist the scan-started stamp on this DB so a crash mid-scan is
            # detectable cross-process via the staleness window above.
            _record_scan_started(scan_conn, spec, now=started_at)
            scan_conn.commit()
            result = check_external_content_fts_integrity(scan_conn, spec)
        finally:
            scan_conn.close()

        meta_conn = sqlite3.connect(db_path, timeout=timeout, check_same_thread=False)
        try:
            meta_conn.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}")
            status = result.get("status")
            if status == "pass":
                _record_integrity_checked(meta_conn, spec, now=started_at)
                _clear_integrity_failed(meta_conn, spec)
            elif status == "fail":
                _record_integrity_failed(
                    meta_conn, spec, detail=result.get("detail", ""), now=started_at
                )
                logger.warning(
                    "Background FTS integrity-check found corruption in '%s': %s. "
                    "Run `/lcm doctor repair apply` to rebuild the index.",
                    spec.table_name,
                    result.get("detail", ""),
                )
            # 'unchecked' (e.g. a read-only DB): leave the throttle marker unset
            # so the next bind retries; do not stamp or flag.
            _clear_scan_started(meta_conn, spec, expected=started_at)
            meta_conn.commit()
        finally:
            meta_conn.close()
    except Exception:  # pragma: no cover - defensive
        logger.exception(
            "Background FTS integrity-check for '%s' failed", spec.table_name
        )
        try:
            cleanup = sqlite3.connect(db_path, timeout=timeout, check_same_thread=False)
            try:
                _clear_scan_started(cleanup, spec, expected=started_at)
                cleanup.commit()
            finally:
                cleanup.close()
        except sqlite3.DatabaseError:
            pass
    finally:
        with _integrity_scan_lock:
            if _integrity_scan_threads.get(key) is threading.current_thread():
                _integrity_scan_threads.pop(key, None)


def _dispatch_background_integrity_scan(
    conn: sqlite3.Connection, spec: ExternalContentFtsSpec, *, now: float | None = None
) -> bool:
    """Try to run the deep FTS integrity-check on a daemon thread.

    Returns ``True`` when the caller should NOT run the check synchronously —
    either a scan was dispatched here or one is already in flight (in-process, or
    in another process per a fresh ``fts_integrity_scan_started_at`` stamp).
    Returns ``False`` to fall back to the synchronous check (e.g. an in-memory or
    anonymous DB that cannot be reopened from another thread).
    """
    db_path = _database_path_for_connection(conn)
    if not db_path or db_path == ":memory:":
        return False
    current = time.time() if now is None else now
    key = (db_path, spec.table_name)
    with _integrity_scan_lock:
        existing = _integrity_scan_threads.get(key)
        if existing is not None and existing.is_alive():
            return True
        started = _load_scan_started_at(conn, spec)
        if started is not None and (current - started) < INTEGRITY_SCAN_STALE_SECONDS:
            # A recent scan (this or another process) owns this table; let it
            # stamp the marker. The bind returns fast without a duplicate scan.
            return True
        # Durably claim the scan cross-process BEFORE starting the thread. The
        # spawned thread stamps ``scan_started_at`` on its own connection, but
        # ``thread.start()`` returns before that stamp is committed — a second
        # process racing ``ensure_external_content_fts`` in that window would read
        # no stamp and dispatch a duplicate deep scan (F6). Writing the stamp here
        # under BEGIN IMMEDIATE closes that window; best-effort (a transient lock
        # just falls back to the thread's own stamp).
        claim_timeout = SQLITE_BUSY_TIMEOUT_MS / 1000.0
        try:
            claim_conn = sqlite3.connect(
                db_path, timeout=claim_timeout, check_same_thread=False
            )
            try:
                claim_conn.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}")
                claim_conn.execute("BEGIN IMMEDIATE")
                _record_scan_started(claim_conn, spec, now=current)
                claim_conn.commit()
            finally:
                claim_conn.close()
        except sqlite3.DatabaseError:
            pass

        thread = threading.Thread(
            target=_run_background_integrity_scan,
            args=(db_path, spec, current),
            name=f"lcm-fts-integrity-{spec.table_name}",
            daemon=True,
        )
        _integrity_scan_threads[key] = thread
        thread.start()
        return True


def join_background_integrity_scans(timeout: float | None = None) -> None:
    """Block until in-flight background integrity scans finish.

    Test/diagnostic helper so callers can deterministically observe the marker or
    failure flag a dispatched scan writes."""
    with _integrity_scan_lock:
        threads = list(_integrity_scan_threads.values())
    for thread in threads:
        thread.join(timeout)


def _fts_needs_rebuild(
    conn: sqlite3.Connection,
    spec: ExternalContentFtsSpec,
    *,
    now: float | None = None,
    throttle: bool = False,
) -> bool:
    if _fts_needs_rebuild_structural(conn, spec):
        return True
    # Structurally sound: the FTS5 integrity-check is O(index size) and was the
    # dominant startup cost on large databases (issue #235). On the startup path
    # (``throttle=True``) skip it when already checked within the interval.
    # Explicit repair (e.g. ``/lcm doctor repair apply``) uses ``throttle=False``
    # so it always runs the deep check and can fix same-row-count drift that the
    # structural checks cannot see.
    if throttle and not _should_run_integrity_check(conn, spec, now=now):
        return False
    # The deep check is due. On the startup path, dispatch it to a background
    # thread so the bind returns immediately (issue #6); the scan flags any
    # corruption via metadata rather than rebuilding here. The kill-switch and
    # non-file DBs fall back to the exact old synchronous behavior below.
    if throttle and _background_integrity_enabled():
        if _dispatch_background_integrity_scan(conn, spec, now=now):
            return False
    result = check_external_content_fts_integrity(conn, spec)
    if result["status"] == "pass":
        _record_integrity_checked(conn, spec, now=now)
    return result["status"] == "fail"


# SQLite/FTS5 error substrings that denote genuine corruption or index drift
# (SQLITE_CORRUPT / SQLITE_NOTADB, and the FTS5 integrity-check's own
# ``checksum mismatch`` for same-row-count stale drift). Everything else a
# writable integrity-check can raise — SQLITE_BUSY / SQLITE_LOCKED "database is
# locked", timeouts — is transient and must classify as ``unchecked``, never
# ``fail`` (which records a corruption flag).
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

    FTS5 exposes this as a special INSERT command. Wrap it in a savepoint and
    roll it back so diagnostics can verify the index without leaving any state
    behind on the shared connection.
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
        # A transient lock/busy/timeout — or any other non-corruption error — must
        # NOT be reported as corruption: the background scan would otherwise wedge
        # a false ``fts_integrity_failed`` flag (F3). Only an actual corruption
        # signature (malformed / disk image / not-a-database) fails the check.
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


def _drop_fts_triggers(conn: sqlite3.Connection, trigger_sqls: Sequence[str]) -> None:
    for trigger_sql in trigger_sqls:
        trigger_name = _extract_trigger_name(trigger_sql)
        if trigger_name:
            conn.execute(f"DROP TRIGGER IF EXISTS {quote_sql_identifier(trigger_name)}")


def _drop_fts_artifacts(conn: sqlite3.Connection, spec: ExternalContentFtsSpec) -> None:
    _drop_fts_triggers(conn, spec.trigger_sqls)
    _drop_fts_table(conn, spec.table_name)


def _check_disk_space(db_path: str) -> bool:
    try:
        parent = os.path.dirname(os.path.abspath(db_path)) or "."
        return shutil.disk_usage(parent).free >= _MIN_DISK_SPACE_BYTES
    except (OSError, AttributeError):
        return True


def _fts_missing_triggers(conn: sqlite3.Connection, spec: ExternalContentFtsSpec) -> bool:
    expected = {
        trigger_name
        for trigger_name in (_extract_trigger_name(sql) for sql in spec.trigger_sqls)
        if trigger_name
    }
    if not expected:
        return False
    placeholders = ",".join("?" for _ in expected)
    rows = conn.execute(
        f"SELECT name FROM sqlite_master WHERE type='trigger' AND name IN ({placeholders})",
        tuple(sorted(expected)),
    ).fetchall()
    existing = {str(row[0]) for row in rows if row and row[0]}
    return bool(expected - existing)


def external_content_fts_needs_repair(conn: sqlite3.Connection, spec: ExternalContentFtsSpec) -> bool:
    return _fts_needs_rebuild_structural(conn, spec) or _fts_missing_triggers(conn, spec)


@contextmanager
def _fts_repair_ownership(conn: sqlite3.Connection):
    """Own the short FTS structural-repair transaction.

    Fresh startup connections are normally outside a transaction, so
    ``BEGIN IMMEDIATE`` serializes the structural recheck and all FTS DDL
    across independent processes. A caller that already owns a transaction
    gets a savepoint instead; committing that caller-owned transaction remains
    the historical behavior of the repair helper.
    """
    if conn.in_transaction:
        savepoint = quote_sql_identifier("lcm_fts_repair_ownership")
        conn.execute(f"SAVEPOINT {savepoint}")
        try:
            yield False
        except BaseException:
            try:
                conn.execute(f"ROLLBACK TO {savepoint}")
            finally:
                conn.execute(f"RELEASE {savepoint}")
            raise
        else:
            conn.execute(f"RELEASE {savepoint}")
        return

    previous_timeout = conn.execute("PRAGMA busy_timeout").fetchone()
    previous_timeout_ms = int(previous_timeout[0]) if previous_timeout else 0
    conn.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}")
    try:
        # A loser waits for the winner, then takes a current write snapshot and
        # rechecks the complete FTS state before deciding whether to repair.
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield True
        except BaseException:
            conn.rollback()
            raise
        else:
            try:
                conn.commit()
            except BaseException:
                conn.rollback()
                raise
    finally:
        conn.execute(f"PRAGMA busy_timeout={previous_timeout_ms}")


def repair_external_content_fts(
    conn: sqlite3.Connection,
    spec: ExternalContentFtsSpec,
    *,
    now: float | None = None,
    throttle: bool = False,
) -> dict[str, bool]:
    rebuilt = False
    degraded = False
    fts_structure_needs_rebuild = _fts_needs_rebuild_structural(conn, spec)
    deep_repair_needed = False
    if not fts_structure_needs_rebuild or not throttle:
        # Preserve the cheap startup path and its background integrity-scan
        # behavior whenever the FTS table and shadow structure can support a
        # deep check. Missing triggers alone must not hide same-row-count index
        # drift. Explicit repair remains unthrottled for every repair state.
        deep_repair_needed = _fts_needs_rebuild(conn, spec, now=now, throttle=throttle)
        if not deep_repair_needed:
            # A trigger can disappear after the initial complete-state check.
            # Return on the healthy fast path only while it is still complete;
            # any observed trigger repair must pass through write ownership below.
            if not _fts_missing_triggers(conn, spec):
                _clear_integrity_failed(conn, spec)
                conn.commit()
                return {
                    "rebuilt": False,
                    "degraded": False,
                    "triggers_recreated": False,
                }

    with _fts_repair_ownership(conn) as owns_transaction:
        # This is the decisive cross-process recheck. If another process won
        # while we waited, accept its complete table/shadow/trigger state and do
        # not drop or recreate it.
        winner_state_needs_repair = external_content_fts_needs_repair(conn, spec)
        owner_rebuild_needed = (
            _fts_needs_rebuild_structural(conn, spec) if winner_state_needs_repair else False
        )
        if not owner_rebuild_needed and deep_repair_needed:
            # Explicit repair (and synchronous startup when background scans are
            # disabled) must revalidate same-row-count token drift while owning
            # the write boundary. A repaired winner is accepted without a second
            # destructive rebuild.
            owner_rebuild_needed = _fts_needs_rebuild(
                conn, spec, now=now, throttle=False
            )
        if owner_rebuild_needed:
            db_path = conn.execute("PRAGMA database_list").fetchone()
            low_disk = False
            if db_path:
                db_file = db_path[2]
                low_disk = bool(db_file and not _check_disk_space(db_file))
            if low_disk:
                logger.warning(
                    "Low disk space for FTS rebuild of '%s' (%d MB needed), degrading to LIKE search",
                    spec.table_name,
                    _MIN_DISK_SPACE_BYTES // (1024 * 1024),
                )
                _drop_fts_artifacts(conn, spec)
                degraded = True
            else:
                _drop_fts_table(conn, spec.table_name)
                conn.execute(
                    f"""
                    CREATE VIRTUAL TABLE {quote_sql_identifier(spec.table_name)} USING fts5(
                        {quote_sql_identifier(spec.indexed_column)},
                        content={quote_sql_identifier(spec.content_table)},
                        content_rowid={quote_sql_identifier(spec.content_rowid)}
                    )
                    """
                )
                conn.execute(
                    f"INSERT INTO {quote_sql_identifier(spec.table_name)}({quote_sql_identifier(spec.table_name)}) VALUES('rebuild')"
                )
                rebuilt = True

        if degraded:
            triggers_were_missing = False
        else:
            triggers_were_missing = _fts_missing_triggers(conn, spec)
            for trigger_sql in spec.trigger_sqls:
                conn.execute(trigger_sql)
        if rebuilt:
            # A freshly rebuilt index is known-consistent; record the marker so
            # the next startup can skip the deep integrity-check within the
            # interval.
            _record_integrity_checked(conn, spec, now=now)
        # A completed repair resolves any prior background-scan corruption flag:
        # clear it in the SAME transaction that commits the rebuild.
        _clear_integrity_failed(conn, spec)

    if not owns_transaction:
        # Preserve the helper's historical behavior for callers that supplied an
        # already-active transaction while keeping the startup path's ownership
        # boundary isolated and rollback-safe.
        conn.commit()
    return {"rebuilt": rebuilt, "degraded": degraded, "triggers_recreated": triggers_were_missing}


def ensure_external_content_fts(
    conn: sqlite3.Connection, spec: ExternalContentFtsSpec, *, now: float | None = None
) -> None:
    # Startup path: throttle the deep integrity-check. Explicit repair callers
    # use ``repair_external_content_fts(..., throttle=False)`` for a forced check.
    repair_external_content_fts(conn, spec, now=now, throttle=True)


def run_versioned_migrations(conn: sqlite3.Connection) -> None:
    refuse_schema_version_too_new(conn)

    ensure_metadata_table(conn)
    ensure_migration_state_table(conn)

    refuse_schema_version_too_new(conn)
    current_version = get_schema_version(conn)
    if current_version < 2:
        mark_migration_step_complete(conn, "v2_external_content_fts_triggers")
        current_version = 2

    if current_version < 3:
        ensure_lifecycle_state_table(conn)
        mark_migration_step_complete(conn, "v3_lifecycle_state")
        current_version = 3
    else:
        ensure_lifecycle_state_table(conn)

    ensure_lifecycle_state_columns(conn)
    if current_version < 4:
        mark_migration_step_complete(conn, "v4_lifecycle_debt_columns")
        current_version = 4

    ensure_message_origin_columns(conn)
    if current_version < 5:
        mark_migration_step_complete(conn, "v5_message_conversation_id")
        current_version = 5

    set_schema_version(conn, current_version)
