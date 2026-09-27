"""The tools' reader of stored messages: the ``messages`` view over the record.

It writes nothing; the record is written by ``RecordStore`` at compactions.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import stat
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from .db_bootstrap import (
    LockedConnection,
    StoreRefusedError,
    close_connection,
    open_store,
    refuse_cross_vm_filesystem,
)
from .sqlite_util import _create_private_sqlite_file

logger = logging.getLogger(__name__)


_MESSAGE_SELECT_COLUMNS = (
    "store_id, session_id, source, role, content, tool_call_id, "
    "tool_calls, tool_name, timestamp, token_estimate, pinned, conversation_id, "
    "ingested_at, observed_at, observed_at_source, seq, revises_node_id, uncounted_images, content_type"
)
_UNKNOWN_SOURCE = "unknown"


def _same_directory_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _restrict_created_sqlite_directory(path: Path) -> None:
    """Restrict a newly created directory without following a replacement."""
    if os.name != "posix":  # pragma: no cover - Windows compatibility fallback
        path.chmod(0o700)
        return

    parent = path.parent
    expected_parent = os.stat(parent, follow_symlinks=False)
    if not stat.S_ISDIR(expected_parent.st_mode):
        raise OSError(f"database directory parent is not a real directory: {parent}")

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_CLOEXEC", 0)
    parent_fd = os.open(parent, flags)
    try:
        opened_parent = os.fstat(parent_fd)
        if (
            not stat.S_ISDIR(opened_parent.st_mode)
            or not _same_directory_identity(expected_parent, opened_parent)
        ):
            raise OSError(f"database directory parent changed during validation: {parent}")

        expected = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISDIR(expected.st_mode):
            raise OSError(f"database directory is not a real directory: {path}")
        fd = os.open(path.name, flags, dir_fd=parent_fd)
        try:
            opened = os.fstat(fd)
            if (
                not stat.S_ISDIR(opened.st_mode)
                or not _same_directory_identity(expected, opened)
            ):
                raise OSError(f"database directory changed during validation: {path}")
            os.fchmod(fd, 0o700)
            current = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
            if not _same_directory_identity(opened, current):
                raise OSError(f"database directory changed while restricting permissions: {path}")
        finally:
            os.close(fd)
    finally:
        os.close(parent_fd)


def _prepare_private_sqlite_storage(db_path: Path) -> None:
    """Create a new store file with mode 0600 before SQLite opens it.

    An existing regular file is left alone: it is never opened and closed here,
    because closing a descriptor on a live database releases SQLite's locks on it
    for every connection in the process. Whether an existing file is a store of
    this plugin is decided by its identity row when SQLite opens it. Anything else
    at the path (a directory, a symbolic link to a missing file) is refused.
    """
    if os.path.isfile(db_path):
        return
    refuse_cross_vm_filesystem(db_path)
    try:
        db_path.parent.mkdir(parents=True, mode=0o700)
    except FileExistsError:
        pass
    else:
        _restrict_created_sqlite_directory(db_path.parent)
    try:
        _create_private_sqlite_file(db_path)
    except OSError as exc:
        message = (
            f"LCM cannot create its store at {db_path}: {exc}. "
            f"Nothing was created."
        )
        logger.error(message)
        raise StoreRefusedError(message) from exc


def _legacy_blank_source_clause(column: str) -> str:
    # SQLite TRIM() only strips spaces unless given an explicit character set.
    # Match Python's write-time `str.strip()` behavior for common ASCII whitespace
    # so legacy tabs/newlines do not become a fake attributed source bucket.
    whitespace_chars = "char(9) || char(10) || char(11) || char(12) || char(13) || char(32)"
    return f"({column} IS NULL OR TRIM({column}, {whitespace_chars}) = '')"


def _normalize_source_value(source: str | None) -> str:
    normalized = (source or "").strip()
    return normalized or _UNKNOWN_SOURCE


def _normalize_conversation_id_value(conversation_id: str | None) -> str:
    return (conversation_id or "").strip()


class MessageStore:
    """SQLite-backed immutable message store."""

    def __init__(self, db_path: str | Path, *, hermes_home: str = ""):
        self.db_path = Path(db_path)
        _prepare_private_sqlite_storage(self.db_path)
        self._hermes_home = hermes_home or str(self.db_path.parent)
        self._conn: Optional[sqlite3.Connection] = None
        # Every read runs under the lock close() takes: a close waits for the read,
        # or the read raises StoreClosedError (LockedConnection).
        self._lock = threading.RLock()
        self._locked = LockedConnection(lambda: self._conn, self._lock)
        self._init_db()

    def _init_db(self):
        self._conn = sqlite3.connect(str(self.db_path), timeout=5.0, check_same_thread=False)
        try:
            # Refuses a database this plugin did not write and creates the store in an
            # empty one. No DDL otherwise.
            open_store(self._conn, self.db_path)
        except BaseException:
            self._conn.close()
            self._conn = None
            raise

    # -- Read operations (over the record's views) ---------------------------- ----------------------------------------------------

    def get(self, store_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve a single message by store_id."""
        row = self._locked.execute(
            f"SELECT {_MESSAGE_SELECT_COLUMNS} FROM messages WHERE store_id = ?", (store_id,)
        ).fetchone()
        return self._row_to_dict(row) if row else None

    def get_returned_tail(self, session_id: str) -> List[Dict[str, Any]]:
        """The fresh tail as it was returned: the record entries of the session's
        latest effective return, in their return positions."""
        rows = self._locked.execute(
            f"""SELECT {_MESSAGE_SELECT_COLUMNS} FROM messages
               WHERE session_id = ? AND store_id IN (
                   SELECT r.record_id FROM latest_returns l JOIN records r ON r.handle = l.record
                   WHERE l.kind = 'record' AND r.session = ?)
               ORDER BY seq""",
            (session_id, session_id),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_session_count(self, session_id: str) -> int:
        """Count messages in a session."""
        row = self._locked.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return row[0] if row else 0

    def get_session_uncounted_images(self, session_id: str) -> int:
        """The images the session's token estimates left uncounted (#35)."""
        row = self._locked.execute(
            "SELECT COALESCE(SUM(uncounted_images), 0) FROM messages WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return row[0] if row else 0

    def get_session_token_total(self, session_id: str) -> int:
        """Sum of token estimates for a session."""
        row = self._locked.execute(
            "SELECT COALESCE(SUM(token_estimate), 0) FROM messages WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return row[0] if row else 0

    def get_source_stats(self, session_id: str | None = None) -> Dict[str, int]:
        """Return raw source-bucket counts for diagnostics."""
        where = ""
        args: list[Any] = []
        if session_id is not None:
            where = "WHERE session_id = ?"
            args.append(session_id)

        legacy_blank_clause = _legacy_blank_source_clause("source")
        query = f"""
            SELECT COUNT(*) AS messages_total,
                   COALESCE(SUM(CASE WHEN source = ? THEN 1 ELSE 0 END), 0) AS normalized_unknown_messages,
                   COALESCE(SUM(CASE WHEN {legacy_blank_clause} THEN 1 ELSE 0 END), 0) AS legacy_blank_source_messages,
                   COALESCE(SUM(CASE WHEN NOT {legacy_blank_clause} AND source != ? THEN 1 ELSE 0 END), 0) AS attributed_messages
            FROM messages
            {where}
            """
        query_args: list[Any] = [_UNKNOWN_SOURCE, _UNKNOWN_SOURCE, *args]
        row = self._locked.execute(query, query_args).fetchone()

        messages_total = int(row[0] or 0) if row else 0
        normalized_unknown = int(row[1] or 0) if row else 0
        legacy_blank = int(row[2] or 0) if row else 0
        attributed = int(row[3] or 0) if row else 0
        return {
            "messages_total": messages_total,
            "attributed_messages": attributed,
            "normalized_unknown_messages": normalized_unknown,
            "legacy_blank_source_messages": legacy_blank,
            "effective_unknown_messages": normalized_unknown + legacy_blank,
        }

    # -- Helpers ------------------------------------------------------------

    def _row_to_dict(self, row) -> Dict[str, Any]:
        """Convert a sqlite3 row to a dict."""
        if row is None:
            return {}
        cols = [
            "store_id", "session_id", "source", "role", "content", "tool_call_id",
            "tool_calls", "tool_name", "timestamp", "token_estimate", "pinned", "conversation_id",
            "ingested_at", "observed_at", "observed_at_source", "seq", "revises_node_id",
            "uncounted_images", "content_type",
        ]
        d = dict(zip(cols, row[:len(cols)]))
        d["source"] = _normalize_source_value(d.get("source"))
        d["conversation_id"] = _normalize_conversation_id_value(d.get("conversation_id"))
        # Deserialize tool_calls JSON
        if d.get("tool_calls"):
            try:
                d["tool_calls"] = json.loads(d["tool_calls"])
            except (json.JSONDecodeError, TypeError):
                pass
        return d


    # -- Connection access --------------------------------------------------

    @property
    def connection(self) -> LockedConnection:
        """The connection as diagnostics use it: every call runs under the lock
        :meth:`close` takes, and ``execute`` returns its rows already fetched. Once
        closed, every use raises ``StoreClosedError``.

        Exposed for read-oriented diagnostics and inspection -- integrity /
        quick checks, schema health -- that need ad-hoc
        queries the store does not wrap in a purpose-built method. Callers must
        treat it as read-only: the tables behind it are the record's, written
        only by ``RecordStore``.
        """
        return self._locked

    # -- Lifecycle ----------------------------------------------------------

    def close(self, reason: str = "closed") -> None:
        """Close the connection once a read on another thread has finished (the
        lock every read takes); later use raises. The engine closes its helpers at
        plugin unload and when the engine is collected (``LCMEngine.close``)."""
        with self._lock:
            self._conn = close_connection(self._conn, db_path=self.db_path, reason=reason,
                                          owner="the message reader")
