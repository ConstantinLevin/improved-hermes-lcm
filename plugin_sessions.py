"""The plugin's own sessions (#20; #29, W1 and W6).

A plugin session is named at the host's signal that a session begins and is
independent of every host identifier. The host gives no kind of beginning to the
engine, so (ruling 1):

- a host session id this store has not seen names a new plugin session, with the
  signal recorded as it was received and no kind unless the signal itself carries one;
- a host session id it has seen continues the plugin session whose record line ends
  at it: the session that began at it, or the session a confirmed compaction carried
  to it (the host's identifier after the compaction);
- explicit signals about a session that already exists are appended as facts;
- a platform a signal states is the fact ``platform``, appended when it differs from
  the session's last platform fact.

Session records are written when their signal arrives, between compactions too. The
tables are insert-only; nothing here is ever changed.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional

from .db_bootstrap import close_connection, open_store
from .handles import SESSION, new_handle

logger = logging.getLogger(__name__)

_HANDLE_DRAWS = 8


class PluginSessions:
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

    def close(self, reason: str = "closed") -> None:
        """Close the connection once a session write on another thread (a hook's) has
        finished (the helper's lock). Later use raises."""
        with self._lock:
            self._conn = close_connection(self._conn, db_path=self.db_path, reason=reason, owner="the sessions")

    def find(self, host_session_id: str) -> Optional[str]:
        row = self._conn.execute(
            "SELECT handle FROM sessions WHERE host_session_id = ?",
            (host_session_id,),
        ).fetchone()
        if row:
            return str(row[0])
        row = self._conn.execute(
            "SELECT c.session FROM confirmations f JOIN compactions c ON c.compaction_id = f.compaction "
            "WHERE f.host_session_after = ? ORDER BY f.compaction DESC LIMIT 1",
            (host_session_id,),
        ).fetchone()
        return str(row[0]) if row else None

    def name_session(
        self,
        host_session_id: str,
        *,
        signal: str,
        kind: Optional[str] = None,
    ) -> tuple[str, bool]:
        """Return ``(handle, created)`` for the plugin session a host id names.

        The check and the insert run under the write lock, so two processes that
        meet the same unknown id name one session.
        """
        if not host_session_id:
            raise ValueError("a plugin session is named only from a non-empty host session id")
        with self._lock:
            conn = self._conn
            try:
                conn.execute("BEGIN IMMEDIATE")
                try:
                    existing = self.find(host_session_id)
                    if existing is not None:
                        conn.execute("COMMIT")
                        return existing, False
                    handle = self._insert_session(conn, host_session_id, signal, kind)
                    conn.execute("COMMIT")
                except BaseException:
                    conn.execute("ROLLBACK")
                    raise
            except Exception:
                logger.error(
                    "LCM could not record the plugin session for host session %s (signal %s) in %s",
                    host_session_id,
                    signal,
                    self.db_path,
                    exc_info=True,
                )
                raise
        logger.info(
            "LCM named plugin session %s at %s for host session %s",
            handle,
            signal,
            host_session_id,
        )
        return handle, True

    @staticmethod
    def _insert_session(
        conn: sqlite3.Connection,
        host_session_id: str,
        signal: str,
        kind: Optional[str],
    ) -> str:
        for _ in range(_HANDLE_DRAWS):
            handle = new_handle(SESSION)
            try:
                conn.execute(
                    "INSERT INTO sessions(handle, began_at, signal, kind, host_session_id) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (handle, time.time(), signal, kind, host_session_id),
                )
                return handle
            except sqlite3.IntegrityError:
                if conn.execute("SELECT 1 FROM sessions WHERE handle = ?", (handle,)).fetchone():
                    continue
                raise
        raise RuntimeError(f"LCM drew {_HANDLE_DRAWS} session handles that were all taken")

    # --- The host's hooks that name a session ------------------------------------
    # Hooks are the host's, not an engine copy's: they write only rows keyed by the
    # host ids they carry, and never touch the session an engine copy acts for.

    def on_session_reset_hook(self, **payload) -> None:
        """``on_session_reset(session_id, platform, reason, …)``: a session begins.

        With ``reason="new_session"`` (CLI and gateway ``/new``) the kind of the
        beginning is known: a new plugin session gets it as its kind; a session
        already named for that id (the CLI binds the engine first) gets it as the
        fact ``began_as``. Without a reason (the TUI, at every agent build) the id is
        named and no kind is recorded.
        """
        host_session_id = str(payload.get("session_id") or "")
        if not host_session_id:
            return
        kind = "new_session" if payload.get("reason") == "new_session" else None
        handle, created = self.name_session(
            host_session_id,
            signal="hook:on_session_reset",
            kind=kind,
        )
        if kind and not created:
            self.add_fact(handle, "began_as", kind, signal="hook:on_session_reset",
                          host_session_id=host_session_id)
        self.note_platform(handle, str(payload.get("platform") or ""),
                           signal="hook:on_session_reset", host_session_id=host_session_id)

    def subagent_start_hook(self, **payload) -> None:
        """``subagent_start(parent_session_id, child_session_id, …)``: the child's parent.

        The hook arrives after the child's engine has started, on the parent's
        thread. The parent is recorded as a fact of the child's session, naming the
        parent's plugin session.
        """
        child_host = str(payload.get("child_session_id") or "")
        parent_host = str(payload.get("parent_session_id") or "")
        if not child_host or not parent_host:
            return
        child, _ = self.name_session(child_host, signal="hook:subagent_start")
        parent, _ = self.name_session(parent_host, signal="hook:subagent_start")
        self.add_fact(child, "parent", parent, signal="hook:subagent_start",
                      host_session_id=parent_host)

    def note_platform(
        self,
        session: str,
        platform: str,
        *,
        signal: str,
        host_session_id: Optional[str] = None,
    ) -> None:
        """Append the fact ``platform`` when a signal states one that differs from
        the session's last platform fact. The read and the insert run under the
        write lock, so two processes stating the same platform append it once."""
        if not platform:
            return
        with self._lock:
            conn = self._conn
            try:
                conn.execute("BEGIN IMMEDIATE")
                try:
                    row = conn.execute(
                        "SELECT value FROM session_facts WHERE session = ? AND kind = 'platform' "
                        "ORDER BY fact_id DESC LIMIT 1",
                        (session,),
                    ).fetchone()
                    if row is None or row[0] != platform:
                        conn.execute(
                            "INSERT INTO session_facts(session, kind, value, signal, host_session_id, at) "
                            "VALUES (?, 'platform', ?, ?, ?, ?)",
                            (session, platform, signal, host_session_id or None, time.time()),
                        )
                    conn.execute("COMMIT")
                except BaseException:
                    conn.execute("ROLLBACK")
                    raise
            except Exception:
                logger.error(
                    "LCM could not record the fact platform=%s on plugin session %s (signal %s) in %s",
                    platform,
                    session,
                    signal,
                    self.db_path,
                    exc_info=True,
                )
                raise

    def add_fact(
        self,
        session: str,
        kind: str,
        value: str,
        *,
        signal: str,
        host_session_id: Optional[str] = None,
    ) -> None:
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT INTO session_facts(session, kind, value, signal, host_session_id, at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (session, kind, value, signal, host_session_id or None, time.time()),
                )
            except Exception:
                logger.error(
                    "LCM could not record the fact %s=%s on plugin session %s (signal %s) in %s",
                    kind,
                    value,
                    session,
                    signal,
                    self.db_path,
                    exc_info=True,
                )
                raise
