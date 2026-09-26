"""The tools' reader of summaries: the ``summary_nodes`` view over the record.

A node is a summary of one chunk of stored messages, as the session's latest
effective return holds it (the cover). It writes nothing; the record is written
by ``RecordStore`` at compactions. Condensed summaries come with #34.
"""

import json
import logging
import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .db_bootstrap import (
    LockedConnection,
    close_connection,
    open_store,
)


logger = logging.getLogger(__name__)


@dataclass
class SummaryNode:
    """A single node in the summary DAG."""
    node_id: int = 0
    session_id: str = ""
    depth: int = 0
    summary: str = ""
    token_count: int = 0
    source_token_count: int = 0  # total tokens of source material
    # Images in the source the estimate left uncounted (#21, #35).
    source_uncounted_images: int = 0
    source_ids: List[int] = field(default_factory=list)  # store_ids or node_ids
    source_type: str = "messages"  # "messages" or "nodes"
    created_at: float = 0.0
    earliest_at: float | None = None
    latest_at: float | None = None
    # Transcript order of the summary in the session's cover (the view's ``seq``).
    seq: int = 0


class SummaryDAG:
    """SQLite-backed DAG of summary nodes."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self._conn: Optional[sqlite3.Connection] = None
        # Every read runs under the lock close() takes: a close waits for the read,
        # or the read raises StoreClosedError (LockedConnection).
        self._db_lock = threading.RLock()
        self._locked = LockedConnection(lambda: self._conn, self._db_lock)
        self._init_db()

    @property
    def connection(self) -> LockedConnection:
        """The connection as diagnostics use it: every call runs under the lock
        :meth:`close` takes, and ``execute`` returns its rows already fetched. Once
        closed, every use raises ``StoreClosedError``.

        Exposed for read-oriented diagnostics and inspection -- FTS sync counts,
        integrity checks, latest-node lookups -- that need ad-hoc queries the DAG
        does not wrap in a purpose-built method. Callers must treat it as
        read-only: the tables behind it are the record's, written only by
        ``RecordStore``.
        """
        return self._locked

    def _init_db(self):
        self._conn = sqlite3.connect(str(self.db_path), timeout=5.0, check_same_thread=False)
        try:
            open_store(self._conn, self.db_path)
        except BaseException:
            self._conn.close()
            self._conn = None
            raise

    # -- Read ---------------------------------------------------------------

    def get_node(self, node_id: int) -> Optional[SummaryNode]:
        row = self._locked.execute(
            "SELECT * FROM summary_nodes WHERE node_id = ?", (node_id,)
        ).fetchone()
        return self._row_to_node(row) if row else None

    def get_session_nodes(self, session_id: str,
                          depth: int | None = None,
                          limit: int = 1000) -> List[SummaryNode]:
        """Get nodes for a session, optionally filtered by depth."""
        with self._db_lock:
            if depth is not None:
                rows = self._locked.execute(
                    """SELECT * FROM summary_nodes
                       WHERE session_id = ? AND depth = ?
                       ORDER BY seq LIMIT ?""",
                    (session_id, depth, limit),
                ).fetchall()
            else:
                rows = self._locked.execute(
                    """SELECT * FROM summary_nodes
                       WHERE session_id = ?
                       ORDER BY depth, seq LIMIT ?""",
                    (session_id, limit),
                ).fetchall()
        return [self._row_to_node(r) for r in rows]


    def get_session_node_count(self, session_id: str) -> int:
        """Count summary nodes for a session without loading node rows."""
        row = self._locked.execute(
            "SELECT COUNT(*) FROM summary_nodes WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return int(row[0] if row else 0)

    def get_session_depth_stats(self, session_id: str) -> Dict[int, Dict[str, int]]:
        """Aggregate per-depth node/token stats for a session."""
        rows = self._locked.execute(
            """SELECT depth,
                      COUNT(*) AS count,
                      COALESCE(SUM(token_count), 0) AS tokens,
                      COALESCE(SUM(source_token_count), 0) AS source_tokens,
                      COALESCE(SUM(source_uncounted_images), 0) AS source_uncounted_images
               FROM summary_nodes
               WHERE session_id = ?
               GROUP BY depth
               ORDER BY depth""",
            (session_id,),
        ).fetchall()
        return {
            int(row[0]): {
                "count": int(row[1] or 0),
                "tokens": int(row[2] or 0),
                "source_tokens": int(row[3] or 0),
                "source_uncounted_images": int(row[4] or 0),
            }
            for row in rows
        }

    # -- Helpers ------------------------------------------------------------

    def _row_to_node(self, row) -> SummaryNode:
        return SummaryNode(
            node_id=row[0],
            session_id=row[1],
            depth=row[2],
            summary=row[3],
            token_count=row[4],
            source_token_count=row[5],
            source_uncounted_images=int(row[6] or 0),
            source_ids=json.loads(row[7]) if row[7] else [],
            source_type=row[8],
            created_at=row[9],
            earliest_at=row[10],
            latest_at=row[11],
            seq=int(row[12] or 0) if len(row) > 12 else 0,
        )

    def close(self, reason: str = "closed") -> None:
        """Close the connection once a read of this helper on another thread has
        finished (its lock); later use raises. The engine closes its helpers at plugin
        unload and when the engine is collected (``LCMEngine.close``)."""
        with self._db_lock:
            self._conn = close_connection(self._conn, db_path=self.db_path, reason=reason, owner="the summary reader")
