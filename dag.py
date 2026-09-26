"""The tools' reader of summaries: the ``summary_nodes`` view over the record.

A node is a summary of one chunk of stored messages, as the session's latest
effective return holds it (the cover). It writes nothing; the record is written
by ``RecordStore`` at compactions. Condensed summaries come with #34.
"""

import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .db_bootstrap import (
    NODES_FTS_SPEC,
    ExternalContentFtsSpec,
    LockedConnection,
    close_connection,
    open_store,
)
from .search_query import (
    AGE_DECAY_RATE,
    compute_search_candidate_cap,
    compute_directness_rank_bonus_upper_bound,
    compute_directness_score,
    compute_like_fallback_fetch_limit,
    compute_search_fetch_limit,
    contains_risky_fts_ascii,
    count_term_matches,
    escape_like,
    extract_quoted_phrases,
    extract_search_terms,
    normalize_search_sort,
    requires_like_fallback,
    sanitize_fts5_query,
    sanitize_like_query,
    should_apply_directness_rank_adjustment,
)
from .store import _normalize_source_value, _UNKNOWN_SOURCE, _legacy_blank_source_clause


logger = logging.getLogger(__name__)


def _build_search_order_by(sort: str | None, recency_expr: str, order_expr: str = "n.seq") -> str:
    """Recency is the cover order (``order_expr``); ``recency_expr`` only ages a hit
    for the hybrid blend."""
    normalized = normalize_search_sort(sort)
    if normalized == "relevance":
        return f"rank ASC, {order_expr} DESC"
    if normalized == "hybrid":
        return (
            f"(rank / (1 + (MAX(0.0, ((strftime('%s','now') - {recency_expr}) / 3600.0)) * {AGE_DECAY_RATE}))) ASC, "
            f"{order_expr} DESC"
        )
    return f"{order_expr} DESC"


def _fallback_result_sort_key(node: "SummaryNode", sort: str | None) -> tuple[float, float, float]:
    normalized = normalize_search_sort(sort)
    score = float(node.search_rank or 0.0) * -1.0
    recency = float(node.latest_at or node.created_at or 0.0)
    order = float(node.seq or 0)
    directness = float(node.search_directness or 0.0)

    if normalized == "relevance":
        return (-score, -directness, -order)
    if normalized == "hybrid":
        age_hours = max(0.0, (time.time() - recency) / 3600.0)
        blended = score / (1 + (age_hours * AGE_DECAY_RATE))
        return (-blended, -directness, -order)
    return (-order, -score, -directness)


def _fts_result_sort_key(node: "SummaryNode", sort: str | None) -> tuple[float, float, float]:
    normalized = normalize_search_sort(sort)
    rank = node.search_rank
    rank_value = float(rank) if rank is not None else float("inf")
    recency = float(node.latest_at or node.created_at or 0.0)
    order = float(node.seq or 0)
    directness = float(node.search_directness or 0.0)

    if normalized == "relevance":
        return (rank_value, -directness, -order)
    if normalized == "hybrid":
        age_hours = max(0.0, (time.time() - recency) / 3600.0)
        strength = (-rank_value) if rank is not None else float("-inf")
        blended_strength = strength / (1 + (age_hours * AGE_DECAY_RATE)) if rank is not None else float("-inf")
        return (-blended_strength, -directness, -order)
    return (-order, rank_value, 0.0)


def _fts_primary_value(node: "SummaryNode", sort: str | None) -> float:
    normalized = normalize_search_sort(sort)
    rank = node.search_rank
    rank_value = float(rank) if rank is not None else float("inf")
    if normalized == "hybrid":
        recency = float(node.latest_at or node.created_at or 0.0)
        age_hours = max(0.0, (time.time() - recency) / 3600.0)
        strength = (-rank_value) if rank is not None else float("-inf")
        blended_strength = strength / (1 + (age_hours * AGE_DECAY_RATE)) if rank is not None else float("-inf")
        return -blended_strength
    return rank_value


def build_nodes_fts_spec() -> ExternalContentFtsSpec:
    return NODES_FTS_SPEC


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
    # No longer written (#9: nothing in a reply is recognised by pattern); rows written
    # before carry the text a pattern took from the summary. Its tool fields go with #18.
    expand_hint: str = ""
    search_rank: float | None = None
    search_directness: float = 0.0
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

    # -- Search -------------------------------------------------------------

    def search(self, query: str, session_id: str | None = None,
               limit: int = 20, sort: str | None = None,
               source: str | None = None) -> List[SummaryNode]:
        """FTS5 search across summary nodes.

        Retrieval contract:
        - ``session_id`` limits which sessions are eligible
        - ``session_id=None`` means all sessions; an empty string is treated as
          a literal session id
        - ``source`` filters summaries by descendant raw-message lineage, not by
          session-level source presence
        - mixed-source nodes may match more than one ``source`` filter
        """
        safe_query = sanitize_fts5_query(query)
        terms = extract_search_terms(safe_query)
        phrases = extract_quoted_phrases(safe_query)
        # LIKE is the fallback for text sanitization LOSES (CJK/emoji) and for a
        # query with no term left after it. A raw natural-language question is
        # NOT one of those: it sanitizes to a term form the index answers, so it
        # stays on the FTS path (F31 §3).
        if requires_like_fallback(query, safe_query):
            return self._search_like(query, session_id=session_id, limit=limit, sort=sort, source=source)

        order_by = _build_search_order_by(sort, "COALESCE(n.latest_at, n.created_at)")
        fetch_limit = compute_search_fetch_limit(limit, terms, phrases)
        candidate_cap = compute_search_candidate_cap(limit)
        apply_directness_adjustment = should_apply_directness_rank_adjustment(terms, phrases)
        max_rank_bonus = compute_directness_rank_bonus_upper_bound(terms, phrases) * 2e-7
        offset = 0
        scanned_rows = 0
        results: list[SummaryNode] = []
        source_match_cache: dict[int, bool] = {}
        while True:
            try:
                with self._db_lock:
                    if session_id is not None:
                        rows = self._locked.execute(
                            f"""SELECT n.*, rank as search_rank FROM nodes_fts fts
                               JOIN summary_nodes n ON n.node_id = fts.rowid
                               WHERE nodes_fts MATCH ? AND n.session_id = ?
                               ORDER BY {order_by} LIMIT ? OFFSET ?""",
                            (safe_query, session_id, fetch_limit, offset),
                        ).fetchall()
                    else:
                        rows = self._locked.execute(
                            f"""SELECT n.*, rank as search_rank FROM nodes_fts fts
                               JOIN summary_nodes n ON n.node_id = fts.rowid
                               WHERE nodes_fts MATCH ?
                               ORDER BY {order_by} LIMIT ? OFFSET ?""",
                            (safe_query, fetch_limit, offset),
                        ).fetchall()
                scanned_rows += len(rows)
            except sqlite3.Error as exc:
                logger.warning("FTS node search failed, falling back to LIKE: %s", exc)
                return self._search_like(query, session_id=session_id, limit=limit, sort=sort, source=source)

            raw_nodes = [self._row_to_node(r) for r in rows]
            for node in raw_nodes:
                if source and not self._node_matches_source(node.node_id, source, cache=source_match_cache):
                    continue
                node.search_directness = compute_directness_score(node.summary, terms, phrases)
                if apply_directness_adjustment and node.search_rank is not None:
                    rank_adjustment = max(float(node.search_directness), 0.0)
                    node.search_rank = float(node.search_rank) - (rank_adjustment * 2e-7)
                results.append(node)
            results.sort(key=lambda node: _fts_result_sort_key(node, sort))

            exhausted = len(rows) < fetch_limit or scanned_rows >= candidate_cap
            if source and not exhausted:
                offset += len(rows)
                remaining = candidate_cap - scanned_rows
                if remaining <= 0:
                    return results[:limit]
                fetch_limit = min(fetch_limit * 2, remaining)
                continue

            if exhausted or not apply_directness_adjustment or len(results) <= limit:
                return results[:limit]

            worst_visible_primary = _fts_primary_value(results[min(limit, len(results)) - 1], sort)
            last_fetched_primary = _fts_primary_value(raw_nodes[-1], sort)
            best_unseen_primary = last_fetched_primary - max_rank_bonus
            if best_unseen_primary > worst_visible_primary:
                return results[:limit]

            offset += len(rows)
            remaining = candidate_cap - scanned_rows
            if remaining <= 0:
                return results[:limit]
            fetch_limit = min(fetch_limit * 2, remaining)

    def _search_like(self, query: str, session_id: str | None = None,
                     limit: int = 20, sort: str | None = None,
                     source: str | None = None) -> List[SummaryNode]:
        # LIKE keeps every character the index cannot spell (emoji, punctuation)
        # because substring matching is the only way to find those rows.
        safe_query = sanitize_like_query(query)
        terms = extract_search_terms(safe_query)
        phrases = extract_quoted_phrases(safe_query)
        if not terms:
            return []
        fetch_limit = compute_search_fetch_limit(limit, terms, phrases)

        where: list[str] = ["summary IS NOT NULL"]
        args: list[Any] = []
        if session_id is not None:
            where.append("session_id = ?")
            args.append(session_id)
        like_clauses = []
        for term in terms:
            like_clauses.append("summary LIKE ? ESCAPE '\\'")
            args.append(f"%{escape_like(term)}%")
        where.append("(" + " OR ".join(like_clauses) + ")")
        fetch_limit = compute_like_fallback_fetch_limit(limit, terms, phrases)
        base_args = list(args)
        collapse_risky_repeats = contains_risky_fts_ascii(query)
        candidate_cap = compute_search_candidate_cap(limit)
        offset = 0
        scanned_rows = 0
        nodes: list[SummaryNode] = []
        source_match_cache: dict[int, bool] = {}
        while True:
            with self._db_lock:
                rows = self._locked.execute(
                    f"""SELECT * FROM summary_nodes
                        WHERE {' AND '.join(where)}
                        LIMIT ? OFFSET ?""",
                    [*base_args, fetch_limit, offset],
                ).fetchall()
            scanned_rows += len(rows)
            for row in rows:
                node = self._row_to_node(row)
                if source and not self._node_matches_source(node.node_id, source, cache=source_match_cache):
                    continue
                score = sum(
                    min(count_term_matches(node.summary, term), 1) if collapse_risky_repeats else count_term_matches(node.summary, term)
                    for term in terms
                )
                if score <= 0:
                    continue
                node.search_rank = -float(score)
                node.search_directness = compute_directness_score(node.summary, terms, phrases)
                nodes.append(node)

            nodes.sort(key=lambda node: _fallback_result_sort_key(node, sort))
            if not source or len(rows) < fetch_limit or scanned_rows >= candidate_cap:
                return nodes[:limit]

            offset += len(rows)
            remaining = candidate_cap - scanned_rows
            if remaining <= 0:
                return nodes[:limit]
            fetch_limit = min(fetch_limit * 2, remaining)

    # -- DAG traversal ------------------------------------------------------


    def _node_matches_source(
        self,
        node_id: int,
        source: str,
        *,
        cache: dict[int, bool] | None = None,
    ) -> bool:
        if not source:
            return True
        normalized_source = _normalize_source_value(source)
        if cache is not None and node_id in cache:
            return cache[node_id]
        legacy_blank_clause = _legacy_blank_source_clause("m.source")
        row = self._locked.execute(
            f"""
            WITH RECURSIVE source_walk(source_type, source_id) AS (
                SELECT n.source_type, CAST(j.value AS INTEGER)
                FROM summary_nodes n, json_each(n.source_ids) j
                WHERE n.node_id = ?

                UNION ALL

                SELECT child.source_type, CAST(j.value AS INTEGER)
                FROM summary_nodes child
                JOIN source_walk walk
                  ON walk.source_type = 'nodes'
                 AND child.node_id = walk.source_id
                JOIN json_each(child.source_ids) j
            )
            SELECT 1
            FROM source_walk walk
            JOIN messages m
              ON walk.source_type = 'messages'
             AND m.store_id = walk.source_id
            WHERE CASE
                    WHEN ? = ? THEN (m.source = ? OR {legacy_blank_clause})
                    ELSE m.source = ?
                  END
            LIMIT 1
            """,
            (node_id, normalized_source, _UNKNOWN_SOURCE, normalized_source, normalized_source),
        ).fetchone()
        matched = row is not None
        if cache is not None:
            cache[node_id] = matched
        return matched

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
            expand_hint=row[12] or "",
            seq=int(row[13] or 0) if len(row) > 13 else 0,
            search_rank=row[14] if len(row) > 14 else None,
        )

    def close(self, reason: str = "closed") -> None:
        """Close the connection once a read of this helper on another thread has
        finished (its lock); later use raises. The engine closes its helpers at plugin
        unload and when the engine is collected (``LCMEngine.close``)."""
        with self._db_lock:
            self._conn = close_connection(self._conn, db_path=self.db_path, reason=reason, owner="the summary reader")
