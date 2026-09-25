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
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .db_bootstrap import (
    MESSAGES_FTS_SPEC,
    ExternalContentFtsSpec,
    LockedConnection,
    StoreRefusedError,
    close_connection,
    open_store,
    refuse_cross_vm_filesystem,
)
from .search_query import (
    build_snippet,
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
    AGE_DECAY_RATE,
    should_apply_directness_rank_adjustment,
)
from .sqlite_util import _create_private_sqlite_file

logger = logging.getLogger(__name__)


_MESSAGE_ROLE_BIAS_SQL = "CASE m.role WHEN 'user' THEN 0 WHEN 'assistant' THEN 1 WHEN 'tool' THEN 2 ELSE 1 END"
_MESSAGE_SELECT_COLUMNS = (
    "store_id, session_id, source, role, content, tool_call_id, "
    "tool_calls, tool_name, timestamp, token_estimate, pinned, conversation_id, "
    "ingested_at, observed_at, observed_at_source, seq, revises_node_id"
)
_MESSAGE_SELECT_COLUMN_COUNT = len(_MESSAGE_SELECT_COLUMNS.split(","))
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


def _source_filter_clause(column: str, source: str | None) -> tuple[str | None, list[str]]:
    normalized = _normalize_source_value(source) if source is not None else ""
    if not normalized:
        return None, []
    if normalized == _UNKNOWN_SOURCE:
        return f"({column} = ? OR {_legacy_blank_source_clause(column)})", [_UNKNOWN_SOURCE]
    return f"{column} = ?", [normalized]


def _conversation_filter_clause(column: str, conversation_id: str | None) -> tuple[str | None, list[str]]:
    normalized = _normalize_conversation_id_value(conversation_id)
    if not normalized:
        return None, []
    return f"{column} = ?", [normalized]


def _message_role_bias(role: str | None) -> float:
    if role == "user":
        return 0.0
    if role == "assistant":
        return 1.0
    if role == "tool":
        return 2.0
    return 1.0


def _message_directness_score(role: str | None, content: str | None, terms: List[str], phrases: List[str] | None = None) -> float:
    score = compute_directness_score(content or "", terms, phrases)
    if role == "tool":
        stripped = (content or "").lstrip()
        if stripped.startswith("{") or stripped.startswith("["):
            score -= 4.0
    return score


def _build_search_order_by(
    sort: str | None,
    timestamp_expr: str,
    role_penalty_expr: str | None = None,
    order_expr: str = "m.seq",
) -> str:
    """Recency is the transcript order (``order_expr``, the view's ``seq``), never
    the ids; ``timestamp_expr`` only ages a hit for the hybrid blend."""
    normalized = normalize_search_sort(sort)
    order_parts: list[str] = []
    if normalized == "relevance":
        if role_penalty_expr:
            order_parts.extend(["rank ASC", f"{role_penalty_expr} ASC", f"{order_expr} DESC"])
        else:
            order_parts.extend(["rank ASC", f"{order_expr} DESC"])
        return ", ".join(order_parts)
    if normalized == "hybrid":
        blended = f"(rank / (1 + (MAX(0.0, ((strftime('%s','now') - {timestamp_expr}) / 3600.0)) * {AGE_DECAY_RATE})))"
        if role_penalty_expr:
            order_parts.extend([f"{blended} ASC", f"{role_penalty_expr} ASC", f"{order_expr} DESC"])
        else:
            order_parts.extend([f"{blended} ASC", f"{order_expr} DESC"])
        return ", ".join(order_parts)
    order_parts.append(f"{order_expr} DESC")
    if role_penalty_expr:
        order_parts.append(f"{role_penalty_expr} ASC")
    order_parts.append("rank ASC")
    return ", ".join(order_parts)


def _fallback_result_sort_key(result: Dict[str, Any], sort: str | None) -> tuple[float, float, float, float]:
    normalized = normalize_search_sort(sort)
    score = float(result.get("_fallback_score") or 0.0)
    directness = float(result.get("_directness_score") or 0.0)
    timestamp = float(result.get("timestamp") or 0.0)
    order = float(result.get("seq") or 0.0)
    role_bias = _message_role_bias(result.get("role"))

    if normalized == "relevance":
        return (-score, -directness, role_bias, -order)
    if normalized == "hybrid":
        age_hours = max(0.0, (time.time() - timestamp) / 3600.0)
        blended = score / (1 + (age_hours * AGE_DECAY_RATE))
        return (-blended, -directness, role_bias, -order)
    return (-order, role_bias, -score, -directness)


def _fts_result_sort_key(result: Dict[str, Any], sort: str | None) -> tuple[float, float, float, float]:
    normalized = normalize_search_sort(sort)
    rank = result.get("search_rank")
    rank_value = float(rank) if rank is not None else float("inf")
    directness = float(result.get("_directness_score") or 0.0)
    timestamp = float(result.get("timestamp") or 0.0)
    order = float(result.get("seq") or 0.0)
    role_bias = _message_role_bias(result.get("role"))

    if normalized == "relevance":
        return (rank_value, -directness, role_bias, -order)
    if normalized == "hybrid":
        age_hours = max(0.0, (time.time() - timestamp) / 3600.0)
        blended = rank_value / (1 + (age_hours * AGE_DECAY_RATE)) if rank is not None else float("inf")
        return (blended, -directness, role_bias, -order)
    return (-order, role_bias, rank_value, 0.0)


def _fts_primary_value(result: Dict[str, Any], sort: str | None) -> float:
    normalized = normalize_search_sort(sort)
    rank = result.get("search_rank")
    rank_value = float(rank) if rank is not None else float("inf")
    if normalized == "hybrid":
        timestamp = float(result.get("timestamp") or 0.0)
        age_hours = max(0.0, (time.time() - timestamp) / 3600.0)
        return rank_value / (1 + (age_hours * AGE_DECAY_RATE)) if rank is not None else float("inf")
    return rank_value


def build_message_fts_spec() -> ExternalContentFtsSpec:
    return MESSAGES_FTS_SPEC


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
            # Refuses a database this plugin did not write, creates the store in an
            # empty one, and repairs a damaged full-text index. No DDL otherwise.
            open_store(self._conn, self.db_path, check_fts=True)
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

    def get_batch(self, store_ids: List[int]) -> Dict[int, Dict[str, Any]]:
        """Retrieve multiple messages by store_id in a single query.

        Returns a dict mapping store_id → message dict.
        """
        if not store_ids:
            return {}
        placeholders = ",".join("?" for _ in store_ids)
        rows = self._locked.execute(
            f"SELECT {_MESSAGE_SELECT_COLUMNS} FROM messages WHERE store_id IN ({placeholders})",
            store_ids,
        ).fetchall()
        return {row[0]: self._row_to_dict(row) for row in rows}

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

    # -- Search -------------------------------------------------------------

    def search(self, query: str, session_id: str | None = None,
               limit: int = 20, sort: str | None = None,
               source: str | None = None,
               conversation_id: str | None = None,
               role: str | None = None,
               time_from: float | None = None,
               time_to: float | None = None,
               allow_operators: bool = False) -> List[Dict[str, Any]]:
        """FTS5 search across raw messages.

        Retrieval contract:
        - ``session_id`` limits which sessions are eligible
        - ``session_id=None`` means all sessions; an empty string is treated as
          a literal session id
        - ``source`` limits which raw rows inside those sessions are eligible
        - ``source='unknown'`` means the explicit unknown-source bucket, with
          legacy blank-source rows treated as equivalent for back-compat
        - ``conversation_id`` limits rows to one gateway conversation/session key
        - ``allow_operators`` marks a query the CALLER composed as FTS5 syntax,
          keeping its bare AND/OR/NOT/NEAR. Never set it for user or agent text
        """
        safe_query = sanitize_fts5_query(query, allow_operators=allow_operators)
        terms = extract_search_terms(safe_query)
        phrases = extract_quoted_phrases(safe_query)
        # LIKE is the fallback for text sanitization LOSES (CJK/emoji) and for a
        # query with no term left after it. A raw natural-language question is
        # NOT one of those: it sanitizes to a term form the index answers, so it
        # stays on the FTS path (F31 §3).
        if requires_like_fallback(query, safe_query):
            return self._search_like(
                query,
                session_id=session_id,
                limit=limit,
                sort=sort,
                source=source,
                conversation_id=conversation_id,
                role=role,
                time_from=time_from,
                time_to=time_to,
            )

        order_by = _build_search_order_by(
            sort,
            "m.timestamp",
            _MESSAGE_ROLE_BIAS_SQL,
        )
        fetch_limit = compute_search_fetch_limit(limit, terms, phrases)
        candidate_cap = compute_search_candidate_cap(limit)
        apply_directness_adjustment = should_apply_directness_rank_adjustment(terms, phrases)
        max_rank_bonus = compute_directness_rank_bonus_upper_bound(terms, phrases) * 3e-7
        source_clause, source_args = _source_filter_clause("m.source", source)
        conversation_clause, conversation_args = _conversation_filter_clause("m.conversation_id", conversation_id)
        offset = 0
        scanned_rows = 0
        results: list[Dict[str, Any]] = []
        while True:
            try:
                where = ["messages_fts MATCH ?"]
                args: list[Any] = [safe_query]
                if session_id is not None:
                    where.append("m.session_id = ?")
                    args.append(session_id)
                if source_clause:
                    where.append(source_clause)
                    args.extend(source_args)
                if conversation_clause:
                    where.append(conversation_clause)
                    args.extend(conversation_args)
                if role is not None:
                    where.append("m.role = ?")
                    args.append(role)
                if time_from is not None:
                    where.append("m.timestamp >= ?")
                    args.append(time_from)
                if time_to is not None:
                    where.append("m.timestamp <= ?")
                    args.append(time_to)
                args.extend([fetch_limit, offset])
                rows = self._locked.execute(
                    f"""SELECT m.store_id, m.session_id, m.source, m.role, m.content, m.tool_call_id,
                              m.tool_calls, m.tool_name, m.timestamp, m.token_estimate, m.pinned, m.conversation_id,
                              m.ingested_at, m.observed_at, m.observed_at_source, m.seq, m.revises_node_id,
                              rank as search_rank,
                              snippet(messages_fts, 0, '>>>', '<<<', '...', 40) as snippet
                       FROM messages_fts fts
                       JOIN messages m ON m.store_id = fts.rowid
                       WHERE {' AND '.join(where)}
                       ORDER BY {order_by} LIMIT ? OFFSET ?""",
                    args,
                ).fetchall()
                scanned_rows += len(rows)
            except sqlite3.Error as exc:
                logger.warning("FTS message search failed, falling back to LIKE: %s", exc)
                return self._search_like(
                    query,
                    session_id=session_id,
                    limit=limit,
                    sort=sort,
                    source=source,
                    conversation_id=conversation_id,
                    role=role,
                    time_from=time_from,
                    time_to=time_to,
                )

            raw_primary_values: list[float] = []
            for r in rows:
                d = self._row_to_dict(r)
                base_columns = _MESSAGE_SELECT_COLUMN_COUNT
                d["search_rank"] = r[base_columns] if len(r) > base_columns else None
                d["snippet"] = r[base_columns + 1] if len(r) > (base_columns + 1) else ""
                d["_directness_score"] = _message_directness_score(d.get("role"), d.get("content"), terms, phrases)
                if apply_directness_adjustment and d["search_rank"] is not None:
                    rank_adjustment = max(float(d["_directness_score"]), 0.0)
                    d["search_rank"] = float(d["search_rank"]) - (rank_adjustment * 3e-7)
                raw_primary_values.append(_fts_primary_value(d, sort))
                results.append(d)
            results.sort(key=lambda result: _fts_result_sort_key(result, sort))

            if not apply_directness_adjustment or len(rows) < fetch_limit or len(results) <= limit:
                return results[:limit]

            worst_visible_primary = _fts_primary_value(results[min(limit, len(results)) - 1], sort)
            last_fetched_primary = raw_primary_values[-1]
            best_unseen_primary = last_fetched_primary - max_rank_bonus
            if best_unseen_primary > worst_visible_primary:
                return results[:limit]

            if scanned_rows >= candidate_cap:
                return results[:limit]

            offset += len(rows)
            remaining = candidate_cap - scanned_rows
            if remaining <= 0:
                return results[:limit]
            fetch_limit = min(fetch_limit * 2, remaining)

    def _search_like(self, query: str, session_id: str | None = None,
                     limit: int = 20, sort: str | None = None,
                     source: str | None = None,
                     conversation_id: str | None = None,
                     role: str | None = None,
                     time_from: float | None = None,
                     time_to: float | None = None) -> List[Dict[str, Any]]:
        # LIKE keeps every character the index cannot spell (emoji, punctuation)
        # because substring matching is the only way to find those rows.
        safe_query = sanitize_like_query(query)
        terms = extract_search_terms(safe_query)
        phrases = extract_quoted_phrases(safe_query)
        if not terms:
            return []
        fetch_limit = compute_search_fetch_limit(limit, terms, phrases)

        where: list[str] = ["content IS NOT NULL"]
        args: list[Any] = []
        if session_id is not None:
            where.append("session_id = ?")
            args.append(session_id)
        source_clause, source_args = _source_filter_clause("source", source)
        if source_clause:
            where.append(source_clause)
            args.extend(source_args)
        conversation_clause, conversation_args = _conversation_filter_clause("conversation_id", conversation_id)
        if conversation_clause:
            where.append(conversation_clause)
            args.extend(conversation_args)
        if role is not None:
            where.append("role = ?")
            args.append(role)
        if time_from is not None:
            where.append("timestamp >= ?")
            args.append(time_from)
        if time_to is not None:
            where.append("timestamp <= ?")
            args.append(time_to)
        like_clauses = []
        for term in terms:
            like_clauses.append("content LIKE ? ESCAPE '\\'")
            args.append(f"%{escape_like(term)}%")
        where.append("(" + " OR ".join(like_clauses) + ")")
        fetch_limit = compute_like_fallback_fetch_limit(limit, terms, phrases)
        base_args = list(args)
        normalized_sort = normalize_search_sort(sort)
        results: List[Dict[str, Any]] = []
        collapse_risky_repeats = contains_risky_fts_ascii(query)
        order_by = ""
        order_args: list[Any] = []
        role_bias = "CASE role WHEN 'user' THEN 0 WHEN 'assistant' THEN 1 WHEN 'tool' THEN 2 ELSE 1 END"

        def count_expr(term: str) -> tuple[str, list[Any]]:
            return (
                "((LENGTH(LOWER(content)) - LENGTH(REPLACE(LOWER(content), LOWER(?), ''))) "
                "/ NULLIF(LENGTH(?), 0))",
                [term, term],
            )

        if normalized_sort == "recency":
            score_exprs: list[str] = []
            for term in terms:
                if collapse_risky_repeats:
                    score_exprs.append("CASE WHEN content LIKE ? ESCAPE '\\' THEN 1 ELSE 0 END")
                    order_args.append(f"%{escape_like(term)}%")
                else:
                    expr, expr_args = count_expr(term)
                    score_exprs.append(expr)
                    order_args.extend(expr_args)
            score_expr = " + ".join(score_exprs) if score_exprs else "0"

            def build_unique_exprs(selected_terms: list[str]) -> tuple[str, list[Any]]:
                parts: list[str] = []
                expr_args: list[Any] = []
                for selected_term in selected_terms:
                    expr, args_for_expr = count_expr(selected_term)
                    parts.append(f"CASE WHEN ({expr}) > 0 THEN 1 ELSE 0 END")
                    expr_args.extend(args_for_expr)
                return (" + ".join(parts) if parts else "0", expr_args)

            def build_total_exprs(selected_terms: list[str]) -> tuple[str, list[Any]]:
                parts: list[str] = []
                expr_args: list[Any] = []
                for selected_term in selected_terms:
                    expr, args_for_expr = count_expr(selected_term)
                    parts.append(expr)
                    expr_args.extend(args_for_expr)
                return (" + ".join(parts) if parts else "0", expr_args)

            directness_args: list[Any] = []
            unique_score_expr, expr_args = build_unique_exprs(terms)
            directness_args.extend(expr_args)
            normalized_phrases = {(phrase or "").strip().lower() for phrase in phrases if (phrase or "").strip()}
            if phrases:
                phrase_hit_exprs: list[str] = []
                for phrase in phrases:
                    phrase_hit_exprs.append("CASE WHEN INSTR(LOWER(content), LOWER(?)) > 0 THEN 1 ELSE 0 END")
                    directness_args.append(phrase)
                phrase_hit_expr = " + ".join(phrase_hit_exprs) if phrase_hit_exprs else "0"
                non_phrase_terms = [term for term in terms if term.strip().lower() not in normalized_phrases]
                non_phrase_total_expr, expr_args = build_total_exprs(non_phrase_terms)
                directness_args.extend(expr_args)
                non_phrase_unique_expr, expr_args = build_unique_exprs(non_phrase_terms)
                directness_args.extend(expr_args)
                repetition_expr = f"MAX(({non_phrase_total_expr}) - ({non_phrase_unique_expr}), 0)"
                directness_expr = f"(({unique_score_expr}) * 5.0) + (({phrase_hit_expr}) * 8.0) - MIN(({repetition_expr}), 6)"
            else:
                total_repetition_expr, expr_args = build_total_exprs(terms)
                directness_args.extend(expr_args)
                unique_repetition_expr, expr_args = build_unique_exprs(terms)
                directness_args.extend(expr_args)
                repetition_expr = f"MAX(({total_repetition_expr}) - ({unique_repetition_expr}), 0)"
                directness_expr = f"(({unique_score_expr}) * 5.0) - MIN(({repetition_expr}), 6)"
            order_args.extend(directness_args)
            order_by = (
                f"ORDER BY seq DESC, {role_bias} ASC, ({score_expr}) DESC, "
                f"({directness_expr}) DESC, store_id DESC"
            )

        def add_rows(rows: list[sqlite3.Row]) -> None:
            for row in rows:
                result = self._row_to_dict(row)
                content = result.get("content") or ""
                score = sum(
                    min(count_term_matches(content, term), 1) if collapse_risky_repeats else count_term_matches(content, term)
                    for term in terms
                )
                if score <= 0:
                    continue
                result["search_rank"] = -float(score)
                result["snippet"] = build_snippet(content, terms)
                result["_fallback_score"] = float(score)
                result["_directness_score"] = _message_directness_score(result.get("role"), content, terms, phrases)
                results.append(result)

        if normalized_sort == "recency":
            candidate_cap = compute_search_candidate_cap(limit)
            offset = 0
            scanned_rows = 0
            while True:
                batch_limit = min(fetch_limit, candidate_cap - scanned_rows)
                if batch_limit <= 0:
                    break
                rows = self._locked.execute(
                    f"""SELECT {_MESSAGE_SELECT_COLUMNS}
                        FROM messages
                        WHERE {' AND '.join(where)}
                        {order_by}
                        LIMIT ? OFFSET ?""",
                    [*base_args, *order_args, batch_limit, offset],
                ).fetchall()
                scanned_rows += len(rows)
                add_rows(rows)
                offset += len(rows)
                if len(rows) < batch_limit:
                    break
                if scanned_rows >= candidate_cap:
                    boundary_timestamp = rows[-1][8]
                    boundary_role_bias = _message_role_bias(rows[-1][3])
                    while True:
                        tie_rows = self._locked.execute(
                            f"""SELECT {_MESSAGE_SELECT_COLUMNS}
                                FROM messages
                                WHERE {' AND '.join(where)}
                                {order_by}
                                LIMIT ? OFFSET ?""",
                            [*base_args, *order_args, fetch_limit, offset],
                        ).fetchall()
                        if not tie_rows:
                            break
                        matching_tie_rows = []
                        reached_next_primary_group = False
                        for tie_row in tie_rows:
                            if tie_row[8] == boundary_timestamp and _message_role_bias(tie_row[3]) == boundary_role_bias:
                                matching_tie_rows.append(tie_row)
                            else:
                                reached_next_primary_group = True
                                break
                        add_rows(matching_tie_rows)
                        if reached_next_primary_group or len(tie_rows) < fetch_limit:
                            break
                        offset += len(tie_rows)
                    break
        else:
            # Deterministic relevance/hybrid candidate scan for LIKE fallback.
            # Apply the same coarse score/directness ordering before the hard
            # candidate cap that Python uses below; otherwise a recent-biased
            # window can exclude older but materially better relevance matches.
            score_exprs: list[str] = []
            order_args = []
            for term in terms:
                if collapse_risky_repeats:
                    score_exprs.append("CASE WHEN content LIKE ? ESCAPE '\\' THEN 1 ELSE 0 END")
                    order_args.append(f"%{escape_like(term)}%")
                else:
                    expr, expr_args = count_expr(term)
                    score_exprs.append(expr)
                    order_args.extend(expr_args)
            score_expr = " + ".join(score_exprs) if score_exprs else "0"
            exact_query = (query or "").strip()
            exact_expr = "CASE WHEN LOWER(content) = LOWER(?) THEN 1 ELSE 0 END" if exact_query else "0"
            exact_args: list[Any] = [exact_query] if exact_query else []
            directness_expr = "0.0 + 0"

            if normalized_sort == "hybrid":
                primary_expr = (
                    f"(({score_expr}) / (1 + (MAX(0.0, "
                    f"((strftime('%s','now') - timestamp) / 3600.0)) * {AGE_DECAY_RATE})))"
                )
            else:
                primary_expr = f"({score_expr})"

            order_by = (
                f"ORDER BY {primary_expr} DESC, ({exact_expr}) DESC, ({directness_expr}) DESC, "
                f"{role_bias} ASC, seq DESC"
            )
            candidate_cap = compute_search_candidate_cap(limit)
            offset = 0
            while offset < candidate_cap:
                batch_limit = min(fetch_limit, candidate_cap - offset)
                rows = self._locked.execute(
                    f"""SELECT {_MESSAGE_SELECT_COLUMNS}
                        FROM messages
                        WHERE {' AND '.join(where)}
                        {order_by}
                        LIMIT ? OFFSET ?""",
                    [*base_args, *order_args, *exact_args, batch_limit, offset],
                ).fetchall()
                if not rows:
                    break
                add_rows(rows)
                offset += len(rows)
                if len(rows) < batch_limit:
                    break

        results.sort(key=lambda result: _fallback_result_sort_key(result, sort))
        for result in results:
            result.pop("_fallback_score", None)
        return results[:limit]

    # -- Helpers ------------------------------------------------------------

    def _row_to_dict(self, row) -> Dict[str, Any]:
        """Convert a sqlite3 row to a dict."""
        if row is None:
            return {}
        cols = [
            "store_id", "session_id", "source", "role", "content", "tool_call_id",
            "tool_calls", "tool_name", "timestamp", "token_estimate", "pinned", "conversation_id",
            "ingested_at", "observed_at", "observed_at_source", "seq", "revises_node_id",
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
        quick checks, FTS sync counts, schema health -- that need ad-hoc
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
