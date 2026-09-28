"""Tool handlers for LCM — the code that runs when the LLM calls each tool."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, TYPE_CHECKING

from . import expansion
from . import grep as grep_tool
from . import query as query_tool
from .diagnostics import doctor_guidance_for_checks
from .db_bootstrap import inspect_lcm_schema_health

if TYPE_CHECKING:
    from .engine import LCMEngine


logger = logging.getLogger(__name__)


def _require_engine(kwargs: Dict[str, Any]) -> "LCMEngine | None":
    engine = kwargs.get("engine")
    return engine if engine is not None else None


def _parse_strict_int(value: Any, name: str) -> tuple[int | None, str | None]:
    try:
        if isinstance(value, bool):
            raise ValueError
        return int(value), None
    except (TypeError, ValueError, OverflowError):
        return None, f"{name} must be an integer"


_LCM_INSPECT_DEFAULT_LIMIT = 20
_LCM_INSPECT_HARD_LIMIT_CAP = 200
_LCM_INSPECT_MAX_RESPONSE_CHARS = 20_000
_OPERATOR_TEXT_FIELD_MAX_CHARS = 1_000


def _bounded_operator_field(value: object) -> tuple[str, bool]:
    text = str(value or "")
    if len(text) <= _OPERATOR_TEXT_FIELD_MAX_CHARS:
        return text, False
    suffix = "..."
    return text[: _OPERATOR_TEXT_FIELD_MAX_CHARS - len(suffix)] + suffix, True


def _bound_operator_strings(value: Any) -> tuple[Any, int]:
    """Return a JSON-compatible copy with every free-text field bounded."""
    if isinstance(value, str):
        bounded, truncated = _bounded_operator_field(value)
        return bounded, int(truncated)
    if isinstance(value, list):
        result: list[Any] = []
        truncated_fields = 0
        for item in value:
            bounded, count = _bound_operator_strings(item)
            result.append(bounded)
            truncated_fields += count
        return result, truncated_fields
    if isinstance(value, dict):
        result: dict[Any, Any] = {}
        truncated_fields = 0
        for key, item in value.items():
            bounded, count = _bound_operator_strings(item)
            result[key] = bounded
            truncated_fields += count
        return result, truncated_fields
    return value, 0


def _bounded_inspect_json(response: dict[str, Any]) -> str:
    """Serialize ``lcm_inspect`` under one final response-size invariant."""
    payload, truncated_fields = _bound_operator_strings(response)
    total_truncated_fields = truncated_fields
    payload["char_limit"] = _LCM_INSPECT_MAX_RESPONSE_CHARS
    payload["truncated"] = bool(total_truncated_fields)
    if total_truncated_fields:
        payload["truncated_field_count"] = total_truncated_fields
    encoded = json.dumps(payload, ensure_ascii=False)
    if len(encoded) <= _LCM_INSPECT_MAX_RESPONSE_CHARS:
        return encoded

    # If cardinality rather than one text field exceeds the cap, keep whole
    # top-level sections in a deterministic priority order.  Never cut encoded
    # JSON mid-token; omitted sections are reported explicitly.
    priority = [
        "read_only",
        "session_id",
        "conversation_id",
        "limit",
        "runtime_identity",
        "lineage",
        "messages",
        "compaction",
        "dag",
        "filters",
        "limit_clamped_from",
    ]
    compact: dict[str, Any] = {
        "char_limit": _LCM_INSPECT_MAX_RESPONSE_CHARS,
        "truncated": True,
        "truncation": {
            "reason": "response_char_limit",
            "omitted_top_level_sections": [],
        },
    }
    if total_truncated_fields:
        compact["truncated_field_count"] = total_truncated_fields
    retained: list[str] = []
    omitted: list[str] = []
    ordered_keys = priority + [key for key in payload if key not in priority]
    for key in dict.fromkeys(ordered_keys):
        if key in {"char_limit", "truncated", "truncated_field_count"}:
            continue
        if key not in payload:
            continue
        compact[key] = payload[key]
        if len(json.dumps(compact, ensure_ascii=False)) <= _LCM_INSPECT_MAX_RESPONSE_CHARS - 1_000:
            retained.append(key)
        else:
            compact.pop(key)
            omitted.append(key)
    compact["truncation"]["omitted_top_level_sections"] = omitted
    encoded = json.dumps(compact, ensure_ascii=False)
    while len(encoded) > _LCM_INSPECT_MAX_RESPONSE_CHARS and retained:
        key = retained.pop()
        compact.pop(key, None)
        omitted.append(key)
        compact["truncation"]["omitted_top_level_sections"] = omitted
        encoded = json.dumps(compact, ensure_ascii=False)
    return encoded


def lcm_grep(args: Dict[str, Any], **kwargs) -> Any:
    """Search what the session's agent said and did for a term, and return the chunks it
    lies in (``grep``, #18 D2). The result is the final string, at most the host's spill
    threshold."""
    engine = _require_engine(kwargs)
    if engine is None:
        return json.dumps({"error": "LCM engine not initialized"})
    try:
        return grep_tool.grep(engine, args, messages=kwargs.get("messages"))
    except expansion.ExpansionError as exc:
        return json.dumps({"error": str(exc)}, ensure_ascii=False)
    except Exception as exc:
        # A failure is said as what it is, never shown as "no hits", and never given a cause
        # the code did not establish (any step may have raised).
        logger.warning("lcm_grep failed", exc_info=True)
        return json.dumps({"error": f"lcm_grep failed ({type(exc).__name__}: {exc})"}, ensure_ascii=False)


_LCM_EXPAND_REMOVED_ARGUMENTS = (
    "node_id", "store_id", "max_tokens", "source_offset", "source_limit", "content_offset",
    "include_exact_ref", "externalized_ref",
)


def lcm_expand(args: Dict[str, Any], **kwargs) -> Any:
    """Look behind a handle: one page of what it opens into (``expansion``, #18). The
    result is the final string, or the ``_multimodal`` envelope where the page holds an
    image; either way at most the host's spill threshold."""
    engine = _require_engine(kwargs)
    if engine is None:
        return json.dumps({"error": "LCM engine not initialized"})
    removed = [name for name in _LCM_EXPAND_REMOVED_ARGUMENTS if name in args]
    if removed:
        return json.dumps({
            "error": "lcm_expand no longer accepts: " + ", ".join(removed)
                     + ". It takes a handle (m, t, c or s and eight characters), raw, and page.",
        })
    unknown = [name for name in args if name not in ("handle", "raw", "page")]
    if unknown:
        return json.dumps({"error": "lcm_expand takes handle, raw and page; not " + ", ".join(unknown)})
    try:
        return expansion.expand(engine, args, messages=kwargs.get("messages"))
    except expansion.ExpansionError as exc:
        return json.dumps({"error": str(exc)}, ensure_ascii=False)
    except Exception as exc:
        # A failure is said as what it is, never shown as an empty stretch, and never given a
        # cause the code did not establish (any step may have raised).
        logger.warning("lcm_expand failed", exc_info=True)
        return json.dumps({"error": f"lcm_expand failed ({type(exc).__name__}: {exc})"}, ensure_ascii=False)


def lcm_query(args: Dict[str, Any], **kwargs) -> Any:
    """Ask a question over what stands behind summaries or chunks the agent holds (``query``,
    #19): one call to the summariser's model over the raw of every chunk, a hedged report and
    excerpts checked against the store. The result is the final string, at most the host's
    spill threshold."""
    engine = _require_engine(kwargs)
    if engine is None:
        return json.dumps({"error": "LCM engine not initialized"})
    try:
        return query_tool.query(engine, args, messages=kwargs.get("messages"), interrupted=kwargs.get("interrupted"))
    except expansion.ExpansionError as exc:
        return json.dumps({"error": str(exc)}, ensure_ascii=False)
    except Exception as exc:
        # A failure is said as what it is, never as an answer, and never given a cause the
        # code did not establish (any step may have raised).
        logger.warning("lcm_query failed", exc_info=True)
        return json.dumps({"error": f"lcm_query failed ({type(exc).__name__}: {exc})"}, ensure_ascii=False)


def _summary_quality_stats(engine: "LCMEngine", session_id: str) -> dict[str, Any]:
    """Return read-only summary compression quality diagnostics for one session."""
    conn = engine._dag.connection
    if conn is None:
        raise RuntimeError("LCM DAG connection is not initialized")
    rows = conn.execute(
        """
        SELECT node_id, session_id, depth, token_count, source_token_count, source_uncounted_images
        FROM summary_nodes
        WHERE session_id = ? AND source_token_count > 0
        ORDER BY
            CASE WHEN token_count <= 0 THEN 1 ELSE 0 END DESC,
            CASE WHEN token_count > 0
                 THEN CAST(source_token_count AS REAL) / token_count
                 ELSE source_token_count
            END DESC
        LIMIT 5
        """,
        (session_id,),
    ).fetchall()
    totals = conn.execute(
        """
        SELECT
            COUNT(*),
            COALESCE(SUM(source_token_count), 0),
            COALESCE(SUM(token_count), 0),
            SUM(CASE WHEN source_token_count >= 100000
                      AND token_count < 500 THEN 1 ELSE 0 END),
            SUM(CASE WHEN token_count > 0
                      AND CAST(source_token_count AS REAL) / token_count >= 400
                     THEN 1 ELSE 0 END),
            COALESCE(SUM(source_uncounted_images), 0)
        FROM summary_nodes
        WHERE session_id = ?
        """,
        (session_id,),
    ).fetchone()
    total_nodes = int(totals[0] or 0)
    total_source_tokens = int(totals[1] or 0)
    total_summary_tokens = int(totals[2] or 0)
    tiny_large_source_nodes = int(totals[3] or 0)
    extreme_ratio_nodes = int(totals[4] or 0)
    total_source_uncounted_images = int(totals[5] or 0)
    overall_ratio = (
        round(total_source_tokens / total_summary_tokens, 1)
        if total_summary_tokens > 0
        else 0.0
    )
    worst_nodes = []
    for node_id, session_id, depth, token_count, source_token_count, source_uncounted_images in rows:
        ratio = (
            round(float(source_token_count) / float(token_count), 1)
            if token_count and token_count > 0
            else None
        )
        worst_nodes.append({
            "node_id": int(node_id),
            "session_id": session_id,
            "depth": int(depth),
            "source_token_count": int(source_token_count or 0),
            "source_uncounted_images": int(source_uncounted_images or 0),
            "token_count": int(token_count or 0),
            "compression_ratio": ratio,
        })
    return {
        "total_nodes": total_nodes,
        "session_id": session_id,
        "total_source_tokens": total_source_tokens,
        "total_source_uncounted_images": total_source_uncounted_images,
        "total_summary_tokens": total_summary_tokens,
        "overall_compression_ratio": overall_ratio,
        "extreme_ratio_threshold": 400,
        "tiny_large_source_threshold": {
            "source_token_count_min": 100000,
            "token_count_max": 500,
        },
        "extreme_ratio_nodes": extreme_ratio_nodes,
        "tiny_large_source_nodes": tiny_large_source_nodes,
        "worst_nodes": worst_nodes,
        "recommendation": (
            "Inspect worst_nodes with lcm_expand; tiny summaries for very large sources often indicate degraded fallback summarization."
            if extreme_ratio_nodes or tiny_large_source_nodes
            else "summary compression ratios are within the diagnostic thresholds"
        ),
    }


def _inspect_message_metadata(row: dict[str, Any]) -> dict[str, Any]:
    """Return row metadata only; never include raw message content."""
    content = row.get("content") or ""
    item: dict[str, Any] = {
        "store_id": row.get("store_id"),
        "session_id": row.get("session_id") or "",
        "source": row.get("source") or "",
        "conversation_id": row.get("conversation_id") or "",
        "role": row.get("role") or "unknown",
        "timestamp": row.get("timestamp", 0),
        "token_estimate": row.get("token_estimate", 0),
        "uncounted_images": int(row.get("uncounted_images") or 0),
        "content_chars": len(content),
    }
    if row.get("tool_call_id"):
        item["tool_call_id"] = row.get("tool_call_id")
    if row.get("tool_name"):
        item["tool_name"] = row.get("tool_name")
    return item


def _inspect_last_compacted_store_id(engine: "LCMEngine", session_id: str) -> int | None:
    """The last record in transcript order that a summary covers (ids are the order
    of writing, not the transcript's)."""
    row = engine._store.connection.execute(
        """
        SELECT m.store_id FROM messages m
        WHERE m.session_id = ? AND m.store_id NOT IN (
            SELECT r.record_id FROM latest_returns l JOIN records r ON r.handle = l.record
            WHERE l.kind = 'record' AND r.session = ?)
          AND m.revises_node_id IS NULL
        ORDER BY m.seq DESC LIMIT 1
        """,
        (session_id, session_id),
    ).fetchone()
    return int(row[0]) if row else None


def lcm_inspect(args: Dict[str, Any], **kwargs) -> str:
    """Return a read-only metadata inventory of the current LCM session."""
    engine = _require_engine(kwargs)
    if engine is None:
        return json.dumps({"error": "LCM engine not initialized"})

    raw_limit_arg = args.get("limit", _LCM_INSPECT_DEFAULT_LIMIT)
    parsed_limit, limit_error = _parse_strict_int(raw_limit_arg, "limit")
    if limit_error:
        return json.dumps({"error": limit_error})
    if parsed_limit is None or parsed_limit <= 0:
        return json.dumps({"error": "limit must be a positive integer"})
    requested_limit = parsed_limit
    limit = min(requested_limit, _LCM_INSPECT_HARD_LIMIT_CAP)

    session_id = engine.current_session_id
    conversation_id = engine.current_conversation_id
    if not session_id:
        full_status = engine.get_status()
        return _bounded_inspect_json({
            "error": "No active session",
            "read_only": True,
            "runtime_identity": full_status.get("runtime_identity") or engine.get_runtime_identity(),
        })

    full_status = engine.get_status()
    runtime_identity = full_status.get("runtime_identity") or engine.get_runtime_identity()

    store_totals_row = engine._store.connection.execute(
        """
        SELECT COUNT(*), COALESCE(SUM(token_estimate), 0),
               (SELECT store_id FROM messages WHERE session_id = ? ORDER BY seq ASC LIMIT 1),
               (SELECT store_id FROM messages WHERE session_id = ? ORDER BY seq DESC LIMIT 1),
               COALESCE(SUM(uncounted_images), 0)
        FROM messages
        WHERE session_id = ?
        """,
        (session_id, session_id, session_id),
    ).fetchone()
    message_total = int(store_totals_row[0] or 0) if store_totals_row else 0
    estimated_tokens = int(store_totals_row[1] or 0) if store_totals_row else 0
    first_store_id = store_totals_row[2] if store_totals_row else None
    last_store_id = store_totals_row[3] if store_totals_row else None
    estimated_uncounted_images = int(store_totals_row[4] or 0) if store_totals_row else 0
    # The fresh tail as the latest effective compaction returned it, in its positions.
    fresh_tail_rows = engine._store.get_returned_tail(session_id)
    fresh_tail_tokens = sum(int(row.get("token_estimate") or 0) for row in fresh_tail_rows)
    fresh_tail_uncounted_images = sum(int(row.get("uncounted_images") or 0) for row in fresh_tail_rows)
    fresh_tail_display_rows = fresh_tail_rows[-limit:]
    fresh_tail_items = [
        _inspect_message_metadata(row)
        for row in fresh_tail_display_rows
    ]

    depth_stats = engine._dag.get_session_depth_stats(session_id)
    total_dag_nodes = sum(info["count"] for info in depth_stats.values())
    total_dag_tokens = sum(info["tokens"] for info in depth_stats.values())
    total_dag_source_tokens = sum(info["source_tokens"] for info in depth_stats.values())
    total_dag_source_uncounted_images = sum(info["source_uncounted_images"] for info in depth_stats.values())
    latest_node_rows = engine._dag.connection.execute(
        """
        SELECT node_id, session_id, depth, token_count, source_token_count,
               source_type, created_at, earliest_at, latest_at, source_uncounted_images
        FROM summary_nodes
        WHERE session_id = ?
        ORDER BY seq DESC
        LIMIT ?
        """,
        (session_id, limit),
    ).fetchall()
    latest_nodes = [
        {
            "node_id": int(row[0]),
            "session_id": row[1],
            "depth": int(row[2]),
            "token_count": int(row[3] or 0),
            "source_token_count": int(row[4] or 0),
            "source_uncounted_images": int(row[9] or 0),
            "source_type": row[5],
            "created_at": row[6],
            "earliest_at": row[7],
            "latest_at": row[8],
        }
        for row in latest_node_rows
    ]

    last_compacted_store_id = _inspect_last_compacted_store_id(engine, session_id)

    platform = engine.current_session_platform

    response: dict[str, Any] = {
        "read_only": True,
        "session_id": session_id,
        "conversation_id": conversation_id,
        "limit": limit,
        "runtime_identity": runtime_identity,
        "lineage": {
            "session_id": session_id,
            "conversation_id": conversation_id,
            "session_platform": platform,
            "source_lineage": full_status.get("source_lineage"),
        },
        "messages": {
            "total": message_total,
            "estimated_tokens": estimated_tokens,
            "estimated_uncounted_images": estimated_uncounted_images,
            "first_store_id": first_store_id,
            "last_store_id": last_store_id,
            "effective_fresh_tail_count": len(fresh_tail_rows),
            "effective_fresh_tail_tokens": fresh_tail_tokens,
            "effective_fresh_tail_uncounted_images": fresh_tail_uncounted_images,
            "pre_tail_message_count": max(0, message_total - len(fresh_tail_rows)),
            "fresh_tail": {
                "returned": len(fresh_tail_items),
                "items": fresh_tail_items,
            },
        },
        "compaction": {
            "last": {
                "status": full_status.get("last_compression_status", "idle"),
                "noop_reason": full_status.get("last_compression_noop_reason", ""),
                "compression_count": engine.compression_count,
                "last_prompt_tokens": engine.last_prompt_tokens,
                "threshold_tokens": engine.threshold_tokens,
            },
            "frontier": {
                "last_compacted_store_id": last_compacted_store_id,
            },
        },
        "dag": {
            "total_nodes": total_dag_nodes,
            "total_tokens": total_dag_tokens,
            "total_source_tokens": total_dag_source_tokens,
            "total_source_uncounted_images": total_dag_source_uncounted_images,
            "depths": {f"d{depth}": info for depth, info in sorted(depth_stats.items())},
            "latest_nodes": latest_nodes,
        },
    }
    if requested_limit > _LCM_INSPECT_HARD_LIMIT_CAP:
        response["limit_clamped_from"] = requested_limit
    return _bounded_inspect_json(response)


def lcm_status(args: Dict[str, Any], **kwargs) -> str:
    """Quick health overview of the LCM engine for the current session."""
    engine = _require_engine(kwargs)
    if engine is None:
        return json.dumps({"error": "LCM engine not initialized"})

    session_id = engine.current_session_id
    if not session_id:
        return json.dumps({
            "error": "No active session",
            "runtime_identity": engine.get_runtime_identity(),
        })

    # Store stats
    store_messages = engine._store.get_session_count(session_id)
    store_tokens = engine._store.get_session_token_total(session_id)
    store_uncounted_images = engine._store.get_session_uncounted_images(session_id)

    # DAG stats by depth
    depths = engine._dag.get_session_depth_stats(session_id)

    total_dag_tokens = sum(d["tokens"] for d in depths.values())
    total_source_tokens = sum(d["source_tokens"] for d in depths.values())
    total_dag_nodes = sum(d["count"] for d in depths.values())
    compression_ratio = round(total_source_tokens / total_dag_tokens, 1) if total_dag_tokens > 0 else 0
    full_status = engine.get_status()
    source_lineage = full_status.get("source_lineage")
    runtime_identity = full_status.get("runtime_identity")
    config_sources = full_status.get("config_sources") or {}
    config_source_warnings = full_status.get("config_source_warnings") or []
    ignored_config_yaml_lcm_keys = full_status.get("ignored_config_yaml_lcm_keys") or []

    return json.dumps({
        "session_id": session_id,
        "compression_count": engine.compression_count,
        "total_compactions": full_status.get("total_compactions", 0),
        "total_compactions_scope": full_status.get("total_compactions_scope", ""),
        "last_compression_status": full_status.get("last_compression_status", "idle"),
        "last_compression_noop_reason": full_status.get("last_compression_noop_reason", ""),
        "model": full_status.get("model", ""),
        "provider": full_status.get("provider", ""),
        "raw_context_length": full_status.get("raw_context_length", engine.context_length),
        "context_length": engine.context_length,
        "effective_context_length_cap": full_status.get("effective_context_length_cap"),
        "effective_context_length_reason": full_status.get("effective_context_length_reason", ""),
        "context_length_source": full_status.get("context_length_source", ""),
        "geometry": full_status.get("geometry"),
        "tau": full_status.get("tau"),
        "tau_raised": full_status.get("tau_raised"),
        "target": full_status.get("target"),
        "turn": full_status.get("turn"),
        "native_compaction_refused": full_status.get("native_compaction_refused"),
        "threshold_tokens": engine.threshold_tokens,
        "last_prompt_tokens": engine.last_prompt_tokens,
        "last_input_tokens": engine.last_input_tokens,
        "last_output_tokens": engine.last_output_tokens,
        "last_cache_read_tokens": engine.last_cache_read_tokens,
        "last_cache_write_tokens": engine.last_cache_write_tokens,
        "last_reasoning_tokens": engine.last_reasoning_tokens,
        "cache_metrics_available": engine.cache_metrics_available,
        "cache_read_ratio": round(engine.cache_read_ratio, 4),
        "store": {
            "messages": store_messages,
            "estimated_tokens": store_tokens,
            "estimated_uncounted_images": store_uncounted_images,
        },
        "dag": {
            "total_nodes": total_dag_nodes,
            "total_tokens": total_dag_tokens,
            "compression_ratio": f"{compression_ratio}:1",
            "depths": {
                f"d{depth}": info for depth, info in sorted(depths.items())
            },
        },
        "config": {
            "fixed_prefix": engine._fixed_prefix_label(),
            "chunk_tokens": engine._config.chunk_tokens,
            "estimate_ratio": engine._config.estimate_ratio,
            "chunk": engine._chunk_label(),
            "summary_model": (
                engine._summariser_route()[1] or f"(the session's model: {engine.provider}/{engine.model})"
            ),
            "summary_reasoning_effort_default": engine._config.summary_reasoning_effort,
            "summary_calls_in_flight": engine._config.summary_calls_in_flight,
            "summary_calls_per_endpoint": dict(engine._config.summary_calls_per_endpoint or {}),
        },
        "config_sources": config_sources,
        "config_source_warnings": config_source_warnings,
        "ignored_config_yaml_lcm_keys": ignored_config_yaml_lcm_keys,
        "source_lineage": source_lineage,
        "runtime_identity": runtime_identity,
    })


def lcm_doctor(args: Dict[str, Any], **kwargs) -> str:
    """Run diagnostics on the LCM database and configuration."""
    engine = _require_engine(kwargs)
    if engine is None:
        return json.dumps({"error": "LCM engine not initialized"})

    checks: list[dict] = []
    session_id = engine.current_session_id

    # 1. Database integrity
    try:
        result = engine._store.connection.execute("PRAGMA integrity_check").fetchone()
        ok = result and result[0] == "ok"
        checks.append({
            "check": "database_integrity",
            "status": "pass" if ok else "fail",
            "detail": result[0] if result else "no response",
        })
    except Exception as e:
        checks.append({
            "check": "database_integrity",
            "status": "fail",
            "detail": str(e),
        })

    try:
        conn = engine._store.connection
        if conn is None:
            raise RuntimeError("LCM store connection is not initialized")
        schema_health = inspect_lcm_schema_health(
            conn,
            database_path=str(engine._store.db_path),
        )
        missing_tables = schema_health.get("missing_tables")
        has_missing = isinstance(missing_tables, list) and bool(missing_tables)
        checks.append({
            "check": "schema_core_tables",
            "status": "fail" if has_missing or schema_health.get("error") else "pass",
            "detail": schema_health,
        })
    except Exception as e:
        checks.append({
            "check": "schema_core_tables",
            "status": "fail",
            "detail": str(e),
        })

    # 2. SQLite storage posture
    try:
        journal_mode_row = engine._store.connection.execute("PRAGMA journal_mode").fetchone()
        quick_check_row = engine._store.connection.execute("PRAGMA quick_check").fetchone()
        db_path = Path(engine._store.db_path)
        wal_path = Path(str(db_path) + "-wal")
        checks.append({
            "check": "sqlite_storage",
            "status": "pass" if quick_check_row and quick_check_row[0] == "ok" else "fail",
            "detail": {
                "database_path": str(db_path),
                "database_exists": db_path.exists(),
                "journal_mode": journal_mode_row[0] if journal_mode_row else "unknown",
                "quick_check": quick_check_row[0] if quick_check_row else "unknown",
                "database_size_bytes": db_path.stat().st_size if db_path.exists() else 0,
                "wal_size_bytes": wal_path.stat().st_size if wal_path.exists() else 0,
                "stores_left_beside": [str(p) for p in engine._stores_left_beside(db_path)],
            },
        })
    except Exception as e:
        checks.append({
            "check": "sqlite_storage",
            "status": "fail",
            "detail": str(e),
        })

    # 3. Orphaned DAG nodes (nodes referencing store_ids that don't exist)
    try:
        all_nodes = engine._dag.get_session_nodes(session_id)
        orphaned = 0
        for node in all_nodes:
            if node.source_type == "messages":
                for sid in node.source_ids:
                    stored = engine._store.get(sid)
                    if stored is None:
                        orphaned += 1
                        break
        checks.append({
            "check": "orphaned_dag_nodes",
            "status": "pass" if orphaned == 0 else "warn",
            "detail": f"{orphaned} nodes reference missing store messages" if orphaned else "all nodes have valid sources",
        })
    except Exception as e:
        checks.append({
            "check": "orphaned_dag_nodes",
            "status": "fail",
            "detail": str(e),
        })

    try:
        summary_quality = _summary_quality_stats(engine, session_id)
        degraded_count = (
            summary_quality.get("extreme_ratio_nodes", 0)
            + summary_quality.get("tiny_large_source_nodes", 0)
        )
        checks.append({
            "check": "summary_quality",
            "status": "warn" if degraded_count else "pass",
            "detail": summary_quality,
        })
    except Exception as e:
        checks.append({
            "check": "summary_quality",
            "status": "fail",
            "detail": str(e),
        })

    # 4. Configuration validation
    config_warnings = []
    c = engine._config
    if engine.context_length and engine._geometry is None:
        config_warnings.append(f"no compaction: {engine._geometry_error}")
    if engine._native_compaction_refusal:
        config_warnings.append(engine._native_compaction_refusal)
    for warning in getattr(c, "config_source_warnings", []) or []:
        config_warnings.append(warning)
    for key in getattr(c, "ignored_config_yaml_lcm_keys", []) or []:
        config_warnings.append(
            f"config.yaml lcm.{key} is not a supported LCM config.yaml key and was ignored; use the matching LCM_* env var if this setting is intentional"
        )

    checks.append({
        "check": "config_validation",
        "status": "pass" if not config_warnings else "warn",
        "detail": config_warnings if config_warnings else "all settings within normal ranges",
    })

    # 5. Source-lineage hygiene
    try:
        source_stats = engine._store.get_source_stats()
        checks.append({
            "check": "source_lineage_hygiene",
            "status": "pass",
            "detail": {
                **source_stats,
                "normalization_mode": "backcompat-normalization",
            },
        })
    except Exception as e:
        checks.append({
            "check": "source_lineage_hygiene",
            "status": "fail",
            "detail": str(e),
        })

    # 6. The record's invariant (#29 W7, #34 D5), for every session's latest
    # effective compaction.
    try:
        reports = engine._records.check_invariant()
        failing = [report for report in reports if report["status"] != "pass"]
        checks.append({
            "check": "record_invariant",
            "status": "fail" if failing else "pass",
            "detail": {"sessions_checked": len(reports), "failing": failing}
            if failing else f"{len(reports)} session(s) checked; every record on each branch is reached exactly once",
        })
    except Exception as e:
        checks.append({
            "check": "record_invariant",
            "status": "fail",
            "detail": str(e),
        })

    # 7. Context pressure
    if engine.context_length > 0 and engine._geometry is not None:
        tau = engine._geometry.tau
        checks.append({
            "check": "context_pressure",
            "status": "pass" if engine.last_prompt_tokens < tau else "warn",
            "detail": f"the last prompt was {engine.last_prompt_tokens} provider tokens; compaction runs at "
                      f"τ {tau} ({engine._geometry.label()})",
        })

    overall = "healthy"
    if any(ch["status"] == "fail" for ch in checks):
        overall = "unhealthy"
    elif any(ch["status"] == "warn" for ch in checks):
        overall = "warnings"

    # The store's identity and its recent events (each one something the plugin
    # could not do), listed, not judged: they are history, not the store's state.
    try:
        store_identity = engine._records.identity()
    except Exception as e:
        store_identity = {"error": str(e)}
    try:
        store_events = engine._records.recent_events()
    except Exception as e:
        store_events = [{"error": str(e)}]
    return json.dumps({
        "overall": overall,
        "runtime_identity": engine.get_runtime_identity(),
        "store_identity": store_identity,
        "store_events_recent": store_events,
        # The daily backup slot (#6): read with stat, never opened.
        "backup": engine._backup.describe(),
        "checks": checks,
        "guidance": doctor_guidance_for_checks(checks),
    })
