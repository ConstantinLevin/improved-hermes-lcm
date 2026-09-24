"""Slash-style /lcm command helpers for Hermes."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import os
import sqlite3
from typing import Any

from .db_bootstrap import (
    check_external_content_fts_integrity,
    external_content_fts_needs_repair,
    inspect_lcm_schema_health,
    join_background_integrity_scans,
    load_integrity_failed,
    repair_external_content_fts,
)
from .diagnostics import (
    _has_lifecycle_fragmentation,
    _state_db_path_for_engine,
    doctor_guidance_for_checks,
)
from .dag import build_nodes_fts_spec
from .presets import (
    explicit_operator_overrides,
    get_preset,
    invalid_operator_overrides,
    preset_confidence_reasons,
    preset_env_diff,
    preset_match_confidence,
    shipped_presets,
    suggest_preset_for_engine,
    unsupported_runtime_fields_text,
)
from .maintenance import backup_database, rotate_backup_database
from .store import build_message_fts_spec


def _fmt_bool(value: Any) -> str:
    return "yes" if bool(value) else "no"


def _fmt_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    unit = 0
    while value >= 1024 and unit < len(units) - 1:
        value /= 1024
        unit += 1
    precision = 0 if value >= 100 else 1 if value >= 10 else 2
    return f"{value:.{precision}f} {units[unit]}"


def _help_text(error: str | None = None) -> str:
    lines = []
    if error:
        lines.append(error)
        lines.append("")
    lines.extend([
        "LCM command help",
        "- /lcm or /lcm status: show current LCM runtime/session status",
        "- /lcm doctor: run read-only LCM health checks",
        "- /lcm doctor repair: read-only scan for SQLite/FTS index repair needs",
        "- /lcm doctor repair apply: backup-first repair/rebuild of message and summary FTS indexes",
        "- /lcm backup: create a timestamped SQLite backup",
        "- /lcm rotate: preview a tail-preserving in-place compact of the active session (read-only)",
        "- /lcm rotate apply: backup-first rotate that advances the lifecycle frontier past pre-tail raw messages",
        "- /lcm preset show [name]: inspect shipped preset metadata and benchmark provenance",
        "- /lcm preset suggest: preview the best shipped preset for the current engine state",
        "- /lcm preset apply <name> --dry-run: preview env-var changes without mutating live config",
        "- /lcm help: show this help",
    ])
    return "\n".join(lines)


def _status_text(engine) -> str:
    status = engine.get_status()
    db_path = Path(engine._store.db_path)
    db_exists = db_path.exists()
    db_size = db_path.stat().st_size if db_exists else 0
    session_bound = bool(engine.current_session_id)
    source_stats = status.get("source_lineage") or {}
    runtime_identity = status.get("runtime_identity") or {}
    source_stats = {
        "messages_total": int(source_stats.get("messages_total", 0) or 0),
        "attributed_messages": int(source_stats.get("attributed_messages", 0) or 0),
        "normalized_unknown_messages": int(source_stats.get("normalized_unknown_messages", 0) or 0),
        "legacy_blank_source_messages": int(source_stats.get("legacy_blank_source_messages", 0) or 0),
        "effective_unknown_messages": int(source_stats.get("effective_unknown_messages", 0) or 0),
        **({"error": source_stats.get("error")} if source_stats.get("error") else {}),
    }
    config_sources = status.get("config_sources") or {}
    config_source_warnings = status.get("config_source_warnings") or []
    ignored_config_yaml_lcm_keys = status.get("ignored_config_yaml_lcm_keys") or []

    uninitialized = "(uninitialized)"
    unknown = "(unknown)"
    model = (engine.model or unknown) if session_bound else uninitialized
    provider = (engine.provider or unknown) if session_bound else uninitialized
    context_length_source = (
        (getattr(engine, "_context_length_source", "") or unknown)
        if session_bound
        else uninitialized
    )

    lines = [
        "LCM status",
        f"engine: {status.get('engine', engine.name)}",
        f"plugin_name: {runtime_identity.get('plugin_name', '(unknown)')}",
        f"plugin_version: {runtime_identity.get('plugin_version', '(unknown)')}",
        f"plugin_path: {runtime_identity.get('plugin_path', '(unknown)')}",
        f"module_path: {runtime_identity.get('module_path', '(unknown)')}",
        f"plugin_git_commit: {runtime_identity.get('plugin_git_commit') or '(unavailable)'}",
        f"plugin_git_branch: {runtime_identity.get('plugin_git_branch') or '(unavailable)'}",
        f"plugin_git_dirty: {runtime_identity.get('plugin_git_dirty') if runtime_identity.get('plugin_git_dirty') is not None else '(unavailable)'}",
        f"hermes_home: {runtime_identity.get('hermes_home', '') or '(unset)'}",
        f"session_id: {engine.current_session_id or '(unbound)'}",
        f"session_platform: {engine.current_session_platform or ('(unbound)' if not session_bound else '(unknown)')}",
        f"model: {model}",
        f"provider: {provider}",
        f"database_path: {db_path}",
        f"database_path_source: {runtime_identity.get('database_path_source', '(unknown)')}",
        f"database_exists: {_fmt_bool(db_exists)}",
        f"database_size: {_fmt_size(db_size) if db_exists else 'missing'}",
        f"compression_count: {engine.compression_count}",
        f"total_compactions: {status.get('total_compactions', 0)}",
        f"total_compactions_scope: {status.get('total_compactions_scope', 'current_conversation')}",
        f"last_compression_status: {status.get('last_compression_status', 'idle')}",
        f"last_compression_noop_reason: {status.get('last_compression_noop_reason', '') or '(none)'}",
        f"context_length: {engine.context_length if session_bound else '(uninitialized)'}",
        f"raw_context_length: {status.get('raw_context_length', 0) if session_bound else '(uninitialized)'}",
        f"effective_context_length_cap: {status.get('effective_context_length_cap') or '(none)'}",
        f"effective_context_length_reason: {status.get('effective_context_length_reason') or '(none)'}",
        f"context_length_source: {context_length_source}",
        f"configured_context_threshold: {status.get('configured_context_threshold', engine._config.context_threshold)}",
        f"context_threshold: {status.get('context_threshold', engine._config.context_threshold)}",
        f"context_threshold_source: {status.get('context_threshold_source', config_sources.get('context_threshold', 'manual_or_default'))}",
        f"context_threshold_autoraised: {status.get('context_threshold_autoraised') or '(none)'}",
        f"threshold_tokens: {engine.threshold_tokens if session_bound else '(uninitialized)'}",
        f"cache_metrics_available: {_fmt_bool(status.get('cache_metrics_available'))}",
        f"last_input_tokens: {status.get('last_input_tokens', 0)}",
        f"last_output_tokens: {status.get('last_output_tokens', 0)}",
        f"last_cache_read_tokens: {status.get('last_cache_read_tokens', 0)}",
        f"last_cache_write_tokens: {status.get('last_cache_write_tokens', 0)}",
        f"last_reasoning_tokens: {status.get('last_reasoning_tokens', 0)}",
        f"cache_read_ratio: {float(status.get('cache_read_ratio', 0.0) or 0.0) * 100:.1f}%",
        # Filter classification for current_session_id (the foreground view).
        # When a side channel is in flight, get_status() reports the bound
        # session's flags; we read the engine properties instead so this row
        # stays consistent with the session_id row above.
        f"session_ignored: {_fmt_bool(engine.current_session_ignored)}",
        f"session_stateless: {_fmt_bool(engine.current_session_stateless)}",
        f"side_channel_active: {_fmt_bool(engine.side_channel_active)}",
        f"conversation_id: {runtime_identity.get('conversation_id', '') or '(unbound)'}",
        f"lifecycle_current_session_id: {runtime_identity.get('lifecycle_current_session_id', '') or '(none)'}",
        f"lifecycle_last_finalized_session_id: {runtime_identity.get('lifecycle_last_finalized_session_id', '') or '(none)'}",
        f"source_messages_total: {source_stats['messages_total']}",
        f"source_attributed_messages: {source_stats['attributed_messages']}",
        f"source_unknown_messages: {source_stats['normalized_unknown_messages']}",
        f"source_legacy_blank_messages: {source_stats['legacy_blank_source_messages']}",
        f"source_effective_unknown_messages: {source_stats['effective_unknown_messages']}",
    ]

    last_rotate_at = status.get("last_rotate_at")
    if last_rotate_at:
        lines.append(
            f"last_rotate_at: "
            f"{datetime.fromtimestamp(float(last_rotate_at), tz=timezone.utc).isoformat(timespec='seconds')}"
        )
        rotate_backup_size = int(status.get("rotate_backup_size", 0) or 0)
        if rotate_backup_size:
            lines.append(f"rotate_backup_size: {_fmt_size(rotate_backup_size)}")
    else:
        lines.append("last_rotate_at: (never)")
    if status.get("rotate_backup_path"):
        lines.append(f"rotate_backup_path: {status['rotate_backup_path']}")

    if session_bound:
        lines.extend([
            f"store_messages: {status.get('store_messages', 0)}",
            f"dag_nodes: {status.get('dag_nodes', 0)}",
        ])
    else:
        lines.append(
            "note: no active Hermes session has initialized LCM in this process yet — after a fresh restart, send one normal message first if you want live per-session runtime details"
        )

    if "ignore_session_patterns_source" in status:
        lines.append(
            f"ignore_session_patterns_source: {status.get('ignore_session_patterns_source')}"
        )
    if "stateless_session_patterns_source" in status:
        lines.append(
            f"stateless_session_patterns_source: {status.get('stateless_session_patterns_source')}"
        )
    if config_source_warnings:
        lines.append("config_source_warnings: " + "; ".join(config_source_warnings))
    if ignored_config_yaml_lcm_keys:
        lines.append(
            "ignored_config_yaml_lcm_keys: "
            + ", ".join(f"lcm.{key}" for key in ignored_config_yaml_lcm_keys)
        )
    if source_stats.get("error"):
        lines.append(f"source_lineage_error: {source_stats['error']}")
    return "\n".join(lines)


def _rotate_text(engine) -> str:
    preview = engine.rotate_active_session(apply=False)
    if not preview.get("ok"):
        reason = preview.get("reason", "unknown")
        lines = [
            "LCM rotate",
            "status: refused",
            f"reason: {reason}",
        ]
        session_id = preview.get("session_id")
        if session_id:
            lines.append(f"session_id: {session_id}")
        lines.append("note: read-only preview — no changes were made")
        return "\n".join(lines)

    backup_path = engine.rotate_backup_path()
    lines = [
        "LCM rotate",
        f"status: {'noop' if preview.get('noop') else 'preview'}",
        f"session_id: {preview['session_id']}",
        f"conversation_id: {preview['conversation_id']}",
        f"total_message_count: {preview['total_message_count']}",
        f"fresh_tail_count: {preview['fresh_tail_count']}",
        f"fresh_tail_max_tokens: {preview['fresh_tail_max_tokens']}",
        f"effective_fresh_tail_count: {preview['effective_fresh_tail_count']}",
        f"effective_fresh_tail_tokens: {preview['effective_fresh_tail_tokens']}",
        f"pre_tail_message_count: {preview.get('pre_tail_message_count', 0)}",
        f"current_frontier_store_id: {preview['current_frontier_store_id']}",
        f"new_frontier_store_id: {preview['new_frontier_store_id']}",
        f"rotate_backup_path: {backup_path}",
    ]
    if preview.get("noop"):
        lines.append(f"reason: {preview.get('reason', 'no_change')}")
        lines.append("note: read-only preview — rotate apply would be a no-op for this session")
    else:
        lines.append("note: read-only preview — use `/lcm rotate apply` to advance the frontier (backup-first)")
        lines.append("note: pre-tail raw messages remain in the store")
    return "\n".join(lines)


def _rotate_apply_text(engine) -> str:
    # Pre-flight refusal AND noop check before touching disk. This avoids
    # both writing a backup for a session that would refuse and overwriting
    # the previous known-good rolling backup when the apply would be a no-op
    # (e.g., idempotent rerun on an already-rotated session).
    pre = engine.rotate_active_session(apply=False)
    if not pre.get("ok"):
        reason = pre.get("reason", "unknown")
        lines = [
            "LCM rotate apply",
            "status: refused",
            f"reason: {reason}",
        ]
        session_id = pre.get("session_id")
        if session_id:
            lines.append(f"session_id: {session_id}")
        lines.append("note: rotate apply refused; no backup was created and no lifecycle state was changed")
        return "\n".join(lines)

    if pre.get("noop"):
        # Surface the same shape as a successful apply but with status:noop so
        # operators get the standard fields without a fresh backup write
        # destroying the previous known-good snapshot.
        lines = [
            "LCM rotate apply",
            "status: noop",
            f"session_id: {pre['session_id']}",
            f"conversation_id: {pre['conversation_id']}",
            f"total_message_count: {pre['total_message_count']}",
            f"fresh_tail_count: {pre['fresh_tail_count']}",
            f"fresh_tail_max_tokens: {pre['fresh_tail_max_tokens']}",
            f"effective_fresh_tail_count: {pre['effective_fresh_tail_count']}",
            f"effective_fresh_tail_tokens: {pre['effective_fresh_tail_tokens']}",
            f"pre_tail_message_count: {pre.get('pre_tail_message_count', 0)}",
            f"previous_frontier_store_id: {pre['current_frontier_store_id']}",
            f"new_frontier_store_id: {pre['new_frontier_store_id']}",
            f"reason: {pre.get('reason', 'no_change')}",
            "note: rotate is a no-op; rolling backup was not written so the previous rotate-latest snapshot is preserved",
        ]
        return "\n".join(lines)

    backup = rotate_backup_database(engine)
    if not backup["ok"]:
        return "\n".join([
            "LCM rotate apply",
            "status: error",
            f"database_path: {backup['db_path']}",
            f"error: backup failed: {backup['error']}",
            "note: rotate apply aborted before any lifecycle mutation",
        ])

    result = engine.rotate_active_session(apply=True)
    if not result.get("ok"):
        return "\n".join([
            "LCM rotate apply",
            "status: refused",
            f"reason: {result.get('reason', 'unknown')}",
            f"rotate_backup_path: {backup['backup_path']}",
            f"rotate_backup_size: {_fmt_size(int(backup['backup_size']))}",
            "note: backup was created before rotate refused; lifecycle state unchanged",
        ])

    is_noop = bool(result.get("noop"))
    lines = [
        "LCM rotate apply",
        f"status: {'noop' if is_noop else 'ok'}",
        f"session_id: {result['session_id']}",
        f"conversation_id: {result['conversation_id']}",
        f"rotate_backup_path: {backup['backup_path']}",
        f"rotate_backup_size: {_fmt_size(int(backup['backup_size']))}",
        f"total_message_count: {result['total_message_count']}",
        f"fresh_tail_count: {result['fresh_tail_count']}",
        f"fresh_tail_max_tokens: {result['fresh_tail_max_tokens']}",
        f"effective_fresh_tail_count: {result['effective_fresh_tail_count']}",
        f"effective_fresh_tail_tokens: {result['effective_fresh_tail_tokens']}",
        f"pre_tail_message_count: {result.get('pre_tail_message_count', 0)}",
        f"previous_frontier_store_id: {result['current_frontier_store_id']}",
        f"new_frontier_store_id: {result.get('applied_frontier_store_id', result['new_frontier_store_id'])}",
    ]
    if is_noop:
        lines.append(f"reason: {result.get('reason', 'no_change')}")
        lines.append("note: lifecycle state already at or ahead of the target frontier")
    else:
        lines.append("note: pre-tail raw messages remain in the store")
        lines.append("note: rolling backup overwrites the previous rotate-latest slot")
    return "\n".join(lines)


def _scan_fts_repair(engine) -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}
    specs = {
        "messages_fts": build_message_fts_spec(),
        "nodes_fts": build_nodes_fts_spec(),
    }
    conn = engine._store.connection
    for label, spec in specs.items():
        try:
            structural_needs_repair = external_content_fts_needs_repair(conn, spec)
            integrity_check = check_external_content_fts_integrity(conn, spec)
            integrity_status = str(integrity_check.get("status") or "fail")
            needs_repair = structural_needs_repair or integrity_status == "fail"
            content_count = int(conn.execute(
                f"SELECT COUNT(*) FROM {spec.content_table}"
            ).fetchone()[0])
            try:
                fts_count = int(conn.execute(f"SELECT COUNT(*) FROM {spec.table_name}").fetchone()[0])
            except sqlite3.Error:
                fts_count = None
            checks[label] = {
                "ok": not needs_repair,
                "needs_repair": needs_repair,
                "content_rows": content_count,
                "fts_rows": fts_count,
                "integrity_status": integrity_status,
                "integrity_detail": integrity_check.get("detail"),
                "error": None,
            }
        except Exception as exc:  # pragma: no cover - defensive
            checks[label] = {
                "ok": False,
                "needs_repair": True,
                "content_rows": None,
                "fts_rows": None,
                "integrity_status": "error",
                "integrity_detail": str(exc),
                "error": str(exc),
            }
    return {
        "checks": checks,
        "needs_repair": any(item["needs_repair"] for item in checks.values()),
    }


def _doctor_repair_text(engine) -> str:
    scan = _scan_fts_repair(engine)
    lines = [
        "LCM doctor repair",
        f"status: {'repair-needed' if scan['needs_repair'] else 'ok'}",
    ]
    for label, item in scan["checks"].items():
        state = "repair-needed" if item["needs_repair"] else "ok"
        lines.append(f"{label}: {state}")
        if item["error"]:
            lines.append(f"{label}_error: {item['error']}")
        else:
            lines.append(f"{label}_content_rows: {item['content_rows']}")
            lines.append(f"{label}_fts_rows: {item['fts_rows']}")
            lines.append(f"{label}_integrity_status: {item['integrity_status']}")
    lines.append("note: read-only scan only — no FTS tables were repaired")
    if scan["needs_repair"]:
        lines.append("note: use `/lcm doctor repair apply` to create a backup and repair FTS indexes")
    return "\n".join(lines)


def _doctor_repair_apply_text(engine) -> str:
    backup = backup_database(engine)
    if not backup["ok"]:
        return "\n".join([
            "LCM doctor repair apply",
            "status: error",
            f"database_path: {backup['db_path']}",
            f"error: backup failed: {backup['error']}",
            "note: repair apply aborted before any FTS tables were repaired",
        ])

    # Join any in-flight background integrity scan first: otherwise a scan still
    # mid-flight can error out (or re-check) after the repair commits and re-write
    # a fresh fts_integrity_failed marker, reproducing F1's stuck false-positive
    # via a race (F3).
    join_background_integrity_scans()

    conn = engine._store.connection
    try:
        messages_result = repair_external_content_fts(conn, build_message_fts_spec())
        nodes_result = repair_external_content_fts(conn, build_nodes_fts_spec())
    except sqlite3.Error as exc:
        return "\n".join([
            "LCM doctor repair apply",
            "status: error",
            f"database_path: {backup['db_path']}",
            f"backup_path: {backup['backup_path']}",
            f"backup_size: {_fmt_size(int(backup['backup_size']))}",
            f"error: FTS repair failed: {exc}",
            "note: backup was created before repair apply",
        ])

    return "\n".join([
        "LCM doctor repair apply",
        "status: ok",
        f"database_path: {backup['db_path']}",
        f"backup_path: {backup['backup_path']}",
        f"backup_size: {_fmt_size(int(backup['backup_size']))}",
        f"messages_fts_rebuilt: {_fmt_bool(messages_result['rebuilt'])}",
        f"messages_fts_triggers_recreated: {_fmt_bool(messages_result['triggers_recreated'])}",
        f"messages_fts_degraded: {_fmt_bool(messages_result['degraded'])}",
        f"nodes_fts_rebuilt: {_fmt_bool(nodes_result['rebuilt'])}",
        f"nodes_fts_triggers_recreated: {_fmt_bool(nodes_result['triggers_recreated'])}",
        f"nodes_fts_degraded: {_fmt_bool(nodes_result['degraded'])}",
        "note: backup created before repair apply",
    ])


def _doctor_text(engine) -> str:
    db_path = Path(engine._store.db_path)
    runtime_identity = engine.get_runtime_identity()
    store_conn = engine._store.connection
    dag_conn = engine._dag.connection

    issues: list[str] = []
    recommended_actions: list[str] = []
    schema_health = inspect_lcm_schema_health(store_conn, database_path=str(db_path))
    schema_missing_raw = schema_health.get("missing_tables")
    schema_missing_tables = [str(name) for name in schema_missing_raw] if isinstance(schema_missing_raw, list) else []
    schema_existing_raw = schema_health.get("existing_tables")
    schema_existing_tables = [str(name) for name in schema_existing_raw] if isinstance(schema_existing_raw, list) else []
    schema_core_status = "error" if schema_health.get("error") else "missing" if schema_missing_tables else "ok"
    if schema_missing_tables or schema_health.get("error"):
        issues.append("schema_core_tables")

    def _safe_count(conn, query: str, issue_key: str) -> int | str:
        try:
            return int(conn.execute(query).fetchone()[0])
        except Exception as exc:  # pragma: no cover - defensive
            issues.append(issue_key)
            return f"error: {exc}"

    try:
        integrity_row = store_conn.execute("PRAGMA integrity_check").fetchone()
        integrity = str(integrity_row[0]) if integrity_row else "unknown"
    except Exception as exc:  # pragma: no cover - defensive
        integrity = f"error: {exc}"
        issues.append("sqlite_integrity")

    def _fts_text_status(result: dict[str, Any]) -> str:
        status = str(result.get("status") or "fail")
        return "ok" if status == "pass" else status

    try:
        store_fts_count = int(store_conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0])
        store_fts_integrity = check_external_content_fts_integrity(store_conn, build_message_fts_spec())
        store_fts = _fts_text_status(store_fts_integrity)
        if store_fts == "fail":
            issues.append("messages_fts")
        elif store_fts == "unchecked":
            recommended_actions.append("rerun `/lcm doctor` with read-write SQLite access if a deep messages FTS check is needed")
    except Exception as exc:  # pragma: no cover - defensive
        store_fts_count = f"error: {exc}"
        store_fts = f"error: {exc}"
        store_fts_integrity = {"status": "fail", "detail": str(exc)}
        issues.append("messages_fts")

    try:
        node_fts_count = int(dag_conn.execute("SELECT COUNT(*) FROM nodes_fts").fetchone()[0])
        node_fts_integrity = check_external_content_fts_integrity(dag_conn, build_nodes_fts_spec())
        node_fts = _fts_text_status(node_fts_integrity)
        if node_fts == "fail":
            issues.append("nodes_fts")
        elif node_fts == "unchecked":
            recommended_actions.append("rerun `/lcm doctor` with read-write SQLite access if a deep nodes FTS check is needed")
    except Exception as exc:  # pragma: no cover - defensive
        node_fts_count = f"error: {exc}"
        node_fts = f"error: {exc}"
        node_fts_integrity = {"status": "fail", "detail": str(exc)}
        issues.append("nodes_fts")

    # A prior non-blocking background integrity scan (issue #6) records a
    # persisted ``fts_integrity_failed:<table>`` flag when it finds corruption
    # without rebuilding. Surface it even when this doctor run's live deep check
    # could not confirm it (e.g. read-only access), pointing at the explicit
    # repair path.
    try:
        store_fts_failed_flag = load_integrity_failed(store_conn, build_message_fts_spec())
    except Exception:  # pragma: no cover - defensive
        store_fts_failed_flag = None
    try:
        node_fts_failed_flag = load_integrity_failed(dag_conn, build_nodes_fts_spec())
    except Exception:  # pragma: no cover - defensive
        node_fts_failed_flag = None
    if store_fts_failed_flag and "messages_fts" not in issues:
        issues.append("messages_fts")
    if node_fts_failed_flag and "nodes_fts" not in issues:
        issues.append("nodes_fts")

    total_messages = _safe_count(store_conn, "SELECT COUNT(*) FROM messages", "messages_total")
    total_message_sessions = _safe_count(
        store_conn,
        "SELECT COUNT(DISTINCT session_id) FROM messages",
        "message_sessions_total",
    )
    total_nodes = _safe_count(dag_conn, "SELECT COUNT(*) FROM summary_nodes", "summary_nodes_total")
    total_node_sessions = _safe_count(
        dag_conn,
        "SELECT COUNT(DISTINCT session_id) FROM summary_nodes",
        "summary_node_sessions_total",
    )

    db_exists = db_path.exists()
    db_size = db_path.stat().st_size if db_exists else 0
    wal_path = Path(str(db_path) + "-wal")
    wal_size = wal_path.stat().st_size if wal_path.exists() else 0
    try:
        journal_row = store_conn.execute("PRAGMA journal_mode").fetchone()
        journal_mode = str(journal_row[0]) if journal_row else "unknown"
    except Exception as exc:  # pragma: no cover - defensive
        journal_mode = f"error: {exc}"
        issues.append("sqlite_journal_mode")
    try:
        quick_row = store_conn.execute("PRAGMA quick_check").fetchone()
        quick_check = str(quick_row[0]) if quick_row else "unknown"
    except Exception as exc:  # pragma: no cover - defensive
        quick_check = f"error: {exc}"
        issues.append("sqlite_quick_check")

    debt_rows = []
    lifecycle_conn = getattr(getattr(engine, "_lifecycle", None), "connection", None)
    if lifecycle_conn is not None:
        try:
            debt_rows = lifecycle_conn.execute(
                """
                SELECT conversation_id, debt_kind, debt_size_estimate
                FROM lcm_lifecycle_state
                WHERE debt_kind IS NOT NULL AND debt_size_estimate > 0
                ORDER BY updated_at DESC
                """
            ).fetchall()
        except Exception as exc:  # pragma: no cover - defensive
            issues.append("lifecycle_state")
            debt_rows = [(f"error: {exc}", "error", 0)]

    observations: list[str] = []

    if schema_health.get("error"):
        observations.append(f"schema_core_tables: error: {schema_health['error']}")
        recommended_actions.append(
            "verify SQLite can read sqlite_master for the database inspected by Hermes"
        )
    elif schema_missing_tables:
        observations.append(
            "schema_core_tables: missing " + ", ".join(schema_missing_tables)
        )
        recommended_actions.append(
            "verify HERMES_HOME/LCM_DATABASE_PATH point at the database inspected by Hermes"
        )
    else:
        observations.append("schema_core_tables: ok")

    if debt_rows:
        first = debt_rows[0]
        observations.append(
            f"maintenance_debt: {len(debt_rows)} conversation(s) currently carry deferred maintenance debt; first={first[0]} kind={first[1]} size={first[2]}"
        )
        recommended_actions.append(
            "let normal compaction turns reduce maintenance debt"
        )

    try:
        source_stats = engine._store.get_source_stats()
    except Exception as exc:  # pragma: no cover - defensive
        issues.append("source_lineage")
        source_stats = {
            "messages_total": 0,
            "attributed_messages": 0,
            "normalized_unknown_messages": 0,
            "legacy_blank_source_messages": 0,
            "effective_unknown_messages": 0,
            "error": str(exc),
        }
    observations.append(
        "source_lineage: "
        f"attributed={source_stats['attributed_messages']} "
        f"unknown={source_stats['normalized_unknown_messages']} "
        f"legacy_blank={source_stats['legacy_blank_source_messages']} "
        f"effective_unknown={source_stats['effective_unknown_messages']}"
    )
    if source_stats.get("error"):
        observations.append(f"source_lineage_error: {source_stats['error']}")
    if source_stats["legacy_blank_source_messages"]:
        observations.append(
            "legacy blank-source rows are normalized as `source=unknown` for back-compat filters"
        )

    try:
        lifecycle_stats = engine._lifecycle.get_fragmentation_stats(
            state_db_path=_state_db_path_for_engine(engine)
        )
    except Exception as exc:  # pragma: no cover - defensive
        issues.append("lifecycle_fragmentation")
        lifecycle_stats = {"error": str(exc)}
    else:
        observations.append(
            "lifecycle_fragmentation: "
            f"lifecycle_rows={lifecycle_stats['lifecycle_rows']} "
            f"empty_lifecycle_rows={lifecycle_stats.get('empty_lifecycle_rows', 0)} "
            f"message_sessions={lifecycle_stats['distinct_message_sessions']} "
            f"node_sessions={lifecycle_stats['distinct_node_sessions']} "
            f"current_missing_in_lcm_any={lifecycle_stats['lifecycle_current_missing_in_lcm_any']} "
            f"last_finalized_missing_in_lcm_any={lifecycle_stats['lifecycle_last_finalized_missing_in_lcm_any']} "
            f"current_missing_in_state={lifecycle_stats['lifecycle_current_missing_in_state']} "
            f"last_finalized_missing_in_state={lifecycle_stats['lifecycle_last_finalized_missing_in_state']} "
            f"message_sessions_missing_in_state={lifecycle_stats['lcm_message_sessions_missing_in_state']} "
            f"node_sessions_missing_in_state={lifecycle_stats['lcm_node_sessions_missing_in_state']} "
            f"message_sessions_without_lifecycle_current={lifecycle_stats['message_sessions_without_lifecycle_current']} "
            f"message_sessions_without_lifecycle_reference={lifecycle_stats['message_sessions_without_lifecycle_reference']} "
            f"node_sessions_without_lifecycle_reference={lifecycle_stats['node_sessions_without_lifecycle_reference']} "
            f"state_sessions_missing_in_lcm_any={lifecycle_stats['state_sessions_missing_in_lcm_any']}"
        )
        if lifecycle_stats.get("state_db_error"):
            observations.append(f"lifecycle_fragmentation_state_db_error: {lifecycle_stats['state_db_error']}")
        classification = lifecycle_stats.get("classification") or {}
        categories = classification.get("categories") or []
        if classification:
            observations.append(
                "lifecycle_fragmentation_classification: "
                f"{classification.get('status', 'unknown')}; {len(categories)} categories need review"
            )
            for category in categories:
                sample = ",".join(category.get("sample_session_ids") or []) or "(none)"
                observations.append(
                    "lifecycle_category "
                    f"{category.get('name')}: count={category.get('count', 0)} sample={sample}"
                )
        if _has_lifecycle_fragmentation(lifecycle_stats):
            recommended_actions.append(
                "inspect lifecycle fragmentation before any cleanup/repair behavior mutates state"
            )
            recommended_actions.append(
                "treat this as read-only evidence; do not infer every mismatch is harmful"
            )

    if store_fts_failed_flag:
        observations.append(
            "messages_fts_integrity: a background integrity scan flagged corruption "
            f"(detail: {store_fts_failed_flag['detail'] or 'unknown'})"
        )
        recommended_actions.append(
            "run `/lcm doctor repair`, then `/lcm backup` and `/lcm doctor repair apply` to rebuild messages_fts"
        )
    if node_fts_failed_flag:
        observations.append(
            "nodes_fts_integrity: a background integrity scan flagged corruption "
            f"(detail: {node_fts_failed_flag['detail'] or 'unknown'})"
        )
        recommended_actions.append(
            "run `/lcm doctor repair`, then `/lcm backup` and `/lcm doctor repair apply` to rebuild nodes_fts"
        )

    triage_checks: list[dict[str, Any]] = []
    if integrity != "ok":
        triage_checks.append({"check": "database_integrity", "status": "fail", "detail": integrity})
    if schema_health.get("error") or schema_missing_tables:
        triage_checks.append({"check": "schema_core_tables", "status": "fail", "detail": schema_health})
    if store_fts != "ok":
        triage_checks.append({
            "check": "messages_fts_integrity",
            "status": "warn" if store_fts == "unchecked" else "fail",
            "detail": store_fts_integrity,
        })
    if node_fts != "ok":
        triage_checks.append({
            "check": "nodes_fts_integrity",
            "status": "warn" if node_fts == "unchecked" else "fail",
            "detail": node_fts_integrity,
        })
    if store_fts_failed_flag and store_fts != "fail":
        triage_checks.append({
            "check": "messages_fts_integrity",
            "status": "fail",
            "detail": {"status": "fail", "background_flag": store_fts_failed_flag},
        })
    if node_fts_failed_flag and node_fts != "fail":
        triage_checks.append({
            "check": "nodes_fts_integrity",
            "status": "fail",
            "detail": {"status": "fail", "background_flag": node_fts_failed_flag},
        })
    if source_stats.get("error"):
        triage_checks.append({"check": "source_lineage_hygiene", "status": "fail", "detail": source_stats})
    if lifecycle_stats.get("error") or _has_lifecycle_fragmentation(lifecycle_stats):
        lifecycle_status = "fail" if lifecycle_stats.get("error") else "warn"
        triage_checks.append({"check": "lifecycle_fragmentation", "status": lifecycle_status, "detail": lifecycle_stats})
    triage_guidance = doctor_guidance_for_checks(triage_checks)

    doctor_status = "issues-found" if integrity != "ok" or issues else (
        "action-recommended" if recommended_actions else "ok"
    )
    lines = [
        "LCM doctor",
        f"status: {doctor_status}",
        f"plugin_name: {runtime_identity.get('plugin_name', '(unknown)')}",
        f"plugin_version: {runtime_identity.get('plugin_version', '(unknown)')}",
        f"plugin_path: {runtime_identity.get('plugin_path', '(unknown)')}",
        f"module_path: {runtime_identity.get('module_path', '(unknown)')}",
        f"plugin_git_commit: {runtime_identity.get('plugin_git_commit') or '(unavailable)'}",
        f"plugin_git_branch: {runtime_identity.get('plugin_git_branch') or '(unavailable)'}",
        f"plugin_git_dirty: {runtime_identity.get('plugin_git_dirty') if runtime_identity.get('plugin_git_dirty') is not None else '(unavailable)'}",
        f"database_path: {db_path}",
        f"database_exists: {_fmt_bool(db_exists)}",
        f"database_size: {_fmt_size(db_size) if db_exists else 'missing'}",
        f"wal_size: {_fmt_size(wal_size)}",
        f"schema_core_tables: {schema_core_status}",
        f"schema_missing_tables: {', '.join(schema_missing_tables) or '(none)'}",
        f"schema_existing_tables: {', '.join(schema_existing_tables) or '(none)'}",
        f"journal_mode: {journal_mode}",
        f"quick_check: {quick_check}",
        f"sqlite_integrity: {integrity}",
        f"messages_total: {total_messages}",
        f"message_sessions_total: {total_message_sessions}",
        f"summary_nodes_total: {total_nodes}",
        f"summary_node_sessions_total: {total_node_sessions}",
        f"messages_fts: {store_fts}",
        f"messages_fts_rows: {store_fts_count}",
        f"nodes_fts: {node_fts}",
        f"nodes_fts_rows: {node_fts_count}",
    ]
    if issues:
        lines.append(f"issues: {', '.join(issues)}")
    else:
        lines.append("issues: none")
    lines.append("observations:")
    for item in observations:
        lines.append(f"- {item}")
    lines.append("recommended_actions:")
    if recommended_actions:
        for item in recommended_actions:
            lines.append(f"- {item}")
    else:
        lines.append("- none")
    lines.append("triage_guidance:")
    if triage_guidance:
        for item in triage_guidance:
            warning_suffix = " warning-only" if item.get("warning_only") else ""
            lines.append(
                "- "
                f"{item['check']}: {item['action']}{warning_suffix} — "
                f"{item['operator_action']}"
            )
    else:
        lines.append("- none")
    return "\n".join(lines)


def _backup_text(engine) -> str:
    backup = backup_database(engine)
    if not backup["ok"]:
        return "\n".join([
            "LCM backup",
            "status: error",
            f"database_path: {backup['db_path']}",
            f"error: {backup['error']}",
        ])

    return "\n".join([
        "LCM backup",
        "status: ok",
        f"database_path: {backup['db_path']}",
        f"backup_path: {backup['backup_path']}",
        f"backup_size: {_fmt_size(int(backup['backup_size']))}",
        "note: backup created before any future cleanup/apply workflow",
    ])


def _unknown_preset_text(name: str) -> str:
    available = ", ".join(preset.name for preset in shipped_presets()) or "(none)"
    return "\n".join([
        "LCM preset",
        "status: error",
        f"error: unknown preset {name}",
        f"available_presets: {available}",
    ])


def _preset_show_text(tokens: list[str], engine) -> str:
    if len(tokens) > 1:
        return _help_text("`/lcm preset show` accepts at most one preset name.")
    preset = get_preset(tokens[0] if tokens else None)
    if preset is None:
        return _unknown_preset_text(tokens[0])
    provenance = dict(preset.provenance)
    metric_summary = dict(provenance.get("metric_summary") or {})
    fixture_suite = ", ".join(str(item) for item in provenance.get("fixture_suite") or []) or "(unknown)"
    applies_to = ", ".join(preset.applies_to) if preset.applies_to else "(unspecified)"
    lines = [
        "LCM preset show",
        f"preset: {preset.name}",
        f"family: {preset.family}",
        f"description: {preset.description}",
        f"policy_version: {preset.policy_version}",
        f"policy_path: {preset.policy_path}",
        f"benchmark_version: {provenance.get('benchmark_version', '(unknown)')}",
        f"fixture_suite: {fixture_suite}",
        f"score: {metric_summary.get('score', '(unknown)')}",
        f"baseline_score: {metric_summary.get('baseline_score', '(unknown)')}",
        f"retrieval_canary_recall: {metric_summary.get('retrieval_canary_recall', '(unknown)')}",
        f"applies_to: {applies_to}",
        "runtime_env:",
    ]
    for item in preset_env_diff(preset, engine._config):
        lines.append(f"- {item}")
    lines.extend([
        f"unsupported_runtime_fields: {unsupported_runtime_fields_text(preset)}",
        "operator_config_precedence: explicit preset-managed LCM_* overrides win",
        "runtime_mutation: no",
        f"notes: {preset.notes}",
    ])
    return "\n".join(lines)


def _preset_suggest_text(engine) -> str:
    preset, reason = suggest_preset_for_engine(engine)
    lines = ["LCM preset suggest"]
    if preset is None:
        lines.extend([
            "suggested_preset: (none)",
            f"reason: {reason}",
            "note: run deterministic benchmarks before promoting a runtime preset",
            "note: suggestion only; no live config was changed",
        ])
        return "\n".join(lines)

    explicit = explicit_operator_overrides()
    invalid = invalid_operator_overrides()
    invalid_text = ", ".join(
        f"{env_var}={os.environ.get(env_var, '')}" for env_var in sorted(invalid.values())
    ) if invalid else "(none)"
    lines.extend([
        f"suggested_preset: {preset.name}",
        f"reason: {reason}",
        f"match_confidence: {preset_match_confidence(engine, preset)}",
        f"policy_version: {preset.policy_version}",
        f"benchmark_version: {preset.provenance.get('benchmark_version', '(unknown)')}",
        "explicit_overrides: " + (", ".join(sorted(explicit.values())) if explicit else "(none)"),
        f"invalid_overrides: {invalid_text}",
        "confidence_reasons:",
    ])
    for item in preset_confidence_reasons(engine, preset, reason):
        lines.append(f"- {item}")
    lines.extend([
        "preview:",
    ])
    for item in preset_env_diff(
        preset,
        engine._config,
        runtime_context_threshold=getattr(engine, "context_threshold", None),
        runtime_context_threshold_source=getattr(engine, "_context_threshold_source", ""),
    ):
        lines.append(f"- {item}")
    lines.extend([
        f"unsupported_runtime_fields: {unsupported_runtime_fields_text(preset)}",
        "note: suggestion only; no live config was changed",
    ])
    return "\n".join(lines)


def _preset_apply_text(tokens: list[str], engine) -> str:
    if not tokens:
        return _help_text("`/lcm preset apply` requires a preset name and `--dry-run`.")
    dry_run = "--dry-run" in tokens
    selected = [token for token in tokens if token != "--dry-run"]
    if len(selected) != 1:
        return _help_text("`/lcm preset apply` accepts exactly one preset name and optional `--dry-run`.")
    preset_name = selected[0]
    preset = get_preset(preset_name)
    if preset is None:
        return _unknown_preset_text(preset_name)
    if not dry_run:
        return "\n".join([
            "LCM preset apply",
            "status: denied",
            "error: preset apply is preview-only for now; pass --dry-run",
            "note: no live config was changed",
        ])

    lines = [
        "LCM preset apply",
        "status: dry-run",
        f"preset: {preset.name}",
        "would_set:",
    ]
    for item in preset_env_diff(
        preset,
        engine._config,
        runtime_context_threshold=getattr(engine, "context_threshold", None),
        runtime_context_threshold_source=getattr(engine, "_context_threshold_source", ""),
    ):
        lines.append(f"- {item}")
    lines.extend([
        f"unsupported_runtime_fields: {unsupported_runtime_fields_text(preset)}",
        "operator_config_precedence: explicit preset-managed LCM_* overrides win",
        "note: no live config was changed",
    ])
    return "\n".join(lines)


def _preset_text(tokens: list[str], engine) -> str:
    if not tokens:
        return _help_text("`/lcm preset` requires `show`, `suggest`, or `apply`.")
    subcommand = tokens[0].lower()
    rest = tokens[1:]
    if subcommand == "show":
        return _preset_show_text(rest, engine)
    if subcommand == "suggest":
        if rest:
            return _help_text("`/lcm preset suggest` does not accept extra arguments.")
        return _preset_suggest_text(engine)
    if subcommand == "apply":
        return _preset_apply_text(rest, engine)
    return _help_text("`/lcm preset` supports `show`, `suggest`, and `apply`.")


def handle_lcm_command(raw_args: str | None, engine) -> str:
    tokens = [part.strip() for part in (raw_args or "").strip().split() if part.strip()]
    if not tokens:
        return _status_text(engine)

    head = tokens[0].lower()
    rest = tokens[1:]

    if head == "status":
        if rest:
            return _help_text("`/lcm status` does not accept extra arguments.")
        return _status_text(engine)

    if head == "doctor":
        if not rest:
            return _doctor_text(engine)
        if len(rest) == 1 and rest[0].lower() == "repair":
            return _doctor_repair_text(engine)
        if len(rest) == 2 and rest[0].lower() == "repair" and rest[1].lower() == "apply":
            return _doctor_repair_apply_text(engine)
        return _help_text("`/lcm doctor` currently supports `repair` and `repair apply` as extra subcommands.")

    if head == "backup":
        if rest:
            return _help_text("`/lcm backup` does not accept extra arguments.")
        return _backup_text(engine)

    if head == "rotate":
        if not rest:
            return _rotate_text(engine)
        if len(rest) == 1 and rest[0].lower() == "apply":
            return _rotate_apply_text(engine)
        return _help_text("`/lcm rotate` accepts an optional `apply` subcommand.")


    if head == "preset":
        return _preset_text(rest, engine)

    if head == "help":
        return _help_text()

    return _help_text(f"Unknown subcommand: {tokens[0]}")
