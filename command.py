"""Slash-style /lcm command helpers for Hermes."""

from __future__ import annotations

from pathlib import Path
import sqlite3
from typing import Any

from .db_bootstrap import (
    check_external_content_fts_integrity,
    external_content_fts_needs_repair,
    inspect_lcm_schema_health,
    repair_external_content_fts,
)
from .diagnostics import doctor_guidance_for_checks
from .dag import build_nodes_fts_spec
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
        "- /lcm doctor repair apply: rebuild the message and summary FTS indexes from the stored records",
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
        f"stores_left_beside: {', '.join(runtime_identity.get('stores_left_beside') or []) or '(none)'}",
        f"database_exists: {_fmt_bool(db_exists)}",
        f"database_size: {_fmt_size(db_size) if db_exists else 'missing'}",
        f"compression_count: {engine.compression_count}",
        f"total_compactions: {status.get('total_compactions', 0)}",
        f"total_compactions_scope: {status.get('total_compactions_scope', 'current_conversation')}",
        f"last_compression_status: {status.get('last_compression_status', 'idle')}",
        f"last_compression_noop_reason: {status.get('last_compression_noop_reason', '') or '(none)'}",
        f"native_compaction_refused: {status.get('native_compaction_refused') or '(not configured)'}",
        f"context_length: {engine.context_length if session_bound else '(uninitialized)'}",
        f"raw_context_length: {status.get('raw_context_length', 0) if session_bound else '(uninitialized)'}",
        f"effective_context_length_cap: {status.get('effective_context_length_cap') or '(none)'}",
        f"effective_context_length_reason: {status.get('effective_context_length_reason') or '(none)'}",
        f"context_length_source: {context_length_source}",
        f"geometry: {status.get('geometry') or '(none)'}",
        f"turn: {status.get('turn') or '(none)'}",
        f"threshold_tokens: {engine.threshold_tokens if session_bound else '(uninitialized)'}",
        f"cache_metrics_available: {_fmt_bool(status.get('cache_metrics_available'))}",
        f"last_input_tokens: {status.get('last_input_tokens', 0)}",
        f"last_output_tokens: {status.get('last_output_tokens', 0)}",
        f"last_cache_read_tokens: {status.get('last_cache_read_tokens', 0)}",
        f"last_cache_write_tokens: {status.get('last_cache_write_tokens', 0)}",
        f"last_reasoning_tokens: {status.get('last_reasoning_tokens', 0)}",
        f"cache_read_ratio: {float(status.get('cache_read_ratio', 0.0) or 0.0) * 100:.1f}%",
        f"conversation_id: {runtime_identity.get('conversation_id', '') or '(unbound)'}",
        f"host_session_id: {runtime_identity.get('host_session_id', '') or '(unbound)'}",
        f"source_messages_total: {source_stats['messages_total']}",
        f"source_attributed_messages: {source_stats['attributed_messages']}",
        f"source_unknown_messages: {source_stats['normalized_unknown_messages']}",
        f"source_legacy_blank_messages: {source_stats['legacy_blank_source_messages']}",
        f"source_effective_unknown_messages: {source_stats['effective_unknown_messages']}",
    ]

    if session_bound:
        lines.extend([
            f"store_messages: {status.get('store_messages', 0)}",
            f"dag_nodes: {status.get('dag_nodes', 0)}",
        ])
    else:
        lines.append(
            "note: no active Hermes session has initialized LCM in this process yet — after a fresh restart, send one normal message first if you want live per-session runtime details"
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
        lines.append("note: use `/lcm doctor repair apply` to rebuild the FTS indexes from the stored records")
    return "\n".join(lines)


def _doctor_repair_apply_text(engine) -> str:
    # The full-text indexes are derived from the record, which is insert-only, and are
    # rebuilt from it; a repair cannot lose a record. The store's backup is the daily
    # slot (#6), not a copy per command.
    db_path = engine._store.db_path
    conn = engine._store.connection
    try:
        messages_result = repair_external_content_fts(conn, build_message_fts_spec())
        nodes_result = repair_external_content_fts(conn, build_nodes_fts_spec())
    except sqlite3.Error as exc:
        return "\n".join([
            "LCM doctor repair apply",
            "status: error",
            f"database_path: {db_path}",
            f"error: FTS repair failed: {exc}",
        ])

    return "\n".join([
        "LCM doctor repair apply",
        "status: ok",
        f"database_path: {db_path}",
        f"messages_fts_rebuilt: {_fmt_bool(messages_result['rebuilt'])}",
        f"messages_fts_triggers_recreated: {_fmt_bool(messages_result['triggers_recreated'])}",
        f"messages_fts_degraded: {_fmt_bool(messages_result['degraded'])}",
        f"nodes_fts_rebuilt: {_fmt_bool(nodes_result['rebuilt'])}",
        f"nodes_fts_triggers_recreated: {_fmt_bool(nodes_result['triggers_recreated'])}",
        f"nodes_fts_degraded: {_fmt_bool(nodes_result['degraded'])}",
        "note: the indexes are rebuilt from the stored records",
    ])


def _backup_lines(engine) -> list[str]:
    """The daily backup slot (#6), as the doctor shows it."""
    backup = engine._backup.describe()
    if not backup["taken"]:
        return [f"backup_slot: {backup['slot']}", "backup_taken: none yet"]
    return [
        f"backup_slot: {backup['slot']}",
        f"backup_taken: {backup['age_hours']} h ago ({_fmt_size(int(backup['size_bytes']))})",
    ]


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

    # The record's invariant (#29 W7, #34 D5) and what the plugin could not do.
    try:
        store_identity = engine._records.identity()
        invariant_reports = engine._records.check_invariant()
        store_events = engine._records.recent_events()
        invariant_error = ""
    except Exception as exc:  # pragma: no cover - defensive
        store_identity, invariant_reports, store_events = {}, [], []
        invariant_error = str(exc)
    invariant_failing = [report for report in invariant_reports if report["status"] != "pass"]
    if invariant_error or invariant_failing:
        issues.append("record_invariant")

    observations: list[str] = []
    if invariant_error:
        observations.append(f"record_invariant: error: {invariant_error}")
    for report in invariant_failing:
        observations.append(
            f"record_invariant: session {report['session']} (compaction {report['compaction']}): "
            f"{report['problem_count']} problem(s): " + "; ".join(report["problems"])
        )
    if store_events:
        observations.append(
            "store_events (newest first): "
            + "; ".join(f"{event['kind']} ({event['session'] or '-'}): {event['detail'] or ''}" for event in store_events)
        )

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
    if source_stats.get("error"):
        triage_checks.append({"check": "source_lineage_hygiene", "status": "fail", "detail": source_stats})
    if invariant_error or invariant_failing:
        triage_checks.append({"check": "record_invariant", "status": "fail",
                              "detail": invariant_error or invariant_failing})
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
        f"stores_left_beside: {', '.join(runtime_identity.get('stores_left_beside') or []) or '(none)'}",
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
        f"store_format: {store_identity.get('format', '(unknown)')}",
        f"store_uuid: {store_identity.get('store_uuid', '(unknown)')}",
        "record_invariant: "
        + ("error" if invariant_error else "fail" if invariant_failing else "pass")
        + f" ({len(invariant_reports)} session(s) checked)",
        f"store_events_recent: {len(store_events)}",
        *_backup_lines(engine),
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

    if head == "help":
        return _help_text()

    return _help_text(f"Unknown subcommand: {tokens[0]}")
