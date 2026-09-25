"""Shared read-only diagnostic helpers for LCM tools and commands."""

from __future__ import annotations

from typing import Any


DOCTOR_ACTION_SAFE_IGNORE = "safe/ignore"
DOCTOR_ACTION_INSPECT = "inspect"
DOCTOR_ACTION_BACKUP_FIRST_CLEANUP = "backup-first cleanup"


def doctor_guidance_for_check(check: dict[str, Any]) -> dict[str, Any] | None:
    """Return operator triage guidance for one lcm_doctor check.

    Guidance is deliberately conservative: most warning classes are inspect-only
    evidence, and any mutation path is framed as preview/backup/apply rather than
    implied automatic cleanup.
    """
    status = str(check.get("status") or "")
    if status not in {"warn", "fail"}:
        return None

    name = str(check.get("check") or "unknown")
    detail = check.get("detail")
    action = DOCTOR_ACTION_INSPECT
    command = "inspect the reported detail and confirm the active HERMES_HOME/LCM_DATABASE_PATH"
    warning_only = False
    rationale = "operator review required before changing persisted LCM state"

    if name == "database_integrity":
        command = "stop and inspect the SQLite database path; restore from backup if integrity_check is not ok"
    elif name == "schema_core_tables":
        command = "verify HERMES_HOME/LCM_DATABASE_PATH points at the intended LCM database before repair or restore"
    elif name in {"messages_fts_integrity", "nodes_fts_integrity", "fts_index_sync"}:
        if status == "warn" and isinstance(detail, dict) and detail.get("status") == "unchecked":
            action = DOCTOR_ACTION_INSPECT
            command = "rerun the doctor with read-write SQLite access if a deep FTS integrity result is needed"
            warning_only = True
            rationale = "the deep FTS check could not run, but this is not evidence that the index is corrupt"
        else:
            action = DOCTOR_ACTION_BACKUP_FIRST_CLEANUP
            command = "back up the database, then rebuild the FTS index from the stored rows"
            rationale = "FTS repair is rebuildable, but it still mutates SQLite indexes"
    elif name == "sqlite_storage":
        command = "inspect journal/quick_check output and database/WAL size; restore from backup if SQLite reports corruption"
    elif name == "orphaned_dag_nodes":
        command = "inspect affected DAG/source IDs; do not auto-delete summaries without confirming recall impact"
        if status == "warn":
            warning_only = True
        else:
            rationale = "DAG diagnostic failures mean doctor could not read summary/source state reliably"
    elif name == "summary_quality":
        command = "inspect worst_nodes and retrieval behavior; treat as summary quality evidence, not cleanup input"
        if status == "warn":
            warning_only = True
        else:
            rationale = "summary-quality diagnostic failures mean doctor could not read DAG quality state reliably"
    elif name == "config_validation":
        command = "inspect LCM_* environment/config values and adjust only intentional operator overrides"
    elif name == "source_lineage_hygiene" and status == "warn":
        action = DOCTOR_ACTION_SAFE_IGNORE
        command = "safe to ignore legacy blank-source observations"
        rationale = "legacy blank sources are read as unknown for compatibility"
    elif name == "source_lineage_hygiene":
        command = "inspect source-lineage diagnostics and SQLite read errors"
        rationale = "source-lineage failures indicate the doctor could not read attribution state reliably"
    elif name == "context_pressure":
        action = DOCTOR_ACTION_SAFE_IGNORE
        command = "safe to ignore if compaction proceeds normally; inspect lcm_status only if pressure stays high or compaction loops"
        warning_only = True
        rationale = "context pressure is an operating state, not persisted-state corruption"

    return {
        "check": name,
        "status": status,
        "action": action,
        "operator_action": command,
        "warning_only": warning_only,
        "rationale": rationale,
    }


def doctor_guidance_for_checks(checks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return actionable guidance for warning/failing lcm_doctor checks."""
    guidance = []
    for check in checks:
        item = doctor_guidance_for_check(check)
        if item is not None:
            guidance.append(item)
    return guidance
