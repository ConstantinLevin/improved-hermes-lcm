"""LCM Engine — Lossless Context Management.

Implements the ContextEngine ABC. Replaces the built-in ContextCompressor
with a DAG-based summarization system that preserves every message.
"""

import copy
import json
import logging
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.context_engine import ContextEngine

from .codex_routing import (
    _codex_oauth_context_cap,
    _is_codex_gpt55_route,
)
from .config import LCMConfig
from .dag import SummaryDAG, SummaryNode
from .db_bootstrap import STORE_FILENAME, StoreRefusedError
from .engine_registry import (
    _ACTIVE_ENGINE_REGISTRY_LOCK,
    _ACTIVE_ENGINES_BY_CONVERSATION_ID,
    _ACTIVE_ENGINES_BY_SESSION_ID,
    _remove_registry_entries_for_engine,
    resolve_active_lcm_engine,  # noqa: F401  (re-exported: hosts import it from .engine)
)
from .escalation import (
    SummaryCircuitBreaker,
    SummarySpendGuard,
    summarize_with_escalation,
)
from .externalize import load_externalized_payload
from .extraction import (
    sanitize_pre_compaction_content,
    sanitize_pre_compaction_tool_arguments,
    strip_injected_context_blocks,
)
from .ingest_protection import (
    _is_hermes_persisted_output_marker,
    extract_ingest_externalized_refs,
    protect_inline_payloads_in_text,
    protect_messages_for_ingest,
    quarantine_suspicious_assistant_messages,
    restore_ingest_payload_placeholders,
)
from .runtime_identity import (
    _PLUGIN_ROOT,
    _git_runtime_identity,
    _plugin_metadata,
)
from .schemas import (
    LCM_DOCTOR,
    LCM_EXPAND,
    LCM_EXPAND_QUERY,
    LCM_GREP,
    LCM_INSPECT,
    LCM_STATUS,
)
from .sanitize import (
    _clean_active_assistant_message,
)
from .message_analysis import (
    _is_synthetic_assistant_noise,
    _matched_tool_call_ids,
    _tool_call_id,
)
from .fresh_tail import FreshTailBoundary, resolve_fresh_tail_boundary
from .placeholder_ledger import PlaceholderLedgerMixin
from .reconcile import ReconcileMixin, _PRESERVED_OBJECTIVE_CONTEXT_PREFIX
from .compaction import CompactionMixin
from .reset_state import ResetStateMixin
from .plugin_sessions import PluginSessions
from .record_store import RecordStore
from .record_write import RecordWriteMixin
from .message_content import (
    normalize_content_value,
    text_content_for_pattern_matching,
)
from .store import MessageStore
from .tokens import count_message_tokens, count_messages_tokens, count_tokens
from . import tools as lcm_tools

logger = logging.getLogger(__name__)


_CODEX_GPT55_COMPACTION_THRESHOLD = 0.85
_TOTAL_COMPACTIONS_SCOPE = "plugin_session"

# Auto-focus topic derivation: infer a compact focus hint from the most recent
# real user turns so that summarization can prioritise current user intent.
# Mirrors Hermes upstream fix/compression-auto-focus-topic (#44687 branch).
_AUTO_FOCUS_MAX_TURNS = 3
_AUTO_FOCUS_TURN_MAX_CHARS = 260
_AUTO_FOCUS_MAX_CHARS = 700

_PRESERVED_TODO_CONTEXT_PREFIX = "[Your active task list was preserved across context compression]"


class ReviewForkDetachRefused(RuntimeError):
    """Raised when the host detaches an engine copy from every session.

    The host makes this call, ``bind_session_state(session_db=None, session_id="")``,
    on the engine copy of its background-review fork and enables the fork's
    compaction only when it succeeds. The host neither commits nor confirms a fork's
    compaction and names no parent for the fork, so the plugin could record it only
    by guessing. Refusing keeps the fork's compaction disabled, as it was.
    """


def _normalize_total_compactions(value: Any) -> int:
    """Return a persisted compaction total only when it is a valid counter."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


class LCMEngine(
    CompactionMixin,
    RecordWriteMixin,
    ResetStateMixin,
    ReconcileMixin,
    PlaceholderLedgerMixin,
    ContextEngine,
):
    """Lossless Context Management engine.

    Automatic LCM compaction is routine background maintenance. Hosts that
    support user-visible compaction status opt-outs should keep successful
    automatic LCM passes silent unless the user explicitly asks for diagnostics.

    Architecture:
      1. When context pressure builds, older messages outside the fresh tail
         are summarized into leaf nodes (D0) in a SummaryDAG
      2. When enough nodes accumulate at a depth, they're condensed into
         higher-depth nodes (D1, D2, ...)
      3. The agent gets tools (lcm_grep, lcm_expand, lcm_expand_query) to
         search and drill into compacted history
      4. Active context = system prompt + DAG summaries + fresh tail
    """

    def __init__(self, config: LCMConfig | None = None,
                 hermes_home: str = ""):
        self._config = config or LCMConfig.from_env()
        self._hermes_home = hermes_home

        db_path = self._resolve_db_path(hermes_home)
        self._bind_storage(db_path, hermes_home)

        self._session_id: str = ""
        self._session_platform: str = ""
        self._conversation_id: str = ""
        # The plugin session this engine copy acts for, set only from this copy's
        # own bind_session_state() and on_session_start(). Nothing else names it:
        # not a hook's session id, not a turn id, not another copy's session.
        self._plugin_session: str = ""
        # This copy's attempts whose return is written, by compaction id, until all
        # their returned entries are bound; confirmation and binding read the
        # returned dicts from here. The attempt that returned last, for a rejection;
        # a confirmation whose compaction the next list must name.
        self._returned_attempts: dict = {}
        self._last_returned_attempt = None
        self._pending_confirmation = None

        # Track which store_ids have been ingested into the DAG
        self._last_compacted_store_id: int = 0

        # Cursor: index in the current messages list up to which all
        # messages have been persisted.  After compress() shortens the
        # list, the cursor resets to len(compressed) so that only
        # genuinely new messages (appended after compaction) get ingested.
        # The cursor is process-local; existing sessions rebound after a
        # gateway restart reconcile it against the durable store on the
        # next ingest.
        self._ingest_cursor: int = 0
        self._ingest_cursor_needs_reconcile = False
        self._last_ingest_reconciliation: Dict[str, Any] = {
            "action": "none",
            "reason": "not run",
        }

        # State required by ContextEngine ABC and run_agent.py compatibility
        self.model = ""
        self.base_url = ""
        self.api_key = ""
        self.provider = ""
        self.api_mode = ""
        self.raw_context_length = 0
        self.context_length = 0
        self.effective_context_length_cap: int | None = None
        self.effective_context_length_reason = ""
        self._context_length_source = ""
        self._update_model_pending_session_start = False
        self.threshold_tokens = 0
        self.context_threshold = self._config.context_threshold
        self.threshold_percent = self.context_threshold
        self._context_threshold_source = (
            self._config.config_sources.get("context_threshold", "manual_or_default")
            if getattr(self._config, "config_sources", None)
            else "manual_or_default"
        )
        self._context_threshold_autoraised: dict[str, float] | None = None
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0
        self.last_input_tokens = 0
        self.last_output_tokens = 0
        self.last_cache_read_tokens = 0
        self.last_cache_write_tokens = 0
        self.last_reasoning_tokens = 0
        self.cache_metrics_available = False
        self.compression_count = 0
        # Distinguishes this reset-scoped process counter from overlapping or
        # previous runtimes that write the same conversation telemetry row.
        self._compaction_telemetry_counter_epoch = uuid.uuid4().hex
        self._compaction_telemetry_counter_rebaseline_pending = True
        self._compaction_telemetry_turn_reset_pending = False
        # Wall-clock of the last leaf compaction (ms); surfaced via telemetry only.
        self._last_compaction_duration_ms = 0.0
        # run_agent.py reads these for preflight checks
        self.protect_first_n = 3
        self.protect_last_n = self._config.fresh_tail_count
        # run_agent.py reads these for context probing
        self._context_probed = False
        self._context_probe_persistable = False
        # Host compatibility: LCM treats successful automatic compaction as
        # silent maintenance. Manual /lcm diagnostics and warning/error paths
        # remain explicit.
        self.emit_automatic_compaction_status = False
        self.quiet_mode = True
        self.summary_model = self._config.summary_model
        self._summary_circuit_breaker = SummaryCircuitBreaker(
            failure_threshold=self._config.summary_circuit_breaker_failure_threshold,
            cooldown_seconds=self._config.summary_circuit_breaker_cooldown_seconds,
        )
        # Summary spend guard: process-local sliding window so a loop that
        # keeps succeeding cannot burn auxiliary-model budget without bound. When
        # tripped, escalation falls back to deterministic L3 truncation. Set
        # summary_spend_max_calls=0 to disable.
        self._summary_spend_guard = SummarySpendGuard(
            max_calls=int(self._config.summary_spend_max_calls),
            window_seconds=float(self._config.summary_spend_window_seconds),
            backoff_seconds=float(self._config.summary_spend_backoff_seconds),
        )
        self._last_overflow_recovery_failed = False
        self._last_condensation_suppressed_reason = ""
        self._last_threshold_full_sweep: dict[str, Any] = {
            "status": "never_run",
            "leaf_passes": 0,
            "condensation_passes": 0,
            "total_passes": 0,
            "duration_ms": 0.0,
            "tokens_before": 0,
            "tokens_after": 0,
            "summary_prefix_tokens_before": 0,
            "summary_prefix_tokens_after": 0,
            "summary_prefix_target_tokens": 0,
            "stop_reason": "",
            "budget_exhausted": False,
        }
        self._last_compression_status = "idle"
        self._last_compression_noop_reason = ""
        # Read by the host after compress(): an aborted compaction returned its input
        # unchanged, and the host shows "⚠ Compression aborted: <_last_summary_error>".
        self._last_compress_aborted = False
        self._last_summary_error: Optional[str] = None
        # Ingest-failure tracking. The core promise is that nothing is ever
        # lost, but a swallowed persistence error (disk full, DB locked,
        # corruption) silently breaks it: the turn continues while messages
        # exist only in the volatile host list. Surface it instead of hiding
        # it in a debug log so get_status()/doctor can escalate. Store-scoped,
        # not session-scoped, so it is not cleared on session reset.
        self._ingest_failure_count = 0
        self._consecutive_ingest_failures = 0
        self._last_ingest_error = ""
        self._last_ingest_error_time: float = 0
        # Cooldown timestamp to prevent compression cascade after boundary skip.
        # Set when skip-carry-over path is taken in _continue_compression_boundary.
        self._last_boundary_skip_time: float = 0
        # One-shot handoff from preflight: adopt an already-durable replay
        # cleanup during boundary cooldown without running summary work.
        self._preflight_cleanup_only_due_to_boundary_cooldown = False
        # Temporary source window used only while compress() assembles context.
        # _assemble_context also serves tests and recovery paths directly, so
        # keep anchoring opt-in rather than changing its public behavior.
        self._pending_context_anchor_messages: Optional[List[Dict[str, Any]]] = None
        self._current_compress_store_ids_by_message_id: dict[int, int] = {}
        self._last_active_replay_source_identities: list[tuple[Any, ...]] = []
        self._last_active_replay_messages: list[Dict[str, Any]] = []
        self._pending_reset_session_id: str = ""
        self._pending_reset_conversation_id: str = ""
        self._pending_reset_frontier_store_id: int = 0

    def clone_for_agent(self) -> "LCMEngine":
        """Return a fresh runtime engine for one AIAgent instance.

        Hermes registers plugin context engines process-wide, while gateway
        runtimes may keep multiple cached AIAgent instances alive at once
        (different platforms, chats, cron jobs, etc.).  LCM stores mutable
        session binding and ingest cursor state on the engine object itself, so
        sharing one registered instance across agents can let one conversation
        rebind another conversation's raw-message ingest and lifecycle state.

        The clone shares the same durable SQLite database path/configuration,
        but gets independent session/cursor/lifecycle runtime state. Runtime
        model and context-window metadata is copied so the clone is immediately
        budget-aware even before a compatible Hermes host calls update_model().
        """
        clone = type(self)(
            config=copy.deepcopy(self._config),
            hermes_home=self._hermes_home,
        )
        clone.model = self.model
        clone.base_url = self.base_url
        clone.api_key = self.api_key
        clone.provider = self.provider
        clone.api_mode = self.api_mode
        if self._context_length_source:
            clone._set_context_length(
                self.raw_context_length,
                source=self._context_length_source,
                model=self.model,
                provider=self.provider,
            )
        elif self.raw_context_length or self.context_length:
            clone._set_context_length(
                self.raw_context_length or self.context_length,
                source="clone_for_agent",
                model=self.model,
                provider=self.provider,
            )
        # ``update_model()`` authority is a per-runtime lifecycle edge, not
        # durable metadata.  Compatible hosts call update_model() on the clone
        # before binding it; hosts that bind only through on_session_start()
        # must still be able to replace the copied prototype route.
        clone._update_model_pending_session_start = False
        return clone

    def __deepcopy__(self, memo: dict[int, object]) -> "LCMEngine":
        """Copy the plugin runtime without pickling SQLite-backed helpers.

        Hermes core may deepcopy plugin context engines while creating isolated
        AIAgent instances. A default object deepcopy walks into MessageStore,
        SummaryDAG, PluginSessions and RecordStore sqlite3.Connection handles, which
        cannot be pickled. LCM already exposes clone_for_agent() as the safe
        boundary: share durable configuration/database path, but allocate fresh
        per-agent runtime/storage helper objects.
        """
        clone = self.clone_for_agent()
        memo[id(self)] = clone
        return clone

    def _resolve_db_path(self, hermes_home: str = "") -> Path:
        """Resolve the store's path: ``LCM_DATABASE_PATH``, else the host-given home.

        With neither, the location is not known, and the plugin does not guess one.
        """
        if self._config.database_path:
            return Path(self._config.database_path)
        if hermes_home:
            return Path(hermes_home) / STORE_FILENAME
        message = (
            "LCM has no store location: the host gave no Hermes home and "
            "LCM_DATABASE_PATH is not set."
        )
        logger.error(message)
        raise StoreRefusedError(message)

    def _bind_storage(self, db_path: str | Path, hermes_home: str = "") -> None:
        """Bind the store's helpers to one SQLite database: the record and its
        sessions, and the tools' readers over its views."""
        try:
            self._store = MessageStore(
                db_path,
                ingest_protection_config=self._config,
                hermes_home=hermes_home,
            )
            self._dag = SummaryDAG(db_path)
            self._sessions = PluginSessions(db_path)
            self._records = RecordStore(db_path)
        except Exception:
            self._close_storage()
            raise

    def _close_storage(self) -> None:
        """Best-effort close of currently bound SQLite helpers."""
        for attr in (
            "_store",
            "_dag",
            "_sessions",
            "_records",
        ):
            helper = getattr(self, attr, None)
            close = getattr(helper, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    logger.debug("LCM failed closing %s during profile rebind", attr, exc_info=True)


    def _reset_profile_runtime_state(self) -> None:
        """Clear process-local session state that cannot cross profile homes."""
        self._unregister_active_engine_binding()
        self._session_id = ""
        self._session_platform = ""
        self._conversation_id = ""
        self._plugin_session = ""
        self._returned_attempts = {}
        self._last_returned_attempt = None
        self._pending_confirmation = None
        self._clear_pending_reset_boundary()
        self._reset_session_scoped_runtime_state()

    def _rebind_storage_for_home(self, hermes_home: str = "") -> bool:
        """Switch SQLite-backed state when a reused engine serves another profile.

        Hermes core passes the active ``hermes_home`` on session start.  Older
        Hermes versions may still reuse the same plugin/context-engine object
        after ``HERMES_HOME`` changes, so the plugin must not assume the store
        captured during ``register()`` is still correct.
        """
        if not hermes_home:
            return False
        if self._config.database_path:
            current_home = str(self._hermes_home or "")
            current_store_home = str(getattr(getattr(self, "_store", None), "_hermes_home", "") or "")
            if current_home == str(hermes_home) and current_store_home == str(hermes_home):
                return False
            self._hermes_home = hermes_home
            store = getattr(self, "_store", None)
            if store is not None:
                store._hermes_home = hermes_home
            self._reset_profile_runtime_state()
            logger.info("LCM rebound Hermes home for configured database path %s", hermes_home)
            return True

        db_path = self._resolve_db_path(hermes_home)
        current_db = Path(getattr(getattr(self, "_store", None), "db_path", ""))
        if current_db == db_path and str(self._hermes_home or "") == str(hermes_home):
            return False

        self._close_storage()
        self._hermes_home = hermes_home
        self._bind_storage(db_path, hermes_home)
        self._reset_profile_runtime_state()
        logger.info("LCM rebound storage for Hermes home %s", hermes_home)
        return True

    def _runtime_context_threshold(
        self,
        *,
        model: str | None = None,
        provider: str | None = None,
    ) -> tuple[float, str, dict[str, float] | None]:
        configured = float(self._config.context_threshold)
        source = (
            self._config.config_sources.get("context_threshold", "manual_or_default")
            if getattr(self._config, "config_sources", None)
            else "manual_or_default"
        )
        explicit_lcm_override = source in {
            "env:LCM_CONTEXT_THRESHOLD",
            "config_yaml:lcm.context_threshold",
        }
        route_model = self.model if model is None else model
        route_provider = self.provider if provider is None else provider
        if (
            _is_codex_gpt55_route(route_model, route_provider)
            and self._config.codex_gpt55_autoraise_enabled
            and not explicit_lcm_override
            and configured < _CODEX_GPT55_COMPACTION_THRESHOLD
        ):
            return (
                _CODEX_GPT55_COMPACTION_THRESHOLD,
                "codex_gpt55_autoraise",
                {"from": configured, "to": _CODEX_GPT55_COMPACTION_THRESHOLD},
            )
        return configured, source, None

    def _effective_context_length(
        self,
        raw_context_length: int,
        *,
        model: str | None = None,
        provider: str | None = None,
    ) -> tuple[int, int | None, str]:
        route_model = self.model if model is None else model
        route_provider = self.provider if provider is None else provider
        cap = _codex_oauth_context_cap(route_model, route_provider)
        if cap is not None and raw_context_length > cap:
            return (
                cap,
                cap,
                "codex_oauth_context_cap",
            )
        return raw_context_length, None, ""

    def _effective_threshold_tokens(self, context_threshold_tokens: int) -> int:
        """Return the host-visible preflight trigger token count.

        Hermes core uses ``threshold_tokens`` as a cheap gate before it pays for
        the full request estimate that includes system prompt and tool schemas.
        LCM can enforce a stricter active-context assembly cap than the normal
        context-threshold value, so expose the stricter cap here; otherwise a
        tool/schema-heavy request can skip host preflight entirely.
        """
        assembly_cap = self._effective_assembly_token_cap()
        if assembly_cap is not None and assembly_cap > 0:
            if context_threshold_tokens > 0:
                return min(context_threshold_tokens, assembly_cap)
            return assembly_cap
        return context_threshold_tokens

    def _set_context_length(
        self,
        context_length: Any,
        *,
        source: str,
        model: str | None = None,
        provider: str | None = None,
    ) -> bool:
        try:
            parsed_context_length = int(context_length)
        except (TypeError, ValueError):
            logger.debug("LCM ignored invalid %s context_length: %r", source, context_length)
            return False
        if parsed_context_length <= 0:
            logger.debug(
                "LCM cleared non-positive %s context_length: %r",
                source,
                context_length,
            )
            self.raw_context_length = 0
            self.context_length = 0
            self.effective_context_length_cap = None
            self.effective_context_length_reason = ""
            self._context_length_source = source
            self.threshold_tokens = 0
            self.context_threshold, self._context_threshold_source, self._context_threshold_autoraised = (
                self._runtime_context_threshold(model=model, provider=provider)
            )
            self.threshold_percent = self.context_threshold
            return True
        self.raw_context_length = parsed_context_length
        effective_context_length, cap, reason = self._effective_context_length(
            parsed_context_length,
            model=model,
            provider=provider,
        )
        self.context_length = effective_context_length
        self.effective_context_length_cap = cap
        self.effective_context_length_reason = reason
        self._context_length_source = source
        self.context_threshold, self._context_threshold_source, self._context_threshold_autoraised = (
            self._runtime_context_threshold(model=model, provider=provider)
        )
        self.threshold_percent = self.context_threshold
        context_threshold_tokens = int(
            effective_context_length * self.context_threshold
        )
        self.threshold_tokens = self._effective_threshold_tokens(
            context_threshold_tokens
        )
        return True

    def _session_metadata_matches_active_runtime(
        self,
        kwargs: Dict[str, Any],
        *,
        ignore_empty_optional: bool = False,
    ) -> bool:
        if "model" in kwargs and str(kwargs.get("model") or "") != self.model:
            return False
        for key in ("provider", "base_url", "api_key", "api_mode"):
            if key not in kwargs:
                continue
            incoming = str(kwargs.get(key) or "")
            if ignore_empty_optional and not incoming:
                continue
            if incoming != str(getattr(self, key, "") or ""):
                return False
        return True

    @property
    def name(self) -> str:
        return "lcm"


    def _mark_preflight_compression_requested(self) -> bool:
        """Record that preflight found work and clear any stale no-op reason."""
        self._last_compression_status = "pending"
        self._last_compression_noop_reason = ""
        return True

    @property
    def bound_session_id(self) -> str:
        """The host session identifier this engine copy is bound to."""
        return self._session_id

    @property
    def current_session_id(self) -> str:
        """The plugin session this engine copy serves; the tools read this one. It
        spans the host identifiers a compaction rotates through."""
        return self._plugin_session

    @property
    def current_session_platform(self) -> str:
        """Platform string paired with ``current_session_id``."""
        return self._session_platform

    @property
    def current_conversation_id(self) -> str:
        """Conversation id paired with ``current_session_id``."""
        return self._conversation_id

    # -- ContextEngine required methods ------------------------------------

    def update_from_response(self, usage: Dict[str, Any]) -> None:
        self.last_prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        self.last_completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        self.last_total_tokens = int(usage.get("total_tokens", 0) or 0)

        cache_keys = {"cache_read_tokens", "cache_write_tokens"}
        self.cache_metrics_available = any(key in usage for key in cache_keys)
        self.last_input_tokens = int(usage.get("input_tokens", self.last_prompt_tokens) or 0)
        self.last_output_tokens = int(
            usage.get("output_tokens", self.last_completion_tokens) or 0
        )
        self.last_cache_read_tokens = int(usage.get("cache_read_tokens", 0) or 0)
        self.last_cache_write_tokens = int(usage.get("cache_write_tokens", 0) or 0)
        self.last_reasoning_tokens = int(usage.get("reasoning_tokens", 0) or 0)

    @property
    def cache_read_ratio(self) -> float:
        if self.last_prompt_tokens <= 0:
            return 0.0
        return self.last_cache_read_tokens / self.last_prompt_tokens

    def _compaction_telemetry_counter_delta(
        self,
        existing: Dict[str, Any],
    ) -> tuple[int, bool, int]:
        """Return persisted baseline, reset state, and unrecorded compactions."""
        prev_count = int(existing.get("compression_count_at_record", 0) or 0)
        epoch_baseline = None
        watermarks = existing.get("counter_epoch_watermarks", [])
        if isinstance(watermarks, list):
            for item in watermarks:
                if (
                    isinstance(item, list)
                    and len(item) == 2
                    and item[0] == self._compaction_telemetry_counter_epoch
                    and isinstance(item[1], int)
                    and not isinstance(item[1], bool)
                    and item[1] >= 0
                ):
                    epoch_baseline = item[1]
                    break
        if epoch_baseline is not None:
            prev_count = epoch_baseline
        rebaseline_pending = bool(
            existing
            and self._compaction_telemetry_counter_rebaseline_pending
        )
        delta = (
            max(0, self.compression_count - epoch_baseline)
            if epoch_baseline is not None
            else self.compression_count
            if rebaseline_pending
            else max(0, self.compression_count - prev_count)
        )
        return prev_count, rebaseline_pending, delta

    def _record_successful_compaction_telemetry(self) -> None:
        """Durably count a completed leaf compaction before returning it."""
        conversation_id = self._conversation_id
        if not conversation_id:
            return
        try:
            existing = self._store.read_compaction_telemetry(conversation_id) or {}
            _, _, compaction_delta = self._compaction_telemetry_counter_delta(existing)
            if compaction_delta <= 0:
                return

            updates = {
                "conversation_id": conversation_id,
                "counter_epoch": self._compaction_telemetry_counter_epoch,
                "compression_count_at_record": self.compression_count,
                "turns_since_leaf_compaction": 0,
                "peak_prompt_tokens_since_leaf_compaction": 0,
                "last_leaf_compaction_at": time.time(),
                "last_compaction_duration_ms": round(self._last_compaction_duration_ms, 3),
            }
            self._store.increment_compaction_telemetry(
                conversation_id,
                compaction_delta,
                updates,
            )
            self._compaction_telemetry_counter_rebaseline_pending = False
            # The response hook still owns per-turn token/cache fields. Keep its
            # first post-compaction snapshot at turn zero without recounting.
            self._compaction_telemetry_turn_reset_pending = True
        except Exception:
            logger.debug("LCM successful compaction telemetry update failed", exc_info=True)

    def _record_turn_compaction_telemetry(self) -> None:
        """Persist a per-conversation compaction-telemetry snapshot for this turn.

        Best-effort and diagnostic only: any failure is logged at debug and never
        affects the turn. Turns with no token or cache signal are skipped so idle
        turns do not churn the record. The since-compaction accumulators reset off
        the monotonic ``compression_count``. Session resets mark the next
        telemetry write for an explicit zero-baseline comparison so compactions
        that happen before that write are not mistaken for an old baseline.
        """
        conversation_id = self._conversation_id
        if not conversation_id:
            return
        prompt_tokens = self.last_prompt_tokens
        cache_read = self.last_cache_read_tokens
        cache_write = self.last_cache_write_tokens
        if (
            prompt_tokens <= 0
            and cache_read <= 0
            and cache_write <= 0
            and not self.cache_metrics_available
        ):
            return
        try:
            existing = self._store.read_compaction_telemetry(conversation_id) or {}

            if cache_read > 0 or cache_write > 0:
                cache_state = "hot"
            elif self.cache_metrics_available:
                cache_state = "cold"
            else:
                cache_state = "unknown"
            cold_streak = int(existing.get("consecutive_cold_observations", 0) or 0)
            if cache_state == "hot":
                cold_streak = 0
            elif cache_state == "cold":
                cold_streak += 1

            (
                prev_count,
                counter_rebaseline_pending,
                compaction_delta,
            ) = self._compaction_telemetry_counter_delta(existing)
            compacted = compaction_delta > 0
            rebaselined = (
                self._compaction_telemetry_turn_reset_pending
                or counter_rebaseline_pending
                or self.compression_count != prev_count
            )
            if rebaselined:
                turns_since = 0
                peak_tokens_since = prompt_tokens
            else:
                turns_since = int(existing.get("turns_since_leaf_compaction", 0) or 0) + 1
                peak_tokens_since = max(
                    int(existing.get("peak_prompt_tokens_since_leaf_compaction", 0) or 0),
                    prompt_tokens,
                )
            total_compactions = _normalize_total_compactions(
                existing.get("total_compactions", 0)
            )
            if compacted:
                total_compactions += compaction_delta
                last_leaf_compaction_at = time.time()
                last_compaction_duration_ms = round(self._last_compaction_duration_ms, 3)
            else:
                last_leaf_compaction_at = existing.get("last_leaf_compaction_at")
                last_compaction_duration_ms = existing.get("last_compaction_duration_ms")

            record = dict(existing)
            record.update({
                "conversation_id": conversation_id,
                "last_observed_prompt_tokens": prompt_tokens,
                "last_observed_cache_read": cache_read,
                "last_observed_cache_write": cache_write,
                "cache_state": cache_state,
                "consecutive_cold_observations": cold_streak,
                "turns_since_leaf_compaction": turns_since,
                "peak_prompt_tokens_since_leaf_compaction": peak_tokens_since,
                # Reserved carry-forward field; no live 'medium'/'high' computation yet.
                "activity_band": existing.get("activity_band", "low"),
                "provider": self.provider or existing.get("provider"),
                "model": self.model or existing.get("model"),
                "last_api_call_at": time.time(),
                "last_leaf_compaction_at": last_leaf_compaction_at,
                "last_compaction_duration_ms": last_compaction_duration_ms,
                "total_compactions": total_compactions,
                "counter_epoch": self._compaction_telemetry_counter_epoch,
                "compression_count_at_record": self.compression_count,
            })
            if cache_state == "hot":
                record["last_cache_hit_at"] = time.time()
            # Even zero-delta snapshots use the transactional updater so an
            # overlapping snapshot cannot overwrite a newly incremented total.
            self._store.increment_compaction_telemetry(
                conversation_id,
                compaction_delta,
                record,
            )
            self._compaction_telemetry_counter_rebaseline_pending = False
            self._compaction_telemetry_turn_reset_pending = False
        except Exception:
            logger.debug("LCM compaction telemetry update failed", exc_info=True)

    def _compression_boundary_cooldown_active(self) -> bool:
        """Return true while a boundary skip is in its short no-compress window."""
        if self._last_boundary_skip_time <= 0:
            return False
        elapsed = time.time() - self._last_boundary_skip_time
        if elapsed < 60:
            logger.debug(
                "LCM compression cooldown active: %.1f seconds since boundary skip",
                elapsed,
            )
            return True
        self._last_boundary_skip_time = 0
        return False

    def _record_ingest_success(self) -> None:
        self._consecutive_ingest_failures = 0

    def _record_ingest_failure(self, where: str, error: Exception) -> None:
        """Track a swallowed ingest error so it is operator-visible.

        Escalates to error level once failures are consecutive: a single
        transient lock is a warning, but a sustained inability to persist
        means the lossless guarantee is broken and must not stay hidden.
        """
        self._ingest_failure_count += 1
        self._consecutive_ingest_failures += 1
        self._last_ingest_error = f"{type(error).__name__}: {error}"
        self._last_ingest_error_time = time.time()
        message = "LCM ingest failed (%s): %s [consecutive=%d, total=%d]"
        args = (
            where,
            error,
            self._consecutive_ingest_failures,
            self._ingest_failure_count,
        )
        if self._consecutive_ingest_failures >= 3:
            logger.error(message, *args)
        else:
            logger.warning(message, *args)

    def ingest(self, messages: List[Dict[str, Any]]) -> None:
        """Persist messages to the durable store every turn.

        Called by the post_llm_call plugin hook so messages land in LCM
        regardless of whether compression triggers — short WebUI
        conversations never hit the compression threshold and never
        expire like Telegram sessions do, so without this they'd never
        be ingested.

        Uses the same _ingest_messages cursor as compress(), so if
        compression runs later the same turn, already-ingested messages
        are skipped (no duplicates).
        """
        if self._session_id and messages:
            try:
                self._ingest_messages(messages)
                self._record_ingest_success()
                logger.debug(
                    "Per-turn ingest OK: session=%s msgs=%d cursor=%d",
                    self._session_id, len(messages), self._ingest_cursor,
                )
            except Exception as e:
                self._record_ingest_failure("per-turn ingest()", e)

    def _is_retry_worthy_leaf_summary_error(self, exc: Exception) -> bool:
        if isinstance(exc, TimeoutError):
            return True
        message = str(exc).lower()
        retry_markers = (
            "context length",
            "maximum context",
            "max context",
            "too many tokens",
            "token limit",
            "prompt is too long",
            "input too long",
            "request too large",
            "timed out",
            "timeout",
        )
        return any(marker in message for marker in retry_markers)

    def _next_leaf_rescue_chunk(
        self,
        current_chunk: List[Dict[str, Any]],
        current_source_tokens: int,
    ) -> List[Dict[str, Any]]:
        if len(current_chunk) <= 1:
            return []

        floor_tokens = max(1, self._config.leaf_chunk_tokens)
        shrink_targets = [
            max(floor_tokens, int(current_source_tokens * 0.75)),
            max(floor_tokens, int(current_source_tokens * 0.50)),
        ]

        for target in shrink_targets:
            if target >= current_source_tokens:
                continue
            smaller = self._select_oldest_leaf_chunk(current_chunk, target)
            if smaller and len(smaller) < len(current_chunk):
                return smaller

        return current_chunk[:-1]

    def _summarize_leaf_chunk_with_rescue(
        self,
        initial_chunk: List[Dict[str, Any]],
        focus_topic: Optional[str] = None,
        deadline: Optional[float] = None,
    ) -> tuple[List[Dict[str, Any]], int, str, int, int]:
        attempt_chunk = list(initial_chunk)
        max_attempts = 3
        attempt_number = 0

        while attempt_chunk and attempt_number < max_attempts:
            attempt_number += 1
            source_tokens = count_messages_tokens(attempt_chunk)
            serialized = self._serialize_messages(attempt_chunk)
            source_store_ids = sorted(dict.fromkeys(
                self._current_compress_store_ids_by_message_id[id(message)]
                for message in attempt_chunk
                if id(message) in self._current_compress_store_ids_by_message_id
            ))
            token_budget = max(2000, int(source_tokens * 0.20))
            token_budget = min(token_budget, 12000)

            try:
                timeout_seconds = self._config.summary_timeout_ms / 1000
                if deadline is not None:
                    remaining_seconds = deadline - time.monotonic()
                    if remaining_seconds <= 0:
                        raise TimeoutError("threshold full sweep time budget exhausted")
                    timeout_seconds = min(timeout_seconds, remaining_seconds)
                summary_text, level = summarize_with_escalation(
                    text=serialized,
                    source_tokens=source_tokens,
                    token_budget=token_budget,
                    depth=0,
                    model=self._config.summary_model,
                    fallback_models=self._config.summary_fallback_models,
                    circuit_breaker=self._summary_circuit_breaker,
                    spend_guard=self._summary_spend_guard,
                    timeout=timeout_seconds,
                    l2_budget_ratio=self._config.l2_budget_ratio,
                    l3_truncate_tokens=self._config.l3_truncate_tokens,
                    focus_topic=focus_topic or "",
                    custom_instructions=self._config.custom_instructions,
                    source_provenance={
                        "source_type": "messages",
                        "store_ids": source_store_ids,
                        "message_count": len(attempt_chunk),
                    },
                )
                return attempt_chunk, source_tokens, summary_text, level, attempt_number
            except Exception as exc:
                if attempt_number >= max_attempts or not self._is_retry_worthy_leaf_summary_error(exc):
                    raise
                smaller_chunk = self._next_leaf_rescue_chunk(attempt_chunk, source_tokens)
                if not smaller_chunk or len(smaller_chunk) >= len(attempt_chunk):
                    raise
                logger.warning(
                    "LCM leaf summarization retrying with smaller oldest chunk after retry-worthy failure: %s (attempt %d/%d, %d→%d messages)",
                    exc,
                    attempt_number,
                    max_attempts,
                    len(attempt_chunk),
                    len(smaller_chunk),
                )
                attempt_chunk = smaller_chunk

        raise RuntimeError("adaptive leaf rescue exhausted without a valid chunk")

    # -- ContextEngine optional methods ------------------------------------


    def _bind_lifecycle_state(
        self,
        session_id: str,
        *,
        conversation_id: str | None = None,
    ) -> None:
        state = self._lifecycle.bind_session(session_id, conversation_id=conversation_id)
        self._conversation_id = state.conversation_id
        self._last_compacted_store_id = state.current_frontier_store_id
        self._register_active_engine_binding()

    def _register_active_engine_binding(self) -> None:
        session_id = str(self._session_id or "")
        conversation_id = str(self._conversation_id or "")
        if not session_id:
            return
        with _ACTIVE_ENGINE_REGISTRY_LOCK:
            _remove_registry_entries_for_engine(
                self,
                keep_session_id=session_id,
                keep_conversation_id=conversation_id,
            )
            _ACTIVE_ENGINES_BY_SESSION_ID[session_id] = self
            if conversation_id:
                _ACTIVE_ENGINES_BY_CONVERSATION_ID[conversation_id] = self

    def _unregister_active_engine_binding(self) -> None:
        with _ACTIVE_ENGINE_REGISTRY_LOCK:
            _remove_registry_entries_for_engine(self)

    def _persist_frontier_marker(self) -> None:
        if not self._session_id or not self._conversation_id:
            return
        self._lifecycle.advance_frontier(
            self._conversation_id,
            self._session_id,
            self._last_compacted_store_id,
        )

    def _clear_pending_reset_boundary(self) -> None:
        self._pending_reset_session_id = ""
        self._pending_reset_conversation_id = ""
        self._pending_reset_frontier_store_id = 0

    def _finalize_pending_reset_boundary(self, session_id: str) -> None:
        if not self._pending_reset_session_id:
            return
        if self._pending_reset_session_id != session_id:
            self._clear_pending_reset_boundary()
            return
        if not self._pending_reset_conversation_id:
            self._clear_pending_reset_boundary()
            return
        state = self._lifecycle.get_by_conversation(self._pending_reset_conversation_id)
        frontier_store_id = self._pending_reset_frontier_store_id
        if state is not None and state.current_session_id == session_id:
            frontier_store_id = max(
                frontier_store_id,
                int(state.current_frontier_store_id or 0),
            )
        self._lifecycle.finalize_session(
            self._pending_reset_conversation_id,
            self._pending_reset_session_id,
            frontier_store_id=frontier_store_id,
        )
        self._clear_pending_reset_boundary()

    def _fresh_tail_boundary(self, messages: List[Dict[str, Any]]) -> FreshTailBoundary:
        return resolve_fresh_tail_boundary(
            messages,
            fresh_tail_count=self._config.fresh_tail_count,
            fresh_tail_max_tokens=self._config.fresh_tail_max_tokens,
        )

    def _fresh_tail_start(self, messages: List[Dict[str, Any]]) -> int:
        return self._fresh_tail_boundary(messages).start

    def _get_session_fresh_tail(
        self,
        session_id: str,
        *,
        minimum_count: int = 0,
    ) -> tuple[List[Dict[str, Any]], FreshTailBoundary]:
        """Load and resolve a stored tail, expanding backward for tool pairing."""
        total_count = int(self._store.get_session_count(session_id))
        configured_count = max(minimum_count, int(self._config.fresh_tail_count or 0))
        if self._config.fresh_tail_max_tokens > 0:
            configured_count = max(1, configured_count)
        if total_count <= 0 or configured_count <= 0:
            return [], resolve_fresh_tail_boundary(
                [],
                fresh_tail_count=configured_count,
                fresh_tail_max_tokens=self._config.fresh_tail_max_tokens,
            )

        load_limit = min(total_count, configured_count)
        while True:
            rows = self._store.get_session_tail(session_id, load_limit)
            boundary = resolve_fresh_tail_boundary(
                rows,
                fresh_tail_count=configured_count,
                fresh_tail_max_tokens=self._config.fresh_tail_max_tokens,
            )
            selected = rows[boundary.start:]
            unresolved_tool_boundary = bool(
                selected
                and selected[0].get("role") == "tool"
                and not boundary.tool_group_extended
                and load_limit < total_count
            )
            if not unresolved_tool_boundary:
                return selected, boundary
            load_limit = min(total_count, max(load_limit + 1, load_limit * 2))

    @staticmethod
    def _leading_anchor_count(messages: List[Dict[str, Any]]) -> int:
        """Return the number of non-compactable leading messages.

        Only the system prompt is a safe permanent anchor. Hermes gateway
        sessions can begin with a user message when core passes conversation
        history without a system prompt; preserving that first user turn as raw
        active context lets stale requests look current after later compaction.
        """
        if messages and isinstance(messages[0], dict) and messages[0].get("role") == "system":
            return 1
        return 0

    def _apply_session_start_metadata(self, session_id: str, kwargs: Dict[str, Any]) -> None:
        self._session_id = session_id
        self._session_platform = str(kwargs.get("platform") or "")
        if "hermes_home" in kwargs:
            self._hermes_home = kwargs["hermes_home"]

        update_model_is_authoritative = (
            self._context_length_source == "update_model"
            and self._update_model_pending_session_start
        )

        # Pick up context_length from kwargs if provided, but do not let stale
        # session metadata undo the authoritative runtime update_model() call.
        # Hermes Agent calls update_model() with the resolver output before it
        # binds a fresh agent/session.  Older or buggy host paths can still pass
        # a context_length copied from the previously bound runtime; treating
        # that as authoritative makes /model switches keep compressing against
        # the old model window.
        if "context_length" in kwargs:
            incoming_context_length = kwargs["context_length"]
            try:
                parsed_context_length = int(incoming_context_length)
            except (TypeError, ValueError):
                logger.debug(
                    "LCM ignored invalid session-start context_length: %r",
                    incoming_context_length,
                )
                self._update_model_pending_session_start = False
                return
            if parsed_context_length <= 0:
                if update_model_is_authoritative:
                    if self._session_metadata_matches_active_runtime(
                        kwargs,
                        ignore_empty_optional=True,
                    ):
                        logger.debug(
                            "LCM ignored missing session-start context_length=%r for model=%s; active update_model context_length=%s",
                            incoming_context_length,
                            self.model or str(kwargs.get("model") or ""),
                            self.context_length,
                        )
                    else:
                        logger.warning(
                            "LCM ignored stale session-start runtime metadata for model=%s; active update_model model=%s",
                            str(kwargs.get("model") or ""),
                            self.model,
                        )
                    self._update_model_pending_session_start = False
                    return
                self._set_context_length(parsed_context_length, source="session_start")
                update_model_is_authoritative = False
            else:
                if (
                    update_model_is_authoritative
                    and parsed_context_length not in {self.context_length, self.raw_context_length}
                ):
                    logger.warning(
                        "LCM ignored stale session-start context_length=%s for model=%s; active update_model raw_context_length=%s effective_context_length=%s",
                        parsed_context_length,
                        self.model or str(kwargs.get("model") or ""),
                        self.raw_context_length,
                        self.context_length,
                    )
                    self._update_model_pending_session_start = False
                    return
                if update_model_is_authoritative:
                    if not self._session_metadata_matches_active_runtime(kwargs):
                        logger.warning(
                            "LCM ignored stale session-start runtime metadata for model=%s; active update_model model=%s",
                            str(kwargs.get("model") or ""),
                            self.model,
                        )
                        self._update_model_pending_session_start = False
                        return
                else:
                    self._set_context_length(
                        parsed_context_length,
                        source="session_start",
                        model=str(kwargs.get("model") or self.model),
                        provider=str(kwargs.get("provider") or self.provider),
                    )
                    update_model_is_authoritative = False
        if (
            update_model_is_authoritative
            and not self._session_metadata_matches_active_runtime(kwargs)
        ):
            logger.warning(
                "LCM ignored stale session-start runtime metadata for model=%s; active update_model model=%s",
                str(kwargs.get("model") or ""),
                self.model,
            )
            self._update_model_pending_session_start = False
            return
        if "model" in kwargs:
            self.model = str(kwargs.get("model") or "")
        route_affects_context = "model" in kwargs or "provider" in kwargs
        for key in ("base_url", "api_key", "provider", "api_mode"):
            if key in kwargs:
                setattr(self, key, str(kwargs.get(key) or ""))
        if (
            "context_length" not in kwargs
            and route_affects_context
            and (self.raw_context_length or self.context_length)
        ):
            self._set_context_length(
                self.raw_context_length or self.context_length,
                source=self._context_length_source or "session_start",
                model=self.model,
                provider=self.provider,
            )
        self._update_model_pending_session_start = False

    def _continue_compression_boundary(
        self,
        session_id: str,
        old_session_id: str,
        kwargs: Dict[str, Any],
    ) -> None:
        previous_session_id = self._session_id
        requested_conversation_id = kwargs.get("conversation_id")
        session_state = self._lifecycle.get_by_session(old_session_id)
        conversation_state = self._lifecycle.get_by_conversation(old_session_id)

        def _state_conversation_matches(state: Any) -> bool:
            return bool(
                state
                and (
                    not requested_conversation_id
                    or state.conversation_id == requested_conversation_id
                )
            )

        def _has_summary_nodes(candidate_session_id: str | None) -> bool:
            return bool(candidate_session_id and self._dag.get_session_nodes(candidate_session_id))

        def _host_source_from_conversation_state(state: Any) -> tuple[str, Any]:
            if not _state_conversation_matches(state):
                return "", None
            if state.current_session_id == old_session_id and _has_summary_nodes(old_session_id):
                return old_session_id, state
            if (
                state.conversation_id == old_session_id
                and state.current_session_id
                and _has_summary_nodes(state.current_session_id)
            ):
                return state.current_session_id, state
            if (
                state.current_session_id is None
                and state.last_finalized_session_id
                and _has_summary_nodes(state.last_finalized_session_id)
            ):
                return state.last_finalized_session_id, state
            return "", None

        def _host_source_from_session_state(state: Any) -> tuple[str, Any]:
            if not _state_conversation_matches(state):
                return "", None
            if state.current_session_id == old_session_id and _has_summary_nodes(old_session_id):
                return old_session_id, state
            if (
                state.current_session_id is None
                and state.last_finalized_session_id == old_session_id
                and _has_summary_nodes(old_session_id)
            ):
                return old_session_id, state
            return "", None

        host_source_session_id, host_source_state = _host_source_from_conversation_state(
            conversation_state
        )
        if not host_source_session_id:
            host_source_session_id, host_source_state = _host_source_from_session_state(
                session_state
            )

        source_session_id = host_source_session_id or old_session_id
        source_state = host_source_state or session_state

        if previous_session_id and previous_session_id != old_session_id:
            # Hermes passes the session that actually crossed the compression
            # boundary as old_session_id. A different bound session can be a
            # short-lived subagent/cron/WebUI side channel that ran after the
            # foreground compaction. Prefer the host-authoritative source when
            # durable lifecycle + DAG evidence proves it belongs to LCM, then
            # fall back to the older bound-session recovery path. When the host
            # old_session_id is the durable conversation id, use that row's
            # current/finalized LCM source instead of unrelated auxiliary rows
            # where the id appears only as last_finalized_session_id.
            if host_source_session_id:
                logger.warning(
                    "LCM compression boundary using host old_session_id %s as carry-over source=%s despite bound session drift=%s",
                    old_session_id,
                    host_source_session_id,
                    previous_session_id,
                )
            else:
                bound_state = self._lifecycle.get_by_session(previous_session_id)
                bound_conversation_matches = bool(
                    bound_state
                    and (not self._conversation_id or bound_state.conversation_id == self._conversation_id)
                    and (
                        not requested_conversation_id
                        or bound_state.conversation_id == requested_conversation_id
                    )
                )
                bound_is_active_source = bool(
                    bound_state and bound_state.current_session_id == previous_session_id
                )
                bound_is_finalized_source = bool(
                    bound_state
                    and bound_state.current_session_id is None
                    and bound_state.last_finalized_session_id == previous_session_id
                )
                bound_has_summary_nodes = bool(self._dag.get_session_nodes(previous_session_id))
                if (
                    bound_conversation_matches
                    and (bound_is_active_source or bound_is_finalized_source)
                    and bound_has_summary_nodes
                ):
                    source_session_id = previous_session_id
                    source_state = bound_state
                    logger.warning(
                        "LCM compression boundary using bound session %s as carry-over source; host old_session_id=%s does not match",
                        previous_session_id,
                        old_session_id,
                    )
                else:
                    # Fallback: sibling chain with zero-DAG parent.
                    # When stale old_session_id has no DAG nodes AND the
                    # bound session belongs to a different conversation_id
                    # but shares the same last_finalized_session_id
                    # (parent) — prefer the bound session despite the
                    # conversation_id mismatch. This handles the lifecycle
                    # fork case where two sessions on the same channel
                    # received different conversation_ids.
                    bound_shares_parent_with_host = bool(
                        bound_state
                        and bound_state.last_finalized_session_id == old_session_id
                    )
                    host_has_no_dag = not bool(
                        self._dag.get_session_nodes(old_session_id)
                    )
                    if (
                        bound_shares_parent_with_host
                        and host_has_no_dag
                        and (bound_is_active_source or bound_is_finalized_source)
                        and bound_has_summary_nodes
                    ):
                        source_session_id = previous_session_id
                        source_state = bound_state
                        logger.warning(
                            "LCM compression boundary using bound session %s on sibling chain as carry-over source; host old_session_id=%s has zero DAG, parent=%s matches",
                            previous_session_id,
                            old_session_id,
                            bound_state.last_finalized_session_id,
                        )
                    else:
                        source_session_id = ""
                        source_state = None

        conversation_id = (
            (source_state.conversation_id if source_state else None)
            or kwargs.get("conversation_id")
            or self._conversation_id
            or source_session_id
            or old_session_id
            or session_id
        )
        process_local_frontier = (
            int(self._last_compacted_store_id or 0)
            if source_session_id and previous_session_id == source_session_id
            else 0
        )
        pending_reset_frontier = int(
            self._pending_reset_frontier_store_id
            if self._pending_reset_session_id
            and self._pending_reset_session_id == source_session_id
            else 0
        )
        frontier = max(
            process_local_frontier,
            int(source_state.current_frontier_store_id if source_state else 0),
            int(source_state.last_finalized_frontier_store_id if source_state else 0),
            pending_reset_frontier,
        )
        can_reassign = bool(
            source_session_id
            and session_id
            and source_session_id != session_id
        )
        boundary_placeholder_budget = {}
        boundary_placeholder_ordinals: dict[str, set[int]] = {}
        if can_reassign:
            if previous_session_id == source_session_id:
                boundary_placeholder_budget = self._active_replay_generated_placeholder_digest_budget()
                boundary_placeholder_ordinals = self._generated_placeholder_digest_ordinals_for_active_replay(
                    self._last_active_replay_messages
                )
            if not boundary_placeholder_budget:
                boundary_placeholder_budget = self._load_generated_ignored_placeholder_hash_counts(
                    self._session_scoped_hash_metadata_keys(
                        "ignored_active_replay_placeholder_hash_counts",
                        source_session_id,
                    )
                )
            if not boundary_placeholder_ordinals:
                boundary_placeholder_ordinals = self._load_generated_ignored_placeholder_hash_ordinals(
                    self._session_scoped_hash_metadata_keys(
                        "ignored_active_replay_placeholder_hash_ordinals",
                        source_session_id,
                    )
                )
            for digest, ordinals in boundary_placeholder_ordinals.items():
                boundary_placeholder_budget[digest] = max(
                    boundary_placeholder_budget.get(digest, 0),
                    len(ordinals),
                )

        if can_reassign:
            self._lifecycle.finalize_session(
                conversation_id,
                source_session_id,
                frontier_store_id=frontier,
            )
            self._copy_generated_ignore_hashes_to_session(source_session_id, session_id)
            self._write_generated_ignored_placeholder_hash_counts(
                boundary_placeholder_budget,
                self._session_scoped_hash_metadata_keys(
                    "ignored_active_replay_placeholder_hash_counts",
                    session_id,
                ),
            )
            self._write_generated_ignored_placeholder_hash_ordinals(
                boundary_placeholder_ordinals,
                self._session_scoped_hash_metadata_keys(
                    "ignored_active_replay_placeholder_hash_ordinals",
                    session_id,
                ),
            )
            # Compression rollover carries derived context forward, but raw
            # messages remain owned by the session that produced them. Moving
            # raw rows here makes session-scoped transcript recovery report the
            # old/child session as missing even though its payload was only
            # reassigned to the next compression segment.
            moved_nodes = self._dag.reassign_session_nodes(source_session_id, session_id)
            logger.debug(
                "LCM compression boundary continued %s -> %s: carried %d DAG nodes; preserved raw message ownership",
                source_session_id,
                session_id,
                moved_nodes,
            )
        elif old_session_id:
            logger.warning(
                "LCM compression boundary skipped carry-over: old_session_id=%s does not match bound session=%s",
                old_session_id,
                previous_session_id,
            )
            self._finalize_pending_reset_boundary(previous_session_id)
            self._reset_session_scoped_runtime_state()
            self._last_boundary_skip_time = time.time()
            self._apply_session_start_metadata(session_id, kwargs)
            self._bind_lifecycle_state(
                session_id,
                conversation_id=kwargs.get("conversation_id"),
            )
            self._schedule_ingest_cursor_reconciliation()
            self._clear_pending_reset_boundary()
            return

        self._apply_session_start_metadata(session_id, kwargs)
        self._bind_lifecycle_state(session_id, conversation_id=conversation_id)
        if frontier > 0:
            state = self._lifecycle.advance_frontier(
                self._conversation_id,
                session_id,
                frontier,
            )
            if state is not None:
                self._last_compacted_store_id = state.current_frontier_store_id
        self._clear_pending_reset_boundary()

    def on_session_start(self, session_id: str, **kwargs) -> None:
        if "hermes_home" in kwargs:
            self._rebind_storage_for_home(str(kwargs.get("hermes_home") or ""))

        boundary_reason = str(kwargs.get("boundary_reason") or "")
        old_session_id = str(kwargs.get("old_session_id") or "")
        if boundary_reason == "compression":
            # The host committed a compaction: the record line (#20, #29 W2 step 7).
            self._record_confirmation(old_session_id, session_id)
        previous_session_id = self._session_id
        previous_conversation_id = self._conversation_id
        requested_conversation_id = str(kwargs.get("conversation_id") or "")
        # Every session is an ordinary session: a delegate, a background fork or
        # a cron agent is bound and compacted like any other.
        if boundary_reason == "compression":
            # A compaction boundary is not a beginning: the plugin session continues
            # under the host's new identifier, and the record line is the
            # confirmation above. Nothing is carried over; the store holds it.
            self._apply_session_start_metadata(session_id, kwargs)
            self._conversation_id = requested_conversation_id or previous_conversation_id or session_id
            self._register_active_engine_binding()
            return

        if previous_session_id and previous_session_id != session_id:
            self._reset_session_scoped_runtime_state()
        else:
            if (
                previous_conversation_id
                and (requested_conversation_id or session_id) != previous_conversation_id
            ):
                self._reset_session_counters()
            self._last_overflow_recovery_failed = False
        self._apply_session_start_metadata(session_id, kwargs)
        self._conversation_id = requested_conversation_id or session_id
        self._register_active_engine_binding()
        self._name_plugin_session(
            session_id,
            signal="on_session_start",
            platform=str(kwargs.get("platform") or "") or None,
        )

    def _name_plugin_session(self, host_session_id: str, *, signal: str, platform: str | None) -> None:
        """Bind this engine copy to the plugin session a host session id names.

        An id the store has not seen names a new plugin session; a known id
        continues its session (ruling 1). Only this copy's own calls reach here.
        """
        if not host_session_id:
            return
        handle, _created = self._sessions.name_session(host_session_id, signal=signal)
        self._plugin_session = handle
        if platform:
            self._sessions.note_platform(handle, platform, signal=signal,
                                         host_session_id=host_session_id)

    def bind_session_state(self, session_db: Any = None, session_id: str = "") -> None:
        """The host's binding of this engine copy to a session id.

        The host calls it for every agent it builds, before ``on_session_start``, and
        after a switch that only resets (CLI ``/new``, ``/resume``, ``/branch``; ACP
        ``/reset``), which is how this copy learns the new id there. The plugin uses
        only the id, never the host's database. An empty id is the host detaching the
        copy of its background-review fork; it is recorded on this copy's own session
        and refused (``ReviewForkDetachRefused``).
        """
        if session_id:
            self._name_plugin_session(session_id, signal="bind_session_state", platform=None)
            return
        if self._plugin_session:
            self._sessions.add_fact(
                self._plugin_session,
                "detach",
                "refused",
                signal="bind_session_state",
            )
        message = (
            "LCM refuses bind_session_state(session_id=''), the host's detach of a "
            "background-review fork: the host neither commits nor confirms a fork's "
            "compaction and names no parent for it, so the plugin cannot record it "
            "without guessing. The host keeps this fork's compaction disabled. "
            "See issue #20 and ask A-32.4 (a session id of its own for the fork)."
        )
        logger.warning(message)
        raise ReviewForkDetachRefused(message)

    def on_session_end(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        """Not a session boundary, and nothing is written here.

        The host calls this inside every compaction it commits (with the list as
        it was before), and at the real ends of a session late or not at all
        (#20). The plugin draws no boundary from it and flushes nothing.
        """
        return None

    def on_session_reset(self) -> None:
        """/new or /reset: the next session is a new one; nothing is carried over."""
        super().on_session_reset()
        self._reset_session_scoped_runtime_state()

    def carry_over_new_session_context(self, old_session_id: str, new_session_id: str) -> int:
        """Nothing is carried into a new session: /new is new, and what a session
        holds stays in the store under its own session. Returns the number moved, 0."""
        return 0


    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            LCM_GREP,
            LCM_EXPAND,
            LCM_EXPAND_QUERY,
            LCM_STATUS,
            LCM_INSPECT,
            LCM_DOCTOR,
        ]

    def handle_tool_call(self, name: str, args: Dict[str, Any], **kwargs) -> str:
        # The tools read the store, which is filled at compaction; what the agent's
        # context holds now is in its context.
        handlers = {
            "lcm_grep": lcm_tools.lcm_grep,
            "lcm_expand": lcm_tools.lcm_expand,
            "lcm_expand_query": lcm_tools.lcm_expand_query,
            "lcm_status": lcm_tools.lcm_status,
            "lcm_inspect": lcm_tools.lcm_inspect,
            "lcm_doctor": lcm_tools.lcm_doctor,
        }
        handler = handlers.get(name)
        if handler:
            return handler(args, engine=self)
        return json.dumps({"error": f"Unknown LCM tool: {name}"})

    def _database_path_source(self) -> str:
        if self._config.database_path:
            return "config.database_path"
        if self._hermes_home:
            return "hermes_home"
        return "default_home"

    def get_runtime_identity(self) -> Dict[str, Any]:
        """Return operator-facing identity for the loaded LCM runtime."""
        metadata = _plugin_metadata()
        git_identity = _git_runtime_identity(_PLUGIN_ROOT)
        session_id = self.current_session_id
        conversation_id = self.current_conversation_id
        identity: Dict[str, Any] = {
            "engine": self.name,
            "plugin_name": metadata.get("name", "hermes-lcm"),
            "plugin_version": metadata.get("version", "unknown"),
            "plugin_path": str(_PLUGIN_ROOT),
            "module_path": str(Path(__file__).resolve()),
            "hermes_home": str(self._hermes_home or ""),
            "database_path": str(self._store.db_path),
            "database_path_source": self._database_path_source(),
            "session_id": session_id,
            "host_session_id": self._session_id,
            "session_platform": self.current_session_platform,
            "session_bound": bool(session_id),
            "conversation_id": conversation_id,
        }
        identity.update(git_identity)
        return identity

    def get_status(self) -> Dict[str, Any]:
        status = super().get_status()
        status.update({
            "compression_count": self.compression_count,
            "last_prompt_tokens": self.last_prompt_tokens,
            "last_completion_tokens": self.last_completion_tokens,
            "last_total_tokens": self.last_total_tokens,
            "last_input_tokens": self.last_input_tokens,
            "last_output_tokens": self.last_output_tokens,
            "last_cache_read_tokens": self.last_cache_read_tokens,
            "last_cache_write_tokens": self.last_cache_write_tokens,
            "last_reasoning_tokens": self.last_reasoning_tokens,
            "cache_metrics_available": self.cache_metrics_available,
            "cache_read_ratio": round(self.cache_read_ratio, 4),
            "raw_context_length": self.raw_context_length,
            "context_length": self.context_length,
            "effective_context_length_cap": self.effective_context_length_cap,
            "effective_context_length_reason": self.effective_context_length_reason,
            "threshold_tokens": self.threshold_tokens,
            "last_compression_status": self._last_compression_status,
            "last_compression_noop_reason": self._last_compression_noop_reason,
            "last_compress_aborted": self._last_compress_aborted,
            "last_summary_error": self._last_summary_error,
            "threshold_full_sweep": dict(self._last_threshold_full_sweep),
            "ingest_failure_count": self._ingest_failure_count,
            "consecutive_ingest_failures": self._consecutive_ingest_failures,
            "last_ingest_error": self._last_ingest_error,
            "last_ingest_error_time": self._last_ingest_error_time,
            "model": self.model,
            "provider": self.provider,
            "context_length_source": self._context_length_source,
            "configured_context_threshold": self._config.context_threshold,
            "context_threshold": self.context_threshold,
            "context_threshold_source": self._context_threshold_source,
            "context_threshold_autoraised": self._context_threshold_autoraised,
            "config_sources": dict(getattr(self._config, "config_sources", {}) or {}),
            "config_source_warnings": list(getattr(self._config, "config_source_warnings", []) or []),
            "ignored_config_yaml_lcm_keys": list(getattr(self._config, "ignored_config_yaml_lcm_keys", []) or []),
        })
        session_id = self.current_session_id
        conversation_id = self.current_conversation_id
        # The compactions of this plugin session that took effect, from the store.
        try:
            status["total_compactions"] = self._records.effective_count(session_id) if session_id else 0
        except Exception as exc:  # pragma: no cover - defensive
            status["total_compactions"] = f"error: {exc}"
        status["total_compactions_scope"] = _TOTAL_COMPACTIONS_SCOPE
        status["engine"] = "lcm"
        status["runtime_identity"] = self.get_runtime_identity()
        try:
            status["source_lineage"] = self._store.get_source_stats(session_id or None)
        except Exception as exc:  # pragma: no cover - defensive
            status["source_lineage"] = {"error": str(exc)}
        if session_id:
            status["store_messages"] = self._store.get_session_count(session_id)
            status["dag_nodes"] = self._dag.get_session_node_count(session_id)
            status["session_platform"] = self.current_session_platform
            status["ingest_reconciliation"] = dict(self._last_ingest_reconciliation)
            status["overflow_recovery_failed"] = self._last_overflow_recovery_failed
            status["condensation_suppressed_reason"] = self._last_condensation_suppressed_reason
            status["conversation_id"] = conversation_id
        return status

    def update_model(self, model: str, context_length: int,
                     base_url: str = "", api_key: str = "",
                     provider: str = "",
                     api_mode: str = "") -> None:
        self.model = str(model or "")
        self.base_url = str(base_url or "")
        self.api_key = str(api_key or "")
        self.provider = str(provider or "")
        self.api_mode = str(api_mode or "")
        self._set_context_length(context_length, source="update_model")
        self._update_model_pending_session_start = True

    # -- Internal: message ingestion ---------------------------------------

    def _schedule_ingest_cursor_reconciliation(self) -> None:
        """Mark existing-session rebinds for cursor repair on next ingest."""
        self._ingest_cursor_needs_reconcile = False
        if not self._session_id:
            return
        try:
            self._ingest_cursor_needs_reconcile = self._store.get_session_count(self._session_id) > 0
        except Exception as exc:  # pragma: no cover - defensive only
            logger.debug("LCM ingest cursor reconciliation probe failed: %s", exc)
            self._ingest_cursor_needs_reconcile = False


    def _content_has_externalized_placeholder_ref(self, content: str) -> bool:
        return bool(extract_ingest_externalized_refs(content))


    @staticmethod
    def _copy_active_replay_messages(
        active_replay_messages: List[Dict[str, Any]],
    ) -> list[Dict[str, Any]]:
        return [dict(message) for message in active_replay_messages]

    def _remember_active_replay_messages(
        self,
        original_messages: List[Dict[str, Any]],
        active_replay_messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        self._last_active_replay_source_identities = [
            self._message_replay_identity(message) for message in original_messages
        ]
        self._last_active_replay_messages = self._copy_active_replay_messages(
            active_replay_messages
        )
        self._write_generated_ignored_placeholder_hash_counts(
            self._generated_placeholder_digest_budget_for_active_replay(active_replay_messages)
        )
        self._write_generated_ignored_placeholder_hash_ordinals(
            self._generated_placeholder_digest_ordinals_for_active_replay(active_replay_messages)
        )
        return active_replay_messages

    def _cached_active_replay_messages(
        self,
        original_messages: List[Dict[str, Any]],
    ) -> Optional[List[Dict[str, Any]]]:
        identities = [self._message_replay_identity(message) for message in original_messages]
        if identities == getattr(self, "_last_active_replay_source_identities", None):
            cached = getattr(self, "_last_active_replay_messages", None)
            if cached is not None:
                return self._copy_active_replay_messages(cached)
        return None

    def _is_replayed_context_scaffold_message(self, msg: Dict[str, Any]) -> bool:
        """Return true for active-context scaffolding that should not be re-ingested."""
        role = str(msg.get("role") or "")
        content = normalize_content_value(msg.get("content")) or ""
        if role == "system":
            return (
                "[Note: This conversation uses Lossless Context Management (LCM)." in content
                and "Earlier turns have been compacted into hierarchical summaries below." in content
            )
        if content.lstrip().startswith(_PRESERVED_OBJECTIVE_CONTEXT_PREFIX):
            return True
        if "[Expand for details:" not in content:
            return False
        return bool(
            re.search(
                r"\[(?:Recent|Session Arc|Durable|Depth-\d+) Summary \(d\d+, node \d+\)\]",
                content,
            )
        )

    def _restore_ingest_payload_placeholders_in_value(self, value: Any, *, session_id: str) -> Any:
        if isinstance(value, dict):
            return {
                self._restore_ingest_payload_placeholders_in_value(key, session_id=session_id)
                if isinstance(key, str)
                else key: self._restore_ingest_payload_placeholders_in_value(val, session_id=session_id)
                for key, val in value.items()
            }
        if isinstance(value, list):
            return [self._restore_ingest_payload_placeholders_in_value(item, session_id=session_id) for item in value]
        if isinstance(value, str):
            return restore_ingest_payload_placeholders(
                value,
                config=self._config,
                hermes_home=self._hermes_home,
                session_id=session_id,
            )
        return value

    def _restore_ingest_payload_placeholders_in_content_identity(self, content: str, *, session_id: str) -> str:
        if not content:
            return content
        try:
            decoded = json.loads(content)
        except (TypeError, ValueError, json.JSONDecodeError):
            return restore_ingest_payload_placeholders(
                content,
                config=self._config,
                hermes_home=self._hermes_home,
                session_id=session_id,
            )
        restore_as_structured = False
        if isinstance(decoded, (dict, list)) and normalize_content_value(decoded) == content:
            for ref in extract_ingest_externalized_refs(content):
                payload = load_externalized_payload(
                    ref,
                    config=self._config,
                    hermes_home=self._hermes_home,
                )
                payload_session_id = (payload or {}).get("session_id") or ""
                if session_id and payload_session_id and payload_session_id != session_id:
                    continue
                field_path = str((payload or {}).get("field_path") or "")
                if field_path and field_path != "content":
                    restore_as_structured = True
                    break
        if restore_as_structured:
            restored = self._restore_ingest_payload_placeholders_in_value(decoded, session_id=session_id)
            return normalize_content_value(restored) or ""
        return restore_ingest_payload_placeholders(
            content,
            config=self._config,
            hermes_home=self._hermes_home,
            session_id=session_id,
        )


    def _ingest_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Persist new messages to the store.

        Uses a cursor to track which portion of the current messages list
        has already been persisted.  After compress() shortens the list,
        the cursor is reset to len(compressed), so only messages appended
        after compaction are ingested — regardless of how the store count
        compares to the current list length.

        Returns a replay-safe copy of ``messages`` with obviously broken
        assistant loops replaced by quarantine placeholders. Existing callers may
        ignore the return value when they only need durable persistence.
        """
        if not self._session_id:
            logger.debug("Ingest skipped: no session_id")
            return self._copy_active_replay_messages(messages)

        n = len(messages)
        cursor = min(max(self._ingest_cursor, 0), n)
        scan_start = 0 if self._ingest_cursor_needs_reconcile else cursor
        externalize_messages = [idx >= scan_start for idx in range(n)]
        prefer_existing_externalized = [idx < scan_start for idx in range(n)]
        replay_messages = quarantine_suspicious_assistant_messages(
            messages,
            session_id=self._session_id,
            config=self._config,
            hermes_home=self._hermes_home,
            externalize=externalize_messages,
            prefer_existing_externalized=prefer_existing_externalized,
        )
        replay_messages = self._copy_active_replay_messages(replay_messages)
        if self._ingest_cursor_needs_reconcile:
            self._ingest_cursor = self._reconcile_ingest_cursor_from_store(replay_messages)
            self._ingest_cursor_needs_reconcile = False
        cursor = min(max(self._ingest_cursor, 0), n)
        if cursor > 0:
            cached_source_identities = getattr(self, "_last_active_replay_source_identities", None)
            cached_active_replay_messages = getattr(self, "_last_active_replay_messages", None)
            if (
                cached_source_identities is not None
                and cached_active_replay_messages is not None
                and len(cached_source_identities) >= cursor
                and len(cached_active_replay_messages) >= cursor
            ):
                current_prefix_identities = [
                    self._message_replay_identity(message) for message in messages[:cursor]
                ]
                if current_prefix_identities == cached_source_identities[:cursor]:
                    replay_messages = (
                        self._copy_active_replay_messages(
                            cached_active_replay_messages[:cursor]
                        )
                        + replay_messages[cursor:]
                    )
        logger.debug(
            "Ingest: session=%s cursor=%d incoming=%d",
            self._session_id, cursor, n,
        )

        new_messages = replay_messages[cursor:] if cursor < n else []
        original_new_messages = messages[cursor:] if cursor < n else []

        if not new_messages:
            cached_replay = self._cached_active_replay_messages(messages)
            if cached_replay is not None:
                return cached_replay
            return self._remember_active_replay_messages(messages, replay_messages)

        messages_to_store_with_index: list[tuple[int, Dict[str, Any]]] = []
        for offset, (original_msg, replay_msg) in enumerate(zip(original_new_messages, new_messages)):
            absolute_idx = cursor + offset
            replay_text = text_content_for_pattern_matching(replay_msg.get("content")) or ""
            original_text = text_content_for_pattern_matching(original_msg.get("content")) or ""
            volatile_digest = self._active_replay_placeholder_digest(replay_text)
            generated_volatile_placeholder = self._is_volatile_ignored_quarantine_placeholder(
                replay_msg,
                replay_text,
            ) and (
                original_text != replay_text
                or (
                    volatile_digest is not None
                    and volatile_digest in self._load_generated_ignored_placeholder_hashes()
                )
            )
            if generated_volatile_placeholder:
                if volatile_digest is not None:
                    self._remember_generated_ignored_placeholder_hash(volatile_digest)
                logger.debug(
                    "LCM did not store a generated quarantine placeholder for %s message: %r",
                    original_msg.get("role", "unknown"),
                    original_text[:80].replace("\n", " "),
                )
                continue
            store_msg = replay_msg
            if (
                str(original_msg.get("role") or "") == "tool"
                and _is_hermes_persisted_output_marker(
                    normalize_content_value(original_msg.get("content")) or ""
                )
            ):
                store_msg = original_msg
            messages_to_store_with_index.append((absolute_idx, store_msg))

        if not messages_to_store_with_index:
            self._ingest_cursor = n
            return self._remember_active_replay_messages(messages, replay_messages)

        protected_messages = protect_messages_for_ingest(
            [msg for _idx, msg in messages_to_store_with_index],
            session_id=self._session_id,
            config=self._config,
            hermes_home=self._hermes_home,
        )
        estimates = [count_message_tokens(m) for m in protected_messages]
        self._store._append_protected_batch(
            self._session_id,
            protected_messages,
            estimates,
            source=self._session_platform,
            conversation_id=self._conversation_id,
        )
        self._ingest_cursor = n
        logger.debug("Ingested %d messages into LCM store", len(messages_to_store_with_index))
        # ``protected_messages`` changes are storage-only: inline media and
        # data/base64 substrings stay provider-usable in active replay.
        return self._remember_active_replay_messages(messages, replay_messages)


    def _get_store_ids_for_messages(self, messages: List[Dict[str, Any]]) -> List[int]:
        ids_by_message_id = self._get_store_id_map_for_messages(messages)
        return [ids_by_message_id[id(msg)] for msg in messages if id(msg) in ids_by_message_id]

    # -- Internal: summarization -------------------------------------------


    def _serialize_messages(self, messages: List[Dict[str, Any]]) -> str:
        """Serialize messages into labeled text for the summarizer."""
        parts = []
        matched_tool_ids = _matched_tool_call_ids(messages)
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content") or ""
            if role == "tool":
                tool_id = str(msg.get("tool_call_id") or "").strip()
                content = sanitize_pre_compaction_content(content)
                if len(content) > 3000:
                    content = content[:2000] + "\n...[truncated]...\n" + content[-800:]
                parts.append(f"[TOOL RESULT {tool_id}]: {content}")
                continue

            content = sanitize_pre_compaction_content(content)

            if role == "assistant":
                tool_calls = msg.get("tool_calls", [])
                matched_tool_calls = [
                    tc for tc in tool_calls
                    if not _tool_call_id(tc) or _tool_call_id(tc) in matched_tool_ids
                ]
                if _is_synthetic_assistant_noise(content):
                    if not matched_tool_calls:
                        continue
                    content = ""
                if len(content) > 3000:
                    content = content[:2000] + "\n...[truncated]...\n" + content[-800:]
                if matched_tool_calls:
                    tc_parts = []
                    for tc in matched_tool_calls:
                        if isinstance(tc, dict):
                            fn = tc.get("function", {})
                            name = fn.get("name", "?")
                            args = fn.get("arguments", "")
                            args = sanitize_pre_compaction_tool_arguments(args)
                            if len(args) > 500:
                                args = args[:400] + "..."
                            tc_parts.append(f"  {name}({args})")
                    content += "\n[Tool calls:\n" + "\n".join(tc_parts) + "\n]"
                parts.append(f"[ASSISTANT]: {content}")
                continue

            if len(content) > 3000:
                content = content[:2000] + "\n...[truncated]...\n" + content[-800:]
            parts.append(f"[{role.upper()}]: {content}")

        return "\n\n".join(parts)

    # -- Internal: tool-pair sanitization ------------------------------------

    def _sanitize_active_context_messages(
        self,
        messages: List[Dict[str, Any]],
        *,
        insert_missing_tool_stubs: bool = True,
    ) -> List[Dict[str, Any]]:
        """Drop unsafe assistant-only noise, then repair tool sequencing.

        This is intentionally active-context-only: callers pass the selected
        provider replay context, and this helper never mutates stored rows,
        source mappings, or DAG nodes.
        """
        cleaned: list[Dict[str, Any]] = []
        dropped_assistant_messages = 0
        stripped_assistant_messages = 0
        for msg in messages:
            msg = self._sanitize_active_preserved_objective_message(msg)
            if msg.get("role") == "assistant":
                cleaned_msg = _clean_active_assistant_message(msg)
                if cleaned_msg is None:
                    dropped_assistant_messages += 1
                    continue
                if cleaned_msg is not msg:
                    stripped_assistant_messages += 1
                cleaned.append(cleaned_msg)
                continue
            cleaned.append(msg)

        if dropped_assistant_messages:
            logger.info(
                "LCM active-context cleanup: dropped %d assistant message(s) with no visible content",
                dropped_assistant_messages,
            )
        if stripped_assistant_messages:
            logger.info(
                "LCM active-context cleanup: stripped internal content from %d assistant message(s)",
                stripped_assistant_messages,
            )

        return self._sanitize_tool_pairs(
            cleaned,
            insert_missing_tool_stubs=insert_missing_tool_stubs,
        )


    def _sanitize_tool_pairs(
        self,
        messages: List[Dict[str, Any]],
        *,
        insert_missing_tool_stubs: bool = True,
    ) -> List[Dict[str, Any]]:
        """Return provider-safe active-context tool-call/result sequencing.

        Raw store and DAG history remain lossless. This guardrail only sanitizes
        the active context emitted back to providers, where assistant tool calls
        must be followed immediately by their contiguous tool results. Late,
        duplicate, out-of-order, and orphan tool results are dropped; missing
        direct results get synthetic stubs.
        """
        sanitized: List[Dict[str, Any]] = []
        dropped_tool_results = 0
        inserted_stub_results = 0

        i = 0
        while i < len(messages):
            msg = messages[i]

            if msg.get("role") == "tool":
                dropped_tool_results += 1
                i += 1
                continue

            sanitized.append(msg)

            if msg.get("role") == "assistant":
                expected_ids = [
                    call_id
                    for call_id in (_tool_call_id(tool_call) for tool_call in (msg.get("tool_calls") or []))
                    if call_id
                ]

                for expected_id in expected_ids:
                    matched_direct_result = False
                    while i + 1 < len(messages) and messages[i + 1].get("role") == "tool":
                        next_msg = messages[i + 1]
                        next_id = str(next_msg.get("tool_call_id") or "").strip()
                        if next_id == expected_id:
                            sanitized.append(next_msg)
                            i += 1
                            matched_direct_result = True
                            break
                        dropped_tool_results += 1
                        i += 1

                    if not matched_direct_result and insert_missing_tool_stubs:
                        sanitized.append({
                            "role": "tool",
                            "content": "[Result from earlier conversation — see context summary above]",
                            "tool_call_id": expected_id,
                        })
                        inserted_stub_results += 1

                while i + 1 < len(messages) and messages[i + 1].get("role") == "tool":
                    dropped_tool_results += 1
                    i += 1

            i += 1

        if dropped_tool_results:
            logger.info(
                "LCM tool-pair guardrail: dropped %d late/orphan/duplicate tool result(s)",
                dropped_tool_results,
            )
        if inserted_stub_results:
            logger.info(
                "LCM tool-pair guardrail: inserted %d missing tool-result stub(s)",
                inserted_stub_results,
            )

        return sanitized

    # -- Internal: condensation --------------------------------------------

    def _should_allow_follow_on_condensation(
        self,
        *,
        uncondensed_count: int,
        leaf_compacted_this_turn: bool,
        force_overflow: bool,
    ) -> tuple[bool, str]:
        if not leaf_compacted_this_turn:
            return True, ""
        if not self._config.cache_friendly_condensation_enabled:
            return True, ""
        if force_overflow:
            return True, ""

        fanin = max(1, self._config.condensation_fanin)
        debt_threshold = fanin * max(1, self._config.cache_friendly_min_debt_groups)
        if uncondensed_count >= debt_threshold:
            return True, ""
        if uncondensed_count == fanin:
            return False, "cache_friendly_single_group"
        return False, "cache_friendly_low_debt"

    def _maybe_condense(
        self,
        focus_topic: Optional[str] = None,
        *,
        leaf_compacted_this_turn: bool = False,
        force_overflow: bool = False,
    ) -> None:
        """Check if any depth level has enough nodes for condensation."""
        self._last_condensation_suppressed_reason = ""

        max_depth = self._config.incremental_max_depth
        if max_depth == 0:
            return  # condensation disabled

        # When max_depth is -1 (unlimited), derive the upper bound from
        # the deepest existing node + 1, so condensation can always
        # create the next depth level.
        if max_depth < 0:
            all_nodes = self._dag.get_session_nodes(self._session_id)
            upper = (max(n.depth for n in all_nodes) + 1) if all_nodes else 1
        else:
            upper = max_depth

        condensed_any = False
        suppression_reason = ""
        fanin = max(1, self._config.condensation_fanin)

        for depth in range(upper):
            uncondensed = self._dag.get_uncondensed_at_depth(
                self._session_id, depth
            )
            if len(uncondensed) < fanin:
                continue

            allow_condense, reason = self._should_allow_follow_on_condensation(
                uncondensed_count=len(uncondensed),
                leaf_compacted_this_turn=leaf_compacted_this_turn,
                force_overflow=force_overflow,
            )
            if not allow_condense:
                suppression_reason = reason or suppression_reason
                continue

            # Take the first fanin nodes and condense
            to_condense = uncondensed[:fanin]
            source_tokens, summary_tokens, level = self._condense_summary_nodes(
                to_condense,
                focus_topic=focus_topic,
            )
            condensed_any = True

            logger.info(
                "LCM condensation: d%d × %d → d%d (L%d, %d→%d tokens)",
                depth, len(to_condense), depth + 1, level,
                source_tokens, summary_tokens,
            )

            if leaf_compacted_this_turn and self._config.cache_friendly_condensation_enabled:
                break

        if not condensed_any and leaf_compacted_this_turn and self._config.cache_friendly_condensation_enabled:
            self._last_condensation_suppressed_reason = suppression_reason

    def _condense_summary_nodes(
        self,
        nodes: List[SummaryNode],
        *,
        focus_topic: Optional[str] = None,
        deadline: Optional[float] = None,
    ) -> tuple[int, int, int]:
        """Persist one same-depth condensation and return source/output tokens and level."""
        if not nodes:
            raise ValueError("condensation requires at least one summary node")
        depth = nodes[0].depth
        if any(node.depth != depth for node in nodes):
            raise ValueError("condensation requires same-depth summary nodes")
        self._require_live_write()
        combined_text = "\n\n---\n\n".join(node.summary for node in nodes)
        source_tokens = sum(node.token_count for node in nodes)
        token_budget = max(1000, int(source_tokens * 0.40))
        timeout_seconds = self._config.summary_timeout_ms / 1000
        if deadline is not None:
            remaining_seconds = deadline - time.monotonic()
            if remaining_seconds <= 0:
                raise TimeoutError("threshold full sweep time budget exhausted")
            timeout_seconds = min(timeout_seconds, remaining_seconds)
        summary_text, level = summarize_with_escalation(
            text=combined_text,
            source_tokens=source_tokens,
            token_budget=token_budget,
            depth=depth + 1,
            model=self._config.summary_model,
            fallback_models=self._config.summary_fallback_models,
            circuit_breaker=self._summary_circuit_breaker,
            spend_guard=self._summary_spend_guard,
            timeout=timeout_seconds,
            l2_budget_ratio=self._config.l2_budget_ratio,
            l3_truncate_tokens=self._config.l3_truncate_tokens,
            focus_topic=focus_topic or "",
            custom_instructions=self._config.custom_instructions,
            source_provenance={
                "source_type": "summary_nodes",
                "node_ids": [node.node_id for node in nodes],
                "source_depth": depth,
            },
        )
        earliest_at, latest_at = self._dag.get_source_time_window(
            [node.node_id for node in nodes]
        )
        summary_tokens = count_tokens(summary_text)
        condensed_node = SummaryNode(
            session_id=self._session_id,
            depth=depth + 1,
            summary=summary_text,
            token_count=summary_tokens,
            source_token_count=source_tokens,
            source_ids=[node.node_id for node in nodes],
            source_type="nodes",
            created_at=time.time(),
            earliest_at=earliest_at,
            latest_at=latest_at,
            expand_hint=self._extract_expand_hint(summary_text),
        )
        self._require_live_write()
        self._dag.add_node(condensed_node)
        return source_tokens, summary_tokens, level

    def _summary_frontier_nodes(self) -> List[SummaryNode]:
        """Return all provider-visible summary frontier nodes for the active session."""
        all_nodes = self._dag.get_session_nodes(self._session_id, limit=100_000)
        referenced = {
            source_id
            for node in all_nodes
            if node.source_type == "nodes"
            for source_id in node.source_ids
        }
        return [node for node in all_nodes if node.node_id not in referenced]

    def _summary_frontier_tokens(self) -> int:
        return sum(node.token_count for node in self._summary_frontier_nodes())

    def _select_threshold_sweep_condensation_group(self) -> List[SummaryNode]:
        """Prefer routine fanin/depth, then allow bounded pressure condensation."""
        by_depth: dict[int, list[SummaryNode]] = {}
        for node in self._summary_frontier_nodes():
            by_depth.setdefault(node.depth, []).append(node)
        if not by_depth:
            return []
        fanin = max(2, self._config.condensation_fanin)
        preferred_max_depth = self._config.incremental_max_depth
        for depth in sorted(by_depth):
            nodes = by_depth[depth]
            within_preferred_depth = preferred_max_depth < 0 or depth < preferred_max_depth
            if within_preferred_depth and len(nodes) >= fanin:
                return nodes[:fanin]
        # The frontier still exceeds its sweep target but no routine group is
        # available. Permit a same-depth partial group or depth beyond the
        # preferred routine maximum; the outer sweep budget keeps this bounded.
        for depth in sorted(by_depth):
            nodes = by_depth[depth]
            if len(nodes) >= 2:
                return nodes[: min(fanin, len(nodes))]
        return []

    def _run_threshold_sweep_condensation(
        self,
        *,
        target_tokens: int,
        pass_budget: int,
        deadline: float,
        focus_topic: Optional[str] = None,
    ) -> tuple[int, str]:
        """Condense an oversized summary frontier within the remaining sweep budget."""
        passes = 0
        while self._summary_frontier_tokens() > target_tokens:
            if passes >= pass_budget:
                return passes, "pass_budget_exhausted"
            if time.monotonic() >= deadline:
                return passes, "time_budget_exhausted"
            group = self._select_threshold_sweep_condensation_group()
            if not group:
                return passes, "no_same_depth_condensation_group"
            before = self._summary_frontier_tokens()
            try:
                self._condense_summary_nodes(
                    group,
                    focus_topic=focus_topic,
                    deadline=deadline,
                )
            except Exception as exc:
                logger.warning(
                    "LCM threshold full sweep condensation stopped after %d pass(es): %s",
                    passes,
                    exc,
                )
                return passes, "condensation_error"
            passes += 1
            after = self._summary_frontier_tokens()
            if after >= before:
                return passes, "condensation_no_progress"
        return passes, "summary_prefix_target_reached"

    # -- Internal: context assembly ----------------------------------------

    @staticmethod
    def _append_lcm_note_to_content(content: Any) -> Any:
        note = (
            "\n\n[Note: This conversation uses Lossless Context Management (LCM). "
            "Earlier turns have been compacted into hierarchical summaries below. "
            "Use lcm_grep to search history "
            "and lcm_expand to recover original details from any summary.]"
        )
        if isinstance(content, str):
            return content + note
        note_part = {"type": "text", "text": note.lstrip()}
        if content is None:
            return note.lstrip()
        if isinstance(content, list):
            return list(content) + [note_part]
        normalized = normalize_content_value(content) or ""
        return normalized + note

    @staticmethod
    def _is_preserved_todo_context_message(message: Dict[str, Any]) -> bool:
        content = text_content_for_pattern_matching(message.get("content")) or ""
        return content.lstrip().startswith(_PRESERVED_TODO_CONTEXT_PREFIX)

    @staticmethod
    def _preserved_objective_context_content(message: Dict[str, Any]) -> str:
        content = text_content_for_pattern_matching(message.get("content")) or ""
        return content if content.lstrip().startswith(_PRESERVED_OBJECTIVE_CONTEXT_PREFIX) else ""

    def _sanitized_preserved_objective_context_content(self, message: Dict[str, Any]) -> str:
        preserved_objective = self._preserved_objective_context_content(message)
        if not preserved_objective:
            return ""
        return self._sanitize_preserved_objective_content(
            preserved_objective,
            role=str(message.get("role") or "user"),
        )

    def _sanitize_active_preserved_objective_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        sanitized_content = self._sanitized_preserved_objective_context_content(message)
        if not sanitized_content or sanitized_content == message.get("content"):
            return message
        sanitized = dict(message)
        sanitized["content"] = sanitized_content
        return sanitized

    def _sanitize_preserved_objective_content(self, content: str, role: str = "user") -> str:
        content = strip_injected_context_blocks(content)
        content = protect_inline_payloads_in_text(
            content,
            role=role,
            session_id=self._session_id,
            field_path="preserved_objective.content",
            config=self._config,
            hermes_home=self._hermes_home,
        )
        return content

    def _build_preserved_objective_summary_part(self, message: Dict[str, Any]) -> str:
        content = text_content_for_pattern_matching(message.get("content")) or ""
        content = self._sanitize_preserved_objective_content(
            content,
            role=str(message.get("role") or "user"),
        )
        return f"{_PRESERVED_OBJECTIVE_CONTEXT_PREFIX}\n{content}"

    def _latest_user_context_anchor(
        self,
        messages: List[Dict[str, Any]],
        selected_tail: List[Dict[str, Any]],
    ) -> Optional[str]:
        """Return a scaffolded newest real user objective omitted from the tail.

        Tool-heavy turns can push the operative user request outside the fresh
        tail while retaining only assistant/tool traces from that turn.  The
        returned text is active-context scaffolding, not raw conversation: it is
        emitted inside the summary block so restart reconciliation ignores it
        instead of ingesting a duplicate non-contiguous user message.

        Previous preserved-objective scaffolds are derived context, not real
        user turns, so they are not eligible as the next anchor source. Once a
        reverse scan reaches one, older user turns are stale relative to that
        synthetic continuity marker and must not be promoted as current intent.
        """
        selected_tail_messages = [msg for msg in selected_tail if isinstance(msg, dict)]
        for message in reversed(messages):
            if not isinstance(message, dict):
                continue
            content_text = text_content_for_pattern_matching(message.get("content")) or ""
            if self._is_volatile_ignored_quarantine_placeholder(message, content_text):
                continue
            if self._preserved_objective_context_content(message):
                return None
            if message.get("role") != "user":
                continue
            if self._is_preserved_todo_context_message(message):
                continue
            if any(message == selected for selected in selected_tail_messages):
                return None
            return self._build_preserved_objective_summary_part(message)
        return None


    def _assemble_context(
        self,
        system_msg: Optional[Dict[str, Any]],
        tail_messages: List[Dict[str, Any]],
        assembly_cap_override: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Build the active context from DAG summaries + fresh tail.

        Structure:
          [leading anchor, normally system prompt]
          [highest-depth summary nodes first, then lower]
          [fresh tail messages]
        """
        result = []

        # Leading anchor with optional LCM annotation. Only a true system prompt
        # is a safe permanent anchor; gateway sessions can start directly with
        # user messages, and those user turns must remain compactable.
        leading_msg = system_msg.copy() if system_msg is not None else None
        if leading_msg is not None:
            if (
                leading_msg.get("role") == "system"
                and self.compression_count == 0
            ):
                leading_msg["content"] = self._append_lcm_note_to_content(
                    leading_msg.get("content", "")
                )
            result.append(leading_msg)

        assembly_cap = (
            assembly_cap_override
            if assembly_cap_override is not None
            else self._effective_assembly_token_cap()
        )

        tail_selected = tail_messages
        anchor_source = getattr(self, "_pending_context_anchor_messages", None)
        if anchor_source is None:
            anchor_source = tail_messages
        anchor_part: Optional[str] = None
        summary_budget = None
        if assembly_cap is not None:
            used = count_message_tokens(leading_msg) if leading_msg is not None else 0
            kept_tail_reversed: list[Dict[str, Any]] = []
            tail_token_total = 0
            tail_for_selection = self._sanitize_active_context_messages(
                tail_messages,
                insert_missing_tool_stubs=False,
            )
            skipped_tail_gap = False
            for msg in reversed(tail_for_selection):
                msg_tokens = count_message_tokens(msg)
                if used + tail_token_total + msg_tokens > assembly_cap:
                    if self._is_budget_droppable_tail_message(msg):
                        skipped_tail_gap = True
                        continue
                    break
                if skipped_tail_gap:
                    break
                kept_tail_reversed.append(msg)
                tail_token_total += msg_tokens
            tail_selected = list(reversed(kept_tail_reversed))
            summary_budget = max(0, assembly_cap - used - tail_token_total)
        if anchor_source is not None:
            anchor_part = self._latest_user_context_anchor(anchor_source, tail_selected)

        # Collect DAG summaries — highest depth first for context hierarchy
        summary_parts: list[str] = []
        last_role = result[-1].get("role", "system") if result else "system"
        if not result or result[-1].get("role") == "system":
            # The summary becomes the first provider-visible message: either no
            # leading anchor exists (gateway-style assembly) or the system
            # prompt is the only anchor, which Anthropic extracts into a
            # separate field. Either way messages[0] must be role "user"; an
            # assistant summary here is rejected with HTTP 400 after the second
            # compaction.
            summary_role = "user"
        else:
            summary_role = "assistant" if last_role != "assistant" else "user"
        if anchor_part is not None:
            anchor_msg = {"role": summary_role, "content": anchor_part}
            if summary_budget is None or count_message_tokens(anchor_msg) <= summary_budget:
                summary_parts.append(anchor_part)

        all_nodes = self._dag.get_session_nodes(self._session_id)
        if all_nodes:
            # Group by depth, take the most recent uncondensed at each level
            # For active context, we want the highest-level summaries
            # that haven't been condensed into even higher levels
            depths = sorted(set(n.depth for n in all_nodes), reverse=True)
            for d in depths:
                uncondensed = self._dag.get_uncondensed_at_depth(self._session_id, d)
                for node in uncondensed:
                    depth_label = {
                        0: "Recent",
                        1: "Session Arc",
                        2: "Durable",
                    }.get(d, f"Depth-{d}")
                    summary_parts.append(
                        f"[{depth_label} Summary (d{d}, node {node.node_id})]\n"
                        f"{node.summary}\n"
                        f"[Expand for details: {node.expand_hint}]"
                    )

        if summary_parts:
            selected_parts = summary_parts
            if summary_budget is not None:
                selected_parts = []
                for part in summary_parts:
                    candidate = "\n\n---\n\n".join(selected_parts + [part])
                    candidate_msg = {"role": summary_role, "content": candidate}
                    if count_message_tokens(candidate_msg) > summary_budget:
                        if part == anchor_part:
                            continue
                        continue
                    selected_parts.append(part)
            if selected_parts:
                combined = "\n\n---\n\n".join(selected_parts)
                # The host's own field for "this row is a summary, not the user's
                # words" (#29, Decided); it also lets the plugin know its row by flag.
                result.append({"role": summary_role, "content": combined, "_compressed_summary": True})

        # Fresh tail
        result.extend(tail_selected)

        # ── Active-context cleanup / tool-pair guardrail ──
        # Drop assistant turns that carry only blank/internal structured content,
        # then ensure provider-valid tool-call/result sequencing.
        result = self._sanitize_active_context_messages(result)
        if leading_msg is None:
            while result and result[0].get("role") in {"assistant", "tool"}:
                result = result[1:]
        if (
            assembly_cap is not None
            and anchor_part is not None
            and count_messages_tokens(result) > assembly_cap
        ):
            trimmed_result: list[Dict[str, Any]] = []
            for msg in result:
                content = normalize_content_value(msg.get("content")) or ""
                if _PRESERVED_OBJECTIVE_CONTEXT_PREFIX not in content:
                    trimmed_result.append(msg)
                    continue
                parts = [
                    part for part in content.split("\n\n---\n\n")
                    if not part.lstrip().startswith(_PRESERVED_OBJECTIVE_CONTEXT_PREFIX)
                ]
                if parts:
                    trimmed = msg.copy()
                    trimmed["content"] = "\n\n---\n\n".join(parts)
                    trimmed_result.append(trimmed)
            result = self._sanitize_active_context_messages(trimmed_result)

        return result

    def _untouched_return(
        self,
        messages: List[Dict[str, Any]],
        assembled: List[Dict[str, Any]],
        working_tail: List[Dict[str, Any]],
        *,
        has_system: bool,
        index_by_working: Optional[Dict[int, int]],
    ) -> List[Dict[str, Any]]:
        """The context returned at a compaction.

        The host's system row in place (ruling 12), the summary row the assembly made,
        and the fresh tail as the host's own dicts, untouched: no sanitiser, no copy
        (ruling 5; #29 W3 finds them again by the key they carry). When the working
        copies cannot be mapped back to the host's entries, the assembled list is
        returned and the event recorded.
        """
        if index_by_working is None:
            return assembled
        positions = [index_by_working.get(id(message)) for message in working_tail]
        system_ok = not has_system or (bool(messages) and messages[0].get("role") == "system")
        if any(position is None for position in positions) or not system_ok:
            self._records.event(
                "untouched_return_unavailable",
                session=self._plugin_session or None,
                detail={"unmapped_tail_entries": sum(1 for p in positions if p is None), "system_ok": system_ok},
            )
            return assembled
        summaries = [
            message for message in assembled
            if isinstance(message, dict) and message.get("_compressed_summary") is True
        ]
        head = [messages[0]] if has_system else []
        return head + summaries + [messages[position] for position in positions]

    def _is_budget_droppable_tail_message(self, message: Dict[str, Any]) -> bool:
        """Return whether an over-budget tail message may be evicted.

        User turns are prompt-bearing context and stop tail selection when they
        cannot fit. Assistant/tool turns are derived context; if one bulky turn
        blocks older prompt material, skip it and keep scanning for budgetable
        user intent or compact status that still fits.
        """
        role = message.get("role")
        if role not in {"assistant", "tool"}:
            return False
        content = normalize_content_value(message.get("content")) or ""
        if _PRESERVED_TODO_CONTEXT_PREFIX in content:
            return False
        if _PRESERVED_OBJECTIVE_CONTEXT_PREFIX in content:
            return False
        return True

    def _should_force_overflow_recovery(
        self,
        observed_tokens: Optional[int] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        assembly_cap = self._effective_assembly_token_cap()
        if assembly_cap is None:
            return False

        tokens = self._overflow_recovery_signal_tokens(
            observed_tokens=observed_tokens,
            messages=messages,
        )
        if tokens is None:
            return False
        return tokens >= assembly_cap

    def _overflow_recovery_signal_tokens(
        self,
        observed_tokens: Optional[int] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[int]:
        candidates: list[int] = []
        if observed_tokens is not None and observed_tokens > 0:
            candidates.append(observed_tokens)
        if messages is not None:
            candidates.append(count_messages_tokens(messages))
        if not candidates:
            return None
        return max(candidates)

    def _overflow_recovery_assembly_cap(
        self,
        observed_tokens: Optional[int] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[int]:
        assembly_cap = self._effective_assembly_token_cap()
        if assembly_cap is None:
            return None
        if messages is None or observed_tokens is None or observed_tokens <= 0:
            return assembly_cap

        message_tokens = count_messages_tokens(messages)
        overhead_tokens = max(0, observed_tokens - message_tokens)
        return max(1, assembly_cap - overhead_tokens)

    def _effective_assembly_token_cap(self) -> Optional[int]:
        """Return the active assembly cap, if any.

        Two knobs can constrain the assembled active context:
        - max_assembly_tokens: explicit hard cap
        - reserve_tokens_floor: keep headroom inside context_length
        """
        caps: list[int] = []

        if self._config.max_assembly_tokens > 0:
            caps.append(self._config.max_assembly_tokens)

        if self.context_length > 0 and self._config.reserve_tokens_floor > 0:
            reserve_cap = self.context_length - self._config.reserve_tokens_floor
            if reserve_cap > 0:
                caps.append(reserve_cap)
            else:
                logger.warning(
                    "LCM reserve_tokens_floor=%d disables reserve-based assembly cap because context_length=%d",
                    self._config.reserve_tokens_floor,
                    self.context_length,
                )

        if not caps:
            return None

        return max(1, min(caps))

    # -- Internal: helpers -------------------------------------------------

    def _derive_auto_focus_topic(
        self,
        messages: List[Dict[str, Any]],
    ) -> Optional[str]:
        """Infer a compact focus hint from the most recent real user turns.

        Walks the message list backwards, collecting up to
        ``_AUTO_FOCUS_MAX_TURNS`` user messages (skipping context summaries
        and empty turns).  Returns a brief text block suitable for injection
        into the summarizer prompt as ``focus_topic``.

        The ``messages`` parameter must be ``working_messages`` (output of
        ``_ingest_messages``), not raw messages.

        Mirrors Hermes upstream ``ContextCompressor._derive_auto_focus_topic``
        from ``fix/compression-auto-focus-topic``.
        """
        candidates: list[str] = []
        for idx in range(len(messages) - 1, -1, -1):
            msg = messages[idx]
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            # Skip context compaction summaries — they are synthetic, not
            # real user intent.
            if self._is_context_summary_content(content):
                continue
            text = (text_content_for_pattern_matching(content) or "").strip()
            if self._is_volatile_ignored_quarantine_placeholder(msg, text):
                continue
            if not text:
                continue
            text = " ".join(text.split())
            if len(text) > _AUTO_FOCUS_TURN_MAX_CHARS:
                text = text[: _AUTO_FOCUS_TURN_MAX_CHARS - 1].rstrip() + "…"
            candidates.append(text)
            if len(candidates) >= _AUTO_FOCUS_MAX_TURNS:
                break

        if not candidates:
            return None

        candidates.reverse()
        focus = "Recent user focus:\n" + "\n".join(f"- {item}" for item in candidates)
        if len(focus) > _AUTO_FOCUS_MAX_CHARS:
            focus = focus[: _AUTO_FOCUS_MAX_CHARS - 1].rstrip() + "…"
        return focus

    @staticmethod
    def _is_context_summary_content(content: Any) -> bool:
        """Check whether message content is a synthetic context summary.

        Only checks string content — LCM/ Hermes compression summaries are
        always stored as plain strings, never as structured multimodal parts.
        """
        if not isinstance(content, str):
            return False
        return (
            "CONTEXT COMPACTION" in content
            or "CONTEXT SUMMARY" in content
            or "Earlier turns have been compacted" in content
            or "Earlier turns were compacted" in content
        )

    @staticmethod
    def _extract_expand_hint(summary: str) -> str:
        """Extract the 'Expand for details about:' line from a summary."""
        marker = "Expand for details about:"
        idx = summary.rfind(marker)
        if idx >= 0:
            hint = summary[idx + len(marker):].strip()
            # Take first line only
            return hint.split("\n")[0].strip()
        return ""

    # -- Backup path -------------------------------------------------------

    def backup_dir(self) -> Path:
        """Return the directory where ``maintenance.backup_database`` writes."""
        db_path = Path(self._store.db_path)
        backup_root = (
            Path(self._hermes_home).expanduser()
            if getattr(self, "_hermes_home", "")
            else db_path.parent
        )
        return backup_root / "backups" / "lcm"

    # -- Lifecycle ---------------------------------------------------------

    def shutdown(self):
        self._unregister_active_engine_binding()
        self._store.close()
        self._dag.close()
