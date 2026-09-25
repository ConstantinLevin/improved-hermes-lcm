"""LCM Engine — Lossless Context Management.

Implements the ContextEngine ABC. Replaces the built-in ContextCompressor with the
plugin's own compaction over its record, which keeps what the host hands over.
"""

import atexit
import copy
import json
import logging
import weakref
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.context_engine import ContextEngine

from .codex_routing import (
    _codex_oauth_context_cap,
    _is_codex_gpt55_route,
)
from .config import LCMConfig
from .dag import SummaryDAG
from .db_bootstrap import STORE_FILENAME, StoreClosedError, StoreRefusedError
from .engine_registry import (
    _ACTIVE_ENGINE_REGISTRY_LOCK,
    _ACTIVE_ENGINES_BY_CONVERSATION_ID,
    _ACTIVE_ENGINES_BY_SESSION_ID,
    _remove_registry_entries_for_engine,
    resolve_active_lcm_engine,  # noqa: F401  (re-exported: hosts import it from .engine)
)
from .escalation import configured_route_problem
from .extraction import (
    sanitize_pre_compaction_content,
    sanitize_pre_compaction_tool_arguments,
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
from .message_analysis import (
    _is_synthetic_assistant_noise,
    _matched_tool_call_ids,
    _tool_call_id,
)
from .fresh_tail import FreshTailBoundary, resolve_fresh_tail_boundary
from .backup import DailyBackup
from .compaction import CompactionMixin
from .reset_state import ResetStateMixin
from .plugin_sessions import PluginSessions
from .record_store import RecordStore
from .record_write import RecordWriteMixin
from .store import MessageStore
from .tokens import count_messages_tokens
from . import tools as lcm_tools

logger = logging.getLogger(__name__)


_CODEX_GPT55_COMPACTION_THRESHOLD = 0.85
_TOTAL_COMPACTIONS_SCOPE = "plugin_session"

# Set by an exit handler registered right after the first finalizer (which registers
# the finalizers' own exit handler), so it runs before them: an engine's finalizer
# can then say whether it runs at process exit or at collection.
_PROCESS_EXITING = False
_EXIT_MARK_REGISTERED = False


def _mark_process_exiting() -> None:
    global _PROCESS_EXITING
    _PROCESS_EXITING = True


def _register_exit_mark() -> None:
    global _EXIT_MARK_REGISTERED
    if not _EXIT_MARK_REGISTERED:
        atexit.register(_mark_process_exiting)
        _EXIT_MARK_REGISTERED = True


def _close_helpers(helpers: tuple, label: str, reason_box: list, backup: Optional[DailyBackup] = None) -> None:
    """Close one engine's store helpers and say so. Run by ``LCMEngine.close`` with
    its reason, and otherwise by the engine's finalizer, when the engine is collected
    or the process exits. It holds the helpers, never the engine. A daily backup the
    engine started is stopped first and waited for; the slot stays as it was."""
    reason = reason_box[0] or ("the process is exiting" if _PROCESS_EXITING else "its engine was collected")
    if backup is not None:
        try:
            backup.stop()
        except Exception:
            logger.warning("LCM could not stop the daily backup of %s (%s)", label, reason, exc_info=True)
    for helper in helpers:
        try:
            helper.close(reason)
        except Exception:
            logger.warning("LCM could not close the %s of %s (%s)", type(helper).__name__, label, reason,
                           exc_info=True)
    logger.info("LCM closed the store connections of %s: %s", label, reason)


class ReviewForkDetachRefused(RuntimeError):
    """Raised when the host detaches an engine copy from every session.

    The host makes this call, ``bind_session_state(session_db=None, session_id="")``,
    on the engine copy of its background-review fork and enables the fork's
    compaction only when it succeeds. The host neither commits nor confirms a fork's
    compaction and names no parent for the fork, so the plugin could record it only
    by guessing. Refusing keeps the fork's compaction disabled, as it was.
    """


class LCMEngine(
    CompactionMixin,
    RecordWriteMixin,
    ResetStateMixin,
    ContextEngine,
):
    """Lossless Context Management engine.

    Automatic LCM compaction is routine background maintenance. Hosts that
    support user-visible compaction status opt-outs should keep successful
    automatic LCM passes silent unless the user explicitly asks for diagnostics.

    The plugin's record is the only store, filled at each compaction
    (``record_store``, ``record_write``); the compaction path is ``compaction``;
    the tools read views over the record (``store``, ``dag``, ``tools``).
    """

    def __init__(self, config: LCMConfig | None = None,
                 hermes_home: str = ""):
        self._config = config or LCMConfig.from_env()
        self._hermes_home = hermes_home
        # A configured summariser the plugin cannot use is refused here, visibly, and
        # every compaction aborts with the same cause (#9).
        problem = configured_route_problem(self._config)
        if problem is not None:
            logger.error("LCM refuses its configured summariser: %s", problem)
        # Why this engine's store connections were closed; None while they are open.
        self._closed_reason: Optional[str] = None

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
        self._last_overflow_recovery_failed = False
        self._last_compression_status = "idle"
        self._last_compression_noop_reason = ""
        # Read by the host after compress(): an aborted compaction returned its input
        # unchanged, and the host shows "⚠ Compression aborted: <_last_summary_error>".
        self._last_compress_aborted = False
        self._last_summary_error: Optional[str] = None

    def clone_for_agent(self) -> "LCMEngine":
        """Return a fresh runtime engine for one AIAgent instance.

        Hermes registers plugin context engines process-wide, while gateway
        runtimes may keep multiple cached AIAgent instances alive at once
        (different platforms, chats, cron jobs, etc.).  LCM keeps its session
        binding and its returned attempts on the engine object itself, so sharing
        one registered instance across agents would let one conversation act for
        another's session.

        The clone shares the same durable SQLite database path/configuration,
        but gets independent session and attempt state. Runtime
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
        sessions, and the tools' readers over its views.

        Their connections are closed by :meth:`close`, and otherwise by a finalizer
        when this engine is collected or the process exits: the host gives an engine
        copy no teardown call (#20). The finalizer holds the helpers, never the engine.
        """
        helpers = []
        try:
            for build in (
                lambda: MessageStore(db_path, hermes_home=hermes_home),
                lambda: SummaryDAG(db_path),
                lambda: PluginSessions(db_path),
                lambda: RecordStore(db_path),
            ):
                helpers.append(build())
        except Exception:
            _close_helpers(tuple(helpers), f"an engine on {db_path}", ["its store could not be opened"])
            raise
        self._store, self._dag, self._sessions, self._records = helpers
        self._close_box: list = [None]
        # The store's daily backup (#6); it holds the record helper, never the engine.
        self._backup = DailyBackup(db_path, self._records)
        self._storage_finalizer = weakref.finalize(
            self, _close_helpers, tuple(helpers), f"engine {id(self):#x} on {db_path}", self._close_box,
            self._backup,
        )
        _register_exit_mark()

    def _close_storage(self, reason: str) -> None:
        """Close the bound helpers now, once, with the reason given."""
        finalizer = getattr(self, "_storage_finalizer", None)
        if finalizer is not None and finalizer.alive:
            self._close_box[0] = reason
            finalizer()

    def close(self, reason: str = "closed by its owner") -> None:
        """Close this engine's store connections. Idempotent.

        Each helper closes once any statement or transaction it runs on another
        thread has finished; a transaction still open is rolled back with a warning.
        Afterwards every use of the store raises ``StoreClosedError``: a closed
        engine is never reopened or reused. Called at plugin unload for the engine
        registered with the host; an engine copy is closed when it is collected.
        """
        if self._closed_reason is not None:
            return
        self._closed_reason = reason
        self._unregister_active_engine_binding()
        self._close_storage(reason)


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
        if self._closed_reason is not None:
            raise StoreClosedError(
                f"LCM's engine was closed ({self._closed_reason}); a closed engine is never reopened"
            )

        self._close_storage(f"the engine was rebound to the store of {hermes_home}")
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
            if key == "api_key":
                # Compared as the objects they are; a credential is never stringified.
                incoming_key = kwargs.get(key) or ""
                if ignore_empty_optional and not incoming_key:
                    continue
                if incoming_key != (self.api_key or ""):
                    return False
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

    # -- ContextEngine optional methods ------------------------------------

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

    def _fresh_tail_boundary(self, messages: List[Dict[str, Any]]) -> FreshTailBoundary:
        # The newest message is always in the tail (#31's floor): no setting makes
        # the tail empty, so the return always ends with the host's own newest dict.
        return resolve_fresh_tail_boundary(
            messages,
            fresh_tail_count=max(1, int(self._config.fresh_tail_count or 0)),
            fresh_tail_max_tokens=self._config.fresh_tail_max_tokens,
        )

    def _fresh_tail_start(self, messages: List[Dict[str, Any]]) -> int:
        return self._fresh_tail_boundary(messages).start

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
        for key in ("base_url", "provider", "api_mode"):
            if key in kwargs:
                setattr(self, key, str(kwargs.get(key) or ""))
        if "api_key" in kwargs:
            # A credential passes through as the host gave it, never stringified.
            self.api_key = kwargs.get("api_key") or ""
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
            self._start_daily_backup()
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
        self._start_daily_backup()

    def _start_daily_backup(self) -> None:
        """Start the store's daily backup on its own thread when one is due (#6)."""
        if self._closed_reason is None:
            self._backup.start_if_due()

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
        # context holds now is in its context. A list the host hands over is settled
        # first, as at a turn's end and at the preflight: a confirmation waiting for
        # its compaction's name, a return adopted without one, the bindings. So a
        # summary a compaction inside this turn put into the context can be expanded
        # at once.
        if self._closed_reason is not None:
            return json.dumps({"error": f"LCM's store connections of this engine were closed "
                                        f"({self._closed_reason}); a closed engine is never reused"})
        messages = kwargs.get("messages")
        if messages:
            self._bind_from_list(messages)
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
            status["overflow_recovery_failed"] = self._last_overflow_recovery_failed
            status["conversation_id"] = conversation_id
        return status

    def update_model(self, model: str, context_length: int,
                     base_url: str = "", api_key: str = "",
                     provider: str = "",
                     api_mode: str = "") -> None:
        self.model = str(model or "")
        self.base_url = str(base_url or "")
        # The credential exactly as the host gave it: a string, or a callable the host
        # resolves itself (key_cmd, Entra). Never stringified, never called here (#9).
        self.api_key = api_key if api_key else ""
        self.provider = str(provider or "")
        self.api_mode = str(api_mode or "")
        self._set_context_length(context_length, source="update_model")
        self._update_model_pending_session_start = True

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

    # -- Internal: overflow recovery ----------------------------------------

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
