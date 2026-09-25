"""Hermes LCM Plugin — Lossless Context Management.

Replaces the built-in ContextCompressor with a context engine that keeps what the
host hands it at each compaction in its own record, returns summaries of the older
part of the context, and provides tools to retrieve the originals.

Based on the LCM paper by Ehrlich & Blackman (Voltropy PBC, Feb 2026).
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def get_recall_policy() -> str:
    """Load the canonical product policy without making bare imports package-dependent."""
    from .guidance import get_recall_policy as _get_recall_policy

    return _get_recall_policy()


def _env_flag_enabled(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _make_wrapped_handler(tool_name: str, engine):
    """Route a registered lcm_* tool through the engine dispatch path."""
    def _wrapped(args: dict, **kwargs) -> str:
        return engine.handle_tool_call(tool_name, args, **kwargs)
    return _wrapped


def _host_forwards_registered_tool_messages(ctx) -> bool:
    """Return whether ctx.register_tool handlers receive active messages.

    Hermes Agent's current registry dispatch passes task_id/user_task to
    plugin tools, but not the active conversation messages list. Registering
    duplicate lcm_* tool names on that host makes the model call the registry
    handler instead of the native context-engine dispatch branch, so LCM loses
    the live list it settles before a tool answers (a summary a compaction
    inside the turn returned could not be expanded at once).

    Keep plugin-side tool registration opt-in until a host explicitly
    advertises that registered context-engine handlers receive messages.
    """
    capability = getattr(ctx, "context_engine_tool_handlers_receive_messages", False)
    if callable(capability):
        try:
            capability = capability()
        except Exception:
            return False
    return bool(capability)


def _session_context_value(name: str) -> str:
    """Read task-local host session metadata with legacy env compatibility."""
    try:
        from gateway.session_context import get_session_env
    except ImportError:
        return str(os.environ.get(name, "") or "")
    try:
        return str(get_session_env(name, "") or "")
    except Exception:
        # Once a concurrent host exposes task-local session context, never fall
        # back to process-global env after a read failure: it may name another
        # lane. Unbound is safer than cross-session command dispatch.
        logger.debug("LCM plugin command could not read %s", name, exc_info=True)
        return ""


def _command_engine_for_current_session(engine, resolve_active_lcm_engine):
    """Resolve the runtime serving the current plugin-command invocation.

    Gateway hosts bind task-local session/lane metadata before dispatching a
    plugin slash command. Prefer the already-active AIAgent clone registered for
    that lane. If no clone exists yet, keep the process-wide prototype's genuine
    ``(unbound)`` cold-start status rather than mutating shared runtime state.
    """
    session_id = _session_context_value("HERMES_SESSION_ID")
    conversation_id = _session_context_value("HERMES_SESSION_KEY")
    if session_id or conversation_id:
        active_engine = resolve_active_lcm_engine(
            session_id=session_id,
            conversation_id=conversation_id,
        )
        if active_engine is not None:
            return active_engine
    return engine


def _make_command_handler(handle_lcm_command, engine, resolve_active_lcm_engine):
    def _handler(raw_args: str):
        return handle_lcm_command(
            raw_args,
            _command_engine_for_current_session(
                engine,
                resolve_active_lcm_engine,
            ),
        )

    return _handler


def register(ctx):
    """Plugin entry point — register the LCM context engine and tools."""
    from .config import LCMConfig
    from .engine import LCMEngine, resolve_active_lcm_engine
    from .schemas import (
        LCM_GREP,
        LCM_EXPAND,
        LCM_EXPAND_QUERY,
        LCM_STATUS,
        LCM_INSPECT,
        LCM_DOCTOR,
    )

    config = LCMConfig.from_env()

    # The store lives under the home the host gives. Without one (and without
    # LCM_DATABASE_PATH) the engine refuses to guess a location.
    hermes_home = ""
    try:
        from hermes_cli.config import get_hermes_home
        hermes_home = str(get_hermes_home())
    except Exception:
        hermes_home = os.environ.get("HERMES_HOME", "")

    engine = LCMEngine(config=config, hermes_home=hermes_home)

    # Register as the context engine (replaces ContextCompressor)
    ctx.register_context_engine(engine)

    # At unload (a forced rediscovery, the plugin doctor's load) the host drops this
    # engine and the next register() builds another; its store connections close with
    # it. The copies the host made for its agents stay open: they belong to those
    # agents, and a running session keeps its store (#20).
    on_unload = getattr(ctx, "on_unload", None)
    if callable(on_unload):
        def _close_registered_engine() -> None:
            engine.close("the plugin was unloaded")

        on_unload(_close_registered_engine)

    # Ship the same recall contract through both Hermes plugin skill
    # registration (explicit qualified loads) and the installer's ordinary
    # profile skill link (normal discovery). Older hosts simply lack this
    # capability and keep their existing schema-driven behavior.
    skill_root = Path(__file__).resolve().parent / "skills" / "hermes-lcm"
    register_skill = getattr(ctx, "register_skill", None)
    if callable(register_skill):
        try:
            register_skill(
                "hermes-lcm",
                skill_root,
                description=(
                    "Use, configure, diagnose, and recall exact evidence "
                    "with the Hermes-LCM lossless context plugin."
                ),
            )
        except Exception as exc:
            logger.warning(
                "LCM bundled skill registration did not complete; normal "
                "profile skill discovery may still be available: %s",
                exc,
            )

    register_hook = getattr(ctx, "register_hook", None)
    if callable(register_hook):
        # Hermes invokes this hook after the context engine has received
        # on_session_start(). Resolve through LCM's own registry so merely
        # loading the plugin cannot inject guidance when another context
        # engine is serving the turn. Capture one validated policy value for
        # deterministic, byte-stable injection across eligible turns.
        try:
            recall_policy = get_recall_policy()

            def _on_pre_llm_call(**payload):
                session_id = str(payload.get("session_id") or "")
                conversation_id = str(
                    payload.get("conversation_id")
                    or payload.get("gateway_session_key")
                    or ""
                )
                active_engine = resolve_active_lcm_engine(
                    session_id=session_id,
                    conversation_id=conversation_id,
                )
                if active_engine is None or getattr(active_engine, "name", None) != "lcm":
                    return None
                return {"context": recall_policy}

            register_hook("pre_llm_call", _on_pre_llm_call)
        except Exception as exc:
            logger.warning(
                "LCM recall-policy hook registration did not complete; "
                "tool schemas remain available: %s",
                exc,
            )

        # The host's explicit session signals: /new, and a delegate's parent.
        # They write into the store of the home this plugin was loaded for.
        def _on_session_reset(**payload):
            engine._sessions.on_session_reset_hook(**payload)

        def _on_subagent_start(**payload):
            engine._sessions.subagent_start_hook(**payload)

        register_hook("on_session_reset", _on_session_reset)
        register_hook("subagent_start", _on_subagent_start)

        # The per-turn signals, per host session id, process-wide (#32 §1, D1):
        # a turn begins, a tool of it ran, a response of it arrived. The turn's end
        # is the engine's on_turn_complete. Each observes and returns nothing.
        from . import turn_signals

        def _on_pre_llm_call_turn(**payload):
            turn_signals.turn_began(str(payload.get("session_id") or ""), str(payload.get("turn_id") or ""),
                                    payload.get("conversation_history"))
            return None

        def _on_post_tool_call(**payload):
            turn_signals.tool_ran(str(payload.get("session_id") or ""), str(payload.get("turn_id") or ""))

        def _on_post_api_request(**payload):
            turn_signals.response_arrived(str(payload.get("session_id") or ""), str(payload.get("turn_id") or ""))

        register_hook("pre_llm_call", _on_pre_llm_call_turn)
        register_hook("post_tool_call", _on_post_tool_call)
        register_hook("post_api_request", _on_post_api_request)

    # Register tools via the plugin registry only on hosts that preserve the
    # active messages=... contract for registered context-engine tools.
    # Older/current Hermes hosts already expose lcm_* correctly through the
    # native context-engine schema/dispatch path (Path B). Registering duplicate
    # names through the plugin registry (Path A) on message-blind hosts would
    # shadow Path B and lose the live list a tool call settles, so the Path B fallback is the
    # expected healthy behavior there.
    _TOOLS = [
        ("lcm_grep", LCM_GREP, "🔍"),
        ("lcm_expand", LCM_EXPAND, "🔎"),
        ("lcm_expand_query", LCM_EXPAND_QUERY, "❓"),
        ("lcm_status", LCM_STATUS, "💚"),
        ("lcm_inspect", LCM_INSPECT, "🧭"),
        ("lcm_doctor", LCM_DOCTOR, "🏥"),
    ]
    register_tool = getattr(ctx, "register_tool", None)
    if callable(register_tool) and _host_forwards_registered_tool_messages(ctx):
        for name, schema, emoji in _TOOLS:
            try:
                register_tool(
                    name=name,
                    toolset="context_engine",
                    schema=schema,
                    handler=_make_wrapped_handler(name, engine),
                    description=schema.get("description", ""),
                    emoji=emoji,
                )
            except Exception as exc:
                logger.warning(
                    "LCM plugin-registry tool registration for %s did not complete; "
                    "LCM tools remain available through context-engine schemas: %s",
                    name,
                    exc,
                )
    elif callable(register_tool):
        logger.info(
            "LCM tools are available through context-engine schemas "
            "(expected Path B fallback on this Hermes host). Standalone "
            "plugin-registry tool registration (Path A) requires message-aware "
            "handlers and is not required here."
        )
    else:
        logger.info(
            "LCM tools are available through context-engine schemas (Path B); "
            "plugin-registry tool registration is unavailable on this Hermes "
            "host and is not required."
        )

    register_command = getattr(ctx, "register_command", None)
    slash_enabled = _env_flag_enabled("LCM_ENABLE_SLASH_COMMAND", default=False)
    if callable(register_command) and slash_enabled:
        from .command import handle_lcm_command

        register_command(
            "lcm",
            _make_command_handler(
                handle_lcm_command,
                engine,
                resolve_active_lcm_engine,
            ),
            description="LCM status and diagnostics",
        )
    elif callable(register_command):
        logger.info("LCM slash command registration disabled (set LCM_ENABLE_SLASH_COMMAND=1 to enable /lcm)")
    else:
        logger.info("LCM slash command registration unavailable on this Hermes host; continuing without /lcm")

    logger.info("LCM plugin loaded — lossless context management active")
