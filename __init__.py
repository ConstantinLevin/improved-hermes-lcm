"""Hermes LCM Plugin — Lossless Context Management.

Replaces the built-in ContextCompressor with a context engine that keeps what the
host hands it at each compaction in its own record, returns summaries of the older
part of the context, and provides tools to retrieve the originals.

Based on the LCM paper by Ehrlich & Blackman (Voltropy PBC, Feb 2026).
"""

import logging
import os

logger = logging.getLogger(__name__)


def _instruction_not_delivered(engine, why: str) -> None:
    """The instruction cannot reach the agent: an error in the log and a store event,
    which the doctor lists (#16). Compaction goes on; its summaries hedge themselves."""
    logger.error("LCM cannot deliver its instruction to the agent: %s", why)
    try:
        engine._records.event("instruction_not_delivered", detail=why)
    except Exception:
        logger.debug("LCM could not record that its instruction is not delivered", exc_info=True)


class _Instruction:
    """What the registration established: the section's text and the host's framing, or
    why the plugin registered no section. The delivery check reads it (#16)."""

    def __init__(self) -> None:
        self.text = None
        self.host = None
        self.refused = None


def _register_instruction(ctx, engine, resolve_active_lcm_engine) -> "_Instruction":
    """The plugin's instruction as a section of the host's system prompt, and its two
    skills, ``hermes-lcm:summaries`` and ``hermes-lcm:setup`` (#16).

    The section's content is a callable the host gives the session's metadata when it
    renders the prompt: it returns the instruction only for a session this plugin's
    engine serves, found in the engine registry by the host's session id, and "" (which
    the host skips) for any other, so that an agent whose context engine is not LCM is
    told nothing about LCM's summaries. Each render that returns the text is noted on the
    engine copy it was returned for, which the delivery check reads."""
    from .guidance import SECTION_ID, SKILLS, HostSections, InstructionRefused, instruction_text

    instruction = _Instruction()
    register_section = getattr(ctx, "register_system_prompt_section", None)
    if not callable(register_section):
        instruction.refused = "the host offers no register_system_prompt_section to this plugin"
    else:
        try:
            host = HostSections()
            text = instruction_text(host.max_chars)

            def _section(info):
                session_id = str((info or {}).get("session_id") or "")
                active = resolve_active_lcm_engine(session_id=session_id) if session_id else None
                if active is None or getattr(active, "name", None) != "lcm":
                    return ""
                active._instruction_rendered = getattr(active, "_instruction_rendered", 0) + 1
                return text

            register_section(SECTION_ID, _section, position="after_memory", max_chars=host.max_chars)
            instruction.text, instruction.host = text, host
        except InstructionRefused as exc:
            instruction.refused = str(exc)
        except Exception as exc:
            instruction.refused = f"the host refused the section ({type(exc).__name__}: {exc})"
    if instruction.refused is not None:
        _instruction_not_delivered(engine, instruction.refused)

    register_skill = getattr(ctx, "register_skill", None)
    if not callable(register_skill):
        logger.warning("LCM's skills are not registered: the host offers no register_skill to this plugin")
        return instruction
    for name, path, description in SKILLS:
        try:
            register_skill(name, path, description=description)
        except Exception as exc:
            logger.warning("LCM could not register its skill %s (%s): %s", name, path, exc)
    return instruction


def _check_instruction_delivered(instruction, resolve_active_lcm_engine, payload) -> None:
    """The request's system prompt carries the plugin's section, for a session an LCM
    engine serves; checked once per distinct prompt (#16). Never raises into the host."""
    from .guidance import check_delivery

    session_id = str(payload.get("session_id") or "")
    active = resolve_active_lcm_engine(session_id=session_id) if session_id else None
    if active is None or getattr(active, "name", None) != "lcm":
        return

    def _record(kind, detail):
        try:
            active._records.event(kind, detail=detail)
        except Exception:
            logger.debug("LCM could not record %s", kind, exc_info=True)

    try:
        check_delivery(active, payload.get("system_prompt"), host_session_id=session_id, text=instruction.text,
                       host=instruction.host, refused=instruction.refused, record=_record)
    except Exception:
        logger.warning("LCM could not check that its instruction reached the agent", exc_info=True)


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

    # The instruction to the agent, as a system-prompt section, and the two skills
    # (#16). No hook injects it: a hook's text rides on the user's message and is
    # replayed on every later request, as if the user had written it (#36).
    instruction = _register_instruction(ctx, engine, resolve_active_lcm_engine)

    register_hook = getattr(ctx, "register_hook", None)
    if callable(register_hook):
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

        def _on_pre_api_request(**payload):
            # R10: the list a request sends, until the session's fixed prefix is measured.
            turn_signals.request_sent(str(payload.get("session_id") or ""), str(payload.get("turn_id") or ""),
                                      payload.get("conversation_history"))
            # #16: the system prompt this request sends carries the plugin's section.
            _check_instruction_delivered(instruction, resolve_active_lcm_engine, payload)

        register_hook("pre_api_request", _on_pre_api_request)
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
