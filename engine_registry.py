"""Process-wide registry of active LCM runtime clones by session/lane.

Isolated from ``engine.py`` (WS5 seam): LCM clones register their own
session/conversation binding so the system-prompt section of the instruction and
the ``/lcm`` command find the active clone instead of the process-wide plugin
singleton. The lock and the two weak
registries are held in process state that outlives a reload of the plugin (below),
alongside the pure resolver/matcher helpers that read them. ``engine.py`` imports the shared lock, the two registries, the removal
helper, and the public ``resolve_active_lcm_engine`` entry point; the binding
methods on ``LCMEngine`` mutate the same shared objects by reference.
"""

from __future__ import annotations

import os
import sys
import threading
import types
import weakref
from typing import Any

# The registry outlives a forced reload of the plugin. The host's loader evicts this
# package and every submodule before it loads the plugin again (``hermes_cli/
# plugins_loader.py`` ``_load_directory_module``, ``_evict_modules``), while the agents
# already running keep the engine copies bound here; a registry held in this module would
# be replaced by an empty one, and the reloaded section and delivery check would find no
# engine for them. So the registry, and the lock of the delivery check's seen-prompt sets,
# live in one process-wide object under a name outside the plugin's package, created
# once and taken over by every later load. So do the store events whose store was
# closed before they could be written (``record_store``): they wait there, by database
# path, for the next store opened on the same database.
_PROCESS_STATE_NAME = "_hermes_lcm_process_state"


def _process_state() -> types.ModuleType:
    state = sys.modules.get(_PROCESS_STATE_NAME)
    if state is None:
        fresh = types.ModuleType(_PROCESS_STATE_NAME, "Hermes-LCM state that outlives a reload of the plugin.")
        state = sys.modules.setdefault(_PROCESS_STATE_NAME, fresh)
    # Each field is added once; a later load of a newer version adds the fields it needs
    # to the state an earlier load created, and takes over those it finds.
    for name, make in (
        ("registry_lock", threading.RLock),
        ("by_session_id", weakref.WeakValueDictionary),
        ("by_conversation_id", weakref.WeakValueDictionary),
        ("seen_lock", threading.Lock),
        ("unbound_sessions_noted", set),
        ("unbound_sessions_pending", set),
        ("orphan_lock", threading.Lock),
        ("orphan_events", dict),
    ):
        if not hasattr(state, name):
            setattr(state, name, make())
    return state


PROCESS_STATE = _process_state()
_ACTIVE_ENGINE_REGISTRY_LOCK = PROCESS_STATE.registry_lock
_ACTIVE_ENGINES_BY_SESSION_ID = PROCESS_STATE.by_session_id
_ACTIVE_ENGINES_BY_CONVERSATION_ID = PROCESS_STATE.by_conversation_id


def same_home(left: str, right: str) -> bool:
    """Whether two Hermes homes are the same directory; an empty one is none."""
    if not left or not right:
        return False
    return os.path.realpath(left) == os.path.realpath(right)


def bound_engines(hermes_home: str) -> list:
    """Every engine copy of this process bound to a host session and serving the Hermes
    home ``hermes_home``, once each. The registry spans the process, and the host scopes
    plugins and their registrations by home: a copy serving another profile belongs to
    that profile's load, not to this one."""
    with _ACTIVE_ENGINE_REGISTRY_LOCK:
        seen, engines = set(), []
        for engine in list(_ACTIVE_ENGINES_BY_SESSION_ID.values()):
            if id(engine) in seen or not same_home(str(getattr(engine, "_hermes_home", "") or ""), hermes_home):
                continue
            seen.add(id(engine))
            engines.append(engine)
        return engines


def _is_usable_lcm_engine(engine: Any) -> bool:
    return bool(engine is not None and getattr(engine, "name", None) == "lcm")


def _engine_matches_session_binding(engine: Any, session_id: str) -> bool:
    return bool(
        _is_usable_lcm_engine(engine)
        and session_id
        and str(getattr(engine, "_session_id", "") or "") == session_id
    )


def _engine_matches_conversation_binding(engine: Any, conversation_id: str) -> bool:
    return bool(
        _is_usable_lcm_engine(engine)
        and conversation_id
        and str(getattr(engine, "_conversation_id", "") or "") == conversation_id
    )


def _remove_registry_entries_for_engine(
    engine: Any,
    *,
    keep_session_id: str = "",
    keep_conversation_id: str = "",
) -> None:
    for registered_session_id, registered_engine in list(_ACTIVE_ENGINES_BY_SESSION_ID.items()):
        if registered_engine is engine and registered_session_id != keep_session_id:
            _ACTIVE_ENGINES_BY_SESSION_ID.pop(registered_session_id, None)
    for registered_conversation_id, registered_engine in list(
        _ACTIVE_ENGINES_BY_CONVERSATION_ID.items()
    ):
        if registered_engine is engine and registered_conversation_id != keep_conversation_id:
            _ACTIVE_ENGINES_BY_CONVERSATION_ID.pop(registered_conversation_id, None)


def resolve_active_lcm_engine(
    session_id: str = "",
    conversation_id: str = "",
) -> Any:
    """Return the LCM runtime clone most recently bound to a session/lane.

    Hooks and plugin commands receive only session/lane ids. LCM clones register
    their own session binding when ``on_session_start`` runs, so a hook or a
    command reaches the active clone instead of the process-wide plugin singleton.
    """
    session_id = str(session_id or "")
    conversation_id = str(conversation_id or "")
    with _ACTIVE_ENGINE_REGISTRY_LOCK:
        if session_id:
            engine = _ACTIVE_ENGINES_BY_SESSION_ID.get(session_id)
            if _engine_matches_session_binding(engine, session_id):
                return engine
            if engine is not None:
                _ACTIVE_ENGINES_BY_SESSION_ID.pop(session_id, None)
        if conversation_id:
            engine = _ACTIVE_ENGINES_BY_CONVERSATION_ID.get(conversation_id)
            conversation_matches = _engine_matches_conversation_binding(
                engine,
                conversation_id,
            )
            session_matches = not session_id or _engine_matches_session_binding(
                engine,
                session_id,
            )
            if conversation_matches and session_matches:
                return engine
            if engine is not None and not conversation_matches:
                _ACTIVE_ENGINES_BY_CONVERSATION_ID.pop(conversation_id, None)
    return None
