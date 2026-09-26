"""The plugin's own instruction to the agent (#16), the skills it registers, and the
check that the instruction reached the request.

The instruction is told globally, not per summary, and the plugin delivers it itself:
as a section of the host's system prompt (``register_system_prompt_section``), never as
text the user seems to have written. Its words are the prompt step's (#10); until then
the text is ``skills/hermes-lcm/references/recall-policy.md``.

What the host does with a section, read at Hermes 34343e7:

- it renders the sections when it first builds a session's prompt and again at every
  compaction it commits (``agent/system_prompt.py`` ``_frozen_plugin_prompt_sections``,
  ``invalidate_system_prompt``), and freezes them between;
- a continuing session, restored from the host's database (a resume, a gateway turn on a
  fresh agent), is not rendered: its sections are parsed back out of the stored prompt
  (``agent/conversation_loop.py`` ``restore_plugin_prompt_sections``), so a prompt stored
  before this section existed stays without it until the next compaction;
- a section over its own limit (4,000 characters) is left out whole, and so is a section
  that would take the sections' total past 8,000 characters, in the order of their ids;
  either only with a warning in the host's own log (``hermes_cli/plugins_dispatch.py``
  ``render_system_prompt_sections``); a section whose callable raises is skipped the same
  way, and the previous bytes are kept only when the whole render raises.

So the plugin checks its text against the host's limit itself and refuses it visibly, and
it checks every distinct system prompt the host sends for its exact section
(:func:`check_delivery`), recording ``instruction_not_delivered`` where it is missing:
as the fact it knows, never with a cause it does not. The host's limits and its framing
are read from the host, never assumed.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Callable, Optional

from .engine_registry import PROCESS_STATE

logger = logging.getLogger(__name__)

_SKILLS = Path(__file__).resolve().parent / "skills"

INSTRUCTION_PATH = _SKILLS / "hermes-lcm" / "references" / "recall-policy.md"
SECTION_ID = "lcm"

# (name, SKILL.md, description): each resolves as ``hermes-lcm:<name>`` through the
# host's skill_view (#16).
SKILLS = (
    ("summaries", _SKILLS / "hermes-lcm" / "SKILL.md",
     "Working with the summaries in your context: what they are, and how to look behind their handles."),
    ("setup", _SKILLS / "hermes-lcm-setup" / "SKILL.md",
     "Setting up Hermes-LCM and checking that it is set up and healthy."),
)


class InstructionRefused(RuntimeError):
    """The instruction text cannot be delivered as the host's section."""


class HostSections:
    """The host's own limit and framing for one plugin section, read from the host."""

    def __init__(self) -> None:
        try:
            from hermes_cli.plugins_dispatch import (  # type: ignore
                MAX_SYSTEM_PROMPT_SECTION_CHARS,
                format_system_prompt_section,
            )
        except Exception as exc:
            raise InstructionRefused(
                f"the host's limit and framing for a system-prompt section cannot be read "
                f"({type(exc).__name__}: {exc}); the plugin does not assume them"
            ) from exc
        self.max_chars: int = int(MAX_SYSTEM_PROMPT_SECTION_CHARS)
        self.frame: Callable[[str, str], str] = format_system_prompt_section


def instruction_text(max_chars: int) -> str:
    """The instruction as the section carries it: the file's text, stripped as the host
    strips a section. Raises ``InstructionRefused`` when it is empty or over the host's
    limit, since the host would leave it out."""
    text = INSTRUCTION_PATH.read_text(encoding="utf-8").strip()
    if not text:
        raise InstructionRefused(f"the instruction text is empty: {INSTRUCTION_PATH}")
    if len(text) > max_chars:
        raise InstructionRefused(
            f"the instruction text has {len(text)} characters, more than the host's {max_chars} for one "
            f"system-prompt section; the host would leave it out whole: {INSTRUCTION_PATH}"
        )
    return text


SYSTEM_LEVEL_ROLES = ("system", "developer")


def sent_system_prompt(system_prompt: Any, request_messages: Any) -> Optional[str]:
    """The system-level prompt of one request as the host sent it, as one text; None when
    the request carries none.

    The hook's ``system_prompt`` is the request's own ``system`` (Anthropic Messages, a
    string or text blocks; Bedrock Converse, ``{"text"}`` blocks and a ``cachePoint``) or
    ``instructions`` (Codex Responses), else the first request message when its role is
    ``system`` (``agent/conversation_loop.py:512``). Chat Completions sends GPT-5 and Codex
    models that message with the role ``developer`` (``agent/transports/chat_completions.py:
    321``), which the hook's field does not recognise, so the first message of
    ``request_messages``, the request's own list, is read for either role.

    Where the prompt is text blocks, their texts are joined with nothing between them, as
    the host joins them back (``agent/prompt_caching.py:145``, ``:173``): the host splits
    one prompt string into a static prefix and the rest to mark them for its cache, so the
    same prompt is the same text, and the same key, in either shape."""
    value = system_prompt
    if value is None and isinstance(request_messages, list) and request_messages:
        first = request_messages[0]
        if isinstance(first, dict) and first.get("role") in SYSTEM_LEVEL_ROLES:
            value = first.get("content")
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        # A block without text is skipped, never joined: Bedrock's ``{"cachePoint": ...}``
        # carries none, and neither would any non-text block.
        texts: list[str] = []
        for block in value:
            part = block.get("text") if isinstance(block, dict) else None
            if isinstance(part, str):
                texts.append(part)
        return "".join(texts) if texts else None
    return None


# The host's runtimes whose requests never pass pre_api_request: the whole turn goes to a
# subprocess before any request is assembled (``agent/conversation_loop.py:1581-1582``),
# which is sent the prompt as ``developerInstructions`` (``agent/codex_runtime.py:490``).
UNCHECKED_API_MODES = ("codex_app_server",)


def note_unverifiable_delivery(engine: Any, api_mode: str) -> None:
    """On a runtime whose requests the plugin never sees, record that the instruction's
    delivery cannot be verified: a WARNING and an event, once per engine copy (#16). The
    runtime is the ``api_mode`` the host hands the engine in ``update_model``, at agent
    creation and at every switch of model or provider; at creation it arrives before the
    copy is bound to a host session (``agent/agent_init.py:1971``, ``:1994``), so the note
    waits for the binding, which calls this again. Whether the plugin should serve such a
    runtime at all is not decided here (#24)."""
    host_session_id = str(getattr(engine, "_session_id", "") or "")
    if (api_mode not in UNCHECKED_API_MODES or not host_session_id
            or not _first_sight(engine, f"api_mode {api_mode}")):
        return
    fact = f"delivery of the instruction cannot be verified: api_mode {api_mode} bypasses pre_api_request"
    logger.warning("LCM: %s (host session %s)", fact, host_session_id)
    try:
        engine._records.event_deferred("instruction_not_delivered",
                                       detail={"host_session_id": host_session_id, "fact": fact})
    except Exception:
        logger.error("LCM could not hand over the store event for: %s", fact, exc_info=True)


def _sections_present(prompt: str) -> tuple:
    """The plugin sections the host framed into a prompt, as the host itself parses them
    back on a resume; () where the host's parser is not there or finds none."""
    try:
        from agent.system_prompt import _restore_plugin_prompt_sections  # type: ignore
    except Exception:
        return ()
    try:
        return tuple(_restore_plugin_prompt_sections(prompt))
    except Exception:
        return ()


# Guards each engine copy's set of the prompts it has checked; held only for a set lookup.
# It lives in the process state that outlives a reload of the plugin, as the engine
# copies whose sets it guards do (``engine_registry``).
_SEEN_LOCK = PROCESS_STATE.seen_lock
_NO_SYSTEM_PROMPT = "no system-level prompt"


def note_unbound_session(host_session_id: str, record: Callable[..., None]) -> None:
    """The hook ran for a host session no LCM engine copy is bound to, while the host's
    ``context.engine`` names ``lcm``: the plugin cannot check that session's prompt, and
    records exactly that, once per host session id in this process (#16).

    A session counts as noted only once its event is written: until then it is pending,
    so that a request in flight does not record it twice, and an event whose store was
    closed first waits in the process for the next store opened on its database
    (``record_store``), where it is written and the session noted."""
    if not host_session_id:
        return
    with _SEEN_LOCK:
        if (host_session_id in PROCESS_STATE.unbound_sessions_noted
                or host_session_id in PROCESS_STATE.unbound_sessions_pending):
            return
        PROCESS_STATE.unbound_sessions_pending.add(host_session_id)
    if _configured_context_engine() != "lcm":
        _mark_unbound_noted(host_session_id)
        return
    fact = ("the plugin cannot check that its instruction reached this session: the host's context.engine is "
            "lcm, and no LCM engine copy is bound to this host session id")
    logger.warning("LCM: %s (host session %s)", fact, host_session_id)
    handed = record("instruction_not_delivered", {"host_session_id": host_session_id, "fact": fact},
                    on_written=lambda: _mark_unbound_noted(host_session_id))
    if not handed:
        # Not handed over at all: the next request of this session tries again.
        with _SEEN_LOCK:
            PROCESS_STATE.unbound_sessions_pending.discard(host_session_id)


def _mark_unbound_noted(host_session_id: str) -> None:
    with _SEEN_LOCK:
        PROCESS_STATE.unbound_sessions_pending.discard(host_session_id)
        PROCESS_STATE.unbound_sessions_noted.add(host_session_id)


def _configured_context_engine() -> str:
    """``context.engine`` as the host reads it for each agent it builds
    (``agent/agent_init.py`` ``_select_context_engine``), "" where it cannot be read."""
    try:
        from hermes_cli.config import load_config_readonly  # type: ignore
        cfg = load_config_readonly()
    except Exception:
        return ""
    context = cfg.get("context", {}) if isinstance(cfg, dict) else {}
    return str((context.get("engine", "compressor") if isinstance(context, dict) else "") or "compressor")


def _first_sight(engine: Any, key: str) -> bool:
    """True the first time this engine copy sees ``key`` (a prompt's digest), atomically.
    Every request of the copy's host session id is checked against the same set, a review
    fork's that runs under its parent's id included, so no prompt is recorded twice."""
    with _SEEN_LOCK:
        seen = getattr(engine, "_instruction_prompts_seen", None)
        if seen is None:
            seen = set()
            engine._instruction_prompts_seen = seen
        if key in seen:
            return False
        seen.add(key)
        return True


def check_delivery(engine: Any, system_prompt: Any, request_messages: Any, *, host_session_id: str,
                   text: Optional[str], host: Optional[HostSections], refused: Optional[str],
                   record: Callable[..., None]) -> None:
    """Check one request's system prompt for the plugin's exact section, once per distinct
    prompt of this engine copy (#16). Where it is missing, log a warning and record
    ``instruction_not_delivered`` with what the plugin knows: that it registered no section,
    and why, or that the prompt the host sent does not carry it. What can be seen in that
    prompt, the plugin sections the host's own parser finds there and their sizes, is added
    as an observation; the plugin names no cause it does not know.

    ``text`` is the section's content and ``host`` the host's framing, so the section is
    looked for exactly as the host frames it; ``refused`` is why the plugin registered
    none. ``engine`` is the LCM copy serving the request's session. ``record`` must not
    wait on the store: it runs on the host's hook thread."""
    prompt = sent_system_prompt(system_prompt, request_messages)
    if prompt is None:
        if _first_sight(engine, _NO_SYSTEM_PROMPT):
            fact = "the request the host sent carries no system-level prompt"
            logger.warning("LCM's instruction is not in the request of host session %s: %s", host_session_id, fact)
            record("instruction_not_delivered", {"host_session_id": host_session_id, "fact": fact})
        return
    digest = hashlib.sha256(prompt.encode("utf-8", "surrogatepass")).hexdigest()
    if not _first_sight(engine, digest):
        return
    framed = host.frame(SECTION_ID, text) if host is not None and text else None
    if framed is not None and framed in prompt:
        return
    if refused is not None:
        fact = f"the plugin registered no section: {refused}"
    else:
        fact = "the system prompt the host sent does not carry the plugin's section"
    present = _sections_present(prompt)
    observed = [{"id": section.id, "chars": len(host.frame(section.id, section.content)) if host else
                 len(section.content)} for section in present]
    detail = {"host_session_id": host_session_id, "fact": fact,
              "sections_the_hosts_parser_finds_in_it": observed}
    logger.warning("LCM's instruction is not in the system prompt of host session %s: %s; plugin sections the "
                   "host's parser finds in it: %s", host_session_id, fact,
                   ", ".join(f"{item['id']} ({item['chars']} characters)" for item in observed) or "none")
    record("instruction_not_delivered", detail)
