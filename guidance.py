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
  way, and the previous bytes are kept only when the whole render raises;
- a plugin's hook and section callbacks are stored as given and run in their caller's
  home scope (``hermes_cli/plugins.py`` ``register_hook``; ``hermes_cli/plugins_dispatch.py``
  ``invoke_hook``); the host reads a manager's own config under an explicit scope of that
  manager's home (``plugins.py`` ``_tool_override_allowed``).

So the section's text is static and is given where the ``context.engine`` of this load's
own home, captured when it was registered, is ``lcm``. The plugin checks the text against
the host's limit itself and refuses it visibly, and :class:`DeliveryCheck` looks for the
exact section in every distinct system prompt the host sends, logging a warning where it
is missing: the facts it knows, never a cause it does not. The host's limit and framing
are read from the host, never assumed.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from pathlib import Path
from typing import Any, Callable, Optional

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


def configured_context_engine(hermes_home: str) -> Optional[str]:
    """``context.engine`` of the Hermes home ``hermes_home``, as the host reads it for each
    agent it builds (``agent/agent_init.py`` ``_select_context_engine``), read under an
    explicit override of that home for this thread's context only
    (``hermes_constants.set_hermes_home_override``, what the host's ``_plugin_home_scope``
    sets); None where it cannot be read."""
    if not hermes_home:
        return None
    try:
        from hermes_cli.config import load_config_readonly  # type: ignore
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override  # type: ignore
    except Exception:
        return None
    token = set_hermes_home_override(hermes_home)
    try:
        cfg = load_config_readonly()
    except Exception:
        return None
    finally:
        reset_hermes_home_override(token)
    context = cfg.get("context", {}) if isinstance(cfg, dict) else {}
    return str((context.get("engine", "compressor") if isinstance(context, dict) else "") or "compressor")


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


class DeliveryCheck:
    """One load's check that its section reached the request (#16).

    It belongs to one registration: its home, captured when the load was registered, its
    text and the host's framing, or why the load registered no section, and the set of the
    prompts it has checked, under a lock held only for the set operation. A reload
    registers a new one and the unload removes the old one's hook with it; nothing is
    shared between loads, homes or processes. It writes nothing to the store: where the
    section is missing it logs a warning in the host's log, once per distinct prompt, and
    how the status view shows such warnings is #18's."""

    def __init__(self, hermes_home: str, *, text: Optional[str], host: Optional[HostSections],
                 refused: Optional[str]) -> None:
        self.hermes_home = hermes_home
        self.text = text
        self.host = host
        self.refused = refused
        self._lock = threading.Lock()
        self._seen: set[str] = set()

    def section(self, info: Any) -> str:
        """The section's content for the host's render: the static text where this load's
        home names ``lcm`` as its context engine, else "" (which the host skips)."""
        if self.text is None or configured_context_engine(self.hermes_home) != "lcm":
            return ""
        return self.text

    def _first_sight(self, key: str) -> bool:
        with self._lock:
            if key in self._seen:
                return False
            self._seen.add(key)
            return True

    def check(self, host_session_id: str, system_prompt: Any, request_messages: Any) -> None:
        """Look for the section in one request's system-level prompt, as it was sent."""
        prompt = sent_system_prompt(system_prompt, request_messages)
        key = (hashlib.sha256(prompt.encode("utf-8", "surrogatepass")).hexdigest() if prompt is not None
               else f"no system-level prompt {host_session_id}")
        if not self._first_sight(key):
            return
        framed = self.host.frame(SECTION_ID, self.text) if self.host is not None and self.text else None
        if prompt is not None and framed is not None and framed in prompt:
            return
        configured = configured_context_engine(self.hermes_home)
        if configured is not None and configured != "lcm":
            return  # the section is given only where this home's context engine is LCM
        where = (f"; the context.engine of {self.hermes_home or 'this load’s home'} cannot be read"
                 if configured is None else "")
        if prompt is None:
            logger.warning("LCM's instruction is not in the request of host session %s: the request the host "
                           "sent carries no system-level prompt%s", host_session_id, where)
            return
        if self.refused is not None:
            fact = f"the plugin registered no section: {self.refused}"
        else:
            fact = "the system prompt the host sent does not carry the plugin's section"
        observed = ", ".join(
            f"{section.id} ({len(self.host.frame(section.id, section.content)) if self.host else len(section.content)}"
            f" characters)" for section in _sections_present(prompt)) or "none"
        logger.warning("LCM's instruction is not in the system prompt of host session %s: %s%s; plugin sections "
                       "the host's parser finds in it: %s", host_session_id, fact, where, observed)


# The host's runtimes whose requests never pass pre_api_request: the whole turn goes to a
# subprocess before any request is assembled (``agent/conversation_loop.py:1581-1582``),
# which is sent the prompt as ``developerInstructions`` (``agent/codex_runtime.py:490``).
UNCHECKED_API_MODES = ("codex_app_server",)
_UNVERIFIABLE_LOCK = threading.Lock()


def warn_unverifiable_delivery(engine: Any, api_mode: str) -> None:
    """On a runtime whose requests the plugin never sees, log that the instruction's
    delivery cannot be verified, once per engine copy (#16). The runtime is the
    ``api_mode`` the host hands the engine in ``update_model``, at agent creation and at
    every switch of model or provider. Whether the plugin should serve such a runtime at
    all is not decided here (#24)."""
    if api_mode not in UNCHECKED_API_MODES:
        return
    with _UNVERIFIABLE_LOCK:
        if getattr(engine, "_instruction_unverifiable_warned", False):
            return
        engine._instruction_unverifiable_warned = True
    logger.warning("LCM: delivery of the instruction cannot be verified: api_mode %s bypasses pre_api_request "
                   "(model %s, host session %s)", api_mode, getattr(engine, "model", ""),
                   getattr(engine, "_session_id", "") or "not yet bound")
