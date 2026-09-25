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
  ``render_system_prompt_sections``).

So the plugin checks its text against the host's limit itself and refuses it visibly, and
it checks every distinct system prompt the host sends for its exact section
(:func:`check_delivery`), recording ``instruction_not_delivered`` where it is missing.
The host's limits and its framing are read from the host, never assumed.
"""

from __future__ import annotations

import hashlib
import logging
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
    """The host's own limits and framing for plugin sections, read from the host."""

    def __init__(self) -> None:
        try:
            from hermes_cli.plugins_dispatch import (  # type: ignore
                MAX_SYSTEM_PROMPT_SECTION_CHARS,
                MAX_SYSTEM_PROMPT_SECTIONS_TOTAL_CHARS,
                format_system_prompt_section,
                format_system_prompt_sections,
            )
        except Exception as exc:
            raise InstructionRefused(
                f"the host's limits and framing for a system-prompt section cannot be read "
                f"({type(exc).__name__}: {exc}); the plugin does not assume them"
            ) from exc
        self.max_chars: int = int(MAX_SYSTEM_PROMPT_SECTION_CHARS)
        self.total_chars: int = int(MAX_SYSTEM_PROMPT_SECTIONS_TOTAL_CHARS)
        self.frame: Callable[[str, str], str] = format_system_prompt_section
        self.frame_all: Callable[[list], str] = format_system_prompt_sections


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


def _prompt_texts(system_prompt: Any) -> Optional[list[str]]:
    """The system prompt's text as the request carries it: a string, or the text of each
    content block (a provider that takes the system prompt as blocks). None when the
    request carries no system prompt the plugin can read."""
    if isinstance(system_prompt, str):
        return [system_prompt]
    if isinstance(system_prompt, list):
        texts = [block.get("text") for block in system_prompt if isinstance(block, dict)]
        texts = [text for text in texts if isinstance(text, str)]
        return texts or None
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


def check_delivery(engine: Any, system_prompt: Any, *, host_session_id: str, text: Optional[str],
                   host: Optional[HostSections], refused: Optional[str], record: Callable[..., None]) -> None:
    """Check one request's system prompt for the plugin's exact section, once per distinct
    prompt of this engine copy (#16). Where it is missing, record ``instruction_not_delivered``
    with the cause the plugin knows, and log a warning.

    ``text`` is the section's content and ``host`` the host's framing, so the section is
    looked for exactly as the host frames it; ``refused`` is why the plugin registered
    none. ``engine`` is the LCM copy serving the request's session."""
    framed = host.frame(SECTION_ID, text) if host is not None and text else None
    texts = _prompt_texts(system_prompt)
    if texts is None:
        if not getattr(engine, "_instruction_prompt_unreadable_logged", False):
            engine._instruction_prompt_unreadable_logged = True
            logger.warning(
                "LCM cannot check that its instruction reached the agent: the request of host session %s "
                "carries no system prompt the plugin can read (%s)", host_session_id, type(system_prompt).__name__)
        return
    digest = hashlib.sha256("\x00".join(texts).encode("utf-8", "surrogatepass")).hexdigest()
    if getattr(engine, "_instruction_checked_prompt", None) == digest:
        return
    engine._instruction_checked_prompt = digest
    if framed is not None and any(framed in text for text in texts):
        return
    if refused is not None:
        cause = f"the plugin registered no section: {refused}"
    elif not getattr(engine, "_instruction_rendered", 0):
        cause = ("the host sent a system prompt no render of the section reached: it reused a prompt built "
                 "earlier, as it does when it restores a continuing session from its database (a prompt "
                 "stored before the section existed stays without it until the next compaction) or when a "
                 "review fork takes its parent's prompt")
    else:
        present = _sections_present("\n\n".join(texts))
        with_this = len(host.frame_all([*present, _Section(SECTION_ID, text)])) if host and text else 0
        if host is not None and present and with_this > host.total_chars:
            others = sum(len(host.frame(section.id, section.content)) for section in present)
            cause = (f"the host left it out for its total of {host.total_chars} characters for all plugin "
                     f"sections: the sections present ({', '.join(section.id for section in present)}) take "
                     f"{others}, and with this one the container would take {with_this}")
        else:
            cause = ("the section was rendered for this session and the host left it out; the causes at "
                     "Hermes 34343e7 are its total for all plugin sections and its count of 32 sections")
    detail = {"host_session_id": host_session_id, "cause": cause}
    logger.warning("LCM's instruction is not in the system prompt of host session %s: %s", host_session_id, cause)
    record("instruction_not_delivered", detail)


class _Section:
    """A section as the host's framing reads one (``id``, ``content``)."""

    def __init__(self, id: str, content: str) -> None:
        self.id = id
        self.content = content
