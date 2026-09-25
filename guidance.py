"""The plugin's own instruction to the agent (#16), and the skills it registers.

The instruction is told globally, not per summary, and the plugin delivers it itself:
as a section of the host's system prompt (``register_system_prompt_section``), which the
host renders once per session and again after every compaction, never cut, and never
as text the user seems to have written. Its words are the prompt step's (#10); until
then the text is ``skills/hermes-lcm/references/recall-policy.md``.

The host leaves a section over its limit out whole, with a warning in its own log
(``hermes_cli/plugins_dispatch.py`` at Hermes 7b761da: 4,000 characters per section).
So the plugin checks its text against that limit itself and refuses it visibly.
"""

from pathlib import Path

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

try:  # the host's limit for one plugin section
    from hermes_cli.plugins_dispatch import MAX_SYSTEM_PROMPT_SECTION_CHARS as SECTION_MAX_CHARS  # type: ignore
except Exception:  # pragma: no cover - as at Hermes 7b761da, hermes_cli/plugins_dispatch.py:70
    SECTION_MAX_CHARS = 4_000


class InstructionRefused(RuntimeError):
    """The instruction text cannot be delivered as the host's section."""


def instruction_text() -> str:
    """The instruction as the section carries it: the file's text, stripped as the host
    strips a section. Raises ``InstructionRefused`` when it is empty or over the host's
    limit, since the host would leave it out."""
    text = INSTRUCTION_PATH.read_text(encoding="utf-8").strip()
    if not text:
        raise InstructionRefused(f"the instruction text is empty: {INSTRUCTION_PATH}")
    if len(text) > SECTION_MAX_CHARS:
        raise InstructionRefused(
            f"the instruction text has {len(text)} characters, more than the host's {SECTION_MAX_CHARS} for one "
            f"system-prompt section; the host would leave it out whole: {INSTRUCTION_PATH}"
        )
    return text
