"""The plugin's own table of what it knows about the models it calls (#9, R17).

"Known, or nothing": a fact the plugin's behaviour rests on comes from here or from the
host's explicit signal, never from the shape of a model's name. A row is found by the
route that calls the model and the model's exact id (9.6): the provider as the host or the
configuration named it (``openrouter``, ``openai-codex``, ``anthropic`` …) and the id as
that route names it (``anthropic/claude-opus-5``, ``claude-opus-5``). A route row holds
what is established for that route. Where there is none, the vendor's documented facts
for the id apply, labelled as the vendor's for that route (the orchestrator's ruling on
9.6): behind a custom URL a proxy's lower limit then fails visibly, as a request the
provider rejects. An id with neither row has no facts, and every column is unknown and
treated as unknown, never guessed. Every lookup says which of these it is (``basis``).

Columns, each ``None`` where not established:

- ``reads_images``: whether the model reads image parts (#8, #35);
- ``output_cap``: the most the model writes in one reply, in provider tokens; passed as
  ``max_tokens`` so that the plugin's call never has a lower limit than the model
  (R5 b); a reply stopped there is a failed summary (#7);
- ``context_window``: the model's whole window in provider tokens, the bound on what one
  summariser call can read (the tiny-chunk rule on #52; #34 D4);
- ``image_rule``: how the provider counts one image, from its documentation (#35,
  #21): ``AnthropicImageRule`` (the tier's edge and token limits, the documented resize,
  28-px visual tokens, plus a constant measured on a route; Anthropic has no ``detail``)
  or ``OpenAIImageRule`` (per ``detail`` level a pixel limit and possibly a resizing patch
  budget, the level ``auto`` stands for, and the model's multiplier; the algorithm is in
  ``tokens``). Where no rule is documented an image is not counted, and the estimate
  says how many it left uncounted;
- ``encrypted_reasoning``: the kinds of encrypted or signed reasoning items, by the host
  field that carries them (``reasoning_details``, ``codex_reasoning_items``,
  ``anthropic_content_blocks``, ``bedrock_content_blocks``), that the model on this route
  takes as input. Read today only to name what is withheld from the summariser (#8);
  what it may be given waits for the producer's stamp (A-P, R3). Where the item was
  produced matters too: the source of each row says what is known about that.

- ``effort``: how a request on this route carries the reasoning effort (``EffortRule``),
  from the provider's own documentation, cited with its date. The session's route
  takes the effort through the host's ``reasoning_config``, so nothing in this build
  reads the column: it is the documented mapping per row, kept for the session's
  provider rows (the orchestrator's ruling on the Codex review of 40eda93; a configured
  summariser, #68, would send it).

A reasoning-summaries column is not built: readable reasoning reaches every summariser as
text (R2), so nothing would read it (the orchestrator's ruling on 9.6).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional, Union


@dataclass(frozen=True)
class AnthropicImageRule:
    max_edge_px: int
    max_tokens: int
    measured_constant: int


@dataclass(frozen=True)
class OpenAISizing:
    """One ``detail`` level's sizing: the pixel limit on either side, and the resizing
    patch budget where the level has one (``None``: no resize to a budget)."""
    max_edge_px: int
    patch_budget: Optional[int]


@dataclass(frozen=True)
class OpenAIImageRule:
    """The documented sizing per ``detail`` level (OpenAI, "Images and vision", model
    sizing behavior, 2026-09-25), the level ``auto`` behaves as, and the multiplier."""
    sizing: tuple[tuple[str, OpenAISizing], ...]
    auto_as: str
    multiplier: float

    def for_detail(self, detail: str) -> Optional[OpenAISizing]:
        """The sizing of a part's ``detail``; ``auto`` or none is the model's default.
        A level the model does not support has no sizing (the image is uncounted)."""
        level = (detail or "auto").strip().lower()
        if level == "auto":
            level = self.auto_as
        return dict(self.sizing).get(level)


ImageRule = Union[AnthropicImageRule, OpenAIImageRule]


@dataclass(frozen=True)
class ModelFacts:
    vendor: str
    model: str
    reads_images: Optional[bool]
    output_cap: Optional[int]
    context_window: Optional[int]
    source: str
    image_rule: Optional[ImageRule] = None
    # The route this row is for (the provider as the host names it); "" for the vendor's
    # documented facts, which apply where no route row exists.
    route: str = ""
    # Every id the route names the model by.
    ids: tuple[str, ...] = ()
    encrypted_reasoning: frozenset = field(default_factory=frozenset)
    # How a request on this route carries the reasoning effort, from the provider's own
    # documentation (``EffortRule``); None where it is not documented or not read.
    effort: Optional["EffortRule"] = None
    # How a lookup found this row (set by ``lookup``).
    basis: str = ""


@dataclass(frozen=True)
class EffortRule:
    """The request fields that carry a reasoning effort on one wire, the levels the
    provider documents, and where that is written. ``form`` is "openrouter_reasoning"
    (``reasoning: {effort}`` in the body) or "anthropic_output_config" (``thinking`` and
    ``output_config.effort``)."""

    wire: str
    form: str
    levels: tuple[str, ...]
    source: str

    def fields(self, level: str) -> Optional[dict]:
        """The request fields for ``level``, or None where the level is not documented."""
        if level not in self.levels:
            return None
        if self.form == "openrouter_reasoning":
            return {"extra_body": {"reasoning": {"effort": level}}}
        if self.form == "anthropic_output_config":
            if level == "none":
                return {"thinking": {"type": "disabled"}}
            return {"thinking": {"type": "adaptive"}, "output_config": {"effort": level}}
        return None


# OpenRouter's unified reasoning parameter ("Reasoning Tokens", "Reasoning Effort Level",
# openrouter.ai/docs/use-cases/reasoning-tokens, read 2026-09-26): max, xhigh, high,
# medium, low, minimal, none; each table model lists "reasoning" among its
# supported_parameters in the models API (read 2026-09-26).
_OPENROUTER_EFFORT = EffortRule(
    wire="chat_completions", form="openrouter_reasoning",
    levels=("max", "xhigh", "high", "medium", "low", "minimal", "none"),
    source="OpenRouter, Reasoning Tokens, Reasoning Effort Level (read 2026-09-26); models API supported_parameters "
           "'reasoning' (read 2026-09-26)")
# Claude Opus 5 (the claude-api reference, cached 2026-06-24, "Thinking & Effort"): adaptive
# thinking with output_config.effort low to max; {type: "disabled"} accepted at effort high
# or below.
_OPUS_5_EFFORT = EffortRule(
    wire="anthropic_messages", form="anthropic_output_config",
    levels=("low", "medium", "high", "xhigh", "max", "none"),
    source="Anthropic, the claude-api reference (cached 2026-06-24), Thinking & Effort: Claude Opus 5, "
           "output_config.effort low-max, adaptive thinking, disabled accepted")


# Anthropic's own models (the claude-api reference, cached 2026-06-24: models.md for the
# windows and output caps; model-migration.md for thinking blocks). Its thinking blocks
# are signed; "regular thinking blocks aren't origin-locked - they replay across models"
# on Anthropic's API, but blocks of Claude Fable 5.1 are read only by Fable 5.1 and
# Mythos 5.1, and those of Claude Opus 5.5 and Fable 5.1 are bound to the conversation
# that produced them, so a summariser's request (another conversation) may be refused.
_ANTHROPIC_THINKING = ("thinking blocks: the claude-api reference (model-migration.md, cached 2026-06-24); "
                       "replayable across Claude models except Fable 5.1 / Opus 5.5 blocks, bound to model "
                       "and conversation")
# OpenAI's Responses reasoning items: the host replays encrypted_content only to the
# issuer and the model that produced it ("encrypted_content is sealed to its issuer and
# model", agent/codex_responses_adapter.py _replay_reasoning_items at Hermes 916e1688ba).
# OpenAI's own documentation was not read.
_OPENAI_ITEMS = ("reasoning items: the host's reading (_replay_reasoning_items: sealed to issuer and model); "
                 "OpenAI's documentation not read; effort: not established, OpenAI's documentation cannot be "
                 "reached from where this was written (network limited to gh, git and OpenRouter)")
# OpenRouter's reasoning_details are passed back unmodified, for Anthropic and OpenAI
# models alike ("Preserving Reasoning", openrouter.ai/docs/use-cases/reasoning-tokens,
# read 2026-09-26); the host keeps reasoning_details on OpenRouter's wire only.
_OPENROUTER_DETAILS = ("reasoning_details: OpenRouter's \"Preserving Reasoning\" (read 2026-09-26); the host "
                       "keeps them on OpenRouter's wire")
# OpenRouter's models API (openrouter.ai/api/v1/models, read 2026-09-26): context_length,
# top_provider.max_completion_tokens, architecture.input_modalities.
_OPENROUTER_API = "OpenRouter models API, read 2026-09-26"

_OPUS_5_IMAGES = AnthropicImageRule(max_edge_px=2576, max_tokens=4784, measured_constant=0)
_HAIKU_45_IMAGES = AnthropicImageRule(max_edge_px=1568, max_tokens=1568, measured_constant=0)
_GPT_6_ASTRA_IMAGES = OpenAIImageRule(
    sizing=(("low", OpenAISizing(512, None)), ("high", OpenAISizing(65_535, 2500)),
            ("original", OpenAISizing(65_535, None))),
    auto_as="original", multiplier=1.2)
_GPT_54_MINI_IMAGES = OpenAIImageRule(
    sizing=(("low", OpenAISizing(2048, 6144)), ("high", OpenAISizing(2048, 2500)),
            ("original", OpenAISizing(6000, 10_000))),
    auto_as="high", multiplier=1.2)

ROWS: tuple[ModelFacts, ...] = (
    # --- The vendors' documented facts -------------------------------------------------
    ModelFacts(
        "anthropic", "claude-opus-5", reads_images=True, output_cap=128_000, context_window=1_000_000,
        source="window and output: the claude-api reference, cached 2026-06-24 (#31); images: Anthropic docs "
               "2026-09-24, high-resolution tier (#9, #35), no constant measured on this route; " + _ANTHROPIC_THINKING,
        image_rule=_OPUS_5_IMAGES, ids=("claude-opus-5", "anthropic/claude-opus-5"),
        encrypted_reasoning=frozenset({"anthropic_content_blocks"}), effort=_OPUS_5_EFFORT,
    ),
    ModelFacts(
        "anthropic", "claude-haiku-4.5", reads_images=True, output_cap=64_000, context_window=200_000,
        source="window and output: the claude-api reference (models.md: 200K, 64K), cached 2026-06-24; images: "
               "Anthropic docs 2026-09-24, standard tier (#9, #35), no constant measured on this route; "
               + _ANTHROPIC_THINKING + "; effort: none on Haiku 4.5 (the claude-api reference, Thinking & Effort: "
               "effort errors on Haiku 4.5, thinking takes budget_tokens, which no effort level names)",
        image_rule=_HAIKU_45_IMAGES,
        ids=("claude-haiku-4-5", "anthropic/claude-haiku-4-5", "claude-haiku-4.5", "anthropic/claude-haiku-4.5"),
        encrypted_reasoning=frozenset({"anthropic_content_blocks"}),
    ),
    ModelFacts(
        "openai", "gpt-6-astra", reads_images=True, output_cap=128_000, context_window=1_050_000,
        source="images: OpenAI docs 2026-09-25 (low: within 512 px; high: 65,535 px and a 2,500-patch "
               "budget; original: 65,535 px, no budget, rejected above 30,000 patches; auto as original; "
               "x1.2), measured = rule at original (#9, #35); output and window: third-party metadata "
               "caches, not the provider's own (#31); " + _OPENAI_ITEMS,
        image_rule=_GPT_6_ASTRA_IMAGES, ids=("gpt-6-astra", "openai/gpt-6-astra"),
        encrypted_reasoning=frozenset({"codex_reasoning_items"}),
    ),
    ModelFacts(
        "openai", "gpt-5.4-mini", reads_images=True, output_cap=None, context_window=None,
        source="images: OpenAI docs 2026-09-25 (low: 2,048 px and a 6,144-patch budget; high: 2,048 px "
               "and 2,500; original: 6,000 px and 10,000; auto as high; x1.2), measured = rule at high "
               "(#9, #35); " + _OPENAI_ITEMS,
        image_rule=_GPT_54_MINI_IMAGES, ids=("gpt-5.4-mini", "openai/gpt-5.4-mini"),
        encrypted_reasoning=frozenset({"codex_reasoning_items"}),
    ),
    ModelFacts(
        "openai", "gpt-5-nano", reads_images=True, output_cap=None, context_window=None,
        source="images: OpenAI docs 2026-09-24 (#9, #35); its sizing is not documented, so no rule: "
               "its images are uncounted; " + _OPENAI_ITEMS,
        ids=("gpt-5-nano", "openai/gpt-5-nano"),
        encrypted_reasoning=frozenset({"codex_reasoning_items"}),
    ),
    # --- Routes ------------------------------------------------------------------------
    ModelFacts(
        "anthropic", "claude-opus-5", reads_images=True, output_cap=128_000, context_window=1_000_000,
        source=f"{_OPENROUTER_API}; images: Anthropic's rule, measured via OpenRouter as rule + 3 on eight "
               f"images (#9, #35); {_OPENROUTER_DETAILS}",
        image_rule=replace(_OPUS_5_IMAGES, measured_constant=3), route="openrouter",
        ids=("anthropic/claude-opus-5",), encrypted_reasoning=frozenset({"reasoning_details"}), effort=_OPENROUTER_EFFORT,
    ),
    ModelFacts(
        "anthropic", "claude-haiku-4.5", reads_images=True, output_cap=64_000, context_window=200_000,
        source=f"{_OPENROUTER_API}; images: Anthropic's rule, measured via OpenRouter as rule + 4 (#9, #35); "
               f"{_OPENROUTER_DETAILS}",
        image_rule=replace(_HAIKU_45_IMAGES, measured_constant=4), route="openrouter",
        ids=("anthropic/claude-haiku-4.5",), encrypted_reasoning=frozenset({"reasoning_details"}),
        effort=_OPENROUTER_EFFORT,
    ),
    ModelFacts(
        "openai", "gpt-6-astra", reads_images=True, output_cap=128_000, context_window=1_050_000,
        source=f"{_OPENROUTER_API}; images: OpenAI's rule, measured via OpenRouter = rule (#35); "
               f"{_OPENROUTER_DETAILS}",
        image_rule=_GPT_6_ASTRA_IMAGES, route="openrouter",
        ids=("openai/gpt-6-astra",), encrypted_reasoning=frozenset({"reasoning_details"}), effort=_OPENROUTER_EFFORT,
    ),
    ModelFacts(
        "openai", "gpt-5.4-mini", reads_images=True, output_cap=128_000, context_window=400_000,
        source=f"{_OPENROUTER_API}; images: OpenAI's rule, measured via OpenRouter = rule (#35); "
               f"{_OPENROUTER_DETAILS}",
        image_rule=_GPT_54_MINI_IMAGES, route="openrouter",
        ids=("openai/gpt-5.4-mini",), encrypted_reasoning=frozenset({"reasoning_details"}), effort=_OPENROUTER_EFFORT,
    ),
    ModelFacts(
        "openai", "gpt-5-nano", reads_images=True, output_cap=128_000, context_window=400_000,
        source=f"{_OPENROUTER_API}; images: sizing not documented, uncounted (#35); {_OPENROUTER_DETAILS}",
        route="openrouter", ids=("openai/gpt-5-nano",), encrypted_reasoning=frozenset({"reasoning_details"}),
        effort=_OPENROUTER_EFFORT,
    ),
    ModelFacts(
        "openai", "gpt-6-astra", reads_images=True, output_cap=None, context_window=272_000,
        source="window: the host's Codex OAuth window for the id (agent/model_metadata.py "
               "_CODEX_OAUTH_CONTEXT_FALLBACK at Hermes 916e1688ba; the -900k variant is another id); output: not "
               "established on this route; images: OpenAI's rule (#35); " + _OPENAI_ITEMS,
        image_rule=_GPT_6_ASTRA_IMAGES, route="openai-codex", ids=("gpt-6-astra", "openai/gpt-6-astra"),
        encrypted_reasoning=frozenset({"codex_reasoning_items"}),
    ),
)

_BY_ROUTE: dict[tuple[str, str], ModelFacts] = {}
_BY_ID: dict[str, ModelFacts] = {}
for _row in ROWS:
    for _id in _row.ids:
        if _row.route:
            _BY_ROUTE[(_row.route, _id)] = _row
        else:
            _BY_ID[_id] = _row


def lookup(model_id: str, provider: str = "") -> Optional[ModelFacts]:
    """The facts for a model id exactly as a route names it, called through ``provider``
    (the provider as the host or the configuration named it): that route's row, else the
    vendor's documented facts for the id, labelled so; None where the table has neither.
    The result's ``basis`` says which."""
    model = str(model_id or "").strip()
    route = str(provider or "").strip().lower()
    row = _BY_ROUTE.get((route, model)) if route else None
    if row is not None:
        return replace(row, basis=f"the table's row for {model} on {route}")
    row = _BY_ID.get(model)
    if row is None:
        return None
    if route == row.vendor:
        return replace(row, basis=f"{row.vendor}'s documented facts for {model}, on {row.vendor}'s own API")
    where = f"on {route}" if route else "on a route not named"
    return replace(row, basis=f"{row.vendor}'s documented facts for {model}, not established for this route "
                              f"({where}): a lower limit of the route fails visibly")
