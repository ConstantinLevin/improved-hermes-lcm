"""The plugin's own table of what it knows about the models it calls (#9, R17).

"Known, or nothing": a fact the plugin's behaviour rests on comes from here or from the
host's explicit signal, never from the shape of a model's name. A model is found only
by its exact id, bare (``claude-opus-5``) or with its vendor (``anthropic/claude-opus-5``),
the way a route names it; a model not in the table has no row, and every column that
needs a row is unknown and treated as unknown, never guessed.

Columns, each ``None`` where not established:

- ``reads_images``: whether the model reads image parts (#8, #35);
- ``output_cap``: the most the model writes in one reply, in provider tokens; passed as
  ``max_tokens`` so that the plugin's call never has a lower limit than the model
  (R5 b); a reply stopped there is a failed summary (#7);
- ``context_window``: the model's whole window in provider tokens, the bound on what one
  summariser call can read (the tiny-chunk rule on #52; #34 D4);
- ``image_rule``: how the provider counts one image, from its documentation (#35,
  #21): ``AnthropicImageRule`` (the tier's edge and token limits, the documented resize,
  28-px visual tokens, plus a measured constant; Anthropic has no ``detail``) or
  ``OpenAIImageRule`` (per ``detail`` level a pixel limit and possibly a resizing patch
  budget, the level ``auto`` stands for, and the model's multiplier; the algorithm is in
  ``tokens``). Where no rule is documented an image is not counted, and the estimate
  says how many it left uncounted.

#21 adds the session window's bounds to this table. Each row says where its values
come from.
"""

from __future__ import annotations

from dataclasses import dataclass
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


ROWS: tuple[ModelFacts, ...] = (
    ModelFacts(
        "anthropic", "claude-opus-5", reads_images=True, output_cap=128_000, context_window=1_000_000,
        source="images: Anthropic docs 2026-09-24 (high-resolution tier) and a measurement via OpenRouter, "
               "rule + 3 on eight images (#9, #35); output and window: the claude-api reference, "
               "cached 2026-06-24 (#31)",
        image_rule=AnthropicImageRule(max_edge_px=2576, max_tokens=4784, measured_constant=3),
    ),
    ModelFacts(
        "anthropic", "claude-haiku-4.5", reads_images=True, output_cap=None, context_window=200_000,
        source="images: Anthropic docs 2026-09-24 (standard tier) and a measurement via OpenRouter, "
               "rule + 4 (#9, #35); window: the claude-api reference (#31); output: not established",
        image_rule=AnthropicImageRule(max_edge_px=1568, max_tokens=1568, measured_constant=4),
    ),
    ModelFacts(
        "openai", "gpt-6-astra", reads_images=True, output_cap=128_000, context_window=1_050_000,
        source="images: OpenAI docs 2026-09-25 (low: within 512 px; high: 65,535 px and a 2,500-patch "
               "budget; original: 65,535 px, no budget, rejected above 30,000 patches; auto as original; "
               "x1.2), measured = rule at original (#9, #35); output and window: third-party metadata "
               "caches, not the provider's own (#31)",
        image_rule=OpenAIImageRule(
            sizing=(("low", OpenAISizing(512, None)), ("high", OpenAISizing(65_535, 2500)),
                    ("original", OpenAISizing(65_535, None))),
            auto_as="original", multiplier=1.2),
    ),
    ModelFacts(
        "openai", "gpt-5.4-mini", reads_images=True, output_cap=None, context_window=None,
        source="images: OpenAI docs 2026-09-25 (low: 2,048 px and a 6,144-patch budget; high: 2,048 px "
               "and 2,500; original: 6,000 px and 10,000; auto as high; x1.2), measured = rule at high "
               "(#9, #35)",
        image_rule=OpenAIImageRule(
            sizing=(("low", OpenAISizing(2048, 6144)), ("high", OpenAISizing(2048, 2500)),
                    ("original", OpenAISizing(6000, 10_000))),
            auto_as="high", multiplier=1.2),
    ),
    ModelFacts(
        "openai", "gpt-5-nano", reads_images=True, output_cap=None, context_window=None,
        source="images: OpenAI docs 2026-09-24 (#9, #35); its sizing is not documented, so no rule: "
               "its images are uncounted",
    ),
)

_BY_ID: dict[str, ModelFacts] = {}
for _row in ROWS:
    _BY_ID[_row.model] = _row
    _BY_ID[f"{_row.vendor}/{_row.model}"] = _row


def lookup(model_id: str) -> Optional[ModelFacts]:
    """The row for a model id exactly as a route names it, or None."""
    return _BY_ID.get(str(model_id or "").strip())
