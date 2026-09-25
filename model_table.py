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
  summariser call can read (the tiny-chunk rule on #52; #34 D4).

#21 adds the session window's bounds and #35's image-counting rules to this table.
Each row says where its values come from.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ModelFacts:
    vendor: str
    model: str
    reads_images: Optional[bool]
    output_cap: Optional[int]
    context_window: Optional[int]
    source: str


ROWS: tuple[ModelFacts, ...] = (
    ModelFacts(
        "anthropic", "claude-opus-5", reads_images=True, output_cap=128_000, context_window=1_000_000,
        source="images: Anthropic docs 2026-09-24 and a measurement via OpenRouter (#9, #35); "
               "output and window: the claude-api reference, cached 2026-06-24 (#31)",
    ),
    ModelFacts(
        "anthropic", "claude-haiku-4.5", reads_images=True, output_cap=None, context_window=200_000,
        source="images: Anthropic docs 2026-09-24 and a measurement via OpenRouter (#9, #35); "
               "window: the claude-api reference (#31); output: not established",
    ),
    ModelFacts(
        "openai", "gpt-6-astra", reads_images=True, output_cap=128_000, context_window=1_050_000,
        source="images: OpenAI docs 2026-09-24 and a measurement via OpenRouter (#9, #35); "
               "output and window: third-party metadata caches, not the provider's own (#31)",
    ),
    ModelFacts(
        "openai", "gpt-5.4-mini", reads_images=True, output_cap=None, context_window=None,
        source="images: OpenAI docs 2026-09-24 and a measurement via OpenRouter (#9, #35)",
    ),
    ModelFacts(
        "openai", "gpt-5-nano", reads_images=True, output_cap=None, context_window=None,
        source="images: OpenAI docs 2026-09-24 (#9, #35); its sizing rule is a hypothesis there",
    ),
)

_BY_ID: dict[str, ModelFacts] = {}
for _row in ROWS:
    _BY_ID[_row.model] = _row
    _BY_ID[f"{_row.vendor}/{_row.model}"] = _row


def lookup(model_id: str) -> Optional[ModelFacts]:
    """The row for a model id exactly as a route names it, or None."""
    return _BY_ID.get(str(model_id or "").strip())
