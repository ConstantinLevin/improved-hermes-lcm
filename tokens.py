"""The plugin's token estimate (#21, R9): characters divided by four, over what the
provider receives, labelled an estimate wherever it is shown.

The plugin knowingly does not know the number of tokens; it counts by this estimate and
says so ("Known, or nothing"). For a message, the characters are those the provider
receives:

- the content, with the ``api_content`` sidecar in its place for a user or assistant
  row where the host sends the sidecar (``substitute_api_content``); a list's text
  parts by their text, any other non-image part by its JSON; the ``_multimodal``
  envelope by its parts;
- each tool call's function name and arguments;
- ``reasoning_content`` only where the provider receives it (``Estimator.reasoning_sent``,
  the host's ``needs_reasoning_echo`` for the route; what the estimate must be good for
  with reasoning is #21's open item 2).

An image is not counted by the length of its data: it is counted by the rule of the
model the count concerns, from the plugin's model table (#35), reading its size from
its header; an image with no rule or no readable size is left out and counted as
"uncounted". The count is ``ceil(characters / 4)`` plus the images' tokens.

On Claude, for tool results, the provider's count was 1.51 times this estimate at p50
and 2.37 at p99 (#31); a value compared with provider tokens states its conversion
where it is used.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

from .message_content import content_parts, is_image_part

CHARS_PER_TOKEN = 4
ESTIMATE_LABEL = "an estimate: characters / 4, images by the model table's rule"


@dataclass(frozen=True)
class Estimate:
    tokens: int
    uncounted_images: int = 0

    def __add__(self, other: "Estimate") -> "Estimate":
        return Estimate(self.tokens + other.tokens, self.uncounted_images + other.uncounted_images)

    def label(self) -> str:
        """The estimate's label, as it is shown."""
        if self.uncounted_images:
            return f"{ESTIMATE_LABEL}; {self.uncounted_images} images uncounted"
        return ESTIMATE_LABEL


def count_tokens(text: Any) -> int:
    """``ceil(characters / 4)`` of a text; a non-string value by its JSON."""
    if not text:
        return 0
    if not isinstance(text, str):
        try:
            text = json.dumps(text, ensure_ascii=False)
        except (TypeError, ValueError):
            text = str(text)
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def _anthropic_image_tokens(rule, width: int, height: int) -> int:
    scale = min(1.0, rule.max_edge_px / max(width, height))
    w, h = width * scale, height * scale
    tiles = math.ceil(w / 28) * math.ceil(h / 28)
    if tiles > rule.max_tokens:
        factor = math.sqrt(rule.max_tokens / tiles)
        w, h = w * factor, h * factor
        tiles = min(rule.max_tokens, math.ceil(w / 28) * math.ceil(h / 28))
    return tiles + rule.measured_constant


def _openai_image_tokens(rule, width: int, height: int) -> int:
    w, h = float(width), float(height)
    if rule.max_edge_px:
        scale = min(1.0, rule.max_edge_px / max(w, h))
        w, h = w * scale, h * scale
    patches = math.ceil(w / 32) * math.ceil(h / 32)
    if patches > rule.patch_budget:
        factor = math.sqrt(rule.patch_budget / patches)
        w, h = w * factor, h * factor
        patches = min(rule.patch_budget, math.ceil(w / 32) * math.ceil(h / 32))
    return math.ceil(patches * rule.multiplier)


@dataclass(frozen=True)
class Estimator:
    """The estimate for one model's context: ``image_model`` names whose image rule
    counts images; ``reasoning_sent`` whether that provider receives
    ``reasoning_content``."""

    image_model: str = ""
    reasoning_sent: bool = False

    def image(self, part: dict) -> Optional[int]:
        """One image's tokens by the model's rule, or None (uncounted)."""
        from .image_size import image_dimensions
        from .model_table import AnthropicImageRule, OpenAIImageRule, lookup

        facts = lookup(self.image_model)
        rule = facts.image_rule if facts is not None else None
        if rule is None:
            return None
        size = image_dimensions(part)
        if size is None:
            return None
        if isinstance(rule, AnthropicImageRule):
            return _anthropic_image_tokens(rule, *size)
        if isinstance(rule, OpenAIImageRule):
            return _openai_image_tokens(rule, *size)
        return None

    def _content(self, content: Any) -> tuple[int, int, int]:
        """(characters, image tokens, uncounted images) of a content value."""
        if content is None:
            return 0, 0, 0
        if isinstance(content, str):
            return len(content), 0, 0
        parts = content_parts(content)
        if parts is None:
            return len(json.dumps(content, ensure_ascii=False, default=str)), 0, 0
        chars = images = uncounted = 0
        for part in parts:
            if is_image_part(part):
                tokens = self.image(part)
                if tokens is None:
                    uncounted += 1
                else:
                    images += tokens
            elif isinstance(part, dict) and part.get("type") in ("text", "input_text", "output_text") \
                    and isinstance(part.get("text"), str):
                chars += len(part["text"])
            elif isinstance(part, str):
                chars += len(part)
            else:
                chars += len(json.dumps(part, ensure_ascii=False, default=str))
        return chars, images, uncounted

    def message(self, message: Dict[str, Any]) -> Estimate:
        if not isinstance(message, dict):
            return Estimate(0)
        content = message.get("content")
        sidecar = message.get("api_content")
        if isinstance(sidecar, str) and sidecar and message.get("role") in ("user", "assistant"):
            content = sidecar
        chars, image_tokens, uncounted = self._content(content)
        for call in message.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            function = call.get("function") if isinstance(call.get("function"), dict) else {}
            chars += len(str(function.get("name") or ""))
            arguments = function.get("arguments")
            chars += len(arguments) if isinstance(arguments, str) else len(
                json.dumps(arguments, ensure_ascii=False, default=str)) if arguments else 0
        reasoning = message.get("reasoning_content")
        if self.reasoning_sent and isinstance(reasoning, str):
            chars += len(reasoning)
        return Estimate(math.ceil(chars / CHARS_PER_TOKEN) + image_tokens, uncounted)

    def messages(self, messages: Iterable[Dict[str, Any]]) -> Estimate:
        total = Estimate(0)
        for message in messages:
            total = total + self.message(message)
        return total


_PLAIN = Estimator()


def count_message_tokens(message: Dict[str, Any], estimator: Optional[Estimator] = None) -> int:
    """A message's estimate in tokens (see the module docstring)."""
    return (estimator or _PLAIN).message(message).tokens


def count_messages_tokens(messages: Iterable[Dict[str, Any]], estimator: Optional[Estimator] = None) -> int:
    """A list's estimate in tokens (see the module docstring)."""
    return (estimator or _PLAIN).messages(messages).tokens
