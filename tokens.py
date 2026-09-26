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
its header and, on OpenAI, the part's own ``detail``; the algorithms are the providers'
documented ones, cited where they are implemented. An image with no rule, no readable
size or a ``detail`` its model does not support is left out and counted as "uncounted". The count is ``ceil(characters / 4)`` plus the images' tokens.

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
    # The part of ``tokens`` counted by images' rules: provider tokens already, so a
    # conversion of the estimate into provider tokens applies only to the rest.
    image_tokens: int = 0

    def __add__(self, other: "Estimate") -> "Estimate":
        return Estimate(self.tokens + other.tokens, self.uncounted_images + other.uncounted_images,
                        self.image_tokens + other.image_tokens)

    def in_provider_tokens(self, ratio: float) -> int:
        """The estimate converted into provider tokens: its characters / 4 part times
        ``ratio``, its images as their rule counted them."""
        return math.ceil((self.tokens - self.image_tokens) * ratio) + self.image_tokens

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


def _anthropic_resized_size(width: int, height: int, max_edge: int, max_tokens: int) -> tuple[int, int]:
    """The size Claude resizes an image to before padding: Anthropic's reference
    implementation, as published in "Coordinates and bounding boxes", section "Resize
    your image before uploading"
    (https://platform.claude.com/docs/en/build-with-claude/vision-coordinates, read
    2026-09-25), with the tier's limits from "Vision", "Resolution and token cost"
    (https://platform.claude.com/docs/en/build-with-claude/vision)."""

    def fits(w: int, h: int) -> bool:
        return (
            math.ceil(w / 28) * 28 <= max_edge
            and math.ceil(h / 28) * 28 <= max_edge
            and math.ceil(w / 28) * math.ceil(h / 28) <= max_tokens
        )

    if fits(width, height):
        return (width, height)
    if height > width:
        resized_h, resized_w = _anthropic_resized_size(height, width, max_edge, max_tokens)
        return (resized_w, resized_h)
    # Binary search along the long edge for the largest aspect-preserving size that fits.
    aspect_ratio = width / height
    lo, hi = 1, width  # lo always fits; hi never fits
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if fits(mid, max(round(mid / aspect_ratio), 1)):
            lo = mid
        else:
            hi = mid
    return (lo, max(round(lo / aspect_ratio), 1))


def _anthropic_image_tokens(rule, width: int, height: int) -> int:
    """``ceil(w/28) * ceil(h/28)`` visual tokens of the resized image ("Vision",
    "Resolution and token cost"), plus the constant measured on the route (#35).
    Anthropic's image block has no ``detail``; a part's ``detail`` changes nothing."""
    w, h = _anthropic_resized_size(width, height, rule.max_edge_px, rule.max_tokens)
    return math.ceil(w / 28) * math.ceil(h / 28) + rule.measured_constant


def _openai_image_tokens(rule, width: int, height: int, detail: str = "") -> Optional[int]:
    """OpenAI's patch-based image tokenization, steps A-D as published in "Images and
    vision", section "Calculating costs" / "Patch-based image tokenization"
    (https://developers.openai.com/api/docs/guides/images-vision, read 2026-09-25), with
    the ``detail`` level's limits from the same page's "Model sizing behavior". A level
    the model does not support gives None (uncounted).

    An image above 30,000 patches is rejected by the API, not resized; it is counted as
    computed here, so that the estimate never reads lower than what the list holds."""
    sizing = rule.for_detail(detail)
    if sizing is None:
        return None
    # First, fit within the level's pixel-dimension limit, preserving aspect ratio,
    # rounding to integer pixels, never enlarging. The page says "rounding"; it is read
    # here as rounding to the nearest pixel, halves up.
    w, h = width, height
    if max(w, h) > sizing.max_edge_px:
        scale = sizing.max_edge_px / max(w, h)
        w, h = max(1, math.floor(w * scale + 0.5)), max(1, math.floor(h * scale + 0.5))
    # A. Patches covering the image.
    patches = math.ceil(w / 32) * math.ceil(h / 32)
    # B. and C. Only where the level has a resizing patch budget and the image exceeds it.
    if sizing.patch_budget is not None and patches > sizing.patch_budget:
        shrink = math.sqrt((32 ** 2 * sizing.patch_budget) / (w * h))
        adjusted = shrink * min(
            math.floor(w * shrink / 32) / (w * shrink / 32),
            math.floor(h * shrink / 32) / (h * shrink / 32),
        )
        resized_w, resized_h = math.floor(w * adjusted), math.floor(h * adjusted)
        patches = math.ceil(resized_w / 32) * math.ceil(resized_h / 32)
    # D. The model's multiplier, rounded up.
    return math.ceil(patches * rule.multiplier)


def image_detail(part: dict) -> str:
    """The ``detail`` a part carries: ``image_url.detail`` (Chat Completions) or the
    part's own ``detail`` (Responses ``input_image``); "" when it has none."""
    if not isinstance(part, dict):
        return ""
    inner = part.get("image_url")
    if isinstance(inner, dict) and isinstance(inner.get("detail"), str):
        return inner["detail"]
    return part["detail"] if isinstance(part.get("detail"), str) else ""


@dataclass(frozen=True)
class Estimator:
    """The estimate for one model's context: ``image_model`` names whose image rule
    counts images, on the route ``image_provider`` names (the model table is keyed on
    both, 9.6); ``reasoning_sent`` whether that provider receives
    ``reasoning_content``."""

    image_model: str = ""
    reasoning_sent: bool = False
    image_provider: str = ""

    def image(self, part: dict) -> Optional[int]:
        """One image's tokens by the model's rule, or None (uncounted)."""
        from .image_size import image_dimensions
        from .model_table import AnthropicImageRule, OpenAIImageRule, lookup

        facts = lookup(self.image_model, self.image_provider)
        rule = facts.image_rule if facts is not None else None
        if rule is None:
            return None
        size = image_dimensions(part)
        if size is None:
            return None
        if isinstance(rule, AnthropicImageRule):
            return _anthropic_image_tokens(rule, *size)
        if isinstance(rule, OpenAIImageRule):
            # The part's own detail decides; the model's default only where it has none.
            return _openai_image_tokens(rule, *size, detail=image_detail(part))
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
        return Estimate(math.ceil(chars / CHARS_PER_TOKEN) + image_tokens, uncounted, image_tokens)

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
