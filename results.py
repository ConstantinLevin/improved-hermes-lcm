"""What a tool result is when it leaves the plugin: the final string the engine hands
the host (``LCMEngine.handle_tool_call``), or the ``_multimodal`` envelope, which the
host carries as it is. One function makes the string, so that a page is measured on the
exact string the host receives and compares with its spill threshold (#18)."""

from __future__ import annotations

import json
from typing import Any

from .tokens import ESTIMATE_LABEL

# The keys under which the tools show the plugin's own estimate (the views' token
# columns, derived from records.est_tokens and derivations.est_tokens, and their sums).
# The host's own counts (last_prompt_tokens and the like) are not estimates and are not
# labelled. Beside every such count that can hold images stands the number of images
# its estimate left uncounted (records.est_uncounted_images and its sums, #35).
_ESTIMATE_KEYS = frozenset({
    "token_count", "source_token_count", "est_tokens", "tokens", "source_tokens", "token_estimate",
    "estimated_tokens", "effective_fresh_tail_tokens", "total_tokens", "total_source_tokens",
    "total_summary_tokens",
})
_UNCOUNTED_NOTE = "; beside each count that can hold images, the images it left uncounted (*uncounted_images)"


def _has_token_count(value: Any) -> bool:
    if isinstance(value, dict):
        return any(k in _ESTIMATE_KEYS or _has_token_count(v) for k, v in value.items())
    if isinstance(value, list):
        return any(_has_token_count(v) for v in value)
    return False


def is_envelope(value: Any) -> bool:
    """The host's multimodal tool result (``_is_multimodal_tool_result``,
    agent/tool_dispatch_helpers.py:305 at Hermes d0288be5b3)."""
    return isinstance(value, dict) and value.get("_multimodal") is True and isinstance(value.get("content"), list)


def final_result(result: Any) -> Any:
    """The tool result as the engine returns it. An envelope passes as it is; a payload
    (a dict) or a JSON string becomes the final string, with the label its estimated
    token counts carry (#21) at the top of the object. A string that is not a JSON
    object passes unchanged. Idempotent."""
    if is_envelope(result):
        return result
    payload = result
    if isinstance(result, str):
        try:
            payload = json.loads(result)
        except (TypeError, ValueError):
            return result
    if not isinstance(payload, dict):
        return result if isinstance(result, str) else json.dumps(payload, ensure_ascii=False)
    if _has_token_count(payload) and "token_counts" not in payload:
        payload = {"token_counts": ESTIMATE_LABEL + _UNCOUNTED_NOTE, **payload}
    elif isinstance(result, str):
        return result
    return json.dumps(payload, ensure_ascii=False)
