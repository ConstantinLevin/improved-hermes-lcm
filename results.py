"""What a tool result is when it leaves the plugin: the final string the engine hands
the host (``LCMEngine.handle_tool_call``), or the ``_multimodal`` envelope, which the host
carries as it is. One function makes the string, so that a page is measured on the exact
string the host receives and compares with its spill threshold (#18)."""

from __future__ import annotations

import json
from typing import Any

from .tokens import ESTIMATE_LABEL

# The label of a result that holds the plugin's own estimates (the views' token columns,
# derived from records.est_tokens and derivations.est_tokens, and their sums). Which keys
# hold them is declared by the tool that wrote them (``tools.ESTIMATES``, #78, A12): never
# found by walking the result, which can hold the host's data under any key (a stored
# message's ``token_count`` is a host column). The host's own counts (last_prompt_tokens
# and the like) are not estimates and are never declared. Beside every such count that can
# hold images stands the number of images its estimate left uncounted
# (records.est_uncounted_images and its sums, #35).
_UNCOUNTED_NOTE = "; beside each count that can hold images, the images it left uncounted (*uncounted_images)"


def is_envelope(value: Any) -> bool:
    """The host's multimodal tool result, by the host's own test (``_is_multimodal_tool_result``,
    agent/tool_dispatch_helpers.py:305-307 at Hermes 1c535d9689), called, never copied. Where
    the host's test cannot be read this raises, and the tool's call answers with that error
    (``LCMEngine.handle_tool_call``)."""
    from agent.tool_dispatch_helpers import _is_multimodal_tool_result  # type: ignore

    return bool(_is_multimodal_tool_result(value))


def final_result(result: Any, estimates: tuple = ()) -> Any:
    """The tool result as the engine returns it: an envelope passes as it is; a payload (a
    dict) or a JSON string becomes the final string. Where the tool that wrote it declares
    the keys that hold its estimates (``estimates``), the label names them at the top of the
    object (#21, #78); nothing else adds a label. A string that is not a JSON object passes
    unchanged. Idempotent."""
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
    if estimates and "token_counts" not in payload and "error" not in payload:
        payload = {"token_counts": f"{ESTIMATE_LABEL}, under the keys {', '.join(estimates)}{_UNCOUNTED_NOTE}",
                   **payload}
    elif isinstance(result, str):
        return result
    return json.dumps(payload, ensure_ascii=False)
