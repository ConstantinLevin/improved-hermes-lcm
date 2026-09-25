"""What the summariser reads for a chunk (#8): the chunk's records as the messages they
were, never a serialisation of them (R1).

Each message is read from the store's ``raw`` (the host's dict as it came) and handed
over as the provider saw it, a reader over the raw that changes nothing stored:

- ``api_content``, the exact bytes the host sent where it differs from ``content``,
  replaces ``content`` for user and assistant messages, as the host's own request
  builder does (``substitute_api_content``, agent/turn_context.py at 7b761da);
- the host's bookkeeping fields go: its persistence-only fields
  (``PERSISTENCE_ONLY_MESSAGE_FIELDS``) and every key starting with an underscore,
  which no provider receives;
- tool calls, their arguments and their results stay whole;
- readable reasoning, the host's merged ``reasoning`` field (else a non-blank
  ``reasoning_content``), goes into the message as a text part of its own under a
  neutral label (R2; the label's words are #10's);
- encrypted reasoning is withheld: the plugin does not know which provider produced
  a record, and encrypted items go only to a summariser of that provider (R3). Every
  reasoning field is dropped after the readable account is taken. That is a named
  loss, asked of Hermes (A-P: stamp the answering provider and model on each
  assistant message);
- an image, found by structure, goes in only when the model table says the
  summariser reads images; otherwise it is replaced by a placeholder naming its media
  type and the record it belongs to. The placeholder exists only in this input and is
  never stored.

The chunk stands between the summariser's instructions (the system message, today's
text) and a closing user message that asks for the summary. A user turn inside the
chunk can read to the summariser as an instruction; the JSON envelope answered that,
and how the chunk is framed so that it is summarised and not continued is #10's.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Iterable, Optional

from .message_content import content_parts, image_media_type, is_image_part

try:  # the host's own set of bookkeeping fields no provider receives
    from agent.message_metadata import PERSISTENCE_ONLY_MESSAGE_FIELDS as _PERSISTENCE_ONLY  # type: ignore
except Exception:  # pragma: no cover - as at Hermes 7b761da, agent/message_metadata.py:14
    _PERSISTENCE_ONLY = frozenset({"timestamp", "display_kind", "display_metadata", "_row_id"})

# Every field that carries a provider's reasoning on a host message.
_REASONING_FIELDS = (
    "reasoning",
    "reasoning_content",
    "reasoning_details",
    "codex_reasoning_items",
    "codex_message_items",
    "anthropic_content_blocks",
    "bedrock_content_blocks",
)

# The neutral label of the readable-reasoning part (R2; the words are #10's).
READABLE_REASONING_LABEL = "[Reasoning the provider returned with this message, as it returned it]"

# The closing request (#10 owns the words).
CLOSING_REQUEST = "Summarize the conversation above, as the system instructions say."


def _readable_reasoning(message: dict) -> Optional[str]:
    for key in ("reasoning", "reasoning_content"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _image_placeholder(part: dict, record: str) -> dict:
    return {"type": "text", "text": f"[An image ({image_media_type(part)}) of record {record}, not shown "
                                     f"to this summariser, which does not read images]"}


def _content_for_summariser(content: Any, record: str, reads_images: bool) -> Any:
    parts = content_parts(content)
    if parts is None:
        return content
    out = []
    for part in parts:
        if is_image_part(part) and not reads_images:
            out.append(_image_placeholder(part, record))
        else:
            out.append(part)
    return out


def summariser_message(raw_message: dict, record: str, *, reads_images: bool) -> dict:
    """One record's message as the summariser receives it (see the module docstring)."""
    message = copy.deepcopy(raw_message)
    sidecar = message.pop("api_content", None)
    if isinstance(sidecar, str) and sidecar and message.get("role") in ("user", "assistant"):
        message["content"] = sidecar
    readable = _readable_reasoning(message)
    for key in list(message):
        if key in _PERSISTENCE_ONLY or key in _REASONING_FIELDS or (isinstance(key, str) and key.startswith("_")):
            del message[key]
    if "tool_calls" in message and not message["tool_calls"]:
        # The host's own "no calls" (None or []), which carries nothing a provider reads.
        del message["tool_calls"]
    if "content" in message:
        message["content"] = _content_for_summariser(message["content"], record, reads_images)
    if readable is not None and message.get("role") == "assistant":
        reasoning_part = {"type": "text", "text": f"{READABLE_REASONING_LABEL}\n{readable}"}
        content = message.get("content")
        if isinstance(content, list):
            message["content"] = [reasoning_part] + content
        elif isinstance(content, str) and content:
            message["content"] = [reasoning_part, {"type": "text", "text": content}]
        else:
            message["content"] = [reasoning_part]
    return message


def summariser_messages(
    records: Iterable[tuple[str, dict]],
    *,
    instructions: str,
    request: dict,
    reads_images: bool,
) -> list[dict]:
    """The whole input of one summariser call: the instructions, the chunk's records
    as messages, and the closing request with its fields (focus topic, custom
    instructions) where there are any."""
    closing = CLOSING_REQUEST
    if request:
        closing += "\nrequest: " + json.dumps(request, ensure_ascii=False)
    return (
        [{"role": "system", "content": instructions}]
        + [summariser_message(raw, record, reads_images=reads_images) for record, raw in records]
        + [{"role": "user", "content": closing}]
    )
