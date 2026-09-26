"""What the summariser reads for a chunk (#8): the chunk's records as the messages the
provider saw, never a serialisation of them (R1).

Each record's ``raw`` (the host's dict as it came) is made into the message the host
would send for it, by the host's own rules, and then the plugin's own rules are
applied on top. Nothing stored changes.

**The host's rules** are those of ``build_api_messages`` (agent/turn_context.py at
Hermes 7b761da, lines 1216-1268), which cannot be called here because it needs a live
agent. Its field rules are reproduced exactly, calling the host's own functions where
they take a message:

1. a structural clone (``agent.conversation_loop._clone_message_for_send``);
2. ``api_content`` popped, and for a user or assistant row a non-empty string sidecar
   becomes ``content`` (the exact bytes sent);
3. the host's ``PERSISTENCE_ONLY_MESSAGE_FIELDS`` popped;
4. the reasoning field the summariser's provider needs, decided by the host:
   ``apply_reasoning_content_policy`` with ``needs_reasoning_echo`` for the
   summariser's route (agent/message_sanitization.py): DeepSeek, Kimi and MiMo get
   ``reasoning_content`` on every assistant turn exactly as the host would send it,
   every other provider gets none;
5. ``reasoning`` and ``finish_reason`` popped;
6. an empty non-final user or assistant message filled by the host's
   ``fill_empty_non_final_wire_payload``;
7. ``_length_continuation_fragment`` and ``_length_continuation_nudge`` popped.

``build_api_messages`` keeps every other field (underscore keys and
``reasoning_details`` included); so does this input. The host's own adapter on the way
to the provider then applies its rules, as on the main request path: the Chat
Completions transport strips underscore keys, native carriers and, except on
OpenRouter and Nous, ``reasoning_details``; the Anthropic and Bedrock converters
rebuild the message from their carriers. Not reproduced: the host's
``canonicalize_replay_history``, which rewrites old rows by the time of the request
(a dangerous-command confirmation older than a minute becomes a sentinel); the
summariser reads what the row said. And the strict-API tool-call scrub, which the Chat
Completions transport repeats on the way.

**The plugin's rules** on top:

- R2: readable reasoning (the host's merged ``reasoning``, else a non-blank
  ``reasoning_content``) is also given as a labelled text part of its message, so that
  the summariser reads it whatever the provider does with the reasoning field; where
  the message is rebuilt from a native carrier, the part goes into the carrier too;
- R3: encrypted items are withheld, and only those: signed or encrypted
  ``reasoning_details`` entries, ``codex_reasoning_items`` carrying encrypted content,
  and the signed or redacted thinking blocks of ``anthropic_content_blocks`` and
  ``bedrock_content_blocks``, whose other blocks stay in their order (ask A-P);
- an image, found by structure, goes in only where the model table says the
  summariser reads images, else a placeholder naming its media type and record; where
  the summariser's wire is the host's Anthropic converter, the images the converter
  would evict for its per-request limit are replaced the same way, oldest first as the
  host picks them (ask A-8.2); the placeholder is the mechanism's layer, never stored;
- a tool call whose arguments are not valid JSON, which the host's Anthropic and
  Bedrock converters replace with ``{}``, gets a labelled text part carrying the stored
  string verbatim (ask A-8.3).

The chunk stands between the summariser's instructions (the system message, today's
text) and a closing user message that asks for the summary. A user turn inside the
chunk can read to the summariser as an instruction; how the chunk is framed so that
it is summarised and not continued is #10's.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from .message_content import content_parts, image_media_type, is_image_part

try:  # the host's own set of bookkeeping fields no provider receives
    from agent.message_metadata import PERSISTENCE_ONLY_MESSAGE_FIELDS as _PERSISTENCE_ONLY  # type: ignore
except Exception:  # pragma: no cover - as at Hermes 7b761da, agent/message_metadata.py:14
    _PERSISTENCE_ONLY = frozenset({"timestamp", "display_kind", "display_metadata", "_row_id"})

# Labels of the mechanism's layer in the input (the words are #10's).
READABLE_REASONING_LABEL = "[Reasoning the provider returned with this message, as it returned it]"
CLOSING_REQUEST = "Summarize the conversation above, as the system instructions say."
_ONLY_WITHHELD_REASONING = ("[This message carried only reasoning the summariser is not given: "
                            "encrypted, and its producer is not known]")


@dataclass(frozen=True)
class WireFacts:
    """What the summariser's route means for its input, from the host's own rules.
    ``reads_images`` is None where the model table has no row for the summariser: its
    images are then not sent either, and their placeholder says it is not known."""

    reads_images: Optional[bool]
    needs_reasoning_echo: bool
    anthropic_converter: bool


def _host_clone(message: dict) -> dict:
    try:
        from agent.conversation_loop import _clone_message_for_send  # type: ignore
        return _clone_message_for_send(message)
    except Exception:
        return copy.deepcopy(message)


def _host_reasoning_policy(source: dict, message: dict, needs_echo: bool) -> None:
    try:
        from agent.message_sanitization import apply_reasoning_content_policy  # type: ignore
    except Exception:
        message.pop("reasoning_content", None)
        return
    apply_reasoning_content_policy(source, message, needs_echo)


def _host_fill_empty(message: dict) -> None:
    try:
        from agent.agent_runtime_helpers import fill_empty_non_final_wire_payload  # type: ignore
    except Exception:
        return
    fill_empty_non_final_wire_payload(message, is_final=False)


def wire_facts(provider: str, model: str, base_url: str, api_mode: str,
               reads_images: Optional[bool]) -> WireFacts:
    """The route's facts for the input: whether the host sends ``reasoning_content``
    to it (``needs_reasoning_echo``, agent/message_sanitization.py at 7b761da) and
    whether its wire is the host's Anthropic converter (provider anthropic, or the
    anthropic_messages API mode)."""
    try:
        from agent.message_sanitization import needs_reasoning_echo  # type: ignore
        echo = bool(needs_reasoning_echo(provider, model, base_url))
    except Exception:
        echo = False
    try:
        from hermes_cli.config_providers import _canonical_api_mode  # type: ignore
        mode = str(_canonical_api_mode(str(api_mode or ""))).lower()
    except Exception:
        mode = str(api_mode or "").strip().lower()
    try:
        from agent.auxiliary_client import _normalize_aux_provider  # type: ignore
        dispatched = str(_normalize_aux_provider(provider))
    except Exception:
        dispatched = str(provider or "").strip().lower()
    anthropic = mode == "anthropic_messages" or (dispatched == "anthropic" and mode in ("", "anthropic_messages"))
    return WireFacts(reads_images=reads_images, needs_reasoning_echo=echo, anthropic_converter=anthropic)


# --- The host's build_api_messages, field by field (see the module docstring) -------------

def _as_the_host_sends_it(raw: dict, *, needs_echo: bool) -> dict:
    message = _host_clone(raw)
    sidecar = message.pop("api_content", None)
    for key in _PERSISTENCE_ONLY:
        message.pop(key, None)
    if isinstance(sidecar, str) and sidecar and raw.get("role") in ("user", "assistant"):
        message["content"] = sidecar
    _host_reasoning_policy(raw, message, needs_echo)
    message.pop("reasoning", None)
    message.pop("finish_reason", None)
    _host_fill_empty(message)
    message.pop("_length_continuation_fragment", None)
    message.pop("_length_continuation_nudge", None)
    return message


# --- R3: withhold encrypted items only ------------------------------------------------------

def _signed_or_encrypted_detail(entry: Any) -> bool:
    if not isinstance(entry, dict):
        return False
    kind = str(entry.get("type") or "")
    return bool(entry.get("signature")) or "encrypted" in kind or kind == "redacted_thinking" \
        or bool(entry.get("data"))


def _signed_anthropic_block(block: Any) -> bool:
    if not isinstance(block, dict):
        return False
    kind = block.get("type")
    return kind == "redacted_thinking" or (kind == "thinking" and bool(block.get("signature")))


def _signed_bedrock_block(block: Any) -> bool:
    if not isinstance(block, dict) or not isinstance(block.get("reasoningContent"), dict):
        return False
    reasoning = block["reasoningContent"]
    text = reasoning.get("reasoningText")
    return "redactedContent" in reasoning or (isinstance(text, dict) and bool(text.get("signature")))


def _keep(message: dict, key: str, drop) -> int:
    """Drop the items of ``key`` that ``drop`` names; returns how many were dropped."""
    values = message.get(key)
    if not isinstance(values, list):
        return 0
    kept = [v for v in values if not drop(v)]
    if kept:
        message[key] = kept
    else:
        message.pop(key, None)
    return len(values) - len(kept)


# The kinds of encrypted reasoning, by the host field that carries them; the model
# table's ``encrypted_reasoning`` column names the same kinds (#8, 9.6).
ENCRYPTED_KINDS = ("reasoning_details", "codex_reasoning_items", "anthropic_content_blocks", "bedrock_content_blocks")


def _withhold_encrypted(message: dict) -> dict[str, int]:
    """Withhold the encrypted items (R3), and say how many of each kind: the loss is named
    where it happens (#8, "Reasoning": never brushed over)."""
    counts = {
        "reasoning_details": _keep(message, "reasoning_details", _signed_or_encrypted_detail),
        "codex_reasoning_items": _keep(message, "codex_reasoning_items",
                                       lambda i: isinstance(i, dict) and bool(i.get("encrypted_content"))),
        "anthropic_content_blocks": _keep(message, "anthropic_content_blocks", _signed_anthropic_block)
        + _keep(message, "_anthropic_content_blocks", _signed_anthropic_block),
        "bedrock_content_blocks": _keep(message, "bedrock_content_blocks", _signed_bedrock_block),
    }
    return {kind: count for kind, count in counts.items() if count}


# --- Images ----------------------------------------------------------------------------------

_NOT_READ = "not shown to this summariser, which does not read images"
# The orchestrator's ruling on #8b: a summariser the model table has no row for is not
# said not to read images; its images are not sent, and it is said that it is not known.
_NOT_KNOWN = "image not sent: whether this summariser reads images is unknown"
_EVICTED = "left out of this call: the host's Anthropic converter drops it for its per-request image limit"


def _image_placeholder(part: dict, record: str, why: str) -> dict:
    return {"type": "text", "text": f"[An image ({image_media_type(part)}) of record {record}, {why}]"}


def _replace_images(parts: list, record: str, why: str) -> list:
    return [_image_placeholder(p, record, why) if is_image_part(p) else p for p in parts]


def _replace_images_in_message(message: dict, record: str, why: str) -> None:
    content = message.get("content")
    if isinstance(content, dict) and content.get("_multimodal") is True and isinstance(content.get("content"), list):
        message["content"] = dict(content, content=_replace_images(content["content"], record, why))
    elif isinstance(content, list):
        message["content"] = _replace_images(content, record, why)
    stashed = message.get("_anthropic_content_blocks")
    if isinstance(stashed, list):
        message["_anthropic_content_blocks"] = _replace_images(stashed, record, why)


def _image_count(message: dict) -> int:
    count = sum(1 for p in (content_parts(message.get("content")) or []) if is_image_part(p))
    stashed = message.get("_anthropic_content_blocks")
    if not count and isinstance(stashed, list):
        count = sum(1 for p in stashed if is_image_part(p))
    return count


def _evict_as_the_host_would(messages: list[dict], records: list[str]) -> None:
    """The host's Anthropic converter retires the images of the oldest image-bearing tool
    results once a request crosses its limit, counting every image, user uploads
    included, and never touching uploads (``_evict_old_screenshots``,
    agent/anthropic_message_convert.py:605; ``outbound_image_retire_count`` with
    ``OUTBOUND_IMAGE_LIMIT`` 20, agent/image_eviction_policy.py at 7b761da). The same
    count, from the host's own function, picks the same carriers here."""
    try:
        from agent.image_eviction_policy import outbound_image_retire_count  # type: ignore
    except Exception:
        return
    carriers = [i for i, m in enumerate(messages) if m.get("role") == "tool" and _image_count(m)]
    reserved = sum(_image_count(m) for m in messages if m.get("role") != "tool")
    newest_first = list(reversed(carriers))
    retire = outbound_image_retire_count([_image_count(messages[i]) for i in newest_first], reserved)
    for index in (newest_first[len(newest_first) - retire:] if retire else []):
        _replace_images_in_message(messages[index], records[index], _EVICTED)


# --- Labelled parts ---------------------------------------------------------------------------

def _readable_reasoning(raw: dict) -> Optional[str]:
    for key in ("reasoning", "reasoning_content"):
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _malformed_argument_parts(message: dict) -> list[dict]:
    parts = []
    for call in message.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        function = call.get("function") if isinstance(call.get("function"), dict) else {}
        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            continue
        try:
            json.loads(arguments)
        except (json.JSONDecodeError, ValueError):
            parts.append({"type": "text", "text": (
                f"[The arguments of tool call {call.get('id') or '?'} ({function.get('name') or '?'}) as stored; "
                f"they are not valid JSON:]\n{arguments}")})
    return parts


def _content_as_parts(content: Any) -> list:
    if isinstance(content, list):
        return list(content)
    if isinstance(content, str) and content:
        return [{"type": "text", "text": content}]
    return []


def _add_parts(message: dict, before: list[dict], after: list[dict]) -> None:
    if not before and not after:
        return
    message["content"] = before + _content_as_parts(message.get("content")) + after
    blocks = message.get("anthropic_content_blocks")
    if isinstance(blocks, list):
        message["anthropic_content_blocks"] = [{"type": "text", "text": p["text"]} for p in before] + blocks
    blocks = message.get("bedrock_content_blocks")
    if isinstance(blocks, list):
        message["bedrock_content_blocks"] = [{"text": p["text"]} for p in before] + blocks


def _has_payload(message: dict) -> bool:
    content = message.get("content")
    return bool((isinstance(content, str) and content.strip()) or (isinstance(content, list) and content)
                or message.get("tool_calls") or message.get("anthropic_content_blocks")
                or message.get("bedrock_content_blocks") or message.get("codex_message_items")
                or (isinstance(message.get("reasoning_content"), str) and message["reasoning_content"].strip()))


def summariser_message(raw: dict, record: str, facts: WireFacts,
                       withheld: Optional[dict[str, int]] = None) -> dict:
    """One record's message as the summariser receives it (see the module docstring).
    The encrypted items withheld from it are added to ``withheld`` by kind."""
    message = _as_the_host_sends_it(raw, needs_echo=facts.needs_reasoning_echo)
    for kind, count in _withhold_encrypted(message).items():
        if withheld is not None:
            withheld[kind] = withheld.get(kind, 0) + count
    if facts.reads_images is None:
        _replace_images_in_message(message, record, _NOT_KNOWN)
    elif not facts.reads_images:
        _replace_images_in_message(message, record, _NOT_READ)
    if message.get("role") == "assistant":
        readable = _readable_reasoning(raw)
        before = [{"type": "text", "text": f"{READABLE_REASONING_LABEL}\n{readable}"}] if readable else []
        _add_parts(message, before, _malformed_argument_parts(message))
        if not _has_payload(message):
            message["content"] = [{"type": "text", "text": _ONLY_WITHHELD_REASONING}]
    return message


def summariser_messages(
    records: Iterable[tuple[str, dict]],
    *,
    instructions: str,
    request: dict,
    facts: WireFacts,
    withheld: Optional[dict[str, int]] = None,
) -> list[dict]:
    """The whole input of one summariser call: the instructions, the chunk's records
    as messages, and the closing request with its fields (focus topic, custom
    instructions) where there are any. ``withheld`` receives the count of encrypted
    items withheld, by kind."""
    pairs = list(records)
    body = [summariser_message(raw, record, facts, withheld) for record, raw in pairs]
    if facts.reads_images and facts.anthropic_converter:
        _evict_as_the_host_would(body, [record for record, _raw in pairs])
    closing = CLOSING_REQUEST
    if request:
        closing += "\nrequest: " + json.dumps(request, ensure_ascii=False)
    return [{"role": "system", "content": instructions}] + body + [{"role": "user", "content": closing}]
