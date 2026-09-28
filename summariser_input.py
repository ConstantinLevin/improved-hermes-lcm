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

from .message_content import content_parts, image_media_type, is_image_part, readable_reasoning, sent_content, \
    sidecar_sent

try:  # the host's own set of bookkeeping fields no provider receives
    from agent.message_metadata import PERSISTENCE_ONLY_MESSAGE_FIELDS as _PERSISTENCE_ONLY  # type: ignore
except Exception:  # pragma: no cover - as at Hermes 7b761da, agent/message_metadata.py:14
    _PERSISTENCE_ONLY = frozenset({"timestamp", "display_kind", "display_metadata", "_row_id"})

# Labels of the mechanism's layer in the input (the words are #10's).
READABLE_REASONING_LABEL = "[Reasoning the provider returned with this message, as it returned it]"
CLOSING_REQUEST = "Summarize the conversation above, as the system instructions say."
_ONLY_WITHHELD_REASONING = ("[This message carried only reasoning the summariser is not given: "
                            "encrypted, and its producer is not known]")
# The strict projection's words (the query, #19): its model is the summariser's, and its
# labels name that model as "this model", never as the summariser.
_ONLY_WITHHELD_REASONING_STRICT = ("[This message carried only reasoning this model is not given: "
                                   "encrypted, and its producer is not known]")


@dataclass(frozen=True)
class WireFacts:
    """What the summariser's route means for its input, from the host's own rules.
    ``reads_images`` is None where the model table has no row for the summariser: its
    images are then not sent either, and their placeholder says it is not known."""

    reads_images: Optional[bool]
    needs_reasoning_echo: bool
    anthropic_converter: bool


class HostUnavailable(Exception):
    """A host function the strict projection needs cannot be read (expansion refuses the
    call; the summariser's own path keeps its fallbacks, which are not D1's)."""


def _strict_import(what: str, module: str, name: str,
                   so: str = "the message as the host sends it is not known") -> Any:
    """The host's function, or ``HostUnavailable`` naming it and what cannot be known without it."""
    try:
        return getattr(__import__(module, fromlist=[name]), name)
    except Exception as exc:
        raise HostUnavailable(f"the host's {what} ({module}.{name}) cannot be read ({type(exc).__name__}: {exc}), "
                              f"so {so}") from None


def _host_clone(message: dict, strict: bool = False) -> dict:
    if strict:
        return _strict_import("message clone", "agent.conversation_loop", "_clone_message_for_send")(message)
    try:
        from agent.conversation_loop import _clone_message_for_send  # type: ignore
        return _clone_message_for_send(message)
    except Exception:
        return copy.deepcopy(message)


def _host_reasoning_policy(source: dict, message: dict, needs_echo: bool, strict: bool = False) -> None:
    if strict:
        _strict_import("reasoning policy", "agent.message_sanitization", "apply_reasoning_content_policy")(
            source, message, needs_echo)
        return
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


def host_fill_text(row: dict) -> Optional[str]:
    """The host's own stand-in for an empty non-final message (``fill_empty_non_final_wire_payload``,
    agent/agent_runtime_helpers.py:2582-2590, applied at agent/turn_context.py:1264), or None
    where the host sends the message as it is. ``row`` must be the host's own input to the
    fill: ``host_row_before_fill`` of the record, before any transformation of the plugin's
    (LEARNINGSFÜRPLÄNE A10). Run on a copy; strict."""
    fill = _strict_import("empty-message fill", "agent.agent_runtime_helpers", "fill_empty_non_final_wire_payload")
    probe = dict(row)
    return str(probe.get("content")) if fill(probe, is_final=False) else None


def host_reasoning_pad(provider: str, model: str, base_url: str) -> bool:
    """The host's own reasoning-echo decision for a route (``_needs_thinking_reasoning_pad``,
    agent/reasoning_params.py:163-179 at Hermes 8afaab3703), from the route the engine holds:
    the DeepSeek, Kimi and MiMo families exactly as the host tests them. The host's opt-in
    (``model.reasoning_echo``, the agent's ``_reasoning_echo_flag``) is the agent's and cannot
    be read here: a caller that needs it asks both ways. Strict."""
    mixin = _strict_import("reasoning-echo decision", "agent.reasoning_params", "ReasoningParamsMixin")

    class _Route(mixin):  # the host's own methods, over the engine's route
        pass

    route = _Route()
    route.provider, route.model, route.base_url = provider, model, base_url
    return bool(route._needs_thinking_reasoning_pad())


def wire_facts(provider: str, model: str, base_url: str, api_mode: str,
               reads_images: Optional[bool], *, strict: bool = False) -> WireFacts:
    """The route's facts for the input: whether the host sends ``reasoning_content``
    to it (``needs_reasoning_echo``, agent/message_sanitization.py at 7b761da) and
    whether its wire is the host's Anthropic converter (provider anthropic, or the
    anthropic_messages API mode). ``strict`` (the query, #19): every host function is
    called, and one that cannot be read raises ``HostUnavailable`` naming it; the
    summariser's call keeps its three fallbacks until #72 is worked."""
    if strict:
        # Each names the fact that cannot be known without it; nothing is sent either way.
        echo = bool(_strict_import(
            "reasoning-echo test", "agent.message_sanitization", "needs_reasoning_echo",
            so="whether the host sends reasoning_content to this route is not known; nothing was sent")(
            provider, model, base_url))
        mode = str(_strict_import(
            "API mode canonicalisation", "hermes_cli.config_providers", "_canonical_api_mode",
            so="the route's API mode as the host reads it is not known; nothing was sent")(
            str(api_mode or ""))).lower()
        dispatched = str(_strict_import(
            "provider normalisation", "agent.auxiliary_client", "_normalize_aux_provider",
            so="the route's provider as the host dispatches it is not known; nothing was sent")(
            provider))
        anthropic = mode == "anthropic_messages" or (dispatched == "anthropic" and mode in ("", "anthropic_messages"))
        return WireFacts(reads_images=reads_images, needs_reasoning_echo=echo, anthropic_converter=anthropic)
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

def _row_before_fill(raw: dict, *, needs_echo: bool, strict: bool) -> dict:
    """``build_api_messages``' per-row steps up to the fill (agent/turn_context.py:1221-1260):
    the clone, the sidecar, the persistence fields, the reasoning copy, the pops."""
    message = _host_clone(raw, strict)
    persistence_only = (_strict_import("persistence fields", "agent.message_metadata",
                                       "PERSISTENCE_ONLY_MESSAGE_FIELDS") if strict else _PERSISTENCE_ONLY)
    message.pop("api_content", None)
    for key in persistence_only:
        message.pop(key, None)
    if sidecar_sent(raw):                     # one rule with what grep searches (#18 D2)
        message["content"] = sent_content(raw)
    _host_reasoning_policy(raw, message, needs_echo, strict)
    message.pop("reasoning", None)
    message.pop("finish_reason", None)
    return message


def _as_the_host_sends_it(raw: dict, *, needs_echo: bool, strict: bool = False, fill: bool = True) -> dict:
    message = _row_before_fill(raw, needs_echo=needs_echo, strict=strict)
    if fill:
        _host_fill_empty(message)
    message.pop("_length_continuation_fragment", None)
    message.pop("_length_continuation_nudge", None)
    return message


def host_row_before_fill(raw: dict, *, pad: bool) -> dict:
    """The record's row as the host builds it for a request up to its fill, with the host's
    reasoning pad for the route (``host_reasoning_pad``), every host function called
    strictly: the host's own input to ``fill_empty_non_final_wire_payload``."""
    return _row_before_fill(raw, needs_echo=pad, strict=True)


def item_message(row: dict) -> dict:
    """What expansion shows of a host row (the plan of #71, §5.1; the re-plan C2): a copy
    of ``host_row_before_fill``, then only the plugin's own transformations, after every host
    function has had its input: the encrypted items withheld (R3), ``reasoning_content``
    popped (the readable reasoning is shown as a field of its own), the continuation marks
    popped."""
    message = copy.deepcopy(row)
    _withhold_encrypted(message)
    message.pop("reasoning_content", None)
    message.pop("_length_continuation_fragment", None)
    message.pop("_length_continuation_nudge", None)
    return message


# --- R3: withhold encrypted items only ------------------------------------------------------

def _signed_or_encrypted_detail(entry: Any) -> bool:
    """A ``reasoning_details`` entry that is signed or encrypted, as the host persists it
    (agent/chat_completion_helpers.py:1709-1721; Anthropic's cleaned thinking blocks,
    agent/transports/anthropic.py:84-87; Bedrock's redacted reasoning, bedrock_adapter.py:850),
    or a ``<provider>.native_assistant`` entry: another provider's signed replay
    (agent/transports/chat_completions.py:386-401)."""
    if not isinstance(entry, dict):
        return False
    kind = str(entry.get("type") or "")
    return bool(entry.get("signature")) or "encrypted" in kind or kind == "redacted_thinking" \
        or bool(entry.get("data")) or kind.endswith(".native_assistant")


def _signed_anthropic_block(block: Any) -> bool:
    if not isinstance(block, dict):
        return False
    kind = block.get("type")
    return kind == "redacted_thinking" or (kind == "thinking" and bool(block.get("signature")))


def _signed_bedrock_block(block: Any) -> bool:
    """A ``bedrock_content_blocks`` block that is signed or redacted. The host persists the
    flat form ``{"reasoningContent": {"text", "signature", "redactedContentBase64"}}``
    (agent/bedrock_adapter.py:836-852, stored by chat_completion_helpers.py:1728-1731); the
    wire form nests ``reasoningText.{text, signature}`` or carries ``redactedContent``
    (bedrock_adapter.py:721-731). Both are read (#74)."""
    if not isinstance(block, dict) or not isinstance(block.get("reasoningContent"), dict):
        return False
    reasoning = block["reasoningContent"]
    text = reasoning.get("reasoningText")
    return ("redactedContent" in reasoning or bool(reasoning.get("redactedContentBase64"))
            or bool(reasoning.get("signature")) or (isinstance(text, dict) and bool(text.get("signature"))))


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
# The host's stored carriers from which its converters replay a message's text instead of
# reading ``content`` (the query's strict projection sends none of them, ``strict_message``).
TEXT_REPLAY_CARRIERS = ("codex_message_items", "anthropic_content_blocks", "bedrock_content_blocks")


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
# The strict projection's words (the query's model, #19).
_NOT_READ_STRICT = "not shown to this model, which does not read images"
_NOT_KNOWN_STRICT = "image not sent: whether this model reads images is unknown"
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


def image_count(message: dict) -> int:
    """The images one message as the summariser receives it carries (placeholders are
    text and do not count)."""
    return _image_count(message)


def wire_image_limit(facts: WireFacts) -> Optional[int]:
    """The most images one summariser request may carry before the host's converter for
    the summariser's wire retires some of them unseen, or None where none does.

    Retiring is a loss (#8: the summariser never sees those images; the orchestrator's
    ruling on the Codex review of 6f4a351), so the cut keeps every chunk within this
    limit, as it keeps it within B. Read at Hermes origin/main d0288be5b3:
    - the Anthropic Messages converter (``_evict_old_screenshots``,
      agent/anthropic_message_convert.py:605) retires tool-result images once a request
      carries more than ``OUTBOUND_IMAGE_LIMIT`` (20, agent/image_eviction_policy.py),
      every image counted, uploads included; it runs on the auxiliary path too;
    - the Chat Completions and Responses paths of ``call_llm`` retire none:
      ``evict_stale_outbound_tool_images`` runs only on the main agent's send path
      (agent/chat_completion_helpers.py:2247, agent/turn_request_assembly.py:153);
    - where the summariser is sent no images (it does not read them, or that is not
      known), there is nothing to retire.
    Raises where the host's limit cannot be read."""
    if not (facts.reads_images and facts.anthropic_converter):
        return None
    from agent.image_eviction_policy import OUTBOUND_IMAGE_LIMIT  # type: ignore

    return int(OUTBOUND_IMAGE_LIMIT)


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

# The one rule for readable reasoning, shared with expansion and grep (message_content).
_readable_reasoning = readable_reasoning


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


FILL_NOTE_LABEL = "[The host sends this empty message to the provider with its own stand-in as content:]"


def _json_kind(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


# --- The strict projection's closed domain (the query, #19, PR #83 plan §4-§5) -------------
#
# What the query gives its model of a record is closed: a content that is a string, None, or a
# list of text parts and image parts; tool calls of the shape the host writes; readable
# reasoning as labelled text parts. Every other value is given as a labelled rendering of its
# JSON (the orchestrator's ruling OD-E, one level down), refused by name, or withheld and
# counted. ``given`` receives every string the query gives from the record, with its path in
# the record as expansion shows it and its standing, so that an excerpt can be checked against
# the field it was drawn from (plan §2).

GIVEN_CONTENT, GIVEN_RESULT, GIVEN_CALL, GIVEN_REASONING = "content", "result", "call", "reasoning"

# The keys the projection reads as the record's transcript; every other key is either the
# host's own bookkeeping (``_host_metadata_keys``) or unknown, and then given as labelled JSON
# (ruling OD-3a).
_TRANSCRIPT_KEYS = frozenset({
    "role", "content", "tool_calls", "tool_call_id", "reasoning", "reasoning_content", "reasoning_details",
    "codex_reasoning_items", "codex_message_items", "anthropic_content_blocks", "bedrock_content_blocks",
    "_anthropic_content_blocks", "api_content",
})
# The fields a ``reasoning_details`` entry or a Codex reasoning item holds readable text in, as
# the host reads them (agent/agent_runtime_helpers.py ``_extract_reasoning`` 1376-1378, and
# agent/codex_responses_adapter.py 1023-1027, 1080-1083, at Hermes 375930d089).
_READABLE_DETAIL_KEYS = ("summary", "thinking", "content", "text")


def _host_metadata_keys() -> frozenset:
    """The keys the host writes on a message as its own bookkeeping, from the host's own
    constants at run time (ruling OD-3a): the session schema's columns
    (``hermes_state_messages._MESSAGE_SCHEMA_KEYS``), the message core keys
    (``agent.message_sanitization._MESSAGE_CORE_KEYS``, ``name`` among them), the keys the Chat
    Completions transport strips (``_STRIP_MSG_KEYS``) and the persistence-only fields; and
    every key that begins with an underscore, which the host strips from every wire."""
    so = "which of a message's keys are the host's own bookkeeping is not known; nothing was sent"
    schema = _strict_import("session schema keys", "hermes_state_messages", "_MESSAGE_SCHEMA_KEYS", so=so)
    core = _strict_import("message core keys", "agent.message_sanitization", "_MESSAGE_CORE_KEYS", so=so)
    strip = _strict_import("Chat Completions strip keys", "agent.transports.chat_completions", "_STRIP_MSG_KEYS",
                           so=so)
    persistence = _strict_import("persistence fields", "agent.message_metadata", "PERSISTENCE_ONLY_MESSAGE_FIELDS",
                                 so=so)
    return frozenset(schema) | frozenset(core) | frozenset(strip) | frozenset(persistence)


def _leaves(value: Any, path: tuple) -> Iterable[tuple[tuple, str]]:
    """Every string of a value, with its path; a number or a boolean as its JSON text."""
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, bool) or isinstance(value, (int, float)):
        yield path, json.dumps(value)
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _leaves(item, path + (key,))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _leaves(item, path + (index,))


@dataclass
class Given:
    """What the strict projection gives of one record beside the message: every string with
    its path and standing (``values``), every text part the query added (``added``), the
    record's problems that refuse the query (``problems``), and counts for the header."""

    values: list
    added: list
    problems: list
    reasoning_parts: int = 0


def _json_part(label: str, value: Any, path: tuple, standing: str, given: Given) -> dict:
    part = {"type": "text", "text": f"{label}\n{json.dumps(value, ensure_ascii=False)}"}
    given.added.append(part["text"])
    given.values.extend((leaf_path, text, standing) for leaf_path, text in _leaves(value, path))
    return part


def image_part(part: Any) -> bool:
    """An image part by structure (``message_content.is_image_part``), for any stored value: a
    ``type`` that is not a string (a list, an object) is no image type and is never hashed."""
    return isinstance(part, dict) and isinstance(part.get("type"), str) and is_image_part(part)


def _canonical_parts(parts: list, prefix: tuple, standing: str, given: Given) -> list:
    """A list content's members, each a text part, an image part, or a labelled part holding
    the member's JSON (plan §4.2)."""
    shown = []
    for index, part in enumerate(parts):
        if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
            shown.append(part)
            given.values.append((prefix + (index, "text"), part["text"], standing))
        elif image_part(part):
            shown.append(part)
        else:
            shown.append(_json_part(f"[Member {index + 1} of the stored content is not a text or image part; shown "
                                    f"as its JSON by the query:]", part, prefix + (index,), standing, given))
    return shown


def _canonical_content(content: Any, standing: str, given: Given) -> Any:
    """The strict projection's content (plan §4.2): a string or None as stored; a list as its
    canonical members; the host's ``_multimodal`` envelope as its parts, its other keys in a
    labelled part; any other value (an object, a number, a boolean) as one labelled part
    holding its JSON (ruling OD-E)."""
    base = ("message", "content")
    if content is None:
        return None
    if isinstance(content, str):
        given.values.append((base, content, standing))
        return content
    if isinstance(content, list):
        return _canonical_parts(content, base, standing, given)
    parts = content_parts(content)
    if parts is not None and isinstance(content, dict):
        shown = _canonical_parts(parts, base + ("content",), standing, given)
        rest = {key: value for key, value in content.items() if key not in ("_multimodal", "content")}
        if rest:
            shown.append(_json_part("[The rest of this stored multimodal envelope, shown as its JSON by the query:]",
                                    rest, base, standing, given))
        return shown
    return [_json_part(f"[The stored content is a JSON {_json_kind(content)}, shown as its JSON by the query:]",
                       content, base, standing, given)]


def _canonical_calls(message: dict, given: Given) -> list[dict]:
    """The message's tool calls as the wire can carry them (plan §4.2): a list of calls of
    the shape the host writes (a dict, a dict ``function`` with a non-blank string ``name``
    and a string ``arguments``, a non-blank string ``id``). A falsy value is dropped (the host's
    Anthropic converter raises on ``None``); every other value or call is given as a labelled
    part holding its JSON, after the content, and is not a call on the wire. Returns those
    parts."""
    calls = message.get("tool_calls")
    base = ("message", "tool_calls")
    if not calls:
        message.pop("tool_calls", None)
        return []
    if not isinstance(calls, list):
        message.pop("tool_calls", None)
        return [_json_part(f"[The stored tool calls are a JSON {_json_kind(calls)}, not a list of calls; shown as "
                           f"their JSON by the query:]", calls, base, GIVEN_CALL, given)]
    kept, after = [], []
    for index, call in enumerate(calls):
        function = call.get("function") if isinstance(call, dict) else None
        if (isinstance(function, dict) and isinstance(function.get("name"), str) and function["name"].strip()
                and isinstance(function.get("arguments"), str) and isinstance(call.get("id"), str)
                and call["id"].strip()):
            kept.append(call)
            given.values.append((base + (index, "function", "name"), function["name"], GIVEN_CALL))
            given.values.append((base + (index, "function", "arguments"), function["arguments"], GIVEN_CALL))
        else:
            after.append(_json_part(f"[Tool call {index + 1} as stored is not a call the wire can carry; shown as "
                                    f"its JSON by the query:]", call, base + (index,), GIVEN_CALL, given))
    if kept:
        message["tool_calls"] = kept
    else:
        message.pop("tool_calls", None)
    return after


def _given_texts(given: Given, standings: tuple) -> list[str]:
    return [text for _path, text, standing in given.values if standing in standings]


def _parsed(arguments: str) -> tuple[Any, bool]:
    """A call's arguments parsed as JSON, or (None, False) where they are not JSON the parser can
    read (not valid, or nested past its recursion limit): then the stored string is given in a
    labelled part and the call is compared by its name."""
    try:
        return json.loads(arguments), True
    except (ValueError, RecursionError):
        return None, False


def _call_arguments(call: dict) -> tuple[str, Any, bool]:
    function = call["function"]
    arguments, parsed = _parsed(function["arguments"])
    return function["name"], arguments, parsed


def _unparsed_argument_parts(message: dict) -> list[dict]:
    """The strict projection's ``_malformed_argument_parts``, over its canonical calls: a
    labelled part carrying, verbatim, each call's arguments that are not JSON the parser reads."""
    parts = []
    for call in message.get("tool_calls") or []:
        function = call["function"]
        if not _parsed(function["arguments"])[1]:
            parts.append({"type": "text", "text": (
                f"[The arguments of tool call {call.get('id') or '?'} ({function.get('name') or '?'}) as stored; "
                f"they are not valid JSON:]\n{function['arguments']}")})
    return parts


def _readable_block(path: tuple, text: Any, readable: list) -> None:
    if isinstance(text, str) and text.strip():
        readable.append((path, text))


def _readable_strings(value: Any, path: tuple) -> list[tuple[tuple, str]]:
    """The readable texts under a reasoning field: its non-blank strings, not those under a
    ``type`` key (a part's kind, not its text)."""
    return [(leaf, text) for leaf, text in _leaves(value, path)
            if isinstance(text, str) and leaf[-1] != "type" and text.strip() and not _is_json_scalar(value, leaf, path)]


def _is_json_scalar(value: Any, leaf: tuple, path: tuple) -> bool:
    """Whether the leaf at ``leaf`` is a number or a boolean (``_leaves`` gives those as their
    JSON text); only strings are readable text."""
    node = value
    for step in leaf[len(path):]:
        node = node[step]
    return not isinstance(node, str)


def _reasoning_and_carriers(raw: dict, message: dict, given: Given, withheld_text: dict) -> tuple[list, list]:
    """The reasoning fields and replay carriers of a record, faced block by block (plan §5.2).
    Returns the labelled parts to put before the stored content (readable reasoning) and after
    it (values the query renders). Readable text held anywhere but the text the query already
    gives as the record's readable reasoning is given as its own labelled part naming its path
    (exact containment, ruling OD-4a); a carrier's message text must stand in the given content
    and a carrier's call must be one of the stored calls, else the record is a problem that
    refuses the query; an image a carrier holds is given as an image part; an unknown block is
    given as its JSON. ``withheld_text`` counts, per kind, the signed items whose readable text
    is given (their signatures are withheld)."""
    role = raw.get("role")
    readable: list = []       # (path, text) of readable reasoning held outside the main field
    before: list = []
    after: list = []
    main = readable_reasoning(raw)
    for key in ("reasoning", "reasoning_content"):
        value = raw.get(key)
        if isinstance(value, str):
            if value.strip() and value != main:
                readable.append((("message", key), value))
        elif value is not None:
            after.append(_json_part(f"[{key} as stored is not text; shown as its JSON by the query:]", value,
                                    ("message", key), GIVEN_REASONING, given))
    details = raw.get("reasoning_details")
    if isinstance(details, list):
        for index, entry in enumerate(details):
            if isinstance(entry, dict):
                texts = [pair for key in _READABLE_DETAIL_KEYS if key in entry
                         for pair in _readable_strings(entry[key], ("message", "reasoning_details", index, key))]
                readable.extend(texts)
                if texts and _signed_or_encrypted_detail(entry):
                    withheld_text["reasoning_details"] = withheld_text.get("reasoning_details", 0) + 1
            elif entry is not None:
                after.append(_json_part(f"[reasoning_details[{index}] as stored is not an entry; shown as its JSON "
                                        f"by the query:]", entry, ("message", "reasoning_details", index),
                                        GIVEN_REASONING, given))
    elif details is not None:
        after.append(_json_part("[reasoning_details as stored is not a list; shown as its JSON by the query:]",
                                details, ("message", "reasoning_details"), GIVEN_REASONING, given))
    items = raw.get("codex_reasoning_items")
    if isinstance(items, list):
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                if item is not None:
                    after.append(_json_part(f"[codex_reasoning_items[{index}] as stored is not an item; shown as its "
                                            f"JSON by the query:]", item, ("message", "codex_reasoning_items", index),
                                            GIVEN_REASONING, given))
                continue
            texts = [pair for key in ("summary", "text", "content") if key in item
                     for pair in _readable_strings(item[key], ("message", "codex_reasoning_items", index, key))]
            readable.extend(texts)
            if texts and item.get("encrypted_content"):
                withheld_text["codex_reasoning_items"] = withheld_text.get("codex_reasoning_items", 0) + 1
    # A carrier's message text must stand in what is given as the message's text: its content,
    # or its readable reasoning (a Codex commentary item's text is the host's ``reasoning``,
    # agent/codex_responses_adapter.py 1125-1134).
    content_texts = _given_texts(given, (GIVEN_CONTENT, GIVEN_RESULT)) + ([main] if main else [])
    calls = [_call_arguments(call) for call in (message.get("tool_calls") or [])]

    def carried(text: str, path: tuple) -> None:
        if isinstance(text, str) and text.strip() and not any(text in given_text for given_text in content_texts):
            given.problems.append(f"its replay carrier holds text its content does not ({'.'.join(map(str, path))})")

    def called(name: Any, arguments: Any, path: tuple) -> None:
        if not any(name == stored_name and (not parsed or arguments == stored_arguments)
                   for stored_name, stored_arguments, parsed in calls):
            given.problems.append(f"its replay carrier holds a call its tool calls do not "
                                  f"({'.'.join(map(str, path))}, {name if isinstance(name, str) else '?'})")

    def image(block: Any, path: tuple) -> None:
        source = block.get("source") if isinstance(block, dict) else None
        url = None
        if isinstance(source, dict) and source.get("type") == "base64" and isinstance(source.get("data"), str):
            url = f"data:{source.get('media_type') or 'application/octet-stream'};base64,{source['data']}"
        elif isinstance(source, dict) and isinstance(source.get("url"), str):
            url = source["url"]
        if url is None:
            after.append(_json_part(f"[{'.'.join(map(str, path))} is an image block of no shape the query reads; "
                                    f"shown as its JSON by the query:]", block, path, GIVEN_CONTENT, given))
            return
        label = {"type": "text", "text": f"[An image the provider returned in {'.'.join(map(str, path))}, which the "
                                         f"stored content does not hold:]"}
        given.added.append(label["text"])
        after.extend([label, {"type": "image_url", "image_url": {"url": url}}])

    blocks = raw.get("anthropic_content_blocks")
    for index, block in enumerate(blocks if isinstance(blocks, list) else []):
        path = ("message", "anthropic_content_blocks", index)
        kind = block.get("type") if isinstance(block, dict) else None
        if kind == "text":
            carried(block.get("text"), path + ("text",))
        elif kind == "thinking":
            _readable_block(path + ("thinking",), block.get("thinking"), readable)
            if block.get("signature") and isinstance(block.get("thinking"), str) and block["thinking"].strip():
                withheld_text["anthropic_content_blocks"] = withheld_text.get("anthropic_content_blocks", 0) + 1
        elif kind == "redacted_thinking":
            pass                                         # encrypted: withheld and counted (R3)
        elif kind == "tool_use":
            called(block.get("name"), block.get("input"), path)
        elif kind == "image":
            image(block, path)
        else:
            after.append(_json_part(f"[anthropic_content_blocks[{index}] is a block of no type the query knows; shown "
                                    f"as its JSON by the query:]", block, path, GIVEN_CONTENT, given))
    blocks = raw.get("bedrock_content_blocks")
    for index, block in enumerate(blocks if isinstance(blocks, list) else []):
        path = ("message", "bedrock_content_blocks", index)
        if isinstance(block, dict) and isinstance(block.get("text"), str):
            carried(block["text"], path + ("text",))
        elif isinstance(block, dict) and isinstance(block.get("reasoningContent"), dict):
            reasoning = block["reasoningContent"]
            nested = reasoning.get("reasoningText") if isinstance(reasoning.get("reasoningText"), dict) else {}
            texts: list = []
            for key, text in (("text", reasoning.get("text")), ("reasoningText.text", nested.get("text"))):
                _readable_block(path + ("reasoningContent",) + tuple(key.split(".")), text, texts)
            readable.extend(texts)
            if texts and _signed_bedrock_block(block):
                withheld_text["bedrock_content_blocks"] = withheld_text.get("bedrock_content_blocks", 0) + 1
        elif isinstance(block, dict) and isinstance(block.get("toolUse"), dict):
            called(block["toolUse"].get("name"), block["toolUse"].get("input"), path)
        else:
            after.append(_json_part(f"[bedrock_content_blocks[{index}] is a block of no shape the query knows; shown "
                                    f"as its JSON by the query:]", block, path, GIVEN_CONTENT, given))
    items = raw.get("codex_message_items")
    for index, item in enumerate(items if isinstance(items, list) else []):
        path = ("message", "codex_message_items", index)
        parts = item.get("content") if isinstance(item, dict) and item.get("type") == "message" else None
        if not isinstance(parts, list):
            after.append(_json_part(f"[codex_message_items[{index}] is an item of no shape the query knows; shown as "
                                    f"its JSON by the query:]", item, path, GIVEN_CONTENT, given))
            continue
        for number, part in enumerate(parts):
            if isinstance(part, dict) and part.get("type") in ("output_text", "text") and isinstance(part.get("text"),
                                                                                                     str):
                carried(part["text"], path + ("content", number, "text"))
            else:
                after.append(_json_part(f"[codex_message_items[{index}].content[{number}] is a part of no shape the "
                                        f"query knows; shown as its JSON by the query:]", part,
                                        path + ("content", number), GIVEN_CONTENT, given))
    stash = raw.get("_anthropic_content_blocks")
    standing = GIVEN_RESULT if role == "tool" else GIVEN_CONTENT
    for index, block in enumerate(stash if isinstance(stash, list) else []):
        path = ("message", "_anthropic_content_blocks", index)
        kind = block.get("type") if isinstance(block, dict) else None
        if kind == "text" and isinstance(block.get("text"), str):
            if block["text"].strip() and not any(block["text"] in text for text in content_texts):
                part = {"type": "text", "text": f"[A text block the host stashed in _anthropic_content_blocks[{index}]"
                                                f", which the stored content does not hold:]\n{block['text']}"}
                given.added.append(part["text"])
                given.values.append((path + ("text",), block["text"], standing))
                after.append(part)
        elif kind == "redacted_thinking":
            pass                                         # encrypted: withheld and counted (R3)
        elif kind == "thinking":
            _readable_block(path + ("thinking",), block.get("thinking"), readable)
            if block.get("signature") and isinstance(block.get("thinking"), str) and block["thinking"].strip():
                withheld_text["anthropic_content_blocks"] = withheld_text.get("anthropic_content_blocks", 0) + 1
        elif kind == "image" or image_part(block):
            if image_part(block) and kind != "image":
                label = {"type": "text", "text": f"[An image the host stashed in _anthropic_content_blocks[{index}]:]"}
                given.added.append(label["text"])
                after.extend([label, block])
            else:
                image(block, path)
        elif block is not None:
            after.append(_json_part(f"[_anthropic_content_blocks[{index}] is a block of no type the query knows; shown "
                                    f"as its JSON by the query:]", block, path, standing, given))
    if stash is not None and not isinstance(stash, list):
        after.append(_json_part("[_anthropic_content_blocks as stored is not a list; shown as its JSON by the query:]",
                                stash, ("message", "_anthropic_content_blocks"), standing, given))
    # Readable reasoning: the main field, then every other readable text not contained in
    # what is given as reasoning already (exact containment, ruling OD-4a).
    shown: list[str] = []
    if main:
        before.append({"type": "text", "text": f"{READABLE_REASONING_LABEL}\n{main}"})
        given.added.append(before[-1]["text"])
        given.values.append((("lcm", "reasoning") if role == "assistant" else
                             ("message", "reasoning" if raw.get("reasoning") == main else "reasoning_content"),
                             main, GIVEN_REASONING))
        shown.append(main)
    for path, text in readable:
        if any(text in earlier for earlier in shown):
            continue
        part = {"type": "text", "text": f"[Readable reasoning the provider returned in {'.'.join(map(str, path))}, "
                                        f"as stored:]\n{text}"}
        before.append(part)
        given.added.append(part["text"])
        given.values.append((path, text, GIVEN_REASONING))
        given.reasoning_parts += 1
        shown.append(text)
    return before, after


def _unknown_keys(message: dict, given: Given) -> list[dict]:
    """A key that is neither the record's transcript nor the host's bookkeeping is given as a
    labelled part holding its JSON (ruling OD-3a)."""
    known = _host_metadata_keys()
    return [_json_part(f"[The stored key {key!r} is no field the query knows; shown as its JSON by the query:]",
                       value, ("message", key), GIVEN_CONTENT, given)
            for key, value in list(message.items())
            if key not in _TRANSCRIPT_KEYS and key not in known and not str(key).startswith("_")]


def strict_message(raw: dict, record: str, facts: WireFacts, withheld: dict, given: Given) -> dict:
    """One record's message as the query gives it (plan §4-§5): the host's per-row rules, each
    host function called strictly, then the closed domain. ``given`` receives what is given of
    the record; ``withheld`` the encrypted items withheld, by kind, and the signed items whose
    readable text is given (their signatures only withheld)."""
    row = _row_before_fill(raw, needs_echo=facts.needs_reasoning_echo, strict=True)
    fill = host_fill_text(row)
    message = copy.deepcopy(row)
    message.pop("_length_continuation_fragment", None)
    message.pop("_length_continuation_nudge", None)
    standing = GIVEN_RESULT if raw.get("role") == "tool" else GIVEN_CONTENT
    if "content" in message:
        message["content"] = _canonical_content(message["content"], standing, given)
    call_parts = _canonical_calls(message, given)
    withheld_text: dict[str, int] = {}
    before, after = _reasoning_and_carriers(raw, message, given, withheld_text)
    after = call_parts + after + _unknown_keys(message, given)
    withheld_here = _withhold_encrypted(message)
    for kind, count in withheld_text.items():
        withheld_here[kind] = max(0, withheld_here.get(kind, 0) - count)
        withheld_here[f"{kind}: signatures withheld, their readable text given"] = count
    for kind, count in withheld_here.items():
        if count:
            withheld[kind] = withheld.get(kind, 0) + count
    # The host's replay carriers and the private stash are not sent: the stored content with the
    # query's parts is what the model reads (ruling OD-G); their encrypted items were withheld
    # and counted, their other blocks faced above.
    for carrier in TEXT_REPLAY_CARRIERS + ("_anthropic_content_blocks",):
        message.pop(carrier, None)
    malformed = _unparsed_argument_parts(message)
    given.added.extend(part["text"] for part in malformed)
    _add_parts(message, before, after + malformed)
    # The image rules, over every image the message now holds (the stored ones and those a
    # carrier or the stash held): sent where the model reads images, else a placeholder.
    if not facts.reads_images:
        present = {id(part) for part in (content_parts(message.get("content")) or [])}
        _replace_images_in_message(message, record, _NOT_KNOWN_STRICT if facts.reads_images is None
                                   else _NOT_READ_STRICT)
        given.added.extend(part["text"] for part in (content_parts(message.get("content")) or [])
                           if id(part) not in present and isinstance(part, dict) and isinstance(part.get("text"), str))
    if not _has_payload(message) and any(withheld_here.values()):
        message["content"] = [{"type": "text", "text": _ONLY_WITHHELD_REASONING_STRICT}]
        given.added.append(_ONLY_WITHHELD_REASONING_STRICT)
    if fill is not None:
        note = f"{FILL_NOTE_LABEL} {json.dumps(fill, ensure_ascii=False)}"
        _add_parts(message, [{"type": "text", "text": note}], [])
        given.added.append(note)
    return message


def summariser_message(raw: dict, record: str, facts: WireFacts,
                       withheld: Optional[dict[str, int]] = None) -> dict:
    """One record's message as the summariser receives it (see the module docstring).
    The encrypted items withheld from it are added to ``withheld`` by kind. (The query's
    strict projection is ``strict_message``.)"""
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
