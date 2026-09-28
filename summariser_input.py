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
from dataclasses import dataclass, field
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
# What the query gives its model of a record is closed (PLAN-83d §7): each stored message is
# given as its role, content, tool calls and readable reasoning; any stored value the query
# cannot give in that shape is given as a labelled rendering of its JSON in its place (the
# orchestrator's ruling OD-E, one level down); an image the model is not given is replaced by a
# placeholder that says so; signed or encrypted material is withheld and counted; the host's
# bookkeeping keys and a replay carrier's metadata are not given; null, empty strings, empty
# lists and blank reasoning carry nothing and are not shown. ``given`` receives every string the
# query gives from the record, with its path in the record as expansion shows it and its
# standing, so that an excerpt can be checked against the field it was drawn from, and the
# origin of every part it puts into the message's content, recorded where the part is made.

GIVEN_CONTENT, GIVEN_RESULT, GIVEN_CALL, GIVEN_REASONING = "content", "result", "call", "reasoning"

# Where a part of the strict message's content came from (PLAN-83d §2): a stored content part
# kept as it is; a stored content member, or the stored content itself, given as its JSON; a
# stored image replaced by its placeholder; a value of another stored key (readable reasoning,
# a carrier's image, the stash, a call not of the host's shape, an unknown key, arguments that
# are not JSON); a text of the query's own.
STORED, RENDERED, REPLACED, FIELD, NOTE = "stored", "rendered", "replaced", "field", "note"

# The keys the projection reads as the record's transcript; every other key is either the
# host's own bookkeeping (``_host_metadata_keys``) or unknown, and then given as labelled JSON
# (ruling OD-3a).
_TRANSCRIPT_KEYS = frozenset({
    "role", "content", "tool_calls", "tool_call_id", "reasoning", "reasoning_content", "reasoning_details",
    "codex_reasoning_items", "codex_message_items", "anthropic_content_blocks", "bedrock_content_blocks",
    "_anthropic_content_blocks", "api_content",
})
# The keys of an entry of a reasoning field, by the host's own classification (PLAN-83d §3; Hermes
# 375930d089). Readable text: the keys the host merges into the stored ``reasoning``
# (agent/agent_runtime_helpers.py ``extract_reasoning`` 1376-1378), the union of its readers'
# sets (context_compressor.py 1327-1337, auxiliary_client.py 7087-7088 and 8136,
# reasoning_summaries.py 39-46); no host reader treats a signature or data as making the text
# beside it unreadable (anthropic_message_convert.py 562-563 demotes signed thinking to text).
# Opaque: what the host calls the "signed/base64 envelope" (context_compressor.py 1328, 1365) and
# ciphertext (codex ``encrypted_content``; bedrock ``redactedContent*``): withheld and counted.
# Replay metadata: the host's backfill and stamp keys (reasoning_summaries.py 36, 71-73;
# codex_responses_adapter.py 1067-1083): not given.
_READABLE_DETAIL_KEYS = ("summary", "thinking", "content", "text")
_OPAQUE_DETAIL_KEYS = frozenset({"signature", "data", "encrypted_content", "redactedContentBase64", "redactedContent"})
_REPLAY_METADATA_KEYS = frozenset({"type", "id", "format", "index", "_issuer_kind", "_issuer_model"})
# The host's stored replay carriers of a message's reasoning (the query sends none of them, ruling
# OD-P2a on #83: their readable text is given as the message's reasoning or as its own part).
REASONING_REPLAY_CARRIERS = ("reasoning_details", "codex_reasoning_items")


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
    its path and standing (``values``), the record's problems that refuse the query
    (``problems``), the origin of every part it puts into the message's content (``origins``,
    recorded where the part is made, PLAN-83d §2), and counts for the header."""

    values: list
    problems: list
    origins: dict = field(default_factory=dict)       # id(part) -> (part, origin)
    reasoning_parts: int = 0

    def made(self, part: dict, origin: str) -> dict:
        self.origins[id(part)] = (part, origin)
        return part

    def origin(self, part: Any) -> Optional[str]:
        entry = self.origins.get(id(part))
        return entry[1] if entry is not None and entry[0] is part else None

    @property
    def added(self) -> list[str]:
        """The texts of every part that is not a stored content part kept as it is."""
        return [part["text"] for part, origin in self.origins.values()
                if origin != STORED and isinstance(part.get("text"), str)]


def _json_part(label: str, value: Any, path: tuple, standing: str, given: Given, origin: str = FIELD) -> dict:
    part = {"type": "text", "text": f"{label}\n{json.dumps(value, ensure_ascii=False)}"}
    given.values.extend((leaf_path, text, standing) for leaf_path, text in _leaves(value, path))
    return given.made(part, origin)


def image_part(part: Any) -> bool:
    """An image part by structure (``message_content.is_image_part``), for any stored value: a
    ``type`` that is not a string (a list, an object) is no image type and is never hashed."""
    return isinstance(part, dict) and isinstance(part.get("type"), str) and is_image_part(part)


def text_part(part: Any) -> bool:
    """A text part as the host writes it: exactly a ``type`` "text" and a string ``text``
    (agent/image_routing.py 558, turn_context.py 149-157 at Hermes 375930d089). A member with
    further keys is outside the query's domain and is given as its JSON, so no key of it is
    dropped (PLAN-83d §2)."""
    return (isinstance(part, dict) and set(part) == {"type", "text"} and part["type"] == "text"
            and isinstance(part["text"], str))


def _canonical_parts(parts: list, prefix: tuple, standing: str, given: Given) -> list:
    """A list content's members, each a text part, an image part, or a labelled part holding
    the member's JSON (plan §4.2)."""
    shown = []
    for index, part in enumerate(parts):
        if text_part(part):
            shown.append(given.made(part, STORED))
            given.values.append((prefix + (index, "text"), part["text"], standing))
        elif image_part(part):
            shown.append(given.made(part, STORED))
        else:
            shown.append(_json_part(f"[Member {index + 1} of the stored content is not a text part (a type and a "
                                    f"text only) or an image part; shown as its JSON by the query:]", part,
                                    prefix + (index,), standing, given, RENDERED))
    return shown


def _canonical_content(content: Any, standing: str, given: Given) -> Any:
    """The strict projection's content (plan §4.2): a string or None as stored; a list as its
    canonical members; the host's ``_multimodal`` envelope as its parts, every other key of it
    (its ``_multimodal`` mark included, so that the part says the content was stored as an
    envelope) in one labelled part; any other value (an object, a number, a boolean) as one
    labelled part holding its JSON (ruling OD-E)."""
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
        rest = {key: value for key, value in content.items() if key != "content"}
        shown.append(_json_part("[The rest of this stored multimodal envelope (every key but its content), shown as "
                                "its JSON by the query:]", rest, base, standing, given, RENDERED))
        return shown
    return [_json_part(f"[The stored content is a JSON {_json_kind(content)}, shown as its JSON by the query:]",
                       content, base, standing, given, RENDERED)]


def canonical_call(call: Any) -> bool:
    """A stored tool call of the shape the host writes (``_assistant_tool_call_dict``,
    agent/chat_completion_helpers.py 1622-1663 at Hermes 375930d089): a dict, a dict ``function``
    with a non-blank string ``name`` and a string ``arguments``, a non-blank string ``id``. The one
    test of which stored calls the query gives as calls on the wire (the projection and the
    query's labels both ask it)."""
    function = call.get("function") if isinstance(call, dict) else None
    return (isinstance(function, dict) and isinstance(function.get("name"), str) and bool(function["name"].strip())
            and isinstance(function.get("arguments"), str) and isinstance(call.get("id"), str)
            and bool(call["id"].strip()))


def _canonical_calls(message: dict, given: Given) -> list[dict]:
    """The message's tool calls as the wire can carry them (plan §4.2): a list of calls of
    the shape the host writes (a dict, a dict ``function`` with a non-blank string ``name``
    and a string ``arguments``, a non-blank string ``id``). Null or an empty list carries
    nothing and is dropped (the host's Anthropic converter raises on ``None``); every other value
    or call is given as a labelled part holding its JSON, after the content, and is not a call on
    the wire. Returns those parts."""
    calls = message.get("tool_calls")
    base = ("message", "tool_calls")
    if calls is None or (isinstance(calls, list) and not calls):
        message.pop("tool_calls", None)
        return []
    if not isinstance(calls, list):
        message.pop("tool_calls", None)
        return [_json_part(f"[The stored tool calls are a JSON {_json_kind(calls)}, not a list of calls; shown as "
                           f"their JSON by the query:]", calls, base, GIVEN_CALL, given)]
    kept, after = [], []
    for index, call in enumerate(calls):
        if canonical_call(call):
            function = call["function"]
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


def _path_text(path: tuple) -> str:
    return ".".join(map(str, path))


def _count(counts: dict, kind: str) -> None:
    counts[kind] = counts.get(kind, 0) + 1


def _not_text(value: Any, path: tuple, after: list, given: Given) -> None:
    """A value under a readable key of a reasoning field that is not a string: its JSON."""
    after.append(_json_part(f"[{_path_text(path)} as stored is not text; shown as its JSON by the query:]", value,
                            path, GIVEN_REASONING, given))


def _reasoning_entry(entry: dict, path: tuple, readable: list, after: list, given: Given) -> bool:
    """One entry of ``reasoning_details`` by the host's key classification (PLAN-83d §3): every
    readable key's string is readable text, signed or not (a structured value there is given as
    its JSON); an opaque key is withheld; replay metadata is not given; any other key is given as
    its JSON. A ``<provider>.native_assistant`` entry is another provider's private replay carrier
    (agent/transports/chat_completions.py:388, providers/base.py 100-102): its top-level readable
    strings are given (the host merged the first of them into ``reasoning``), everything else of
    it is withheld (ruling OD-P2b). Returns whether anything of the entry was withheld."""
    kind = entry.get("type")
    native = isinstance(kind, str) and kind.endswith(".native_assistant")
    withheld = False
    for key, value in entry.items():
        if key in _READABLE_DETAIL_KEYS and isinstance(value, str):
            _readable_block(path + (key,), value, readable)
        elif key in _REPLAY_METADATA_KEYS or value is None:
            continue
        elif native or key in _OPAQUE_DETAIL_KEYS:
            withheld = withheld or bool(value) or native
        elif key in _READABLE_DETAIL_KEYS:
            _not_text(value, path + (key,), after, given)
        else:
            after.append(_json_part(f"[{_path_text(path + (key,))} is a key of no kind the query knows; shown as its "
                                    f"JSON by the query:]", value, path + (key,), GIVEN_REASONING, given))
    return withheld


def _codex_item(item: dict, path: tuple, readable: list, after: list, given: Given) -> bool:
    """One Codex reasoning item (agent/codex_responses_adapter.py 1059-1084 at Hermes 375930d089):
    its summary texts, ``text`` and ``content`` are readable; ``encrypted_content`` is withheld;
    its type, id and issuer stamps are replay metadata; any other value is given as its JSON.
    Returns whether its encrypted content was withheld."""
    withheld = False
    for key, value in item.items():
        if value is None or key in _REPLAY_METADATA_KEYS:
            continue
        if key == "summary" and isinstance(value, list):
            for number, piece in enumerate(value):
                where = path + (key, number)
                if isinstance(piece, dict):
                    for inner, text in piece.items():
                        if inner == "text" and isinstance(text, str):
                            _readable_block(where + (inner,), text, readable)
                        elif inner != "type" and text is not None:
                            after.append(_json_part(f"[{_path_text(where + (inner,))} is a key of no kind the query "
                                                    f"knows; shown as its JSON by the query:]", text,
                                                    where + (inner,), GIVEN_REASONING, given))
                elif piece is not None:
                    _not_text(piece, where, after, given)
        elif key in ("summary", "text", "content"):
            if isinstance(value, str):
                _readable_block(path + (key,), value, readable)
            else:
                _not_text(value, path + (key,), after, given)
        elif key in _OPAQUE_DETAIL_KEYS:
            withheld = withheld or bool(value)
        else:
            after.append(_json_part(f"[{_path_text(path + (key,))} is a key of no kind the query knows; shown as its "
                                    f"JSON by the query:]", value, path + (key,), GIVEN_REASONING, given))
    return withheld


def _not_a_list(key: str, value: Any, after: list, given: Given, standing: str) -> None:
    after.append(_json_part(f"[{key} as stored is not a list; shown as its JSON by the query:]", value,
                            ("message", key), standing, given))


def _reasoning_and_carriers(raw: dict, message: dict, given: Given, opaque: dict,
                            standing: str) -> tuple[list, list]:
    """The reasoning fields and replay carriers of a record, faced entry by entry and block by
    block (PLAN-83d §3, §7). Returns the labelled parts to put before the stored content
    (readable reasoning) and after it (values the query renders). Readable text held anywhere
    but the text the query already gives as the record's readable reasoning, or an earlier
    part, is given as its own labelled part naming its path (exact containment, ruling OD-4a); a
    carrier's message text must stand in the given content and a carrier's call must be one of
    the stored calls, else the record is a problem that refuses the query; an image a carrier
    holds is given as an image part; an unknown block is given as its JSON; a value outside
    content, calls and reasoning takes the record's own ``standing``. ``opaque`` counts, per
    field, the entries and blocks whose signature, encrypted content or opaque data is
    withheld."""
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
            path = ("message", "reasoning_details", index)
            if isinstance(entry, dict):
                kind = entry.get("type")
                if _reasoning_entry(entry, path, readable, after, given):
                    _count(opaque, f"reasoning_details ({kind}, another provider's private replay carrier)"
                           if isinstance(kind, str) and kind.endswith(".native_assistant") else "reasoning_details")
            elif entry is not None:
                after.append(_json_part(f"[reasoning_details[{index}] as stored is not an entry; shown as its JSON "
                                        f"by the query:]", entry, path, GIVEN_REASONING, given))
    elif details is not None:
        _not_a_list("reasoning_details", details, after, given, GIVEN_REASONING)
    items = raw.get("codex_reasoning_items")
    if isinstance(items, list):
        for index, item in enumerate(items):
            path = ("message", "codex_reasoning_items", index)
            if isinstance(item, dict):
                if _codex_item(item, path, readable, after, given):
                    _count(opaque, "codex_reasoning_items")
            elif item is not None:
                after.append(_json_part(f"[codex_reasoning_items[{index}] as stored is not an item; shown as its "
                                        f"JSON by the query:]", item, path, GIVEN_REASONING, given))
    elif items is not None:
        _not_a_list("codex_reasoning_items", items, after, given, GIVEN_REASONING)
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
            after.append(_json_part(f"[{_path_text(path)} is an image block of no shape the query reads; shown as its "
                                    f"JSON by the query:]", block, path, standing, given))
            return
        label = {"type": "text", "text": f"[An image the provider returned in {_path_text(path)}, which the stored "
                                         f"content does not hold:]"}
        after.extend([given.made(label, FIELD), given.made({"type": "image_url", "image_url": {"url": url}}, FIELD)])

    def text_block(text: Any, path: tuple) -> None:
        """A carrier's text block: a string is checked as carried text (a blank one carries
        nothing); any other value is given as its JSON."""
        if isinstance(text, str):
            carried(text, path)
        elif text is not None:
            after.append(_json_part(f"[{_path_text(path)} as stored is not text; shown as its JSON by the query:]",
                                    text, path, standing, given))

    blocks = raw.get("anthropic_content_blocks")
    for index, block in enumerate(blocks if isinstance(blocks, list) else []):
        path = ("message", "anthropic_content_blocks", index)
        kind = block.get("type") if isinstance(block, dict) else None
        if kind == "text":
            text_block(block.get("text"), path + ("text",))
        elif kind == "thinking":
            _readable_block(path + ("thinking",), block.get("thinking"), readable)
            if block.get("signature"):
                _count(opaque, "anthropic_content_blocks")
        elif kind == "redacted_thinking":
            _count(opaque, "anthropic_content_blocks")   # encrypted: withheld and counted
        elif kind == "tool_use":
            called(block.get("name"), block.get("input"), path)
        elif kind == "image":
            image(block, path)
        else:
            after.append(_json_part(f"[anthropic_content_blocks[{index}] is a block of no type the query knows; shown "
                                    f"as its JSON by the query:]", block, path, standing, given))
    if blocks is not None and not isinstance(blocks, list):
        _not_a_list("anthropic_content_blocks", blocks, after, given, standing)
    blocks = raw.get("bedrock_content_blocks")
    for index, block in enumerate(blocks if isinstance(blocks, list) else []):
        path = ("message", "bedrock_content_blocks", index)
        if isinstance(block, dict) and "text" in block:
            text_block(block["text"], path + ("text",))
        elif isinstance(block, dict) and isinstance(block.get("reasoningContent"), dict):
            reasoning = block["reasoningContent"]
            nested = reasoning.get("reasoningText") if isinstance(reasoning.get("reasoningText"), dict) else {}
            for key, text in (("text", reasoning.get("text")), ("reasoningText.text", nested.get("text"))):
                _readable_block(path + ("reasoningContent",) + tuple(key.split(".")), text, readable)
            if _signed_bedrock_block(block):
                _count(opaque, "bedrock_content_blocks")
        elif isinstance(block, dict) and isinstance(block.get("toolUse"), dict):
            called(block["toolUse"].get("name"), block["toolUse"].get("input"), path)
        else:
            after.append(_json_part(f"[bedrock_content_blocks[{index}] is a block of no shape the query knows; shown "
                                    f"as its JSON by the query:]", block, path, standing, given))
    if blocks is not None and not isinstance(blocks, list):
        _not_a_list("bedrock_content_blocks", blocks, after, given, standing)
    items = raw.get("codex_message_items")
    for index, item in enumerate(items if isinstance(items, list) else []):
        path = ("message", "codex_message_items", index)
        parts = item.get("content") if isinstance(item, dict) and item.get("type") == "message" else None
        if not isinstance(parts, list):
            after.append(_json_part(f"[codex_message_items[{index}] is an item of no shape the query knows; shown as "
                                    f"its JSON by the query:]", item, path, standing, given))
            continue
        for number, part in enumerate(parts):
            if isinstance(part, dict) and part.get("type") in ("output_text", "text") and "text" in part:
                text_block(part["text"], path + ("content", number, "text"))
            else:
                after.append(_json_part(f"[codex_message_items[{index}].content[{number}] is a part of no shape the "
                                        f"query knows; shown as its JSON by the query:]", part,
                                        path + ("content", number), standing, given))
    if items is not None and not isinstance(items, list):
        _not_a_list("codex_message_items", items, after, given, standing)
    stash = raw.get("_anthropic_content_blocks")
    for index, block in enumerate(stash if isinstance(stash, list) else []):
        path = ("message", "_anthropic_content_blocks", index)
        kind = block.get("type") if isinstance(block, dict) else None
        if kind == "text" and isinstance(block.get("text"), str):
            if block["text"].strip() and not any(block["text"] in text for text in content_texts):
                part = {"type": "text", "text": f"[A text block the host stashed in _anthropic_content_blocks[{index}]"
                                                f", which the stored content does not hold:]\n{block['text']}"}
                given.values.append((path + ("text",), block["text"], standing))
                after.append(given.made(part, FIELD))
        elif kind == "redacted_thinking":
            _count(opaque, "anthropic_content_blocks")   # encrypted: withheld and counted
        elif kind == "thinking":
            _readable_block(path + ("thinking",), block.get("thinking"), readable)
            if block.get("signature"):
                _count(opaque, "anthropic_content_blocks")
        elif kind == "image" or image_part(block):
            if image_part(block) and kind != "image":
                label = {"type": "text", "text": f"[An image the host stashed in _anthropic_content_blocks[{index}]:]"}
                after.extend([given.made(label, FIELD), given.made(block, FIELD)])
            else:
                image(block, path)
        elif block is not None:
            after.append(_json_part(f"[_anthropic_content_blocks[{index}] is a block of no type the query knows; shown "
                                    f"as its JSON by the query:]", block, path, standing, given))
    if stash is not None and not isinstance(stash, list):
        _not_a_list("_anthropic_content_blocks", stash, after, given, standing)
    # Readable reasoning: the main field, then every other readable text not contained in
    # the main field or an earlier such part (exact containment, ruling OD-4a).
    shown: list[str] = []
    if main:
        before.append(given.made({"type": "text", "text": f"{READABLE_REASONING_LABEL}\n{main}"}, FIELD))
        given.values.append((("lcm", "reasoning") if role == "assistant" else
                             ("message", "reasoning" if raw.get("reasoning") == main else "reasoning_content"),
                             main, GIVEN_REASONING))
        shown.append(main)
    for path, text in readable:
        if any(text in earlier for earlier in shown):
            continue
        part = {"type": "text", "text": f"[Readable reasoning the provider returned in {_path_text(path)}, "
                                        f"as stored:]\n{text}"}
        before.append(given.made(part, FIELD))
        given.values.append((path, text, GIVEN_REASONING))
        given.reasoning_parts += 1
        shown.append(text)
    return before, after


def _unknown_keys(message: dict, given: Given, standing: str) -> list[dict]:
    """A key that is neither the record's transcript nor the host's bookkeeping is given as a
    labelled part holding its JSON (ruling OD-3a), under the record's own standing, and removed
    from the message: the rendering stands in its place (PLAN-83d §7)."""
    known = _host_metadata_keys()
    unknown = [key for key in list(message) if key not in _TRANSCRIPT_KEYS and key not in known
               and not str(key).startswith("_")]
    return [_json_part(f"[The stored key {key!r} is no field the query knows; shown as its JSON by the query:]",
                       message.pop(key), ("message", key), standing, given) for key in unknown]


def _assemble(message: dict, before: list[dict], after: list[dict], given: Given) -> None:
    """The strict message's content: ``before``, the stored content's parts, ``after``. A stored
    string becomes one stored text part; an empty one carries nothing. With nothing to add the
    content stays as it is (a string stays a string)."""
    if not before and not after:
        return
    content = message.get("content")
    if isinstance(content, list):
        stored = list(content)
    elif isinstance(content, str) and content:
        stored = [given.made({"type": "text", "text": content}, STORED)]
    else:
        stored = []
    message["content"] = before + stored + after


def strict_message(raw: dict, record: str, facts: WireFacts, withheld: dict, given: Given) -> dict:
    """One record's message as the query gives it (PLAN-83d §2-§3, §7): the host's per-row rules,
    each host function called strictly, then the closed domain. ``given`` receives what is given
    of the record and the origin of every part of its content; ``withheld`` counts, per stored
    field, the entries and blocks whose signature, encrypted content or opaque data is withheld
    (their readable text is given). No replay carrier of the message's text or reasoning is sent,
    nor the host's stash (rulings OD-G, OD-P2a)."""
    row = _row_before_fill(raw, needs_echo=facts.needs_reasoning_echo, strict=True)
    fill = host_fill_text(row)
    message = copy.deepcopy(row)
    message.pop("_length_continuation_fragment", None)
    message.pop("_length_continuation_nudge", None)
    standing = GIVEN_RESULT if raw.get("role") == "tool" else GIVEN_CONTENT
    if "content" in message:
        message["content"] = _canonical_content(message["content"], standing, given)
    call_parts = _canonical_calls(message, given)
    opaque: dict[str, int] = {}
    before, after = _reasoning_and_carriers(raw, message, given, opaque, standing)
    after = call_parts + after + _unknown_keys(message, given, standing)
    for kind, count in opaque.items():
        withheld[kind] = withheld.get(kind, 0) + count
    # The host's replay carriers of the message's text and of its reasoning, and the private
    # stash, are not sent: the stored content with the query's parts is what the model reads
    # (rulings OD-G, OD-P2a); their readable text was given, their opaque material withheld and
    # counted, their other blocks faced above.
    for carrier in TEXT_REPLAY_CARRIERS + REASONING_REPLAY_CARRIERS + ("_anthropic_content_blocks",):
        message.pop(carrier, None)
    malformed = [given.made(part, FIELD) for part in _unparsed_argument_parts(message)]
    _assemble(message, before, after + malformed, given)
    # The image rules, over every image the message now holds (the stored ones and those a
    # carrier or the stash held): sent where the model reads images, else a placeholder.
    if not facts.reads_images and isinstance(message.get("content"), list):
        why = _NOT_KNOWN_STRICT if facts.reads_images is None else _NOT_READ_STRICT
        message["content"] = [given.made(_image_placeholder(part, record, why), REPLACED) if image_part(part)
                              else part for part in message["content"]]
    if not _has_payload(message) and any(opaque.values()):
        message["content"] = [given.made({"type": "text", "text": _ONLY_WITHHELD_REASONING_STRICT}, NOTE)]
    if fill is not None:
        note = f"{FILL_NOTE_LABEL} {json.dumps(fill, ensure_ascii=False)}"
        _assemble(message, [given.made({"type": "text", "text": note}, NOTE)], [], given)
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
