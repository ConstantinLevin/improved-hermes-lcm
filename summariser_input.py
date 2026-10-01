"""Prepare recorded originals for an original reader, without replaying the main agent.

Source snapshots and row ownership are explicit. Original content is retained;
labelled annotations expose the other recorded values, including readable reasoning
and metadata. Only identified opaque fields are withheld, with their paths named.
Ambiguous sidecar purpose and unsupported native carriers refuse that record.

This is preparation evidence, not evidence of dispatch or provider receipt. The
actual invocation must establish whether its selected conversion preserves this
input. The strict host helpers below remain for expansion's separate replay view.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from .message_content import content_parts, is_image_part, readable_reasoning, sent_content, sidecar_sent

# The closing request belongs to the mechanism, never to a recorded original.
CLOSING_REQUEST = "Summarize the conversation above, as the system instructions say."
# Expansion's existing display view selects reasoning, then reasoning_content.
# Original-reader preparation below exposes every recorded field independently.
_readable_reasoning = readable_reasoning

SourcePath = tuple[str | int, ...]


@dataclass(frozen=True)
class WithheldField:
    source_path: SourcePath
    kind: str
    reason: str


@dataclass(frozen=True)
class ReaderSource:
    record: str
    request_index: int
    original_json: str
    prepared_json: str
    readable_paths: tuple[SourcePath, ...]
    withheld: tuple[WithheldField, ...]


@dataclass(frozen=True)
class ReaderInput:
    # Request rows are owned mutable copies; the evidence uses immutable snapshots.
    messages: list[dict]
    sources: tuple[ReaderSource, ...]


class ReaderUnavailable(Exception):
    def __init__(self, record: str, source_path: SourcePath, reason: str):
        self.record, self.source_path, self.reason = record, source_path, reason
        super().__init__(f"record {record}, field {source_path!r}: {reason}")


@dataclass(frozen=True)
class WireFacts:
    """Prospective route hints for planning, not evidence of an actual invocation.
    Original-reader preparation retains images independently of these hints."""

    reads_images: Optional[bool]
    needs_reasoning_echo: bool
    anthropic_converter: bool


class HostUnavailable(Exception):
    """A host function expansion's strict replay projection needs cannot be read."""


def _strict_import(what: str, module: str, name: str) -> Any:
    try:
        return getattr(__import__(module, fromlist=[name]), name)
    except Exception as exc:
        raise HostUnavailable(f"the host's {what} ({module}.{name}) cannot be read ({type(exc).__name__}: {exc}), "
                              f"so the message as the host sends it is not known") from None


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

def _row_before_fill(raw: dict, *, needs_echo: bool) -> dict:
    """``build_api_messages``' per-row steps up to the fill (agent/turn_context.py:1221-1260):
    the clone, the sidecar, the persistence fields, the reasoning copy, the pops."""
    message = _host_clone(raw, True)
    persistence_only = _strict_import("persistence fields", "agent.message_metadata",
                                     "PERSISTENCE_ONLY_MESSAGE_FIELDS")
    message.pop("api_content", None)
    for key in persistence_only:
        message.pop(key, None)
    if sidecar_sent(raw):                     # one rule with what grep searches (#18 D2)
        message["content"] = sent_content(raw)
    _host_reasoning_policy(raw, message, needs_echo, True)
    message.pop("reasoning", None)
    message.pop("finish_reason", None)
    return message




def host_row_before_fill(raw: dict, *, pad: bool) -> dict:
    """The record's row as the host builds it for a request up to its fill, with the host's
    reasoning pad for the route (``host_reasoning_pad``), every host function called
    strictly: the host's own input to ``fill_empty_non_final_wire_payload``."""
    return _row_before_fill(raw, needs_echo=pad)


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

# The orchestrator's ruling on #8b: a summariser the model table has no row for is not
# said not to read images; its images are not sent, and it is said that it is not known.








def _image_count(message: dict) -> int:
    count = sum(1 for p in (content_parts(message.get("content")) or []) if is_image_part(p))
    stashed = message.get("_anthropic_content_blocks")
    if not count and isinstance(stashed, list):
        count = sum(1 for p in stashed if is_image_part(p))
    return count


def image_count(message: dict) -> int:
    """Count image parts structurally in one prepared message."""
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


# --- The original reader --------------------------------------------------------------

_NATIVE_CARRIERS = frozenset({
    "reasoning_details", "codex_reasoning_items", "codex_message_items",
    "anthropic_content_blocks", "_anthropic_content_blocks", "bedrock_content_blocks",
})
_CARRIER_TYPES = {
    "reasoning_details": frozenset({"reasoning.text", "reasoning.summary", "reasoning.encrypted",
                                  "thinking", "redacted_thinking"}),
    "codex_reasoning_items": frozenset({"reasoning"}),
    "codex_message_items": frozenset({"reasoning", "message", "function_call", "function_call_output"}),
    "anthropic_content_blocks": frozenset({"text", "thinking", "redacted_thinking", "tool_use", "tool_result"}),
    "_anthropic_content_blocks": frozenset({"text", "thinking", "redacted_thinking", "tool_use", "tool_result"}),
}


def _reader_json(value: Any, record: str, path: SourcePath) -> str:
    def check(node: Any, where: SourcePath) -> None:
        if isinstance(node, dict):
            for key, child in node.items():
                if not isinstance(key, str):
                    raise ReaderUnavailable(record, where, "the original object has a non-string field name")
                check(child, (*where, key))
        elif isinstance(node, list):
            for index, child in enumerate(node):
                check(child, (*where, index))
        elif node is not None and not isinstance(node, (str, bool, int, float)):
            raise ReaderUnavailable(record, where, "unsupported original JSON value")

    try:
        check(value, path)
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError, RecursionError) as exc:
        raise ReaderUnavailable(record, path, f"the original JSON value cannot be represented ({exc})") from None


def _field_parts(value: Any, path: SourcePath, readable: list[SourcePath]) -> list[dict]:
    """Expose recorded field values with their types; string values remain verbatim."""
    label = f"[Recorded original field {json.dumps(path, ensure_ascii=False)}"
    readable.append(path)
    if isinstance(value, dict):
        parts = [{"type": "text", "text": label + "; retained object keys]\n"
                  + json.dumps(list(value), ensure_ascii=False)}]
        for key, child in value.items():
            parts.extend(_field_parts(child, (*path, key), readable))
        return parts
    if isinstance(value, list):
        parts = [{"type": "text", "text": label + f"; array length {len(value)}]"}]
        for index, child in enumerate(value):
            parts.extend(_field_parts(child, (*path, index), readable))
        return parts
    if isinstance(value, str):
        text = label + f"; string length {len(value)}]\n" + value
    else:
        text = label + "; JSON value]\n" + json.dumps(value, ensure_ascii=False, allow_nan=False)
    return [{"type": "text", "text": text}]


def _carrier_projection(value: Any, path: SourcePath, kind: str, record: str,
                        omitted: list[WithheldField], *, native: bool = True) -> Any:
    """Copy known native fields, omitting opaque fields rather than their neighboring text."""
    if isinstance(value, list):
        return [_carrier_projection(child, (*path, index), kind, record, omitted, native=native)
                for index, child in enumerate(value)]
    if not isinstance(value, dict):
        return value
    block_type = value.get("type") if native else None
    if native and "type" in value and not isinstance(block_type, str):
        raise ReaderUnavailable(record, (*path, "type"), "unsupported native-carrier type value")
    if native and (block_type in {"image", "image_url", "input_image", "input_audio", "audio", "video"}
                   or (kind == "bedrock_content_blocks" and "image" in value)):
        raise ReaderUnavailable(record, path, "native multimedia has no established original-reader carriage")
    result = {}
    for key, child in value.items():
        reason = None
        if native and key == "signature" and (
            block_type in {"thinking", "reasoning.text", "reasoning.summary"}
            or (kind == "bedrock_content_blocks" and path[-1] in {"reasoningContent", "reasoningText"})
        ):
            reason = "opaque replay signature"
        elif native and key == "data" and block_type in {"reasoning.encrypted", "redacted_thinking"}:
            reason = "declared encrypted reasoning payload"
        elif native and key == "encrypted_content" and block_type == "reasoning":
            reason = "declared encrypted reasoning payload"
        elif native and kind == "bedrock_content_blocks" and key in {"redactedContent", "redactedContentBase64"}:
            reason = "declared redacted reasoning payload"
        if reason is not None and child not in (None, ""):
            if not isinstance(child, str):
                raise ReaderUnavailable(record, (*path, key), "unsupported opaque-field value")
            omitted.append(WithheldField((*path, key), kind, reason))
            continue
        # Tool arguments are ordinary recorded values, even when their keys resemble native fields.
        child_native = native and key in {"content", "summary", "reasoningContent", "reasoningText"}
        result[key] = _carrier_projection(child, (*path, key), kind, record, omitted, native=child_native)
    return result


def _reader_carrier(value: Any, key: str, record: str, omitted: list[WithheldField]) -> Any:
    if value is None or value == []:
        return copy.deepcopy(value)
    if not isinstance(value, list):
        raise ReaderUnavailable(record, (key,), "unsupported native-carrier container")
    for index, entry in enumerate(value):
        if not isinstance(entry, dict):
            raise ReaderUnavailable(record, (key, index), "unsupported native-carrier item")
        if key in _CARRIER_TYPES and (
            not isinstance(entry.get("type"), str) or entry["type"] not in _CARRIER_TYPES[key]
        ):
            raise ReaderUnavailable(record, (key, index, "type"), "native-carrier meaning is not established")
        if key == "bedrock_content_blocks" and not any(
            field in entry for field in ("text", "reasoningContent", "toolUse", "toolResult")
        ):
            raise ReaderUnavailable(record, (key, index), "Bedrock carrier meaning is not established")
    return _carrier_projection(value, (key,), key, record, omitted)


def _reader_content(raw: dict, record: str, readable: list[SourcePath]) -> list[dict]:
    if "content" not in raw:
        return [{"type": "text", "text": "[The recorded original has no content field]"}]
    value = raw["content"]
    if isinstance(value, str) and value:
        readable.append(("content",))
        return [{"type": "text", "text": value}]
    parts = content_parts(value)
    if parts is not None:
        for index, part in enumerate(parts):
            if not isinstance(part, dict) or not isinstance(part.get("type"), str) or part["type"] not in {
                "text", "input_text", "output_text", "image_url", "image", "input_image",
            }:
                raise ReaderUnavailable(record, ("content", index), "unsupported original content part")
            if part.get("type") in {"text", "input_text", "output_text"} and not isinstance(part.get("text"), str):
                raise ReaderUnavailable(record, ("content", index, "text"), "unsupported original text-part value")
        path = ("content", "content") if isinstance(value, dict) else ("content",)
        readable.append(path)
        result = copy.deepcopy(parts)
        if isinstance(value, dict):
            result.append({"type": "text", "text": "[Recorded original content envelope; object keys]\n"
                           + json.dumps(list(value), ensure_ascii=False)})
            for key, child in value.items():
                if key != "content":
                    result.extend(_field_parts(child, ("content", key), readable))
        if not parts:
            result.extend(_field_parts([], path, readable))
        return result
    if isinstance(value, dict):
        raise ReaderUnavailable(record, ("content",), "original content-object meaning is not established")
    # Empty strings, null, numbers and booleans are source values, never invented utterances.
    return _field_parts(value, ("content",), readable)


def summariser_message(raw: dict, record: str, facts: WireFacts,
                       withheld: Optional[dict[str, int]] = None, *,
                       _sources: Optional[list[ReaderSource]] = None, _request_index: int = 0) -> dict:
    """Own one original and expose its values; actual transport eligibility is the caller's."""
    original = copy.deepcopy(raw)
    if not isinstance(original, dict):
        raise ReaderUnavailable(record, (), "the recorded original is not an object")
    original_json = _reader_json(original, record, ())
    role = original.get("role")
    if not isinstance(role, str) or not role:
        raise ReaderUnavailable(record, ("role",), "the original role is not a nonempty string")
    sidecar = original.get("api_content")
    if sidecar not in (None, "") and sidecar != original.get("content"):
        raise ReaderUnavailable(record, ("api_content",),
                                "the differing replay sidecar's event/mechanism purpose is not established")
    readable: list[SourcePath] = [("role",)]
    omitted: list[WithheldField] = []
    message = {"role": role, "content": _reader_content(original, record, readable)}
    for key, value in original.items():
        if key in {"role", "content"}:
            continue
        projected = _reader_carrier(value, key, record, omitted) if key in _NATIVE_CARRIERS else value
        message["content"].extend(_field_parts(projected, (key,), readable))
        if key in {"tool_calls", "tool_call_id", "name"}:
            message[key] = copy.deepcopy(value)
    for field in omitted:
        message["content"].append({"type": "text", "text":
            f"[Recorded original field {json.dumps(field.source_path, ensure_ascii=False)} withheld: {field.reason}]"})
        if withheld is not None:
            withheld[field.kind] = withheld.get(field.kind, 0) + 1
    # A container containing an omitted field is not a wholly readable original value.
    readable = [path for path in readable if not any(
        field.source_path[:len(path)] == path for field in omitted
    )]
    if _sources is not None:
        _sources.append(ReaderSource(
            record, _request_index, original_json, _reader_json(message, record, ()),
            tuple(readable), tuple(omitted),
        ))
    return message


def summariser_messages(
    records: Iterable[tuple[str, dict]],
    *,
    instructions: str,
    request: dict,
    facts: WireFacts,
    withheld: Optional[dict[str, int]] = None,
) -> ReaderInput:
    """Prepare owned messages and explicit original/row evidence for one invocation."""
    sources: list[ReaderSource] = []
    body = [summariser_message(raw, record, facts, withheld, _sources=sources, _request_index=index)
            for index, (record, raw) in enumerate(records, start=1)]
    closing = CLOSING_REQUEST
    if request:
        closing += "\nrequest: " + json.dumps(request, ensure_ascii=False)
    messages = [{"role": "system", "content": instructions}] + body + [{"role": "user", "content": closing}]
    return ReaderInput(messages, tuple(sources))
