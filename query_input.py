"""What the query gives its model of one stored record (#19, PR #83; PLAN-83d, PLAN-83e, PLAN-83g).

The query's description states a rule to the agent: each stored message is given as its role,
content, tool calls and readable reasoning; any other stored value is given as labelled JSON; an
image the model is not given as an image is replaced by a placeholder that says so, or named where
it stood and counted; material the host treats as opaque replay is withheld and counted wherever it
is stored outside the transcript's own content, tool-call arguments and tool results; the host's
bookkeeping keys and a replay carrier's metadata are not given; null, empty and blank values carry
nothing and are not given as parts. Each clause is made true here by one mechanism every record
passes through, never by a sentence applied at each site by hand:

- ``carries_nothing`` is the one test of "carries nothing", asked at every place the query
  decides whether and how to give a stored value (a key, a member of a list it walks, a key of
  an entry or block it classifies); inside a value given as JSON nothing is removed, since the
  JSON is the stored value; a content that carries nothing is given as "" (PLAN-83g §3.6);
- ``_render`` is the only place a stored value becomes labelled JSON, and it walks the value
  carrying the kind of the stored field it came from (M1, PLAN-83g §3.1): the kind changes only
  where table T says (``_step``), a value under a key of the host's opaque vocabulary is withheld
  in every kind but the transcript's own content and a call's name and arguments, and every leaf
  is recorded with the standing its kind gives it (M2);
- ``_image_outcome`` is the one function that decides every image the query meets (M3', PLAN-19
  §2.2): given as an image (as stored) where the host's own per-part function delivers it on every
  leg, replaced by a placeholder, named where it stood (a field the query does not give as an image,
  or an image the host's converter does not deliver on a leg, with its own text), or, where the legs
  disagree, a problem that refuses the query; each marker or label written from that decision;
- one key classification (``_CLASSES``), read by one walker per stored field, faces every key of
  every entry of every reasoning field and every block of every replay carrier, and of every
  stored tool call (PLAN-83e §4); a field stored on a role whose domain lacks it is walked by its
  own walker, its outputs given as parts (M4);
- the message sent is built positively from the stored role's domain (``strict_message``,
  PLAN-83e §5): role, content, the calls an agent message gives as calls (M-PAIR, PLAN-19 §2.4;
  every other call of the host's shape given as labelled JSON), the call id on a record sent as a
  tool result (a stored tool result no call given as a call answers is sent as a user message,
  without it, its id on its label), and ``reasoning_content`` as the host's echo policy leaves it;
  every other stored key is given as a part, withheld and counted, or the host's bookkeeping; the
  standing of what is given is the stored role's (PLAN-19 §2.8);
- every part of the content is recorded with its origin where it is made (``Given``), so that
  what a label says about a part comes from how the part was made, and the query asserts that
  every part has one (``unrecorded_parts``); a stored place and a stored role are each written one
  way (``_path_text``, ``stored_role_text``, M5).

Host facts at Hermes 375930d089 (PR #83, PLAN-83e §0, PLAN-83g §0: the readers and the host's own
lines named where they are used).
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from .message_content import content_parts, image_media_type, is_image_part, readable_reasoning, sidecar_sent
from .summariser_input import (HostUnavailable, WireFacts, _image_placeholder, _row_before_fill, _strict_import,
                               host_fill_text)

# The standing of a value the query gives: a claim about its origin, which only the writer of the
# field settles (PLAN-19 re-derived, M-STANDING; Hermes 375930d089). Four the host's writers
# settle: the message's content as stored (chat_completion_helpers.py 1608-1619, 1675; the codex
# projector verbatim, codex_event_projector.py 69-78), a tool result, a call's name and arguments
# (1622-1663; a carrier's tool_use/toolUse block, transports/anthropic.py 305-306, bedrock_adapter.py
# toolUse), the readable reasoning (agent_runtime_helpers.py 1362-1395; a carrier's thinking,
# reasoningText and summary texts; a codex message item the host stamps as a commentary or analysis
# phase, codex_responses_adapter.py 1125-1139). Three no writer settles, each named by what it is
# and given all the same: ``sidecar``, the text the host sends in place of a message's stored
# content (``api_content``; on an agent message from the model's reasoning when its reply had no
# content, turn_final_response.py 225/276-277/346, from a hook's output, 346, or its own
# interruption placeholder, turn_api_call.py 200-203, conversation_loop.py 348; on a user message
# the user's text with what the host injected, turn_context.py 888-932, session_persistence.py
# 192-231, conversation_loop.py 355, turn_facade_lease.py 343; no producer recorded); ``carrier``,
# a text a replay carrier or the stash holds that the content does not (the carriers keep the
# provider's text before the host's strip, redaction and flattening, whose classes the carrier does
# not record, transports/anthropic.py 65-106, bedrock_adapter.py 879-899, codex_responses_adapter.py
# 1124-1139; the stash has no writer at this host); ``stored``, a value under a key no producer of
# this host writes on the message's role.
GIVEN_CONTENT, GIVEN_RESULT, GIVEN_CALL, GIVEN_REASONING = "content", "result", "call", "reasoning"
GIVEN_SIDECAR, GIVEN_CARRIER, GIVEN_STORED = "sidecar", "carrier", "stored"

# Where a part of the message's content came from, recorded where the part is made: a stored
# content part kept as it is; a stored content member, or the stored content itself, given as its
# JSON; a placeholder in place of an image; a value of another stored key; a text of the query's
# own; an image lifted out of a stored content value given as JSON; the query's label.
STORED, RENDERED, REPLACED, FIELD, NOTE, LIFTED, LABEL = (
    "stored", "rendered", "replaced", "field", "note", "lifted", "label")

# The stored replay carriers of a message's text (the host's converters replay them instead of
# reading ``content``) and of its reasoning, and the host's private stash of a tool result's
# blocks: the query sends none of them (rulings OD-G, OD-P2a); each is faced key by key.
TEXT_REPLAY_CARRIERS = ("codex_message_items", "anthropic_content_blocks", "bedrock_content_blocks")
REASONING_REPLAY_CARRIERS = ("reasoning_details", "codex_reasoning_items")
STASH = "_anthropic_content_blocks"

# The keys the host's producers write per role (Hermes 375930d089, reader H1 of PLAN-83e):
# tool calls, the reasoning fields and the carriers on agent messages only
# (agent/chat_completion_helpers.py 1666-1761), the call id on tool results
# (agent/tool_dispatch_helpers.py 434-466), the sidecar on user and agent messages
# (agent/turn_context.py 904, agent/turn_final_response.py 225-346); the stash has no writer and
# is read on tool results (agent/anthropic_message_convert.py 443-447). Any other stored role
# (a client's history on the API server keeps any role string, a role-less row reloads as
# "unknown") is given as a user message (ruling OD-E2) whose domain is its content.
_TRANSCRIPT_KEYS = frozenset({
    "content", "api_content", "tool_calls", "tool_call_id", "reasoning", "reasoning_content",
    *REASONING_REPLAY_CARRIERS, *TEXT_REPLAY_CARRIERS, STASH,
})
# ``name`` is written on tool results only (the tool's name, which the call's label names:
# bookkeeping); no host producer writes it on a user, agent or system message (reader H4 of
# PLAN-83e: agent/tool_dispatch_helpers.py 450-456, turn_tool_round.py 106-113,
# turn_tool_validation.py 44-52, turn_loop_errors.py 136-141; every user and agent producer builds
# its dict without it), so there it is outside the role's domain.
_TRANSCRIPT_KEYS = _TRANSCRIPT_KEYS | {"name"}
_DOMAIN = {
    "user": frozenset({"content", "api_content"}),
    "assistant": frozenset({"content", "api_content", "tool_calls", "reasoning", "reasoning_content",
                            *REASONING_REPLAY_CARRIERS, *TEXT_REPLAY_CARRIERS}),
    "tool": frozenset({"content", "tool_call_id", "name", STASH}),
}
_OTHER_DOMAIN = frozenset({"content"})
_WIRE_ROLES = frozenset(_DOMAIN)

# Keys of a message the query never gives, as the host's own bookkeeping beside the constants it
# reads at run time (``_host_metadata_keys``): the host's per-message stamps and a tool result's
# tool name (the call's label names the call), and a cache marker (agent/prompt_caching.py writes
# it on request copies only). ``name``: see ``_DOMAIN``.
_BOOKKEEPING_KEYS = frozenset({"finish_reason", "timestamp", "tool_name", "effect_disposition", "cache_control"})

FILL_NOTE_LABEL = "[The host sends this empty message to the provider with its own stand-in as content:]"
_ONLY_WITHHELD = ("[This message held only reasoning the query does not give (signed, encrypted, or another "
                  "provider's private replay carrier), counted in the header's encrypted_withheld]")
_NOT_READ = "not shown to this model, which does not read images"
_NOT_KNOWN = "image not sent: whether this model reads images is unknown"


@dataclass(frozen=True)
class EchoInputs:
    """The two inputs of the reasoning pad the host's own agent applies on the query's route, each
    as the host's own function reads it (PLAN-83g §3.8, ruling OD-F3)."""

    provider: str
    model: str
    base_url: str
    family: bool          # agent.message_sanitization.needs_reasoning_echo(provider, model, base_url)
    opt_in: bool          # model.reasoning_echo, read by ReasoningParamsMixin._read_reasoning_echo_from_config

    @property
    def pad(self) -> bool:
        return self.family or self.opt_in


def query_wire_facts(route: Any, reads_images: Optional[bool]) -> tuple[WireFacts, EchoInputs]:
    """The query's wire facts, computed once per call; every reader of the pad (the projection
    through ``_row_before_fill``, the Estimator, the header) takes it from here (PLAN-83g §3.8).

    The host's own agent pads ``reasoning_content`` by ``_needs_thinking_reasoning_pad()``
    (agent/reasoning_params.py 157-173 at Hermes 375930d089): the DeepSeek, Kimi and MiMo family
    test, or its ``_reasoning_echo_flag``, which it reads at creation and ``switch_model`` with the
    staticmethod ``_read_reasoning_echo_from_config`` (180-187; agent_init.py 2452-2453,
    agent_runtime_helpers.py 2121-2122). The method needs the agent, which the query does not read
    and does not stand in for, so its two inputs are read with the host's own functions: the family
    test ``needs_reasoning_echo`` (message_sanitization.py 621-623, the same test line by line for
    this route) and the opt-in reader itself. Each is imported strictly. The Anthropic-converter
    fact is read as ``summariser_input.wire_facts`` reads it, strictly."""
    provider, model, base_url = str(route.target_provider), str(route.target_model), str(route.target_base_url or "")
    family = bool(_strict_import("reasoning-echo family test", "agent.message_sanitization", "needs_reasoning_echo",
                                 so="whether the host's agent pads reasoning_content on this route is not known")(
        provider, model, base_url))
    mixin = _strict_import("reasoning parameters", "agent.reasoning_params", "ReasoningParamsMixin",
                           so="whether the host's agent pads reasoning_content on this route is not known")
    reader = getattr(mixin, "_read_reasoning_echo_from_config", None)
    if not callable(reader):
        raise HostUnavailable("the host's reasoning-echo opt-in reader (agent.reasoning_params.ReasoningParamsMixin."
                              "_read_reasoning_echo_from_config) cannot be read, so whether the host's agent pads "
                              "reasoning_content on this route is not known")
    opt_in = bool(reader())
    mode = str(_strict_import("API mode canonicalisation", "hermes_cli.config_providers", "_canonical_api_mode",
                              so="the route's API mode as the host reads it is not known")(
        str(route.target_api_mode or ""))).lower()
    dispatched = str(_strict_import("provider normalisation", "agent.auxiliary_client", "_normalize_aux_provider",
                                    so="the route's provider as the host dispatches it is not known")(provider))
    anthropic = mode == "anthropic_messages" or (dispatched == "anthropic" and mode in ("", "anthropic_messages"))
    echo = EchoInputs(provider=provider, model=model, base_url=base_url, family=family, opt_in=opt_in)
    return WireFacts(reads_images=reads_images, needs_reasoning_echo=echo.pad, anthropic_converter=anthropic), echo


def json_kind(value: Any) -> str:
    """A JSON value's kind, in JSON's own names."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def carries_nothing(value: Any) -> bool:
    """Null, a string with nothing but whitespace, an empty array or an empty object carry
    nothing (ruling OD-E5); a number or a boolean always states something (``0``, ``false``)."""
    return (value is None or (isinstance(value, str) and not value.strip())
            or (isinstance(value, (list, dict)) and not value))


def blank(value: Any) -> bool:
    """A string that carries nothing (the stored shape a label names)."""
    return isinstance(value, str) and not value.strip()


def _path_text(path: tuple) -> str:
    """A stored path as the query's labels write it, the one renderer of a stored place (M5):
    ``reasoning_details[0].summary``."""
    steps = list(path[1:] if path and path[0] == "message" else path)
    text = ""
    for step in steps:
        text += f"[{step}]" if isinstance(step, int) else (f".{step}" if text else str(step))
    return text or "message"


def stored_role_text(raw: dict) -> str:
    """The stored role as every label of the query says it, one way (M5, PLAN-83g §3.5):
    ``with role "system"``, ``with role null``, or ``with no role`` where the key is absent."""
    if "role" not in raw:
        return "with no role"
    return f"with role {json.dumps(raw['role'], ensure_ascii=False)}"


@dataclass
class Given:
    """What the query gives of one record beside the message: every string with its path and
    standing (``values``); the record's problems that refuse the query (``problems``); the origin
    of every part of the message's content, recorded where the part is made, with the stored path
    it came from and, for a placeholder, the origin of what it replaced (``origins``); the images
    of what the record holds that are not given as images, per class (M3, PLAN-83g §3.3); counts
    for the header."""

    values: list = field(default_factory=list)
    problems: list = field(default_factory=list)      # images the legs disagree on (M3', PLAN-19 §2.2)
    origins: dict = field(default_factory=dict)       # id(part) -> (part, origin, path, was)
    images_replaced: int = 0                          # replaced by a placeholder that says so
    images_elsewhere: list = field(default_factory=list)     # paths: a field the query does not give as an image
    images_ungivable: list = field(default_factory=list)     # paths: the host's converter delivers none as an image
    images_behind_sidecar: int = 0                    # in a stored content the host sends api_content in place of
    reasoning_parts: int = 0
    lifted: int = 0
    joined: bool = False      # the content was joined into one string by the query (``query._join``)
    record: str = ""
    reads_images: Optional[bool] = None
    # The legs the host can send the call on (``query._Leg``: name, wire, prepass), the role the record is sent
    # under, and the host's stop, read before each image's per-part functions run (PLAN-19 §2.2, §2.7).
    legs: tuple = ()
    role: str = ""
    check: Any = None
    call_positions: list = field(default_factory=list)   # the stored positions of the calls given as calls

    def made(self, part: dict, origin: str, path: tuple = (), was: Optional[str] = None) -> dict:
        self.origins[id(part)] = (part, origin, path, was)
        return part

    def _entry(self, part: Any) -> Optional[tuple]:
        entry = self.origins.get(id(part))
        return entry if entry is not None and entry[0] is part else None

    def origin(self, part: Any) -> Optional[str]:
        entry = self._entry(part)
        return entry[1] if entry is not None else None

    def replaced(self, part: Any) -> Optional[str]:
        """For a placeholder, the origin of the image it replaced."""
        entry = self._entry(part)
        return entry[3] if entry is not None else None

    @property
    def images_not_sent(self) -> int:
        """The four classes of M3, each image counted in exactly one (``images_not_sent_is``)."""
        return (self.images_replaced + len(self.images_elsewhere) + len(self.images_ungivable)
                + self.images_behind_sidecar)


def unrecorded_parts(message: dict, given: Given) -> int:
    """How many parts of the message's content have no recorded origin (the query refuses a
    record with any: PLAN-83d §2's property, asserted over every message, PLAN-83e §8.3)."""
    content = message.get("content")
    return sum(1 for part in content if given.origin(part) is None) if isinstance(content, list) else 0


class _Marker(str):
    """A text the query puts inside a value given as JSON in place of an image's payload or of a
    withheld value; not a value of the record."""


# --- M3: one image function (PLAN-83g §3.3) ---------------------------------------------------

def image_part(part: Any) -> bool:
    """An image part by structure (``message_content.is_image_part``, the union of the host's
    tests on message parts: agent/context_compressor.py 1569-1579, model_metadata.py 2450), for
    any stored value: a ``type`` that is not a string is no image type and is never hashed."""
    return isinstance(part, dict) and isinstance(part.get("type"), str) and is_image_part(part)


# The keys of an image part that make its image, per type; every other key of the part is given as
# its JSON beside the image (``cache_control`` is the host's cache marker, bookkeeping).
_IMAGE_KEYS = {"image_url": ("type", "image_url"), "input_image": ("type", "image_url", "detail"),
               "image": ("type", "source")}


def _as_sent(block: dict) -> dict:
    """The image part the query sends for a stored image block: the block as stored, with only the keys that make
    its image (``_IMAGE_KEYS``; an ``input_image``'s ``detail`` only as a string); the plugin converts nothing
    (PLAN-19 §2.2: the host's converters decide what reaches the model)."""
    kind = block["type"]
    image = {key: block[key] for key in _IMAGE_KEYS[kind] if key in block and key != "detail"}
    if kind == "input_image" and isinstance(block.get("detail"), str):
        image["detail"] = block["detail"]
    return image


_CHAT_UNKNOWN = "the host's Chat Completions converter sends it as stored; whether this endpoint reads it is not known"
_DROPPED = "dropped"


def _text_of(blocks: list) -> str:
    """The converter's own text for an image it does not deliver, where it wrote one, else "dropped"."""
    for block in blocks:
        if isinstance(block, dict) and isinstance(block.get("text"), str) and block["text"].strip():
            return block["text"]
    return _DROPPED


def _delivered_on(part: dict, role: str, wire: str, prepass: bool) -> tuple[bool, str]:
    """Whether the host's own converter for ``wire`` delivers one image part as an image, and its own text where it
    does not (M3', PLAN-19 §2.2, rulings D-1, X1-X3), by the host's per-part functions at Hermes 375930d089, each
    called on a copy of exactly the part the query sends:

    - where ``prepass``, first ``_convert_openai_images_to_anthropic`` (agent/auxiliary_client.py 6408-6436), which
      raises on an ``image_url`` that is a string (6424);
    - Anthropic Messages: ``_convert_content_part_to_anthropic`` on a user or agent message,
      ``_content_parts_to_anthropic_blocks([part])`` on a tool result (anthropic_message_convert.py 190-209,
      264-276);
    - Responses: ``_chat_content_to_responses_parts([part], role=…)`` (codex_responses_adapter.py 249-266; a tool
      result's parts are converted with the role user, 534);
    - Chat Completions: the transport passes a part as stored (transports/chat_completions.py 375-433); with the
      host's image conversion off only ``{"type": "image_url", "image_url": {"url": <a non-empty string>}}`` is
      delivered as an image (ruling D-1), with it on only the block that conversion makes, an Anthropic ``image``
      block, whose endpoint's own shape it is.

    An image block out is delivered; anything else is not, with the converter's own text or "dropped"."""
    so = "what the host's converter does to an image part is not known"
    probe = copy.deepcopy(part)
    if prepass:
        convert = _strict_import("image-block conversion", "agent.auxiliary_client",
                                 "_convert_openai_images_to_anthropic", so=so)
        try:
            probe = convert([{"role": role, "content": [probe]}])[0]["content"][0]
        except AttributeError as exc:
            return False, (f"the host's image conversion for this endpoint (_convert_openai_images_to_anthropic) "
                           f"raises on it (AttributeError: {exc})")
    if wire == "chat_completions":
        if prepass:
            return (probe.get("type") == "image", "" if probe.get("type") == "image" else _CHAT_UNKNOWN)
        value = probe.get("image_url")
        url = value.get("url") if isinstance(value, dict) else None
        ok = probe.get("type") == "image_url" and isinstance(url, str) and bool(url)
        return ok, "" if ok else _CHAT_UNKNOWN
    if wire == "anthropic_messages":
        if role == "tool":
            blocks = _strict_import("tool-result block conversion", "agent.anthropic_message_convert",
                                    "_content_parts_to_anthropic_blocks", so=so)([probe])
        else:
            block = _strict_import("content-part conversion", "agent.anthropic_message_convert",
                                   "_convert_content_part_to_anthropic", so=so)(probe)
            blocks = [block] if block is not None else []
        ok = any(isinstance(block, dict) and block.get("type") == "image" for block in blocks)
        return ok, "" if ok else _text_of(blocks)
    if wire == "codex_responses":
        parts = _strict_import("Responses part conversion", "agent.codex_responses_adapter",
                               "_chat_content_to_responses_parts", so=so)([probe], role="user" if role == "tool" else role)
        ok = any(isinstance(item, dict) and item.get("type") == "input_image" for item in parts)
        return ok, "" if ok else _text_of(parts)
    return False, f"the {wire} wire, which the plugin has not established"


def _deliveries(part: dict, given: Given) -> list[tuple[str, bool, str]]:
    """``(leg, delivered, the converter's text)`` for one image part on every leg of the call, under every value of
    the leg's image conversion."""
    if not given.legs:
        raise AssertionError("an image met at an image site with no leg to decide it (PLAN-19 §2.2)")
    out = []
    for leg in given.legs:
        for prepass in leg.prepass:
            if callable(given.check):
                given.check()   # the host's stop, read before each leg's per-part functions (PLAN-19 §2.7)
            # Named where it runs, and where it is one of two values checked.
            value = (f", the host's image conversion {'on' if prepass else 'off'}" if prepass or len(leg.prepass) > 1
                     else "")
            ok, text = _delivered_on(part, given.role or "user", leg.wire, prepass)
            out.append((f"{leg.wire} ({leg.name}{value})", ok, text))
    return out


def _other_image_keys(block: dict) -> dict:
    """What an image part holds beside what makes its image (and the host's cache marker)."""
    named = _IMAGE_KEYS[block["type"]]
    other = {key: value for key, value in block.items() if key not in named and key != "cache_control"}
    if block["type"] == "input_image" and "detail" in block and not isinstance(block["detail"], str):
        other["detail"] = block["detail"]
    return other


GIVEN_IMAGE, REPLACED_IMAGE, NOT_HERE, NOT_GIVABLE = "given", "replaced", "not here", "not givable"


def _image_outcome(block: dict, path: tuple, image_site: bool, given: Given) -> tuple[str, Optional[dict], str]:
    """The one decision on an image the query meets (M3', PLAN-19 §2.2): ``(outcome, part, text)``.

    At a place where the query does not give an image as an image (inside a value rendered from
    a reasoning field, a call, a carrier's other keys, a ``codex_message_items`` part, an unknown
    key: ELSEWHERE) it is named where it stood; where the model does not read images, or that is
    not known, it is replaced by the placeholder that says so; else, at an image site (a content
    member, an envelope member, an image lifted from the stored content, a carrier's or the stash's
    image block), the host's own per-part function of every leg decides (``_deliveries``): where
    every leg delivers it, it is given as stored (``_as_sent``); where none does, it is named with
    each leg's own text; where the legs disagree, the record is a problem that refuses the query
    (whether the model sees it would depend on which leg answers). Each image is counted in exactly
    one class of ``Given``; ``text`` is what a marker or a note says of it."""
    media = image_media_type(block)
    if not image_site:
        given.images_elsewhere.append(path)
        return NOT_HERE, None, (f"[an image part ({media}) stored here, in a field the query does not give as an "
                                f"image; not given as one]")
    if not given.reads_images:
        given.images_replaced += 1
        placeholder = _image_placeholder(block, given.record, _NOT_KNOWN if given.reads_images is None else _NOT_READ)
        return REPLACED_IMAGE, placeholder, placeholder["text"]
    image = _as_sent(block)
    deliveries = _deliveries(image, given)
    if all(ok for _leg, ok, _text in deliveries):
        given.lifted += 1
        return GIVEN_IMAGE, image, f"[image {given.lifted} of this record, given as its own part after this one]"
    missing = [(leg, text) for leg, ok, text in deliveries if not ok]
    if len(missing) < len(deliveries):
        delivering = ", ".join(leg for leg, ok, _text in deliveries if ok)
        given.problems.append(f"{_path_text(path)} holds an image ({media}) that {delivering} deliver(s) as an image "
                              f"and {'; '.join(f'{leg} does not ({text})' for leg, text in missing)}")
    given.images_ungivable.append(path)
    return NOT_GIVABLE, None, (f"[{_path_text(path)} holds an image ({media}) that the host's converter does not "
                               f"deliver as an image: {'; '.join(f'{leg}: {text}' for leg, text in missing)}; not "
                               f"given]")


def _image_parts(block: dict, path: tuple, given: Given, origin: str, kind: str, standing: str,
                 label: Optional[str] = None) -> tuple[list[dict], bool]:
    """An image block at an image site given as parts, in its place: the image (after ``label``,
    written only where the image is given), a placeholder, or a note that says why it is not given;
    in every outcome its other stored keys as their JSON beside it (the orchestrator's ruling on
    PI-9, 2026-09-28). Returns the parts and whether anything under a key of the opaque vocabulary
    was withheld from the other keys."""
    outcome, part, text = _image_outcome(block, path, True, given)
    if outcome == REPLACED_IMAGE:
        parts = [given.made(part, REPLACED, path, origin)]
    elif outcome == NOT_GIVABLE:
        parts = [given.made({"type": "text", "text": text}, NOTE, path)]
    else:
        parts = [given.made({"type": "text", "text": label}, FIELD, path)] if label else []
        parts.append(given.made(part, origin, path))
    other, withheld = [], False
    rest = _other_image_keys(block)
    if rest:
        other, withheld = _render(f"[{_path_text(path)}: the other stored keys of this image part; shown as their JSON "
                                  f"by the query:]", rest, path, kind, standing, given, RENDERED if origin == STORED
                                  else FIELD)
    return parts + other, withheld


def _images_in(value: Any, path: tuple) -> int:
    """How many image parts a stored value holds, at any depth, each once: ``_render``'s own walk
    with lifting off, run into a Given of its own so that nothing of it is given (M3, the images of
    a stored content the host sends api_content in place of; PLAN-83g §3.3)."""
    scratch = Given()
    _step(value, path, CONTENT, GIVEN_CONTENT, _Walk(scratch, lift=False))
    return len(scratch.images_elsewhere)


# --- M1: one walk carrying the kind of the stored field (PLAN-83g §3.1) --------------------------
#
# The kinds of a stored value: a message's content (members, an envelope, api_content, a tool
# result's content); a tool-call dict (canonical or not, wherever stored); a call's name and
# arguments and everything inside them; reasoning (the reasoning fields, the reasoning replay
# carriers, a thinking block, everything under them); a text replay carrier or the stash
# (everything under it unless a transition names another kind); an unknown top-level key. Three
# more names mark the places where a transition of table T is taken inside one walk: a call dict,
# a call's ``function``, a tool_use block.
CONTENT, CALL, CALL_VALUE, REASONING, CARRIER, STORED_KIND = (
    "content", "call", "call value", "reasoning", "carrier", "stored")
_CALL_DICT, _FUNCTION, _TOOL_USE = "call dict", "function", "tool use"
# The sidecar's kind: the ``api_content`` string the host sends in place of the content (``sidecar_sent``),
# walked as content (its members, an envelope) but standing as the sidecar (M-STANDING).
SIDECAR = "sidecar"

# The host's opaque vocabulary (PLAN-83g §3.1; Hermes 375930d089, reader RH1 Q2): the keys under
# which the host keeps replay material no other model can read, at its own places —
# ``signature`` (anthropic_message_convert.py 297-298, 559, 590; bedrock_adapter.py 726-727,
# 846-848; reasoning_summaries.py 36, 40), ``data`` (anthropic_message_convert.py 302, 559;
# bedrock_adapter.py 755, 851), ``encrypted_content`` (codex_responses_adapter.py 787-803,
# 1059-1067; model_metadata.py 2469-2478, 2525-2528), ``redactedContent`` (bedrock_adapter.py
# 578-586, 849), ``redactedContentBase64`` (bedrock_adapter.py 729-733, 852); and on a tool-call
# dict ``extra_content``, withheld whole, which holds the thought signatures
# (chat_completion_helpers.py 1643-1662; transports/chat_completions.py 284-288;
# gemini_native_adapter.py 305-308). The host never treats these names as opaque inside a tool
# call's arguments or a tool's result (chat_completion_helpers.py 1641-1645: arguments kept
# verbatim), so the transcript's own content and a call's name and arguments never withhold them
# (ruling OD-F1).
OPAQUE_VOCABULARY = frozenset({"signature", "data", "encrypted_content", "redactedContent", "redactedContentBase64"})
_CALL_VOCABULARY = OPAQUE_VOCABULARY | {"extra_content"}
_NEVER_WITHHELD = frozenset({CONTENT, CALL_VALUE})
_CALL_KINDS = frozenset({CALL, CALL_VALUE, _CALL_DICT, _FUNCTION, _TOOL_USE})
_WITHHELD = ("[withheld: opaque replay material under this key; counted in the header's encrypted_withheld]")

# Table T (PLAN-83g §3.1), the transitions taken inside one walk: a member of the stored content
# typed as one of the host's blocks (T11, ruling OD-G1) is walked by that block's row; a call
# dict's ``function`` name and arguments are the call's values (T1); ``extra_content`` is
# withheld whole (T2); a call dict's other keys are the call's (T2′). The transitions at the first
# level of a carrier, an entry or a block (T3-T10) are taken by the walkers, which hand a value to
# ``_render`` with the kind their row names; every depth not named inherits its parent's kind.
_CONTENT_BLOCK_KINDS = {"thinking": REASONING, "redacted_thinking": REASONING, "tool_use": _TOOL_USE}


def _withholds(key: Any, kind: str) -> bool:
    """M1's one rule: a value under a key of the host's opaque vocabulary is withheld in every kind
    but the transcript's own content and a call's name and arguments (ruling OD-F1), at every depth,
    the first level of an entry, a block, a call or a stored key included."""
    if kind in _NEVER_WITHHELD:
        return False
    return key in (_CALL_VOCABULARY if kind in (_CALL_DICT, CALL) else OPAQUE_VOCABULARY)


def _standing(kind: str, base: str) -> str:
    """M2, re-derived (M-STANDING): the one place a value's standing is decided, from the kind of the
    field it stands in, so that an origin no writer settles is never told as content or result. A
    value of a record whose base is ``stored`` (a field stored on a role whose domain lacks it, M4)
    is stored whatever its kind; reasoning → reasoning; a call or its values → call; a carrier's
    value → carrier; an unknown key or a non-string sidecar → stored; the sidecar → sidecar;
    content → the record's own (content, or result on a tool row; ruling OD-F2)."""
    if base == GIVEN_STORED:
        return GIVEN_STORED
    if kind == REASONING:
        return GIVEN_REASONING
    if kind in _CALL_KINDS:
        return GIVEN_CALL
    if kind == CARRIER:
        return GIVEN_CARRIER
    if kind == STORED_KIND:
        return GIVEN_STORED
    if kind == SIDECAR:
        return GIVEN_SIDECAR
    return base


@dataclass
class _Walk:
    given: Given
    lift: bool
    images: list = field(default_factory=list)       # (path, image part) given after the JSON
    withheld: bool = False
    # The members of a rendered list under which something was withheld (their index at ``depth`` of a path), so
    # that a list is counted per member (ruling D-6, PLAN-19 §2.8).
    depth: int = 0
    withheld_at: set = field(default_factory=set)


def _step(value: Any, path: tuple, kind: str, base: str, walk: _Walk) -> Any:
    """One value of the walk with its kind: the value as given inside the JSON, every image part
    decided by ``_image_outcome``, every value under a key of the opaque vocabulary withheld where
    the kind withholds, every leaf recorded with its kind's standing (M1, M2)."""
    if isinstance(value, _Marker):
        return value
    if isinstance(value, str):
        walk.given.values.append((path, value, _standing(kind, base)))
        return value
    if isinstance(value, bool) or isinstance(value, (int, float)):
        walk.given.values.append((path, json.dumps(value), _standing(kind, base)))
        return value
    if isinstance(value, list):
        return [_step(item, path + (index,), kind, base, walk) for index, item in enumerate(value)]
    if not isinstance(value, dict):
        return value
    if image_part(value):
        site = walk.lift and kind == CONTENT
        outcome, part, text = _image_outcome(value, path, site, walk.given)
        if outcome == GIVEN_IMAGE:
            walk.images.append((path, part))
        payload = "source" if value["type"] == "image" else "image_url"
        faced = {}
        for key, item in value.items():
            faced[key] = _Marker(text) if key == payload else _step(item, path + (key,), kind, base, walk)
        if payload not in value:
            faced[payload] = _Marker(text)
        return faced
    if kind == CONTENT and isinstance(value.get("type"), str) and value["type"] in (*_CONTENT_BLOCK_KINDS,
                                                                                   "tool_result"):
        if value["type"] == "tool_result":
            # T11: a tool_result block's content is a result; its tool_use_id the block's metadata.
            return {key: _step(item, path + (key,), CONTENT, GIVEN_RESULT if key == "content" else base, walk)
                    for key, item in value.items()}
        kind = _CONTENT_BLOCK_KINDS[value["type"]]
    faced = {}
    for key, item in value.items():
        where = path + (key,)
        if _withholds(key, kind) and not carries_nothing(item):
            walk.withheld = True
            walk.withheld_at.add(where[walk.depth] if len(where) > walk.depth else None)
            faced[key] = _Marker(_WITHHELD)
            continue
        if kind == _CALL_DICT:
            child = _FUNCTION if key == "function" and isinstance(item, dict) else CALL
        elif kind == _FUNCTION:
            child = CALL_VALUE if key in ("name", "arguments") else CALL
        elif kind == _TOOL_USE:
            child = CALL_VALUE if key in ("name", "input") else CALL
        else:
            child = kind
        faced[key] = _step(item, where, child, base, walk)
    return faced


def _render(label: str, value: Any, path: tuple, kind: str, base: str, given: Given, origin: str = FIELD, *,
            lift: bool = False) -> tuple[list[dict], int]:
    """The only place a stored value becomes labelled JSON (M1): nothing where the value carries
    nothing; else the label and the value's JSON as the walk gives it, and, where the value stood in
    the stored content (``lift``), each image given as an image after it. Returns the parts and how
    many units held something withheld under a key of the opaque vocabulary (the caller counts
    them under its stored field): for a list, the members that did (ruling D-6); for any other
    value, 1 or 0."""
    if carries_nothing(value):
        return [], 0
    walk = _Walk(given, lift, depth=len(path))
    faced = _step(value, path, kind, base, walk)
    part = {"type": "text", "text": f"{label}\n{json.dumps(faced, ensure_ascii=False)}"}
    parts = [given.made(part, origin, path)]
    for image_path, image in walk.images:
        parts.append(given.made(image, LIFTED, image_path))
    if not walk.withheld:
        return parts, 0
    return parts, (len(walk.withheld_at) if isinstance(value, list) else 1)


def text_part(part: Any) -> bool:
    """A text part as the host writes it: exactly a ``type`` "text" and a string ``text``
    (agent/image_routing.py 558, turn_context.py 149-157 at Hermes 375930d089). A member with
    further keys is given as its JSON, so no key of it is dropped (PLAN-83d §2)."""
    return (isinstance(part, dict) and set(part) == {"type", "text"} and part["type"] == "text"
            and isinstance(part["text"], str))


def _canonical_parts(parts: list, prefix: tuple, standing: str, given: Given, face: "_Record") -> list:
    """A list content's members: a text part as stored; an image part by the image function (M3);
    any other member as a labelled part holding its JSON, walked as content (its image parts lifted
    out; a member typed as one of the host's blocks walked by that block's row, T11); a member that
    carries nothing not at all."""
    shown = []
    for index, part in enumerate(parts):
        where = prefix + (index,)
        if carries_nothing(part) or (text_part(part) and carries_nothing(part["text"])):
            continue
        if text_part(part):
            shown.append(given.made(part, STORED, where))
            given.values.append((where + ("text",), part["text"], standing))
        elif image_part(part):
            parts_, withheld = _image_parts(part, where, given, STORED, CONTENT, standing)
            shown.extend(parts_)
            face.withheld_content += 1 if withheld else 0      # per member (ruling D-6)
        else:
            rendered, withheld = _render(f"[{_path_text(where)} is not a text part (a type and a text only) or an image "
                                         f"part; shown as its JSON by the query:]", part, where, CONTENT, standing,
                                         given, RENDERED, lift=True)
            shown.extend(rendered)
            face.withheld_content += 1 if withheld else 0      # per member (ruling D-6)
    return shown


def _canonical_content(content: Any, standing: str, given: Given, face: "_Record") -> Any:
    """The message's content: a string or null as stored; a list as its members; the host's
    ``_multimodal`` envelope as its parts and one labelled part holding every other key of it (its
    ``_multimodal`` mark included, so that the part says the content was stored as an envelope);
    any other value as one labelled part holding its JSON (ruling OD-E). A content that carries
    nothing, or of which nothing is given, is given as "" (ruling OD-F4); null stays null."""
    base = ("message", "content")
    if content is None:
        return None
    if isinstance(content, str):
        if carries_nothing(content):
            return ""
        given.values.append((base, content, standing))
        return content
    if isinstance(content, list):
        return _canonical_parts(content, base, standing, given, face) or ""
    parts = content_parts(content)
    if parts is not None and isinstance(content, dict):
        shown = _canonical_parts(parts, base + ("content",), standing, given, face)
        rest = {key: value for key, value in content.items() if key != "content"}
        rendered, withheld = _render("[The rest of this stored multimodal envelope (every key but its content), shown "
                                     "as its JSON by the query:]", rest, base, CONTENT, standing, given, RENDERED,
                                     lift=True)
        # The envelope's keys beside its content, rendered as one object, count as one unit; its content's members
        # were counted each above (ruling D-6).
        face.withheld_content += 1 if withheld else 0
        return (shown + rendered) or ""
    rendered, withheld = _render(f"[The stored content is a JSON {json_kind(content)}, shown as its JSON by the "
                                 f"query:]", content, base, CONTENT, standing, given, RENDERED, lift=True)
    face.withheld_content += withheld                          # per member of a list (ruling D-6)
    return rendered or ""


def canonical_call(call: Any) -> bool:
    """A stored tool call of the shape the host writes (``_assistant_tool_call_dict``,
    agent/chat_completion_helpers.py 1622-1663 at Hermes 375930d089): a dict, a dict ``function``
    with a non-blank string ``name`` and a string ``arguments``, a non-blank string ``id``. The
    one test of which stored calls the query gives as calls (its labels ask it too)."""
    if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
        return False
    function = call["function"]
    return (isinstance(function.get("name"), str) and bool(function["name"].strip())
            and isinstance(function.get("arguments"), str)
            and isinstance(call.get("id"), str) and bool(call["id"].strip()))


def parsed_arguments(arguments: str) -> tuple[Any, Optional[str]]:
    """A call's arguments parsed as JSON, and None; or None and why they do not parse (then the
    stored string is given in a labelled part and the call is compared by its name)."""
    try:
        return json.loads(arguments), None
    except ValueError:
        return None, "they are not valid JSON"
    except RecursionError:
        return None, "they are JSON nested deeper than the parser reads"


# --- One key classification (PLAN-83e §4) ---------------------------------------------------
#
# Every key of every entry of the reasoning fields, of every block of the replay carriers and the
# stash, and of every stored tool call, by what the host writes there (Hermes 375930d089, reader H2
# of PLAN-83e; lines per row): readable reasoning; a message text that must stand in what is given
# of the message (ruling OD-G, rule 3); a text of the stash, given where the content lacks it; a
# call that must be one of the stored calls; an image; opaque material, withheld and counted per
# stored field; the carrier's metadata, not given. Any key the table does not name is given as
# labelled JSON at its path, walked with the kind table T names for its container (M1).
READABLE, MESSAGE_TEXT, STASH_TEXT, CALL_KEY, IMAGE, OPAQUE, METADATA, CITED = (
    "readable", "message_text", "stash_text", "call", "image", "opaque", "metadata", "cited")

# No host reader treats a signature or data as making the text beside it unreadable (#24):
# readable are the keys the host merges into ``reasoning`` and its readers read
# (agent/agent_runtime_helpers.py 1376-1378, context_compressor.py 1327-1337, auxiliary_client.py
# 7087-7088 and 8136, reasoning_summaries.py 39-46); opaque is what the host calls the signed or
# base64 envelope and ciphertext (context_compressor.py 1328, 1365; codex ``encrypted_content``;
# bedrock ``redactedContent*``); metadata is the host's backfill and issuer stamps
# (reasoning_summaries.py 36, 71-73; codex_responses_adapter.py 1067-1083).
_READABLE_KEYS = ("summary", "thinking", "content", "text")
_OPAQUE_KEYS = ("signature", "data", "encrypted_content", "redactedContentBase64", "redactedContent")
_ENTRY_METADATA = ("type", "id", "format", "index", "_issuer_kind", "_issuer_model")


def _table(**classes: tuple) -> dict:
    """A class per key, from keyword arguments named by the classes above."""
    return {key: cls for cls, keys in classes.items() for key in keys}


_CLASSES = {
    # reasoning_details entries: provider dicts kept verbatim (chat_completion_helpers.py 1712-1724,
    # reasoning_summaries.py 49-75), Anthropic thinking blocks (transports/anthropic.py 82-87). T9.
    ("reasoning_details", "entry"): _table(readable=_READABLE_KEYS, opaque=_OPAQUE_KEYS, metadata=_ENTRY_METADATA),
    # codex_reasoning_items (codex_responses_adapter.py 1059-1084): summary pieces {type, text}. T9.
    ("codex_reasoning_items", "item"): _table(opaque=("encrypted_content",), metadata=_ENTRY_METADATA,
                                              readable=("text", "content")),
    ("codex_reasoning_items", "piece"): _table(metadata=("type",), readable=("text",)),
    # anthropic_content_blocks, by block type (anthropic_message_convert.py 287-329): text blocks
    # keep citations and a cache marker (T3); thinking blocks are reasoning (T4); tool_use its
    # sanitised id (T5); image its source (T6). Citations survive only here (the stored content is
    # the prose of the text blocks, transports/anthropic.py 80-81, 102): given as their JSON (ruling
    # OD-E3).
    ("anthropic_content_blocks", "text"): _table(metadata=("type", "cache_control"), message_text=("text",),
                                                 cited=("citations",)),
    ("anthropic_content_blocks", "thinking"): _table(metadata=("type", "cache_control"), readable=("thinking",),
                                                     opaque=("signature", "data")),
    ("anthropic_content_blocks", "redacted_thinking"): _table(metadata=("type", "cache_control"), opaque=("data",)),
    ("anthropic_content_blocks", "tool_use"): _table(metadata=("type", "id", "cache_control"),
                                                     call=("name", "input")),
    ("anthropic_content_blocks", "image"): _table(metadata=("type", "cache_control"), image=("source",)),
    # The stash of a tool result's blocks (read at anthropic_message_convert.py 443-447; no writer).
    (STASH, "text"): _table(metadata=("type", "cache_control"), stash_text=("text",), cited=("citations",)),
    (STASH, "thinking"): _table(metadata=("type", "cache_control"), readable=("thinking",),
                                opaque=("signature", "data")),
    (STASH, "redacted_thinking"): _table(metadata=("type", "cache_control"), opaque=("data",)),
    (STASH, "tool_use"): _table(metadata=("type", "id", "cache_control"), call=("name", "input")),
    (STASH, "image"): _table(metadata=("type", "cache_control"), image=("source",)),
    # bedrock_content_blocks, stored flattened (bedrock_adapter.py 824-899, 909-998): a block can
    # hold a text and a reasoning together (977).
    ("bedrock_content_blocks", "block"): _table(message_text=("text",)),
    ("bedrock_content_blocks", "reasoningContent"): _table(readable=("text",), opaque=(
        "signature", "redactedContentBase64", "redactedContent")),
    ("bedrock_content_blocks", "reasoningText"): _table(readable=("text",), opaque=("signature",)),
    ("bedrock_content_blocks", "toolUse"): _table(metadata=("toolUseId",), call=("name", "input")),
    # codex_message_items (codex_responses_adapter.py 367-373, 1124-1139): one output_text part (T7).
    ("codex_message_items", "item"): _table(metadata=("type", "role", "status", "id", "phase")),
    ("codex_message_items", "part"): _table(metadata=("type",), message_text=("text",)),
    # A stored tool call (chat_completion_helpers.py 1622-1663): its ids beside ``id`` and its
    # type are the host's; ``extra_content`` holds the stored model's thought signature
    # (transports/chat_completions.py 274-288), withheld and counted (ruling OD-P2a; T2).
    ("tool_calls", "call"): _table(metadata=("call_id", "response_item_id"), opaque=("extra_content",)),
}


@dataclass
class _Record:
    """One record's facing: what the classification gives and counts, gathered in one place.
    ``foreign``: the field is stored on a role whose domain lacks it (M4). ``compares_calls``: the
    message is an agent message, whose carrier calls that are among its calls given as calls are
    not repeated (T5); every other carrier call is given as labelled JSON (the orchestrator's ruling
    on PI-2, 2026-09-28; PLAN-19 §2.5).
    ``role_text``: the stored role as the labels say it (``stored_role_text``)."""

    record: str
    given: Given
    standing: str
    opaque: dict
    readable: list = field(default_factory=list)      # (path, text) of readable reasoning
    after: list = field(default_factory=list)         # parts after the content
    content_texts: list = field(default_factory=list)
    calls: list = field(default_factory=list)         # (name, arguments, parsed) of the calls given
    foreign: bool = False
    withheld_content: int = 0     # members of the content under which something was withheld (ruling D-6)
    compares_calls: bool = False
    role_text: str = ""
    stash_site: bool = False      # the host's Anthropic converter sends this tool result's stash (``stash_sent``)

    def json(self, label: str, value: Any, path: tuple, kind: str, standing: Optional[str] = None) -> bool:
        parts, withheld = _render(label, value, path, kind, standing or self.standing, self.given)
        self.after.extend(parts)
        return withheld

    def not_text(self, value: Any, path: tuple, kind: str, standing: Optional[str] = None) -> bool:
        return self.json(f"[{_path_text(path)} as stored is not text; shown as its JSON by the query:]", value, path,
                         kind, standing)

    def readable_text(self, value: Any, path: tuple) -> bool:
        if isinstance(value, str):
            self.readable.append((path, value))
            return False
        return self.not_text(value, path, REASONING)

    def carried(self, value: Any, path: tuple, standing: Optional[str] = None) -> bool:
        """A replay carrier's message text stands in what is given of the message (its content, the
        sidecar the host sends in its place, or its main readable reasoning), or it is given as a
        labelled part of its own, on every message and every wire (M-CARRIER, PLAN-19 §2.5): the query
        sends no carrier (ruling OD-G), so the part is what carries a text only the carrier holds. Its
        standing (M-STANDING): ``carrier``, since the carriers keep the provider's text before the
        host's strip, redaction and flattening and record which class a residue is nowhere; a codex
        message item the host stamped as a commentary or analysis phase passes ``reasoning``, the
        writer's own stamp (``_codex_message_items``); on a foreign record ``stored``."""
        if not isinstance(value, str):
            return self.not_text(value, path, CARRIER)
        if any(value in text for text in self.content_texts):
            return False
        standing = _standing(REASONING if standing == GIVEN_REASONING else CARRIER, self.standing)
        self.given.values.append((path, value, standing))
        if standing == GIVEN_REASONING:      # counted in the header's reasoning_given_apart, as every reasoning part
            self.given.reasoning_parts += 1
        where = ", where the host writes no such field," if self.foreign else ""
        holds = ("which its content and its readable reasoning do not hold" if standing == GIVEN_REASONING
                 else "which its content does not hold")
        stamped = (" (a message item the host stamped as a commentary or analysis phase, which it routes to the "
                   "reasoning channel, codex_responses_adapter.py 1125-1134)" if standing == GIVEN_REASONING else "")
        self.after.append(self.given.made({"type": "text", "text": (
            f"[A text stored in {_path_text(path)} on this message{where} {holds}{stamped}:]\n{value}")},
            FIELD, path))
        return False

    def stashed(self, value: Any, path: tuple) -> bool:
        """The host's Anthropic converter sends a tool result's stash after its content, so a stashed
        text the content lacks is given as its own part. Its standing (M-STANDING): ``carrier``, since
        no writer of this host fills the stash (anthropic_message_convert.py 431-450 reads it; nothing
        writes it at 375930d089); on a foreign record ``stored``."""
        if not isinstance(value, str):
            return self.not_text(value, path, CARRIER)
        if not any(value in text for text in self.content_texts):
            self.given.values.append((path, value, _standing(CARRIER, self.standing)))
            what = (f"[A text stored in {_path_text(path)} on this message, where the host writes no such field, which "
                    f"its content does not hold:]" if self.foreign else
                    f"[A text block stored in {_path_text(path)}, which the stored content does not hold:]")
            self.after.append(self.given.made({"type": "text", "text": f"{what}\n{value}"}, FIELD, path))
        return False

    def called(self, name: Any, arguments: Any) -> bool:
        """Whether a replay carrier's call is one of the calls given as calls: by name and parsed input
        (PLAN-83c §5.2; the block's id is the host's sanitised one, not compared). Where the stored call
        of that name has arguments that do not parse, its input cannot be shown equal: it is not."""
        return any(parsed and arguments == stored_arguments
                   for stored_name, stored_arguments, parsed in self.calls if name == stored_name)

    def call_block(self, block: dict, table: dict, path: tuple) -> bool:
        """A carrier's call block (``tool_use``, ``toolUse``; T5). On an agent message one of its calls
        given as calls (``called``) is not repeated; every other carrier call, on any message and
        every wire, is given as labelled JSON of its name and input, the call's values (standing call;
        the vocabulary is never withheld inside ``input``), and nothing is refused (the orchestrator's
        ruling on PI-2, 2026-09-28; M-CARRIER, PLAN-19 §2.5). Its metadata is not given; every other key
        is the call's. Returns whether opaque material was withheld."""
        withheld = False
        if not (self.compares_calls and not self.foreign and self.called(block.get("name"), block.get("input"))):
            values = {key: block[key] for key in ("name", "input") if key in block}
            withheld = self.json(f"[{_path_text(path)} holds a tool call, stored on this message {self.role_text}, "
                                 f"which is not one of the calls the query gives; its name and input shown as their "
                                 f"JSON by the query:]", values, path, _TOOL_USE)
        return self.keys(block, table, path, CALL, skip=("name", "input")) or withheld

    def keys(self, container: dict, table: dict, path: tuple, other: str, *, skip: tuple = (),
             text_standing: Optional[str] = None) -> bool:
        """Every key of an entry or block by its class; ``other``: the kind (table T) of every key
        the table does not name; ``text_standing``: the standing a message text of this container
        passes to ``carried`` where the writer stamped it (a codex commentary item). Returns whether
        opaque material that carries something was withheld, at this level or inside a key given
        as JSON."""
        withheld = False
        for key, value in container.items():
            if key in skip or carries_nothing(value):
                continue
            cls = table.get(key)
            where = path + (key,)
            if cls == METADATA:
                continue
            if cls == OPAQUE:
                withheld = True
            elif cls == READABLE:
                withheld = self.readable_text(value, where) or withheld
            elif cls == MESSAGE_TEXT:
                withheld = self.carried(value, where, text_standing) or withheld
            elif cls == STASH_TEXT:
                withheld = self.stashed(value, where) or withheld
            elif cls == CITED:
                # Walked with the carrier's kind: standing carrier, stored on a foreign record (M-STANDING).
                withheld = self.json(f"[{_path_text(where)}: the citations stored with this text block; shown as their "
                                     f"JSON by the query:]", value, where, CARRIER) or withheld
            elif _withholds(key, other):
                # A key of the opaque vocabulary the table does not name at this container (M1).
                withheld = True
            else:
                withheld = self.json(f"[{_path_text(where)} is a key of no kind the query knows; shown as its JSON by "
                                     f"the query:]", value, where, other) or withheld
        return withheld

    def count(self, kind: str, units: int = 1) -> None:
        self.opaque[kind] = self.opaque.get(kind, 0) + units

    def not_a_list(self, key: str, value: Any, kind: str) -> bool:
        return self.json(f"[{_path_text(('message', key))} as stored is not a list; shown as its JSON by the query:]",
                         value, ("message", key), kind)

    def members(self, key: str, value: Any, kind: str) -> Iterable[tuple[tuple, dict]]:
        """The members of a stored list field that carry something and are objects; any other
        member, and a value that is not a list, given as JSON with the field's kind, counted under
        the field where something inside was withheld."""
        if carries_nothing(value):
            return
        if not isinstance(value, list):
            if self.not_a_list(key, value, kind):
                self.count(key)
            return
        for index, member in enumerate(value):
            path = ("message", key, index)
            if carries_nothing(member):
                continue
            if isinstance(member, dict):
                yield path, member
            elif self.json(f"[{_path_text(path)} as stored is not an entry of the host's shape; shown as its JSON by "
                           f"the query:]", member, path, kind):
                self.count(key)


def _reasoning_details(value: Any, face: _Record) -> None:
    """``reasoning_details`` entries, reasoning at every depth (T9). A ``<provider>.native_assistant``
    entry is another provider's private replay carrier (agent/transports/chat_completions.py:388,
    providers/base.py 100-102): its top-level readable strings are given (the host merged the
    first into ``reasoning``), all else of it withheld and named by its type (ruling OD-P2b; T10)."""
    table = _CLASSES[("reasoning_details", "entry")]
    for path, entry in face.members("reasoning_details", value, REASONING):
        kind = entry.get("type")
        if isinstance(kind, str) and kind.endswith(".native_assistant"):
            withheld = False
            for key, item in entry.items():
                if carries_nothing(item) or table.get(key) == METADATA:
                    continue
                if key in _READABLE_KEYS and isinstance(item, str):
                    face.readable.append((path + (key,), item))
                else:
                    withheld = True
            if withheld:
                face.count(f"reasoning_details ({kind}, another provider's private replay carrier)")
        elif face.keys(entry, table, path, REASONING):
            face.count("reasoning_details")


def _codex_reasoning_items(value: Any, face: _Record) -> None:
    """``codex_reasoning_items``, reasoning at every depth (T9); a string ``summary`` is readable
    text, as the host reads it (agent_runtime_helpers.py 1376-1378)."""
    for path, item in face.members("codex_reasoning_items", value, REASONING):
        withheld = face.keys(item, _CLASSES[("codex_reasoning_items", "item")], path, REASONING, skip=("summary",))
        summary = item.get("summary")
        where_summary = path + ("summary",)
        if isinstance(summary, list):
            for number, piece in enumerate(summary):
                where = where_summary + (number,)
                if carries_nothing(piece):
                    continue
                if isinstance(piece, dict):
                    withheld = face.keys(piece, _CLASSES[("codex_reasoning_items", "piece")], where, REASONING) \
                        or withheld
                else:
                    withheld = face.json(f"[{_path_text(where)} is not a summary piece of the host's shape; shown as "
                                         f"its JSON by the query:]", piece, where, REASONING) or withheld
        elif isinstance(summary, str) and not carries_nothing(summary):
            face.readable.append((where_summary, summary))
        elif not carries_nothing(summary):
            withheld = face.json(f"[{_path_text(where_summary)} is not a list of summary pieces; shown as its JSON by "
                                 f"the query:]", summary, where_summary, REASONING) or withheld
        if withheld:
            face.count("codex_reasoning_items")


def stash_sent(content: Any) -> bool:
    """Whether the host's Anthropic converter sends a tool result's ``_anthropic_content_blocks`` stash for this
    content (M-STASH, PLAN-19 §2.6): not for a ``_multimodal`` envelope, not where the host's own
    ``_content_parts_to_anthropic_blocks`` makes an image block of the content (anthropic_message_convert.py 431-447
    at Hermes 375930d089, called here on a copy of the content as the host's row carries it)."""
    if isinstance(content, dict) and content.get("_multimodal"):
        return False
    if isinstance(content, list):
        blocks = _strict_import("tool-result block conversion", "agent.anthropic_message_convert",
                                "_content_parts_to_anthropic_blocks",
                                so="whether the host sends this tool result's stash is not known")(copy.deepcopy(content))
        if any(isinstance(block, dict) and block.get("type") == "image" for block in blocks):
            return False
    return True


def _anthropic_blocks(key: str, value: Any, face: _Record) -> None:
    """``anthropic_content_blocks`` of an agent message, or the stash of a tool result, block by
    block and key by key (T3-T6); an image block is given as an image (the host's converter
    replays it, anthropic_message_convert.py 309-311, 443-447); a block of a type the host does not
    write is given whole as its JSON, walked as the carrier's (T8)."""
    for path, block in face.members(key, value, CARRIER):
        kind = block.get("type")
        table = _CLASSES.get((key if key == STASH else "anthropic_content_blocks", kind)) if isinstance(kind, str) \
            else None
        standing = face.standing        # the base ``_standing`` decides over: a carrier value stands as carrier, stored on a
        #                                 foreign record (M-STANDING); an image block's other keys likewise
        if image_part(block) and key == STASH and not (face.stash_site and kind == "image"):
            # M-STASH (PLAN-19 §2.6): the host's Anthropic converter sends the stash only for a tool result whose
            # content is no _multimodal envelope and yields no image block of its own, and replays its blocks as they
            # are, so only an ``image`` block is an image there (anthropic_message_convert.py 431-450): any other
            # image block of the stash, and every one where the stash is not sent, is named where it stood.
            if face.json(f"[{_path_text(path)} holds an image block that the host's Anthropic converter does not send "
                         f"as an image from this field here; shown as its JSON by the query:]", block, path, CARRIER,
                         standing):
                face.count(key)
            continue
        if image_part(block) and (key == STASH or kind == "image"):
            # T6: an image site. An image_url/input_image block of anthropic_content_blocks is not
            # (the host's replay whitelist has no such type): it falls to the block of no type below.
            parts, withheld = _image_parts(block, path, face.given, FIELD, CARRIER, standing,
                                           f"[An image block stored in {_path_text(path)}, given as an image:]")
            face.after.extend(parts)
            if withheld:
                face.count(key)
            continue
        if table is None:
            if face.json(f"[{_path_text(path)} is a block of no type the query knows; shown as its JSON by the query:]",
                         block, path, CARRIER, standing):
                face.count(key)
            continue
        if kind in ("thinking", "redacted_thinking"):
            withheld = face.keys(block, table, path, REASONING)
        elif kind == "tool_use":
            # T5: on an agent message the block's name and input compared with the stored canonical
            # calls (rule 3); on any other message given as labelled JSON (ruling on PI-2).
            withheld = face.call_block(block, table, path)
        else:
            withheld = face.keys(block, table, path, CARRIER)
        if withheld:
            face.count(key)


def _bedrock_blocks(value: Any, face: _Record) -> None:
    """``bedrock_content_blocks`` key by key: a block can hold a text and a reasoning together."""
    field_ = "bedrock_content_blocks"
    for path, block in face.members(field_, value, CARRIER):
        withheld = face.keys(block, _CLASSES[(field_, "block")], path, CARRIER, skip=("reasoningContent", "toolUse"))
        reasoning = block.get("reasoningContent")
        where = path + ("reasoningContent",)
        if isinstance(reasoning, dict):
            withheld = face.keys(reasoning, _CLASSES[(field_, "reasoningContent")], where, REASONING,
                                 skip=("reasoningText",)) or withheld
            nested = reasoning.get("reasoningText")
            if isinstance(nested, dict):
                withheld = face.keys(nested, _CLASSES[(field_, "reasoningText")], where + ("reasoningText",),
                                     REASONING) or withheld
            elif not carries_nothing(nested):
                withheld = face.readable_text(nested, where + ("reasoningText",)) or withheld
        elif not carries_nothing(reasoning):
            withheld = face.json(f"[{_path_text(where)} is not an object of the host's shape; shown as its JSON by the "
                                 f"query:]", reasoning, where, REASONING) or withheld
        tool = block.get("toolUse")
        where = path + ("toolUse",)
        if isinstance(tool, dict):
            withheld = face.call_block(tool, _CLASSES[(field_, "toolUse")], where) or withheld
        elif not carries_nothing(tool):
            withheld = face.json(f"[{_path_text(where)} is not an object of the host's shape; shown as its JSON by the "
                                 f"query:]", tool, where, CALL) or withheld
        if withheld:
            face.count(field_)


def _codex_commentary(item: dict) -> bool:
    """The host's own predicate on a codex message item's ``phase`` stamp (codex_responses_adapter.py
    1125-1134 at Hermes 375930d089: ``normalized_phase = _lower_or_none(getattr(item, "phase", None))``,
    ``is_commentary_phase = normalized_phase in {"commentary", "analysis"}``; the stamp written at
    1136-1139 is that normalised value), mirrored line for line with the host's own normaliser: a
    commentary or analysis item's text the host routes to its reasoning channel."""
    lower_or_none = _strict_import("codex phase normaliser", "agent.codex_responses_adapter", "_lower_or_none")
    return lower_or_none(item.get("phase")) in {"commentary", "analysis"}


def _codex_message_items(value: Any, face: _Record) -> None:
    """``codex_message_items``: message items of one ``output_text`` part (T7; the host's replay
    reads only such parts, codex_responses_adapter.py 457, so an image part there was never an image
    to any converter: ruling OD-E1); any other item or part is the carrier's (T8). A part's text of an
    item the host stamped as a commentary or analysis phase has the standing reasoning, the writer's
    own stamp (``_codex_commentary``); any other phase, absent or unknown, carrier (M-STANDING)."""
    field_ = "codex_message_items"
    for path, item in face.members(field_, value, CARRIER):
        parts = item.get("content")
        withheld = False
        if item.get("type") != "message" or not isinstance(parts, list):
            withheld = face.json(f"[{_path_text(path)} is an item of no shape the query knows; shown as its JSON by the "
                                 f"query:]", item, path, CARRIER)
        else:
            withheld = face.keys(item, _CLASSES[(field_, "item")], path, CARRIER, skip=("content",))
            text_standing = GIVEN_REASONING if _codex_commentary(item) else None
            for number, part in enumerate(parts):
                where = path + ("content", number)
                if carries_nothing(part):
                    continue
                if isinstance(part, dict) and part.get("type") in ("output_text", "text"):
                    withheld = face.keys(part, _CLASSES[(field_, "part")], where, CARRIER,
                                         text_standing=text_standing) or withheld
                else:
                    withheld = face.json(f"[{_path_text(where)} is a part of no shape the query knows; shown as its "
                                         f"JSON by the query:]", part, where, CARRIER) or withheld
        if withheld:
            face.count(field_)


def _tool_calls(value: Any, face: _Record, calls_given: Optional[set] = None,
                call_handles: Optional[dict] = None) -> list[tuple[int, dict]]:
    """An agent message's calls as the wire carries them, with their stored positions: each call of
    the host's shape as its id, type and function's name and arguments (T1); every other key of it
    faced by the table (T2, T2′; a stored type other than "function" given as its JSON, never
    overwritten unsaid); any other value or call given as labelled JSON after the content, walked
    as a call, not a call on the wire. ``calls_given``: the positions of the calls of the host's shape
    the query gives as calls (M-PAIR, PLAN-19 §2.4; None: every one); any other is given as labelled
    JSON, walked as a call, its handle named."""
    base = ("message", "tool_calls")
    if carries_nothing(value):
        return []
    if not isinstance(value, list):
        if face.json(f"[The stored tool calls are a JSON {json_kind(value)}, not a list of calls; shown as their JSON by "
                     f"the query:]", value, base, _CALL_DICT if isinstance(value, dict) else CALL):
            face.count("tool_calls")
        return []
    kept = []
    table = _CLASSES[("tool_calls", "call")]
    for index, call in enumerate(value):
        path = base + (index,)
        if carries_nothing(call):
            continue
        if not canonical_call(call):
            if face.json(f"[{_path_text(path)} as stored is not a call the wire can carry; shown as its JSON by the "
                         f"query:]", call, path, _CALL_DICT if isinstance(call, dict) else CALL):
                face.count("tool_calls")
            continue
        if calls_given is not None and index not in calls_given:
            handle = (call_handles or {}).get(index)
            named = f"tool call {handle} ({_path_text(path)})" if handle else _path_text(path)
            if face.json(f"[{named} is a tool call the query does not give as a call: the query names no result for it "
                         f"among the records it gives, or the host's Anthropic converter would strip it; shown as its "
                         f"JSON by the query:]", call, path, _CALL_DICT):
                face.count("tool_calls")
            continue
        function = call["function"]
        kept.append((index, {"id": call["id"], "type": "function",
                             "function": {"name": function["name"], "arguments": function["arguments"]}}))
        face.given.values.append((path + ("function", "name"), function["name"], _standing(CALL_VALUE, face.standing)))
        face.given.values.append((path + ("function", "arguments"), function["arguments"],
                                  _standing(CALL_VALUE, face.standing)))
        withheld = False
        kind = call.get("type")
        if not carries_nothing(kind) and kind != "function":
            withheld = face.json(f"[{_path_text(path + ('type',))} as stored is not \"function\"; the call is given as a "
                                 f"function call, its stored type shown as its JSON by the query:]", kind,
                                 path + ("type",), CALL)
        withheld = face.keys(call, table, path, CALL, skip=("id", "function", "type")) or withheld
        withheld = face.keys(function, {}, path + ("function",), CALL, skip=("name", "arguments")) or withheld
        if withheld:
            face.count("tool_calls")
    return kept


def _host_metadata_keys() -> frozenset:
    """The keys the host writes on a message as its own bookkeeping, from the host's own constants
    at run time (ruling OD-3a): the session schema's columns (``hermes_state_messages
    ._MESSAGE_SCHEMA_KEYS``), the message core keys (``agent.message_sanitization._MESSAGE_CORE_KEYS``),
    the keys the Chat Completions transport strips (``_STRIP_MSG_KEYS``) and the persistence-only
    fields. Two of these constants hold transcript keys too (the core keys: content, name,
    tool_calls, role; the schema's columns); the union is right at its one call site only because
    ``strict_message`` faces the transcript keys first (PLAN-83g §2; a fact for #72)."""
    so = "which of a message's keys are the host's own bookkeeping is not known"
    schema = _strict_import("session schema keys", "hermes_state_messages", "_MESSAGE_SCHEMA_KEYS", so=so)
    core = _strict_import("message core keys", "agent.message_sanitization", "_MESSAGE_CORE_KEYS", so=so)
    strip = _strict_import("Chat Completions strip keys", "agent.transports.chat_completions", "_STRIP_MSG_KEYS",
                           so=so)
    persistence = _strict_import("persistence fields", "agent.message_metadata", "PERSISTENCE_ONLY_MESSAGE_FIELDS",
                                 so=so)
    return frozenset(schema) | frozenset(core) | frozenset(strip) | frozenset(persistence)


def _is_metadata(key: Any, known: frozenset) -> bool:
    """A key of the host's own bookkeeping: an underscore key (the host strips every one from every
    wire), a key of the host's constants, the stamps of ``_BOOKKEEPING_KEYS``."""
    key = str(key)
    return key.startswith("_") or key in known or key in _BOOKKEEPING_KEYS


def _assemble(message: dict, before: list[dict], after: list[dict], given: Given) -> None:
    """The message's content: ``before``, the stored content's parts, ``after``. A stored string
    becomes one stored text part, one that carries nothing no part. With nothing to add the
    content stays as it is (a string stays a string)."""
    if not before and not after:
        return
    content = message.get("content")
    if isinstance(content, list):
        stored = list(content)
    elif isinstance(content, str) and not carries_nothing(content):
        stored = [given.made({"type": "text", "text": content}, STORED, ("message", "content"))]
    else:
        stored = []
    message["content"] = before + stored + after


def label_message(message: dict, label: str, given: Given) -> None:
    """The query's label as the message's first part, recorded as the query's own."""
    _assemble(message, [given.made({"type": "text", "text": label}, LABEL)], [], given)


def wire_role(raw: dict) -> str:
    """The role the query sends a record under: its stored role where the host's converters carry
    it (user, assistant, tool), else user (ruling OD-E2: a stored system message would overwrite the
    query's instructions on the Anthropic and Codex converters, anthropic_message_convert.py
    722-723, auxiliary_client.py 1462-1476; its label names the stored role)."""
    role = raw.get("role")
    return role if isinstance(role, str) and role in _WIRE_ROLES else "user"


def _foreign(key: str, value: Any, raw: dict, face: _Record, shown: list) -> None:
    """A transcript field stored on a role whose domain lacks it, walked by its own walker (M4,
    ruling OD-F5): the row decides only that its outputs are given as parts, never placed on the
    wire; the first part says where it is stored."""
    # Every value of a foreign field stands as ``stored`` (M-STANDING): no producer of this host writes
    # the field on this role, so no writer settles what it holds; ``_standing`` returns the base for
    # every kind, and ``carried``, ``stashed`` and ``_readable_parts`` take it from the record.
    sub = _Record(face.record, face.given, GIVEN_STORED, face.opaque, content_texts=face.content_texts,
                  foreign=True, role_text=face.role_text)
    path = ("message", key)
    if key in ("reasoning", "reasoning_content"):
        if sub.readable_text(value, path):
            sub.count(key)
    elif key == "reasoning_details":
        _reasoning_details(value, sub)
    elif key == "codex_reasoning_items":
        _codex_reasoning_items(value, sub)
    elif key in ("anthropic_content_blocks", STASH):
        _anthropic_blocks(key, value, sub)
    elif key == "bedrock_content_blocks":
        _bedrock_blocks(value, sub)
    elif key == "codex_message_items":
        _codex_message_items(value, sub)
    elif key == "tool_calls":
        # Every call as labelled JSON, walked as a call (T1-T2′), never a call on the wire and never
        # labelled with a handle.
        if isinstance(value, list):
            for index, call in enumerate(value):
                if not carries_nothing(call) and sub.json(
                        f"[{_path_text(path + (index,))}, a tool call; shown as its JSON by the query:]", call,
                        path + (index,), _CALL_DICT if isinstance(call, dict) else CALL):
                    sub.count("tool_calls")
        elif sub.json(f"[{_path_text(path)}, the stored tool calls; shown as their JSON by the query:]", value, path,
                      _CALL_DICT if isinstance(value, dict) else CALL):
            sub.count("tool_calls")
    elif key == "api_content":
        withheld_units = sub.not_text(value, path, STORED_KIND)
        if withheld_units:
            sub.count("api_content", withheld_units)      # per member of a list (ruling D-6)
    elif sub.json(f"[{_path_text(path)} as stored; shown as its JSON by the query:]", value, path, STORED_KIND):
        # tool_call_id, name: the wire's pairing key and the host's bookkeeping, stored where the host
        # writes neither; given as they are stored, what the vocabulary names inside withheld and counted.
        sub.count(key)
    readable = _readable_parts(sub.readable, face.given, shown, GIVEN_STORED)
    if not (readable or sub.after):
        return
    label = face.given.made({"type": "text", "text": (f"[{_path_text(path)} is stored on this message "
                                                      f"{stored_role_text(raw)}, where the host writes no such field; "
                                                      f"what the query gives of it follows:]")}, NOTE, path)
    face.after.extend([label] + readable + sub.after)


def _readable_parts(readable: list, given: Given, shown: list, base: str = GIVEN_CONTENT) -> list[dict]:
    """Readable reasoning texts as labelled parts, each unless it carries nothing or is contained
    verbatim in the message's reasoning or an earlier such part (exact containment, ruling OD-4a).
    Their standing is ``_standing``'s for the reasoning kind over the record's ``base``: reasoning on
    the record's own reasoning fields and carriers (their writers settle it); stored on a foreign
    record, where the host writes no such field (M-STANDING)."""
    parts = []
    for path, text in readable:
        if carries_nothing(text) or any(text in earlier for earlier in shown):
            continue
        parts.append(given.made({"type": "text", "text": f"[Readable reasoning stored in {_path_text(path)}:]\n{text}"},
                                FIELD, path))
        standing = _standing(REASONING, base)
        given.values.append((path, text, standing))
        if standing == GIVEN_REASONING:      # the header's reasoning_given_apart counts reasoning, never stored
            given.reasoning_parts += 1
        shown.append(text)
    return parts


def strict_message(raw: dict, record: str, facts: WireFacts, withheld: dict, given: Given,
                   call_handles: Optional[dict] = None, legs: tuple = (), check: Any = None,
                   calls_given: Optional[set] = None, as_user: bool = False) -> dict:
    """One record's message as the query gives it (PLAN-83e §2-§5, PLAN-83g §3). The host's per-row
    rules run first, each host function called strictly (the clone, the sidecar, the reasoning-echo
    policy with the pad the host's own agent applies, the fill decided on the host's own row); then
    the message is built from the stored role's domain and every stored key is faced. ``given``
    receives what is given of the record and the origin of every part of its content; ``withheld``
    counts, per stored field, the entries, blocks, calls and keys whose opaque material is withheld;
    ``call_handles`` names, by stored position, the handles the store minted for the record's calls;
    ``calls_given``: the positions of its calls the query gives as calls, every other shown as labelled
    JSON; ``as_user``: a stored tool result sent as a user message, without its call id (M-PAIR,
    PLAN-19 §2.4); ``legs`` and ``check``: what decides its images, and the host's stop (§2.2, §2.7)."""
    given.record, given.reads_images = record, facts.reads_images
    # A stored tool result whose call the query does not give as a call is sent as a user message (M-PAIR, D-2).
    role = "user" if as_user else wire_role(raw)
    given.legs, given.role, given.check = tuple(legs), role, check
    stored_role = raw.get("role")
    domain = _DOMAIN.get(stored_role, _OTHER_DOMAIN) if isinstance(stored_role, str) else _OTHER_DOMAIN
    row = _row_before_fill(raw, needs_echo=facts.needs_reasoning_echo, strict=True)
    fill = host_fill_text(row)
    # The standing is the stored role's, never the role the query sends the record under (PLAN-19 §2.8).
    standing = GIVEN_RESULT if stored_role == "tool" else GIVEN_CONTENT
    message: dict = {"role": role}
    face = _Record(record, given, standing, {}, compares_calls="tool_calls" in domain, role_text=stored_role_text(raw))
    if "content" in row:
        # Where the host sends the sidecar in place of the content (``_row_before_fill`` put it under
        # ``content``, as ``lcm_expand`` shows it), what stands there is the sidecar, whose origin no
        # writer of the host records: standing ``sidecar`` (M-STANDING), at the path expansion shows.
        message["content"] = _canonical_content(row["content"], _standing(SIDECAR if sidecar_sent(raw) else CONTENT,
                                                                          standing), given, face)
    if face.withheld_content:
        face.count("content", face.withheld_content)
    if sidecar_sent(raw) and not carries_nothing(raw.get("content")):
        # M3: the images of a stored content the host sends api_content in place of (the host
        # replaces the content wholesale on every request, agent/turn_context.py 1231-1256), held
        # by the handles and not given; its label says the stored content is not given.
        given.images_behind_sidecar += _images_in(raw["content"], ("message", "content"))
    # The keys in the order the host's own row has them (role, content, reasoning_content,
    # tool_calls; a tool result's call id after its content), so that the request is the host's.
    if "reasoning_content" in domain and isinstance(row.get("reasoning_content"), str):
        message["reasoning_content"] = row["reasoning_content"]   # as the host's echo policy left it
    positions: list = []
    if "tool_calls" in domain:
        kept = _tool_calls(raw.get("tool_calls"), face, calls_given, call_handles)
        if kept:
            positions = [position for position, _call in kept]
            message["tool_calls"] = [call for _position, call in kept]
            face.calls = [(call["function"]["name"], *_call_arguments(call)) for _position, call in kept]
    given.call_positions = list(positions)
    if "tool_call_id" in domain and "tool_call_id" in raw and not as_user:
        message["tool_call_id"] = raw["tool_call_id"]      # the wire's pairing key, as stored
    main = readable_reasoning(raw) if "reasoning" in domain else None
    # What the model reads as the message's content: the content, a tool result, or the sidecar the
    # host sends in the content's place; a carrier text contained in it is not given again.
    face.content_texts = [text for _path, text, standing_ in given.values
                          if standing_ in (GIVEN_CONTENT, GIVEN_RESULT, GIVEN_SIDECAR)] + ([main] if main else [])
    if "reasoning" in domain:
        for key in ("reasoning", "reasoning_content"):
            value = raw.get(key)
            if carries_nothing(value) or value == main:
                continue
            if face.readable_text(value, ("message", key)):
                face.count(key)
        _reasoning_details(raw.get("reasoning_details"), face)
        _codex_reasoning_items(raw.get("codex_reasoning_items"), face)
        _anthropic_blocks("anthropic_content_blocks", raw.get("anthropic_content_blocks"), face)
        _bedrock_blocks(raw.get("bedrock_content_blocks"), face)
        _codex_message_items(raw.get("codex_message_items"), face)
    if STASH in domain:
        if callable(check):
            check()     # the host's stop, read before the host's conversion of the content (PLAN-19 §2.7)
        face.stash_site = stash_sent(row.get("content"))
        _anthropic_blocks(STASH, raw.get(STASH), face)
    if "api_content" in domain and not sidecar_sent(raw):
        # T13: a sidecar that is not a string is given as its JSON; no writer of this host puts one
        # there (every sidecar writer writes a string), so it stands as ``stored`` (M-STANDING).
        withheld_units = face.not_text(raw.get("api_content"), ("message", "api_content"), STORED_KIND)
        if withheld_units:
            face.count("api_content", withheld_units)      # per member of a list (ruling D-6)
    # Readable reasoning: the message's reasoning first, then every other readable text not
    # contained in it or an earlier such part.
    before: list = []
    shown: list[str] = []
    if main:
        field_ = "reasoning" if raw.get("reasoning") == main else "reasoning_content"
        before.append(given.made({"type": "text", "text": f"[Readable reasoning stored in "
                                                          f"{_path_text(('message', field_))}:]\n{main}"}, FIELD,
                                 ("message", field_)))
        given.values.append((("lcm", "reasoning"), main, _standing(REASONING, standing)))
        shown.append(main)
    before.extend(_readable_parts(face.readable, given, shown))
    known = _host_metadata_keys()
    for key, value in raw.items():
        if key == "role" or carries_nothing(value):
            continue
        if key in _TRANSCRIPT_KEYS:
            if key not in domain:
                _foreign(key, value, raw, face, shown)
        elif not _is_metadata(key, known):
            # T14: an unknown top-level key, walked as a stored value of no kind the store tells; one
            # named by the opaque vocabulary is withheld whole (M1), and a note in the record's
            # after-parts marks its place, since no JSON holds a marker there (the orchestrator's
            # ruling on PI-3, 2026-09-28; nothing of it is recorded).
            where = ("message", key)
            if _withholds(key, STORED_KIND):
                face.after.append(given.made({"type": "text", "text": (
                    f"[{_path_text(where)}: withheld, opaque replay material under this key; counted in the header's "
                    f"encrypted_withheld]")}, NOTE, where))
                face.count("other stored keys")
            elif face.json(f"[The stored key {_path_text(where)} is no field the query knows; shown as its JSON by the "
                           f"query:]", value, where, STORED_KIND):
                face.count("other stored keys")
    for kind, count in face.opaque.items():
        withheld[kind] = withheld.get(kind, 0) + count
    unparsed = []
    for index, call in zip(positions, message.get("tool_calls") or []):
        why = parsed_arguments(call["function"]["arguments"])[1]
        if why is not None:
            path = ("message", "tool_calls", index, "function", "arguments")
            named = (call_handles or {}).get(index) or _path_text(("message", "tool_calls", index))
            unparsed.append(given.made({"type": "text", "text": (
                f"[The arguments of tool call {named} ({call['function']['name']}) as stored; {why}:]\n"
                f"{call['function']['arguments']}")}, FIELD, path))
    _assemble(message, before, face.after + unparsed, given)
    # Every image was decided where it was met (M3): none stands in the content of a model that
    # does not read images, or where that is not known.
    content = message.get("content")
    if not facts.reads_images and isinstance(content, list) and any(image_part(part) for part in content):
        raise AssertionError(f"record {record}: an image part stands in the content of a model that is not given "
                             f"images (PLAN-83g §3.3)")
    payload = (isinstance(content, str) and not carries_nothing(content)) or (isinstance(content, list) and content) \
        or message.get("tool_calls")
    if not payload and face.opaque:
        message["content"] = [given.made({"type": "text", "text": _ONLY_WITHHELD}, NOTE)]
    if fill is not None:
        note = f"{FILL_NOTE_LABEL} {json.dumps(fill, ensure_ascii=False)}"
        _assemble(message, [given.made({"type": "text", "text": note}, NOTE)], [], given)
    return message


def _call_arguments(call: dict) -> tuple[Any, bool]:
    arguments, why = parsed_arguments(call["function"]["arguments"])
    return arguments, why is None
