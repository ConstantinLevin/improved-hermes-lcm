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
- ``_image_outcome`` is the one function that decides every image the query meets (M3): given
  as an image, replaced by a placeholder, named where it stood (a field the query does not give as
  an image, or a shape it cannot give as one), each marker or label written from that decision;
- one key classification (``_CLASSES``), read by one walker per stored field, faces every key of
  every entry of every reasoning field and every block of every replay carrier, and of every
  stored tool call (PLAN-83e §4); a field stored on a role whose domain lacks it is walked by its
  own walker, its outputs given as parts (M4);
- the message sent is built positively from the stored role's domain (``strict_message``,
  PLAN-83e §5): role, content, tool calls on an agent message, the call id on a tool result, and
  ``reasoning_content`` as the host's echo policy leaves it; every other stored key is given as
  a part, withheld and counted, or the host's bookkeeping;
- every part of the content is recorded with its origin where it is made (``Given``), so that
  what a label says about a part comes from how the part was made, and the query asserts that
  every part has one (``unrecorded_parts``); a stored place and a stored role are each written one
  way (``_path_text``, ``stored_role_text``, M5).

Host facts at Hermes 375930d089 (PR #83, PLAN-83e §0, PLAN-83g §0: the readers and the host's own
lines named where they are used).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from .message_content import content_parts, image_media_type, is_image_part, readable_reasoning, sidecar_sent
from .summariser_input import (HostUnavailable, WireFacts, _image_placeholder, _row_before_fill, _strict_import,
                               host_fill_text)

# The standing of a value the query gives: what the agent may rest on an excerpt found in it.
GIVEN_CONTENT, GIVEN_RESULT, GIVEN_CALL, GIVEN_REASONING = "content", "result", "call", "reasoning"

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
    problems: list = field(default_factory=list)
    origins: dict = field(default_factory=dict)       # id(part) -> (part, origin, path, was)
    images_replaced: int = 0                          # replaced by a placeholder that says so
    images_elsewhere: list = field(default_factory=list)     # paths: a field the query does not give as an image
    images_ungivable: list = field(default_factory=list)     # paths: a shape the query cannot give as an image
    images_behind_sidecar: int = 0                    # in a stored content the host sends api_content in place of
    reasoning_parts: int = 0
    lifted: int = 0
    joined: bool = False      # the content was joined into one string by the query (``query._join``)
    record: str = ""
    reads_images: Optional[bool] = None

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


def _url_of(value: Any) -> Optional[str]:
    """An ``image_url`` value's URL: a non-blank string, or a dict with a non-blank string ``url``."""
    if isinstance(value, str) and value.strip():
        return value
    if isinstance(value, dict) and isinstance(value.get("url"), str) and value["url"].strip():
        return value["url"]
    return None


def _as_image(block: dict) -> tuple[Optional[dict], str]:
    """The image part the wire carries for a stored image block, or None and why the query cannot
    give it as one. An ``image_url`` part needs an ``image_url`` that is a non-blank string or a dict
    with a non-blank string ``url`` and is given as ``{type, image_url}`` as stored; an
    ``input_image`` part likewise, its ``detail`` kept only as a string; an Anthropic ``image`` block
    needs a base64 source with non-blank data and a non-blank media type (never supplied by the
    plugin: a guess), or a non-blank URL, and is given as the ``image_url`` part it holds."""
    kind = block["type"]
    if kind in ("image_url", "input_image"):
        if _url_of(block.get("image_url")) is None:
            return None, "no url"
        image = {"type": kind, "image_url": block["image_url"]}
        if kind == "input_image" and isinstance(block.get("detail"), str):
            image["detail"] = block["detail"]
        return image, ""
    source = block.get("source")
    if not isinstance(source, dict):
        return None, "no source"
    if source.get("type") == "base64":
        data, media = source.get("data"), source.get("media_type")
        if isinstance(data, str) and data.strip() and isinstance(media, str) and media.strip():
            return {"type": "image_url", "image_url": {"url": f"data:{media};base64,{data}"}}, ""
        return None, "a base64 source without its data or its media type"
    if isinstance(source.get("url"), str) and source["url"].strip():
        return {"type": "image_url", "image_url": {"url": source["url"]}}, ""
    return None, "a source of no shape the query can give"


def _other_image_keys(block: dict) -> dict:
    """What an image part holds beside what makes its image (and the host's cache marker)."""
    named = _IMAGE_KEYS[block["type"]]
    other = {key: value for key, value in block.items() if key not in named and key != "cache_control"}
    if block["type"] == "input_image" and "detail" in block and not isinstance(block["detail"], str):
        other["detail"] = block["detail"]
    return other


GIVEN_IMAGE, REPLACED_IMAGE, NOT_HERE, NOT_GIVABLE = "given", "replaced", "not here", "not givable"


def _image_outcome(block: dict, path: tuple, image_site: bool, given: Given) -> tuple[str, Optional[dict], str]:
    """The one decision on an image the query meets (M3): ``(outcome, part, text)``.

    At a place where the query does not give an image as an image (inside a value rendered from
    a reasoning field, a call, a carrier's other keys, a ``codex_message_items`` part, an unknown
    key: ELSEWHERE) it is named where it stood; at an image site (a content member, an envelope
    member, an image lifted from the stored content, a carrier's or the stash's image block) a
    block of a shape the query cannot give is named with why; where the model does not read
    images, or that is not known, it is replaced by the placeholder that says so; else it is given
    as the image part the wire carries. Each image is counted in exactly one class of ``Given``;
    ``text`` is what a marker or a note says of it."""
    media = image_media_type(block)
    if not image_site:
        given.images_elsewhere.append(path)
        return NOT_HERE, None, (f"[an image part ({media}) stored here, in a field the query does not give as an "
                                f"image; not given as one]")
    image, why = _as_image(block)
    if image is None:
        given.images_ungivable.append(path)
        return NOT_GIVABLE, None, (f"[{_path_text(path)} holds an image block of a shape the query cannot give as an "
                                   f"image ({why}); not given]")
    if not given.reads_images:
        given.images_replaced += 1
        placeholder = _image_placeholder(block, given.record, _NOT_KNOWN if given.reads_images is None else _NOT_READ)
        return REPLACED_IMAGE, placeholder, placeholder["text"]
    given.lifted += 1
    return GIVEN_IMAGE, image, f"[image {given.lifted} of this record, given as its own part after this one]"


def _image_parts(block: dict, path: tuple, given: Given, origin: str, kind: str, standing: str,
                 label: Optional[str] = None) -> tuple[list[dict], bool]:
    """An image block at an image site given as parts, in its place: the image (after ``label``,
    written only where the image is given), and its other stored keys as their JSON beside it; a
    placeholder, or a note that says why it is not given. Returns the parts and whether anything
    under a key of the opaque vocabulary was withheld from the other keys."""
    outcome, part, text = _image_outcome(block, path, True, given)
    if outcome == REPLACED_IMAGE:
        return [given.made(part, REPLACED, path, origin)], False
    if outcome == NOT_GIVABLE:
        return [given.made({"type": "text", "text": text}, NOTE, path)], False
    parts = [given.made({"type": "text", "text": label}, FIELD, path)] if label else []
    parts.append(given.made(part, origin, path))
    other, withheld = [], False
    rest = _other_image_keys(block)
    if rest:
        other, withheld = _render(f"[{_path_text(path)}: the other stored keys of this image part; shown as their JSON "
                                  f"by the query:]", rest, path, kind, standing, given, RENDERED if origin == STORED
                                  else FIELD)
    return parts + other, withheld


def _image_paths(value: Any, path: tuple) -> Iterable[tuple]:
    """Every image part inside a value, at any depth, each once (an image part is not searched
    further): the walk ``_render`` makes, with nothing given (M3, the images behind a sidecar)."""
    if image_part(value):
        yield path
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _image_paths(item, path + (key,))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _image_paths(item, path + (index,))


# --- M1: one walk carrying the kind of the stored field (PLAN-83g §3.1) --------------------------
#
# The kinds of a stored value: a message's content (members, an envelope, api_content, a tool
# result's content); a tool-call dict (canonical or not, wherever stored); a call's name and
# arguments and everything inside them; reasoning (the reasoning fields, the reasoning replay
# carriers, a thinking block, everything under them); a text replay carrier or the stash
# (everything under it unless a transition names another kind); an unknown top-level key. Three
# more names mark the places where a transition of table T is taken inside one walk: a list of
# calls, a call's ``function``, a tool_use block.
CONTENT, CALL, CALL_VALUE, REASONING, CARRIER, STORED_KIND = (
    "content", "call", "call value", "reasoning", "carrier", "stored")
_CALLS, _CALL_DICT, _FUNCTION, _TOOL_USE = "calls", "call dict", "function", "tool use"

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
_CALL_KINDS = frozenset({CALL, CALL_VALUE, _CALLS, _CALL_DICT, _FUNCTION, _TOOL_USE})
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
    """M2: a value's standing is its kind's (PLAN-83g §3.2): reasoning → reasoning; a call or its
    values → call; content, a carrier's value of no other kind and an unknown key → the record's
    own (content, or result on a tool row; ruling OD-F2)."""
    if kind == REASONING:
        return GIVEN_REASONING
    if kind in _CALL_KINDS:
        return GIVEN_CALL
    return base


@dataclass
class _Walk:
    given: Given
    lift: bool
    images: list = field(default_factory=list)       # (path, image part) given after the JSON
    withheld: bool = False


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
        inner = _CALL_DICT if kind == _CALLS else kind
        return [_step(item, path + (index,), inner, base, walk) for index, item in enumerate(value)]
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
    if kind == _CALLS:
        kind = _CALL_DICT
    faced = {}
    for key, item in value.items():
        where = path + (key,)
        if _withholds(key, kind) and not carries_nothing(item):
            walk.withheld = True
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
            lift: bool = False) -> tuple[list[dict], bool]:
    """The only place a stored value becomes labelled JSON (M1): nothing where the value carries
    nothing; else the label and the value's JSON as the walk gives it, and, where the value stood in
    the stored content (``lift``), each image given as an image after it. Returns the parts and
    whether anything under a key of the opaque vocabulary was withheld (the caller counts it under
    its stored field)."""
    if carries_nothing(value):
        return [], False
    walk = _Walk(given, lift)
    faced = _step(value, path, kind, base, walk)
    part = {"type": "text", "text": f"{label}\n{json.dumps(faced, ensure_ascii=False)}"}
    parts = [given.made(part, origin, path)]
    for image_path, image in walk.images:
        parts.append(given.made(image, LIFTED, image_path))
    return parts, walk.withheld


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
            face.withheld_content = face.withheld_content or withheld
        else:
            rendered, withheld = _render(f"[{_path_text(where)} is not a text part (a type and a text only) or an image "
                                         f"part; shown as its JSON by the query:]", part, where, CONTENT, standing,
                                         given, RENDERED, lift=True)
            shown.extend(rendered)
            face.withheld_content = face.withheld_content or withheld
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
        face.withheld_content = face.withheld_content or withheld
        return (shown + rendered) or ""
    rendered, withheld = _render(f"[The stored content is a JSON {json_kind(content)}, shown as its JSON by the "
                                 f"query:]", content, base, CONTENT, standing, given, RENDERED, lift=True)
    face.withheld_content = face.withheld_content or withheld
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
    ``foreign``: the field is stored on a role whose domain lacks it (M4): its carrier texts are
    given as parts, not held against the content."""

    record: str
    given: Given
    standing: str
    opaque: dict
    readable: list = field(default_factory=list)      # (path, text) of readable reasoning
    after: list = field(default_factory=list)         # parts after the content
    content_texts: list = field(default_factory=list)
    calls: list = field(default_factory=list)         # (name, arguments, parsed) of the calls given
    foreign: bool = False
    withheld_content: bool = False

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

    def carried(self, value: Any, path: tuple) -> bool:
        """A replay carrier's message text must stand in what is given of the message: its content,
        or its main readable reasoning (a Codex commentary item's text is the host's ``reasoning``,
        codex_responses_adapter.py 1125-1134); else the record is a problem that refuses the query.
        On a role where the host writes no carrier no converter reads it: a text the content lacks
        is given as its own part (ruling OD-F5)."""
        if not isinstance(value, str):
            return self.not_text(value, path, CARRIER)
        if any(value in text for text in self.content_texts):
            return False
        if self.foreign:
            self.given.values.append((path, value, GIVEN_CONTENT))
            self.after.append(self.given.made({"type": "text", "text": (
                f"[A text stored in {_path_text(path)} on this message, where the host writes no such field, which its "
                f"content does not hold:]\n{value}")}, FIELD, path))
        else:
            self.given.problems.append(f"its replay carrier holds text its content does not ({_path_text(path)})")
        return False

    def stashed(self, value: Any, path: tuple) -> bool:
        """The host's Anthropic converter replaces a tool result's content with its stash, so a
        stashed text the content lacks is given as its own part, a result."""
        if not isinstance(value, str):
            return self.not_text(value, path, CARRIER, GIVEN_RESULT)
        if not any(value in text for text in self.content_texts):
            self.given.values.append((path, value, GIVEN_RESULT))
            what = (f"[A text stored in {_path_text(path)} on this message, where the host writes no such field, which "
                    f"its content does not hold:]" if self.foreign else
                    f"[A text block stored in {_path_text(path)}, which the stored content does not hold:]")
            self.after.append(self.given.made({"type": "text", "text": f"{what}\n{value}"}, FIELD, path))
        return False

    def called(self, name: Any, arguments: Any, path: tuple) -> None:
        """A replay carrier's call must be one of the calls given: by name and parsed input
        (PLAN-83c §5.2; the block's id is the host's sanitised one, not compared). Where the stored
        call of that name has arguments that do not parse, its input cannot be shown equal: the
        record is refused naming the block (T5, the ruled extension of rule 3)."""
        named = [(stored_arguments, parsed) for stored_name, stored_arguments, parsed in self.calls
                 if name == stored_name]
        if any(parsed and arguments == stored_arguments for stored_arguments, parsed in named):
            return
        shown = name if isinstance(name, str) else "?"
        if any(not parsed for _stored_arguments, parsed in named):
            self.given.problems.append(f"its replay carrier holds a call whose input cannot be compared with its stored "
                                       f"arguments, which do not parse ({_path_text(path)}, {shown})")
        else:
            self.given.problems.append(f"its replay carrier holds a call its tool calls do not ({_path_text(path)}, "
                                       f"{shown})")

    def keys(self, container: dict, table: dict, path: tuple, other: str, *, skip: tuple = ()) -> bool:
        """Every key of an entry or block by its class; ``other``: the kind (table T) of every key
        the table does not name. Returns whether opaque material that carries something was
        withheld, at this level or inside a key given as JSON."""
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
                withheld = self.carried(value, where) or withheld
            elif cls == STASH_TEXT:
                withheld = self.stashed(value, where) or withheld
            elif cls == CITED:
                withheld = self.json(f"[{_path_text(where)}: the citations stored with this text block; shown as their "
                                     f"JSON by the query:]", value, where, CARRIER,
                                     GIVEN_RESULT if path[1] == STASH else None) or withheld
            elif _withholds(key, other):
                # A key of the opaque vocabulary the table does not name at this container (M1).
                withheld = True
            else:
                withheld = self.json(f"[{_path_text(where)} is a key of no kind the query knows; shown as its JSON by "
                                     f"the query:]", value, where, other) or withheld
        return withheld

    def count(self, kind: str) -> None:
        self.opaque[kind] = self.opaque.get(kind, 0) + 1

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


def _anthropic_blocks(key: str, value: Any, face: _Record) -> None:
    """``anthropic_content_blocks`` of an agent message, or the stash of a tool result, block by
    block and key by key (T3-T6); an image block is given as an image (the host's converter
    replays it, anthropic_message_convert.py 309-311, 443-447); a block of a type the host does not
    write is given whole as its JSON, walked as the carrier's (T8)."""
    for path, block in face.members(key, value, CARRIER):
        kind = block.get("type")
        table = _CLASSES.get((key if key == STASH else "anthropic_content_blocks", kind)) if isinstance(kind, str) \
            else None
        standing = GIVEN_RESULT if key == STASH else face.standing
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
            # T5: the block's name and input compared with the stored canonical calls (rule 3).
            face.called(block.get("name"), block.get("input"), path)
            withheld = face.keys(block, table, path, CALL, skip=("name", "input"))
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
            face.called(tool.get("name"), tool.get("input"), where)
            withheld = face.keys(tool, _CLASSES[(field_, "toolUse")], where, CALL, skip=("name", "input")) or withheld
        elif not carries_nothing(tool):
            withheld = face.json(f"[{_path_text(where)} is not an object of the host's shape; shown as its JSON by the "
                                 f"query:]", tool, where, CALL) or withheld
        if withheld:
            face.count(field_)


def _codex_message_items(value: Any, face: _Record) -> None:
    """``codex_message_items``: message items of one ``output_text`` part (T7; the host's replay
    reads only such parts, codex_responses_adapter.py 457, so an image part there was never an image
    to any converter: ruling OD-E1); any other item or part is the carrier's (T8)."""
    field_ = "codex_message_items"
    for path, item in face.members(field_, value, CARRIER):
        parts = item.get("content")
        withheld = False
        if item.get("type") != "message" or not isinstance(parts, list):
            withheld = face.json(f"[{_path_text(path)} is an item of no shape the query knows; shown as its JSON by the "
                                 f"query:]", item, path, CARRIER)
        else:
            withheld = face.keys(item, _CLASSES[(field_, "item")], path, CARRIER, skip=("content",))
            for number, part in enumerate(parts):
                where = path + ("content", number)
                if carries_nothing(part):
                    continue
                if isinstance(part, dict) and part.get("type") in ("output_text", "text"):
                    withheld = face.keys(part, _CLASSES[(field_, "part")], where, CARRIER) or withheld
                else:
                    withheld = face.json(f"[{_path_text(where)} is a part of no shape the query knows; shown as its "
                                         f"JSON by the query:]", part, where, CARRIER) or withheld
        if withheld:
            face.count(field_)


def _tool_calls(value: Any, face: _Record) -> list[tuple[int, dict]]:
    """An agent message's calls as the wire carries them, with their stored positions: each call of
    the host's shape as its id, type and function's name and arguments (T1); every other key of it
    faced by the table (T2, T2′; a stored type other than "function" given as its JSON, never
    overwritten unsaid); any other value or call given as labelled JSON after the content, walked
    as a call, not a call on the wire."""
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
        function = call["function"]
        kept.append((index, {"id": call["id"], "type": "function",
                             "function": {"name": function["name"], "arguments": function["arguments"]}}))
        face.given.values.append((path + ("function", "name"), function["name"], GIVEN_CALL))
        face.given.values.append((path + ("function", "arguments"), function["arguments"], GIVEN_CALL))
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
    sub = _Record(face.record, face.given, face.standing, face.opaque, content_texts=face.content_texts,
                  foreign=True)
    path = ("message", key)
    if key in ("reasoning", "reasoning_content"):
        sub.readable_text(value, path)
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
        sub.not_text(value, path, CONTENT)
    else:
        # tool_call_id, name: the wire's pairing key and the host's bookkeeping, stored where the host
        # writes neither; given as they are stored.
        sub.json(f"[{_path_text(path)} as stored; shown as its JSON by the query:]", value, path, STORED_KIND)
    readable = _readable_parts(sub.readable, face.given, shown)
    if not (readable or sub.after):
        return
    label = face.given.made({"type": "text", "text": (f"[{_path_text(path)} is stored on this message "
                                                      f"{stored_role_text(raw)}, where the host writes no such field; "
                                                      f"what the query gives of it follows:]")}, NOTE, path)
    face.after.extend([label] + readable + sub.after)


def _readable_parts(readable: list, given: Given, shown: list) -> list[dict]:
    """Readable reasoning texts as labelled parts, each unless it carries nothing or is contained
    verbatim in the message's reasoning or an earlier such part (exact containment, ruling OD-4a)."""
    parts = []
    for path, text in readable:
        if carries_nothing(text) or any(text in earlier for earlier in shown):
            continue
        parts.append(given.made({"type": "text", "text": f"[Readable reasoning stored in {_path_text(path)}:]\n{text}"},
                                FIELD, path))
        given.values.append((path, text, GIVEN_REASONING))
        given.reasoning_parts += 1
        shown.append(text)
    return parts


def strict_message(raw: dict, record: str, facts: WireFacts, withheld: dict, given: Given,
                   call_handles: Optional[dict] = None) -> dict:
    """One record's message as the query gives it (PLAN-83e §2-§5, PLAN-83g §3). The host's per-row
    rules run first, each host function called strictly (the clone, the sidecar, the reasoning-echo
    policy with the pad the host's own agent applies, the fill decided on the host's own row); then
    the message is built from the stored role's domain and every stored key is faced. ``given``
    receives what is given of the record and the origin of every part of its content; ``withheld``
    counts, per stored field, the entries, blocks, calls and keys whose opaque material is withheld;
    ``call_handles`` names, by stored position, the handles the store minted for the record's calls."""
    given.record, given.reads_images = record, facts.reads_images
    role = wire_role(raw)
    stored_role = raw.get("role")
    domain = _DOMAIN.get(stored_role, _OTHER_DOMAIN) if isinstance(stored_role, str) else _OTHER_DOMAIN
    row = _row_before_fill(raw, needs_echo=facts.needs_reasoning_echo, strict=True)
    fill = host_fill_text(row)
    standing = GIVEN_RESULT if role == "tool" else GIVEN_CONTENT
    message: dict = {"role": role}
    face = _Record(record, given, standing, {})
    if "content" in row:
        message["content"] = _canonical_content(row["content"], standing, given, face)
    if face.withheld_content:
        face.count("content")
    if sidecar_sent(raw) and not carries_nothing(raw.get("content")):
        # M3: the images of a stored content the host sends api_content in place of (the host
        # replaces the content wholesale on every request, agent/turn_context.py 1231-1256), held
        # by the handles and not given; its label says the stored content is not given.
        given.images_behind_sidecar += sum(1 for _path in _image_paths(raw["content"], ("message", "content")))
    # The keys in the order the host's own row has them (role, content, reasoning_content,
    # tool_calls; a tool result's call id after its content), so that the request is the host's.
    if "reasoning_content" in domain and isinstance(row.get("reasoning_content"), str):
        message["reasoning_content"] = row["reasoning_content"]   # as the host's echo policy left it
    positions: list = []
    if "tool_calls" in domain:
        kept = _tool_calls(raw.get("tool_calls"), face)
        if kept:
            positions = [position for position, _call in kept]
            message["tool_calls"] = [call for _position, call in kept]
            face.calls = [(call["function"]["name"], *_call_arguments(call)) for _position, call in kept]
    if "tool_call_id" in domain and "tool_call_id" in raw:
        message["tool_call_id"] = raw["tool_call_id"]      # the wire's pairing key, as stored
    main = readable_reasoning(raw) if "reasoning" in domain else None
    face.content_texts = [text for _path, text, standing_ in given.values
                          if standing_ in (GIVEN_CONTENT, GIVEN_RESULT)] + ([main] if main else [])
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
        _anthropic_blocks(STASH, raw.get(STASH), face)
    if "api_content" in domain and not sidecar_sent(raw):
        # T13: a sidecar that is not a string is content, given as its JSON.
        face.not_text(raw.get("api_content"), ("message", "api_content"), CONTENT)
    # Readable reasoning: the message's reasoning first, then every other readable text not
    # contained in it or an earlier such part.
    before: list = []
    shown: list[str] = []
    if main:
        field_ = "reasoning" if raw.get("reasoning") == main else "reasoning_content"
        before.append(given.made({"type": "text", "text": f"[Readable reasoning stored in "
                                                          f"{_path_text(('message', field_))}:]\n{main}"}, FIELD,
                                 ("message", field_)))
        given.values.append((("lcm", "reasoning"), main, GIVEN_REASONING))
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
            # named by the opaque vocabulary is withheld whole (M1).
            if _withholds(key, STORED_KIND) or face.json(
                    f"[The stored key {_path_text(('message', key))} is no field the query knows; shown as its JSON by "
                    f"the query:]", value, ("message", key), STORED_KIND):
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
