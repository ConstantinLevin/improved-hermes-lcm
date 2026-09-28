"""What the query gives its model of one stored record (#19, PR #83; PLAN-83d, PLAN-83e).

The query's description states a rule to the agent: each stored message is given as its role,
content, tool calls and readable reasoning; any stored value that cannot be given in that shape
is given as labelled JSON in its place; an image the model is not given is replaced by a
placeholder that says so; signed or encrypted material is withheld and counted; the host's
bookkeeping keys and a replay carrier's metadata are not given; null, empty and blank values
carry nothing and are not given as parts. Each clause is made true here by one mechanism every
record passes through, never by a sentence applied at each site by hand:

- ``carries_nothing`` is the one test of "carries nothing", asked at every place the query
  decides whether and how to give a stored value (a key, a member of a list it walks, a key of
  an entry or block it classifies); inside a value given as JSON nothing is removed, since the
  JSON is the stored value;
- ``_render`` is the only place a stored value becomes labelled JSON, and it faces every
  structural image part inside the value (PLAN-83e §3, ruling OD-E1): where it stood in the
  stored content, which the agent's model was shown, it is lifted out and given as an image
  under the image rules; anywhere else no converter ever read it as an image, and it is named
  and counted, not given;
- one key classification (``_CLASSES``), read by one walker per stored field, faces every key of
  every entry of every reasoning field and every block of every replay carrier, and of every
  stored tool call (PLAN-83e §4);
- the message sent is built positively from the stored role's domain (``strict_message``,
  PLAN-83e §5): role, content, tool calls on an agent message, the call id on a tool result, and
  ``reasoning_content`` as the host's echo policy leaves it; every other stored key is given as
  a part, withheld and counted, or the host's bookkeeping;
- every part of the content is recorded with its origin where it is made (``Given``), so that
  what a label says about a part comes from how the part was made, and the query asserts that
  every part has one (``unrecorded_parts``).

Host facts at Hermes 375930d089 (PR #83, PLAN-83e §0: readers H1-H4 and the host's own lines
named where they are used).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from .message_content import content_parts, image_media_type, is_image_part, readable_reasoning, sidecar_sent
from .summariser_input import WireFacts, _image_placeholder, _row_before_fill, _strict_import, host_fill_text

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
    """A stored path as the query's labels write it: ``reasoning_details[0].summary``."""
    steps = list(path[1:] if path and path[0] == "message" else path)
    text = ""
    for step in steps:
        text += f"[{step}]" if isinstance(step, int) else (f".{step}" if text else str(step))
    return text or "message"


@dataclass
class Given:
    """What the query gives of one record beside the message: every string with its path and
    standing (``values``); the record's problems that refuse the query (``problems``); the origin
    of every part of the message's content, recorded where the part is made, with the stored path
    it came from and, for a placeholder, the origin of what it replaced (``origins``); the paths
    of image parts no converter ever read as images, which are named and not given
    (``images_not_given``); counts for the header."""

    values: list = field(default_factory=list)
    problems: list = field(default_factory=list)
    origins: dict = field(default_factory=dict)       # id(part) -> (part, origin, path, was)
    images_not_given: list = field(default_factory=list)
    reasoning_parts: int = 0
    lifted: int = 0

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


def unrecorded_parts(message: dict, given: Given) -> int:
    """How many parts of the message's content have no recorded origin (the query refuses a
    record with any: PLAN-83d §2's property, asserted over every message, PLAN-83e §8.3)."""
    content = message.get("content")
    return sum(1 for part in content if given.origin(part) is None) if isinstance(content, list) else 0


class _Marker(str):
    """A text the query puts inside a value given as JSON in place of an image part; not a
    value of the record."""


def _leaves(value: Any, path: tuple) -> Iterable[tuple[tuple, str]]:
    """Every string of a value, with its path; a number or a boolean as its JSON text."""
    if isinstance(value, _Marker):
        return
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


def image_part(part: Any) -> bool:
    """An image part by structure (``message_content.is_image_part``, the union of the host's
    tests on message parts: agent/context_compressor.py 1569-1579, model_metadata.py 2450), for
    any stored value: a ``type`` that is not a string is no image type and is never hashed."""
    return isinstance(part, dict) and isinstance(part.get("type"), str) and is_image_part(part)


def _as_image(block: Any) -> Optional[dict]:
    """The image part the wire carries for a stored image block: an ``image_url`` or
    ``input_image`` part as stored; an Anthropic ``image`` block with a base64 source of a known
    media type, or a URL source, as the ``image_url`` part it holds; None where the block holds
    no image the query can give as one."""
    if not image_part(block):
        return None
    if block["type"] != "image":
        return block
    source = block.get("source")
    if not isinstance(source, dict):
        return None
    if source.get("type") == "base64" and not carries_nothing(source.get("data")) and isinstance(
            source.get("data"), str) and isinstance(source.get("media_type"), str) and source["media_type"].strip():
        return {"type": "image_url", "image_url": {"url": f"data:{source['media_type']};base64,{source['data']}"}}
    if isinstance(source.get("url"), str) and source["url"].strip():
        return {"type": "image_url", "image_url": {"url": source["url"]}}
    return None


def _give_image(block: Any, path: tuple, given: Given, origin: str, label: Optional[str] = None) -> list[dict]:
    """An image block given as an image part (the image rules then apply to it like to every
    image of the message), or, where it holds no image the query can give as one, a text that
    says so; the image is then counted as not given."""
    image = _as_image(block)
    if image is None:
        given.images_not_given.append(path)
        return [given.made({"type": "text", "text": f"[{_path_text(path)} holds an image block of a shape the query "
                                                    f"cannot give as an image ({image_media_type(block)}); not "
                                                    f"given]"}, FIELD, path)]
    parts = [given.made({"type": "text", "text": label}, FIELD, path)] if label else []
    return parts + [given.made(image if image is not block else block, origin, path)]


def _face(value: Any, path: tuple, lift: bool, given: Given, images: list) -> Any:
    """The value with every structural image part inside it replaced by a text of the query's
    (PLAN-83e §3): lifted (given as its own part after the JSON) or named as not given."""
    if image_part(value):
        if lift:
            given.lifted += 1
            images.append((path, value))
            return _Marker(f"[image {given.lifted} of this record, given as its own part after this one]")
        given.images_not_given.append(path)
        return _Marker(f"[an image part ({image_media_type(value)}) stored here; the agent's context never held it "
                       f"as an image, so it is not given as one]")
    if isinstance(value, dict):
        return {key: _face(item, path + (key,), lift, given, images) for key, item in value.items()}
    if isinstance(value, list):
        return [_face(item, path + (index,), lift, given, images) for index, item in enumerate(value)]
    return value


def _render(label: str, value: Any, path: tuple, standing: str, given: Given, origin: str = FIELD, *,
            lift: bool = False) -> list[dict]:
    """The only place a stored value becomes labelled JSON: nothing where the value carries
    nothing; else the label, the value's JSON with every image part faced, and, where the value
    stood in the stored content (``lift``), each image part inside it as its own part."""
    if carries_nothing(value):
        return []
    images: list = []
    faced = _face(value, path, lift, given, images)
    part = {"type": "text", "text": f"{label}\n{json.dumps(faced, ensure_ascii=False)}"}
    given.values.extend((leaf_path, text, standing) for leaf_path, text in _leaves(faced, path))
    parts = [given.made(part, origin, path)]
    for image_path, image in images:
        parts.extend(_give_image(image, image_path, given, LIFTED))
    return parts


def text_part(part: Any) -> bool:
    """A text part as the host writes it: exactly a ``type`` "text" and a string ``text``
    (agent/image_routing.py 558, turn_context.py 149-157 at Hermes 375930d089). A member with
    further keys is given as its JSON, so no key of it is dropped (PLAN-83d §2)."""
    return (isinstance(part, dict) and set(part) == {"type", "text"} and part["type"] == "text"
            and isinstance(part["text"], str))


def _canonical_parts(parts: list, prefix: tuple, standing: str, given: Given) -> list:
    """A list content's members: a text part or an image part as stored, any other member as a
    labelled part holding its JSON (its image parts lifted out), a member that carries nothing
    not at all."""
    shown = []
    for index, part in enumerate(parts):
        if carries_nothing(part) or (text_part(part) and carries_nothing(part["text"])):
            continue
        if text_part(part):
            shown.append(given.made(part, STORED, prefix + (index,)))
            given.values.append((prefix + (index, "text"), part["text"], standing))
        elif image_part(part):
            shown.append(given.made(part, STORED, prefix + (index,)))
        else:
            shown.extend(_render(f"[Member {index + 1} of the stored content is not a text part (a type and a text "
                                 f"only) or an image part; shown as its JSON by the query:]", part, prefix + (index,),
                                 standing, given, RENDERED, lift=True))
    return shown


def _canonical_content(content: Any, standing: str, given: Given) -> Any:
    """The message's content: a string or null as stored; a list as its members; the host's
    ``_multimodal`` envelope as its parts and one labelled part holding every other key of it (its
    ``_multimodal`` mark included, so that the part says the content was stored as an envelope);
    any other value as one labelled part holding its JSON (ruling OD-E)."""
    base = ("message", "content")
    if content is None:
        return None
    if isinstance(content, str):
        if not carries_nothing(content):
            given.values.append((base, content, standing))
        return content
    if isinstance(content, list):
        return _canonical_parts(content, base, standing, given)
    parts = content_parts(content)
    if parts is not None and isinstance(content, dict):
        shown = _canonical_parts(parts, base + ("content",), standing, given)
        rest = {key: value for key, value in content.items() if key != "content"}
        shown.extend(_render("[The rest of this stored multimodal envelope (every key but its content), shown as its "
                             "JSON by the query:]", rest, base, standing, given, RENDERED, lift=True))
        return shown
    return _render(f"[The stored content is a JSON {json_kind(content)}, shown as its JSON by the query:]", content,
                   base, standing, given, RENDERED, lift=True)


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
# labelled JSON at its path.
READABLE, MESSAGE_TEXT, STASH_TEXT, CALL, IMAGE, OPAQUE, METADATA, CITED = (
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
    # reasoning_summaries.py 49-75), Anthropic thinking blocks (transports/anthropic.py 82-87).
    ("reasoning_details", "entry"): _table(readable=_READABLE_KEYS, opaque=_OPAQUE_KEYS, metadata=_ENTRY_METADATA),
    # codex_reasoning_items (codex_responses_adapter.py 1059-1084): summary pieces {type, text}.
    ("codex_reasoning_items", "item"): _table(opaque=("encrypted_content",), metadata=_ENTRY_METADATA,
                                              readable=("text", "content")),
    ("codex_reasoning_items", "piece"): _table(metadata=("type",), readable=("text",)),
    # anthropic_content_blocks, by block type (anthropic_message_convert.py 287-329): text blocks
    # keep citations and a cache marker; tool_use its sanitised id; image its source.
    # Citations survive only here (the stored content is the prose of the text blocks,
    # transports/anthropic.py 80-81, 102): given as their JSON (ruling OD-E3).
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
    (STASH, "image"): _table(metadata=("type", "cache_control"), image=("source",)),
    # bedrock_content_blocks, stored flattened (bedrock_adapter.py 824-899, 909-998): a block can
    # hold a text and a reasoning together (977).
    ("bedrock_content_blocks", "block"): _table(message_text=("text",)),
    ("bedrock_content_blocks", "reasoningContent"): _table(readable=("text",), opaque=(
        "signature", "redactedContentBase64", "redactedContent")),
    ("bedrock_content_blocks", "reasoningText"): _table(readable=("text",), opaque=("signature",)),
    ("bedrock_content_blocks", "toolUse"): _table(metadata=("toolUseId",), call=("name", "input")),
    # codex_message_items (codex_responses_adapter.py 367-373, 1124-1139): one output_text part.
    ("codex_message_items", "item"): _table(metadata=("type", "role", "status", "id", "phase")),
    ("codex_message_items", "part"): _table(metadata=("type",), message_text=("text",)),
    # A stored tool call (chat_completion_helpers.py 1622-1663): its ids beside ``id`` and its
    # type are the host's; ``extra_content`` holds the stored model's thought signature
    # (transports/chat_completions.py 274-288), withheld and counted (ruling OD-P2a).
    ("tool_calls", "call"): _table(metadata=("call_id", "response_item_id"), opaque=("extra_content",)),
}


@dataclass
class _Record:
    """One record's facing: what the classification gives and counts, gathered in one place."""

    record: str
    given: Given
    standing: str
    opaque: dict
    readable: list = field(default_factory=list)      # (path, text) of readable reasoning
    after: list = field(default_factory=list)         # parts after the content
    content_texts: list = field(default_factory=list)
    calls: list = field(default_factory=list)         # (name, arguments, parsed) of the calls given

    def json(self, label: str, value: Any, path: tuple, standing: Optional[str] = None) -> None:
        self.after.extend(_render(label, value, path, standing or self.standing, self.given))

    def not_text(self, value: Any, path: tuple, standing: Optional[str] = None) -> None:
        self.json(f"[{_path_text(path)} as stored is not text; shown as its JSON by the query:]", value, path,
                  standing)

    def readable_text(self, value: Any, path: tuple) -> None:
        if isinstance(value, str):
            self.readable.append((path, value))
        else:
            self.not_text(value, path, GIVEN_REASONING)

    def carried(self, value: Any, path: tuple) -> None:
        """A replay carrier's message text must stand in what is given of the message: its content,
        or its readable reasoning (a Codex commentary item's text is the host's ``reasoning``,
        codex_responses_adapter.py 1125-1134); else the record is a problem that refuses the query."""
        if not isinstance(value, str):
            self.not_text(value, path)
        elif not any(value in text for text in self.content_texts):
            self.given.problems.append(f"its replay carrier holds text its content does not ({_path_text(path)})")

    def stashed(self, value: Any, path: tuple) -> None:
        """The host's Anthropic converter replaces a tool result's content with its stash, so a
        stashed text the content lacks is given as its own part."""
        if not isinstance(value, str):
            self.not_text(value, path)
        elif not any(value in text for text in self.content_texts):
            self.given.values.append((path, value, self.standing))
            self.after.append(self.given.made({"type": "text", "text": (
                f"[A text block stored in {_path_text(path)}, which the stored content does not hold:]\n{value}")},
                FIELD, path))

    def called(self, name: Any, arguments: Any, path: tuple) -> None:
        """A replay carrier's call must be one of the calls given: by name and parsed input
        (PLAN-83c §5.2; the block's id is the host's sanitised one, not compared)."""
        if not any(name == stored_name and (not parsed or arguments == stored_arguments)
                   for stored_name, stored_arguments, parsed in self.calls):
            self.given.problems.append(f"its replay carrier holds a call its tool calls do not ({_path_text(path)}, "
                                       f"{name if isinstance(name, str) else '?'})")

    def keys(self, container: dict, table: dict, path: tuple, *, skip: tuple = ()) -> bool:
        """Every key of an entry or block by its class; returns whether opaque material that
        carries something was withheld."""
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
                self.readable_text(value, where)
            elif cls == MESSAGE_TEXT:
                self.carried(value, where)
            elif cls == STASH_TEXT:
                self.stashed(value, where)
            elif cls == CITED:
                self.json(f"[{_path_text(where)}: the citations the provider returned with this text, which the "
                          f"stored content does not hold; shown as their JSON by the query:]", value, where)
            else:
                standing = (GIVEN_REASONING if path[1] in REASONING_REPLAY_CARRIERS
                            else GIVEN_CALL if path[1] == "tool_calls" else None)
                self.json(f"[{_path_text(where)} is a key of no kind the query knows; shown as its JSON by the "
                          f"query:]", value, where, standing)
        return withheld

    def count(self, kind: str) -> None:
        self.opaque[kind] = self.opaque.get(kind, 0) + 1

    def not_a_list(self, key: str, value: Any, standing: Optional[str] = None) -> None:
        self.json(f"[{key} as stored is not a list; shown as its JSON by the query:]", value, ("message", key),
                  standing)

    def members(self, key: str, value: Any, standing: Optional[str] = None) -> Iterable[tuple[tuple, dict]]:
        """The members of a stored list field that carry something and are objects; any other
        member, and a value that is not a list, given as JSON."""
        if carries_nothing(value):
            return
        if not isinstance(value, list):
            self.not_a_list(key, value, standing)
            return
        for index, member in enumerate(value):
            path = ("message", key, index)
            if carries_nothing(member):
                continue
            if isinstance(member, dict):
                yield path, member
            else:
                self.json(f"[{_path_text(path)} as stored is not an entry of the host's shape; shown as its JSON by "
                          f"the query:]", member, path, standing)


def _reasoning_details(value: Any, face: _Record) -> None:
    """A ``<provider>.native_assistant`` entry is another provider's private replay carrier
    (agent/transports/chat_completions.py:388, providers/base.py 100-102): its top-level readable
    strings are given (the host merged the first into ``reasoning``), all else of it withheld and
    named by its type (ruling OD-P2b)."""
    table = _CLASSES[("reasoning_details", "entry")]
    for path, entry in face.members("reasoning_details", value, GIVEN_REASONING):
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
        elif face.keys(entry, table, path):
            face.count("reasoning_details")


def _codex_reasoning_items(value: Any, face: _Record) -> None:
    for path, item in face.members("codex_reasoning_items", value, GIVEN_REASONING):
        withheld = face.keys(item, _CLASSES[("codex_reasoning_items", "item")], path, skip=("summary",))
        summary = item.get("summary")
        if isinstance(summary, list):
            for number, piece in enumerate(summary):
                where = path + ("summary", number)
                if carries_nothing(piece):
                    continue
                if isinstance(piece, dict):
                    face.keys(piece, _CLASSES[("codex_reasoning_items", "piece")], where)
                else:
                    face.json(f"[{_path_text(where)} is not a summary piece of the host's shape; shown as its JSON by "
                              f"the query:]", piece, where, GIVEN_REASONING)
        elif not carries_nothing(summary):
            face.json(f"[{_path_text(path + ('summary',))} is not a list of summary pieces; shown as its JSON by the "
                      f"query:]", summary, path + ("summary",), GIVEN_REASONING)
        if withheld:
            face.count("codex_reasoning_items")


def _anthropic_blocks(key: str, value: Any, face: _Record) -> None:
    """``anthropic_content_blocks`` of an agent message, or the stash of a tool result, block by
    block and key by key; an image block is given as an image (the host's converter replays it,
    anthropic_message_convert.py 309-311, 443-447); a block of a type the host does not write is
    given whole as its JSON."""
    for path, block in face.members(key, value):
        kind = block.get("type")
        table = _CLASSES.get((key if key == STASH else "anthropic_content_blocks", kind)) if isinstance(kind, str) \
            else None
        if key == STASH and kind in ("image_url", "input_image"):
            face.after.extend(_give_image(block, path, face.given, FIELD,
                                          f"[An image block stored in {_path_text(path)}, given as an image:]"))
            continue
        if table is None:
            face.json(f"[{_path_text(path)} is a block of no type the query knows; shown as its JSON by the query:]",
                      block, path)
            continue
        if kind == "image":
            face.after.extend(_give_image(block, path, face.given, FIELD,
                                          f"[An image block stored in {_path_text(path)}, given as an image:]"))
            withheld = face.keys(block, table, path, skip=("source",))
        elif kind == "tool_use":
            face.called(block.get("name"), block.get("input"), path)
            withheld = face.keys(block, table, path, skip=("name", "input"))
        else:
            withheld = face.keys(block, table, path)
        if withheld:
            face.count(key)


def _bedrock_blocks(value: Any, face: _Record) -> None:
    """``bedrock_content_blocks`` key by key: a block can hold a text and a reasoning together."""
    field_ = "bedrock_content_blocks"
    for path, block in face.members(field_, value):
        withheld = face.keys(block, _CLASSES[(field_, "block")], path, skip=("reasoningContent", "toolUse"))
        reasoning = block.get("reasoningContent")
        if isinstance(reasoning, dict):
            where = path + ("reasoningContent",)
            withheld = face.keys(reasoning, _CLASSES[(field_, "reasoningContent")], where, skip=("reasoningText",)) \
                or withheld
            nested = reasoning.get("reasoningText")
            if isinstance(nested, dict):
                withheld = face.keys(nested, _CLASSES[(field_, "reasoningText")], where + ("reasoningText",)) \
                    or withheld
            elif not carries_nothing(nested):
                face.readable_text(nested, where + ("reasoningText",))
        elif not carries_nothing(reasoning):
            face.json(f"[{_path_text(path + ('reasoningContent',))} is not an object of the host's shape; shown as its "
                      f"JSON by the query:]", reasoning, path + ("reasoningContent",))
        tool = block.get("toolUse")
        if isinstance(tool, dict):
            face.called(tool.get("name"), tool.get("input"), path + ("toolUse",))
            face.keys(tool, _CLASSES[(field_, "toolUse")], path + ("toolUse",), skip=("name", "input"))
        elif not carries_nothing(tool):
            face.json(f"[{_path_text(path + ('toolUse',))} is not an object of the host's shape; shown as its JSON by "
                      f"the query:]", tool, path + ("toolUse",))
        if withheld:
            face.count(field_)


def _codex_message_items(value: Any, face: _Record) -> None:
    """``codex_message_items``: message items of one ``output_text`` part (the host's replay reads
    only such parts, codex_responses_adapter.py 457, so an image part there was never an image to
    any converter: ruling OD-E1)."""
    field_ = "codex_message_items"
    for path, item in face.members(field_, value):
        parts = item.get("content")
        if item.get("type") != "message" or not isinstance(parts, list):
            face.json(f"[{_path_text(path)} is an item of no shape the query knows; shown as its JSON by the query:]",
                      item, path)
            continue
        face.keys(item, _CLASSES[(field_, "item")], path, skip=("content",))
        for number, part in enumerate(parts):
            where = path + ("content", number)
            if carries_nothing(part):
                continue
            if isinstance(part, dict) and part.get("type") in ("output_text", "text"):
                face.keys(part, _CLASSES[(field_, "part")], where)
            else:
                face.json(f"[{_path_text(where)} is a part of no shape the query knows; shown as its JSON by the "
                          f"query:]", part, where)


def _tool_calls(value: Any, face: _Record) -> list[dict]:
    """An agent message's calls as the wire carries them: each call of the host's shape as its
    id, type and function's name and arguments; every other key of it faced by the table (a
    stored type other than "function" given as its JSON, never overwritten unsaid); any other
    value or call given as labelled JSON after the content, not a call on the wire."""
    base = ("message", "tool_calls")
    if carries_nothing(value):
        return []
    if not isinstance(value, list):
        face.json(f"[The stored tool calls are a JSON {json_kind(value)}, not a list of calls; shown as their JSON by "
                  f"the query:]", value, base, GIVEN_CALL)
        return []
    kept = []
    table = _CLASSES[("tool_calls", "call")]
    for index, call in enumerate(value):
        path = base + (index,)
        if carries_nothing(call):
            continue
        if not canonical_call(call):
            face.json(f"[Tool call {index + 1} as stored is not a call the wire can carry; shown as its JSON by the "
                      f"query:]", call, path, GIVEN_CALL)
            continue
        function = call["function"]
        kept.append({"id": call["id"], "type": "function",
                     "function": {"name": function["name"], "arguments": function["arguments"]}})
        face.given.values.append((path + ("function", "name"), function["name"], GIVEN_CALL))
        face.given.values.append((path + ("function", "arguments"), function["arguments"], GIVEN_CALL))
        kind = call.get("type")
        if not carries_nothing(kind) and kind != "function":
            face.json(f"[{_path_text(path + ('type',))} as stored is not \"function\"; shown as its JSON by the "
                      f"query:]", kind, path + ("type",), GIVEN_CALL)
        if face.keys(call, table, path, skip=("id", "function", "type")):
            face.count("tool_calls")
        face.keys(function, {}, path + ("function",), skip=("name", "arguments"))
    return kept


def _host_metadata_keys() -> frozenset:
    """The keys the host writes on a message as its own bookkeeping, from the host's own constants
    at run time (ruling OD-3a): the session schema's columns (``hermes_state_messages
    ._MESSAGE_SCHEMA_KEYS``), the message core keys (``agent.message_sanitization._MESSAGE_CORE_KEYS``),
    the keys the Chat Completions transport strips (``_STRIP_MSG_KEYS``) and the persistence-only
    fields."""
    so = "which of a message's keys are the host's own bookkeeping is not known; nothing was sent"
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


def strict_message(raw: dict, record: str, facts: WireFacts, withheld: dict, given: Given) -> dict:
    """One record's message as the query gives it (PLAN-83e §2-§5). The host's per-row rules run
    first, each host function called strictly (the clone, the sidecar, the reasoning-echo policy,
    the fill decided on the host's own row); then the message is built from the stored role's
    domain and every stored key is faced. ``given`` receives what is given of the record and the
    origin of every part of its content; ``withheld`` counts, per stored field, the entries and
    blocks whose opaque material is withheld."""
    role = wire_role(raw)
    stored_role = raw.get("role")
    domain = _DOMAIN.get(stored_role, _OTHER_DOMAIN) if isinstance(stored_role, str) else _OTHER_DOMAIN
    row = _row_before_fill(raw, needs_echo=facts.needs_reasoning_echo, strict=True)
    fill = host_fill_text(row)
    standing = GIVEN_RESULT if role == "tool" else GIVEN_CONTENT
    message: dict = {"role": role}
    if "content" in row:
        message["content"] = _canonical_content(row["content"], standing, given)
    face = _Record(record, given, standing, {})
    # The keys in the order the host's own row has them (role, content, reasoning_content,
    # tool_calls; a tool result's call id after its content), so that the request is the host's.
    if "reasoning_content" in domain and isinstance(row.get("reasoning_content"), str):
        message["reasoning_content"] = row["reasoning_content"]   # as the host's echo policy left it
    if "tool_calls" in domain:
        message_calls = _tool_calls(raw.get("tool_calls"), face)
        if message_calls:
            message["tool_calls"] = message_calls
            face.calls = [(call["function"]["name"], *_call_arguments(call)) for call in message_calls]
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
            face.readable_text(value, ("message", key))
        _reasoning_details(raw.get("reasoning_details"), face)
        _codex_reasoning_items(raw.get("codex_reasoning_items"), face)
        _anthropic_blocks("anthropic_content_blocks", raw.get("anthropic_content_blocks"), face)
        _bedrock_blocks(raw.get("bedrock_content_blocks"), face)
        _codex_message_items(raw.get("codex_message_items"), face)
    if STASH in domain:
        _anthropic_blocks(STASH, raw.get(STASH), face)
    if "api_content" in domain and not sidecar_sent(raw):
        face.not_text(raw.get("api_content"), ("message", "api_content"))
    known = _host_metadata_keys()
    for key, value in raw.items():
        if key == "role" or carries_nothing(value):
            continue
        if key in _TRANSCRIPT_KEYS:
            if key not in domain:
                face.json(f"[{key} is stored on this {json.dumps(stored_role) if stored_role is not None else 'role-less'} "
                          f"message, where the host writes no such field; shown as its JSON by the query:]", value,
                          ("message", key))
        elif not _is_metadata(key, known):
            face.json(f"[The stored key {key!r} is no field the query knows; shown as its JSON by the query:]", value,
                      ("message", key))
    for kind, count in face.opaque.items():
        withheld[kind] = withheld.get(kind, 0) + count
    # Readable reasoning: the message's reasoning, then every other readable text not contained in
    # it or an earlier such part (exact containment, ruling OD-4a).
    before: list = []
    shown: list[str] = []
    if main:
        field_ = "reasoning" if raw.get("reasoning") == main else "reasoning_content"
        before.append(given.made({"type": "text", "text": f"[Readable reasoning stored in {field_}:]\n{main}"}, FIELD,
                                 ("message", field_)))
        given.values.append((("lcm", "reasoning"), main, GIVEN_REASONING))
        shown.append(main)
    for path, text in face.readable:
        if carries_nothing(text) or any(text in earlier for earlier in shown):
            continue
        before.append(given.made({"type": "text", "text": f"[Readable reasoning stored in {_path_text(path)}:]\n"
                                                          f"{text}"}, FIELD, path))
        given.values.append((path, text, GIVEN_REASONING))
        given.reasoning_parts += 1
        shown.append(text)
    unparsed = []
    for index, call in enumerate(message.get("tool_calls") or []):
        why = parsed_arguments(call["function"]["arguments"])[1]
        if why is not None:
            unparsed.append(given.made({"type": "text", "text": (
                f"[The arguments of tool call {call['id']} ({call['function']['name']}) as stored; {why}:]\n"
                f"{call['function']['arguments']}")}, FIELD, ("message", "tool_calls", index, "function", "arguments")))
    _assemble(message, before, face.after + unparsed, given)
    # The image rules, over every image the message now holds (stored, lifted, a carrier's or the
    # stash's): sent where the model reads images, else a placeholder that says so.
    if not facts.reads_images and isinstance(message.get("content"), list):
        why = _NOT_KNOWN if facts.reads_images is None else _NOT_READ
        message["content"] = [given.made(_image_placeholder(part, record, why), REPLACED,
                                         given.origins[id(part)][2] if given.origin(part) else (),
                                         given.origin(part)) if image_part(part) else part
                              for part in message["content"]]
    content = message.get("content")
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
