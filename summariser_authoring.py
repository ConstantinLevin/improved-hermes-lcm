"""Facts about the original reader and the reply from its actual invocation.

These facts describe the provider call observed by the plugin. They do not claim
that a remote service received a particular wire payload.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Any, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from .summariser_input import ReaderInput


class AuthoringUnavailable(RuntimeError):
    def __init__(self, operation: str, reason: str, record: str | None = None):
        self.operation = operation
        self.record = record
        self.reason = reason
        owner = f"record {record}: " if record else ""
        super().__init__(f"{owner}{operation}: {reason}")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class AuthoringEvidence:
    """An immutable observation, stored with the derivation it authored."""

    observation_json: str

    def to_json(self) -> str:
        return self.observation_json

    @classmethod
    def from_json(cls, value: str | None) -> AuthoringEvidence:
        if not isinstance(value, str) or not value:
            raise AuthoringUnavailable("reuse saved derivation", "actual authoring input and complete reply are not recorded")
        evidence = cls(value)
        evidence.require_records(evidence.records, operation="read recorded authoring facts")
        return evidence

    @property
    def records(self) -> tuple[str, ...]:
        try:
            value = json.loads(self.observation_json)
            return tuple(source["record"] for source in value["sources"])
        except (TypeError, ValueError, KeyError) as error:
            raise AuthoringUnavailable("read recorded authoring facts", "source ownership is unavailable") from error

    def require_text(self, text: str, *, operation: str = "copy selected derivation") -> None:
        """Check a storage copy after invocation identity selected its author.

        This does not select a response or infer its author from matching text.
        """
        owner = None
        try:
            value = json.loads(self.observation_json)
            owner = value["sources"][0]["record"]
            authored = (value["authored_text"] if value.get("version") == 2 else
                        value["native_reply"]["choices"][0]["message"]["content"])
            if not isinstance(text, str) or text != authored:
                raise ValueError("the selected derivation text differs from its observed authoring reply")
        except (TypeError, ValueError, KeyError, IndexError) as error:
            raise AuthoringUnavailable(operation, str(error), owner) from error

    def require_original(self, record: str, raw: str, *, operation: str = "copy identified original") -> None:
        """Check values of an already identified canonical source record.

        JSON encoding keeps booleans, integers and floats distinct; ordinary
        Python equality would equate values such as True and 1.
        """
        try:
            value = json.loads(self.observation_json)
            sources = [source for source in value["sources"] if source["record"] == record]
            if not sources:
                raise ValueError("the identified original has no authoring snapshot")
            original = _json(json.loads(raw))
            for source in sources:
                if _json(json.loads(source["original_json"])) != original:
                    raise ValueError("the authoring snapshot differs from the identified canonical original")
        except (TypeError, ValueError, KeyError) as error:
            raise AuthoringUnavailable(operation, str(error), record) from error

    def require_records(self, records: Sequence[str], *, operation: str = "admit derivation") -> None:
        """Check facts for this exact ordered stretch, without current-route lookup."""
        owner = next(iter(records), None)
        try:
            value = json.loads(self.observation_json)
            sources = value["sources"]
            if not sources or tuple(source["record"] for source in sources) != tuple(records):
                raise ValueError("authoring source sequence differs from selected records")
            if value.get("version") == 2:
                _require_native_certificate(value)
                return
            reader = value["reader_messages"]
            request = value["invocation"]
            messages = request["messages"]
            session_model = value["session_model"]
            if not isinstance(session_model, str) or not session_model:
                raise ValueError("the author's expected session model is unavailable")
            if request.get("model") != session_model:
                raise ValueError("the actual authoring request differs from its expected session model")
            if request["stream"] is not False:
                raise ValueError("the observed native completion operation was not nonstream")
            if value["client_type"] != "openai.OpenAI":
                raise ValueError("the recorded client has no established native completion contract")
            if value["response_type"] != "openai.types.chat.chat_completion.ChatCompletion":
                raise ValueError("the recorded reply has no established native completion contract")
            if value["native_operation"] != "openai.resources.chat.completions.completions.Completions.create":
                raise ValueError("the recorded actual create operation is unavailable")
            if not isinstance(reader, list) or len(reader) != len(sources) + 2:
                raise ValueError("the original reader rows are unavailable")
            if not isinstance(messages, list) or len(messages) != len(reader):
                raise ValueError("the actual invocation does not retain the original reader rows")
            removed = value["removed_fields"]
            if not isinstance(removed, list) or not all(isinstance(field, str) for field in removed):
                raise ValueError("the observed message conversion is unavailable")
            expected = [{key: field for key, field in row.items()
                         if key not in removed and not key.startswith("_")}
                        for row in reader]
            if messages != expected:
                raise ValueError("the actual invocation differs from the established reader conversion")
            for position, source in enumerate(sources, 1):
                owner = source["record"]
                if source["request_index"] != position:
                    raise ValueError("source row ownership differs from the actual invocation")
                original = json.loads(source["original_json"])
                prepared = json.loads(source["prepared_json"])
                if not isinstance(original, dict) or prepared != reader[position]:
                    raise ValueError("the original reader snapshot is unavailable")
                if original.get("role") != prepared.get("role"):
                    raise ValueError("the original role was changed")
                if not isinstance(source["readable_paths"], list) or not isinstance(source["withheld"], list):
                    raise ValueError("readable and withheld source fields are unavailable")
                covered = {path[0] for path in source["readable_paths"] if path}
                for omission in source["withheld"]:
                    if not omission["source_path"] or not omission["kind"] or not omission["reason"]:
                        raise ValueError("a source omission has no field-specific reason")
                    covered.add(omission["source_path"][0])
                if set(original) - covered:
                    raise ValueError(f"original fields have no readable carriage or named withholding: {sorted(set(original) - covered)}")
                content = messages[position].get("content")
                if isinstance(content, list) and any(isinstance(part, dict) and part.get("type") == "image_url" for part in content):
                    capability = value.get("image_facts")
                    if not isinstance(capability, dict) or capability.get("reads_images") is not True:
                        raise ValueError("the actual image-reading capability is unavailable")
                    if capability.get("model_id") != request.get("model") or not capability.get("source") or not capability.get("basis"):
                        raise ValueError("the image-reading fact is not attached to the actual model")
            reply = value["native_reply"]
            if reply.get("model") != session_model:
                raise ValueError(f"native reply model {reply.get('model')!r} differs from expected session model {session_model!r}; alias identity is not established")
            if reply["object"] != "chat.completion":
                raise ValueError("the native reply is not a complete Chat response")
            choice = reply["choices"][0]
            content = choice["message"]["content"]
            if choice["finish_reason"] != "stop" or not isinstance(content, str) or not content.strip():
                raise ValueError("the native reply has no genuine successful ending and text")
        except (TypeError, ValueError, KeyError, IndexError) as error:
            raise AuthoringUnavailable(operation, str(error), owner) from error


@dataclass(frozen=True)
class AuthoredSummary:
    text: str
    finish_reason: str
    authoring: AuthoringEvidence


def observe_reply(reader: ReaderInput, request: dict[str, Any], response: Any,
                  client: Any, *, removed_fields: Sequence[str],
                  session_model: str,
                  image_facts: dict[str, Any] | None = None) -> AuthoredSummary:
    """Bind a direct native reply to the owned input of this physical call."""
    from openai import OpenAI
    from openai.types.chat import ChatCompletion

    owner = reader.sources[0].record if reader.sources else None
    if type(client) is not OpenAI:
        raise AuthoringUnavailable("accept summariser reply", "the selected client's native ending is not exposed by an established contract", owner)
    if type(response) is not ChatCompletion:
        raise AuthoringUnavailable("accept summariser reply", "the actual returned object has no established native completion ending", owner)
    if getattr(response, "model", None) != session_model:
        raise AuthoringUnavailable("accept summariser reply on session model",
            f"native reply model {getattr(response, 'model', None)!r} differs from expected session model {session_model!r}; alias identity is not established", owner)
    if request.get("stream", False) is not False:
        raise AuthoringUnavailable("accept summariser reply", "the call did not return an observed native nonstream reply", owner)
    observation = {
        "sources": [asdict(source) for source in reader.sources],
        "reader_messages": reader.messages,
        "session_model": session_model,
        # Keep the physical callback input and effective body overrides. The host
        # may move plain-JSON fields into extra_body before the SDK call; this is
        # not a claim about its final keyword layout or network receipt. Headers
        # and timeout objects are not source material and may carry credentials.
        "invocation": {**{key: value for key, value in request.items()
                           if key not in {"extra_headers", "timeout"}}, "stream": False},
        "removed_fields": sorted(set(removed_fields)),
        "client_type": f"{type(client).__module__}.{type(client).__name__}",
        "response_type": f"{type(response).__module__}.{type(response).__name__}",
        "native_operation": "openai.resources.chat.completions.completions.Completions.create",
        "native_reply": response.model_dump(mode="json"),
        "image_facts": image_facts,
    }
    try:
        authoring = AuthoringEvidence(_json(observation))
    except (TypeError, ValueError) as error:
        raise AuthoringUnavailable("record summariser authoring", "the actual invocation or reply cannot be preserved as JSON", owner) from error
    authoring.require_records([source.record for source in reader.sources], operation="accept summariser reply")
    choice = response.choices[0]
    return AuthoredSummary(choice.message.content, choice.finish_reason, authoring)


def _validate_invocation(reader: ReaderInput, request: dict[str, Any], *,
                         reads_images: bool | None) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Check the actual callback input by explicit row ownership, before calling."""
    owner = reader.sources[0].record if reader.sources else None
    if not reader.sources or len(reader.messages) != len(reader.sources) + 2:
        raise AuthoringUnavailable("own original reader for actual invocation", "the ordered original-source rows are unavailable", owner)
    for index, source in enumerate(reader.sources, 1):
        original = json.loads(source.original_json)
        if source.request_index != index or json.loads(source.prepared_json) != reader.messages[index]:
            raise AuthoringUnavailable("own original reader for actual invocation", "the immutable source snapshot does not own this request row", source.record)
        covered = {path[0] for path in source.readable_paths if path}
        for omitted in source.withheld:
            if not omitted.source_path or not omitted.kind or not omitted.reason:
                raise AuthoringUnavailable("own original reader for actual invocation", "a source omission has no field-specific reason", source.record)
            covered.add(omitted.source_path[0])
        if set(original) - covered:
            raise AuthoringUnavailable("own original reader for actual invocation", f"original fields have no readable carriage or named withholding: {sorted(set(original) - covered)}", source.record)
    effective = dict(request)
    extra = request.get("extra_body")
    if isinstance(extra, dict):
        for field in ("messages", "model", "stream"):
            if field in extra:
                effective[field] = extra[field]
    messages = effective.get("messages")
    if not isinstance(messages, list) or len(messages) != len(reader.messages):
        raise AuthoringUnavailable("convert original reader for actual invocation", "the actual request does not retain every owned reader row", owner)
    removed: set[str] = set()
    allowed = {"role", "content", "tool_calls", "tool_call_id", "name"}
    roles = {"system", "developer", "user", "assistant", "tool", "function"}
    for index, (prepared, actual) in enumerate(zip(reader.messages, messages)):
        source = reader.sources[index - 1] if 1 <= index <= len(reader.sources) else None
        record = source.record if source else owner
        if not isinstance(actual, dict):
            raise AuthoringUnavailable("convert original reader for actual invocation", f"request row {index} is not a message object", record)
        omitted = set(prepared) - set(actual)
        if omitted & allowed:
            raise AuthoringUnavailable("convert original reader for actual invocation", f"native message fields were omitted: {sorted(omitted & allowed)}", record)
        if {key: value for key, value in prepared.items() if key not in omitted} != actual:
            raise AuthoringUnavailable("convert original reader for actual invocation", f"owned request row {index} was changed by request assembly", record)
        removed.update(omitted)
        unknown = set(actual) - allowed
        if unknown:
            raise AuthoringUnavailable("convert original reader for actual invocation", f"native carriage of message fields is not established: {sorted(unknown)}", record)
        role = actual.get("role")
        if role not in roles:
            raise AuthoringUnavailable("convert original reader for actual invocation", f"native carriage of original role {role!r} is not established", record)
        content = actual.get("content")
        if not isinstance(content, (str, list)) and not (role in {"assistant", "function"} and content is None):
            raise AuthoringUnavailable("convert original reader for actual invocation", "native carriage of the original content value is not established", record)
        if isinstance(content, list):
            for part_index, part in enumerate(content):
                if not isinstance(part, dict):
                    raise AuthoringUnavailable("convert original reader for actual invocation", f"content[{part_index}] is not an established native part", record)
                kind = part.get("type")
                if kind == "text" and isinstance(part.get("text"), str) and set(part) <= {"type", "text"}:
                    continue
                if kind == "refusal" and role == "assistant" and isinstance(part.get("refusal"), str) and set(part) <= {"type", "refusal"}:
                    continue
                if kind == "image_url" and role == "user":
                    image = part.get("image_url")
                    if (not isinstance(image, dict) or not isinstance(image.get("url"), str)
                            or set(part) - {"type", "image_url"} or set(image) - {"url", "detail"}
                            or image.get("detail", "auto") not in {"auto", "low", "high"}):
                        raise AuthoringUnavailable("convert original reader for actual invocation", f"content[{part_index}].image_url has no established native value", record)
                    if reads_images is not True:
                        raise AuthoringUnavailable("read original image in actual invocation", f"content[{part_index}]: the actual selected model's image-reading capability is unavailable or false", record)
                    continue
                raise AuthoringUnavailable("convert original reader for actual invocation", f"content[{part_index}]: native readable carriage of {kind!r} is not established", record)
    if effective.get("stream", False) is not False:
        raise AuthoringUnavailable("observe complete summariser reply", "the actual request selects streaming but the host does not expose an observed ending frame", owner)
    return effective, tuple(sorted(removed))


def native_client_problem(client: Any) -> str:
    """Native completion contracts established by source, never by provider labels."""
    try:
        from agent.native_invocation import get_native_owner_binding
        if get_native_owner_binding(client) is not None:
            return ""
    except ImportError:
        pass
    return "the actual selected owner has no native input and terminal observation binding"


def _plain(value: Any) -> Any:
    """Preserve actual native data, including binary media, without opaque owners."""
    import base64

    if isinstance(value, bytes):
        return {"$native_bytes_base64": base64.b64encode(value).decode("ascii")}
    if isinstance(value, dict):
        return {key: _plain(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(child) for child in value]
    if hasattr(value, "model_dump"):
        return _plain(value.model_dump(mode="json"))
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise AuthoringUnavailable("retain native invocation", "an original native value cannot be represented")


def _at(value: Any, path: Sequence[str | int]) -> Any:
    for key in path:
        if isinstance(key, int):
            if not isinstance(value, list) or key < 0:
                raise ValueError("native array location is unavailable")
        elif not isinstance(key, str) or not isinstance(value, dict):
            raise ValueError("native object location is unavailable")
        value = value[key]
    return value


def _nodes(value: Any, path: tuple = ()):
    yield path, value
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _nodes(child, (*path, key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _nodes(child, (*path, index))


def _require_sources(sources: list[dict], messages: list[dict]) -> None:
    """Check every typed node, not merely the presence of its top-level field."""
    if not sources or len(messages) != len(sources) + 2:
        raise ValueError("the ordered original-source rows are unavailable")
    for index, source in enumerate(sources, 1):
        original = json.loads(source["original_json"])
        prepared = json.loads(source["prepared_json"])
        if not isinstance(original, dict) or not isinstance(prepared, dict):
            raise ValueError("the immutable original and reader row must be objects")
        if source["request_index"] != index or _json(prepared) != _json(messages[index]):
            raise ValueError("the immutable original snapshot does not own its request row")
        if original.get("role") != prepared.get("role"):
            raise ValueError("the owned original role was changed")
        omissions = {tuple(item["source_path"]) for item in source["withheld"]
                     if item["source_path"] and item["kind"] and item["reason"]}
        if len(omissions) != len(source["withheld"]):
            raise ValueError("a withholding has no exact path and reason")
        from .summariser_input import ReaderUnavailable, _NATIVE_CARRIERS, _reader_carrier

        actual_omissions = []
        try:
            for key, value in original.items():
                if key in _NATIVE_CARRIERS:
                    _reader_carrier(value, key, source["record"], actual_omissions)
        except ReaderUnavailable as error:
            raise ValueError("the original opaque-field projection is unavailable") from error
        if _json([asdict(item) for item in actual_omissions]) != _json(source["withheld"]):
            raise ValueError("the named withholdings differ from the actual original's opaque fields")
        witnesses = {tuple(item["source_path"]): item for item in source["witnesses"]}
        if len(witnesses) != len(source["witnesses"]):
            raise ValueError("original fields have duplicate reader witnesses")
        for path, node in _nodes(original):
            if any(path[:len(omission)] == omission for omission in omissions):
                continue
            witness = witnesses.get(path)
            if witness is None:
                raise ValueError(f"original field {path!r} has no typed reader witness")
            if isinstance(node, dict):
                expected = list(node)
                kind, suffix = "object", "; retained object keys]\n" + json.dumps(expected, ensure_ascii=False)
            elif isinstance(node, list):
                expected = len(node)
                kind, suffix = "array", f"; array length {expected}]"
            elif isinstance(node, str):
                expected = node
                kind, suffix = "string", f"; string length {len(node)}]\n" + node
            else:
                expected = node
                kind, suffix = "scalar", "; JSON value]\n" + json.dumps(node, ensure_ascii=False, allow_nan=False)
            if _json(json.loads(witness["value_json"])) != _json(expected):
                raise ValueError(f"reader witness changed original field {path!r}")
            actual = _at(prepared, witness["prepared_path"])
            if witness["kind"] == "image":
                part_path = path[:next(position + 1 for position, key in enumerate(path) if isinstance(key, int))]
                original_part = _at(original, part_path)
                if (kind != "string" or _image_identity(actual) is None
                        or _image_identity(actual) != _image_identity(original_part)):
                    raise ValueError("the original image has no identified native reader part")
                continue
            if witness["kind"] != kind:
                raise ValueError("the original value type was changed")
            expected_text = ("[Recorded original object keys]\n" + json.dumps(expected, ensure_ascii=False)
                             if not path else
                             f"[Recorded original field {json.dumps(path, ensure_ascii=False)}" + suffix)
            if actual != expected_text:
                raise ValueError(f"reader text does not expose original field {path!r}")


def _image_identity(part: Any) -> tuple | None:
    """Compare the actual typed carrier's reference or decoded media bytes."""
    import base64

    def binary(data: Any, mime: Any) -> tuple | None:
        if not isinstance(data, str) or not isinstance(mime, str):
            return None
        try:
            decoded = base64.b64decode(data)
        except ValueError:
            return None
        return ("bytes", base64.b64encode(decoded).decode("ascii"), mime.split(";", 1)[0].lower())

    def reference(url: Any) -> tuple | None:
        if not isinstance(url, str) or not url:
            return None
        if url.startswith("data:") and ";base64," in url:
            header, data = url.split(";base64,", 1)
            return binary(data, header[5:])
        return ("url", url)

    if not isinstance(part, dict):
        return None
    kind = part.get("type")
    if kind in {"image_url", "input_image"}:
        image = part.get("image_url")
        return reference(image.get("url") if isinstance(image, dict) else image)
    source = part.get("source")
    if kind == "image" and isinstance(source, dict):
        if source.get("type") == "url":
            return reference(source.get("url"))
        if source.get("type") == "base64":
            return binary(source.get("data"), source.get("media_type"))
        return None
    inline = part.get("inlineData", part.get("inline_data"))
    if isinstance(inline, dict):
        return binary(inline.get("data"), inline.get("mimeType", inline.get("mime_type")))
    if isinstance(part.get("image"), dict):
        return _image_identity(part["image"])
    if isinstance(source, dict) and isinstance(source.get("bytes"), dict):
        return binary(source["bytes"].get("$native_bytes_base64"), "image/" + str(part.get("format", "")))
    return None


def _native_image_at(body: dict, path: Sequence, api_mode: str) -> tuple | None:
    """The value must occupy an actual native media block, not a text reference."""
    roots = {"chat_completions": ("messages", "content", "image_url"),
             "responses": ("input", "content", "input_image"),
             "codex_responses": ("input", "content", "input_image"),
             "anthropic_messages": ("messages", "content", "image"),
             "bedrock_converse": ("messages", "content", None),
             "gemini": ("contents", "parts", None)}
    grammar = roots.get(api_mode)
    if grammar is None or len(path) < 4 or path[0] != grammar[0] or path[2] != grammar[1]:
        return None
    if not isinstance(path[1], int) or not isinstance(path[3], int):
        return None
    part = _at(body, path[:4])
    if not isinstance(part, dict):
        return None
    if grammar[2] is not None and part.get("type") != grammar[2]:
        return None
    if api_mode == "bedrock_converse" and not isinstance(part.get("image"), dict):
        return None
    if api_mode == "gemini" and not isinstance(part.get("inlineData", part.get("inline_data")), dict):
        return None
    return _image_identity(part)


def _native_pair_at(body: dict, path: Sequence, api_mode: str, kind: str) -> tuple | None:
    """Read pairing tokens only from the actual owner's call/result grammar."""
    path = tuple(path)
    if len(path) < 3 or not isinstance(path[1], int):
        return None
    row = _at(body, path[:2])
    token = _at(body, path)
    if not isinstance(token, str) or not token:
        return None
    if api_mode == "chat_completions" and path[0] == "messages":
        if kind == "result" and row.get("role") == "tool" and path[2:] == ("tool_call_id",):
            return ("id", token)
        if (kind == "call" and row.get("role") == "assistant" and len(path) == 5
                and path[2] == "tool_calls" and isinstance(path[3], int) and path[4] == "id"):
            return ("id", token)
    if api_mode in {"responses", "codex_responses"} and path[0] == "input":
        expected = "function_call" if kind == "call" else "function_call_output"
        if row.get("type") == expected and path[2:] == ("call_id",):
            return ("id", token)
    if len(path) < 5 or not isinstance(path[3], int):
        return None
    part = _at(body, path[:4])
    if api_mode == "anthropic_messages" and path[0] == "messages" and path[2] == "content":
        expected_type, expected_key = ("tool_use", "id") if kind == "call" else ("tool_result", "tool_use_id")
        if (row.get("role") == ("assistant" if kind == "call" else "user")
                and part.get("type") == expected_type and path[4:] == (expected_key,)):
            return ("id", token)
    if api_mode == "bedrock_converse" and path[0] == "messages" and path[2] == "content":
        expected_key = "toolUse" if kind == "call" else "toolResult"
        if row.get("role") == ("assistant" if kind == "call" else "user") and path[4:] == (expected_key, "toolUseId"):
            return ("id", token)
    if api_mode == "gemini" and path[0] == "contents" and path[2] == "parts":
        expected_key = "functionCall" if kind == "call" else "functionResponse"
        if row.get("role") == ("model" if kind == "call" else "user") and path[4:] in {(expected_key, "id"), (expected_key, "name")}:
            return (path[-1], token)
    return None


def _require_tool_pairs(messages: list[dict], prepared: dict) -> None:
    """Check original-to-native pair associations while IDs remain reader-visible."""
    from collections import defaultdict, deque

    pending = defaultdict(deque)
    for index, message in enumerate(messages):
        fields = []
        for call_index, call in enumerate(message.get("tool_calls") or []):
            if call.get("id"):
                fields.append((("tool_calls", call_index, "id"), "call", call["id"]))
        if message.get("role") == "tool" and message.get("tool_call_id"):
            fields.append((("tool_call_id",), "result", message["tool_call_id"]))
        for source_path, kind, original_id in fields:
            tokens = []
            for occurrence in prepared["input_trace"]["occurrences"]:
                if (occurrence["request_index"] != index or tuple(occurrence["source_path"]) != source_path
                        or occurrence["carrier_kind"] != "tool_id"):
                    continue
                relation = occurrence.get("tool_pair_relation")
                if not relation or tuple(relation) != (kind, original_id):
                    raise ValueError("the native tool occurrence has no original pair association")
                if occurrence["logical_role"] != message.get("role"):
                    raise ValueError("the native tool occurrence changed its original role")
                for location in occurrence["native_locations"]:
                    token = _native_pair_at(prepared["effective_body"], location["path"], prepared["api_mode"], kind)
                    if token is not None and kind == "call":
                        path, body, mode = location["path"], prepared["effective_body"], prepared["api_mode"]
                        if mode == "chat_completions":
                            native_name = _at(body, path[:4])["function"]["name"]
                        elif mode in {"responses", "codex_responses"}:
                            native_name = _at(body, path[:2])["name"]
                        elif mode == "anthropic_messages":
                            native_name = _at(body, path[:4])["name"]
                        elif mode == "bedrock_converse":
                            native_name = _at(body, path[:4])["toolUse"]["name"]
                        else:
                            native_name = _at(body, path[:4])["functionCall"]["name"]
                        original_name = message["tool_calls"][source_path[1]].get("function", {}).get("name")
                        if native_name != original_name:
                            raise ValueError("the actual native call changed the original tool name")
                    if token is not None and token not in tokens:
                        tokens.append(token)
            if not tokens:
                raise ValueError("the original tool association has no actual native call/result token")
            if kind == "call":
                pending[original_id].append(tokens)
            elif pending[original_id]:
                call_tokens = pending[original_id].popleft()
                if not any(token in call_tokens for token in tokens):
                    raise ValueError("native tool-ID translation separated an original call and result")


def _require_carriage(messages: list[dict], prepared: dict, *, reads_images: bool | None) -> None:
    """Check causal occurrences against final actual values after all transforms."""
    body, trace = prepared["effective_body"], prepared["input_trace"]
    _require_tool_pairs(messages, prepared)
    occurrences = trace["occurrences"]
    previous: dict[tuple, tuple] = {}
    previous_rows: dict[str, tuple[int, int]] = {}
    previous_parts: dict[tuple, int] = {}
    for index, message in enumerate(messages):
        content = message.get("content")
        values = [(("content",), content, "text")] if isinstance(content, str) else []
        if isinstance(content, list):
            for part_index, part in enumerate(content):
                if part.get("type") in {"text", "input_text", "output_text"}:
                    values.append((("content", part_index, "text"), part["text"], "text"))
                else:
                    values.append((("content", part_index), part, "image"))
        for path, expected, kind in values:
            matches = [item for item in occurrences if item["request_index"] == index and
                       (tuple(item["source_path"]) == path or
                        (kind == "image" and tuple(item["source_path"])[:len(path)] == path) or
                        (kind == "text" and item["carrier_kind"] == "json"
                         and path[:len(item["source_path"])] == tuple(item["source_path"])))]
            if not matches:
                raise ValueError(f"request row {index}, field {path!r} has no native emission occurrence")
            accepted = False
            for item in matches:
                if item["logical_role"] != message.get("role"):
                    raise ValueError("the native occurrence changed the logical source role")
                locations = item["native_locations"]
                for location in locations:
                    actual = _at(body, location["path"])
                    start, end = location.get("text_start"), location.get("text_end")
                    if start is not None or end is not None:
                        if not isinstance(actual, str) or not isinstance(start, int) or not isinstance(end, int) or not 0 <= start <= end <= len(actual):
                            raise ValueError("native text span is unavailable")
                        actual = actual[start:end]
                    if kind == "text":
                        if item["carrier_kind"] == "json":
                            if not isinstance(actual, str):
                                raise ValueError("native JSON text carriage is unavailable")
                            actual = _at(json.loads(actual), path[len(item["source_path"]):])
                        if actual != expected:
                            continue
                    else:
                        if reads_images is not True:
                            raise ValueError("the selected model's image-reading capability is unavailable or false")
                        if (_image_identity(expected) is None or
                                _native_image_at(body, location["path"], prepared["api_mode"]) != _image_identity(expected)):
                            continue
                    # Ordered occurrences preserve record boundaries through role merges.
                    destination = tuple(location["path"])
                    if (len(destination) > 1 and destination[0] in {"messages", "input", "contents"}
                            and isinstance(destination[1], int)):
                        earlier_row = previous_rows.get(destination[0])
                        if earlier_row is not None and index >= earlier_row[0] and destination[1] < earlier_row[1]:
                            raise ValueError("native emission reordered original reader rows")
                        previous_rows[destination[0]] = (index, destination[1])
                        if (len(destination) > 3 and destination[2] in {"content", "parts"}
                                and isinstance(destination[3], int) and item["carrier_kind"] != "json"):
                            row_path = destination[:2]
                            earlier_part = previous_parts.get(row_path)
                            if earlier_part is not None and destination[3] < earlier_part:
                                raise ValueError("native emission reordered original reader parts")
                            previous_parts[row_path] = destination[3]
                    if start is not None and item["carrier_kind"] != "json":
                        earlier = previous.get(destination)
                        if earlier is not None and (index < earlier[0] or start < earlier[1]):
                            raise ValueError("native role merge reordered owned reader occurrences")
                        previous[destination] = (index, end)
                    accepted = True
                    break
                if accepted:
                    break
            if not accepted:
                raise ValueError(f"request row {index}, field {path!r} is absent or changed in the actual native payload")


def _native_text(api_mode: str, result: Any, events: Any, terminals: Any) -> str:
    """Use the original owner's successful terminal, never compatibility defaults."""
    raw, frames, observed = _plain(result), _plain(events), _plain(terminals)
    frames, observed = frames or [], observed or []
    if api_mode == "chat_completions" and frames:
        chunks = [frame for frame in frames if isinstance(frame, dict) and frame.get("choices")]
        endings = [frame["choices"][0].get("finish_reason") for frame in chunks
                   if frame["choices"][0].get("finish_reason") is not None]
        if not endings or endings[-1] != "stop" or any(ending != "stop" for ending in endings):
            raise ValueError("the original Chat stream has no genuine successful stop")
        text = "".join(choice.get("delta", {}).get("content", "") or ""
                       for frame in chunks for choice in frame["choices"][:1])
        if not text.strip():
            raise ValueError("the original Chat stream has no nonblank text")
        return text
    if api_mode in {"gemini", "gemini_native"} and frames:
        chunks = [frame for frame in frames if isinstance(frame, dict) and frame.get("candidates")]
        endings = [frame["candidates"][0].get("finishReason") for frame in chunks
                   if frame["candidates"][0].get("finishReason") is not None]
        if not endings or endings[-1] != "STOP" or any(ending != "STOP" for ending in endings):
            raise ValueError("the original Gemini stream has no genuine STOP")
        text = "".join(part["text"] for frame in chunks
                       for part in frame["candidates"][0].get("content", {}).get("parts", [])
                       if isinstance(part.get("text"), str) and not part.get("thought"))
        if not text.strip():
            raise ValueError("the original Gemini stream has no nonblank text")
        return text
    if api_mode in {"codex_responses", "responses"} and frames:
        endings = [frame for frame in frames if isinstance(frame, dict)
                   and frame.get("type") in {"response.completed", "response.incomplete", "response.failed"}]
        if not endings:
            endings = [frame for frame in observed if isinstance(frame, dict)
                       and frame.get("type") in {"response.completed", "response.incomplete", "response.failed"}]
        if (not endings or endings[-1].get("type") != "response.completed"
                or not isinstance(endings[-1].get("response"), dict)):
            raise ValueError("the original Responses stream has no completed response frame")
        raw = endings[-1]["response"]
    if not isinstance(raw, dict):
        raise ValueError("the original native result is unavailable")
    if api_mode == "chat_completions":
        choices = raw.get("choices", [])
        if not choices or raw.get("object") != "chat.completion" or choices[0].get("finish_reason") != "stop":
            raise ValueError("the original Chat reply has no genuine stop")
        text = choices[0].get("message", {}).get("content")
    elif api_mode in {"codex_responses", "responses"}:
        if raw.get("status") != "completed" or raw.get("error") or raw.get("incomplete_details"):
            raise ValueError("the original Responses reply did not complete")
        if frames and not any(isinstance(frame, dict) and frame.get("type") == "response.completed" for frame in frames + observed):
            raise ValueError("the original Responses stream has no completed frame")
        text = "".join(part["text"] for item in raw.get("output", []) if item.get("type") == "message"
                       for part in item.get("content", []) if part.get("type") == "output_text" and isinstance(part.get("text"), str))
    elif api_mode == "anthropic_messages":
        if raw.get("stop_reason") != "end_turn":
            raise ValueError("the original Anthropic reply has no end_turn")
        if frames and not any(isinstance(frame, dict) and frame.get("type") == "message_stop" for frame in frames + observed):
            raise ValueError("the original Anthropic stream has no message_stop")
        text = "".join(part["text"] for part in raw.get("content", []) if part.get("type") == "text" and isinstance(part.get("text"), str))
    elif api_mode in {"bedrock_converse", "converse"}:
        if raw.get("stopReason") != "end_turn":
            raise ValueError("the original Converse reply has no end_turn")
        text = "".join(part["text"] for part in raw.get("output", {}).get("message", {}).get("content", []) if isinstance(part.get("text"), str))
    elif api_mode in {"gemini_native", "gemini"}:
        candidates = raw.get("candidates", [])
        if not candidates or candidates[0].get("finishReason") != "STOP":
            raise ValueError("the original Gemini reply has no STOP")
        text = "".join(part["text"] for part in candidates[0].get("content", {}).get("parts", [])
                       if isinstance(part.get("text"), str) and not part.get("thought"))
    else:
        raise ValueError("the original native owner has no established terminal grammar")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("the original native reply has no nonblank text")
    return text


def _prepared_facts(prepared: Any) -> dict:
    return {"api_mode": prepared.api_mode, "model": prepared.model,
            "owner_identity": prepared.binding.identity,
            "effective_body": _plain(prepared.effective_body),
            "input_trace": _plain(asdict(prepared.input_trace)),
            "control_projection": [_plain(asdict(item)) for item in prepared.control_projection]}


def _require_controls(prepared: dict, effort: str) -> None:
    """Check the selected effort's actual owner-policy projection and final values."""
    projections = prepared["control_projection"]
    if not projections:
        raise ValueError("the selected effort has no observed native policy projection")
    matched = False
    for projection in projections:
        if not isinstance(projection["policy_name"], str) or not projection["policy_name"]:
            raise ValueError("the native control projection has no policy owner")
        config = projection["requested_config"]
        if not isinstance(config, dict):
            raise ValueError("the actual native policy input is unavailable")
        selected = config.get("effort")
        enabled = config.get("enabled")
        if effort == "none":
            applies = selected == "none" or enabled is False
        else:
            applies = selected == effort and enabled is not False
        values = projection["native_values"]
        for location, expected in values:
            actual = _at(prepared["effective_body"], location["path"])
            if _json(actual) != _json(expected):
                raise ValueError("final native assembly replaced a selected policy control")
        if not applies:
            continue
        basis = projection.get("omission_basis")
        if effort != "none" and isinstance(basis, str) and basis:
            raise ValueError(f"the actual native policy cannot apply selected effort {effort!r}: {basis}")
        supported = projection.get("supported_efforts")
        if supported is not None:
            for location, expected in values:
                if (location["path"] and location["path"][-1] in {"effort", "reasoning_effort"}
                        and isinstance(expected, str) and expected not in supported):
                    raise ValueError("the projected native effort is outside this owner's established supported values")
        if not values:
            if not isinstance(basis, str) or not basis:
                raise ValueError("native effort omission has no actual policy basis")
            if effort != "none":
                raise ValueError(f"the actual native policy cannot apply selected effort {effort!r}: {basis}")
        matched = True
    if not matched:
        raise ValueError("the actual owner policy input differs from the selected effort")


def _require_native_certificate(value: dict) -> None:
    _require_sources(value["sources"], value["reader_messages"])
    prepared = value["prepared_request"]
    if not isinstance(prepared["owner_identity"], str) or not prepared["owner_identity"]:
        raise ValueError("the observed native owner identity is unavailable")
    if not value["session_model"] or prepared["model"] != value["session_model"]:
        raise ValueError("the actual authoring request differs from its selected model")
    _require_controls(prepared, value["selected_effort"])
    capability = value.get("image_facts") or {}
    if capability and capability.get("model_id") != prepared["model"]:
        raise ValueError("the image capability belongs to another model")
    if capability.get("reads_images") is True and (not capability.get("source") or not capability.get("basis")):
        raise ValueError("the image-reading capability has no actual model fact")
    _require_carriage(value["reader_messages"], prepared, reads_images=capability.get("reads_images"))
    text = _native_text(prepared["api_mode"], value["native_result"], value["native_events"], value["terminal_observations"])
    if text != value["authored_text"]:
        raise ValueError("the selected derivation differs from its original native author's text")


@dataclass(frozen=True)
class SelectedInvocation:
    text: str
    native_result: Any


def _effective_body(request: dict[str, Any]) -> dict[str, Any]:
    """Body controls after the host's plain-JSON extra_body carriage."""
    body = {key: value for key, value in request.items()
            if key not in {"extra_body", "extra_headers", "timeout"} and not key.startswith("_native_")}
    extra = request.get("extra_body")
    if extra is not None and not isinstance(extra, dict):
        raise AuthoringUnavailable("dispatch selected invocation", "the effective request body is unavailable")
    body.update(extra or {})
    body.setdefault("stream", False)
    return body


def invoke_selected(messages: list[dict], *, route: Any, effort: str,
                    timeout: float, max_tokens: int | None, route_info: Any,
                    temperature: float | None = 0.3,
                    validate: Any = None, observe: Any = None,
                    owner: str | None = None) -> Any:
    """One retained host plan, native gate and physical result for both purposes."""
    from agent import auxiliary_client as host
    from agent.native_invocation import native_invocation_scope

    required = ("_plan_aux_call", "_relay_sync_completion", "_relay_aux_call_scope",
                "scoped_runtime_main", "_create_with_progress_once", "_provider_requires_stream",
                "_validate_llm_response", "_acquire_sync_aux_semaphore")
    for name in required:
        if not callable(getattr(host, name, None)):
            raise AuthoringUnavailable("dispatch selected invocation", f"host capability agent.auxiliary_client.{name} is unavailable", owner)
    options = route.plan_kwargs()
    expected_messages = json.loads(_json(messages))
    planned_messages = json.loads(_json(messages))
    semaphore = host._acquire_sync_aux_semaphore(None)
    if semaphore is not None:
        semaphore.acquire()
    req: Any = None
    contexts: dict[str, Any] = {}

    def gate(prepared: Any) -> None:
        if req is None:
            raise AuthoringUnavailable("dispatch native owner", "a native operation occurred before selected planning completed", owner)
        route.require_selected_client(req.client)
        if prepared.binding is not route.target_binding or prepared.model != route.target_model:
            raise AuthoringUnavailable("dispatch native owner", "the actual native owner or model differs from the selected one", owner)
        if not prepared.binding.matches_dispatch(prepared.resource_ref, prepared.create_ref):
            raise AuthoringUnavailable("dispatch native owner", "the actual callable is outside the retained public native owner", owner)
        try:
            _require_controls(_prepared_facts(prepared), effort)
            context = validate(req, prepared) if validate is not None else None
            if validate is None:
                _require_carriage(expected_messages, _prepared_facts(prepared), reads_images=None)
        except (ValueError, KeyError, IndexError, TypeError) as error:
            raise AuthoringUnavailable("carry complete input into native invocation", str(error), owner) from error
        contexts[prepared.attempt_id] = context

    try:
        with host._relay_aux_call_scope((), {"task": None}), host.scoped_runtime_main(options["main_runtime"]), native_invocation_scope(gate) as capture:
            req, _retry, _candidate = host._plan_aux_call(
                None, async_mode=False, **options, messages=planned_messages,
                temperature=temperature, max_tokens=max_tokens or None, tools=None,
                timeout=timeout, extra_body=None,
                reasoning_config={"enabled": effort != "none", "effort": effort},
                extra_headers=None, route_info=route_info,
            )
            route.require_selected_client(req.client)
            if req.final_model != route.target_model:
                raise AuthoringUnavailable("dispatch selected model", "the host replaced the retained selected model", owner)
            provider = str(host._fallback_provider_from_label(req.request_provider) or "").strip().lower()
            if provider != route.target_provider:
                raise AuthoringUnavailable("dispatch selected provider", "the host replaced the retained selected provider", owner)
            captured: list[tuple[Any, Any]] = []
            expected_body = _effective_body(req.kwargs)

            def create(actual_request: dict[str, Any]) -> Any:
                route.require_selected_client(req.client)
                body = _effective_body(actual_request)
                if body.get("model") != route.target_model:
                    raise AuthoringUnavailable("dispatch selected invocation", "the callback replaced the selected model", owner)
                if ({key: value for key, value in body.items() if key not in {"messages", "stream"}}
                        != {key: value for key, value in expected_body.items() if key not in {"messages", "stream"}}):
                    raise AuthoringUnavailable("dispatch selected invocation", "the callback changed the selected effort or purpose controls", owner)
                if (actual_request.get("extra_headers") != req.kwargs.get("extra_headers")
                        or actual_request.get("timeout") != req.kwargs.get("timeout")):
                    raise AuthoringUnavailable("dispatch selected invocation", "the callback changed selected header or timeout controls", owner)

                force_stream = host._provider_requires_stream(req.request_provider, req.base_info or req.resolved_base_url)
                response = host._create_with_progress_once(route.target_call_owner, actual_request, None, force_stream=force_stream)
                native = capture.take_result(response)
                if native is None or native.prepared_request.attempt_id not in contexts:
                    raise AuthoringUnavailable("select physical native result", "the returned callback object has no matching successful gated native attempt", owner)
                try:
                    text = _native_text(native.prepared_request.api_mode, native.original_result,
                                        native.original_events, native.terminal_observations)
                except (ValueError, KeyError, IndexError, TypeError) as error:
                    raise AuthoringUnavailable("accept original native ending", str(error), owner) from error
                result = (observe(req, actual_request, native, contexts[native.prepared_request.attempt_id], text)
                          if observe is not None else SelectedInvocation(text, native))
                captured.append((response, result))
                return response
            response = host._relay_sync_completion(
                route.target_call_owner, req.kwargs, provider=req.request_provider,
                api_mode=req.resolved_api_mode, create=create,
            )
            for raw, result in reversed(captured):
                if response is raw:
                    validated = host._validate_llm_response(
                        response, None, provider=req.request_provider, base_url=req.base_info)
                    if validated is not raw:
                        raise AuthoringUnavailable("select native reply", "host validation replaced the physically selected return", owner)
                    return result
            raise AuthoringUnavailable("select native reply", "host relay returned no captured physical native invocation", owner)
    finally:
        if semaphore is not None:
            semaphore.release()


def invoke_reader(reader: ReaderInput, *, route: Any, effort: str,
                  timeout: float, max_tokens: int | None, route_info: Any,
                  image_capability: Any = None) -> AuthoredSummary:
    """Summary ownership and versioned evidence over the one shared native call."""
    owner = reader.sources[0].record if reader.sources else None
    sources = [asdict(source) for source in reader.sources]
    reader_messages = json.loads(_json(reader.messages))
    try:
        _require_sources(sources, reader_messages)
    except (ValueError, KeyError, IndexError, TypeError) as error:
        raise AuthoringUnavailable("own complete original reader", str(error), owner) from error

    def validate(req: Any, prepared: Any) -> dict | None:
        capability = image_capability(req.request_provider, route.target_model) if image_capability else None
        reads_images = getattr(capability, "reads_images", None)
        image_facts = None if capability is None else {
            "provider": req.request_provider, "model_id": route.target_model,
            "reads_images": reads_images, "source": capability.source,
            "basis": capability.basis, "route": capability.route,
            "vendor": capability.vendor, "model": capability.model,
            "ids": list(capability.ids),
        }
        _require_sources(sources, reader_messages)
        _require_carriage(reader_messages, _prepared_facts(prepared), reads_images=reads_images)
        return image_facts

    def observe(req: Any, actual_request: dict[str, Any], native: Any,
                image_facts: dict | None, text: str) -> AuthoredSummary:
        value = {
            "version": 2, "sources": sources, "reader_messages": reader_messages,
            "session_model": route.target_model, "selected_effort": effort,
            "prepared_request": _prepared_facts(native.prepared_request),
            "native_result": _plain(native.original_result),
            "native_events": _plain(native.original_events),
            "terminal_observations": _plain(native.terminal_observations),
            "authored_text": text, "image_facts": image_facts,
            "invocation_id": native.prepared_request.invocation_id,
            "attempt_id": native.prepared_request.attempt_id,
        }
        evidence = AuthoringEvidence(_json(value))
        evidence.require_records([source.record for source in reader.sources], operation="accept native authored summary")
        return AuthoredSummary(text, "stop", evidence)

    return invoke_selected(reader_messages, route=route, effort=effort, timeout=timeout,
                           max_tokens=max_tokens, route_info=route_info,
                           validate=validate, observe=observe, owner=owner)
