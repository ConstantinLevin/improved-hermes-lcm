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
            authored = value["native_reply"]["choices"][0]["message"]["content"]
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


def invoke_reader(reader: ReaderInput, *, main_runtime: dict[str, Any], effort: str,
                  timeout: float, max_tokens: int | None, route_info: Any,
                  session_model: str,
                  image_capability: Any = None) -> AuthoredSummary:
    """Prepare once; retain that client through the host's physical executor."""
    from agent import auxiliary_client as host
    from openai import OpenAI
    from openai.resources.chat.completions import Completions

    owner = reader.sources[0].record if reader.sources else None
    if not isinstance(session_model, str) or not session_model:
        raise AuthoringUnavailable("dispatch summariser on session model", "the expected session model is unavailable", owner)
    required = ("_plan_aux_call", "_relay_sync_completion", "_relay_aux_call_scope",
                "scoped_runtime_main", "_create_with_progress_once", "_provider_requires_stream",
                "_aux_progress_active", "_validate_llm_response", "_acquire_sync_aux_semaphore")
    for name in required:
        if not callable(getattr(host, name, None)):
            raise AuthoringUnavailable("dispatch original reader", f"host capability agent.auxiliary_client.{name} is unavailable", owner)
    semaphore = host._acquire_sync_aux_semaphore(None)
    if semaphore is not None:
        semaphore.acquire()
    try:
        with host._relay_aux_call_scope((), {"task": None}), host.scoped_runtime_main(main_runtime):
            req, _retry, _candidate = host._plan_aux_call(
                None, async_mode=False, provider=None, model=None, base_url=None,
                api_key=None, main_runtime=main_runtime, messages=reader.messages,
                temperature=0.3, max_tokens=max_tokens or None, tools=None,
                timeout=timeout, extra_body=None,
                reasoning_config={"enabled": effort != "none", "effort": effort},
                extra_headers=None, api_mode=None, route_info=route_info,
            )
            if req.final_model != session_model:
                raise AuthoringUnavailable("dispatch summariser on session model",
                    f"selected host model {req.final_model!r} differs from expected session model {session_model!r}; alias identity is not established", owner)
            captured: list[tuple[Any, AuthoredSummary]] = []

            def create(actual_request: dict[str, Any]) -> Any:
                # This callback sees Relay's actual physical request. The initial
                # prepared request is not used to certify a later rewritten input.
                if type(req.client) is not OpenAI:
                    client_name = f"{type(req.client).__module__}.{type(req.client).__name__}"
                    unavailable = {
                        "CodexAuxiliaryClient": "the native terminal-event observation, status and incomplete/error details are discarded before create returns",
                        "AnthropicAuxiliaryClient": "the native stop_reason is discarded and missing/unknown reasons can be normalized to stop",
                        "BedrockAuxiliaryClient": "the native stopReason is discarded and missing/unknown reasons can be normalized to stop",
                        "GeminiNativeClient": "the native finishReason is discarded and missing/unknown reasons can be normalized to stop",
                    }.get(type(req.client).__name__, "the actual create operation exposes no established original-carriage and provider-ending contract")
                    raise AuthoringUnavailable("observe original conversion and native completion ending",
                        f"{client_name}.create: {unavailable}", owner)
                resource = req.client.chat.completions
                create_method = resource.create
                if (type(resource) is not Completions
                        or getattr(create_method, "__func__", None) is not Completions.create
                        or getattr(resource, "_client", None) is not req.client):
                    raise AuthoringUnavailable("observe actual summariser operation", "the selected create callable is not an established native Chat operation on the retained client", owner)
                if host._aux_progress_active():
                    raise AuthoringUnavailable("observe complete summariser reply", "an active host progress hook selects aggregation without an observed ending witness", owner)
                if host._provider_requires_stream(req.request_provider, req.base_info or req.resolved_base_url):
                    raise AuthoringUnavailable("observe complete summariser reply", "the selected operation requires streaming; the host aggregation does not expose the observed native ending", owner)
                model = actual_request.get("model")
                extra = actual_request.get("extra_body")
                if isinstance(extra, dict) and "model" in extra:
                    model = extra["model"]
                if model != session_model:
                    raise AuthoringUnavailable("dispatch summariser on session model",
                        f"actual effective request model {model!r} differs from expected session model {session_model!r}; alias identity is not established", owner)
                capability = image_capability(req.request_provider, model) if image_capability else None
                reads_images = getattr(capability, "reads_images", None)
                image_facts = None if capability is None else {
                    "provider": req.request_provider, "model_id": model,
                    "reads_images": reads_images, "source": capability.source,
                    "basis": capability.basis, "route": capability.route,
                    "vendor": capability.vendor, "model": capability.model,
                    "ids": list(capability.ids),
                }
                effective, removed = _validate_invocation(reader, actual_request, reads_images=reads_images)
                # Keep one physical attempt inside the existing host executor.
                # The ordinary nonstream branch returns the client's raw value;
                # the outer helper's affordable-cap retry could change kwargs
                # without exposing that later request to this callback.
                response = host._create_with_progress_once(req.client, actual_request, None, force_stream=False)
                authored = observe_reply(reader, effective, response, req.client,
                                         removed_fields=removed, session_model=session_model,
                                         image_facts=image_facts)
                captured.append((response, authored))
                return response

            response = host._relay_sync_completion(
                req.client, req.kwargs, provider=req.request_provider,
                api_mode=req.resolved_api_mode, create=create,
            )
            for raw, authored in reversed(captured):
                if response is raw:
                    validated = host._validate_llm_response(
                        response, None, provider=req.request_provider, base_url=req.base_info)
                    if validated is not raw:
                        raise AuthoringUnavailable("select summariser reply", "host validation replaced the actual authoring response", owner)
                    return authored
            raise AuthoringUnavailable("select summariser reply", "host reply assembly returned a candidate with no captured authoring invocation and complete native reply", owner)
    finally:
        if semaphore is not None:
            semaphore.release()
