"""What the plugin decides about a page's images, from readable and static facts only (the
rulings on PR B after the review of ae6ab7f). Line numbers are Hermes 97bacbbce5.

**Why there is no room.** Whether the host carries a page's images in its next request
depends on that request: the turn's boundary and the assembly's arguments are locals of the
host's loop, fixed after this call runs (conversation_loop.py:1597; turn_iteration_prep.py:198-235),
and no attribute of the agent holds them. The request cannot be built from readable state, so
the plugin does not predict it. What the host does with a request that holds more images than
its ceiling or byte budget is its own visible procedure (the placeholder "[screenshot removed
to save context]"); the store keeps the image and the agent can expand again. Asked of Hermes
as A-D1.5 (#24).

What stands:

- **The route's verdict per image** (``expansion.route_image_check``): the host's converter
  on a one-image request, carried where the result the converter paired with the call holds
  an image block (one slot, by position, never by bytes). The route is read from the agent the
  host bound for this turn (``route_of``), else from the engine's last ``update_model``, and
  the page says so.
- **The host's static limits on our own output** (``static_limits``): the host's constants,
  never its state. A single result's images stay within the block ceiling and the byte budget
  (the rest go on further pages); an image whose payload alone exceeds the budget can never be
  carried, and is held with that cause (image_eviction_policy.py:26-27, 74-106; its payload as
  the host measures it, context_compressor.py:1498-1521).
- **The tool-loop guard: a read of the host's state** (``host_halt``). The host binds the
  running agent for every turn (``bind_subagent_parent``, turn_facade.py:142);
  ``get_active_subagent_parent`` (subagent_lifecycle.py:185-188) returns it here (tool workers
  copy the context, tools/daemon_pool.py:26-36). The host appends its notice to this result,
  and raises on an image page, iff ``agent._tool_guardrails.halt_decision`` stands with
  ``identical_call_streak_halt`` or ``identical_cycle_halt`` (run_agent.py:1303-1307;
  tool_guardrails.py:340-341). Read when this call runs, it is what the host reads when it
  appends: ``lcm_expand`` is never parallel-safe (tool_dispatch_helpers.py:33-47, 193-195),
  every earlier call is observed at its commit (tool_executor.py:1075-1078), and this call's
  own observation adds neither a halt nor a warning to a dict (run_agent.py:1288;
  tool_guardrails.py:373-431, 452-473; display.py:1001-1009). Where the agent, its engine or
  the attribute cannot be read, nothing is guessed: the host's own error stands and the store
  keeps the page.
- **Statements about where a value came from**, bound to the value: wherever it decides the page
  (``ImageRoom``: the route read from the engine; the guard not readable; the host's retirement)."""

from __future__ import annotations

from typing import Any, Optional

NOTE = ("the host retires images with its own placeholder when a request holds more than its ceiling or byte budget "
        "across uploads and earlier results, or when later results of this turn add images; the store keeps every "
        "image on this page, and expanding again returns it")
ROUTE_FROM_ENGINE = "the session's route was read from the engine's last update_model"
GUARD_CAUSE = "the host's tool-loop guard has halted this turn and cannot append its notice to an image page"
GUARD_CODES = ("identical_call_streak_halt", "identical_cycle_halt")
GUARD_UNREAD = ("the host's tool-loop guard could not be read ({why}); if it has halted this turn, the host appends its "
                "notice to this image page and the model receives the host's own error instead; the store keeps every "
                "image on this page")


class RoomUnavailable(Exception):
    """A host function or constant cannot be read."""


def _host(module: str, name: str) -> Any:
    try:
        return getattr(__import__(module, fromlist=[name]), name)
    except Exception as exc:
        raise RoomUnavailable(f"the host's {module}.{name} cannot be read ({type(exc).__name__}: {exc})") from None


def _list(value: Any) -> list:
    return value if isinstance(value, list) else []


# --- The agent the host bound for this turn ------------------------------------------------------

def bound_agent(engine: Any) -> tuple[Optional[Any], str]:
    """The agent the host bound for the running turn, where its engine is ``engine``, and "";
    else None and why not, each cause asked on its own: the host's binding cannot be read, no
    turn is bound, or the turn's agent runs another engine."""
    try:
        agent = _host("agent.subagent_lifecycle", "get_active_subagent_parent")()
    except RoomUnavailable as exc:
        return None, str(exc)
    if agent is None:
        return None, "no running turn was bound"
    if not hasattr(agent, "context_compressor"):
        return None, "the running turn's agent has no context_compressor to compare with this engine"
    if agent.context_compressor is not engine:
        return None, "the running turn's agent runs another engine copy"
    return agent, ""


def route_of(engine: Any) -> tuple[dict, str, Optional[Any], str]:
    """The one snapshot of a tool call's host values, read once at its entry: the session's
    route (api_mode, model, base_url, provider), where it was read (the bound agent's
    attributes, or the engine's last ``update_model``), the bound agent itself (for the
    guard's and the pad's reads), and why the agent's values were not read, where they were
    not. Every reader of these values in the call takes them from here."""
    keys = ("api_mode", "model", "base_url", "provider")
    agent, why = bound_agent(engine)
    if agent is not None:
        try:
            return {key: str(getattr(agent, key) or "") for key in keys}, "agent", agent, ""
        except AttributeError as exc:
            why = f"the bound agent's {exc.name or 'route'} cannot be read"
    return {key: str(getattr(engine, key, "") or "") for key in keys}, "engine", agent, why


def host_halt(agent: Optional[Any], why: str = "no running turn was bound") -> tuple[Optional[str], str]:
    """(GUARD_CAUSE where the host's tool-loop guard will append its notice to this call's
    result, else None; "" where the guard was read, else why not). It reads
    ``agent._tool_guardrails.halt_decision`` of the bound agent (the call's snapshot,
    ``route_of``, with its why where there is none). Where any of it cannot be read: no
    protection and no approximation, and the why is said on every page whose delivered
    images the guard would decide (``ImageRoom.guard_unread``)."""
    if agent is None:
        return None, why
    try:
        halt = agent._tool_guardrails.halt_decision
    except AttributeError as exc:
        return None, f"the host's agent has no {exc.name or 'readable halt'} to read"
    if halt is None:
        return None, ""                                  # read: no halt stands
    if not hasattr(halt, "code"):
        return None, "the host's halt decision carries no code to read"
    return (GUARD_CAUSE if halt.code in GUARD_CODES else None), ""


# --- The host's static limits --------------------------------------------------------------------

def static_limits() -> tuple[int, int, Any]:
    """(the host's block ceiling, its byte budget, its payload measure), from its constants."""
    return (int(_host("agent.image_eviction_policy", "OUTBOUND_IMAGE_LIMIT")),
            int(_host("agent.image_eviction_policy", "OUTBOUND_IMAGE_BUDGET_BYTES")),
            _host("agent.context_compressor", "_image_payload"))


class ImageRoom:
    """What a page may hold: at most the host's block ceiling of images and its byte budget of
    payload (static facts about our own output); none where the host's tool-loop guard has
    halted this turn. Every value here is a constant of the call, read once with its route
    (``expansion.Route.room``) before any item is built. It carries the call's statements
    about where its values came from, each an annotation of the item that value decided
    (``expansion._provenance``, under ``lcm``): ``route_note`` (the route read from the
    engine), where the route decided something on the item (an image it was asked about;
    the host's fill of an empty message); ``guard_unread`` (the guard could not be read) and
    ``note`` (the host's own retirement), where an image is delivered."""

    def __init__(self, note: str = NOTE, guard: Optional[str] = None, route_note: str = "",
                 guard_unread: str = ""):
        self.note, self.guard, self.route_note, self.guard_unread = note, guard, route_note, guard_unread
        try:
            self.limit, self.budget, self.measure = static_limits()
            self.unreadable = ""
        except RoomUnavailable as exc:
            self.limit = self.budget = self.measure = None
            self.unreadable = str(exc)

    def size(self, part: dict) -> int:
        return int(self.measure({"role": "tool", "content": [part]})[1])

    def admits(self, images: list, key: Any = None) -> bool:
        if not images:
            return True
        if self.guard is not None or self.unreadable:
            return False
        return len(images) <= self.limit and sum(self.size(p) for p in images) <= self.budget

    def blocked(self) -> str:
        """Why no request of this call has room for any image, whatever the image: the guard
        has halted the turn, the host's limits cannot be read, or its ceiling admits none; ""
        where a request has room for one. Asked when an item is built (``_images_of``)."""
        if self.guard is not None:
            return self.guard
        if self.unreadable:
            return f"{self.unreadable}, so the host's image limits are not known"
        if 1 > self.limit:
            return f"the host's ceiling of {self.limit} images per request admits none"
        return ""

    def why_not(self, image: dict, key: Any = None) -> str:
        """Why no page admits ``image``: the check that fails, asked on its own."""
        if self.blocked():
            return f"not shown: {self.blocked()}; the store still holds it"
        size = self.size(image)
        if size > self.budget:
            return (f"not shown: this image is {size} bytes as the host measures it, over the host's budget of "
                    f"{self.budget} bytes for the images of one request; the store still holds it")
        raise RoomUnavailable("why_not was asked about an image the host's limits admit")


def host_image_room(route_source: str, agent: Optional[Any], why: str = "") -> ImageRoom:
    """The ``ImageRoom`` of this call, from its snapshot (``route_of``: the source, the bound
    agent, and why the agent's values were not read)."""
    guard, unread = host_halt(agent, why or "no running turn was bound")
    route_note = f"{ROUTE_FROM_ENGINE}, since {why or 'no running turn was bound'}" if route_source != "agent" else ""
    return ImageRoom(note=NOTE, guard=guard, route_note=route_note,
                     guard_unread=GUARD_UNREAD.format(why=unread) if unread else "")


# --- The route's converter on one image: which result, which slot, by position -------------------

def _newest_calls(messages: list) -> Optional[int]:
    return next((i for i in range(len(messages) - 1, -1, -1) if isinstance(messages[i], dict)
                 and messages[i].get("role") == "assistant" and messages[i].get("tool_calls")), None)


def _run(messages: list, at: int) -> list:
    out = []
    for i in range(at + 1, len(messages)):
        if not (isinstance(messages[i], dict) and messages[i].get("role") == "tool"):
            break
        out.append(i)
    return out


def row_of(request: list, k: int) -> Optional[int]:
    """Row k of the run after the newest assistant message with calls, paired with its call k
    by the host's own relation (``tool_result_id_variants`` against ``tool_call_id_variants``)."""
    at = _newest_calls(request)
    if at is None:
        return None
    run, calls = _run(request, at), request[at]["tool_calls"]
    if k >= len(run) or k >= len(calls):
        return None
    call_variants = _host("agent.message_sanitization", "tool_call_id_variants")
    result_variants = _host("agent.message_sanitization", "tool_result_id_variants")
    return run[k] if set(call_variants(calls[k])) & set(result_variants(request[run[k]].get("tool_call_id"))) else None


def convert_request(route: Any, request: list) -> tuple[str, Any]:
    """The route's converter over ``request``, with the kwargs the host passes."""
    mode = route.api_mode
    if mode == "chat_completions":
        transport = _host("agent.transports", "get_transport")(mode)
        profile = _host("providers", "get_provider_profile")(route.provider) if route.provider else None
        out = transport.convert_messages(request, model=route.model, base_url=route.base_url, provider_profile=profile)
        if profile is not None:
            out = profile.prepare_messages(out)
        if _host("agent.gemini_native_adapter", "is_native_gemini_base_url")(route.base_url):
            return "gemini", _host("agent.gemini_native_adapter", "build_gemini_request")(messages=out,
                                                                                         model=route.model)
        return "chat", out
    if mode == "anthropic_messages":
        convert = _host("agent.anthropic_message_convert", "convert_messages_to_anthropic")
        return "anthropic", convert(request, base_url=route.base_url, model=route.model)[1]
    if mode == "codex_responses":
        transport = _host("agent.transports", "get_transport")(mode)
        return "codex", transport.convert_messages(request, model=route.model, base_url=route.base_url)
    if mode == "bedrock_converse":
        transport = _host("agent.transports", "get_transport")(mode)
        return "bedrock", transport.convert_messages(request)[1]
    if not mode:
        raise RoomUnavailable("the session's route has an empty api_mode, so the host's converter for it is not known")
    raise RoomUnavailable(f"how the host carries a tool result's images on this session's route ({mode}) is not known "
                          f"to this plugin")


def route_name(route: Any) -> str:
    """The route as a cause names it: its API mode, and Gemini's native API where the host
    sends a chat_completions request there."""
    mode = str(route.api_mode) if route.api_mode else "an empty api_mode"
    if route.api_mode != "chat_completions":
        return mode
    try:
        native = bool(_host("agent.gemini_native_adapter", "is_native_gemini_base_url")(route.base_url))
    except RoomUnavailable as exc:
        return f"{mode}; whether it is Gemini's native API could not be read: {exc}"
    return f"{mode}, Gemini's native API" if native else mode


def count_images(tree: Any) -> int:
    """The image blocks in a converted result, in any provider's shape (a count, not an identity)."""
    if isinstance(tree, list):
        return sum(count_images(v) for v in tree)
    if not isinstance(tree, dict):
        return 0
    if tree.get("type") in ("image_url", "image", "input_image") or "inlineData" in tree:
        return 1
    if isinstance(tree.get("image"), dict) and "source" in tree["image"]:
        return 1
    return sum(count_images(v) for v in tree.values())


NO_CALLS = "its output holds no assistant turn with tool calls"
NO_RESULT_TURN = "its output holds no turn after the one with the calls"


def wire_result(wire: tuple[str, Any], request: list, k: int) -> tuple[Optional[int], str]:
    """(the image blocks of the result the converter paired with the turn's call k, "") (the
    tool_use id, the function_call's call_id, the functionCall, the toolUseId; chat_completions
    and the profiles keep the message list index for index), or (None, where the pairing
    breaks), each cause named at the step where it breaks."""
    kind, out = wire
    if kind == "chat":
        index = row_of(request, k)
        if index is None:
            return None, "the request holds no result paired with the call"
        if index >= len(_list(out)):
            return None, "its output is shorter than the request it was given"
        return count_images(_list(out[index].get("content"))), ""
    if kind == "anthropic":
        at = next((i for i in range(len(out) - 1, -1, -1) if out[i].get("role") == "assistant" and any(
            isinstance(b, dict) and b.get("type") == "tool_use" for b in _list(out[i].get("content")))), None)
        if at is None:
            return None, NO_CALLS
        if at + 1 >= len(out):
            return None, NO_RESULT_TURN
        uses = [b for b in out[at]["content"] if isinstance(b, dict) and b.get("type") == "tool_use"]
        if k >= len(uses):
            return None, f"its assistant turn holds {len(uses)} tool uses, fewer than the call's position {k + 1}"
        block = next((b for b in _list(out[at + 1].get("content")) if isinstance(b, dict)
                      and b.get("type") == "tool_result" and b.get("tool_use_id") == uses[k].get("id")), None)
        if block is None:
            return None, "its next turn holds no tool_result for the call's tool_use id"
        return count_images(_list(block.get("content"))), ""
    if kind == "codex":
        calls = [item for item in _list(out) if isinstance(item, dict) and item.get("type") == "function_call"]
        at = _newest_calls(request)
        n = len(request[at]["tool_calls"]) if at is not None else 0
        if not n:
            return None, "the request holds no assistant turn with tool calls"
        if len(calls) < n:
            return None, f"its output holds {len(calls)} function calls, fewer than the {n} of the request's turn"
        call_id = calls[len(calls) - n + k].get("call_id")
        item = next((i for i in _list(out) if isinstance(i, dict) and i.get("type") == "function_call_output"
                     and i.get("call_id") == call_id), None)
        if item is None:
            return None, "its output holds no function_call_output for the call's call_id"
        return count_images(_list(item.get("output"))), ""
    if kind == "gemini":
        contents = _list(out.get("contents"))
        at = next((i for i in range(len(contents) - 1, -1, -1) if contents[i].get("role") == "model" and any(
            isinstance(p, dict) and "functionCall" in p for p in _list(contents[i].get("parts")))), None)
        if at is None:
            return None, NO_CALLS
        if at + 1 >= len(contents):
            return None, NO_RESULT_TURN
        calls = [p["functionCall"] for p in contents[at]["parts"] if isinstance(p, dict) and "functionCall" in p]
        responses = [p["functionResponse"] for p in _list(contents[at + 1].get("parts"))
                     if isinstance(p, dict) and "functionResponse" in p]
        if k >= len(calls):
            return None, f"its model turn holds {len(calls)} function calls, fewer than the call's position {k + 1}"
        if calls[k].get("id"):
            response = next((r for r in responses if r.get("id") == calls[k]["id"]), None)
            missing = "its next turn holds no functionResponse with the call's id"
        else:
            response = responses[k] if k < len(responses) and responses[k].get("name") == calls[k].get("name") else None
            missing = "its next turn holds no functionResponse of the call's name at the call's position"
        if response is None:
            return None, missing
        return count_images(_list(response.get("parts"))), ""
    if kind == "bedrock":
        at = next((i for i in range(len(out) - 1, -1, -1) if out[i].get("role") == "assistant" and any(
            isinstance(b, dict) and "toolUse" in b for b in _list(out[i].get("content")))), None)
        if at is None:
            return None, NO_CALLS
        if at + 1 >= len(out):
            return None, NO_RESULT_TURN
        uses = [b["toolUse"] for b in out[at]["content"] if isinstance(b, dict) and "toolUse" in b]
        if k >= len(uses):
            return None, f"its assistant turn holds {len(uses)} tool uses, fewer than the call's position {k + 1}"
        result = next((b["toolResult"] for b in _list(out[at + 1].get("content")) if isinstance(b, dict)
                       and isinstance(b.get("toolResult"), dict)
                       and b["toolResult"].get("toolUseId") == uses[k].get("toolUseId")), None)
        if result is None:
            return None, "its next turn holds no toolResult for the call's toolUseId"
        return count_images(_list(result.get("content"))), ""
    return None, f"its output is of a kind ({kind}) this plugin does not read"
