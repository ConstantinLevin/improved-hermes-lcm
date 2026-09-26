"""The plugin's own client for a configured summariser route (#9, 9.2; #33 D9, D10).

A configured summariser is called with exactly what the configuration names: its base
URL, its key (or no key, where the key is configured as ``none``), its wire and its model.
Nothing of the host's provider resolution is involved, so none of its config entries,
extra headers, credential pools, wire detection or fallback ladder can reach the call
(the orchestrator's ruling on the Codex review of c2efe0e). The session's own route
still goes through the host's ``call_llm`` (``escalation``).

Every configured value goes out exactly (the ruling on the Codex review of 188fb8b): the
model id as configured (no host helper that normalises it is used for the request: the
Anthropic request is assembled here, from the host's message converter only), the base
URL as named (the OpenAI SDK appends ``/chat/completions`` or ``/responses``, the
Anthropic SDK ``/v1/messages``, so an ``anthropic_messages`` URL that ends in ``/v1`` is
refused at load, ``escalation.configured_route_problem``), the key or none, the wire,
and the effort only in the field the model table documents.

Established clients only: the ``openai`` SDK for ``chat_completions`` and
``codex_responses`` (a core dependency of Hermes), the ``anthropic`` SDK for
``anthropic_messages`` (an optional extra of Hermes: where it is not installed, a route
on that wire is refused when the configuration is loaded, never installed lazily).

The messages are sent as ``summariser_message`` prepared them for the wire; a field is
left out only where the wire has no such field. Each wire's form is made by the host's
own converters, which are libraries and route nothing:

- Chat Completions: ``ChatCompletionsTransport.convert_messages`` (the host's
  transport), which drops only what is not a field of a Chat Completions message:
  the host's own bookkeeping (``_``-keys, ``timestamp``, ``platform_message_id``,
  ``effect_disposition``, ``tool_name``, ``api_content``), the carriers of other wires
  (``anthropic_content_blocks``, ``bedrock_content_blocks``, ``codex_reasoning_items``,
  ``codex_message_items``), a tool call's ``call_id`` and ``response_item_id``, ``name``
  on a tool result, an empty ``tool_calls``, and ``reasoning_details`` except on routes
  that replay it (OpenRouter, Nous). The message fields of the wire are those of OpenAI's
  API as the ``openai`` SDK 2.24.0 types them (``ChatCompletion*MessageParam``: role,
  content, name, tool_calls, tool_call_id, refusal, audio, function_call; a tool call:
  id, type, function). ``reasoning_content``, which ``summariser_message`` sets only for
  a route the host's ``needs_reasoning_echo`` names (DeepSeek, Kimi, MiMo), is kept, as
  is ``reasoning_details`` on OpenRouter (its "Preserving Reasoning", read 2026-09-26).
- Anthropic Messages: ``convert_messages_to_anthropic``.
- OpenAI Responses: ``_chat_messages_to_responses_input``.

Every call streams. Each payload with content ticks the host's progress hook installed on
the calling thread (the worker's, #33 D10). The host's deadline bounds the call: it is the
SDK's timeout, and the stream is left when it passes. The reply's text, its ending and
its reasoning come from the provider's own response; the reasoning is never part of the
summary. Only a complete ending is a summary (``ending_failure``): Chat Completions
``stop``, a Responses ``response.completed``, Anthropic ``end_turn``. The SDKs' own errors
are raised as they are; ``escalation`` reads their HTTP status the way it reads the
host's.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional

# The wires a configured route may name, each an established client of its own.
WIRES = ("chat_completions", "codex_responses", "anthropic_messages")
# The key value that means "send no authentication" (the orchestrator's ruling: no
# placeholder is ever invented).
NO_KEY = "none"

# The one complete ending per wire; every other ending fails the summary, with its kind
# (the ruling on the Codex review of 188fb8b). Values by the SDKs' types (openai 2.24.0:
# ``finish_reason`` stop, length, tool_calls, content_filter, function_call;
# ``incomplete_details.reason`` max_output_tokens, content_filter; ``status`` completed,
# failed, in_progress, cancelled, queued, incomplete. anthropic 0.87.0: ``StopReason``
# end_turn, max_tokens, stop_sequence, tool_use, pause_turn, refusal) and OpenRouter's
# normalised ``finish_reason`` "error" (its API reference).
COMPLETE = {"chat_completions": "stop", "codex_responses": "completed", "anthropic_messages": "end_turn"}
_ENDINGS = {
    "chat_completions": {
        "length": ("reply", "the reply was cut at the output limit"),
        "content_filter": ("reply", "the provider's content filter stopped the reply"),
        "tool_calls": ("reply", "the reply called a tool, and none was offered"),
        "function_call": ("reply", "the reply called a function, and none was offered"),
        "error": ("endpoint", "the provider ended the stream with an error"),
    },
    "codex_responses": {
        "incomplete:max_output_tokens": ("reply", "the reply was cut at the output limit"),
        "incomplete:content_filter": ("reply", "the provider's content filter stopped the reply"),
        "failed": ("endpoint", "the provider reported the response failed"),
        "cancelled": ("endpoint", "the provider cancelled the response"),
    },
    "anthropic_messages": {
        "max_tokens": ("reply", "the reply was cut at the output limit"),
        "refusal": ("reply", "the model refused to continue"),
        "tool_use": ("reply", "the reply called a tool, and none was offered"),
        "stop_sequence": ("other", "the reply stopped at a stop sequence, and none was sent"),
        "pause_turn": ("endpoint", "the provider paused the turn"),
        "model_context_window_exceeded": ("request", "the input exceeded the model's window"),
    },
}


def ending_failure(wire: str, ending: Optional[str]) -> Optional[tuple[str, str]]:
    """None where ``ending`` is the wire's complete ending, else (failure kind, why).
    An ending the table does not know (a Responses ``incomplete`` for another reason, a
    value a provider adds) fails as ``other``; no ending at all is the endpoint's."""
    if ending == COMPLETE.get(wire):
        return None
    if not ending:
        return "endpoint", "the stream ended without an ending"
    known = _ENDINGS.get(wire, {}).get(ending)
    if known is not None:
        return known
    return "other", f"the reply ended {ending!r}, not {COMPLETE.get(wire)!r}"


class StreamEnded(Exception):
    """The host's deadline passed while the summary streamed (``deadline``), or the
    stream broke off."""

    def __init__(self, message: str, *, deadline: bool = False) -> None:
        super().__init__(message)
        self.deadline = deadline


def sdk_missing(wire: str) -> Optional[str]:
    """Why the client of ``wire`` cannot be used here, or None: its SDK or the host's
    converter for it is not importable."""
    try:
        if wire == "anthropic_messages":
            import anthropic  # noqa: F401
            from agent.anthropic_adapter import convert_messages_to_anthropic  # noqa: F401
        elif wire == "codex_responses":
            import openai  # noqa: F401
            from agent.codex_responses_adapter import _chat_messages_to_responses_input  # noqa: F401
        elif wire == "chat_completions":
            import openai  # noqa: F401
            from agent.transports.chat_completions import ChatCompletionsTransport  # noqa: F401
        else:
            return f"{wire!r} is not one of {', '.join(WIRES)}"
    except Exception as exc:
        return (f"the client for {wire} cannot be loaded in the host's environment ({type(exc).__name__}: {exc}); "
                f"nothing is installed for it")
    return None


def _progress() -> Callable[[], None]:
    from .inflight import host_progress_hook

    hook = host_progress_hook()

    def tick() -> None:
        if hook is not None:
            try:
                hook()
            except Exception:
                pass
    return tick


def _deadline_check(timeout: Optional[float]) -> Callable[[], None]:
    end = time.monotonic() + timeout if timeout is not None else None

    def check() -> None:
        if end is not None and time.monotonic() > end:
            raise StreamEnded("the host's deadline passed while the summary streamed", deadline=True)
    return check


def _no_key(api_key: Any) -> bool:
    return str(api_key).strip().lower() == NO_KEY


def call(route: Any, messages: list[dict[str, Any]], *, timeout: Optional[float],
         max_tokens: Optional[int], effort: dict) -> tuple[str, Optional[str], dict]:
    """One streamed call on a configured route. Returns (text, the provider's ending as
    ``ending_failure`` reads it, the provider's own reasoning payloads seen). ``effort``
    is the request field that carries the reasoning effort for this route and model, from
    the model table ({} sends none)."""
    wire = route.api_mode
    if wire == "chat_completions":
        return _chat(route, messages, timeout=timeout, max_tokens=max_tokens, effort=effort)
    if wire == "codex_responses":
        return _responses(route, messages, timeout=timeout, max_tokens=max_tokens, effort=effort)
    if wire == "anthropic_messages":
        return _anthropic(route, messages, timeout=timeout, max_tokens=max_tokens, effort=effort)
    raise ValueError(f"no client for the wire {wire!r}")


def _openai_client(route: Any, timeout: Optional[float]):
    """The OpenAI SDK's client with the configured URL and key only. An empty key makes
    the SDK send no authorization header (``auth_headers``); the organization and project
    the SDK would take from the environment are cleared, so nothing but what the
    configuration names is sent."""
    import openai

    client = openai.OpenAI(base_url=route.base_url, api_key="" if _no_key(route.api_key) else route.api_key,
                           max_retries=0, timeout=timeout if timeout is not None else openai.NOT_GIVEN)
    client.organization = None
    client.project = None
    return client


def _anthropic_client(route: Any, timeout: Optional[float]):
    """The Anthropic SDK's client with the configured URL and key only: the key and the
    bearer token the SDK would take from the environment are cleared. With the key
    configured as ``none`` both authorization headers are omitted on the request
    (``_anthropic``): the SDK accepts a request without authentication only where the
    request itself omits them."""
    import anthropic

    no_key = _no_key(route.api_key)
    client = anthropic.Anthropic(base_url=route.base_url, api_key=None if no_key else route.api_key,
                                 max_retries=0, timeout=timeout if timeout is not None else anthropic.NOT_GIVEN)
    client.api_key = None if no_key else route.api_key
    client.auth_token = None
    return client


def chat_body(route: Any, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The messages as the Chat Completions wire takes them (see the module docstring)."""
    from agent.transports.chat_completions import ChatCompletionsTransport

    return ChatCompletionsTransport().convert_messages(messages, model=route.model, base_url=route.base_url)


def _chat(route, messages, *, timeout, max_tokens, effort):
    client = _openai_client(route, timeout)
    kwargs: dict[str, Any] = {"model": route.model, "messages": chat_body(route, messages), "stream": True}
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    kwargs.update(effort)
    tick, check = _progress(), _deadline_check(timeout)
    text: list[str] = []
    reasoning: dict[str, int] = {}
    finish = None
    with client.chat.completions.create(**kwargs) as stream:
        for chunk in stream:
            check()
            for choice in chunk.choices or []:
                delta = choice.delta
                if delta is not None:
                    if delta.content:
                        text.append(delta.content)
                        tick()
                    for field in ("reasoning", "reasoning_content", "reasoning_details"):
                        value = (getattr(delta, "model_extra", None) or {}).get(field)
                        if value:
                            reasoning[field] = reasoning.get(field, 0) + 1
                            tick()
                if choice.finish_reason:
                    finish = choice.finish_reason
    return "".join(text), (str(finish) if finish else None), reasoning


def _responses(route, messages, *, timeout, max_tokens, effort):
    from agent.codex_responses_adapter import _chat_messages_to_responses_input

    client = _openai_client(route, timeout)
    system = "\n\n".join(str(m.get("content") or "") for m in messages if m.get("role") == "system")
    rest = [m for m in messages if m.get("role") != "system"]
    kwargs: dict[str, Any] = {
        "model": route.model, "instructions": system,
        "input": _chat_messages_to_responses_input(rest, replay_encrypted_reasoning=False),
        "store": False, "stream": True,
    }
    if max_tokens:
        kwargs["max_output_tokens"] = max_tokens
    kwargs.update(effort)
    tick, check = _progress(), _deadline_check(timeout)
    text: list[str] = []
    reasoning: dict[str, int] = {}
    ending = None
    for event in client.responses.create(**kwargs):
        check()
        kind = getattr(event, "type", "")
        if kind == "response.output_text.delta":
            text.append(event.delta)
            tick()
        elif kind.startswith("response.reasoning"):
            reasoning[kind] = reasoning.get(kind, 0) + 1
            tick()
        elif kind == "response.completed":
            ending = "completed"
        elif kind == "response.incomplete":
            details = getattr(getattr(event, "response", None), "incomplete_details", None)
            ending = f"incomplete:{getattr(details, 'reason', None)}"
        elif kind == "response.failed":
            ending = "failed"
        elif kind == "response.cancelled":
            ending = "cancelled"
    return "".join(text), ending, reasoning


def _anthropic(route, messages, *, timeout, max_tokens, effort):
    from agent.anthropic_adapter import _resolve_anthropic_messages_max_tokens, convert_messages_to_anthropic

    client = _anthropic_client(route, timeout)
    # The request is assembled here, so that the configured model id goes out exactly;
    # the host's converter makes only the messages (system apart, tool calls and results
    # as blocks, a third-party base URL's signatures stripped). ``max_tokens``, which the
    # wire requires, is the model table's output cap, else the host's documented ceiling
    # for the model (it reads the id and rewrites nothing that is sent).
    system, body = convert_messages_to_anthropic(messages, base_url=route.base_url, model=route.model)
    kwargs: dict[str, Any] = {"model": route.model, "messages": body,
                              "max_tokens": max_tokens or _resolve_anthropic_messages_max_tokens(None, route.model)}
    if system:
        kwargs["system"] = system
    kwargs.update(effort)
    if _no_key(route.api_key):
        import anthropic

        kwargs["extra_headers"] = {"X-Api-Key": anthropic.omit, "Authorization": anthropic.omit}
    tick, check = _progress(), _deadline_check(timeout)
    text: list[str] = []
    reasoning: dict[str, int] = {}
    with client.messages.stream(**kwargs) as stream:
        for event in stream:
            check()
            if getattr(event, "type", "") == "content_block_delta":
                delta = event.delta
                if getattr(delta, "type", "") == "text_delta":
                    text.append(delta.text)
                else:
                    reasoning[delta.type] = reasoning.get(delta.type, 0) + 1
                tick()
        final = stream.get_final_message()
    stop = getattr(final, "stop_reason", None)
    return "".join(text), (str(stop) if stop else None), reasoning
