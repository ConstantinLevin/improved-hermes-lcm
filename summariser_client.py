"""The plugin's own client for a configured summariser route (#9, 9.2; #33 D9, D10).

A configured summariser is called with exactly what the configuration names: its base
URL, its key (or no key, where the key is configured as ``none``), its wire and its model.
Nothing of the host's provider resolution is involved, so none of its config entries,
extra headers, credential pools, wire detection or fallback ladder can reach the call
(the orchestrator's ruling on the Codex review of c2efe0e). The session's own route
still goes through the host's ``call_llm`` (``escalation``).

The base URL is sent as named: the OpenAI SDK appends ``/chat/completions`` or
``/responses`` to it, the Anthropic SDK ``/v1/messages`` (so an ``anthropic_messages``
URL that ends in ``/v1`` is refused at load, ``escalation.configured_route_problem``).

Established clients only: the ``openai`` SDK for ``chat_completions`` and
``codex_responses`` (a core dependency of Hermes), the ``anthropic`` SDK for
``anthropic_messages`` (an optional extra of Hermes: where it is not installed, a route
on that wire is refused when the configuration is loaded, never installed lazily). The
messages are made into each wire's form by the host's own converters, which are
libraries and route nothing: ``convert``/``build_anthropic_kwargs`` for Anthropic's
Messages, ``_chat_messages_to_responses_input`` for OpenAI's Responses.

Every call streams. Each payload with content ticks the host's progress hook installed on
the calling thread (the worker's, #33 D10). The host's deadline bounds the call: it is the
SDK's timeout, and the stream is left when it passes. The reply's text, its finish reason
and its reasoning come from the provider's own response; the reasoning is never part of
the summary. The SDKs' own errors are raised as they are; ``escalation`` reads their HTTP
status the way it reads the host's.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional

# The wires a configured route may name, each an established client of its own.
WIRES = ("chat_completions", "codex_responses", "anthropic_messages")
# The key value that means "send no authentication" (the orchestrator's ruling: no
# placeholder is ever invented).
NO_KEY = "none"
# Chat Completions takes these fields of a message; the plugin's input carries others
# (the host's fields, the placeholders' carriers) that belong to other wires.
_CHAT_FIELDS = ("role", "content", "name", "tool_calls", "tool_call_id")


class StreamEnded(Exception):
    """The provider's stream ended without the event that completes a reply, or the
    host's deadline passed while it streamed. ``deadline`` says which."""

    def __init__(self, message: str, *, deadline: bool = False) -> None:
        super().__init__(message)
        self.deadline = deadline


def sdk_missing(wire: str) -> Optional[str]:
    """Why the client of ``wire`` cannot be used here, or None: its SDK or the host's
    converter for it is not importable."""
    try:
        if wire == "anthropic_messages":
            import anthropic  # noqa: F401
            from agent.anthropic_adapter import build_anthropic_kwargs  # noqa: F401
        elif wire == "codex_responses":
            import openai  # noqa: F401
            from agent.codex_responses_adapter import _chat_messages_to_responses_input  # noqa: F401
        elif wire == "chat_completions":
            import openai  # noqa: F401
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
         max_tokens: Optional[int], effort: dict) -> tuple[str, str, dict]:
    """One streamed call on a configured route. Returns (text, finish reason, the
    provider's own reasoning fields seen). ``effort`` is the request field that carries
    the reasoning effort for this route and model, from the model table ({} sends none)."""
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


def _chat(route, messages, *, timeout, max_tokens, effort):
    client = _openai_client(route, timeout)
    body = [{k: m[k] for k in _CHAT_FIELDS if k in m} for m in messages]
    kwargs: dict[str, Any] = {"model": route.model, "messages": body, "stream": True}
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
    if finish is None:
        raise StreamEnded("the stream ended without a finish reason")
    return "".join(text), str(finish), reasoning


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
    finish = None
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
            finish = "stop"
        elif kind == "response.incomplete":
            details = getattr(getattr(event, "response", None), "incomplete_details", None)
            reason = getattr(details, "reason", None)
            finish = "length" if reason == "max_output_tokens" else f"incomplete: {reason}"
        elif kind == "response.failed":
            error = getattr(getattr(event, "response", None), "error", None)
            raise StreamEnded(f"the provider reported the response failed ({error})")
    if finish is None:
        raise StreamEnded("the stream ended without response.completed")
    return "".join(text), finish, reasoning


def _anthropic(route, messages, *, timeout, max_tokens, effort):
    from agent.anthropic_adapter import build_anthropic_kwargs

    client = _anthropic_client(route, timeout)
    # The host's own builder turns the messages into the Messages API's form (system
    # apart, tool calls and results as blocks, a third-party base URL's signatures
    # stripped); no reasoning config is given to it: the effort is the table's.
    kwargs = build_anthropic_kwargs(model=route.model, messages=messages, tools=None, max_tokens=max_tokens,
                                    reasoning_config=None, base_url=route.base_url)
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
            kind = getattr(event, "type", "")
            if kind == "content_block_delta":
                delta = event.delta
                if getattr(delta, "type", "") == "text_delta":
                    text.append(delta.text)
                    tick()
                else:
                    reasoning[delta.type] = reasoning.get(delta.type, 0) + 1
                    tick()
        final = stream.get_final_message()
    stop = getattr(final, "stop_reason", None)
    if stop is None:
        raise StreamEnded("the stream ended without a stop reason")
    return "".join(text), ("length" if stop == "max_tokens" else str(stop)), reasoning
