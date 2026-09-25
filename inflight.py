"""Summariser calls in flight (#12, #33 D10, D12, D13; the orchestrator's ruling R4).

Every chunk of a compaction is issued at once. Each chunk's call runs on a daemon thread
of the plugin's own, started under a copy of the ``compress()`` thread's context, so it
holds no slot of the host's compression pool (#33 D13). An interpreter exit does not
wait for it: what an exit loses is a summary nothing active depends on, which the next
attempt makes again.

**The limiter.** One per endpoint, process-wide, over the plugin's summariser calls, so
that sessions and engine copies compacting at once share the endpoint and do not
multiply against its limits (#33, "Several sessions in one process"). At most
``limit`` calls run at once (default 8, configurable per endpoint). A provider's
``Retry-After`` holds the whole endpoint: no call to it is dispatched before the time
the provider named. A call holds its slot only while the provider has it: a wait
between retries gives the slot back.

**The registry.** Calls are registered process-wide by (plugin session, the chunk's
member records in order). A second attempt that cuts the same chunk joins the call in
flight instead of making another, and its hook receives the call's progress; where no
call is in flight, a summary of the same records already written by the same
summariser route and effort is reused (#33 D12). A registry is per process: another
process never joins, but reuses a summary once written.

**What the worker carries (D10, R4).** The host's progress hook, read on the
``compress()`` thread, ticks on the worker for every streamed payload of the call
(``aux_progress_hook``); with a joined call the worker ticks every subscribed attempt's
hook. The host's deadline, read on the ``compress()`` thread (``aux_stream_deadline``),
bounds each call: its ``timeout`` is the deadline less the time of dispatch (#33: no
per-call timeout of the plugin's own), and the host's stream consumer stops there. The
worker marks its calls interrupt-protected (``aux_interrupt_protection``) without a
cancellation source: installing the attempt's cancellation check there would make the
host raise inside the worker on a cancel and lose a summary already in flight, which
D13 and Q17a keep. So a user's /stop does not stop a call in flight before the host's
deadline; it stops the attempt, whose ``compress()`` returns at once (the caller polls
the captured check), and no further call is started for it.

**Summaries are written by the worker as they arrive** (#33 Q17a), each as a derivation
of every subscribed attempt's chunk: a summary of a chunk is a fact about that chunk's
content and makes nothing active, so it is written whether or not its attempt is still
current. A call none of whose attempts wants it any more (each was cancelled or
superseded) starts no further dispatch: not a first one waiting for the limiter, not a
retry. These all rest on host internals (the thread-locals of
``agent.auxiliary_client``); ask A-33.1.
"""

from __future__ import annotations

import contextlib
import contextvars
import itertools
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Optional

logger = logging.getLogger(__name__)

try:  # host internals (#33 D10, ask A-33.1)
    from agent.auxiliary_client import (  # type: ignore
        _aux_progress as _HOST_PROGRESS,
        _current_aux_stream_deadline as _host_current_deadline,
        aux_interrupt_protection as _host_interrupt_protection,
        aux_progress_hook as _host_progress_hook,
        aux_stream_deadline as _host_stream_deadline,
    )
except Exception:  # pragma: no cover - older or absent host
    _HOST_PROGRESS = None
    _host_current_deadline = None
    _host_interrupt_protection = None
    _host_progress_hook = None
    _host_stream_deadline = None

DEFAULT_CALLS_PER_ENDPOINT = 8
# How often a waiting thread asks whether its call is still wanted.
_POLL_S = 0.25


def host_progress_hook() -> Optional[Callable[[], Any]]:
    """The host's progress hook installed on the calling thread, or None."""
    hook = getattr(_HOST_PROGRESS, "hook", None) if _HOST_PROGRESS is not None else None
    return hook if callable(hook) else None


def host_deadline() -> Optional[float]:
    """The waiting host's absolute ``time.monotonic()`` deadline on the calling thread, or None."""
    if _host_current_deadline is None:
        return None
    try:
        value = _host_current_deadline()
    except Exception:
        return None
    return float(value) if isinstance(value, (int, float)) else None


def _scope(factory: Optional[Callable[..., Any]], *args: Any, **kwargs: Any):
    return factory(*args, **kwargs) if factory is not None else contextlib.nullcontext()


class CallAbandoned(BaseException):
    """No attempt wants the call any more: it starts no further dispatch. A
    BaseException, so that no ``except Exception`` on the way turns it into a failure."""


class EndpointLimiter:
    """At most ``limit`` calls at once to one endpoint, and none before a ``Retry-After``
    the endpoint named has passed."""

    def __init__(self, key: str) -> None:
        self.key = key
        self._cond = threading.Condition()
        self._active = 0
        self._held_until = 0.0
        self.peak = 0

    @property
    def active(self) -> int:
        with self._cond:
            return self._active

    def acquire(self, limit: int, wanted: Callable[[], bool]) -> bool:
        """A slot, or False as soon as the call is no longer wanted."""
        limit = max(1, int(limit))
        with self._cond:
            while True:
                if not wanted():
                    return False
                now = time.monotonic()
                if self._active < limit and now >= self._held_until:
                    self._active += 1
                    self.peak = max(self.peak, self._active)
                    return True
                wait = _POLL_S if now >= self._held_until else min(_POLL_S, self._held_until - now)
                self._cond.wait(timeout=wait)

    def release(self) -> None:
        with self._cond:
            self._active = max(0, self._active - 1)
            self._cond.notify_all()

    def hold(self, seconds: float) -> None:
        """The endpoint named a ``Retry-After``: nothing is dispatched to it before then."""
        if not seconds or seconds <= 0:
            return
        with self._cond:
            until = time.monotonic() + float(seconds)
            if until > self._held_until:
                self._held_until = until
                logger.warning("LCM holds summariser calls to %s for %.1fs (the provider's Retry-After)",
                               self.key, seconds)
            self._cond.notify_all()


_LIMITERS: dict[str, EndpointLimiter] = {}
_LIMITERS_LOCK = threading.Lock()


def endpoint_key(provider: str, base_url: str) -> str:
    """The endpoint a route reaches: its base URL where it names one, else its provider."""
    url = str(base_url or "").strip().rstrip("/")
    return url if url else f"provider:{str(provider or '').strip()}"


def limiter_for(key: str) -> EndpointLimiter:
    with _LIMITERS_LOCK:
        limiter = _LIMITERS.get(key)
        if limiter is None:
            limiter = _LIMITERS[key] = EndpointLimiter(key)
        return limiter


@dataclass(frozen=True)
class ChunkSummary:
    """A chunk's summary as the summariser wrote it, with its provenance."""

    text: str
    level: Optional[int]
    budget: Optional[int]
    finish_reason: Optional[str]
    model: Optional[str]
    provider: Optional[str]
    effort: Optional[str]


@dataclass
class Outcome:
    """What one attempt got for its chunk: the derivation written for it, or why not."""

    derivation: Optional[str] = None
    failure: Optional[str] = None


class Subscriber:
    """One attempt's interest in one chunk's call.

    ``wanted`` says whether the attempt still wants it (its captured check and whether
    it is the host's current working attempt); ``hook`` and ``deadline`` are the host's
    progress hook and deadline read on its ``compress()`` thread; ``deliver`` writes the
    summary as a derivation of the attempt's own chunk, or records why there is none,
    and returns the ``Outcome``. It runs on the thread that has the result."""

    def __init__(self, *, wanted: Callable[[], bool], hook: Optional[Callable[[], Any]],
                 deadline: Optional[float],
                 deliver: Callable[[Optional[ChunkSummary], Optional[str], bool], Outcome]) -> None:
        self._wanted = wanted
        self.hook = hook
        self.deadline = deadline
        self._deliver = deliver
        self.done = threading.Event()
        self.outcome: Optional[Outcome] = None

    def wanted(self) -> bool:
        try:
            return bool(self._wanted())
        except Exception:
            logger.warning("LCM could not ask whether an attempt still wants its summary", exc_info=True)
            return False

    def finish(self, summary: Optional[ChunkSummary], failure: Optional[str], abandoned: bool = False) -> None:
        try:
            self.outcome = self._deliver(summary, failure, abandoned)
        except BaseException as exc:  # a store closed meanwhile, or anything else: never swallowed silently
            logger.warning("LCM could not deliver a chunk's summary (%s: %s)", type(exc).__name__, exc)
            self.outcome = Outcome(failure=f"the summary could not be written ({type(exc).__name__}: {exc})")
        finally:
            self.done.set()


class ChunkCall:
    """One chunk's summariser call in flight, and the attempts subscribed to it."""

    def __init__(self, key: tuple, limiter: EndpointLimiter, limit: int) -> None:
        self.key = key
        self.limiter = limiter
        self.limit = limit
        self._lock = threading.Lock()
        self._subscribers: list[Subscriber] = []
        self._delivered = 0
        self.closed = False

    def add(self, subscriber: Subscriber) -> bool:
        """Subscribe, unless the call has closed. Called under the registry lock."""
        if self.closed:
            return False
        with self._lock:
            self._subscribers.append(subscriber)
        return True

    def subscribers(self) -> list[Subscriber]:
        with self._lock:
            return list(self._subscribers)

    def take_undelivered(self) -> list[Subscriber]:
        with self._lock:
            pending = self._subscribers[self._delivered:]
            self._delivered = len(self._subscribers)
            return pending

    def wanted(self) -> bool:
        return any(s.wanted() for s in self.subscribers())

    def tick(self) -> None:
        """The host's progress hook of every subscribed attempt: a joined attempt
        receives the call's progress on its own hook (D12)."""
        for subscriber in self.subscribers():
            if subscriber.hook is not None:
                try:
                    subscriber.hook()
                except Exception:
                    logger.debug("LCM: an attempt's progress hook failed", exc_info=True)

    def deadline(self) -> Optional[float]:
        """The latest deadline among the attempts that still want the call; None where
        one of them has no host deadline."""
        deadlines = [s.deadline for s in self.subscribers() if s.wanted()]
        if not deadlines or any(d is None for d in deadlines):
            return None
        return max(deadlines)

    def wait(self, seconds: float) -> None:
        """A wait between retries, given up as soon as no attempt wants the call."""
        end = time.monotonic() + max(0.0, seconds)
        while True:
            if not self.wanted():
                raise CallAbandoned()
            left = end - time.monotonic()
            if left <= 0:
                return
            time.sleep(min(_POLL_S, left))

    @contextlib.contextmanager
    def dispatch(self) -> Iterator[Optional[float]]:
        """One provider call: a slot of the endpoint's limiter, held for the call only,
        and the host's deadline installed for the host's stream consumer. Yields the
        deadline the call is bounded by."""
        if not self.limiter.acquire(self.limit, self.wanted):
            raise CallAbandoned()
        try:
            deadline = self.deadline()
            with _scope(_host_stream_deadline, deadline):
                yield deadline
        finally:
            self.limiter.release()


_REGISTRY: dict[tuple, ChunkCall] = {}
_REGISTRY_LOCK = threading.Lock()
_WORKER_NUMBERS = itertools.count(1)


def in_flight(key: tuple) -> Optional[ChunkCall]:
    with _REGISTRY_LOCK:
        return _REGISTRY.get(key)


def _close(call: ChunkCall, summary: Optional[ChunkSummary], failure: Optional[str], abandoned: bool) -> None:
    """Deliver to every subscriber, then close the call and leave the registry, in that
    order: a summary is in the store before a later attempt can miss the call."""
    while True:
        with _REGISTRY_LOCK:
            pending = call.take_undelivered()
            if not pending:
                call.closed = True
                if _REGISTRY.get(call.key) is call:
                    del _REGISTRY[call.key]
                return
        for subscriber in pending:
            subscriber.finish(summary, failure, abandoned)


def _worker(call: ChunkCall, run: Callable[[ChunkCall], ChunkSummary],
            describe: Callable[[BaseException], str]) -> None:
    summary: Optional[ChunkSummary] = None
    failure: Optional[str] = None
    abandoned = False
    with _scope(_host_interrupt_protection, active=True), _scope(_host_progress_hook, call.tick):
        try:
            summary = run(call)
        except CallAbandoned:
            abandoned = True
            failure = "no attempt wants the summary any more (each was cancelled or superseded)"
        except BaseException as exc:  # a SummaryFailure, or anything else: never swallowed
            failure = describe(exc)
    _close(call, summary, failure, abandoned)


def join_or_start(
    key: tuple,
    subscriber: Subscriber,
    *,
    limiter: EndpointLimiter,
    limit: int,
    reuse: Callable[[], Optional[ChunkSummary]],
    run: Callable[[ChunkCall], ChunkSummary],
    describe: Callable[[BaseException], str],
) -> str:
    """Join the call in flight for ``key``, reuse a summary already written, or start
    the call on a new daemon worker. Returns "joined", "reused" or "started"."""
    with _REGISTRY_LOCK:
        call = _REGISTRY.get(key)
        if call is not None and call.add(subscriber):
            return "joined"
        reused = reuse()
        if reused is None:
            call = ChunkCall(key, limiter, limit)
            call.add(subscriber)
            _REGISTRY[key] = call
            context = contextvars.copy_context()
            threading.Thread(
                target=context.run, args=(_worker, call, run, describe),
                name=f"lcm-summary-{next(_WORKER_NUMBERS)}", daemon=True,
            ).start()
            return "started"
    subscriber.finish(reused, None)
    return "reused"
