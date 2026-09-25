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
member records in order, the summariser route, the effort). A second attempt that cuts
the same chunk with the same summariser and effort joins the call in flight instead of
making another, and its hook receives the call's progress; where no call is in flight,
a summary of the same records already written by the same summariser route and effort
is reused (#33 D12). Joining and reuse apply the one rule. A registry is per process:
another process never joins, but reuses a summary once written. An entry exists only
with a live worker; a call nobody wants any more leaves the registry at the moment
that is decided, so no later attempt joins a call being given up.

**What the worker carries (D10, R4).** The host's progress hook, read on the
``compress()`` thread, ticks on the worker for every streamed payload of the call
(``aux_progress_hook``); with a joined call the worker ticks every subscribed attempt's
hook. The host's deadline, read on the ``compress()`` thread and captured with the call
when an attempt subscribes (``aux_stream_deadline``), bounds each call: its ``timeout``
is the deadline less the time of dispatch (#33: no
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

**What reaches the attempts.** Each subscriber is told the call's failure with its kind
(``escalation.FAILURE_KINDS``), so the attempt can tell the chunk's own failures from
the endpoint's; the first subscriber a failure reaches records it, the others only
read what it recorded, so a call several attempts joined is one trial of the chunk.
The first time the host stamps a dispatch of the call
(``escalation._DispatchStamp``), ``ChunkCall.sent`` tells the attempt that started it.
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
    """What one attempt got for its chunk: the derivation written for it, or why not
    (``failure``), and the cause the host is to show for it (``cause``), which names the
    chunk and says whether the failure counts toward the chunk's own."""

    derivation: Optional[str] = None
    failure: Optional[str] = None
    cause: Optional[str] = None
    # Whether this subscriber wrote the call's failure into the store: only then is the
    # call's failure recorded, and no later subscriber records it again.
    recorded: bool = False


class Subscriber:
    """One attempt's interest in one chunk's call.

    ``wanted`` says whether the attempt still wants it (its captured check and whether
    it is the host's current working attempt); ``hook`` and ``deadline`` are the host's
    progress hook and deadline read on its ``compress()`` thread; ``deliver`` writes the
    summary as a derivation of the attempt's own chunk, or records why there is none,
    and returns the ``Outcome``. It runs on the thread that has the result. A call's
    failure is recorded once (``record``): one call is one trial of the chunk, however
    many attempts joined it. The subscriber asked to record it says whether the write
    succeeded (``Outcome.recorded``); where it did not, the next subscriber is asked."""

    def __init__(self, *, wanted: Callable[[], bool], hook: Optional[Callable[[], Any]],
                 deadline: Optional[float],
                 deliver: Callable[[Optional[ChunkSummary], Optional[str], bool, Optional[str], bool], Outcome],
                 on_done: Optional[Callable[[], None]] = None) -> None:
        self._wanted = wanted
        self.hook = hook
        self.deadline = deadline
        self._deliver = deliver
        self._on_done = on_done
        self.outcome: Optional[Outcome] = None

    def wanted(self) -> bool:
        try:
            return bool(self._wanted())
        except Exception:
            logger.warning("LCM could not ask whether an attempt still wants its summary", exc_info=True)
            return False

    def finish(self, summary: Optional[ChunkSummary], failure: Optional[str], abandoned: bool = False,
               kind: Optional[str] = None, record: bool = True) -> None:
        """Deliver, then say so: the outcome is set before ``on_done`` is called, so
        whoever is told reads a complete outcome. ``kind`` is a failure's kind
        (``escalation.FAILURE_KINDS``); ``record`` whether this subscriber is to record
        the call's failure (no earlier one has) or only read what was recorded."""
        try:
            self.outcome = self._deliver(summary, failure, abandoned, kind, record)
        except BaseException as exc:  # a store closed meanwhile, or anything else: never swallowed silently
            logger.warning("LCM could not deliver a chunk's summary (%s: %s)", type(exc).__name__, exc)
            self.outcome = Outcome(failure=f"the summary could not be written ({type(exc).__name__}: {exc})")
        finally:
            if self._on_done is not None:
                try:
                    self._on_done()
                except Exception:
                    logger.warning("LCM could not report a chunk's outcome to its attempt", exc_info=True)


class ChunkCall:
    """One chunk's summariser call in flight, and the attempts subscribed to it."""

    def __init__(self, key: tuple, limiter: EndpointLimiter, limit: int,
                 on_sent: Optional[Callable[[], None]] = None) -> None:
        self.key = key
        self.limiter = limiter
        self.limit = limit
        # Told once, when the host first dispatches a request of this call to the
        # provider (``sent``; #33 D14: a dispatched chunk is kept on a retry).
        self._on_sent = on_sent
        self._sent = False
        # Whether a subscriber has written the call's failure into the store.
        self._failure_recorded = False
        self._lock = threading.Lock()
        self._subscribers: list[Subscriber] = []
        self._delivered = 0
        self._deadline: Optional[float] = None
        self.closed = False

    def add(self, subscriber: Subscriber) -> bool:
        """Subscribe, unless the call has closed. Called under the registry lock.

        The host's deadline is captured here, as each attempt subscribes, and held with
        the call: the latest of the subscribers' deadlines, or none where one of them
        has none. It does not change when an attempt leaves, so a dispatched call always
        carries it."""
        if self.closed:
            return False
        with self._lock:
            first = not self._subscribers
            self._subscribers.append(subscriber)
            if first:
                self._deadline = subscriber.deadline
            elif self._deadline is not None:
                self._deadline = None if subscriber.deadline is None else max(self._deadline, subscriber.deadline)
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
        """The host's deadline captured with the call (``add``)."""
        with self._lock:
            return self._deadline

    def sent(self) -> None:
        """The host dispatched a request of this call to the provider (its dispatch
        stamp, ``escalation._DispatchStamp``): ``on_sent`` records it, until once it has
        succeeded. A write that raises leaves the dispatch unrecorded, so the host's next
        stamp of this call (a retry, level 2) records it; the exception goes on to the
        stamp, which warns. It runs on the thread the host sends from (the worker, or the
        host's protected provider thread, which carries the hook over)."""
        with self._lock:
            if self._sent:
                return
        if self._on_sent is not None:
            self._on_sent()
        with self._lock:
            self._sent = True

    def failure_recorded(self) -> bool:
        """Whether a subscriber has written the call's failure into the store."""
        with self._lock:
            return self._failure_recorded

    def mark_failure_recorded(self) -> None:
        """The call's failure is in the store: later subscribers only read it."""
        with self._lock:
            self._failure_recorded = True

    def still_needed(self) -> bool:
        """Whether any attempt still wants the call. Where none does, the call leaves
        the registry and closes at once, in one step under the registry lock, before
        its abandonment is delivered: an attempt that comes later starts a call of its
        own instead of joining one that is being given up."""
        if self.wanted():
            return True
        with _REGISTRY_LOCK:
            if self.wanted():
                return True
            self.closed = True
            if _REGISTRY.get(self.key) is self:
                del _REGISTRY[self.key]
        return False

    def wait(self, seconds: float) -> None:
        """A wait between retries, given up as soon as no attempt wants the call."""
        end = time.monotonic() + max(0.0, seconds)
        while True:
            if not self.still_needed():
                raise CallAbandoned()
            left = end - time.monotonic()
            if left <= 0:
                return
            time.sleep(min(_POLL_S, left))

    @contextlib.contextmanager
    def dispatch(self) -> Iterator[Optional[float]]:
        """One provider call: a slot of the endpoint's limiter, held for the call only,
        and the call's captured deadline installed for the host's stream consumer.
        Yields that deadline. Where every attempt left while the slot was being granted,
        nothing is dispatched: the slot is given back and the call is abandoned. A
        ``Retry-After`` is to be passed to ``hold`` inside this scope, before the slot is
        given back, so that no queued call dispatches in between."""
        if not self.limiter.acquire(self.limit, self.still_needed):
            raise CallAbandoned()
        try:
            if not self.still_needed():
                raise CallAbandoned()
            deadline = self.deadline()
            with _scope(_host_stream_deadline, deadline):
                yield deadline
        finally:
            self.limiter.release()


_REGISTRY: dict[tuple, ChunkCall] = {}
_REGISTRY_LOCK = threading.Lock()
_WORKER_NUMBERS = itertools.count(1)


def _close(call: ChunkCall, summary: Optional[ChunkSummary], failure: Optional[str], abandoned: bool,
           kind: Optional[str] = None) -> None:
    """Deliver to every subscriber, then close the call and leave the registry, in that
    order: a summary is in the store before a later attempt can miss the call. An
    abandoned call has already left the registry (``still_needed``)."""
    while True:
        with _REGISTRY_LOCK:
            pending = call.take_undelivered()
            if not pending:
                call.closed = True
                if _REGISTRY.get(call.key) is call:
                    del _REGISTRY[call.key]
                return
        for subscriber in pending:
            # The right to record is used up only by a write that succeeded: where the
            # store refused it, the next subscriber records the failure.
            record = not call.failure_recorded()
            subscriber.finish(summary, failure, abandoned, kind, record=record)
            if record and subscriber.outcome is not None and subscriber.outcome.recorded:
                call.mark_failure_recorded()


def _worker(call: ChunkCall, run: Callable[[ChunkCall], ChunkSummary],
            describe: Callable[[BaseException], str]) -> None:
    summary: Optional[ChunkSummary] = None
    failure: Optional[str] = None
    kind: Optional[str] = None
    abandoned = False
    with _scope(_host_interrupt_protection, active=True), _scope(_host_progress_hook, call.tick):
        try:
            summary = run(call)
        except CallAbandoned:
            abandoned = True
            failure = "no attempt wants the summary any more (each was cancelled or superseded)"
        except BaseException as exc:  # a SummaryFailure, or anything else: never swallowed
            failure = describe(exc)
            # A SummaryFailure names its kind; anything else is of none the plugin knows.
            kind = getattr(exc, "kind", None) or "other"
    _close(call, summary, failure, abandoned, kind)


def join_or_start(
    key: tuple,
    subscriber: Subscriber,
    *,
    limiter: EndpointLimiter,
    limit: int,
    reuse: Callable[[], Optional[ChunkSummary]],
    run: Callable[[ChunkCall], ChunkSummary],
    describe: Callable[[BaseException], str],
    on_sent: Optional[Callable[[], None]] = None,
) -> str:
    """Join the call in flight for ``key``, reuse a summary already written, or start
    the call on a new daemon worker. Returns "joined", "reused", "started", or "failed"
    when the worker could not start: then no entry is registered, and the subscriber is
    told of the failure, visibly. ``on_sent`` is told when the host first dispatches a
    request of a call started here (``ChunkCall.sent``).

    ``key`` holds everything two attempts must share to share a call: the session, the
    chunk's member records in order, the summariser route and the effort (the rule a
    reuse applies too)."""
    start_failure: Optional[str] = None
    with _REGISTRY_LOCK:
        call = _REGISTRY.get(key)
        if call is not None and call.add(subscriber):
            return "joined"
        reused = reuse()
        if reused is None:
            call = ChunkCall(key, limiter, limit, on_sent=on_sent)
            call.add(subscriber)
            context = contextvars.copy_context()
            try:
                threading.Thread(
                    target=context.run, args=(_worker, call, run, describe),
                    name=f"lcm-summary-{next(_WORKER_NUMBERS)}", daemon=True,
                ).start()
            except BaseException as exc:
                call.closed = True
                start_failure = f"the summariser call's worker could not start ({type(exc).__name__}: {exc})"
            else:
                # Registered only with a live worker. The worker cannot close the call
                # before this: closing takes the registry lock, held here.
                _REGISTRY[key] = call
                return "started"
    if start_failure is not None:
        logger.warning("LCM %s", start_failure)
        subscriber.finish(None, start_failure, kind="other")
        return "failed"
    subscriber.finish(reused, None)
    return "reused"
