"""The compaction path (#29 W2 and W3; #34 D5).

``compress()`` runs one attempt against the plugin's record, which is the only store.
In order:

1. The host's list is classified against the session's effective return by identity
   only: the host's ``_row_id`` and the plugin's in-memory key, never by content.
   Where it cannot be, the compaction is aborted: nothing is written, the host keeps
   its list and shows the cause.
2. The tail takes what the target leaves, t = G − F − S − R_in (#13, #31), sized in
   the plugin's estimate with #31's conversion (``_tail_plan``), in whole groups of
   the cut's own grouping over the whole list: a tool call is never separated from
   its result. A tool row with an empty ``tool_call_id``, or one no assistant row
   before it made, is an error wherever it stands, checked once before any boundary.
   The forward check aborts where the boundary would separate a call from its result
   in either direction, or where a call at the boundary has no result. Its floor is
   D1's (the newest message, or the newest tool group); its ceiling leaves the oldest
   group outside; it never reaches back into the summaries the plugin returned. The
   material is every entry before the tail that is not the mechanism's layer (the
   host's system row, a summary the plugin returned, also as the host rewrote it: a
   summary revision holds a summary and is never chunked). Compaction runs at τ,
   forced, or on a provider's rejection; no minimum material applies. Where the floor
   alone is left, the compaction aborts visibly: the chain and the newest turn fill
   the context.
3. The material is cut into chunks of the size c (#31, #12), in list order and only
   between groups: a tool call and its results stay in one chunk. Each row is sized
   as the summariser receives it (``_cut_estimate``). A group larger than c is a chunk
   of its own; between such groups, each run of material W is split equally into
   ceil(W / c) chunks, cut at the group boundaries nearest k·W/n. No chunk below c/4
   is cut where a merge avoids it (ruling on #61, 1): a part of the split that small
   merges into its smaller neighbour, which then exceeds c by the small part it
   absorbs. c flexes by the small parts a chunk absorbs, never above B, the most the
   summariser reads in one call where the model table knows its window
   (``_summariser_bound``): c itself is at most B, and no merge or join makes a chunk
   above B; a part no neighbour takes within B is sent alone, below c/4, with an
   event. An oversized group is a chunk of its own whatever its size. After the cut,
   every chunk, kept and joined ones included, is checked against what the summariser
   can read in one call, and one it cannot read (an indivisible group, a chunk an
   earlier attempt cut) aborts the compaction visibly (D4); where the table does not
   know the window, the provider's own refusal is a failure of kind ``request``,
   visible and counted; a run below c/4 joins the chunk of an adjacent oversized
   group within B, or, at the end of the material, stays raw and the tail begins at
   it. c is 50k provider tokens, cut in the plugin's estimate (characters / 4) as
   50k / 1.51, #31's p50 of the provider's count over that estimate (``_chunk_limit``),
   and at most B. A chunk is frozen once its cut is
   recorded (#33 D14 as revised 2026-09-25, #31): a retry, in any process, keeps every
   recorded chunk of the session's unsettled attempts, found by its members'
   ``_row_id``s, whether or not its call was ever sent: a summarised one with its
   summary, any other retried as the same chunk. A kept chunk wins over the tail's
   start, and one that reaches the floor stays whole in the tail. Only the rest is
   split as above. A kept chunk is frozen, and a summarised one is never summarised
   again; the one exception is a run below c/4 standing before or between kept
   chunks, which joins one of them (D1), and goes on over contiguous neighbours until
   the chunk holds c/4, releasing a summary where it must, with an event. A chunk not
   found again, one without a summary below c/4 by today's estimate that a neighbour
   can now take within B (one no neighbour can take stays frozen), one without a
   summary above B that holds more than one group, and one with
   rows that came without a host identity (the gateway's replayed history, host
   scaffolding never persisted; the ask to Hermes is A1) are cut again, visibly: a
   warning and a store event. When only a rest below c/4 stands outside the tail,
   nothing is compacted (a no-op, not a failure); where such a rest would begin
   before the plugin's last summary (D3's shape), the compaction is aborted,
   unchanged, with an event. Steps 1 to 3 run in one planning transaction
   (``RecordStore.planning``: BEGIN IMMEDIATE, the attempt's captured check asked
   first inside it): every store read the cut depends on, the cut, and the write of
   the compaction, its inputs, the new records and every chunk, all before the first
   summariser call. A planner in another process waits for an earlier attempt's
   commit and reads its chunks.
4. Each chunk is summarised from its records as the store holds them. Every chunk's
   call is issued at once, on daemon workers of the plugin's own, through one limiter
   per endpoint and process; a call for the same records already in flight is joined,
   a summary of them already written is reused (``inflight``, #33). Each summary is
   written as a derivation when it arrives, in any order. The ``compress()`` thread
   waits, asking the attempt's captured check, and returns its input at once when the
   attempt is cancelled or superseded. A summary that cannot be written
   (``escalation.SummaryFailure``: no third level, nothing truncated) fails the
   compaction as a whole: the context stays as it was, and the host shows the cause.
   Each failure of a call is recorded once, with its kind, and judged where it is
   recorded, whether or not an attempt still waits for it. A chunk of the same
   members failing by its own fault (its
   reply rejected, or its request rejected by the provider) in three consecutive
   attempts is the chunk that keeps failing (#7): a ``chunk_keeps_failing`` event and
   a cause that name it, and the next occasion still retries it. The endpoint's,
   the route's and other failures are shown as such and do not count.
5. The return is emitted from the record: the host's system row in place, then the
   cover (the previous return's summaries, each re-emitted from its derivation, then
   the new ones in chunk order), then the tail as the host's own dicts. It is
   recorded exactly as returned.

There is no condensation in this path: the cover grows by one summary per chunk until
#34 builds condensation.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import queue
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from . import turn_signals
from .escalation import (
    OWN_FAILURE_KINDS,
    REASONING_EFFORTS,
    CallPath,
    CallSettings,
    SummariserRoute,
    SummaryFailure,
    _host_provider,
    configured_route_problem,
    failure_text,
    level_one_input,
    prompt_inputs,
    session_route,
    summarize_chunk,
)
from .model_table import lookup as lookup_model
from .handles import MESSAGE
from .summariser_input import image_count, summariser_message, wire_facts, wire_image_limit
from .message_analysis import _tool_call_id
from .record_store import RET_KEY, parse_ret_key, raw_json
from .fresh_tail import ToolPairingError, check_tool_pairing
from .inflight import (
    DEFAULT_CALLS_PER_ENDPOINT,
    ChunkCall,
    ChunkSummary,
    Outcome,
    Subscriber,
    endpoint_key,
    host_deadline,
    host_progress_hook,
    join_or_start,
    limiter_for,
)
from .record_write import _ATTEMPT, AttemptCancelled
from .tokens import Estimator, count_message_tokens, count_messages_tokens

logger = logging.getLogger(__name__)

# The words around a summary row (Decision 8, until #10 writes them). The wrapper carries
# what the plugin knows, the summary's node id, which lcm_expand takes; nothing in it is
# read out of the summary's text (#9, Decided: nothing in a reply is recognised by pattern).
_SUMMARY_HEADER = "[Recent Summary (d0, node {node_id})]"
_SUMMARY_FOOTER = "[Expand for details: node {node_id}]"

# How often the compress() thread, waiting for its chunks, asks the attempt's captured check.
_WAIT_SLICE_S = 0.25

# "The smallest run that stands alone", as a share of the chunk size c: a value for
# #22's table (#31, Decided; orchestrator ruling on #52). A smaller run next to an
# oversized group, or at the end of the material, joins that group's chunk.
_SMALLEST_STANDALONE_RUN = 0.25

# The share of its source a summary's budget asks for (today's prompt, until #10):
# min(max(2000, 0.20 × source), 12000). ρ in the tail's sizing (R9).
_SUMMARY_BUDGET_SHARE = 0.20
_SUMMARY_BUDGET_MIN = 2000
_SUMMARY_BUDGET_MAX = 12000

# "Keeps failing": a chunk of the same members failing in this many consecutive
# attempts is a visible error and a store event naming it (#33, Decided; #7).
_KEEPS_FAILING_ATTEMPTS = 3

# The record handle a row is sized with before it has one (``_cut_estimate``): a
# handle's fixed length, the kind letter and eight characters (``handles``).
_SIZING_HANDLE = MESSAGE + "a" * 8


try:  # the host's own test for its ephemeral recovery scaffolding (the nudge flags)
    from agent.session_persistence import _is_ephemeral_scaffolding as _host_scaffolding  # type: ignore
except Exception:  # pragma: no cover - older or absent host; its flags at 7b761da
    _SCAFFOLDING_FLAGS = ("_empty_recovery_synthetic", "_empty_terminal_sentinel", "_thinking_prefill",
                          "_verification_stop_synthetic", "_pre_verify_synthetic", "_kanban_stop_synthetic",
                          "_dropped_toolcall_nudge")

    def _host_scaffolding(message: Any) -> bool:
        return isinstance(message, dict) and any(message.get(flag) for flag in _SCAFFOLDING_FLAGS)


@dataclass(frozen=True)
class Occasion:
    """What an occasion is by the hooks and the list (#32 D1): ``kind`` one of
    manual, between, no_work, running, gap; ``raised`` whether τ′ applies; ``gap``
    whether the tail's floor reaches back to the newest tool group."""

    kind: str
    raised: bool
    gap: bool
    why: str


@dataclass
class ChunkInput:
    """One chunk as the summariser receives it, built once (``_chunk_input``)."""

    records: List[tuple]
    source: Any
    budget: int
    facts: Any
    wire: Any
    level_one: List[Dict[str, Any]]
    withheld: Optional[dict]


@dataclass
class _Step:
    """The step of the planning phase under way, for the cause of a failure the
    boundary catches (``_compress_impl``)."""

    at: str


@dataclass
class _Planned:
    """What the planning phase hands the dispatch: the recorded cut, each chunk's input,
    and the endpoint's limiter, progress hook and deadline."""

    route: Any
    chunks: List[List[int]]
    kept_state: Dict[tuple, str]
    mechanism: set
    tail_start: int
    endpoint: str
    limiter: Any
    limit: int
    still_wanted: Any
    describe: Any
    finished: Any
    # Per chunk: (number, chunk handle, positions, records, subscriber, the worker's run).
    calls: List[tuple]


def _passes_the_boundary(exc: BaseException) -> bool:
    """The exceptions the planning boundary lets through, to their current path: the
    attempt's own cancellation (``compress()`` returns its input), the host's explicit
    cancellation of an auxiliary attempt (``AuxiliaryExplicitCancellation``,
    agent/auxiliary_client.py:212 at origin/main d0288be5b3), and the interpreter's
    stops (KeyboardInterrupt, SystemExit, GeneratorExit)."""
    if isinstance(exc, (AttemptCancelled, KeyboardInterrupt, SystemExit, GeneratorExit)):
        return True
    try:
        from agent.auxiliary_client import AuxiliaryExplicitCancellation  # type: ignore
    except Exception:
        return False
    return isinstance(exc, AuxiliaryExplicitCancellation)


@dataclass
class TailPlan:
    """Where the tail begins and how it was sized (``_tail_plan``); with the chunks of
    an earlier attempt the cut keeps (``kept``) and those that reach the floor and stay
    whole in the tail (``held``), each as (``FrozenChunk``, positions); and those it
    does not keep after all (``recut``), each as (``FrozenChunk``, why)."""

    start: int
    label: str
    kept: List[tuple] = field(default_factory=list)
    held: List[tuple] = field(default_factory=list)
    recut: List[tuple] = field(default_factory=list)


@dataclass
class FrozenCandidates:
    """The earlier attempts' chunks found in this list (``found``, each as
    (``FrozenChunk``, positions)), those not found (``recut``, each as
    (``FrozenChunk``, why)), and the chunks with rows that came without a host identity
    (``unidentified``, each as (chunk, attempt, those member records); ruling 3 on #61)."""

    found: List[tuple] = field(default_factory=list)
    recut: List[tuple] = field(default_factory=list)
    unidentified: List[tuple] = field(default_factory=list)


def _is_steer(message: Any) -> bool:
    return isinstance(message, dict) and message.get("role") == "user" and message.get("display_kind") == "steer"


def _is_next_user_message(message: Any) -> bool:
    """The row a turn start ends with: a user row not yet persisted (no ``_row_id``),
    neither a steer row nor the host's flagged scaffolding (#32 §1)."""
    return (isinstance(message, dict) and message.get("role") == "user" and "_row_id" not in message
            and not _is_steer(message) and not _host_scaffolding(message))


def _ends_with_tool_results(messages: List[Dict[str, Any]]) -> bool:
    """The list ends with tool results, or with steer rows after them (#32 §1)."""
    at = len(messages) - 1
    while at >= 0 and _is_steer(messages[at]):
        at -= 1
    return at >= 0 and isinstance(messages[at], dict) and messages[at].get("role") == "tool"


def _unanswered_calls_at(messages: List[Dict[str, Any]], boundary: int) -> Optional[tuple]:
    """(position, call ids) of an assistant row at the boundary, the last row before it
    or the first row after it, whose tool calls have no result anywhere in the list;
    None where there is none."""
    answered = {str(m.get("tool_call_id") or "").strip()
                for m in messages if isinstance(m, dict) and m.get("role") == "tool"}
    for position in (boundary - 1, boundary):
        if position < 0 or position >= len(messages) or not isinstance(messages[position], dict):
            continue
        message = messages[position]
        if message.get("role") != "assistant":
            continue
        missing = sorted({_tool_call_id(call) for call in (message.get("tool_calls") or [])} - {""} - answered)
        if missing:
            return position, missing
    return None


def _pairs_crossing(messages: List[Dict[str, Any]], boundary: int) -> Optional[tuple]:
    """(call id, position of its call, position of its result) of a pair the boundary
    separates: a call outside (before the boundary) whose result stands inside, or a
    result outside whose call stands inside; None where there is none. Matched by
    ``tool_call_id``, whether or not the result of a call outside is anywhere else."""
    made: Dict[str, int] = {}
    for position, message in enumerate(messages):
        if isinstance(message, dict) and message.get("role") == "assistant":
            for call in message.get("tool_calls") or []:
                call_id = _tool_call_id(call)
                if call_id:
                    made.setdefault(call_id, position)
    for position, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") != "tool":
            continue
        call_id = str(message.get("tool_call_id") or "").strip()
        call_at = made.get(call_id)
        if call_at is not None and (call_at < boundary) != (position < boundary):
            return call_id, call_at, position
    return None


def _fewest_chunks(sizes: List[int], limit: int, images: Optional[List[int]] = None,
                   image_limit: Optional[int] = None) -> int:
    """The fewest chunks of at most ``limit`` tokens, and of at most ``image_limit``
    images where one is given, that cover ``sizes`` in order, every group within both:
    filling each chunk as far as it goes is optimal (a shorter piece of a chunk that fits
    fits too)."""
    images = images or [0] * len(sizes)
    count, used, pictured = 0, 0, 0
    for size, pictures in zip(sizes, images):
        if count == 0 or used + size > limit or (image_limit is not None and pictured + pictures > image_limit):
            count, used, pictured = count + 1, size, pictures
        else:
            used += size
            pictured += pictures
    return count


def _smallest_for(limit: int) -> int:
    """c/4 for a chunk size c: the smallest run that stands alone (#31). Counts are
    whole tokens, so it is rounded up: a run of 8,278 is below 33,113 / 4 = 8,278.25.
    One value for the cut, the preflight and the gateway's probe (ruling on #61, 1)."""
    return max(1, math.ceil(limit * _SMALLEST_STANDALONE_RUN))


def _split_run(sizes: List[int], limit: int, bound: Optional[int] = None,
               alone: Optional[List[tuple]] = None, images: Optional[List[int]] = None,
               image_limit: Optional[int] = None) -> List[int]:
    """The equal split of one run of groups, each at most ``limit`` (#31): n chunks,
    n = ceil(W / limit) for the run's weight W, or more where the groups cannot be
    packed into that many without a chunk above ``limit``; each cut at the group
    boundary nearest k·W/n that keeps every chunk within ``limit`` and leaves a rest the
    remaining chunks can hold. Returns the group index at which each chunk after the
    first begins.

    No part is below c/4 (ruling on #61, 1): a chunk that small cannot have a shorter
    summary and fails on every attempt. Where the groups leave no other way, as
    [c, 1, c], such a part merges into its smaller neighbour (the following one on a
    tie), or the other where the smaller one would make a chunk above ``bound``, the
    most the summariser reads in one call (B, ``_summariser_bound``; the orchestrator's
    ruling on the Codex review of c2efe0e: every merge is bounded by B). c flexes by
    the small parts a chunk absorbs (orchestrator ruling on #61; it amends #12's "never
    above c"), never above B. Where neither neighbour takes it within B, the part
    stands alone, below c/4, and is named in ``alone`` as (first group, end group,
    weight): the caller records it. The equal cut always finds a boundary: n is at
    least the fewest chunks that cover the run, and every group of a run is at most
    ``limit``.

    ``images`` (per group) and ``image_limit`` (``wire_image_limit``) bound each chunk's
    images the same way (the orchestrator's ruling on the Codex review of 6f4a351): n is
    at least ceil(images / image_limit) and the fewest chunks within both bounds, every
    cut keeps its chunk within both, and no merge makes a chunk above ``image_limit``.
    Every group of a run is within ``image_limit``."""
    images = images or [0] * len(sizes)
    cuts = _equal_cuts(sizes, limit, images, image_limit)
    if not cuts:
        return cuts
    smallest = _smallest_for(limit)
    starts = [0] + cuts + [len(sizes)]
    standing: set = set()   # the first group of each part that stands alone
    while len(starts) > 2:
        weights = [sum(sizes[a:b]) for a, b in zip(starts, starts[1:])]
        pictures = [sum(images[a:b]) for a, b in zip(starts, starts[1:])]
        small = next((i for i, weight in enumerate(weights) if weight < smallest and starts[i] not in standing),
                     None)
        if small is None:
            break
        sides = sorted((side for side in (small - 1, small + 1) if 0 <= side < len(weights)),
                       key=lambda side: (weights[side], 0 if side > small else 1))
        fitting = [side for side in sides
                   if (bound is None or weights[side] + weights[small] <= bound)
                   and (image_limit is None or pictures[side] + pictures[small] <= image_limit)]
        if not fitting:
            standing.add(starts[small])
            continue
        # Removing the boundary between the part and its neighbour merges them.
        del starts[max(small, fitting[0])]
    if alone is not None:
        for a, b in zip(starts, starts[1:]):
            if sum(sizes[a:b]) < smallest:
                alone.append((a, b, sum(sizes[a:b])))
    return starts[1:-1]


def _equal_cuts(sizes: List[int], limit: int, images: Optional[List[int]] = None,
                image_limit: Optional[int] = None) -> List[int]:
    images = images or [0] * len(sizes)
    total, pictured = sum(sizes), sum(images)
    within_images = image_limit is None or pictured <= image_limit
    if total <= limit and within_images:
        return []
    n = max(-(-total // limit), _fewest_chunks(sizes, limit, images, image_limit))
    if image_limit is not None:
        n = max(n, -(-pictured // image_limit))
    prefix, image_prefix = [0], [0]
    for size, count in zip(sizes, images):
        prefix.append(prefix[-1] + size)
        image_prefix.append(image_prefix[-1] + count)
    fewest_from = [_fewest_chunks(sizes[j:], limit, images[j:], image_limit) for j in range(len(sizes))]
    cuts: List[int] = []
    previous = 0
    for k in range(1, n):
        target = k * total / n
        best, best_distance = None, None
        for j in range(previous + 1, len(sizes)):
            if prefix[j] - prefix[previous] > limit:
                break
            if image_limit is not None and image_prefix[j] - image_prefix[previous] > image_limit:
                break
            if fewest_from[j] > n - k:
                continue
            distance = abs(prefix[j] - target)
            if best is None or distance < best_distance:
                best, best_distance = j, distance
        cuts.append(best)
        previous = best
    return cuts


class CompactionMixin:
    def should_compress(self, prompt_tokens: int = None) -> bool:
        """The host's count against τ, or τ′ while a turn runs (#32 D1, D2). No side
        effects: the host asks several times per occasion.

        The host hands one integer and no list, so "a turn runs" is read from the hooks
        alone: ``pre_llm_call`` opened a turn for this session, and the last of its
        events was a tool that ran (``post_tool_call``), so the list ends with tool
        results (#32 §3). After a response of the turn (``post_api_request``), as at the
        gate after a host nudge, or before any tool round, τ applies. ``compress()``
        classifies again with the list."""
        if self._geometry is None:
            return False  # R11: nothing is compacted; the error was recorded where W was set
        tokens = prompt_tokens if prompt_tokens is not None else self.last_prompt_tokens
        if self._should_force_overflow_recovery(observed_tokens=tokens):
            return True
        return tokens >= self._tau(raised=self._turn_state().running())

    def _provider_estimate(self, estimate: int) -> tuple[int, str]:
        """An estimate turned into provider tokens where the plugin must decide from its
        own count, as for c (#31's p50), with its label."""
        ratio = self._config.estimate_ratio
        return int(estimate * ratio), (f"{estimate} by the plugin's estimate (characters / 4) × {ratio}, #31's "
                                       f"p50 of the provider's count over it, = {int(estimate * ratio)} provider tokens")

    def should_compress_preflight(self, messages):
        """Before a request, at turn start (the host asks it only there, after
        ``should_compress`` declined): settle and bind what the list shows, then ask
        for a compaction when the prompt reaches the occasion's threshold and the
        material outside the tail holds a run that stands alone (c/4). No count of the
        host's exists here, so the plugin's estimate decides, converted into provider
        tokens by #31's ratio and labelled. Written to the store: only the settling and
        binding. Changed in memory besides: the hook state of the session's turn, which
        the occasion's classification marks contested at a gap (#32 D1)."""
        self._bind_from_list(messages)
        if self._geometry is None:
            return False
        rough = count_messages_tokens(messages, self._estimator())
        if self._should_force_overflow_recovery(observed_tokens=rough, messages=messages):
            return self._mark_preflight_compression_requested()
        occasion = self._classify_occasion(messages, force=False)
        threshold = self._tau(raised=occasion.raised)
        provider_rough, rough_label = self._provider_estimate(rough)
        if provider_rough >= threshold:
            try:
                material = self._material_outside_tail(messages, occasion)
            except ToolPairingError:
                # compress() records the error and shows it; the preflight asks for it.
                return self._mark_preflight_compression_requested()
            except Exception:
                # Sizing failed (``_sizing_failed``): compress() aborts with the cause and
                # shows it; the preflight asks for it rather than raise into the host.
                logger.warning("LCM preflight could not size the material outside the tail", exc_info=True)
                return self._mark_preflight_compression_requested()
            smallest = self._smallest_run()
            if material.tokens >= smallest:
                return self._mark_preflight_compression_requested()
            reason = (f"the material outside the fresh tail is below the smallest run that stands alone, c/4, "
                      f"by the plugin's estimate ({material.tokens} < {smallest} tokens, {material.label()}; "
                      f"{self._chunk_label()})")
            self._last_compression_status = "noop"
            self._last_compression_noop_reason = reason
            logger.info("LCM preflight compression no-op: %s", reason)
        else:
            logger.debug("LCM preflight: %s is below %s %d (%s)", rough_label,
                         "τ′" if occasion.raised else "τ", threshold, occasion.why)
        return False

    def _material_outside_tail(self, messages: List[Dict[str, Any]], occasion: Occasion):
        """The estimate of what stands outside the fresh tail and the mechanism's layer,
        with the frozen cut, found by id only; nothing is written. Counted in the cut's
        unit (``_cut_estimate``), since it is compared with c/4 as the cut compares it."""
        mechanism = self._mechanism_positions(messages)
        tail_start = self._tail_start(messages, mechanism, occasion)
        estimator, convert = self._cut_estimate()
        return estimator.messages([convert(messages[index]) for index in range(tail_start)
                                   if index not in mechanism])

    def has_content_to_compress(self, messages: List[Dict[str, Any]]) -> bool:
        """The host's probe before the gateway's /compress (``ContextEngine``,
        agent/context_engine.py 176; asked only on ``skip_without_window``,
        agent/conversation_compression_manual.py 104-106): False makes the host say
        "Nothing to compress yet." without calling ``compress()``. False only where
        something stands outside the fresh tail and all of it is below c/4, which
        ``compress()`` leaves raw (ruling on #61, 1); where nothing stands outside it,
        ``compress()`` says why itself. Nothing is written."""
        if self._geometry is None or not messages:
            return True
        try:
            material = self._material_outside_tail(messages, self._classify_occasion(messages, force=True))
        except ToolPairingError:
            return True   # compress() records the error and shows it
        except Exception:
            logger.warning("LCM could not tell whether there is anything to compact", exc_info=True)
            return True
        return not (0 < material.tokens < self._smallest_run())

    def _classify_occasion(self, messages: List[Dict[str, Any]], *, force: bool) -> Occasion:
        """The occasion by the hooks and the list's structure, where both agree (#32
        D1): between turns, a turn with no work in progress, a running turn (τ′), or a
        gap. At a gap the plugin does not guess: τ applies, nothing is re-inserted,
        and the tail's floor reaches back to the newest tool group. Rows are read by
        structure and host fields only, never by text."""
        if force:
            return Occasion("manual", False, False, "the host's /compress (force): between turns, τ (R15)")
        state = self._turn_state()
        last = messages[-1] if messages and isinstance(messages[-1], dict) else None
        if not state.in_turn:
            if _is_next_user_message(last):
                return Occasion("between", False, False, "between turns: the list ends with the next user message")
            return Occasion("gap", False, True, "a gap: the hooks say between turns, but the list does not end "
                                                "with the next user message (the gateway's hygiene path, or "
                                                "another host occasion)")
        opening = state.opening_row_id()
        if opening is not None and last is not None and last.get("_row_id") == opening:
            return Occasion("no_work", False, False, "in a turn with no work in progress: the list ends with the "
                                                     "turn's opening row")
        if _ends_with_tool_results(messages):
            return Occasion("running", True, False, "in a running turn: the list ends with tool results")
        # A gap while the hooks say a turn runs: the turn is contested until the hooks
        # settle it, so that what the plugin publishes without a list (threshold_tokens,
        # should_compress) is τ too, as D1 wants at the gap. The review fork, which runs
        # under its parent's id, never marks the parent's turn. Only the current attempt
        # writes the turn's state (#33 D12; the Codex review of c2efe0e): an attempt the
        # host cancelled, or one a newer attempt replaced, whose worker resumes after the
        # next turn started would otherwise contest that turn. The preflight runs on the
        # host's thread with no attempt, and writes.
        attempt = _ATTEMPT.get()
        if not getattr(self, "_review_fork", False) and self._live_write_allowed(attempt):
            turn_signals.turn_start_seen(self._session_id, state.turn_id)
        if _is_next_user_message(last):
            return Occasion("gap", False, True, "a gap: a turn start while the hooks still say a turn runs (an "
                                                "exit that skipped on_turn_complete)")
        return Occasion("gap", False, True, "a gap: in a turn, the list ends with neither tool results nor the "
                                            "turn's opening row (a host nudge, or another row the host inserts)")

    def _mechanism_positions(self, messages: List[Dict[str, Any]]) -> set:
        """The entries of a list that are the mechanism's layer, by identity: the host's
        system row, and every summary of the session's effective return, known by the
        plugin's key or by its binding. Read only; the preflight's estimate."""
        positions: set = set()
        if messages and isinstance(messages[0], dict) and messages[0].get("role") == "system":
            positions.add(0)
        if not self._plugin_session:
            return positions
        store = self._records
        effective = store.effective_compaction(self._plugin_session)
        if effective is None:
            return positions
        summaries = {p for p, entry in store.return_entries(effective).items() if entry[0] == "summary"}
        bound = store.bound_rows(effective)
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                continue
            key = parse_ret_key(message.get(RET_KEY))
            row_id = message.get("_row_id")
            if key is not None and key[0] == effective and key[1] in summaries:
                positions.add(index)
            elif isinstance(row_id, int) and bound.get(row_id) in summaries:
                positions.add(index)
        return positions

    def _chunk_limit(self, focus_topic: str = "") -> int:
        """c in the unit the plugin cuts in, its estimate (characters / 4, #21).

        #31 decides c = 50k provider tokens (``chunk_tokens``). The plugin cannot count
        provider tokens; it counts by its estimate, and #31 measured the provider's
        count over that estimate on Claude tool results at p50 1.51 (p95 1.95, p99 2.37,
        n = 201; ``estimate_ratio``). #12 states the chunk accordingly: "50k provider
        tokens is about 33k by the estimate". So c is cut at 50,000 / 1.51 = 33,112 by
        the estimate: a chunk of typical text is then about 50k to the provider, and
        one at the p99 error about 78k.

        Only the summariser bounds the chunk (#12): where the model table knows the
        summariser's window, c is at most what it can read in one call, by the
        estimate's worst case (#34 D4), ``_summariser_bound``; the cut uses that
        effective c everywhere it uses c (the split, c/4, the joins). ``focus_topic`` is
        the attempt's focus text, which the summariser's prompt carries (B)."""
        return self._chunk_size(focus_topic)[0]

    def _chunk_size(self, focus_topic: str = "") -> tuple[int, str]:
        """The effective c by the estimate and its label (the orchestrator's rulings on
        D4, in the pre-review of 6a6a7f8 and on the Codex review of c2efe0e):
        c_eff = min(c, B), B the summariser's room in the estimate's unit
        (``_summariser_bound``). Every chunk the cut makes of divisible material is at
        most B: a part of the split is at most c_eff, and a part or run below c/4 merges
        only where the merge stays within B (``_split_run``, ``_cut_chunks``). Only an
        indivisible group larger than the room aborts, the ruling's other clause. Where
        the table does not know the window, c stays as #31 decides."""
        config = self._config
        c = max(1, int(config.chunk_tokens / config.estimate_ratio))
        label = (f"c = {c} tokens by the plugin's estimate: {config.chunk_tokens} provider tokens / "
                 f"{config.estimate_ratio}, #31's p50 of the provider's count over characters / 4")
        bound, bound_label = self._summariser_bound(focus_topic)
        if bound is None or bound >= c:
            return c, label
        return bound, f"c = {bound} tokens by the plugin's estimate, below #31's {c}: B, {bound_label}"

    def _cut_estimate(self) -> tuple[Estimator, Callable[[Any], Any]]:
        """The unit the cut sizes material in (the orchestrator's ruling on the Codex
        review of c2efe0e): each row as the summariser receives it, by
        ``summariser_message`` for the summariser's route (readable reasoning as the text
        part it becomes, the reasoning field its provider is sent, an image it is not
        sent as its placeholder, the record's handle at its fixed length), counted by the
        summariser's estimate. c, c/4 and B are then in the unit the check against the
        summariser's window counts in (D4), so a chunk the cut bounds by B fits. The tail
        and G stay on the session's estimate: they are the session's context. Returns the
        estimator and the conversion of one row. Where there is no summariser route, the
        session's estimate of the row as it stands: ``compress()`` aborts then and says
        why. The host's per-request image eviction on the Anthropic converter, which only
        lowers a chunk's input, is not counted."""
        route, _why = self._summariser_route()
        if route is None:
            return self._estimator(), lambda message: message
        _facts, wire = self._summariser_wire(route)
        estimator = Estimator(image_model=route.model, image_provider=route.table_provider(),
                              reasoning_sent=wire.needs_reasoning_echo)

        def convert(message: Any) -> Any:
            return summariser_message(message, _SIZING_HANDLE, wire) if isinstance(message, dict) else message
        return estimator, convert

    def _summariser_bound(self, focus_topic: str = "") -> tuple[Optional[int], str]:
        """B, the most one chunk may hold by the plugin's estimate so that the summariser
        can read it in one call: its window less its output cap and less its prompt,
        divided by the estimate's worst case observed (#34 D4); None where the model table
        does not know the window or there is no summariser route. The prompt is the
        largest a call of this attempt can carry (the ruling on the Codex review of
        188fb8b): the larger of the two levels' inputs without records, at the largest
        budget, with the custom instructions and the attempt's focus text at their actual
        length (none where the attempt has none: the preflight and the host's automatic
        compaction pass none). Reads nothing of the store."""
        route, _why = self._summariser_route()
        if route is None:
            return None, ""
        facts, wire = self._summariser_wire(route)
        if facts is None or not facts.context_window:
            return None, ""
        worst = float(self._config.estimate_ratio_max)
        room = facts.context_window - (facts.output_cap or 0)
        estimator = Estimator(image_model=route.model, image_provider=route.table_provider(),
                              reasoning_sent=wire.needs_reasoning_echo)
        prompt = max((estimator.messages(inputs) for inputs in prompt_inputs(
            _SUMMARY_BUDGET_MAX, facts=wire, focus_topic=focus_topic or "",
            custom_instructions=self._config.custom_instructions)), key=lambda estimate: estimate.tokens)
        bound = int((room - prompt.in_provider_tokens(worst)) / worst)
        if bound <= 0:
            return None, ""
        return bound, (f"the summariser {route.describe()} reads {room} provider tokens ({facts.context_window} "
                       f"window less {facts.output_cap or 0} output; {facts.basis}), less its prompt of "
                       f"{prompt.tokens}{' with the focus text' if focus_topic else ''}, over {worst}, the "
                       f"estimate's worst case (#34 D4): {bound} by the estimate")

    def _image_limit(self) -> Optional[int]:
        """The most images one chunk may carry to the summariser (``wire_image_limit``),
        or None: none where there is no summariser route (``compress()`` aborts then)."""
        route, _why = self._summariser_route()
        if route is None:
            return None
        return wire_image_limit(self._summariser_wire(route)[1])

    def _smallest_run(self, focus_topic: str = "") -> int:
        """c/4 in the estimate's unit: the smallest run that stands alone (#31), the
        same value the cut uses (``_smallest_for``)."""
        return _smallest_for(self._chunk_limit(focus_topic))

    def _chunk_label(self, focus_topic: str = "") -> str:
        return self._chunk_size(focus_topic)[1]

    @staticmethod
    def _tail_floor(messages: List[Dict[str, Any]], mechanism: set, groups: List[List[int]],
                    occasion: Optional[Occasion] = None) -> int:
        """The least the tail keeps (#31, #32 D1), always the start of one of ``groups``
        (the cut's grouping of every entry outside the mechanism's layer): the group of
        the newest message; where the list ends with steer rows, the group of the row
        before them; at a gap, the newest tool group and everything after it. Never
        before the last entry of the mechanism's layer; a group that stands across it
        is a ``ToolPairingError``."""
        first = max(mechanism) + 1 if mechanism else 0
        group_of = {index: group for group in groups for index in group}
        at = len(messages) - 1
        while at > 0 and _is_steer(messages[at]):
            at -= 1
        if _is_steer(messages[at]):
            at = len(messages) - 1
        start = group_of[at][0] if at in group_of else at
        if occasion is not None and occasion.gap:
            tools = [index for index in group_of
                     if isinstance(messages[index], dict) and messages[index].get("role") == "tool"]
            if tools:
                start = min(start, group_of[max(tools)][0])
        start = max(start, first)
        group = group_of.get(start)
        if group is not None and group[0] != start:
            raise ToolPairingError(f"the tool group from position {group[0]} to {group[-1]} stands across "
                                   f"the summaries the plugin returned")
        return start

    def _fixed_prefix(self) -> tuple[int, str]:
        """F, the fixed prefix in provider tokens (R10): the session's latest measurement
        (``_measure_fixed_prefix``: the one taken with the smallest list), labelled with
        its list and error bound; until the first, the configured hypothesis (32k).

        A store that cannot be read is not "no measurement yet": the error is raised,
        never replaced by the hypothesis (#7), and ``compress()`` aborts with it."""
        fact = self._fixed_prefix_fact()
        if fact is not None:
            return int(fact["F"]), (f"F {fact['F']} (measured with a list of {fact.get('list_estimate')} by the "
                                    f"estimate; up to {fact.get('error_bound')} too high at #31's p99)")
        hypothesis = int(self._config.fixed_prefix_hypothesis_tokens)
        return hypothesis, f"F {hypothesis} (a hypothesis until a response measures it, R10)"

    def _fixed_prefix_label(self) -> str:
        """F as the status views show it; a store that cannot be read is said so."""
        try:
            return self._fixed_prefix()[1]
        except Exception as exc:
            return f"F unreadable: the store could not be read ({type(exc).__name__}: {exc})"

    def _frozen_candidates(self, messages: List[Dict[str, Any]],
                           records: Optional[Dict[int, str]] = None) -> FrozenCandidates:
        """The chunks of the session's unconfirmed attempts a retry keeps (#33 D14),
        found in this list by their members' ``_row_id``s, as (``FrozenChunk``, the
        members' positions). A chunk is found only where every member stands in the
        list, in its order; with ``records`` (the classification's position -> record
        for rows the store already holds) each position must hold that very record,
        so that a row the host rewrote since makes a different chunk. Where no id
        matches, as after a commit by another path, nothing is kept. A chunk that is
        not found, and one with rows that came without a host identity, are re-cut: the
        caller says so visibly (ruling 3 on #61). Nothing is matched by content.

        Every recorded chunk of an unsettled attempt is a candidate, whether or not its
        call was ever sent (#33 D14 as revised): ``compress()`` reads it inside its
        planning transaction, so a chunk an attempt in another process recorded before
        is always seen; a call still in flight here is joined through the registry
        (D12)."""
        result = FrozenCandidates()
        if not self._plugin_session:
            return result
        store = self._records
        frozen, result.unidentified = store.frozen_chunks(
            self._plugin_session, store.effective_compaction(self._plugin_session))
        if not frozen:
            return result
        position_of = {message["_row_id"]: index for index, message in enumerate(messages)
                       if isinstance(message, dict) and isinstance(message.get("_row_id"), int)}
        for chunk in frozen:
            positions = [position_of.get(row_id) for _record, row_id in chunk.members]
            if any(position is None for position in positions):
                result.recut.append((chunk, "a member's host id is not in this list"))
            elif positions != sorted(positions):
                result.recut.append((chunk, "its members' ids stand in this list out of order"))
            elif records is not None and any(records.get(p) != r for p, r in zip(positions, chunk.records)):
                result.recut.append((chunk, "a member row was rewritten since"))
            else:
                result.found.append((chunk, positions))
        result.found.sort(key=lambda item: item[1][0])
        return result

    def _tail_plan(self, messages: List[Dict[str, Any]], mechanism: set,
                   occasion: Optional[Occasion] = None, frozen: Sequence[tuple] = (),
                   fixed: Optional[tuple] = None, focus_topic: str = "") -> "TailPlan":
        """Where the tail begins, and how it was sized (#13, #31). ``focus_topic`` is the
        attempt's focus text, which B counts (``_summariser_bound``).

        The tail takes what the target leaves: after the compaction the context is
        F + S + (the new summaries) + t + R_in, and it should come to G. In the plugin's
        estimate, with k the provider's count over it (#31's p50, ``estimate_ratio``)
        and ρ the share the summariser's prompt asks for (budget over source):

            F + k·S + ρ·k·(X − t) + k·t + R_in = G
            t = (G − F − k·S − R_in − ρ·k·X) / (k·(1 − ρ))

        X is every entry that is not the mechanism's layer, S the summaries the list
        already holds, F per R10, R_in 0 until #14 re-inserts the instruction.

        Every boundary is the start of a group of one grouping, the cut's
        (``_groups``), taken once over every entry outside the mechanism's layer, after
        the pairing is checked over the whole list (``check_tool_pairing``). The walk
        goes back from the floor, whole groups only, while the sum stays within t. The
        floor is D1's; the ceiling leaves the oldest group outside, so that something
        is chunked. The floor wins over the ceiling. The room for #34: condensation
        shrinks S before this is sized, and T_def is its guarantee on t; neither is
        built here.

        On a retry the cut is frozen (#31, #33 D14): the chunks of ``frozen``
        (``_frozen_candidates``) that are whole groups over consecutive entries outside
        the mechanism's layer are kept. Where the tail would begin before the end of a
        kept chunk, the kept chunk wins and the tail begins after it. A kept chunk that
        reaches the floor is never cut: it stays whole in the tail this time (``held``),
        and the tail begins at it. A candidate without a summary below c/4 by today's
        estimate (the same records give the same estimate, so only where c or the
        estimate's ratio changed since it was cut, or where the cut sent it alone
        because no neighbour took it within B) is cut again (``recut``; ruling on #61,
        1) only where a neighbour can now take it within B (``neighbour_takes``); one no
        neighbour can take stays frozen like every recorded chunk, so its failures count
        and no duplicate chunk is recorded (the orchestrator's ruling on 6736c1d). This
        is one departure from "a recorded chunk is frozen"; a multi-group one above B is
        the other (ruled on #33). A summarised
        one is kept whatever its size: it is not sent again, its summary is reused."""
        check_tool_pairing(messages)
        first = max(mechanism) + 1 if mechanism else 0
        outside = [i for i in range(len(messages)) if i not in mechanism]
        groups = self._groups(messages, outside)
        floor = self._tail_floor(messages, mechanism, groups, occasion)
        if self._geometry is None:
            return TailPlan(floor, "no geometry: the floor only")
        estimator = self._estimator()
        sizes = [estimator.message(message).tokens if isinstance(message, dict) else 0 for message in messages]
        system = {0} if messages and isinstance(messages[0], dict) and messages[0].get("role") == "system" else set()
        summaries = sum(sizes[i] for i in mechanism if i not in system)
        total = sum(sizes[i] for i in outside)
        k, rho = float(self._config.estimate_ratio), _SUMMARY_BUDGET_SHARE
        # F as the caller read it before its planning transaction, else read here.
        fixed, fixed_label = fixed if fixed is not None else self._fixed_prefix()
        reinserted = 0  # R_in, until #14
        target = self._geometry.target
        t = (target - fixed - k * summaries - reinserted - rho * k * total) / (k * (1 - rho))
        start = floor
        used = sum(sizes[floor:])
        for group in reversed([group for group in groups if first <= group[0] < floor]):
            weight = sum(sizes[i] for i in group)
            if used + weight > t:
                break
            start, used = group[0], used + weight
        ceiling = groups[1][0] if len(groups) >= 2 else len(messages)
        if start < ceiling:
            start = min(ceiling, floor)
        sized = start

        # The frozen cut (#31, #33 D14).
        group_of = {index: group for group in groups for index in group}
        mechanism_set = set(mechanism)
        kept: List[tuple] = []
        held: List[tuple] = []
        recut: List[tuple] = []
        claimed: set = set()
        smallest = self._smallest_run(focus_topic)
        bound = self._summariser_bound(focus_topic)[0] if frozen else None
        if frozen:
            # A kept chunk is weighed as the cut weighs it (``_cut_estimate``).
            cut_estimator, convert = self._cut_estimate()
            cut_size: Dict[int, int] = {}

            def cut_weight(indices) -> int:
                for i in indices:
                    if i not in cut_size:
                        cut_size[i] = (cut_estimator.message(convert(messages[i])).tokens
                                       if isinstance(messages[i], dict) else 0)
                return sum(cut_size[i] for i in indices)

            cut_picture: Dict[int, int] = {}

            def cut_images(indices) -> int:
                for i in indices:
                    if i not in cut_picture:
                        cut_picture[i] = image_count(convert(messages[i])) if isinstance(messages[i], dict) else 0
                return sum(cut_picture[i] for i in indices)

            limit = self._chunk_limit(focus_topic)
            image_limit = self._image_limit()
            owner = {i: which for which, (_chunk, positions) in enumerate(frozen) for i in positions}

        def within(tokens: int, images: int) -> bool:
            return (bound is None or tokens <= bound) and (image_limit is None or images <= image_limit)

        def neighbour_takes(which: int, positions: List[int], weight: int) -> Optional[str]:
            """Whether a neighbour can take a candidate below c/4 within B and the wire's
            image limit, as the cut would (``_cut_chunks``), and which; None where none
            can (the orchestrator's ruling on 6736c1d: such a chunk, sent alone, stays
            frozen). The neighbours are the groups adjacent to it outside the mechanism's
            layer and before the floor: another candidate is taken as one chunk of its
            weight, a group above c or above the image limit as an oversized chunk, any
            other group as part of a run, which is split again and so always takes it.
            Without B and an image limit nothing bounds a merge."""
            if bound is None and image_limit is None:
                return "any neighbour (the summariser's window is not known, so no bound applies)"
            pictured = cut_images(positions)
            before = next((i for i in range(positions[0] - 1, first - 1, -1) if i not in mechanism_set), None)
            after = next((i for i in range(positions[-1] + 1, floor) if i not in mechanism_set), None)
            for side, index in (("following", after), ("preceding", before)):
                if index is None or index not in group_of:
                    continue
                other = owner.get(index)
                if other is not None and other != which:
                    other_weight = cut_weight(frozen[other][1])
                    if within(weight + other_weight, pictured + cut_images(frozen[other][1])):
                        return f"the {side} recorded chunk ({other_weight} tokens)"
                    continue
                group_weight, group_images = cut_weight(group_of[index]), cut_images(group_of[index])
                if group_weight > limit or (image_limit is not None and group_images > image_limit):
                    if within(weight + group_weight, pictured + group_images):
                        return f"the {side} oversized group ({group_weight} tokens)"
                    continue
                return f"the {side} run"
            return None

        for which, (chunk, positions) in enumerate(frozen):
            low, high = positions[0], positions[-1]
            whole = (low >= first and low in group_of and group_of[low][0] == low and high in group_of
                     and group_of[high][-1] == high
                     and positions == [i for i in range(low, high + 1) if i not in mechanism_set])
            if not whole or claimed.intersection(positions):
                recut.append((chunk, "its members are no longer whole groups over consecutive entries of this list"))
                continue
            weight = cut_weight(positions)
            taker = neighbour_takes(which, positions, weight) if weight < smallest else None
            if weight < smallest and chunk.state != "summarised" and taker is not None:
                # Below c/4 and a neighbour can now take it within B: cut again, so that it
                # joins. One no neighbour can take within B was sent alone by the cut and
                # stays frozen, its failures counted (the orchestrator's ruling on 6736c1d).
                recut.append((chunk, f"it holds {weight} tokens by today's estimate, below c/4 = {smallest}, has "
                                     f"no summary, and {taker} can take it within B"))
                continue
            if (bound is not None and weight > bound and chunk.state != "summarised"
                    and len({tuple(group_of[i]) for i in positions}) > 1):
                # Cut before the summariser's bound applied (another summariser, another
                # c): kept, it would be refused on every attempt; its groups can be cut
                # smaller, so it is cut again, as a chunk below c/4 is.
                recut.append((chunk, f"it holds {weight} tokens by today's estimate, more than the summariser "
                                     f"reads ({bound}), has no summary, and its groups can be cut smaller"))
                continue
            if (image_limit is not None and chunk.state != "summarised"
                    and len({tuple(group_of[i]) for i in positions}) > 1
                    and cut_images(positions) > image_limit):
                # Its images pass what the wire's converter keeps; its groups can be cut
                # within it, as within B (the ruling on the Codex review of 6f4a351).
                recut.append((chunk, f"it carries {cut_images(positions)} images, more than the {image_limit} the "
                                     f"summariser's wire keeps in one request, has no summary, and its groups can "
                                     f"be cut within it"))
                continue
            claimed.update(positions)
            (kept if high < floor else held).append((chunk, positions))
        if kept:
            start = max(start, kept[-1][1][-1] + 1)
        if held:
            start = min(start, held[0][1][0])
        used = sum(sizes[start:])
        label = (f"t {max(0, int(t))} by the estimate ({max(0, int(t * k))} provider tokens): G {target} − "
                 f"{fixed_label} − {k}·S {summaries} − R_in {reinserted} − {rho}·{k}·X {total}, over {k}·(1 − {rho}); "
                 f"the tail holds {used} by the estimate from position {start}")
        if kept and start != sized:
            label += (f" (sized to begin at {sized}; {len(kept)} chunk{'' if len(kept) == 1 else 's'} kept from an "
                      f"earlier attempt end{'s' if len(kept) == 1 else ''} at {kept[-1][1][-1]}: the kept chunk wins)")
        if held:
            label += (f"; {len(held)} kept chunk{'' if len(held) == 1 else 's'} reach{'es' if len(held) == 1 else ''} "
                      f"the floor at {floor} and stay whole in the tail")
        return TailPlan(start, label, kept=kept, held=held, recut=recut)

    def _tail_start(self, messages: List[Dict[str, Any]], mechanism: set,
                    occasion: Optional[Occasion] = None) -> int:
        """The tail's start for an estimate that writes nothing (the preflight, the
        gateway's probe), the frozen cut included; the members are found by id only
        (nothing is classified here), and what is re-cut is recorded only by
        ``compress()``."""
        try:
            frozen = self._frozen_candidates(messages).found
        except Exception:
            logger.warning("LCM preflight could not read the earlier attempts' chunks", exc_info=True)
            frozen = []
        return self._tail_plan(messages, mechanism, occasion, frozen).start

    def _unchanged_return(self, messages: List[Dict[str, Any]], reason: str) -> List[Dict[str, Any]]:
        """No compaction: the host's own list, the same object with the same dicts.

        Nothing is re-assembled, sanitised or dropped: when no compaction happens the
        context stays untouched. The host then logs "made no progress" and commits
        nothing (``_candidate_rejected``, ``agent/conversation_compression.py:3547-3563``).
        """
        self._publish("_last_compression_status", "noop")
        self._publish("_last_compression_noop_reason", reason)
        self._publish("_last_compress_aborted", False)
        self._publish("_last_summary_error", None)
        self._publish("_last_compression_made_progress", False)
        logger.info("LCM compression no-op: %s", reason)
        return messages

    def _abort(self, messages: List[Dict[str, Any]], cause: str) -> List[Dict[str, Any]]:
        """The compaction cannot be done as the model requires: the host keeps its list
        and shows "⚠ Compression aborted: <cause>" (#29 W2 step 1)."""
        self._publish("_last_compression_status", "aborted")
        self._publish("_last_compression_noop_reason", cause)
        self._publish("_last_compress_aborted", True)
        self._publish("_last_summary_error", cause)
        self._publish("_last_compression_made_progress", False)
        logger.warning("LCM compaction aborted: %s", cause)
        return messages

    @staticmethod
    def _rollback_planning(attempt, why: BaseException) -> None:
        """Roll the planning transaction back: an abort after the cut was written but
        before it is committed leaves nothing of this attempt in the store."""
        try:
            attempt.end_planning(why)
        except Exception:
            logger.warning("LCM could not roll back the planning transaction", exc_info=True)
        # The compaction's id was rolled back with it and may be issued again: events of
        # this attempt name no compaction.
        attempt.compaction = None

    def _sizing_failed(self, attempt, messages: List[Dict[str, Any]], what: str,
                       exc: BaseException) -> List[Dict[str, Any]]:
        """Sizing the material failed (a row the summariser's form or the estimate cannot
        take: a ``RecursionError`` from deeply nested tool arguments, any other exception):
        the compaction is aborted with the true cause and an event, never an exception out
        of ``compress()`` (the orchestrator's ruling on the Codex review of 40eda93; #7)."""
        logger.warning("LCM could not size the material while %s (%s: %s)", what, type(exc).__name__, exc)
        self._record_event(attempt, "cut_sizing_failed", {"while": what, "error": f"{type(exc).__name__}: {exc}"})
        return self._abort(messages, f"the material could not be sized while {what} ({type(exc).__name__}: {exc})")

    def _store_read_failed(self, attempt, messages: List[Dict[str, Any]], kind: str, what: str,
                           exc: BaseException) -> List[Dict[str, Any]]:
        """A store read of the compaction failed (a lock another process holds past the
        busy timeout, an I/O error, a damaged page, a store closed meanwhile): the
        compaction is aborted, the context untouched, and the host shows the cause (#7:
        every failure explicit, never an exception out of ``compress()``)."""
        logger.warning("LCM could not read %s from the store (%s: %s)", what, type(exc).__name__, exc)
        self._record_event(attempt, kind, f"{type(exc).__name__}: {exc}")
        return self._abort(messages, f"the store could not be read for {what} ({type(exc).__name__}: {exc})")

    def _summariser_route(self) -> tuple[Optional[SummariserRoute], str]:
        """The summariser's whole route, or why there is none (#9). Reads nothing of the
        store: the preflight and the cut's chunk size ask it too."""
        config = self._config
        problem = configured_route_problem(config)
        if problem is not None:
            # Refused when the configuration was loaded (a summariser other than the
            # session's model, #68), and every compaction says why.
            return None, problem
        if not (self.model and self.provider):
            return None, ("the session's model is not known: the host has not named its route "
                          "through update_model")
        if not self.base_url and _host_provider(self.provider) == "custom":
            return None, ("the session's route names provider custom without a base URL: the host "
                          "would borrow an endpoint of its own, which the session did not name")
        return session_route(self.provider, self.model, self.base_url, self.api_key, self.api_mode), ""

    def _summariser_settings(self) -> tuple[Optional[CallSettings], str]:
        """The summariser's route, effort and output cap, or the reason there is none
        (#9). No part of the route is guessed: it is what the host handed
        ``update_model``."""
        config = self._config
        route, problem = self._summariser_route()
        if route is None:
            return None, problem
        secrets = tuple(s for s in (self.api_key,) if isinstance(s, str) and s)
        effort = None
        if self._plugin_session:
            try:
                effort = self._sessions.latest_fact(self._plugin_session, "effort")
            except Exception as exc:
                return None, f"the session's reasoning effort could not be read ({type(exc).__name__})"
        effort = (effort or config.summary_reasoning_effort or "").strip().lower()
        if effort not in REASONING_EFFORTS:
            return None, (f"the summariser's reasoning effort {effort!r} is not one of the host's levels "
                          f"({', '.join(sorted(REASONING_EFFORTS))})")
        facts = lookup_model(route.model, route.table_provider())
        return CallSettings(
            route=route,
            effort=effort,
            max_tokens=facts.output_cap if facts is not None else None,
            secrets=secrets,
        ), ""

    def compress(self, messages: List[Dict[str, Any]],
                 current_tokens: int = None,
                 focus_topic: Optional[str] = None,
                 force: bool = False,
                 bypass_cooldown: bool = False) -> List[Dict[str, Any]]:
        """Run one compaction attempt.

        The host's cancellation check and the attempt's generation are captured here,
        at entry, on this attempt's own thread (#33 D12). A cancelled attempt returns
        its input at once and sets nothing. Engine attributes are set only when the
        attempt returns and is the host's current working attempt; a terminal status
        is left on every failure under the same rule.

        ``bypass_cooldown``: the host hands a keyword on only to an engine whose
        signature names it (``_supported_compression_kwargs``,
        agent/conversation_compression.py 1771-1792 at 7b761da). It passes this one
        when the provider rejected a request as too long (agent/turn_overflow.py
        162-165, with the request's size as ``current_tokens``) and on its stall retry
        of an occasion already under way (agent/compression_facade.py 282). The
        provider's refusal is the real count: it compacts whatever τ says (#56).
        """
        attempt = self._begin_attempt()
        if attempt.cancelled():
            return messages
        # The host's native compaction switch, read where the host reads it (#32 D2).
        self._check_host_native_compaction()
        token = _ATTEMPT.set(attempt)
        try:
            try:
                result = self._compress_impl(
                    messages,
                    current_tokens=current_tokens,
                    focus_topic=focus_topic,
                    force=force,
                    provider_rejected=bool(bypass_cooldown),
                )
            except BaseException as exc:
                # A planning transaction still open is rolled back (#33 D14 as revised).
                try:
                    attempt.end_planning(exc)
                except Exception:
                    logger.warning("LCM could not roll back the planning transaction", exc_info=True)
                raise
            # An early return inside it (an abort, a no-op) commits what it wrote; a
            # commit that fails is a visible abort (#7), never an exception.
            try:
                attempt.end_planning()
            except Exception as exc:
                logger.warning("LCM could not commit the planning transaction", exc_info=True)
                self._record_event(attempt, "planning_commit_failed", repr(exc))
                if not attempt.outcome.get("_last_compress_aborted"):
                    result = self._abort(messages, f"the store could not commit the compaction's planning ({exc})")
                # An attempt that already aborted keeps its first cause, the one the host shows.
        except AttemptCancelled:
            return messages
        except BaseException:
            if not attempt.cancelled() and self._attempt_is_current(attempt):
                self._last_compression_status = "error"
                self._last_compression_noop_reason = ""
            raise
        finally:
            attempt.over = True
            _ATTEMPT.reset(token)
        if attempt.cancelled():
            return messages
        if self._attempt_is_current(attempt):
            self._apply_attempt_outcome(attempt)
        return result

    def _chunk_input(self, records: List[tuple], route: SummariserRoute, *,
                     focus_topic: Optional[str]) -> "ChunkInput":
        """What the summariser receives for one chunk, built once (the pre-review of
        6a6a7f8, F4): from the chunk's records as the store holds them (#8), the level-1
        input the window check estimates and the call sends, and the encrypted
        reasoning withheld from it. The source is what the summary replaces in the
        session's context, by the session's estimate (R6): the acceptance compares the
        reply with this, by the same estimate."""
        source = self._estimator().messages([message for _record, message in records])
        budget = self._summary_budget(source)
        facts, wire = self._summariser_wire(route)
        counts: Dict[str, int] = {}
        level_one = level_one_input(records, budget, facts=wire, focus_topic=focus_topic or "",
                                    custom_instructions=self._config.custom_instructions, withheld=counts)
        return ChunkInput(records=records, source=source, budget=budget, facts=facts, wire=wire,
                          level_one=level_one, withheld=self._withheld_note(counts, facts))

    def _chunk_run(
        self,
        prepared: "ChunkInput",
        *,
        focus_topic: Optional[str],
        settings: CallSettings,
        chunk_handle: str = "",
    ) -> Callable[[ChunkCall], ChunkSummary]:
        """What a worker runs for one chunk: its summary, or ``SummaryFailure`` (#7),
        with the summariser's route and effort (#9). Everything the call needs is read
        before, on the ``compress()`` thread; the worker reads nothing of the engine.

        The summariser reads the chunk's records as the messages they were, whole
        (#8, ``summariser_input``). The budget is a target in the prompt text, never an
        output limit. The encrypted reasoning withheld from the input is named where it
        happens, when the call is admitted to its endpoint's limiter and goes out (#8,
        "Reasoning": never brushed over): a chunk whose summary is reused, whose call
        another attempt already makes, or whose call is abandoned before admission, sends
        nothing and logs nothing.
        """
        route = settings.route
        withheld = prepared.withheld
        custom_instructions = self._config.custom_instructions

        def run(call: ChunkCall) -> ChunkSummary:
            named = []

            @contextlib.contextmanager
            def dispatch():
                # The withholding is named when the call is admitted (a limiter slot
                # granted, an attempt still waiting), never for a call that is abandoned
                # before it goes out (the Codex review of 6f4a351, P3); once per call.
                with call.dispatch() as deadline:
                    if withheld is not None and not named:
                        named.append(True)
                        logger.warning("LCM withholds from the summariser %s of chunk %s: %s",
                                       withheld["items_text"], chunk_handle, withheld["text"])
                    yield deadline

            text, level, finish_reason = summarize_chunk(
                prepared.records,
                prepared.budget,
                source=prepared.source,
                settings=settings,
                facts=prepared.wire,
                depth=0,
                focus_topic=focus_topic or "",
                custom_instructions=custom_instructions,
                path=CallPath(wait=call.wait, dispatch=dispatch, hold=call.limiter.hold,
                              deadline=call.deadline),
                level_one=prepared.level_one,
            )
            return ChunkSummary(text=text, level=level, budget=prepared.budget, finish_reason=finish_reason,
                                model=route.model, provider=route.provenance_provider(), effort=settings.effort,
                                withheld=json.dumps(withheld["record"]) if withheld is not None else None)

        return run

    @staticmethod
    def _summary_budget(source) -> int:
        """The summary's target length (today's prompt, until #10)."""
        return min(max(_SUMMARY_BUDGET_MIN, int(source.tokens * _SUMMARY_BUDGET_SHARE)), _SUMMARY_BUDGET_MAX)

    @staticmethod
    def _summariser_wire(route: SummariserRoute):
        """The summariser's row in the model table (by its route, 9.6) and what its
        route means for its input: images go in only where the row says it reads them,
        and where there is no row it is not known (#8); the reasoning field and the
        converter by the host's rules for the route."""
        facts = lookup_model(route.model, route.table_provider())
        wire = wire_facts(route.provider, route.model, route.base_url, route.api_mode,
                          reads_images=facts.reads_images if facts is not None else None)
        return facts, wire

    @staticmethod
    def _withheld_note(counts: Dict[str, int], facts) -> Optional[dict]:
        """The named loss of the encrypted items withheld from a summariser (#8; R3 until
        the producer is stamped, ask A-P): per kind, how many, and whether this
        summariser's route takes that kind at all (the model table, 9.6), or that it is
        not known. None where nothing was withheld."""
        if not counts:
            return None
        takes = {}
        for kind in counts:
            takes[kind] = None if facts is None else (kind in facts.encrypted_reasoning)
        words = {True: "this summariser's route takes that kind", False: "this summariser's route does not take "
                 "that kind", None: "whether this summariser's route takes it is not known (no row)"}
        items_text = ", ".join(f"{count} {kind}" for kind, count in sorted(counts.items()))
        text = ("encrypted reasoning, withheld because its producer is not known (ask A-P); "
                + "; ".join(f"{kind}: {words[takes[kind]]}" for kind in sorted(counts)))
        return {"items_text": items_text, "text": text,
                "record": {"withheld": counts, "reason": "producer not known (ask A-P)",
                           "summariser_takes": takes,
                           "table": facts.basis if facts is not None else "no row"}}

    def _calls_in_flight_limit(self, endpoint: str) -> int:
        """The endpoint's limit: its own where configured, else the default (#33)."""
        per_endpoint = self._config.summary_calls_per_endpoint or {}
        value = per_endpoint.get(endpoint, self._config.summary_calls_in_flight)
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            return DEFAULT_CALLS_PER_ENDPOINT

    def _judge_failure(self, attempt, chunk_handle: str, number: int, total: int, records: List[str],
                       unidentified: List[str], failure: str, kind: str,
                       record: bool = True) -> tuple[str, bool]:
        """Record a chunk's failure with its kind and say what the host shows for it,
        where the failure is recorded, whether or not an attempt still waits for it
        (ruling on #61, 4). Returns (the cause, whether this call wrote the failure).
        A call several attempts joined is one trial of the chunk: only the subscriber
        asked to ``record`` writes its failure and the events; the others read the
        streak it left and name the chunk the same way. Where the write raises, the
        failure is not recorded and the next subscriber of the call is asked.

        Only the chunk's own failures count (ruling on #61, 2): a reply the checks
        rejected, or a request the provider rejected. Where a chunk of exactly these
        members failed by its own fault in ``_KEEPS_FAILING_ATTEMPTS`` consecutive
        attempts, this one included, the chunk keeps failing (#7, #33): a store event
        names it, and so does the cause. ``unidentified`` are the chunk's member records
        whose rows came without a host identity (``_row_id``): the gateway's replayed
        history, or host scaffolding the host never persists. Such a chunk's records
        are new at every attempt, so its earlier failures cannot be counted: each own
        failure of it is named, with those rows. Nothing else changes: the next
        occasion retries it."""
        name = (f"chunk {chunk_handle} ({len(records)} message{'' if len(records) == 1 else 's'}, records "
                f"{records[0]} to {records[-1]})" if records else f"chunk {chunk_handle}")
        recorded = False
        try:
            if record:
                streak = self._records.chunk_failed(chunk_handle, failure, kind=kind, session=attempt.session,
                                                    records=records)
                recorded = True
            else:
                streak = self._records.failure_streak(attempt.session, records)
        except Exception as exc:
            logger.warning("LCM could not %s the failure of chunk %d of %d (%s: %s)",
                           "record" if record else "count", number, total, type(exc).__name__, exc)
            streak = None
        if kind not in OWN_FAILURE_KINDS:
            whose = {"route": "another route answered or was resolved (#33 D9)",
                     "endpoint": "the endpoint failed"}.get(kind, "a failure of no kind known as the chunk's own")
            return (f"the summary of {name}, {number} of {total}, failed ({failure}): {whose}, which does not "
                    f"count toward the chunk's own failures"), recorded
        detail = {"chunk": chunk_handle, "members": len(records), "first_record": records[0] if records else None,
                  "last_record": records[-1] if records else None, "error": failure}
        if unidentified:
            if recorded:
                self._record_event(attempt, "chunk_failed_unidentified", {**detail, "without_identity": unidentified})
            return (f"{name} failed with: {failure}. It has rows without a host identity (records "
                    f"{', '.join(unidentified)}), so whether it failed before cannot be counted (the ask to "
                    f"Hermes: A1)"), recorded
        if streak is not None and streak >= _KEEPS_FAILING_ATTEMPTS:
            if recorded:
                self._record_event(attempt, "chunk_keeps_failing", {**detail, "attempts": streak})
            return (f"{name} failed in {streak} consecutive attempts, the last with: {failure}. A chunk that keeps "
                    f"failing is a defect to debug (#7); the next occasion retries it as the same chunk"), recorded
        return f"the summary of {name}, {number} of {total}, failed ({failure})", recorded

    def _deliver_for(self, attempt, chunk_handle: str, number: int, total: int, records: List[str],
                     unidentified: List[str]):
        """How one attempt's chunk receives its call's outcome, on whichever thread has
        it: the summary is written as a derivation of this chunk (#33 Q17a: not fenced,
        a fact about the chunk's content); a failure is recorded and judged there
        (``_judge_failure``), and the cause it gives is the attempt's."""

        def deliver(summary: Optional[ChunkSummary], failure: Optional[str], abandoned: bool,
                    kind: Optional[str] = None, record: bool = True) -> Outcome:
            if summary is not None:
                try:
                    derivation = self._write_summary(
                        attempt, chunk_handle, text=summary.text, level=summary.level, budget=summary.budget,
                        finish_reason=summary.finish_reason,
                        model=summary.model, provider=summary.provider, effort=summary.effort,
                        withheld_reasoning=summary.withheld,
                    )
                except Exception as exc:
                    logger.warning("LCM could not write the summary of chunk %d of %d (%s: %s)",
                                   number, total, type(exc).__name__, exc)
                    try:
                        self._record_event(attempt, "derivation_write_failed", repr(exc))
                    except Exception:
                        pass  # the store itself is gone (closed engine): the warning above says so
                    return Outcome(failure=f"the store could not write the summary ({exc})")
                return Outcome(derivation=derivation)
            if abandoned:
                logger.info("LCM summary call of chunk %d of %d ended unmade: %s", number, total, failure)
                self._record_event(attempt, "summary_call_abandoned",
                                   {"chunk": chunk_handle, "number": number, "reason": failure})
            else:
                kind = kind or "other"
                logger.warning("LCM summary of chunk %d of %d failed (%s): %s", number, total, kind, failure)
                self._record_event(attempt, "summary_failed",
                                   {"chunk": chunk_handle, "number": number, "kind": kind, "error": failure})
                cause, recorded = self._judge_failure(attempt, chunk_handle, number, total, records,
                                                      unidentified, failure or "", kind, record)
                return Outcome(failure=failure, cause=cause, recorded=recorded)
            return Outcome(failure=failure)

        return deliver

    @staticmethod
    def _groups(messages: List[Dict[str, Any]], material: List[int]) -> List[List[int]]:
        """The material in list order as groups a cut may not enter: an assistant row
        that carries tool calls, with every result of those calls in the material and
        whatever stands between them; every other entry alone. Calls and results are
        matched by their ``tool_call_id``, the host's identity for the pair. The span
        closes over every assistant inside it: a group runs from the first call to the
        last result of any call it holds, so interleaved calls stay in one group. A tool
        row that no group of the material opened (an empty or unknown
        ``tool_call_id``, or a call outside the material) is a ``ToolPairingError``."""

        def calls(index: int) -> set:
            message = messages[material[index]]
            if message.get("role") != "assistant":
                return set()
            return {_tool_call_id(call) for call in (message.get("tool_calls") or [])} - {""}

        groups: List[List[int]] = []
        at = 0
        while at < len(material):
            if messages[material[at]].get("role") == "tool":
                raise ToolPairingError(
                    f"the tool row at position {material[at]} answers "
                    f"{str(messages[material[at]].get('tool_call_id') or '').strip() or 'no call'!r}, a call no "
                    f"assistant row of the material before it made")
            end = at
            call_ids = calls(at)
            while call_ids:
                last = end
                for later in range(end + 1, len(material)):
                    other = messages[material[later]]
                    if other.get("role") == "tool" and str(other.get("tool_call_id") or "").strip() in call_ids:
                        last = later
                widened = set().union(*(calls(k) for k in range(at, last + 1)))
                if last == end and widened <= call_ids:
                    break
                end, call_ids = last, call_ids | widened
            groups.append(material[at:end + 1])
            at = end + 1
        return groups

    @classmethod
    def _cut_chunks(cls, messages: List[Dict[str, Any]], material: List[int], limit: int,
                    estimator: Optional[Estimator] = None, kept: Sequence[List[int]] = (),
                    summarised: Sequence[bool] = (),
                    joins: Optional[List[dict]] = None, *,
                    convert: Optional[Callable[[Any], Any]] = None,
                    bound: Optional[int] = None,
                    alone: Optional[List[dict]] = None,
                    image_limit: Optional[int] = None) -> List[List[int]]:
        """The material cut in list order into chunks, only between groups: a tool
        call is never separated from its results (#31, #12). No chunk below c/4 is cut
        where a merge within B can avoid it (ruling on #61, 1): a summary of a chunk that
        small cannot be shorter than its source, so it would fail on every attempt (#7,
        ruling on #52).

        Rows are sized by ``estimator`` after ``convert`` (the caller's
        ``_cut_estimate``: each row as the summariser receives it). ``bound`` is B, the
        most the summariser reads in one call (``_summariser_bound``), or None where the
        model table does not know its window: no merge makes a chunk above it (the
        orchestrator's ruling on the Codex review of c2efe0e). A part or run below c/4
        that no neighbour takes within B stands alone and is named in ``alone``.

        ``image_limit`` (``wire_image_limit``) bounds each chunk's images as B bounds its
        tokens (the orchestrator's ruling on the Codex review of 6f4a351): images are
        counted as the summariser receives each row; a group above it is a chunk of its
        own (the check after the cut refuses it); a run is split within it; no merge or
        join passes it.

        ``kept`` are the chunks of an earlier attempt the cut keeps (#33 D14), each as
        its positions, whole groups of the material, and ``summarised`` says of each
        whether it has a summary: each is one chunk with exactly its members, frozen,
        and only the rest around them is split. Nothing joins a kept chunk (ruling on
        #61), with the one exception below.

        A group larger than ``limit`` (c) is a chunk of its own: a chunk flexes by one
        group. The groups between such groups form runs; each run of W tokens is split
        equally (``_split_run``): ceil(W / c) chunks cut at the group boundaries nearest
        k·W/n, none below c/4 where a merge within B avoids it.

        Where the whole material is below c/4, nothing is cut: it stays raw, and the
        caller makes the compaction a no-op. Otherwise a run below c/4
        (``_smallest_for``) does not stand alone:
        - it joins the chunk of an adjacent oversized group, the following one where
          there is one (the work it opened), else the one before it;
        - else, at the end of the material, alone or after a kept chunk, it stays raw:
          its rows are in no chunk, and the caller begins the tail at the first of
          them, so they join material at a later compaction;
        - else it stands before or between kept chunks, and joins a neighbouring kept
          chunk that has no summary yet, so that no summary is thrown away, the
          following one first; where both are summarised, the following one (ruling on
          #61, D1). Rows arrive only at the end of the list and every recorded chunk
          is kept, so such a run arises only from a chunk without a summary cut again
          after c or the estimate changed, or from rows the host put among kept
          chunks.
        - Where the chunk a join makes is still below c/4 (its neighbour is a
          summarised kept chunk that was cut at a smaller c), it goes on joining
          contiguous neighbours by the same preference, a run last, until it holds
          c/4 (orchestrator ruling on 60a48c9): a kept chunk is taken whole, and a
          summary it releases is lossless, since the joined chunk is summarised again;
          a run takes the joined rows into its equal split. No kept chunk without a
          summary is ever released into a run: one below c/4 was cut again already,
          and one at c/4 or more ends the join.
        - No join makes a chunk above B: a neighbour it would take above B is passed
          over (a run is not, since it is split again). A run below c/4 that no
          neighbour takes within B stays raw at the end of the material, and elsewhere
          stands alone, named in ``alone``; the caller records it.
        Each join is recorded in ``joins``, and the caller records an event for it. A
        joined chunk has new members, so it is summarised afresh; it is recorded with
        this attempt's cut, so a retry keeps it and the join is not made again. Never
        jumping over a chunk, so the chunks stay contiguous.
        """
        groups = cls._groups(messages, material)
        sizes = [sum(count_message_tokens(convert(messages[index]) if convert else messages[index], estimator)
                     for index in group) for group in groups]
        pictures = ([sum(image_count(convert(messages[index]) if convert else messages[index]) for index in group)
                     for group in groups] if image_limit is not None else [0] * len(groups))
        kept_of: Dict[int, int] = {}   # group number -> the kept chunk it belongs to
        first_group = {group[0]: number for number, group in enumerate(groups)}
        for which, positions in enumerate(kept):
            at, numbers = 0, []
            while at < len(positions) and positions[at] in first_group:
                number = first_group[positions[at]]
                numbers.append(number)
                at += len(groups[number])
            if at != len(positions) or [i for n in numbers for i in groups[n]] != list(positions):
                raise ToolPairingError(f"the kept chunk at positions {positions[0]} to {positions[-1]} is not whole "
                                       f"groups of the material")
            kept_of.update({number: which for number in numbers})
        # The material as items in order: a kept chunk, an oversized group, or a run of the others.
        items: List[tuple] = []
        run: List[int] = []
        for number, size in enumerate(sizes):
            if number in kept_of or size > limit or (image_limit is not None and pictures[number] > image_limit):
                if run:
                    items.append(("run", run))
                    run = []
                if number not in kept_of:
                    items.append(("oversized", [number]))
                elif items and items[-1][0] == "kept" and kept_of[items[-1][1][-1]] == kept_of[number]:
                    items[-1][1].append(number)
                else:
                    items.append(("kept", [number]))
            else:
                run.append(number)
        if run:
            items.append(("run", run))

        smallest = _smallest_for(limit)
        # The whole material below c/4 stays raw: nothing can be cut that is not below
        # it (ruling 1; orchestrator ruling on 60a48c9). The caller makes it a no-op.
        if sum(sizes) < smallest:
            return []

        def summarised_kept(numbers: List[int]) -> bool:
            which = kept_of[numbers[0]]
            return bool(summarised[which]) if which < len(summarised) else False

        # Each item: [kind, group numbers, has a summary]. Kinds: "oversized" and
        # "kept" (one chunk each), "joined" (one chunk made by a join), "run" (split
        # equally), "raw" (the end of the material, left raw).
        work: List[list] = [[kind, list(members), kind == "kept" and summarised_kept(members)]
                            for kind, members in items]

        def weight(item: list) -> int:
            return sum(sizes[g] for g in item[1])

        def images_of(item: list) -> int:
            return sum(pictures[g] for g in item[1])

        def positions(item: list) -> List[int]:
            return [groups[item[1][0]][0], groups[item[1][-1]][-1]]

        def fits(at: int, side: int) -> bool:
            # A run takes the joined rows into its equal split, which bounds each part;
            # every other neighbour becomes one chunk with the item, bounded by B and by
            # the wire's image limit.
            if work[side][0] == "run":
                return True
            return ((bound is None or weight(work[at]) + weight(work[side]) <= bound)
                    and (image_limit is None or images_of(work[at]) + images_of(work[side]) <= image_limit))

        def target_for(at: int) -> Optional[int]:
            """Where a tiny item at ``at`` joins: an oversized neighbour, the following
            one first; a tiny run at the very end stays raw (None); else a neighbour
            without a summary, then one with a summary, then a run, the following
            one first on a tie (D1). A neighbour the join would take above B is passed
            over; where none is left, the item stands alone (-1)."""
            item = work[at]
            following = at + 1 if at + 1 < len(work) else None
            preceding = at - 1 if at > 0 else None
            for side in (following, preceding):
                if side is not None and work[side][0] == "oversized" and fits(at, side):
                    return side
            if following is None and item[0] == "run":
                return None
            rank = {"joined": 0, "kept": 1, "run": 3}

            def cost(side: int) -> tuple:
                neighbour = work[side]
                return (rank.get(neighbour[0], 3) + (1 if neighbour[0] == "kept" and neighbour[2] else 0),
                        0 if side == following else 1)

            sides = [side for side in (following, preceding)
                     if side is not None and work[side][0] != "oversized" and fits(at, side)]
            return min(sides, key=cost) if sides else -1

        # A run below c/4 joins its neighbours until the chunk it makes holds at least
        # c/4, over as many contiguous neighbours as that takes: an oversized group
        # ends it at once; a kept chunk is taken whole, releasing its summary where it
        # has one (lossless: the joined chunk is summarised again); a run takes the
        # joined rows into its equal split. Every join is recorded (``joins``).
        at = 0
        while at < len(work):
            item = work[at]
            if item[0] not in ("run", "joined") or weight(item) >= smallest:
                at += 1
                continue
            side = target_for(at)
            if side is None:
                item[0] = "raw"   # stays raw: the tail begins at it
                at += 1
                continue
            if side == -1:
                # No neighbour takes it within B: it stands alone, below c/4, named.
                if alone is not None:
                    alone.append({"positions": positions(item), "tokens": weight(item), "bound": bound,
                                  "neighbours": [{"kind": work[s][0], "tokens": weight(work[s])}
                                                 for s in (at - 1, at + 1) if 0 <= s < len(work)]})
                item[0] = "alone"
                at += 1
                continue
            neighbour = work[side]
            low, high = min(at, side), max(at, side)
            merged_groups = work[low][1] + work[high][1]
            if joins is not None and neighbour[0] in ("kept", "joined", "run"):
                joins.append({"run": positions(item), "tokens": weight(item), "kept": positions(neighbour),
                              "joined": neighbour[0], "summarised": bool(neighbour[2]),
                              "side": "following" if side > at else "preceding",
                              "result_tokens": weight(item) + weight(neighbour)})
            if neighbour[0] == "oversized":
                merged = ["oversized", merged_groups, False]
            elif neighbour[0] == "run":
                merged = ["run", merged_groups, False]
            else:
                merged = ["joined", merged_groups, False]
            # A kept chunk a join takes is released (its summary with it): its groups are
            # no longer kept, and may be split like any other (the orchestrator's ruling on
            # the Codex review of 722d336).
            for g in merged_groups:
                kept_of.pop(g, None)
            work[low:high + 1] = [merged]
            at = low

        chunks: List[List[int]] = []
        for kind, members, _summary in work:
            if kind == "raw":
                continue
            # A part below c/4 left alone (a run, or a join whose kept chunks were released)
            # is still split where its images pass the limit.
            split_alone = (kind == "alone" and image_limit is not None
                           and sum(pictures[g] for g in members) > image_limit
                           and not any(g in kept_of for g in members))
            if kind == "run" or split_alone:
                parts: List[tuple] = []
                starts = [0] + _split_run([sizes[g] for g in members], limit, bound, parts,
                                          [pictures[g] for g in members], image_limit) + [len(members)]
                chunk_groups = [members[a:b] for a, b in zip(starts, starts[1:])]
                if alone is not None:
                    for a, b, tokens in parts:
                        alone.append({"positions": [groups[members[a]][0], groups[members[b - 1]][-1]],
                                      "tokens": tokens, "bound": bound, "neighbours": "parts of the equal split"})
            else:
                chunk_groups = [members]
            for group_numbers in chunk_groups:
                chunks.append([index for g in group_numbers for index in groups[g]])
        return chunks

    def _compress_impl(self, messages: List[Dict[str, Any]],
                       current_tokens: int = None,
                       focus_topic: Optional[str] = None,
                       force: bool = False,
                       provider_rejected: bool = False) -> List[Dict[str, Any]]:
        attempt = _ATTEMPT.get()
        if self._closed_reason is not None:
            return self._abort(
                messages,
                f"LCM's store connections of this engine were closed ({self._closed_reason}); "
                f"a closed engine is never reused",
            )
        if not messages:
            return self._unchanged_return(messages, "empty message list")
        started = time.perf_counter()
        force_overflow = self._should_force_overflow_recovery(
            observed_tokens=current_tokens,
            messages=messages,
        )
        recovery_cap = (
            self._overflow_recovery_assembly_cap(observed_tokens=current_tokens, messages=messages)
            if force_overflow
            else None
        )
        if not attempt.session:
            return self._abort(
                messages,
                "this engine copy is bound to no session of the plugin, and it compacts only for its own",
            )
        attempt.messages = messages
        if self._geometry is None:
            # R11: outside #21's bounds nothing is compacted (recorded where W was set).
            return self._abort(messages, f"LCM compacts nothing: {self._geometry_error}")

        # 0. The occasion (#32 D1) and the threshold. Compaction runs at τ and only
        # there (#11), unless forced (the host's /compress, R15) or on overflow
        # recovery. The count is the host's: its real count at most call sites, its own
        # rough estimate at some (turn_recovery.py:1839). Where the host hands none, the
        # plugin's estimate is converted into provider tokens by #31's ratio, labelled.
        occasion = self._classify_occasion(messages, force=force)
        tau = self._tau(raised=False)
        if current_tokens is not None:
            count, count_label = int(current_tokens), f"the host's count {int(current_tokens)}"
        else:
            count, count_label = self._provider_estimate(count_messages_tokens(messages, self._estimator()))
        if not (force or force_overflow or provider_rejected) and count < tau:
            return self._unchanged_return(
                messages, f"below the threshold: {count_label} < τ {tau} ({occasion.why}; {self._geometry.label()})")
        if force:
            why_now = "the compaction was forced"
        elif force_overflow:
            why_now = "the context overflowed"
        elif count < tau:
            why_now = (f"the host required it after the provider rejected the request or a stalled attempt "
                       f"({count_label}, below τ {tau})")
        else:
            why_now = f"{count_label} is at or above τ {tau}"
        logger.info("LCM compacts: %s; %s", why_now, occasion.why)

        # The summariser, as far as anything can be written: its route and effort (#9).
        # Read before the planning transaction: nothing of the cut depends on it but the
        # check that the summariser can read each chunk.
        settings, why_not = self._summariser_settings()
        if settings is None:
            return self._abort(messages, why_not)

        # What need not be atomic with the cut is read or written before the planning
        # transaction (pre-review of 78c2cbf): settling the previous return commits in a
        # transaction of its own, its changes in memory made only after that commit; the
        # fixed prefix F is a fact of the session helper, read here, so that nothing inside
        # planning takes another lock or connection.
        self._settle_from_list(messages)
        try:
            fixed = self._fixed_prefix()
        except Exception as exc:
            return self._store_read_failed(attempt, messages, "fixed_prefix_unreadable",
                                           "the session's fixed prefix F", exc)

        # The planning phase, from opening the planning transaction to the dispatch of the
        # first call, inside one boundary (the orchestrator's ruling on the Codex review
        # of a6e776e): any exception in it, but the attempt's and the host's cancellations,
        # is a visible abort with its true cause and an event naming the step it was in
        # (``_Step``, ``_planning_failed``). The handlers inside that give a better cause
        # stay; the boundary catches everything else.
        step = _Step("opening the planning transaction")
        try:
            planned = self._planning_phase(attempt, messages, step, occasion=occasion, fixed=fixed,
                                           settings=settings, focus_topic=focus_topic, force=force,
                                           force_overflow=force_overflow, provider_rejected=provider_rejected,
                                           why_now=why_now)
        except BaseException as exc:
            if _passes_the_boundary(exc):
                raise
            return self._planning_failed(attempt, messages, step, exc)
        if not isinstance(planned, _Planned):
            return planned   # an abort or a no-op the planning phase returned itself
        return self._dispatch_phase(attempt, messages, planned, settings=settings, focus_topic=focus_topic,
                                    started=started, recovery_cap=recovery_cap)

    def _planning_failed(self, attempt, messages: List[Dict[str, Any]], step: "_Step",
                         exc: BaseException) -> List[Dict[str, Any]]:
        """An exception inside the planning phase that no handler there named: the
        planning transaction, where still open, is rolled back (nothing of the attempt is
        recorded), an event names the step, and the host shows the true cause."""
        logger.warning("LCM compaction failed while %s", step.at, exc_info=True)
        if attempt.planning is not None:
            self._rollback_planning(attempt, exc)
        try:
            self._record_event(attempt, "planning_phase_failed",
                               {"step": step.at, "error": f"{type(exc).__name__}: {exc}"})
        except Exception:
            logger.warning("LCM could not record the planning failure", exc_info=True)
        return self._abort(messages, f"the compaction failed while {step.at} ({type(exc).__name__}: {exc})")

    def _planning_phase(self, attempt, messages: List[Dict[str, Any]], step: "_Step", *, occasion: Occasion,
                        fixed: tuple, settings: CallSettings, focus_topic: Optional[str], force: bool,
                        force_overflow: bool, provider_rejected: bool, why_now: str):
        """Steps 1 to 3 and the dispatch's preparation, inside ``_compress_impl``'s one
        boundary: returns a ``_Planned`` to dispatch, or the list an abort or a no-op
        returns. ``step.at`` names the step under way."""
        # The planning transaction (#33 D14 as revised 2026-09-25): every store read the
        # cut depends on, from the identity on, and the write of the cut, in one
        # BEGIN IMMEDIATE, after the attempt's captured check is asked inside it. A
        # planner in another process waits for an earlier attempt's commit and reads its
        # chunks. Nothing in it calls a model, does other I/O, or takes another lock or
        # connection. compress() closes it on any early return or exception. A store
        # failure opening or committing it is a visible abort (#7), never an exception.
        try:
            self._begin_planning(attempt)
        except AttemptCancelled:
            raise
        except Exception as exc:
            logger.warning("LCM could not open the planning transaction", exc_info=True)
            attempt.end_planning(exc)
            self._record_event(attempt, "planning_transaction_failed", repr(exc))
            return self._abort(messages, f"the store could not begin the compaction's planning transaction ({exc})")

        # 1. Identity (#29 W3). A list that cannot be classified is not compacted; a store
        # that cannot be read for it is a visible abort (#7), never an exception.
        step.at = "classifying the list"
        try:
            entries = self._classify(attempt, messages)
        except Exception as exc:
            return self._store_read_failed(attempt, messages, "classify_unreadable",
                                           "the records the list is classified against", exc)
        if entries is None:
            return self._abort(messages, attempt.error[1] if attempt.error else "the list could not be classified")
        mechanism = {entry.position for entry in entries if entry.klass == "system"}
        mechanism |= attempt.summary_inputs

        # 2. The tail and the material. The tail takes what the target leaves, in whole
        # groups, from its floor (D1) up to the ceiling that leaves the oldest group
        # outside (#13, #31). At the threshold everything outside it is chunked, with no
        # minimum (#11, #12).
        # On a retry the cut is frozen (#33 D14 as revised): every recorded chunk of the
        # unsettled attempts is found by its members' ids and records. What is cut
        # again is said visibly (ruling 3 on #61).
        step.at = "reading the earlier attempts' chunks"
        try:
            frozen = self._frozen_candidates(
                messages, {entry.position: entry.record for entry in entries
                           if entry.klass in ("reused", "bound") and entry.record is not None})
        except Exception as exc:
            logger.warning("LCM could not read the earlier attempts' chunks", exc_info=True)
            self._record_event(attempt, "frozen_cut_unreadable", repr(exc))
            return self._abort(messages, f"the chunks of the earlier attempt could not be read ({exc})")
        if frozen.unidentified:
            # One line and one event per attempt: a chunk with rows that came without a
            # host identity (the gateway's replayed history, or host scaffolding the host
            # never persists) is never found again by identity.
            attempts = sorted({compaction for _chunk, compaction, _rows in frozen.unidentified})
            named = "; ".join(f"chunk {chunk} (rows without identity: records {', '.join(rows)})"
                              for chunk, _compaction, rows in frozen.unidentified)
            logger.warning("LCM cuts again %d recorded chunks of %d earlier attempts, each with rows "
                           "that came without a host identity, so they cannot be found in this list by identity: %s. "
                           "The ask to Hermes: A1", len(frozen.unidentified), len(attempts), named)
            self._record_event(attempt, "frozen_cut_unidentified",
                               {"attempts": attempts,
                                "chunks": [{"chunk": chunk, "attempt": compaction, "without_identity": list(rows)}
                                           for chunk, compaction, rows in frozen.unidentified]})
        step.at = "placing the tail"
        try:
            plan = self._tail_plan(messages, mechanism, occasion, frozen.found, fixed=fixed,
                                   focus_topic=focus_topic or "")
        except ToolPairingError as exc:
            self._record_event(attempt, "tool_pairing_error", str(exc))
            return self._abort(messages, f"the tail cannot be placed: {exc}")
        except Exception as exc:
            return self._sizing_failed(attempt, messages, "placing the tail", exc)
        for chunk, why in frozen.recut + plan.recut:
            logger.warning("LCM cuts the %s chunk %s of attempt %d again: %s", chunk.state, chunk.chunk,
                           chunk.compaction, why)
            self._record_event(attempt, "frozen_chunk_recut",
                               {"chunk": chunk.chunk, "state": chunk.state, "attempt": chunk.compaction,
                                "reason": why})
        tail_start = plan.start
        logger.info("LCM tail: %s", plan.label)
        for chunk, positions in plan.held:
            # #31 holds that the floor is never inside a kept chunk; where it is (at a
            # gap, the newest tool group can stand in a chunk an earlier occasion cut),
            # the chunk is not cut again: it stays whole in the tail this time.
            self._record_event(attempt, "kept_chunk_reaches_floor",
                               {"chunk": chunk.chunk, "state": chunk.state, "positions": [positions[0], positions[-1]]})
        # The cut, before the boundary is checked: a rest below c/4 at the end of the
        # material stays raw and the tail begins at it (ruling on #61, 1).
        material = [index for index in range(tail_start) if index not in mechanism]
        step.at = "sizing the chunks"
        try:
            # The cut's unit: each row as the summariser receives it (``_cut_estimate``).
            cut_estimator, convert = self._cut_estimate()
            # c and B with the prompt this attempt's calls carry, its focus text included.
            limit = self._chunk_limit(focus_topic or "")
            bound = self._summariser_bound(focus_topic or "")[0]
            # The images one chunk may carry before the wire's converter drops some unseen.
            image_limit = self._image_limit()
        except Exception as exc:
            return self._sizing_failed(attempt, messages, "sizing the chunks", exc)
        chunks: List[List[int]] = []
        waiting: List[int] = []
        if material:
            joins: List[dict] = []
            alone: List[dict] = []
            step.at = "cutting the material"
            try:
                chunks = self._cut_chunks(messages, material, limit, cut_estimator,
                                          kept=[positions for _chunk, positions in plan.kept],
                                          summarised=[chunk.state == "summarised" for chunk, _p in plan.kept],
                                          joins=joins, convert=convert, bound=bound, alone=alone,
                                          image_limit=image_limit)
            except ToolPairingError as exc:
                self._record_event(attempt, "tool_pairing_error", str(exc))
                return self._abort(messages, f"the material cannot be cut: {exc}")
            except Exception as exc:
                return self._sizing_failed(attempt, messages, "cutting the material", exc)
            for part in alone:
                # Every merge is bounded by B (the orchestrator's ruling on the Codex
                # review of c2efe0e): a part below c/4 that no neighbour takes within B
                # is sent alone, and said so.
                logger.warning("LCM sends the rows at positions %d to %d (%d tokens) as a chunk below c/4 = %d: no "
                               "neighbour takes them within B = %s, the most the summariser reads in one call",
                               part["positions"][0], part["positions"][1], part["tokens"], _smallest_for(limit),
                               bound)
                self._record_event(attempt, "chunk_below_smallest_alone", part)
            for join in joins:
                # The one exception to "nothing joins a kept chunk" (ruling on #61, D1),
                # continued until the chunk holds c/4 (ruling on 60a48c9).
                logger.warning("LCM joins material below c/4 (positions %d to %d, %d tokens) to the %s %s at "
                               "positions %d to %d%s, making %d tokens: nothing below c/4 is sent",
                               join["run"][0], join["run"][1], join["tokens"], join["side"],
                               {"kept": "kept chunk", "joined": "joined chunk", "run": "run"}[join["joined"]],
                               join["kept"][0], join["kept"][1],
                               ", releasing its summary (it is summarised again)" if join["summarised"] else "",
                               join["result_tokens"])
                self._record_event(attempt, "kept_chunk_joined", join)
            covered = {index for chunk in chunks for index in chunk}
            waiting = [index for index in material if index not in covered]
            if waiting:
                if any(index not in covered for index in material if index < waiting[0]) or \
                        any(index in covered for index in material if index > waiting[0]):
                    self._record_event(attempt, "cut_left_rows_out", {"positions": waiting})
                    return self._abort(messages, f"the cut left rows outside every chunk that are not the end of "
                                                 f"the material (positions {waiting[0]} to {waiting[-1]})")
                last_summary = max(mechanism) if mechanism else -1
                if waiting[0] < last_summary:
                    # D3's shape (orchestrator, #33): a row the host put before the plugin's
                    # last summary is material. A rest that stays raw there cannot become
                    # the tail, which would then hold the summaries after it; nothing is
                    # changed, and the case is named. Whether the host produces it at all
                    # is not established (its commit-time insertions land after them).
                    self._record_event(attempt, "rest_before_last_summary",
                                       {"positions": waiting, "last_summary": last_summary})
                    return self._abort(
                        messages, f"the rows below c/4 left raw at positions {waiting[0]} to {waiting[-1]} begin "
                                  f"before the plugin's last summary at position {last_summary}: the tail cannot "
                                  f"begin there, so nothing is compacted (the host put a row before a summary)")
                tail_start = waiting[0]
                material = [index for index in material if index < tail_start]
                logger.info("LCM leaves %d rows below c/4 at the end of the material raw (from position %d): the "
                            "tail begins at them, and they join material at a later compaction",
                            len(waiting), tail_start)
        step.at = "checking the tail's boundary"
        unanswered = _unanswered_calls_at(messages, tail_start)
        if unanswered:
            # The forward check: a call at the boundary whose result is not in the list
            # would be split from it (#13). The host never asks with an open call (#32
            # §3); if it does, that is a defect to see, not to guess around.
            self._record_event(attempt, "tool_result_missing_at_boundary",
                               {"position": unanswered[0], "tool_call_ids": unanswered[1]})
            return self._abort(
                messages, f"the assistant row at position {unanswered[0]}, at the tail's boundary, made tool calls "
                          f"whose results are not in the list ({', '.join(unanswered[1])})")
        crossing = _pairs_crossing(messages, tail_start)
        if crossing:
            # The boundary would separate a call from its result (#13): the grouping
            # rules it out, so this is a defect to see.
            self._record_event(attempt, "tool_pair_across_boundary",
                               {"tool_call_id": crossing[0], "call": crossing[1], "result": crossing[2],
                                "boundary": tail_start})
            return self._abort(
                messages, f"the tail's boundary at position {tail_start} would separate the call {crossing[0]!r} "
                          f"at position {crossing[1]} from its result at position {crossing[2]}")
        if not material and waiting and not (force_overflow or provider_rejected):
            # Only a rest below c/4 stands outside the tail: nothing to compact yet, not a
            # failure (ruling on #61, 1), as the preflight says for the same list.
            rest = cut_estimator.messages([convert(messages[index]) for index in waiting])
            return self._unchanged_return(
                messages, f"nothing to compact: what stands outside the fresh tail is below the smallest run that "
                          f"stands alone, c/4 ({rest.tokens} < {_smallest_for(limit)} tokens, {rest.label()}; "
                          f"{self._chunk_label(focus_topic or '')}); it stays raw until more material joins it")
        if not material:
            if force_overflow or provider_rejected:
                self._publish("_last_overflow_recovery_failed", True)
            if waiting:
                return self._abort(
                    messages, f"{why_now}, and what stands outside the fresh tail is below the smallest run that "
                              f"stands alone, c/4 ({len(waiting)} rows): nothing can be compacted now")
            # The visible end (#7): above τ, nothing outside the tail's floor but the
            # mechanism's layer.
            return self._abort(
                messages,
                f"{why_now}, and nothing is left to compact: the chain of summaries and the newest "
                f"{'tool group' if len(messages) - tail_start > 1 else 'message'} fill the context; outside them "
                f"stands only the mechanism's layer (the system row and the summaries)",
            )
        step.at = "estimating the material"
        material_estimate = cut_estimator.messages([convert(messages[index]) for index in material])

        # 3. Transaction 1: the compaction, its inputs, the new records and every chunk,
        # the last step of the planning transaction.
        logger.info("LCM cut %d tokens of material into %d chunk%s (%s)", material_estimate.tokens, len(chunks),
                    "" if len(chunks) == 1 else "s", self._chunk_label(focus_topic or ""))
        # Kept chunks as they were, by their positions: summarised ones reuse their summary
        # whatever route wrote it; they are never summarised again (ruling on #61).
        kept_state = {tuple(positions): chunk.state for chunk, positions in plan.kept}
        if kept_state:
            states = list(kept_state.values())
            logger.info("LCM keeps the cut of earlier attempts (#33 D14): %d recorded chunks as they were "
                        "(%d summarised, %d retried as the same chunk)", len(states), states.count("summarised"),
                        states.count("cut"))
        step.at = "writing the compaction"
        try:
            chunk_handles = self._write_compaction(attempt, entries, chunks, force=force)
        except Exception as exc:
            logger.warning("LCM could not write the compaction", exc_info=True)
            self._record_event(attempt, "compaction_write_failed", repr(exc))
            return self._abort(messages, f"the store could not write the compaction ({exc})")

        # 3b. What the summariser receives for each chunk, built once from the chunk's
        # records as just written (#8; F4 of the pre-review of 6a6a7f8), read inside the
        # planning transaction, so that a chunk the summariser cannot read is refused
        # before its cut is committed: nothing of this attempt is then recorded.
        route = settings.route
        step.at = "reading the chunks' records"
        members = [attempt.records[index] for chunk in chunks for index in chunk]
        try:
            facts = self._records.record_facts(members)
        except Exception as exc:
            self._rollback_planning(attempt, exc)
            return self._store_read_failed(attempt, messages, "chunk_records_unreadable",
                                           "the chunks' records", exc)
        missing = sorted(set(members) - set(facts))
        if missing:
            self._rollback_planning(attempt, RuntimeError("chunk records missing"))
            self._record_event(attempt, "chunk_record_missing", {"records": missing})
            return self._abort(messages, f"the store holds no record {', '.join(missing)} of the chunks it just "
                                         f"recorded")
        step.at = "building the summariser's input"
        prepared: Dict[int, ChunkInput] = {}
        for number, chunk in enumerate(chunks, start=1):
            records = [attempt.records[index] for index in chunk]
            try:
                prepared[number] = self._chunk_input(
                    [(record, json.loads(facts[record][1])) for record in records], route, focus_topic=focus_topic)
            except Exception as exc:
                # F2 of the pre-review: never an exception out of compress().
                self._rollback_planning(attempt, exc)
                logger.warning("LCM could not build the summariser's input for chunk %d of %d (%s: %s)",
                               number, len(chunks), type(exc).__name__, exc)
                self._record_event(attempt, "chunk_input_failed",
                                   {"chunk": number, "error": f"{type(exc).__name__}: {exc}"})
                return self._abort(messages, f"the summariser's input for chunk {number} of {len(chunks)} could "
                                             f"not be built ({type(exc).__name__}: {exc})")
        # A chunk the summariser cannot read in one call would fail on every attempt
        # (#34 D4). The cut bounds every chunk of divisible material by B, in this check's
        # unit (``_chunk_size``, ``_cut_estimate``), so this refuses only a group that cannot
        # be divided, or a kept chunk cut before the summariser's bound applied. What is counted is what the
        # summariser receives (#8b): its level-1 input, an image it is not sent as its
        # placeholder, readable reasoning as the text part it becomes; compared like with
        # like: the estimate's characters / 4 by the worst case observed, never the median,
        # images by the summariser's rule. A kept chunk that has its summary is not sent,
        # so it is not checked.
        # A chunk carrying more images than the summariser wire's converter keeps would
        # lose the rest unseen (#8; the orchestrator's ruling on the Codex review of
        # 6f4a351). The cut keeps divisible material within the limit, so this refuses
        # only a group that cannot be divided, or a chunk an earlier attempt cut: never
        # with placeholders in their place, since a loss is a failure (see the PR).
        step.at = "checking each chunk's images against the wire's limit"
        if image_limit is not None:
            for number, chunk in enumerate(chunks, start=1):
                if kept_state.get(tuple(chunk)) == "summarised":
                    continue
                carried = sum(image_count(summariser_message(raw, record, prepared[number].wire))
                              for record, raw in prepared[number].records)
                if carried > image_limit:
                    groups = len(self._groups(messages, chunk))
                    what = (f"a chunk an earlier attempt cut ({groups} group{'' if groups == 1 else 's'})"
                            if tuple(chunk) in kept_state else
                            "one group that cannot be divided (a message, or a tool call with its results)"
                            if groups == 1 else
                            f"{groups} groups this attempt cut within the image limit (a defect: the cut and this "
                            f"check disagree)")
                    self._rollback_planning(attempt, RuntimeError("chunk over the wire's image limit"))
                    self._record_event(attempt, "chunk_over_image_limit",
                                       {"chunk": number, "groups": groups, "images": carried, "limit": image_limit})
                    return self._abort(
                        messages,
                        f"chunk {number} of {len(chunks)}, {what}, carries {carried} images, more than the "
                        f"{image_limit} the host's converter for the summariser {route.describe()} keeps in one "
                        f"request (agent/image_eviction_policy.py OUTBOUND_IMAGE_LIMIT): the rest would be removed "
                        f"unseen (#8)",
                    )
        step.at = "checking each chunk against the summariser's window"
        model_facts = prepared[1].facts if prepared else None
        if model_facts is not None and model_facts.context_window:
            room = model_facts.context_window - (model_facts.output_cap or 0)
            worst = float(self._config.estimate_ratio_max)
            summariser_estimate = Estimator(image_model=route.model, image_provider=route.table_provider(),
                                            reasoning_sent=prepared[1].wire.needs_reasoning_echo)
            for number, chunk in enumerate(chunks, start=1):
                if kept_state.get(tuple(chunk)) == "summarised":
                    continue
                estimate = summariser_estimate.messages(prepared[number].level_one)
                provider_tokens = estimate.in_provider_tokens(worst)
                if provider_tokens > room:
                    groups = len(self._groups(messages, chunk))
                    if tuple(chunk) in kept_state:
                        what = f"a chunk an earlier attempt cut ({groups} group{'' if groups == 1 else 's'})"
                    elif groups == 1:
                        what = "one group that cannot be divided (a message, or a tool call with its results)"
                    else:
                        # The cut bounds such a chunk by B in the same unit; reaching this
                        # is a defect of the cut, said as what it is.
                        what = (f"{groups} groups this attempt cut within B = {bound} by the cut's estimate (a defect: "
                                f"the cut and this check disagree)")
                    self._rollback_planning(attempt, RuntimeError("chunk over the summariser's window"))
                    self._record_event(attempt, "chunk_over_summariser_window",
                                       {"chunk": number, "groups": groups, "provider_tokens": provider_tokens,
                                        "room": room})
                    return self._abort(
                        messages,
                        f"chunk {number} of {len(chunks)}, {what}, would reach the summariser {route.describe()} as "
                        f"about {provider_tokens} provider tokens ({estimate.tokens} by the plugin's estimate, its "
                        f"characters / 4 times {worst}, the estimate's worst case observed, #34 D4; "
                        f"{estimate.label()}), more than it can read in one call ({model_facts.context_window} "
                        f"window less {model_facts.output_cap or 0} output; {model_facts.basis})",
                    )
        # The cut is recorded: commit the planning transaction before any call starts. A
        # commit that fails (the busy timeout, a full disk) wrote nothing: a visible
        # abort, the context untouched (#7).
        step.at = "committing the planning transaction"
        try:
            held_ms = attempt.end_planning()
        except Exception as exc:
            logger.warning("LCM could not commit the planning transaction", exc_info=True)
            self._record_event(attempt, "planning_commit_failed", repr(exc))
            return self._abort(messages, f"the store could not commit the compaction's cut ({exc})")
        logger.info("LCM held the store's write lock for %.1f ms to plan and record the cut", held_ms or 0.0)

        # The dispatch's preparation, still inside the boundary: nothing is sent yet.
        step.at = "preparing the dispatch"
        endpoint = endpoint_key(route.provider, route.base_url)
        limiter, limit = limiter_for(endpoint), self._calls_in_flight_limit(endpoint)
        # The host's progress hook and deadline, read here on the compress() thread (D10).
        hook, deadline = host_progress_hook(), host_deadline()

        def still_wanted() -> bool:
            return not attempt.over and self._live_write_allowed(attempt)

        def describe(exc: BaseException) -> str:
            # A SummaryFailure, or anything else the call raised: never truncated, never
            # swallowed. What is logged, stored and shown is the failure's class and
            # message with every known secret removed; never a traceback, which could
            # carry request headers.
            return settings.scrub(str(exc)) if isinstance(exc, SummaryFailure) \
                else failure_text(exc, settings.secrets)

        # Every chunk's outcome arrives here, once, after it is complete: the one place
        # the compress() thread reads outcomes from.
        finished: "queue.Queue[int]" = queue.Queue()
        calls: List[tuple] = []
        for number, (chunk_handle, chunk) in enumerate(zip(chunk_handles, chunks), start=1):
            records = [attempt.records[index] for index in chunk]
            # A member whose row has no host identity (the gateway's replayed history, or
            # host scaffolding the host never persists) is recorded anew at every attempt,
            # so this chunk's earlier failures cannot be counted (ruling 3 on #61).
            unidentified = [attempt.records[index] for index in chunk
                            if not isinstance(messages[index].get("_row_id"), int)]
            subscriber = Subscriber(wanted=still_wanted, hook=hook, deadline=deadline,
                                    deliver=self._deliver_for(attempt, chunk_handle, number, len(chunks),
                                                              records, unidentified),
                                    on_done=lambda number=number: finished.put(number))
            # What the worker runs, made before the call is looked up, so that a failure
            # of it is named as its own and not as the reuse read's (F3 of the pre-review).
            try:
                run = self._chunk_run(prepared[number], focus_topic=focus_topic, settings=settings,
                                      chunk_handle=chunk_handle)
            except Exception as exc:
                logger.warning("LCM could not prepare the summariser call of chunk %d of %d (%s: %s)",
                               number, len(chunks), type(exc).__name__, exc)
                self._record_event(attempt, "chunk_call_unprepared",
                                   {"chunk": chunk_handle, "error": f"{type(exc).__name__}: {exc}"})
                return self._abort(messages, f"the summariser call of chunk {number} of {len(chunks)} could not be "
                                             f"prepared ({type(exc).__name__}: {exc})")
            calls.append((number, chunk_handle, chunk, records, subscriber, run))
        return _Planned(route=route, chunks=chunks, kept_state=kept_state, mechanism=mechanism,
                        tail_start=tail_start, endpoint=endpoint, limiter=limiter, limit=limit,
                        still_wanted=still_wanted, describe=describe, finished=finished, calls=calls)

    def _dispatch_phase(self, attempt, messages: List[Dict[str, Any]], planned: "_Planned", *,
                        settings: CallSettings, focus_topic: Optional[str], started: float,
                        recovery_cap: Optional[int]) -> List[Dict[str, Any]]:
        """Steps 4 and 5, after the planning boundary: from the dispatch of the first call
        on, the calls, the wait and the return, each failure with its own handler (#7,
        #33). Everything a call needs was prepared inside the boundary."""
        route, chunks, kept_state = planned.route, planned.chunks, planned.kept_state
        mechanism, tail_start = planned.mechanism, planned.tail_start
        endpoint, limiter, limit = planned.endpoint, planned.limiter, planned.limit
        still_wanted, describe, finished = planned.still_wanted, planned.describe, planned.finished

        # 4. One summary per chunk, every chunk's call issued at once (#12, #33): each
        # joins the call in flight for the same records, reuses a summary of them already
        # written, or starts on a worker of its own.
        subscribers: List[Subscriber] = []
        ways: Dict[str, int] = {}
        for number, chunk_handle, chunk, records, subscriber, run in planned.calls:
            # A cancelled or no longer current attempt starts no further call (#29 W2 step 2).
            if not still_wanted():
                raise AttemptCancelled()
            # Attempts share a call only with the same summariser route and effort, the
            # rule a reuse applies (``summary_of_records``). The reuse reads the store; a
            # read that fails aborts the compaction (#7). Calls already started for
            # earlier chunks run on and write their summaries (#33 D13).
            try:
                way = join_or_start(
                    (attempt.session, tuple(records), route.model, route.provenance_provider(), settings.effort),
                    subscriber, limiter=limiter, limit=limit,
                    reuse=lambda records=records, chunk_handle=chunk_handle, frozen_summary=(
                        kept_state.get(tuple(chunk)) == "summarised"): self._records.summary_of_records(
                        attempt.session, records, exclude_chunk=chunk_handle, model=route.model,
                        provider=route.provenance_provider(), effort=settings.effort, any_route=frozen_summary),
                    run=run,
                    describe=describe,
                )
            except Exception as exc:
                return self._store_read_failed(attempt, messages, "summary_reuse_unreadable",
                                               f"a summary of chunk {number} of {len(chunks)} already written", exc)
            ways[way] = ways.get(way, 0) + 1
            subscribers.append(subscriber)
        logger.info("LCM compaction issued %d chunk%s to %s (%s), at most %d calls in flight there",
                    len(chunks), "" if len(chunks) == 1 else "s", endpoint,
                    ", ".join(f"{count} {way}" for way, count in sorted(ways.items())), limit)

        # The compress() thread waits for its chunks, asking the attempt's captured check
        # all the while (R4): cancelled or superseded, it returns its input at once and
        # sets nothing; the calls in flight run on and their summaries are written (D13).
        # A chunk that failed, or has no derivation, fails the compaction as a whole, the
        # context unchanged (#7), through _abort, never an exception; the summaries
        # written stay for the retry. Each outcome is read once, when its chunk reports
        # it complete, so no outcome is judged half-written.
        received: set = set()
        while len(received) < len(subscribers):
            if not still_wanted():
                raise AttemptCancelled()
            try:
                number = finished.get(timeout=_WAIT_SLICE_S)
            except queue.Empty:
                continue
            received.add(number)
            outcome = subscribers[number - 1].outcome
            if outcome is None or outcome.failure is not None or not outcome.derivation:
                # The failure was judged where it was recorded (ruling on #61, 4); this
                # thread only shows what that said.
                why = outcome.failure if outcome is not None and outcome.failure else "no summary was delivered"
                cause = outcome.cause if outcome is not None and outcome.cause else \
                    f"the summary of chunk {number} of {len(chunks)} failed ({why})"
                return self._abort(messages, cause)
        # The return is ordered by the chunks, whatever order the summaries arrived in.
        new_derivations: List[str] = [s.outcome.derivation for s in subscribers]

        # 5. The return, emitted from the record (#34 D5).
        previous = attempt.effective_returns
        cover = [previous[p][2] for p in sorted(previous) if previous[p][0] == "summary"] + new_derivations
        try:
            texts = self._records.derivations(cover)
        except Exception as exc:
            return self._store_read_failed(attempt, messages, "cover_unreadable", "the summaries of the return", exc)
        absent = [derivation for derivation in cover if derivation not in texts]
        if absent:
            # A summary the return stands on is not in the store: the chain would lose it.
            self._record_event(attempt, "cover_derivation_missing", {"derivations": absent})
            return self._abort(messages, f"the store holds no summary {', '.join(map(str, absent))} of the return")
        result: List[Dict[str, Any]] = []
        returns: List[tuple] = []
        if 0 in mechanism and messages[0].get("role") == "system":
            result.append(messages[0])  # the host's system row, in place, not recorded
        for derivation in cover:
            node_id, text = texts[derivation]
            row = {
                "role": "user",
                "content": "\n".join((_SUMMARY_HEADER.format(node_id=node_id), text,
                                      _SUMMARY_FOOTER.format(node_id=node_id))),
                "_compressed_summary": True,
            }
            returns.append((len(result), "summary", None, derivation, raw_json(row)))
            result.append(row)
        for index in range(tail_start, len(messages)):
            record = attempt.records.get(index)
            if record is None:
                self._record_event(attempt, "tail_entry_without_record", {"index": index})
                return self._abort(messages, "an entry of the fresh tail has no record in the store")
            returns.append((len(result), "record", record, None, None))
            result.append(messages[index])

        # The return fence on the captured check (#29 W2 steps 2 and 6): asked here, and
        # inside the return's transaction once its write lock is granted (_write_return).
        self._require_live_write()
        try:
            self._write_return(attempt, result, returns)
        except Exception as exc:
            logger.warning("LCM could not write the return", exc_info=True)
            self._record_event(attempt, "return_write_failed", repr(exc))
            return self._abort(messages, f"the store could not write the return ({exc})")

        number = self._publish_compaction_counted()
        duration_ms = (time.perf_counter() - started) * 1000.0
        self._publish("_last_compression_status", "compacted")
        self._publish("_last_compression_noop_reason", "")
        self._publish("_last_compress_aborted", False)
        self._publish("_last_summary_error", None)
        # The host sets its re-arm flag from this after the commit, and re-arms its
        # per-turn count when the next prompt is below threshold_tokens (#32 §9).
        self._publish("_last_compression_made_progress", True)
        # The session's context, by the session's estimate, as the recovery cap is
        # computed (``_overflow_recovery_assembly_cap``), never the summariser's.
        session_estimator = self._estimator()
        before, after = session_estimator.messages(messages), session_estimator.messages(result)
        over_cap = recovery_cap is not None and after.tokens > recovery_cap
        self._publish("_last_overflow_recovery_failed", over_cap)
        if over_cap:
            logger.warning(
                "LCM overflow recovery left the context above the cap (%d > %d tokens, %s); nothing is cut to fit",
                after.tokens, recovery_cap, after.label(),
            )
        logger.info(
            "LCM compaction #%d: %d entries -> %d (%d chunk%s, %d summaries in the cover, %d -> %d tokens, "
            "%s -> %s, %.1fms)",
            number, len(messages), len(result), len(chunks), "" if len(chunks) == 1 else "s", len(cover),
            before.tokens, after.tokens, before.label(), after.label(), duration_ms,
        )
        return result
