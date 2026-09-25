"""The compaction path (#29 W2 and W3; #34 D5).

``compress()`` runs one attempt against the plugin's record, which is the only store.
In order:

1. The host's list is classified against the session's effective return by identity
   only: the host's ``_row_id`` and the plugin's in-memory key, never by content.
   Where it cannot be, the compaction is aborted: nothing is written, the host keeps
   its list and shows the cause.
2. The tail is the host's list from ``resolve_fresh_tail_boundary`` on, and never
   reaches back into the summaries the plugin returned. The material is every entry
   before the tail that is not the mechanism's layer (the host's system row, a
   summary the plugin returned, also as the host rewrote it: a summary revision holds
   a summary and is never chunked). The host's count decides pressure: when the host
   calls with its count at or above the threshold, or forces, or recovers from an
   overflow, the plugin's estimate vetoes nothing. No minimum material applies, and
   where the tail leaves no material it yields down to its floor, the newest message
   or the newest tool group inside a turn; only when the floor alone is left is
   there nothing to compact, and that is shown as an abort.
3. The material is cut into chunks, in list order, each at most ``leaf_chunk_tokens``
   (a greedy cut until #12), only between groups: a tool call and its results stay in
   one chunk, and a group larger than a chunk is a chunk of its own. A run smaller
   than a quarter of a chunk next to such a group, or at the end of the material,
   joins the adjacent chunk instead of standing alone. The compaction,
   its inputs, the new records and every chunk are written before the first
   summariser call.
4. Each chunk is summarised from its records as the store holds them. Every chunk's
   call is issued at once, on daemon workers of the plugin's own, through one limiter
   per endpoint and process; a call for the same records already in flight is joined,
   a summary of them already written is reused (``inflight``, #33). Each summary is
   written as a derivation when it arrives, in any order. The ``compress()`` thread
   waits, asking the attempt's captured check, and returns its input at once when the
   attempt is cancelled or superseded. A summary that cannot be written
   (``escalation.SummaryFailure``: no third level, nothing truncated) fails the
   compaction as a whole: the context stays as it was, and the host shows the cause.
5. The return is emitted from the record: the host's system row in place, then the
   cover (the previous return's summaries, each re-emitted from its derivation, then
   the new ones in chunk order), then the tail as the host's own dicts. It is
   recorded exactly as returned.

There is no condensation in this path: the cover grows by one summary per chunk until
#34 builds condensation.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional

from .escalation import (
    REASONING_EFFORTS,
    CallPath,
    CallSettings,
    SummariserRoute,
    SummaryFailure,
    _host_provider,
    configured_route_problem,
    failure_text,
    session_route,
    summarize_chunk,
)
from .model_table import lookup as lookup_model
from .summariser_input import wire_facts
from .message_analysis import _tool_call_id
from .record_store import RET_KEY, parse_ret_key, raw_json
from .fresh_tail import _assistant_group_start
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

# The words around a summary row (Decision 8, until #10 writes them).
_SUMMARY_HEADER = "[Recent Summary (d0, node {node_id})]"
_SUMMARY_FOOTER = "[Expand for details: {hint}]"

# How often the compress() thread, waiting for its chunks, asks the attempt's captured check.
_WAIT_SLICE_S = 0.25

# "The smallest run that stands alone", as a share of the chunk size c: a value for
# #22's table (orchestrator ruling on #52). A smaller run next to an oversized group,
# or at the end of the material, joins the adjacent chunk.
_SMALLEST_STANDALONE_RUN = 0.25


class CompactionMixin:
    def should_compress(self, prompt_tokens: int = None) -> bool:
        tokens = prompt_tokens if prompt_tokens is not None else self.last_prompt_tokens
        if self._should_force_overflow_recovery(observed_tokens=tokens):
            return True
        if self.threshold_tokens <= 0:
            return False
        return tokens >= self.threshold_tokens

    def should_compress_preflight(self, messages):
        """Before a request: settle and bind what the list shows, then ask for a
        compaction when the prompt is over the threshold and the material outside the
        tail reaches a chunk. Nothing else is written here."""
        self._bind_from_list(messages)
        # The plugin's estimate (#21), compared with a threshold in provider tokens: on
        # Claude it reads about 1.5 times lower (#31), so this gate fires later than the
        # host's own pressure would; the conversion is PR-5's (R9).
        rough = count_messages_tokens(messages, self._estimator())
        if self._should_force_overflow_recovery(observed_tokens=rough, messages=messages):
            return self._mark_preflight_compression_requested()
        if self.threshold_tokens > 0 and rough >= self.threshold_tokens:
            mechanism = self._mechanism_positions(messages)
            tail_start = self._tail_start(messages, mechanism)
            material = self._estimator().messages(
                [messages[index] for index in range(tail_start) if index not in mechanism]
            )
            if material.tokens >= self._config.leaf_chunk_tokens:
                return self._mark_preflight_compression_requested()
            # Here no count of the host's exists yet, so the estimate decides, and says
            # so. The host's count after the request decides again (``compress``).
            reason = (f"the material outside the fresh tail is below one leaf chunk by the plugin's estimate "
                      f"({material.tokens} < {self._config.leaf_chunk_tokens} tokens, {material.label()})")
            self._last_compression_status = "noop"
            self._last_compression_noop_reason = reason
            logger.info("LCM preflight compression no-op: %s", reason)
        return False

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

    def _tail_start(self, messages: List[Dict[str, Any]], mechanism: set) -> int:
        """Where the fresh tail begins: the configured boundary, never before the last
        entry of the mechanism's layer, which the cover re-emits."""
        boundary = self._fresh_tail_start(messages)
        return max(boundary, max(mechanism) + 1 if mechanism else 0)

    @staticmethod
    def _tail_floor(messages: List[Dict[str, Any]], mechanism: set) -> int:
        """The least the tail keeps under the host's pressure: the newest message, or,
        where the list ends inside a tool group, that group from the assistant that
        opened it; never before the last entry of the mechanism's layer."""
        start = _assistant_group_start(messages, len(messages) - 1)
        return max(start, max(mechanism) + 1 if mechanism else 0)

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
        logger.info("LCM compression no-op: %s", reason)
        return messages

    def _abort(self, messages: List[Dict[str, Any]], cause: str) -> List[Dict[str, Any]]:
        """The compaction cannot be done as the model requires: the host keeps its list
        and shows "⚠ Compression aborted: <cause>" (#29 W2 step 1)."""
        self._publish("_last_compression_status", "aborted")
        self._publish("_last_compression_noop_reason", cause)
        self._publish("_last_compress_aborted", True)
        self._publish("_last_summary_error", cause)
        logger.warning("LCM compaction aborted: %s", cause)
        return messages

    def _summariser_settings(self) -> tuple[Optional[CallSettings], str]:
        """The summariser's route, effort and output cap, or the reason there is none
        (#9). No part of the route is guessed: the session's route is what the host
        handed ``update_model``; a configured one must name its provider."""
        config = self._config
        problem = configured_route_problem(config)
        if problem is not None:
            # Refused when the configuration was loaded, and every compaction says why.
            return None, problem
        secrets = tuple(s for s in (config.summary_api_key, self.api_key) if isinstance(s, str) and s)
        if config.summary_model:
            route = SummariserRoute(
                provider=config.summary_provider.strip(), model=config.summary_model.strip(),
                base_url=config.summary_base_url.strip(), api_key=config.summary_api_key,
                api_mode=config.summary_api_mode.strip(), source="configured",
            )
        else:
            if not (self.model and self.provider):
                return None, ("the session's model is not known: the host has not named its route "
                              "through update_model, and no summariser is configured")
            if not self.base_url and _host_provider(self.provider) == "custom":
                return None, ("the session's route names provider custom without a base URL: the host "
                              "would borrow an endpoint of its own, which the session did not name")
            route = session_route(self.provider, self.model, self.base_url, self.api_key, self.api_mode)
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
        facts = lookup_model(route.model)
        return CallSettings(
            route=route,
            effort=effort,
            max_tokens=facts.output_cap if facts is not None else None,
            secrets=secrets,
        ), ""

    def compress(self, messages: List[Dict[str, Any]],
                 current_tokens: int = None,
                 focus_topic: Optional[str] = None,
                 force: bool = False) -> List[Dict[str, Any]]:
        """Run one compaction attempt.

        The host's cancellation check and the attempt's generation are captured here,
        at entry, on this attempt's own thread (#33 D12). A cancelled attempt returns
        its input at once and sets nothing. Engine attributes are set only when the
        attempt returns and is the host's current working attempt; a terminal status
        is left on every failure under the same rule.
        """
        attempt = self._begin_attempt()
        if attempt.cancelled():
            return messages
        token = _ATTEMPT.set(attempt)
        try:
            result = self._compress_impl(
                messages,
                current_tokens=current_tokens,
                focus_topic=focus_topic,
                force=force,
            )
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

    def _chunk_run(
        self,
        chunk_messages: List[Dict[str, Any]],
        *,
        focus_topic: Optional[str],
        record_handles: List[str],
        settings: CallSettings,
    ) -> Callable[[ChunkCall], ChunkSummary]:
        """What a worker runs for one chunk: its summary, or ``SummaryFailure`` (#7),
        with the summariser's route and effort (#9). Everything the call needs is read
        here, on the ``compress()`` thread; the worker reads nothing of the engine.

        The summariser reads the chunk's records as the messages they were, whole
        (#8, ``summariser_input``). The budget is a target in the prompt text, never an
        output limit.
        """
        # What the summary replaces in the session's context, by the session's estimate
        # (R6): the acceptance compares the reply with this, by the same estimate.
        source = self._estimator().messages(chunk_messages)
        budget = min(max(2000, int(source.tokens * 0.20)), 12000)
        facts = lookup_model(settings.route.model)
        route = settings.route
        records = list(zip(record_handles, chunk_messages))
        # Images go in only where the model table says the summariser reads them; the
        # reasoning field and the converter by the host's rules for the route.
        wire = wire_facts(route.provider, route.model, route.base_url, route.api_mode,
                          reads_images=bool(facts is not None and facts.reads_images))
        custom_instructions = self._config.custom_instructions

        def run(call: ChunkCall) -> ChunkSummary:
            text, level, finish_reason = summarize_chunk(
                records,
                budget,
                source=source,
                settings=settings,
                facts=wire,
                depth=0,
                focus_topic=focus_topic or "",
                custom_instructions=custom_instructions,
                path=CallPath(wait=call.wait, dispatch=call.dispatch, hold=call.limiter.hold,
                              deadline=call.deadline),
            )
            return ChunkSummary(text=text, level=level, budget=budget, finish_reason=finish_reason,
                                model=route.model, provider=route.provenance_provider(), effort=settings.effort)

        return run

    def _calls_in_flight_limit(self, endpoint: str) -> int:
        """The endpoint's limit: its own where configured, else the default (#33)."""
        per_endpoint = self._config.summary_calls_per_endpoint or {}
        value = per_endpoint.get(endpoint, self._config.summary_calls_in_flight)
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            return DEFAULT_CALLS_PER_ENDPOINT

    def _deliver_for(self, attempt, chunk_handle: str, number: int, total: int):
        """How one attempt's chunk receives its call's outcome, on whichever thread has
        it: the summary is written as a derivation of this chunk (#33 Q17a: not fenced,
        a fact about the chunk's content); a failure is recorded as a store event."""

        def deliver(summary: Optional[ChunkSummary], failure: Optional[str], abandoned: bool) -> Outcome:
            if summary is not None:
                try:
                    derivation = self._write_summary(
                        attempt, chunk_handle, text=summary.text, level=summary.level, budget=summary.budget,
                        finish_reason=summary.finish_reason, expand_hint=self._extract_expand_hint(summary.text),
                        model=summary.model, provider=summary.provider, effort=summary.effort,
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
                logger.warning("LCM summary of chunk %d of %d failed: %s", number, total, failure)
                self._record_event(attempt, "summary_failed",
                                   {"chunk": chunk_handle, "number": number, "error": failure})
            return Outcome(failure=failure)

        return deliver

    @staticmethod
    def _groups(messages: List[Dict[str, Any]], material: List[int]) -> List[List[int]]:
        """The material in list order as groups a cut may not enter: an assistant row
        that carries tool calls, with every result of those calls in the material and
        whatever stands between them; every other entry alone. Calls and results are
        matched by their ``tool_call_id``, the host's identity for the pair."""
        groups: List[List[int]] = []
        at = 0
        while at < len(material):
            message = messages[material[at]]
            call_ids = {_tool_call_id(call) for call in (message.get("tool_calls") or [])} - {""}
            end = at
            if message.get("role") == "assistant" and call_ids:
                for later in range(at + 1, len(material)):
                    other = messages[material[later]]
                    if other.get("role") == "tool" and str(other.get("tool_call_id") or "").strip() in call_ids:
                        end = later
            groups.append(material[at:end + 1])
            at = end + 1
        return groups

    @classmethod
    def _cut_chunks(cls, messages: List[Dict[str, Any]], material: List[int], limit: int,
                    estimator: Optional[Estimator] = None) -> List[List[int]]:
        """The material cut in list order into chunks of at most ``limit`` tokens, and
        only between groups: a tool call is never separated from its results. A group
        larger than the limit is a chunk of its own (#31: a chunk flexes by one group).

        A run smaller than ``limit`` × ``_SMALLEST_STANDALONE_RUN`` does not stand alone
        where it stands next to such an oversized group, or at the end of the material:
        it joins the adjacent chunk, never jumping over a group, so the chunks stay
        contiguous. A chunk exists to bound the summariser's input; a run too small to
        have a shorter summary would fail on every attempt (#7, orchestrator ruling on
        #52). Next to an oversized group on both sides, it joins the following one,
        the work it opened. A run that is the whole material has nothing to join.
        """
        chunks: List[List[int]] = []
        sizes: List[int] = []
        oversized: List[bool] = []
        current: List[int] = []
        used = 0
        for group in cls._groups(messages, material):
            tokens = sum(count_message_tokens(messages[index], estimator) for index in group)
            if current and used + tokens > limit:
                chunks.append(current)
                sizes.append(used)
                oversized.append(False)
                current, used = [], 0
            current.extend(group)
            used += tokens
            if used > limit and len(current) == len(group):
                # A group larger than the limit: a chunk of its own.
                chunks.append(current)
                sizes.append(used)
                oversized.append(True)
                current, used = [], 0
        if current:
            chunks.append(current)
            sizes.append(used)
            oversized.append(False)

        smallest = limit * _SMALLEST_STANDALONE_RUN
        at = 0
        while at < len(chunks):
            if sizes[at] >= smallest or oversized[at] or len(chunks) == 1:
                at += 1
                continue
            if at + 1 < len(chunks) and oversized[at + 1]:
                target = at + 1
            elif at > 0 and oversized[at - 1]:
                target = at - 1
            elif at == len(chunks) - 1:
                target = at - 1
            else:
                at += 1
                continue
            if target > at:
                chunks[target] = chunks[at] + chunks[target]
            else:
                chunks[target] = chunks[target] + chunks[at]
            sizes[target] += sizes[at]
            del chunks[at], sizes[at], oversized[at]
            at = max(0, min(at, target))
        return chunks

    def _compress_impl(self, messages: List[Dict[str, Any]],
                       current_tokens: int = None,
                       focus_topic: Optional[str] = None,
                       force: bool = False) -> List[Dict[str, Any]]:
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

        # 1. Identity (#29 W3). A list that cannot be classified is not compacted.
        self._settle_from_list(messages)
        entries = self._classify(attempt, messages)
        if entries is None:
            return self._abort(messages, attempt.error[1] if attempt.error else "the list could not be classified")
        mechanism = {entry.position for entry in entries if entry.klass == "system"}
        mechanism |= attempt.summary_inputs

        # 2. The tail and the material.
        tail_start = self._tail_start(messages, mechanism)
        material = [index for index in range(tail_start) if index not in mechanism]
        estimator = self._estimator()
        # The host's count decides pressure; the plugin's estimate reads lower than the
        # provider's count (#31) and never vetoes a compaction that count requires.
        # ``current_tokens`` is what the host hands its compaction: its real count at
        # most call sites, its own rough estimate at some (turn_recovery.py:1839); either
        # way the host's, not the plugin's.
        host_pressure = (
            current_tokens is not None and self.threshold_tokens > 0 and current_tokens >= self.threshold_tokens
        )
        pressure = bool(force or force_overflow or host_pressure)
        if not material and pressure:
            floor = self._tail_floor(messages, mechanism)
            if floor > tail_start:
                logger.info("LCM fresh tail yields to its floor under the host's pressure: %d entries -> %d",
                            len(messages) - tail_start, len(messages) - floor)
                tail_start = floor
                material = [index for index in range(tail_start) if index not in mechanism]
        if not material:
            if force_overflow:
                self._publish("_last_overflow_recovery_failed", True)
            if pressure:
                why = (f"the host's count {current_tokens} is at or above the threshold {self.threshold_tokens}"
                       if host_pressure else "the compaction was forced" if force
                       else "the context overflowed")
                return self._abort(
                    messages,
                    f"{why}, and nothing is left to compact: outside the newest "
                    f"{'tool group' if len(messages) - tail_start > 1 else 'message'} "
                    f"stands only the mechanism's layer (the system row and the summaries)",
                )
            return self._unchanged_return(messages, "no material outside the fresh tail")
        material_estimate = estimator.messages([messages[index] for index in material])
        if material_estimate.tokens < self._config.leaf_chunk_tokens and not pressure:
            return self._unchanged_return(
                messages,
                f"without the host's pressure, the material outside the fresh tail is below one leaf chunk "
                f"by the plugin's estimate ({material_estimate.tokens} < {self._config.leaf_chunk_tokens} "
                f"tokens, {material_estimate.label()})",
            )

        # The summariser, as far as anything can be written: its route and effort (#9).
        settings, why_not = self._summariser_settings()
        if settings is None:
            return self._abort(messages, why_not)

        # 3. Transaction 1: the compaction, its inputs, the new records and every chunk.
        chunks = self._cut_chunks(messages, material, max(1, int(self._config.leaf_chunk_tokens)), estimator)
        # A chunk the summariser cannot read in one call would fail on every attempt; it
        # is never cut (the tiny-chunk rule on #52). Only where the model table knows
        # the window; counted by the plugin's estimate, its images by the summariser's rule.
        model_facts = lookup_model(settings.route.model)
        if model_facts is not None and model_facts.context_window:
            room = model_facts.context_window - (model_facts.output_cap or 0)
            summariser_estimate = Estimator(image_model=settings.route.model)
            for number, chunk in enumerate(chunks, start=1):
                estimate = summariser_estimate.messages([messages[index] for index in chunk])
                if estimate.tokens > room:
                    return self._abort(
                        messages,
                        f"chunk {number} of {len(chunks)} holds about {estimate.tokens} tokens "
                        f"({estimate.label()}), more than the summariser {settings.route.describe()} can read "
                        f"in one call ({model_facts.context_window} window less {model_facts.output_cap or 0} "
                        f"output)",
                    )
        try:
            chunk_handles = self._write_compaction(attempt, entries, chunks, force=force)
        except Exception as exc:
            logger.warning("LCM could not write the compaction", exc_info=True)
            self._record_event(attempt, "compaction_write_failed", repr(exc))
            return self._abort(messages, f"the store could not write the compaction ({exc})")

        # 4. One summary per chunk, from the chunk's records as stored, every chunk's call
        # issued at once (#12, #33): each joins the call in flight for the same records,
        # reuses a summary of them already written, or starts on a worker of its own.
        members = [attempt.records[index] for chunk in chunks for index in chunk]
        facts = self._records.record_facts(members)
        route = settings.route
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

        subscribers: List[Subscriber] = []
        ways: Dict[str, int] = {}
        for number, (chunk_handle, chunk) in enumerate(zip(chunk_handles, chunks), start=1):
            # A cancelled or no longer current attempt starts no further call (#29 W2 step 2).
            if not still_wanted():
                raise AttemptCancelled()
            records = [attempt.records[index] for index in chunk]
            chunk_messages = [json.loads(facts[record][1]) for record in records]
            subscriber = Subscriber(wanted=still_wanted, hook=hook, deadline=deadline,
                                    deliver=self._deliver_for(attempt, chunk_handle, number, len(chunks)))
            way = join_or_start(
                (attempt.session, tuple(records)), subscriber, limiter=limiter, limit=limit,
                reuse=lambda records=records, chunk_handle=chunk_handle: self._records.summary_of_records(
                    attempt.session, records, exclude_chunk=chunk_handle, model=route.model,
                    provider=route.provenance_provider(), effort=settings.effort),
                run=self._chunk_run(chunk_messages, focus_topic=focus_topic, record_handles=records,
                                    settings=settings),
                describe=describe,
            )
            ways[way] = ways.get(way, 0) + 1
            subscribers.append(subscriber)
        logger.info("LCM compaction issued %d chunk%s to %s (%s), at most %d calls in flight there",
                    len(chunks), "" if len(chunks) == 1 else "s", endpoint,
                    ", ".join(f"{count} {way}" for way, count in sorted(ways.items())), limit)

        # The compress() thread waits for its chunks, asking the attempt's captured check
        # all the while (R4): cancelled or superseded, it returns its input at once and
        # sets nothing; the calls in flight run on and their summaries are written (D13).
        # A chunk that failed fails the compaction as a whole, the context unchanged (#7);
        # the summaries written stay for the retry.
        while True:
            if not still_wanted():
                raise AttemptCancelled()
            for number, subscriber in enumerate(subscribers, start=1):
                if subscriber.done.is_set() and subscriber.outcome is not None \
                        and subscriber.outcome.failure is not None:
                    return self._abort(
                        messages, f"the summary of chunk {number} of {len(chunks)} failed "
                                  f"({subscriber.outcome.failure})")
            pending = [s for s in subscribers if not s.done.is_set()]
            if not pending:
                break
            pending[0].done.wait(_WAIT_SLICE_S)
        # The return is ordered by the chunks, whatever order the summaries arrived in.
        new_derivations: List[str] = [s.outcome.derivation for s in subscribers]

        # 5. The return, emitted from the record (#34 D5).
        previous = attempt.effective_returns
        cover = [previous[p][2] for p in sorted(previous) if previous[p][0] == "summary"] + new_derivations
        texts = self._records.derivations(cover)
        result: List[Dict[str, Any]] = []
        returns: List[tuple] = []
        if 0 in mechanism and messages[0].get("role") == "system":
            result.append(messages[0])  # the host's system row, in place, not recorded
        for derivation in cover:
            node_id, text, hint = texts[derivation]
            row = {
                "role": "user",
                "content": "\n".join((_SUMMARY_HEADER.format(node_id=node_id), text,
                                      _SUMMARY_FOOTER.format(hint=hint or ""))),
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

        # The return fence on the captured check (#29 W2 steps 2 and 6).
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
        before, after = estimator.messages(messages), estimator.messages(result)
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
