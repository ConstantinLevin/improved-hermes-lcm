"""Leaf-compaction pipeline for the LCM engine (WS5 Seam 6).

The ``CompactionMixin`` holds the compaction gate + pipeline: ``should_compress``
/ ``should_compress_preflight`` (public), the leaf-candidate and chunk-selection
helpers, and the main ``compress`` entry point. These methods were lifted
verbatim out of ``LCMEngine`` and continue to run bound to the engine instance
(``self`` is the ``LCMEngine``), so they read and write the engine's runtime
state (``_ingest_cursor``, ``_store``, ``_dag``, ``_lifecycle``, status/telemetry
fields, per-turn caches) and call back into engine helpers (ingest,
reconciliation, placeholder-ledger, the summarize-with-rescue step, assembly,
lifecycle) through normal attribute lookup. ``LCMEngine`` mixes this in ahead of
``ContextEngine`` so the mixin's ``compress`` / ``should_compress`` /
``should_compress_preflight`` override the ContextEngine protocol defaults.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from .dag import SummaryNode
from .message_content import text_content_for_pattern_matching
from .record_write import _ATTEMPT, AttemptCancelled
from .tokens import count_message_tokens, count_messages_tokens, count_tokens

logger = logging.getLogger(__name__)

_THRESHOLD_FULL_SWEEP_MAX_PASSES = 12
_THRESHOLD_FULL_SWEEP_MAX_SECONDS = 120.0


class CompactionMixin:
    def should_compress(self, prompt_tokens: int = None) -> bool:
        if self._compression_boundary_cooldown_active():
            return False
        tokens = prompt_tokens if prompt_tokens is not None else self.last_prompt_tokens
        if self._should_force_overflow_recovery(observed_tokens=tokens):
            return True
        if self.threshold_tokens <= 0:
            return False
        return tokens >= self.threshold_tokens

    def should_compress_preflight(self, messages):
        """Pre-flight check — also ingests messages into the store."""
        self._bind_from_list(messages)
        self._preflight_cleanup_only_due_to_boundary_cooldown = False
        rough = count_messages_tokens(messages)
        replay_messages = None
        if self._session_id and messages:
            try:
                replay_messages = self._ingest_messages(messages)
                self._record_ingest_success()
            except Exception as e:
                # Fail closed for NORMAL threshold compaction: the store did not
                # accept this turn, so do not compact against a store missing the
                # latest messages - that could rebuild active context without
                # them. But still honor emergency overflow recovery, whose whole
                # job is to keep the prompt under the provider limit; it converges
                # via deterministic L3 truncation without needing the store write.
                self._record_ingest_failure("preflight", e)
                if self._should_force_overflow_recovery(observed_tokens=rough):
                    return True
                return False
        if replay_messages is not None and replay_messages != messages:
            replay_rough = count_messages_tokens(replay_messages)
            cleanup_requested = self._replay_diff_requests_ingest_cleanup(
                messages,
                replay_messages,
            )
            force_overflow_requested = self._should_force_overflow_recovery(
                observed_tokens=rough,
                messages=messages,
            ) or self._should_force_overflow_recovery(
                observed_tokens=replay_rough,
                messages=replay_messages,
            )
            if cleanup_requested:
                if (
                    not force_overflow_requested
                    and self._compression_boundary_cooldown_active()
                ):
                    self._preflight_cleanup_only_due_to_boundary_cooldown = True
                return self._mark_preflight_compression_requested()
            if force_overflow_requested:
                return self._mark_preflight_compression_requested()
            # A boundary skip cools down summary-producing leaf/condensation
            # work. It must not prevent the host from adopting a replay cleanup
            # that ingest has already made durable (for example a live tool
            # result stub); those returns above are deterministic and add no
            # summarizer spend.
            if self._compression_boundary_cooldown_active():
                return False
            eligible, reason = self._leaf_compaction_candidate_status(
                replay_messages,
                allow_partial_leaf=bool(
                    self._config.threshold_full_sweep_enabled
                    and self.threshold_tokens > 0
                    and replay_rough >= self.threshold_tokens
                ),
            )
            if eligible:
                return self._mark_preflight_compression_requested()
            if self.threshold_tokens > 0 and replay_rough >= self.threshold_tokens:
                self._last_compression_status = "noop"
                self._last_compression_noop_reason = reason
                logger.info("LCM preflight compression no-op: %s", reason)
            return False
        if self._compression_boundary_cooldown_active():
            return False
        if self._should_force_overflow_recovery(observed_tokens=rough):
            return self._mark_preflight_compression_requested()
        if self.threshold_tokens > 0 and rough >= self.threshold_tokens:
            eligible, reason = self._leaf_compaction_candidate_status(
                messages,
                allow_partial_leaf=self._config.threshold_full_sweep_enabled,
            )
            if eligible:
                return self._mark_preflight_compression_requested()
            self._last_compression_status = "noop"
            self._last_compression_noop_reason = reason
            logger.info("LCM preflight compression no-op: %s", reason)
        return False

    def _replay_diff_requests_ingest_cleanup(
        self,
        original_messages: List[Dict[str, Any]],
        replay_messages: List[Dict[str, Any]],
    ) -> bool:
        if len(original_messages) != len(replay_messages):
            return True
        for original_msg, replay_msg in zip(original_messages, replay_messages):
            original_text = text_content_for_pattern_matching(original_msg.get("content")) or ""
            replay_text = text_content_for_pattern_matching(replay_msg.get("content")) or ""
            if original_text != replay_text:
                if replay_text.startswith("[Externalized LCM ingest payload:"):
                    return True
                if replay_text.startswith("[LCM active replay placeholder: assistant output quarantined;"):
                    return True
        return False

    def _leaf_compaction_candidate_status(
        self,
        messages: List[Dict[str, Any]],
        *,
        force_overflow: bool = False,
        allow_partial_leaf: bool = False,
    ) -> tuple[bool, str]:
        """Return whether a normal leaf compaction pass can actually run.

        The host asks ``should_compress_preflight`` before it emits user-visible
        compression status. A session can be over the global context threshold
        while all pressure sits in the protected fresh tail, or while the raw
        backlog outside that tail is still smaller than the configured leaf
        chunk. In that case ``compress()`` would immediately no-op, so preflight
        should not advertise a compaction attempt yet.
        """
        if not messages:
            return False, "empty message list"
        fresh_tail_start = self._fresh_tail_start(messages)
        leading_anchor_count = self._leading_anchor_count(messages)
        if fresh_tail_start <= leading_anchor_count:
            return False, "no eligible raw backlog outside fresh tail"

        candidate_raw = messages[leading_anchor_count:fresh_tail_start]
        if not candidate_raw:
            return False, "no eligible raw backlog outside fresh tail"
        generated_placeholder_hashes = self._load_generated_ignored_placeholder_hashes()
        if generated_placeholder_hashes:
            filtered_candidate_raw: list[Dict[str, Any]] = []
            for msg in candidate_raw:
                content_text = text_content_for_pattern_matching(msg.get("content")) or ""
                volatile_digest = self._active_replay_placeholder_digest(content_text)
                if (
                    self._is_volatile_ignored_quarantine_placeholder(msg, content_text)
                    and volatile_digest is not None
                    and volatile_digest in generated_placeholder_hashes
                ):
                    continue
                filtered_candidate_raw.append(msg)
            candidate_raw = filtered_candidate_raw
            if not candidate_raw:
                return False, "no eligible raw backlog outside fresh tail"

        if force_overflow:
            return True, "forced overflow recovery"

        raw_tokens_outside_tail = count_messages_tokens(candidate_raw)
        if allow_partial_leaf:
            return True, "eligible partial threshold-sweep leaf"
        if self._config.dynamic_leaf_chunk_enabled:
            working_leaf_chunk_tokens = self._working_leaf_chunk_tokens(raw_tokens_outside_tail)
        else:
            working_leaf_chunk_tokens = self._config.leaf_chunk_tokens
        if raw_tokens_outside_tail < working_leaf_chunk_tokens:
            return False, "raw backlog outside fresh tail is below leaf chunk threshold"
        return True, "eligible raw backlog outside fresh tail"

    def _working_leaf_chunk_tokens(self, raw_tokens_outside_tail: int) -> int:
        base = max(1, self._config.leaf_chunk_tokens)
        if not self._config.dynamic_leaf_chunk_enabled:
            return base
        ceiling = max(base, self._config.dynamic_leaf_chunk_max)
        working = base
        while working < ceiling and raw_tokens_outside_tail > working * 2:
            working = min(ceiling, working * 2)
        return working

    def _select_oldest_leaf_chunk(
        self,
        candidate_raw: List[Dict[str, Any]],
        working_leaf_chunk_tokens: int,
    ) -> List[Dict[str, Any]]:
        selected: list[Dict[str, Any]] = []
        used = 0
        for msg in candidate_raw:
            msg_tokens = count_message_tokens(msg)
            if used + msg_tokens > working_leaf_chunk_tokens and selected:
                break
            selected.append(msg)
            used += msg_tokens
        return selected

    def _unchanged_return(self, messages: List[Dict[str, Any]], reason: str) -> List[Dict[str, Any]]:
        """No compaction: the host's own list, the same object with the same dicts.

        Nothing is re-assembled, sanitised or dropped: when no compaction happens the
        context stays untouched. The host then logs "made no progress" and commits
        nothing (``_candidate_rejected``, ``agent/conversation_compression.py:3547-3563``).
        """
        self._publish("_last_compression_status", "noop")
        self._publish("_last_compression_noop_reason", reason)
        logger.info("LCM compression no-op: %s", reason)
        return messages

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
            _ATTEMPT.reset(token)
        if attempt.cancelled():
            return messages
        if self._attempt_is_current(attempt):
            self._apply_attempt_outcome(attempt)
        return result

    def _compress_impl(self, messages: List[Dict[str, Any]],
                       current_tokens: int = None,
                       focus_topic: Optional[str] = None,
                       force: bool = False) -> List[Dict[str, Any]]:
        """Main compaction entry point.

        1. Ingest any new messages into the store
        2. Identify messages outside the fresh tail
        3. Summarize them into DAG leaf nodes
        4. Check if condensation is needed
        5. Assemble new active context: summaries + fresh tail
        """
        attempt = _ATTEMPT.get()
        if not messages:
            self._publish("_last_compression_status", "noop")
            self._publish("_last_compression_noop_reason", "empty message list")
            return messages

        _compress_started = time.perf_counter()

        observed_prompt_tokens = current_tokens if current_tokens is not None else None
        force_overflow = self._should_force_overflow_recovery(
            observed_tokens=observed_prompt_tokens,
            messages=messages,
        )
        # NOTE: deliberately do NOT clear the spend guard on force_overflow.
        # force_overflow is automatic (set every turn the prompt exceeds the
        # assembly cap), which is exactly the sustained-over-cap state a runaway
        # compaction loop produces - clearing it per turn would defeat the guard
        # in the case it exists for. A tripped guard still converges the
        # emergency via deterministic L3 truncation (no LLM spend).
        recovery_assembly_cap = (
            self._overflow_recovery_assembly_cap(
                observed_tokens=observed_prompt_tokens,
                messages=messages,
            )
            if force_overflow
            else None
        )

        # Step 1: Ingest new messages into the immutable store. Work from a
        # replay-safe view so quarantined assistant loops do not enter summaries
        # or provider context after the durable row has been written.
        working_messages = self._ingest_messages(messages)
        # The shadow record maps each working copy back to the host's entry by
        # object identity, taken here, before any pass re-slices the working list.
        index_by_working = (
            {id(message): index for index, message in enumerate(working_messages)}
            if len(working_messages) == len(messages)
            else None
        )
        if attempt is not None:
            attempt.messages = messages
            attempt.index_by_working = index_by_working
            if index_by_working is None:
                self._shadow_failed(
                    attempt,
                    "working_list_misaligned",
                    f"{len(working_messages)} working entries for {len(messages)} host entries",
                )
        cleanup_only_due_to_boundary_cooldown = bool(
            self._preflight_cleanup_only_due_to_boundary_cooldown
            and not force_overflow
        )
        self._preflight_cleanup_only_due_to_boundary_cooldown = False
        if cleanup_only_due_to_boundary_cooldown:
            return self._unchanged_return(
                messages,
                "boundary cooldown: no compaction, so the context is returned unchanged",
            )
        anchor_source_messages = list(working_messages)
        pressure_messages = messages if len(messages) == len(working_messages) else working_messages
        leaf_compacted_this_turn = False
        leaf_passes = 0
        estimated_active_tokens = (
            observed_prompt_tokens
            if observed_prompt_tokens is not None and observed_prompt_tokens > 0
            else count_messages_tokens(messages)
        )
        threshold_full_sweep_active = bool(
            self._config.threshold_full_sweep_enabled
            and not force_overflow
            and self.threshold_tokens > 0
            and estimated_active_tokens >= self.threshold_tokens
        )
        sweep_deadline = time.monotonic() + _THRESHOLD_FULL_SWEEP_MAX_SECONDS
        configured_sweep_target = int(self._config.summary_prefix_target_tokens)
        sweep_target_tokens = max(
            1,
            configured_sweep_target
            if configured_sweep_target > 0
            else int(self._config.leaf_chunk_tokens),
        )
        sweep_summary_prefix_before = (
            self._summary_frontier_tokens() if threshold_full_sweep_active else 0
        )
        sweep_state: Dict[str, Any] = {}
        if threshold_full_sweep_active:
            sweep_state = {
                "status": "running",
                "leaf_passes": 0,
                "condensation_passes": 0,
                "total_passes": 0,
                "duration_ms": 0.0,
                "tokens_before": estimated_active_tokens,
                "tokens_after": estimated_active_tokens,
                "summary_prefix_tokens_before": sweep_summary_prefix_before,
                "summary_prefix_tokens_after": sweep_summary_prefix_before,
                "summary_prefix_target_tokens": sweep_target_tokens,
                "stop_reason": "",
                "budget_exhausted": False,
            }
        base_max_leaf_passes = 4 if self._config.dynamic_leaf_chunk_enabled else 1
        max_leaf_passes = base_max_leaf_passes
        if threshold_full_sweep_active:
            max_leaf_passes = _THRESHOLD_FULL_SWEEP_MAX_PASSES

        explicit_focus_topic = focus_topic is not None

        noop_reason = "no eligible raw backlog outside fresh tail"
        sweep_stop_reason = ""
        sweep_raw_drained = False

        while leaf_passes < max_leaf_passes:
            if threshold_full_sweep_active and time.monotonic() >= sweep_deadline:
                sweep_stop_reason = "time_budget_exhausted"
                break
            fresh_tail_start = self._fresh_tail_start(pressure_messages)

            # Keep only a real system prompt anchored. Gateway sessions may
            # pass only conversation messages, so index 0 can be an old user
            # turn; that must remain eligible for compaction instead of being
            # replayed forever as fresh-looking intent.
            leading_anchor_count = self._leading_anchor_count(working_messages)
            if fresh_tail_start <= leading_anchor_count:
                noop_reason = "no eligible raw backlog outside fresh tail"
                if threshold_full_sweep_active:
                    sweep_raw_drained = True
                    sweep_stop_reason = "raw_prefix_drained"
                break

            candidate_start = leading_anchor_count
            while (
                candidate_start < fresh_tail_start
                and self._is_replayed_context_scaffold_message(working_messages[candidate_start])
            ):
                candidate_start += 1
            if candidate_start > leading_anchor_count:
                working_messages = working_messages[:leading_anchor_count] + working_messages[candidate_start:]
                pressure_messages = pressure_messages[:leading_anchor_count] + pressure_messages[candidate_start:]
                candidate_start = leading_anchor_count
                fresh_tail_start = self._fresh_tail_start(pressure_messages)
                if fresh_tail_start <= leading_anchor_count:
                    noop_reason = "selected leaf chunk lacks raw store lineage"
                    break

            if candidate_start < fresh_tail_start:
                self._current_compress_store_ids_by_message_id = self._get_store_id_map_for_messages(
                    working_messages[leading_anchor_count:]
                )
                compactable_pairs = list(
                    zip(
                        working_messages[candidate_start:fresh_tail_start],
                        pressure_messages[candidate_start:fresh_tail_start],
                    )
                )
                kept_working: list[Dict[str, Any]] = []
                kept_pressure: list[Dict[str, Any]] = []
                dropped_generated_placeholder = False
                generated_placeholder_hashes = self._load_generated_ignored_placeholder_hashes()
                for working_msg, pressure_msg in compactable_pairs:
                    content_text = text_content_for_pattern_matching(working_msg.get("content")) or ""
                    volatile_digest = self._active_replay_placeholder_digest(content_text)
                    if (
                        self._is_volatile_ignored_quarantine_placeholder(working_msg, content_text)
                        and volatile_digest is not None
                        and volatile_digest in generated_placeholder_hashes
                    ):
                        dropped_generated_placeholder = True
                        continue
                    kept_working.append(working_msg)
                    kept_pressure.append(pressure_msg)
                if dropped_generated_placeholder:
                    working_messages = (
                        working_messages[:candidate_start]
                        + kept_working
                        + working_messages[fresh_tail_start:]
                    )
                    pressure_messages = (
                        pressure_messages[:candidate_start]
                        + kept_pressure
                        + pressure_messages[fresh_tail_start:]
                    )
                    fresh_tail_start = self._fresh_tail_start(pressure_messages)
                    if fresh_tail_start <= leading_anchor_count:
                        noop_reason = "selected leaf chunk lacks raw store lineage"
                        break

            # Auto-derive focus topic from the post-filter compaction view when
            # not explicitly provided.  The derived focus is summarizer-visible,
            # so it must follow the same filtering as the leaf chunk itself.
            if not explicit_focus_topic:
                focus_topic = self._derive_auto_focus_topic(working_messages)

            candidate_raw = working_messages[leading_anchor_count:fresh_tail_start]
            if not candidate_raw:
                noop_reason = "no eligible raw backlog outside fresh tail"
                if threshold_full_sweep_active:
                    sweep_raw_drained = True
                    sweep_stop_reason = "raw_prefix_drained"
                break

            pressure_candidate_raw = pressure_messages[leading_anchor_count:fresh_tail_start]
            raw_tokens_outside_tail = count_messages_tokens(pressure_candidate_raw)
            if threshold_full_sweep_active:
                working_leaf_chunk_tokens = self._working_leaf_chunk_tokens(
                    raw_tokens_outside_tail
                )
                to_compact = self._select_oldest_leaf_chunk(
                    candidate_raw,
                    working_leaf_chunk_tokens,
                )
            elif self._config.dynamic_leaf_chunk_enabled:
                working_leaf_chunk_tokens = self._working_leaf_chunk_tokens(raw_tokens_outside_tail)
                if raw_tokens_outside_tail < working_leaf_chunk_tokens and not force_overflow:
                    noop_reason = (
                        "raw backlog outside fresh tail is below leaf chunk threshold"
                    )
                    break
                if force_overflow:
                    to_compact = candidate_raw
                else:
                    to_compact = self._select_oldest_leaf_chunk(candidate_raw, working_leaf_chunk_tokens)
            else:
                if raw_tokens_outside_tail < self._config.leaf_chunk_tokens and not force_overflow:
                    noop_reason = (
                        "raw backlog outside fresh tail is below leaf chunk threshold"
                    )
                    break
                to_compact = candidate_raw

            if not to_compact:
                noop_reason = "no eligible leaf chunk selected"
                break

            selected_raw_chunk = to_compact
            # A cancelled attempt starts no further summariser call (#29 W2 step 2).
            if attempt is not None and attempt.cancelled():
                raise AttemptCancelled()
            shadow_chunk = None
            if attempt is not None:
                self._shadow_begin(attempt, force=force)
                shadow_chunk = self._shadow_chunk(attempt, selected_raw_chunk)
            try:
                summary_kwargs: dict[str, Any] = {"focus_topic": focus_topic}
                if threshold_full_sweep_active:
                    summary_kwargs["deadline"] = sweep_deadline
                (
                    compacted_chunk,
                    source_tokens,
                    summary_text,
                    _level,
                    _rescue_attempts,
                ) = self._summarize_leaf_chunk_with_rescue(
                    selected_raw_chunk,
                    **summary_kwargs,
                )
            except Exception as exc:
                if threshold_full_sweep_active and leaf_compacted_this_turn:
                    sweep_stop_reason = "leaf_summary_error"
                    logger.warning(
                        "LCM threshold full sweep stopped after %d persisted leaf pass(es): %s",
                        leaf_passes,
                        exc,
                    )
                    break
                raise
            compacted_summary_ids = {id(message) for message in compacted_chunk}
            compacted_positions = [
                idx for idx, message in enumerate(selected_raw_chunk) if id(message) in compacted_summary_ids
            ]
            last_compacted_raw_pos = max(compacted_positions) if compacted_positions else len(compacted_chunk) - 1
            source_lookup_chunk = selected_raw_chunk[: last_compacted_raw_pos + 1]
            selected_raw_len = len(source_lookup_chunk)
            if attempt is not None:
                if shadow_chunk is not None and len(source_lookup_chunk) != len(selected_raw_chunk):
                    # The rescue summarised a shorter prefix than the chunk it was
                    # given; the summary stands for that prefix (the rescue goes in D).
                    self._records.event(
                        "rescue_shortened_chunk",
                        session=attempt.session,
                        compaction=attempt.compaction,
                        detail={"chunk": shadow_chunk, "given": len(selected_raw_chunk),
                                "summarised": len(source_lookup_chunk)},
                    )
                    shadow_chunk = self._shadow_chunk(attempt, source_lookup_chunk)
                self._shadow_derivation(
                    attempt,
                    shadow_chunk,
                    text=summary_text,
                    level=_level,
                    est_tokens=count_tokens(summary_text),
                )
            remaining_messages = working_messages[leading_anchor_count + selected_raw_len:]
            source_tokens = count_messages_tokens(source_lookup_chunk)

            source_store_ids = self._get_store_ids_for_messages(source_lookup_chunk)
            source_store_ids = sorted(dict.fromkeys(source_store_ids))
            consumed_store_ids = source_store_ids
            earliest_at, latest_at = self._store.get_time_bounds(source_store_ids)
            summary_tokens = count_tokens(summary_text)

            node = SummaryNode(
                session_id=self._session_id,
                depth=0,
                summary=summary_text,
                token_count=summary_tokens,
                source_token_count=source_tokens,
                source_ids=source_store_ids,
                source_type="messages",
                created_at=time.time(),
                earliest_at=earliest_at,
                latest_at=latest_at,
                expand_hint=self._extract_expand_hint(summary_text),
            )
            self._dag.add_node(node)
            self._last_compacted_store_id = max(consumed_store_ids) if consumed_store_ids else 0
            self._persist_frontier_marker()

            pressure_remaining_messages = pressure_messages[leading_anchor_count + selected_raw_len:]
            working_messages = working_messages[:leading_anchor_count] + remaining_messages
            pressure_messages = pressure_messages[:leading_anchor_count] + pressure_remaining_messages
            leaf_compacted_this_turn = True
            leaf_passes += 1
            estimated_active_tokens = max(0, estimated_active_tokens - source_tokens + summary_tokens)

            if threshold_full_sweep_active:
                leading_anchor_count = self._leading_anchor_count(working_messages)
                remaining_fresh_tail_start = self._fresh_tail_start(pressure_messages)
                remaining_raw = working_messages[
                    leading_anchor_count:remaining_fresh_tail_start
                ]
                if not remaining_raw:
                    sweep_raw_drained = True
                    sweep_stop_reason = "raw_prefix_drained"
                    break
                continue

            if not self._config.dynamic_leaf_chunk_enabled:
                break

            if not force_overflow:
                if self.threshold_tokens > 0 and estimated_active_tokens < self.threshold_tokens:
                    break
                leading_anchor_count = self._leading_anchor_count(working_messages)
                remaining_fresh_tail_start = self._fresh_tail_start(pressure_messages)
                remaining_raw = working_messages[
                    leading_anchor_count:remaining_fresh_tail_start
                ]
                if not remaining_raw:
                    break
                pressure_remaining_raw = pressure_messages[
                    leading_anchor_count:remaining_fresh_tail_start
                ]
                remaining_raw_tokens = count_messages_tokens(pressure_remaining_raw)
                remaining_threshold = self._working_leaf_chunk_tokens(remaining_raw_tokens)
                if remaining_raw_tokens < remaining_threshold:
                    break

        if (
            threshold_full_sweep_active
            and not sweep_raw_drained
            and not sweep_stop_reason
            and leaf_passes >= max_leaf_passes
        ):
            sweep_stop_reason = "pass_budget_exhausted"

        if not leaf_compacted_this_turn:
            if force_overflow:
                # No leaf pass: nothing is cut, dropped or truncated to fit.
                self._publish("_last_overflow_recovery_failed", True)
                noop_reason = f"forced overflow recovery made no leaf pass ({noop_reason})"
            if threshold_full_sweep_active:
                duration_ms = (time.perf_counter() - _compress_started) * 1000.0
                self._publish("_last_threshold_full_sweep", {
                    **sweep_state,
                    "status": "noop",
                    "duration_ms": round(duration_ms, 3),
                    "stop_reason": sweep_stop_reason or noop_reason,
                    "budget_exhausted": sweep_stop_reason
                    in {"pass_budget_exhausted", "time_budget_exhausted"},
                })
            return self._unchanged_return(messages, noop_reason)

        # Step 6: Check if condensation is needed. A threshold full sweep only
        # condenses after the eligible raw prefix has been drained, and shares
        # the same total pass/deadline budget as its leaf work.
        condensation_passes = 0
        if attempt is not None and attempt.cancelled():
            raise AttemptCancelled()
        if threshold_full_sweep_active:
            if sweep_raw_drained:
                remaining_passes = max(
                    0,
                    _THRESHOLD_FULL_SWEEP_MAX_PASSES - leaf_passes,
                )
                condensation_passes, sweep_stop_reason = (
                    self._run_threshold_sweep_condensation(
                        target_tokens=sweep_target_tokens,
                        pass_budget=remaining_passes,
                        deadline=sweep_deadline,
                        focus_topic=focus_topic,
                    )
                )
        else:
            self._maybe_condense(
                focus_topic=focus_topic,
                leaf_compacted_this_turn=True,
                force_overflow=force_overflow,
            )

        # Step 7: Assemble new active context
        leading_anchor_count = self._leading_anchor_count(working_messages)
        anchor_leading_count = self._leading_anchor_count(anchor_source_messages)
        self._pending_context_anchor_messages = anchor_source_messages[anchor_leading_count:]
        try:
            compressed = self._assemble_context(
                working_messages[0] if leading_anchor_count else None,
                working_messages[leading_anchor_count:],
                assembly_cap_override=recovery_assembly_cap,
            )
        finally:
            self._pending_context_anchor_messages = None
        compressed = self._untouched_return(
            messages,
            compressed,
            working_messages[leading_anchor_count:],
            has_system=bool(leading_anchor_count),
            index_by_working=index_by_working,
        )
        compaction_number = self._publish_compaction_counted()
        compaction_duration_ms = (time.perf_counter() - _compress_started) * 1000.0
        self._publish("_last_compaction_duration_ms", compaction_duration_ms)
        logger.info(
            "LCM leaf compaction finished in %.1fms", compaction_duration_ms
        )
        self._publish("_last_compression_status", "compacted")
        self._publish("_last_compression_noop_reason", "")
        if recovery_assembly_cap is None:
            self._publish("_last_overflow_recovery_failed", False)
        else:
            overflow_recovery_failed = count_messages_tokens(compressed) > recovery_assembly_cap
            self._publish("_last_overflow_recovery_failed", overflow_recovery_failed)
            if overflow_recovery_failed:
                logger.warning(
                    "LCM overflow recovery could not get under cap=%d after compaction; returning best-effort context (%d tokens)",
                    recovery_assembly_cap,
                    count_messages_tokens(compressed),
                )
        # Reset cursor to the length of the compressed context so that
        # only messages appended *after* this point get ingested next time.
        self._publish("_ingest_cursor", len(compressed))
        self._publish("_ingest_cursor_needs_reconcile", False)

        logger.info(
            "LCM compaction #%d: %d messages → %d (%d leaf pass%s, %d→%d tokens, %d DAG nodes%s)",
            compaction_number,
            len(messages),
            len(compressed),
            leaf_passes,
            "es" if leaf_passes != 1 else "",
            count_messages_tokens(messages),
            count_messages_tokens(compressed),
            len(self._dag.get_session_nodes(self._session_id)),
            ", forced overflow recovery" if force_overflow else "",
        )

        if threshold_full_sweep_active:
            total_passes = leaf_passes + condensation_passes
            duration_ms = (time.perf_counter() - _compress_started) * 1000.0
            final_stop_reason = sweep_stop_reason or "raw_prefix_drained"
            partial_stop_reasons = {
                "pass_budget_exhausted",
                "time_budget_exhausted",
                "leaf_summary_error",
                "condensation_error",
                "condensation_no_progress",
                "no_same_depth_condensation_group",
            }
            self._publish("_last_threshold_full_sweep", {
                "status": "partial" if final_stop_reason in partial_stop_reasons else "completed",
                "leaf_passes": leaf_passes,
                "condensation_passes": condensation_passes,
                "total_passes": total_passes,
                "duration_ms": round(duration_ms, 3),
                "tokens_before": sweep_state["tokens_before"],
                "tokens_after": count_messages_tokens(compressed),
                "summary_prefix_tokens_before": sweep_summary_prefix_before,
                "summary_prefix_tokens_after": self._summary_frontier_tokens(),
                "summary_prefix_target_tokens": sweep_target_tokens,
                "stop_reason": final_stop_reason,
                "budget_exhausted": final_stop_reason
                in {"pass_budget_exhausted", "time_budget_exhausted"},
            })
        self._write_generated_ignored_placeholder_hash_counts(
            self._generated_placeholder_digest_budget_for_active_replay(compressed)
        )
        self._write_generated_ignored_placeholder_hash_ordinals(
            self._generated_placeholder_digest_ordinals_for_active_replay(compressed)
        )
        record_successful_compaction = getattr(
            self,
            "_record_successful_compaction_telemetry",
            None,
        )
        if callable(record_successful_compaction):
            record_successful_compaction()

        # The return fences on the captured check: a cancelled attempt writes no
        # return and hands the host its input (#29 W2 steps 2 and 6).
        if attempt is not None:
            if attempt.cancelled():
                raise AttemptCancelled()
            self._shadow_returns(attempt, compressed)

        return compressed
