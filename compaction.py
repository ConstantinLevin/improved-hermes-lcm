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
   a summary and is never chunked).
3. The material is cut into chunks, in list order, each at most ``leaf_chunk_tokens``
   (a greedy cut until #12), only between groups: a tool call and its results stay in
   one chunk, and a group larger than a chunk is a chunk of its own. The compaction,
   its inputs, the new records and every chunk are written before the first
   summariser call.
4. Each chunk is summarised from its records as the store holds them, one call at a
   time, and the summary is written as a derivation when it arrives. A call that
   fails fails the compaction as a whole: the context stays as it was.
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
from typing import Any, Dict, List, Optional

from .escalation import summarize_with_escalation
from .message_analysis import _tool_call_id
from .record_store import RET_KEY, parse_ret_key, raw_json
from .record_write import _ATTEMPT, AttemptCancelled
from .tokens import count_message_tokens, count_messages_tokens

logger = logging.getLogger(__name__)

# The words around a summary row (Decision 8, until #10 writes them).
_SUMMARY_HEADER = "[Recent Summary (d0, node {node_id})]"
_SUMMARY_FOOTER = "[Expand for details: {hint}]"


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
        rough = count_messages_tokens(messages)
        if self._should_force_overflow_recovery(observed_tokens=rough, messages=messages):
            return self._mark_preflight_compression_requested()
        if self.threshold_tokens > 0 and rough >= self.threshold_tokens:
            mechanism = self._mechanism_positions(messages)
            tail_start = self._tail_start(messages, mechanism)
            material_tokens = count_messages_tokens(
                [messages[index] for index in range(tail_start) if index not in mechanism]
            )
            if material_tokens >= self._config.leaf_chunk_tokens:
                return self._mark_preflight_compression_requested()
            reason = "the material outside the fresh tail is below one leaf chunk"
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

    def _summarize_chunk(
        self,
        chunk_messages: List[Dict[str, Any]],
        *,
        focus_topic: Optional[str],
        store_ids: List[int],
    ) -> tuple[str, int, int]:
        """One summariser call for one chunk: (text, level, budget).

        The chunk's records are serialised as today (``_serialize_messages``, until #8)
        and summarised through the escalation, whose last level truncates (until #7).
        """
        source_tokens = count_messages_tokens(chunk_messages)
        budget = min(max(2000, int(source_tokens * 0.20)), 12000)
        text, level = summarize_with_escalation(
            text=self._serialize_messages(chunk_messages),
            source_tokens=source_tokens,
            token_budget=budget,
            depth=0,
            model=self._config.summary_model,
            fallback_models=self._config.summary_fallback_models,
            circuit_breaker=self._summary_circuit_breaker,
            spend_guard=self._summary_spend_guard,
            timeout=self._config.summary_timeout_ms / 1000,
            l2_budget_ratio=self._config.l2_budget_ratio,
            l3_truncate_tokens=self._config.l3_truncate_tokens,
            focus_topic=focus_topic or "",
            custom_instructions=self._config.custom_instructions,
            source_provenance={
                "source_type": "messages",
                "store_ids": store_ids,
                "message_count": len(chunk_messages),
            },
        )
        return text, level, budget

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
    def _cut_chunks(cls, messages: List[Dict[str, Any]], material: List[int], limit: int) -> List[List[int]]:
        """The material cut in list order into chunks of at most ``limit`` tokens, and
        only between groups: a tool call is never separated from its results. A group
        larger than the limit is a chunk of its own (#31: a chunk flexes by one group)."""
        chunks: List[List[int]] = []
        current: List[int] = []
        used = 0
        for group in cls._groups(messages, material):
            tokens = sum(count_message_tokens(messages[index]) for index in group)
            if current and used + tokens > limit:
                chunks.append(current)
                current, used = [], 0
            current.extend(group)
            used += tokens
        if current:
            chunks.append(current)
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
        material_tokens = count_messages_tokens([messages[index] for index in material])
        if not material:
            if force_overflow:
                self._publish("_last_overflow_recovery_failed", True)
            return self._unchanged_return(messages, "no material outside the fresh tail")
        if material_tokens < self._config.leaf_chunk_tokens and not (force or force_overflow):
            return self._unchanged_return(messages, "the material outside the fresh tail is below one leaf chunk")

        # 3. Transaction 1: the compaction, its inputs, the new records and every chunk.
        chunks = self._cut_chunks(messages, material, max(1, int(self._config.leaf_chunk_tokens)))
        try:
            chunk_handles = self._write_compaction(attempt, entries, chunks, force=force)
        except Exception as exc:
            logger.warning("LCM could not write the compaction", exc_info=True)
            self._record_event(attempt, "compaction_write_failed", repr(exc))
            return self._abort(messages, f"the store could not write the compaction ({exc})")

        # 4. One summary per chunk, from the chunk's records as stored.
        members = [attempt.records[index] for chunk in chunks for index in chunk]
        facts = self._records.record_facts(members)
        new_derivations: List[str] = []
        for number, (chunk_handle, chunk) in enumerate(zip(chunk_handles, chunks), start=1):
            # A cancelled or no longer current attempt starts no further call (#29 W2 step 2).
            if not self._live_write_allowed(attempt):
                raise AttemptCancelled()
            records = [attempt.records[index] for index in chunk]
            chunk_messages = [json.loads(facts[record][1]) for record in records]
            try:
                text, level, budget = self._summarize_chunk(
                    chunk_messages,
                    focus_topic=focus_topic,
                    store_ids=[facts[record][3] for record in records],
                )
            except Exception as exc:
                logger.warning("LCM summary of chunk %d of %d failed", number, len(chunks), exc_info=True)
                self._record_event(attempt, "summary_failed",
                                   {"chunk": chunk_handle, "number": number, "error": repr(exc)})
                return self._abort(messages, f"the summary of chunk {number} of {len(chunks)} failed ({exc})")
            try:
                new_derivations.append(self._write_summary(
                    attempt, chunk_handle, text=text, level=level, budget=budget,
                    expand_hint=self._extract_expand_hint(text),
                ))
            except Exception as exc:
                logger.warning("LCM could not write a summary", exc_info=True)
                self._record_event(attempt, "derivation_write_failed", repr(exc))
                return self._abort(messages, f"the store could not write a summary ({exc})")

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
        over_cap = recovery_cap is not None and count_messages_tokens(result) > recovery_cap
        self._publish("_last_overflow_recovery_failed", over_cap)
        if over_cap:
            logger.warning(
                "LCM overflow recovery left the context above the cap (%d > %d); nothing is cut to fit",
                count_messages_tokens(result), recovery_cap,
            )
        logger.info(
            "LCM compaction #%d: %d entries -> %d (%d chunk%s, %d summaries in the cover, %d -> %d tokens, %.1fms)",
            number, len(messages), len(result), len(chunks), "" if len(chunks) == 1 else "s", len(cover),
            count_messages_tokens(messages), count_messages_tokens(result), duration_ms,
        )
        return result
