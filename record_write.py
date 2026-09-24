"""The write at a compaction, beside the old path (#29 W2 and W3; #33 D12).

The old path stays authoritative for the summariser, the assembly and the tools; the
record written here is a shadow until the read switch. What is written, and when:

- ``compress()`` captures, at its entry and on its own thread, the host's
  cancellation check for this attempt and the attempt's generation, and never reads
  either from the engine again (a newer attempt overwrites the engine attribute).
- At the first leaf pass: the compaction, its inputs, a record for every input entry
  the store does not hold yet, and their tool calls (transaction 1). Input entries
  are classified against the previous effective return by the host's ``_row_id`` and
  the plugin's in-memory key only, never by content.
- Before each summariser call: the chunk with its members. When the summary arrives:
  the derivation, also after a cancellation (a summary is a fact about its chunk and
  makes nothing active).
- Before returning: the captured check is asked; a cancelled attempt writes no return
  and returns its input. Otherwise the return entries are written and each returned
  dict carries the in-memory key.
- At ``on_session_start(boundary_reason="compression")``: the confirmation (the record
  line) and the bindings read off the returned dicts, after asking the attempt's
  captured check. ``record_rejected_compaction()`` writes a rejection. Entries still
  unbound are bound at the next list the plugin sees.

Engine attributes the host or the status read are set only by the attempt that is the
host's current working attempt, and only when it returns (#33 D12). "Current" is the
host's own rule (``_working_attempt_is_current``) applied to the generation captured
at entry; this rests on host internals until Hermes answers asks A-33.1 and A-33.2.

An engine copy writes only for its own plugin session.
"""

from __future__ import annotations

import contextvars
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .record_store import RET_KEY, InputEntry, parse_ret_key

logger = logging.getLogger(__name__)

try:  # host internals; see the module docstring
    from agent.conversation_compression import (  # type: ignore
        _COMPRESSOR_ATTEMPT_GENERATION as _HOST_ATTEMPT_GENERATION,
        _working_attempt_is_current as _host_working_attempt_is_current,
    )
except Exception:  # pragma: no cover - older or absent host
    _HOST_ATTEMPT_GENERATION = None
    _host_working_attempt_is_current = None

_ATTEMPT: contextvars.ContextVar = contextvars.ContextVar("lcm_compress_attempt", default=None)


class AttemptCancelled(Exception):
    """The attempt's captured cancellation check said stop; nothing more is started."""


@dataclass
class CompressAttempt:
    check: Any
    generation: Any
    session: str
    outcome: Dict[str, Any] = field(default_factory=dict)
    count_increment: int = 0
    messages: Optional[List[Dict[str, Any]]] = None
    index_by_working: Optional[Dict[int, int]] = None
    compaction: Optional[int] = None
    records: Dict[int, str] = field(default_factory=dict)
    shadow_ok: bool = True
    returned: Optional[List[Dict[str, Any]]] = None
    returned_positions: set = field(default_factory=set)
    bound_positions: set = field(default_factory=set)

    def cancelled(self) -> bool:
        if not callable(self.check):
            return False
        try:
            return bool(self.check())
        except Exception:
            logger.warning("LCM could not ask the host's cancellation check", exc_info=True)
            return False


def current_attempt() -> Optional[CompressAttempt]:
    return _ATTEMPT.get()


class RecordWriteMixin:
    # --- The attempt ----------------------------------------------------------------

    def _begin_attempt(self) -> CompressAttempt:
        generation = _HOST_ATTEMPT_GENERATION.get() if _HOST_ATTEMPT_GENERATION is not None else None
        return CompressAttempt(
            check=getattr(self, "_compression_cancelled_check", None),
            generation=generation,
            session=self._plugin_session,
        )

    def _attempt_is_current(self, attempt: CompressAttempt) -> bool:
        if attempt.generation is None or _host_working_attempt_is_current is None:
            return True
        try:
            return bool(_host_working_attempt_is_current(self, attempt.generation))
        except Exception:
            logger.warning("LCM could not ask the host whether its attempt is current", exc_info=True)
            return False

    def _publish(self, name: str, value: Any) -> None:
        """Set an attribute now outside an attempt, or at the attempt's return."""
        attempt = _ATTEMPT.get()
        if attempt is None:
            setattr(self, name, value)
        else:
            attempt.outcome[name] = value

    def _publish_compaction_counted(self) -> int:
        attempt = _ATTEMPT.get()
        if attempt is None:
            self.compression_count += 1
            return self.compression_count
        attempt.count_increment += 1
        return self.compression_count + attempt.count_increment

    def _apply_attempt_outcome(self, attempt: CompressAttempt) -> None:
        for name, value in attempt.outcome.items():
            setattr(self, name, value)
        if attempt.count_increment:
            self.compression_count += attempt.count_increment

    # --- Shadow writes --------------------------------------------------------------

    def _shadow_failed(self, attempt: Optional[CompressAttempt], kind: str, detail: Any) -> None:
        if attempt is not None:
            attempt.shadow_ok = False
        self._records.event(
            kind,
            session=attempt.session if attempt is not None else self._plugin_session or None,
            compaction=attempt.compaction if attempt is not None else None,
            detail=detail,
        )

    def _shadow_begin(self, attempt: CompressAttempt, *, force: bool) -> None:
        """Transaction 1, once per attempt, at its first leaf pass."""
        if attempt.compaction is not None or not attempt.shadow_ok:
            return
        if not attempt.session:
            self._shadow_failed(attempt, "no_plugin_session", "compress() on an engine copy that names no plugin session")
            return
        messages = attempt.messages or []
        try:
            store = self._records
            effective = store.effective_compaction(attempt.session)
            returned = store.return_entries(effective) if effective is not None else {}
            bound = store.bound_rows(effective) if effective is not None else {}
            reusable = store.unconfirmed_inputs(attempt.session, effective)
            entries: List[Optional[InputEntry]] = [None] * len(messages)
            late_bindings: list[tuple[int, int]] = []
            last_bound = -1
            for index, message in enumerate(messages):
                row_id = message.get("_row_id")
                position = None
                key = parse_ret_key(message.get(RET_KEY))
                if effective is not None and key is not None and key[0] == effective:
                    position = key[1]
                    if isinstance(row_id, int) and row_id not in bound:
                        late_bindings.append((position, row_id))
                elif effective is not None and isinstance(row_id, int) and row_id in bound:
                    position = bound[row_id]
                if position is None:
                    continue
                kind, record, _derivation = returned.get(position, ("", None, None))
                entries[index] = InputEntry(index, row_id if isinstance(row_id, int) else None, "bound",
                                            record=record if kind == "record" else None)
                last_bound = index
            if effective is not None and last_bound < 0:
                self._shadow_failed(
                    attempt,
                    "return_not_found",
                    f"none of the {len(messages)} entries is bound to compaction {effective}; "
                    "the list carries no identity for it (see ask A1)",
                )
                return
            for index, message in enumerate(messages):
                if entries[index] is not None:
                    continue
                row_id = message.get("_row_id")
                row_id = row_id if isinstance(row_id, int) else None
                if index == 0 and message.get("role") == "system":
                    klass, record = "system", None
                elif row_id is not None and row_id in reusable:
                    klass, record = "reused", reusable[row_id]
                elif index < last_bound:
                    klass, record = "host_insertion", None
                else:
                    klass, record = "transcript", None
                entries[index] = InputEntry(index, row_id, klass, record=record, message=message)
            if late_bindings and effective is not None:
                store.bind(effective, late_bindings)
            compaction, records = store.begin_compaction(
                session=attempt.session,
                kind="full" if force else "threshold",
                host_session_before=self._session_id or None,
                attempt_generation=attempt.generation if isinstance(attempt.generation, int) else None,
                entries=[entry for entry in entries if entry is not None],
                head=store.head(effective),
            )
        except Exception as exc:
            logger.warning("LCM shadow write of the compaction failed", exc_info=True)
            self._shadow_failed(attempt, "compaction_write_failed", repr(exc))
            return
        attempt.compaction = compaction
        attempt.records = records

    def _shadow_chunk(self, attempt: Optional[CompressAttempt], working_members: List[Dict[str, Any]]) -> Optional[str]:
        if attempt is None or not attempt.shadow_ok or attempt.compaction is None:
            return None
        index = attempt.index_by_working or {}
        positions = [index.get(id(message)) for message in working_members]
        members = [attempt.records.get(pos) if pos is not None else None for pos in positions]
        if not members or any(member is None for member in members):
            self._shadow_failed(attempt, "chunk_member_without_record", {"positions": positions})
            return None
        try:
            return self._records.write_chunk(session=attempt.session, compaction=attempt.compaction, members=members)
        except Exception as exc:
            logger.warning("LCM shadow write of a chunk failed", exc_info=True)
            self._shadow_failed(attempt, "chunk_write_failed", repr(exc))
            return None

    def _shadow_derivation(
        self,
        attempt: Optional[CompressAttempt],
        chunk: Optional[str],
        *,
        text: str,
        level: Optional[int],
        est_tokens: Optional[int],
    ) -> None:
        if attempt is None or chunk is None or attempt.compaction is None:
            return
        try:
            self._records.write_derivation(
                compaction=attempt.compaction,
                chunk=chunk,
                text=text,
                model=self._config.summary_model or None,
                provider=None,
                level=level,
                budget=None,
                est_tokens=est_tokens,
            )
        except Exception as exc:
            logger.warning("LCM shadow write of a summary failed", exc_info=True)
            self._shadow_failed(attempt, "derivation_write_failed", repr(exc))

    def _shadow_returns(self, attempt: Optional[CompressAttempt], result: List[Dict[str, Any]]) -> None:
        """Write the return and key the returned dicts. The caller has asked the
        attempt's captured check just before."""
        if attempt is None or not attempt.shadow_ok or attempt.compaction is None:
            return
        by_object = {id(message): pos for pos, message in enumerate(attempt.messages or [])}
        entries: list[tuple[int, str, Optional[str], Optional[str]]] = []
        for position, message in enumerate(result):
            input_position = by_object.get(id(message))
            if input_position is not None and input_position in attempt.records:
                entries.append((position, "record", attempt.records[input_position], None))
            elif message.get("_compressed_summary") is True and input_position is None:
                entries.append((position, "summary", None, None))
            elif input_position == 0 and message.get("role") == "system":
                continue  # the host's system row, returned in place, not recorded
            else:
                self._shadow_failed(attempt, "return_entry_unknown", {"position": position, "role": message.get("role")})
                return
        try:
            self._records.write_returns(attempt.compaction, entries)
        except Exception as exc:
            logger.warning("LCM shadow write of the return failed", exc_info=True)
            self._shadow_failed(attempt, "return_write_failed", repr(exc))
            return
        for position, _kind, _record, _derivation in entries:
            result[position][RET_KEY] = f"{attempt.compaction}:{position}"
        attempt.returned = result
        attempt.returned_positions = {position for position, _k, _r, _d in entries}
        self._returned_attempts[attempt.compaction] = attempt

    def _note_bound(self, attempt: CompressAttempt, written: List[tuple]) -> None:
        attempt.bound_positions.update(position for position, _row_id in written)
        if attempt.returned_positions and attempt.returned_positions <= attempt.bound_positions:
            self._returned_attempts.pop(attempt.compaction, None)

    # --- Confirmation, rejection and binding -------------------------------------------

    def _pending_attempt(self) -> Optional[CompressAttempt]:
        if not self._plugin_session:
            return None
        pending = self._records.latest_pending(self._plugin_session)
        if pending is None:
            return None
        attempt = self._returned_attempts.get(pending)
        if attempt is None:
            attempt = CompressAttempt(check=None, generation=None, session=self._plugin_session, compaction=pending)
        return attempt

    def _record_confirmation(self, old_session_id: str, session_id: str) -> None:
        try:
            attempt = self._pending_attempt()
            if attempt is None:
                self._records.event(
                    "confirmation_without_attempt",
                    session=self._plugin_session or None,
                    detail={"old_session_id": old_session_id, "session_id": session_id},
                )
                return
            if attempt.cancelled():
                self._records.event("confirmation_of_cancelled_attempt", session=attempt.session,
                                    compaction=attempt.compaction)
                return
            self._records.confirm(attempt.compaction, host_session_before=old_session_id,
                                  host_session_after=session_id)
            self._bind_returned(attempt)
        except Exception as exc:
            logger.warning("LCM could not record the host's confirmation", exc_info=True)
            self._records.event("confirmation_write_failed", session=self._plugin_session or None, detail=repr(exc))

    def _bind_returned(self, attempt: CompressAttempt) -> None:
        if not attempt.returned:
            self._records.event("binding_without_returned_objects", session=attempt.session,
                                compaction=attempt.compaction)
            return
        pairs = [
            (position, message["_row_id"])
            for position, message in enumerate(attempt.returned)
            if RET_KEY in message and isinstance(message.get("_row_id"), int)
        ]
        self._note_bound(attempt, self._records.bind(attempt.compaction, pairs))

    def record_rejected_compaction(self, *args: Any, **kwargs: Any) -> None:
        """The host refused the latest returned compaction (for example a grown result)."""
        try:
            attempt = self._pending_attempt()
            if attempt is None:
                self._records.event("rejection_without_attempt", session=self._plugin_session or None)
                return
            self._records.reject(attempt.compaction, how="record_rejected_compaction")
            self._returned_attempts.pop(attempt.compaction, None)
        except Exception as exc:
            logger.warning("LCM could not record the host's rejection", exc_info=True)
            self._records.event("rejection_write_failed", session=self._plugin_session or None, detail=repr(exc))

    def _bind_from_list(self, messages: Optional[List[Dict[str, Any]]]) -> None:
        """Bind keyed entries of a confirmed compaction of this copy's own session that
        carry the host's ``_row_id`` now."""
        if not messages or not self._plugin_session or not self._returned_attempts:
            return
        pairs: dict[int, list[tuple[int, int]]] = {}
        for message in messages:
            if not isinstance(message, dict):
                continue
            key = parse_ret_key(message.get(RET_KEY))
            row_id = message.get("_row_id")
            if key is None or not isinstance(row_id, int) or key[0] not in self._returned_attempts:
                continue
            pairs.setdefault(key[0], []).append((key[1], row_id))
        for compaction, compaction_pairs in pairs.items():
            try:
                if self._records.compaction_session(compaction) != self._plugin_session:
                    continue
                if self._records.effective_compaction(self._plugin_session) != compaction:
                    continue
                written = self._records.bind(compaction, compaction_pairs)
                attempt = self._returned_attempts.get(compaction)
                if attempt is not None:
                    self._note_bound(attempt, written)
            except Exception as exc:
                logger.warning("LCM could not write bindings", exc_info=True)
                self._records.event("binding_write_failed", session=self._plugin_session, compaction=compaction,
                                    detail=repr(exc))

    def on_turn_complete(self, messages: List[Dict[str, Any]], usage: Dict[str, Any] = None, **kwargs: Any) -> None:
        """The first list after a confirmation binds what the confirmation could not."""
        self._bind_from_list(messages)
