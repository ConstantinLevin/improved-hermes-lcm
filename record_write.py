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
- Before every write of live state (the old path's DAG node, its frontier) and before
  the return, the captured check is asked; a cancelled or no longer current attempt
  writes none of them and returns its input.
- The confirmation ``on_session_start(boundary_reason="compression")`` carries no list.
  The committed attempt is the one whose own returned dicts the host stamped with a new
  ``_row_id`` at this commit (``hermes_state_messages.py:515-535``). Each dict is bound
  by the position in its own key; key-less dicts in the committed list are host
  insertions. When the host committed copies instead (its salvage path), nothing is
  stamped on the plugin's objects: the confirmation waits for the next list, whose
  keyed dicts name the compaction and carry the stamped ids.
- A return used without a confirmation (a host without a session database) is found
  by its own returned summary key in the next list, and recorded as an adoption.

Engine attributes the host or the status read are set only by the attempt that is the
host's current working attempt, and only when it returns (#33 D12). "Current" is the
host's own rule (``_working_attempt_is_current``) applied to the generation captured
at entry; this rests on host internals until Hermes answers asks A-33.1 and A-33.2.

An engine copy writes only for its own plugin session.
"""

from __future__ import annotations

import contextvars
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .record_store import RET_KEY, InputEntry, parse_ret_key, raw_json

logger = logging.getLogger(__name__)

try:  # host internals; see the module docstring
    from agent.conversation_compression import (  # type: ignore
        _COMPRESSOR_ATTEMPT_GENERATION as _HOST_ATTEMPT_GENERATION,
        _working_attempt_is_current as _host_working_attempt_is_current,
    )
except Exception:  # pragma: no cover - older or absent host
    _HOST_ATTEMPT_GENERATION = None
    _host_working_attempt_is_current = None

try:  # the host's own list of fields that are bookkeeping, not message content
    from agent.message_metadata import PERSISTENCE_ONLY_MESSAGE_FIELDS as _HOST_PERSISTENCE_ONLY  # type: ignore
except Exception:  # pragma: no cover - as at Hermes 130b8f2c5d, agent/message_metadata.py:14
    _HOST_PERSISTENCE_ONLY = frozenset({"timestamp", "display_kind", "display_metadata", "_row_id"})

# Left out of the rewrite comparison (ruling 6): the host's persistence-only fields,
# its persist marker, and the plugin's own key. The host stamps _row_id and timestamp
# on every dict it commits, so they differ without the row having been rewritten.
_NOT_COMPARED = frozenset(_HOST_PERSISTENCE_ONLY) | {"_db_persisted", RET_KEY}


def _comparable(message: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in message.items() if key not in _NOT_COMPARED}


def rewritten(message: Dict[str, Any], raw: str) -> bool:
    """Whether an identified row differs from what the store holds for it, as JSON
    values. Only ever asked of a row already identified by key or _row_id."""
    current = json.loads(json.dumps(_comparable(message), ensure_ascii=False, allow_nan=False))
    return current != _comparable(json.loads(raw))


_ATTEMPT: contextvars.ContextVar = contextvars.ContextVar("lcm_compress_attempt", default=None)


class AttemptCancelled(BaseException):
    """The attempt may not go on: its captured cancellation check said stop, or it is
    no longer the host's current working attempt. A BaseException, so that no
    ``except Exception`` on the way turns it into an ordinary failure."""


@dataclass
class CompressAttempt:
    check: Any
    generation: Any
    session: str
    outcome: Dict[str, Any] = field(default_factory=dict)
    count_increment: int = 0
    compacted: bool = False
    messages: Optional[List[Dict[str, Any]]] = None
    index_by_working: Optional[Dict[int, int]] = None
    compaction: Optional[int] = None
    records: Dict[int, str] = field(default_factory=dict)
    shadow_ok: bool = True
    # What was returned: the list object, each keyed dict by its position, the
    # _row_id each carried when it was returned, and which positions are summaries.
    returned: Optional[List[Dict[str, Any]]] = None
    objects: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    ids_at_return: Dict[int, Any] = field(default_factory=dict)
    summary_positions: set = field(default_factory=set)
    bound_positions: set = field(default_factory=set)

    def cancelled(self) -> bool:
        if not callable(self.check):
            return False
        try:
            return bool(self.check())
        except Exception:
            logger.warning("LCM could not ask the host's cancellation check", exc_info=True)
            return False


@dataclass
class PendingConfirmation:
    old_session_id: str
    session_id: str
    at: float


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

    def _live_write_allowed(self, attempt: Optional[CompressAttempt]) -> bool:
        return attempt is None or (not attempt.cancelled() and self._attempt_is_current(attempt))

    def _require_live_write(self) -> None:
        """Asked before every write of live state inside an attempt."""
        attempt = _ATTEMPT.get()
        if not self._live_write_allowed(attempt):
            raise AttemptCancelled()

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
        attempt.compacted = True
        return self.compression_count + attempt.count_increment

    def _apply_attempt_outcome(self, attempt: CompressAttempt) -> None:
        for name, value in attempt.outcome.items():
            setattr(self, name, value)
        if attempt.count_increment:
            self.compression_count += attempt.count_increment

    def _record_compaction_telemetry(self) -> None:
        record = getattr(self, "_record_successful_compaction_telemetry", None)
        if callable(record):
            record()

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
            self._settle_from_list(messages)
            classified = self._classify(attempt, messages)
            if classified is None:
                return
            compaction, records = self._records.begin_compaction(
                session=attempt.session,
                kind="full" if force else "threshold",
                host_session_before=self._session_id or None,
                attempt_generation=attempt.generation if isinstance(attempt.generation, int) else None,
                entries=classified,
            )
        except Exception as exc:
            logger.warning("LCM shadow write of the compaction failed", exc_info=True)
            self._shadow_failed(attempt, "compaction_write_failed", repr(exc))
            return
        attempt.compaction = compaction
        attempt.records = records

    def _classify(self, attempt: CompressAttempt, messages: List[Dict[str, Any]]) -> Optional[List[InputEntry]]:
        """Classify the list against the session's effective return (#29 W3, W4).

        Every entry is identified first, by the plugin's key or the host's _row_id, and
        only an identified row is compared with what the store holds for it. An error
        is recorded as an event and nothing is written for this compaction (None).
        """
        store = self._records
        effective = store.effective_compaction(attempt.session)
        returned = store.return_entries(effective) if effective is not None else {}
        bound = store.bound_rows(effective) if effective is not None else {}
        insertions = store.bound_insertions(effective) if effective is not None else set()
        reusable = store.unconfirmed_inputs(attempt.session, effective)

        def fail(kind: str, detail: Any) -> None:
            self._shadow_failed(attempt, kind, detail)

        # 1. Where the effective return stands in the list, by identity.
        found: Dict[int, int] = {}      # list index -> returned position
        at_position: Dict[int, int] = {}
        for index, message in enumerate(messages):
            row_id = message.get("_row_id")
            key = parse_ret_key(message.get(RET_KEY))
            position = None
            if effective is not None and key is not None and key[0] == effective:
                position = key[1]
            elif effective is not None and isinstance(row_id, int) and row_id in bound:
                position = bound[row_id]
            if position is None or position not in returned:
                continue
            if position in at_position:
                fail("bound_entry_twice", {"compaction": effective, "position": position})
                return None
            found[index] = position
            at_position[position] = index
        if effective is not None and not found:
            fail("return_not_found",
                 f"none of the {len(messages)} entries is bound to compaction {effective}; "
                 "the list carries no identity for it (see ask A1)")
            return None

        # 2. Returned rows the host merged into another (its sequence repair records
        # the absorbed ids on the survivor, in memory only).
        absorbed_into: Dict[int, int] = {}  # returned position -> survivor's list index
        position_by_row = {row_id: position for row_id, position in bound.items()}
        for index, message in enumerate(messages):
            for row_id in message.get("_absorbed_row_ids") or ():
                if isinstance(row_id, int) and row_id in position_by_row:
                    absorbed_into[position_by_row[row_id]] = index

        # 3. What is missing: a summary is an error; a record is a revert only as a
        # suffix of the return, anything else is an error.
        present = set(at_position)
        missing = sorted(p for p in returned if p not in present and p not in absorbed_into)
        for position in missing:
            kind = returned[position][0]
            if kind == "summary":
                fail("bound_summary_missing", {"compaction": effective, "position": position})
                return None
            if any(p > position for p in present):
                fail("bound_record_missing_interior", {
                    "compaction": effective, "position": position,
                    "host_row_id": next((r for r, p in bound.items() if p == position), None),
                })
                return None
        reverted = [p for p in missing if returned[p][0] == "record"]

        # 4. A row flagged as a summary that is not the plugin's bound return.
        for index, message in enumerate(messages):
            if index not in found and message.get("_compressed_summary") is True:
                fail("unbound_summary_row", {"index": index, "host_row_id": message.get("_row_id")})
                return None

        # 5. Facts about the records the comparisons and predecessors need.
        wanted = [returned[p][1] for p in returned if returned[p][1]] + list(reusable.values())
        facts = store.record_facts(wanted)

        def predecessor_of(record: Optional[str]) -> Optional[tuple]:
            if record is None or record not in facts:
                return None
            pred = facts[record][0]
            return ("record", pred) if pred else None

        last_bound_index = max(found) if found else -1
        surviving = [returned[found[i]][1] for i in sorted(found) if returned[found[i]][0] == "record"]
        if surviving:
            fallback: Optional[tuple] = ("record", surviving[-1])
        elif reverted:
            fallback = predecessor_of(returned[reverted[0]][1])
        elif effective is not None and store.chain_end(effective):
            fallback = ("record", store.chain_end(effective))
        else:
            fallback = None

        # 6. The entries, in list order. Chain-bearing entries (records on the active
        # branch) move the chain; host insertions and summary revisions stand beside it.
        entries: List[InputEntry] = []
        chain: Optional[tuple] = None
        for index, message in enumerate(messages):
            row_id = message.get("_row_id")
            row_id = row_id if isinstance(row_id, int) else None
            merged = sorted(p for p, survivor in absorbed_into.items() if survivor == index)
            merged_records = [returned[p][1] for p in merged if returned[p][0] == "record" and returned[p][1]]
            beside = chain if chain is not None else fallback
            if index in found:
                position = found[index]
                kind, record, _derivation, raw = returned[position]
                if kind == "summary":
                    if merged_records or (raw is not None and rewritten(message, raw)):
                        entries.append(InputEntry(
                            index, row_id, "revision", message=message, pred=beside,
                            sources=[("return", effective, position)] + [("record", r) for r in merged_records]))
                    else:
                        entries.append(InputEntry(index, row_id, "bound_summary"))
                    continue
                if merged_records or (record in facts and rewritten(message, facts[record][1])):
                    originals = [record] + merged_records if record else merged_records
                    earliest = min(originals, key=lambda r: next(
                        (p for p, e in returned.items() if e[1] == r), 1 << 30))
                    entries.append(InputEntry(
                        index, row_id, "revision", message=message, pred=predecessor_of(earliest),
                        sources=[("record", r) for r in originals]))
                    chain = ("entry", index)
                else:
                    entries.append(InputEntry(index, row_id, "bound", record=record))
                    chain = ("record", record) if record else chain
                continue
            if index == 0 and message.get("role") == "system":
                entries.append(InputEntry(index, row_id, "system"))
            elif row_id is not None and row_id in insertions:
                entries.append(InputEntry(index, row_id, "host_insertion", message=message, pred=beside))
            elif row_id is not None and row_id in reusable:
                record = reusable[row_id]
                if merged_records or (record in facts and rewritten(message, facts[record][1])):
                    entries.append(InputEntry(
                        index, row_id, "revision", message=message, pred=predecessor_of(record),
                        sources=[("record", record)] + [("record", r) for r in merged_records]))
                    chain = ("entry", index)
                else:
                    entries.append(InputEntry(index, row_id, "reused", record=record))
                    chain = ("record", record)
            elif index < last_bound_index:
                entries.append(InputEntry(index, row_id, "host_insertion", message=message, pred=beside))
            elif merged_records:
                # A new row that absorbed a returned row: it holds that row's content.
                earliest = merged_records[0]
                entries.append(InputEntry(
                    index, row_id, "revision", message=message, pred=predecessor_of(earliest),
                    sources=[("record", r) for r in merged_records]))
                chain = ("entry", index)
            else:
                entries.append(InputEntry(index, row_id, "transcript", message=message, pred=beside))
                chain = ("entry", index)
        return entries

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
        entries: list[tuple[int, str, Optional[str], Optional[str], Optional[str]]] = []
        try:
            for position, message in enumerate(result):
                input_position = by_object.get(id(message))
                if input_position is not None and input_position in attempt.records:
                    entries.append((position, "record", attempt.records[input_position], None, None))
                elif message.get("_compressed_summary") is True and input_position is None:
                    entries.append((position, "summary", None, None, raw_json(message)))
                elif input_position == 0 and message.get("role") == "system":
                    continue  # the host's system row, returned in place, not recorded
                else:
                    self._shadow_failed(attempt, "return_entry_unknown",
                                        {"position": position, "role": message.get("role")})
                    return
            self._records.write_returns(attempt.compaction, entries)
        except Exception as exc:
            logger.warning("LCM shadow write of the return failed", exc_info=True)
            self._shadow_failed(attempt, "return_write_failed", repr(exc))
            return
        for position, kind, _record, _derivation, _raw in entries:
            message = result[position]
            message[RET_KEY] = f"{attempt.compaction}:{position}"
            attempt.objects[position] = message
            attempt.ids_at_return[position] = message.get("_row_id")
            if kind == "summary":
                attempt.summary_positions.add(position)
        attempt.returned = result
        self._returned_attempts[attempt.compaction] = attempt
        self._last_returned_attempt = attempt

    # --- Confirmation, adoption, rejection and binding ---------------------------------

    def _stamped(self, attempt: CompressAttempt, message: Dict[str, Any], position: int) -> Optional[int]:
        """The _row_id the host stamped on a returned entry at its commit, or None.

        The host inserts every committed dict as a fresh row and writes the new
        AUTOINCREMENT id into it (``hermes_state_messages.py:515-535``), so a stamped
        id always differs from the id the dict carried when it was returned. A dict
        still carrying its pre-commit id was not committed as itself (the host's
        salvage commits copies, ``agent/context_compressor.py:587``)."""
        row_id = message.get("_row_id")
        if not isinstance(row_id, int) or row_id == attempt.ids_at_return.get(position):
            return None
        return row_id

    def _returns_to_bind(
        self,
        attempt: CompressAttempt,
        messages: List[Dict[str, Any]],
        *,
        own_objects_only: bool,
    ) -> list[tuple[int, int, int]]:
        """(position, stamped row id, index in the list) for each keyed dict of this
        attempt, by the position in its own key. A position seen twice is bound not at
        all, and recorded."""
        found: dict[int, list[tuple[int, int]]] = {}
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                continue
            key = parse_ret_key(message.get(RET_KEY))
            if key is None or key[0] != attempt.compaction:
                continue
            position = key[1]
            if own_objects_only and message is not attempt.objects.get(position):
                continue
            row_id = self._stamped(attempt, message, position)
            if row_id is not None:
                found.setdefault(position, []).append((row_id, index))
        pairs: list[tuple[int, int, int]] = []
        for position, seen in found.items():
            if len(seen) == 1:
                pairs.append((position, seen[0][0], seen[0][1]))
            else:
                self._records.event("binding_position_duplicated", session=attempt.session,
                                    compaction=attempt.compaction, detail={"position": position, "count": len(seen)})
        return pairs

    def _bind(self, attempt: CompressAttempt, returns, insertions=()) -> None:
        written = self._records.bind(attempt.compaction, returns, insertions)
        attempt.bound_positions.update(written)
        if set(attempt.objects) <= attempt.bound_positions:
            self._returned_attempts.pop(attempt.compaction, None)

    def _record_confirmation(self, old_session_id: str, session_id: str) -> None:
        """The host committed a compaction. It names no list; the committed attempt is
        the one whose own returned dicts now carry a _row_id stamped at this commit."""
        try:
            committed = [
                attempt for attempt in self._returned_attempts.values()
                if attempt.returned is not None
                and not self._records.is_settled(attempt.compaction)
                and self._returns_to_bind(attempt, attempt.returned, own_objects_only=True)
            ]
            if len(committed) > 1:
                self._records.event("confirmation_ambiguous", session=self._plugin_session or None,
                                    detail={"compactions": [a.compaction for a in committed]})
                return
            if not committed:
                # Nothing of the plugin's own was committed as itself: the host committed
                # copies, or a list the plugin did not return. The next list's keys say
                # which compaction this was.
                self._pending_confirmation = PendingConfirmation(old_session_id, session_id, time.time())
                self._records.event("confirmation_unattributed", session=self._plugin_session or None,
                                    detail={"old_session_id": old_session_id, "session_id": session_id})
                return
            attempt = committed[0]
            if attempt.cancelled():
                self._records.event("confirmation_of_cancelled_attempt", session=attempt.session,
                                    compaction=attempt.compaction)
                return
            self._records.confirm(attempt.compaction, host_session_before=old_session_id,
                                  host_session_after=session_id)
            returns = self._returns_to_bind(attempt, attempt.returned, own_objects_only=True)
            insertions = [
                (message["_row_id"], index)
                for index, message in enumerate(attempt.returned)
                if isinstance(message, dict) and RET_KEY not in message and isinstance(message.get("_row_id"), int)
            ]
            self._bind(attempt, returns, insertions)
        except Exception as exc:
            logger.warning("LCM could not record the host's confirmation", exc_info=True)
            self._records.event("confirmation_write_failed", session=self._plugin_session or None, detail=repr(exc))

    def _settle_from_list(self, messages: Optional[List[Dict[str, Any]]]) -> None:
        """Read the keys of a list the host hands over: attribute a waiting
        confirmation, find a return adopted without one, and bind stamped entries."""
        if not messages or not self._plugin_session or not self._returned_attempts:
            return
        keyed: set[int] = set()
        for message in messages:
            if isinstance(message, dict):
                key = parse_ret_key(message.get(RET_KEY))
                if key is not None and key[0] in self._returned_attempts:
                    keyed.add(key[0])
        for compaction in sorted(keyed):
            attempt = self._returned_attempts.get(compaction)
            if attempt is None or attempt.session != self._plugin_session:
                continue
            try:
                store = self._records
                if not store.is_settled(compaction):
                    effective = store.effective_compaction(self._plugin_session)
                    if effective is not None and compaction <= effective:
                        continue
                    pending = getattr(self, "_pending_confirmation", None)
                    if pending is not None:
                        store.confirm(compaction, host_session_before=pending.old_session_id,
                                      host_session_after=pending.session_id, at=pending.at)
                        self._pending_confirmation = None
                    elif any(
                        parse_ret_key(m.get(RET_KEY)) == (compaction, position)
                        for m in messages if isinstance(m, dict)
                        for position in attempt.summary_positions
                    ):
                        store.adopt(compaction, evidence="its returned summary entry stands in the next list")
                    else:
                        continue
                if store.effective_compaction(self._plugin_session) != compaction:
                    continue
                self._bind(attempt, self._returns_to_bind(attempt, messages, own_objects_only=False))
            except Exception as exc:
                logger.warning("LCM could not settle a returned compaction", exc_info=True)
                self._records.event("settle_write_failed", session=self._plugin_session, compaction=compaction,
                                    detail=repr(exc))

    def record_rejected_compaction(self, *args: Any, **kwargs: Any) -> None:
        """The host refused the result this copy last returned (for example a grown one)."""
        try:
            attempt = getattr(self, "_last_returned_attempt", None)
            if attempt is None or self._records.is_settled(attempt.compaction):
                self._records.event("rejection_without_attempt", session=self._plugin_session or None)
                return
            self._records.reject(attempt.compaction, how="record_rejected_compaction")
            self._returned_attempts.pop(attempt.compaction, None)
        except Exception as exc:
            logger.warning("LCM could not record the host's rejection", exc_info=True)
            self._records.event("rejection_write_failed", session=self._plugin_session or None, detail=repr(exc))

    def _bind_from_list(self, messages: Optional[List[Dict[str, Any]]]) -> None:
        self._settle_from_list(messages)

    def on_turn_complete(self, messages: List[Dict[str, Any]], usage: Dict[str, Any] = None, **kwargs: Any) -> None:
        """The first list after a commit settles and binds what the confirmation could not."""
        self._bind_from_list(messages)
