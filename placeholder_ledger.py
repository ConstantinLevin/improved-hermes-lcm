"""Ledger of generated quarantine placeholders for the LCM engine (WS5 seam).

The ``PlaceholderLedgerMixin`` keeps the digests of the volatile placeholders
that assistant-output quarantine writes into active replay, so a replayed
placeholder is recognised and not stored as content: session-scoped metadata
keys, the digest list and its copy to a new session at a compression boundary,
and the per-digest counts and ordinals of the placeholders in active replay.
The methods run bound to the engine instance (``self`` is the ``LCMEngine``)
and read the engine's ``_store``, ``_session_id`` and
``_get_store_id_map_for_messages``.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from .message_content import text_content_for_pattern_matching

logger = logging.getLogger(__name__)


class PlaceholderLedgerMixin:
    @staticmethod
    def _is_volatile_ignored_quarantine_placeholder(msg: Dict[str, Any], text: str) -> bool:
        if str(msg.get("role") or "") != "assistant":
            return False
        return bool(
            re.fullmatch(
                r"\[LCM active replay placeholder: assistant output quarantined; "
                r"kind=quarantined_assistant_output; "
                r"reason=[A-Za-z0-9_.:/-]+; "
                r"scope=ignored_message_pattern; field=content; "
                r"chars=\d+; bytes=\d+; "
                r"sha256=[0-9a-f]{16}\]",
                text.strip(),
            )
        )

    @staticmethod
    def _active_replay_placeholder_digest(text: str) -> Optional[str]:
        match = re.search(r"sha256=([0-9a-f]{16})\]$", text.strip())
        return match.group(1) if match else None


    def _ignored_placeholder_metadata_keys(self) -> list[str]:
        return self._session_scoped_hash_metadata_keys("ignored_active_replay_placeholder_hashes")

    def _ignored_placeholder_count_metadata_keys(self) -> list[str]:
        return self._session_scoped_hash_metadata_keys("ignored_active_replay_placeholder_hash_counts")

    def _ignored_placeholder_ordinal_metadata_keys(self) -> list[str]:
        return self._session_scoped_hash_metadata_keys("ignored_active_replay_placeholder_hash_ordinals")


    def _session_scoped_hash_metadata_keys(self, prefix: str, session_id: str | None = None) -> list[str]:
        scoped_session_id = self._session_id if session_id is None else session_id
        keys: list[str] = []
        if scoped_session_id:
            keys.append(f"{prefix}:{scoped_session_id}")
        return list(dict.fromkeys(keys))

    def _copy_generated_ignore_hashes_to_session(
        self,
        source_session_id: str,
        target_session_id: str,
    ) -> None:
        if not source_session_id or not target_session_id or source_session_id == target_session_id:
            return
        source_keys = self._session_scoped_hash_metadata_keys(
            "ignored_active_replay_placeholder_hashes",
            source_session_id,
        )
        target_keys = self._session_scoped_hash_metadata_keys(
            "ignored_active_replay_placeholder_hashes",
            target_session_id,
        )
        for digest in self._load_hash_list_for_metadata_keys(source_keys):
            self._remember_hash_for_metadata_keys(digest, target_keys)

    def _load_hash_list_for_metadata_keys(self, keys: list[str]) -> list[str]:
        if not keys:
            return []
        try:
            ordered: list[str] = []
            seen: set[str] = set()
            for key in keys:
                data = self._store.read_metadata_json(key)
                if isinstance(data, list):
                    for item in data:
                        digest = str(item)
                        if re.fullmatch(r"[0-9a-f]{16}", digest) and digest not in seen:
                            ordered.append(digest)
                            seen.add(digest)
            return ordered
        except Exception:
            logger.debug("LCM scoped hash metadata load failed", exc_info=True)
        return []

    def _remember_hash_for_metadata_keys(self, digest: str, keys: list[str]) -> list[str]:
        if not re.fullmatch(r"[0-9a-f]{16}", digest):
            return []
        ordered_hashes = self._load_hash_list_for_metadata_keys(keys)
        ordered_hashes = [item for item in ordered_hashes if item != digest]
        ordered_hashes.append(digest)
        ordered_hashes = ordered_hashes[-512:]
        if not keys:
            return ordered_hashes
        try:
            payload = json.dumps(ordered_hashes)
            self._store.write_metadata_json(keys, payload)
        except Exception:
            logger.debug("LCM scoped hash metadata write failed", exc_info=True)
        return ordered_hashes

    def _load_generated_ignored_placeholder_hashes(self) -> set[str]:
        return set(self._load_generated_ignored_placeholder_hash_list())

    def _load_generated_ignored_placeholder_hash_list(self) -> list[str]:
        return self._load_hash_list_for_metadata_keys(self._ignored_placeholder_metadata_keys())

    def _load_generated_ignored_placeholder_hash_counts(
        self,
        keys: Optional[list[str]] = None,
    ) -> dict[str, int]:
        count_keys = self._ignored_placeholder_count_metadata_keys() if keys is None else keys
        counts: dict[str, int] = {}
        if not count_keys:
            return counts
        try:
            for key in count_keys:
                data = self._store.read_metadata_json(key)
                if not isinstance(data, dict):
                    continue
                for digest, count in data.items():
                    digest = str(digest)
                    if not re.fullmatch(r"[0-9a-f]{16}", digest):
                        continue
                    try:
                        parsed_count = max(0, int(count))
                    except (TypeError, ValueError):
                        continue
                    counts[digest] = max(counts.get(digest, 0), parsed_count)
        except Exception:
            logger.debug("LCM ignored placeholder count metadata load failed", exc_info=True)
        return counts

    def _write_generated_ignored_placeholder_hash_counts(
        self,
        counts: dict[str, int],
        keys: Optional[list[str]] = None,
    ) -> None:
        count_keys = self._ignored_placeholder_count_metadata_keys() if keys is None else keys
        if not count_keys:
            return
        payload: dict[str, int] = {}
        for digest, count in counts.items():
            digest = str(digest)
            if not re.fullmatch(r"[0-9a-f]{16}", digest):
                continue
            try:
                parsed_count = int(count)
            except (TypeError, ValueError):
                continue
            if parsed_count > 0:
                payload[digest] = parsed_count
        try:
            serialized = json.dumps(payload, sort_keys=True)
            # skip_unchanged avoids the fsync commit (under synchronous=FULL) when
            # the stored value already matches; this runs on every ingest.
            self._store.write_metadata_json(count_keys, serialized, skip_unchanged=True)
        except Exception:
            logger.debug("LCM ignored placeholder count metadata write failed", exc_info=True)

    def _load_generated_ignored_placeholder_hash_ordinals(
        self,
        keys: Optional[list[str]] = None,
    ) -> dict[str, set[int]]:
        ordinal_keys = self._ignored_placeholder_ordinal_metadata_keys() if keys is None else keys
        ordinals: dict[str, set[int]] = {}
        if not ordinal_keys:
            return ordinals
        try:
            for key in ordinal_keys:
                data = self._store.read_metadata_json(key)
                if not isinstance(data, dict):
                    continue
                for digest, values in data.items():
                    digest = str(digest)
                    if not re.fullmatch(r"[0-9a-f]{16}", digest) or not isinstance(values, list):
                        continue
                    bucket = ordinals.setdefault(digest, set())
                    for value in values:
                        try:
                            parsed = int(value)
                        except (TypeError, ValueError):
                            continue
                        if parsed > 0:
                            bucket.add(parsed)
        except Exception:
            logger.debug("LCM ignored placeholder ordinal metadata load failed", exc_info=True)
        return ordinals

    def _write_generated_ignored_placeholder_hash_ordinals(
        self,
        ordinals: dict[str, Any],
        keys: Optional[list[str]] = None,
    ) -> None:
        ordinal_keys = self._ignored_placeholder_ordinal_metadata_keys() if keys is None else keys
        if not ordinal_keys:
            return
        payload: dict[str, list[int]] = {}
        for digest, values in ordinals.items():
            digest = str(digest)
            if not re.fullmatch(r"[0-9a-f]{16}", digest):
                continue
            clean_values: set[int] = set()
            for value in values:
                try:
                    parsed = int(value)
                except (TypeError, ValueError):
                    continue
                if parsed > 0:
                    clean_values.add(parsed)
            clean = sorted(clean_values)
            if clean:
                payload[digest] = clean
        try:
            serialized = json.dumps(payload, sort_keys=True)
            # Skip the write (and its fsync commit) when unchanged; see the counts
            # writer above for rationale.
            self._store.write_metadata_json(ordinal_keys, serialized, skip_unchanged=True)
        except Exception:
            logger.debug("LCM ignored placeholder ordinal metadata write failed", exc_info=True)

    def _active_replay_generated_placeholder_digest_budget(self) -> dict[str, int]:
        return self._generated_placeholder_digest_budget_for_active_replay(
            self._last_active_replay_messages
        )

    def _generated_placeholder_digest_ordinals_for_active_replay(
        self,
        active_replay_messages: List[Dict[str, Any]],
    ) -> dict[str, set[int]]:
        generated_hashes = self._load_generated_ignored_placeholder_hashes()
        if not generated_hashes or not active_replay_messages:
            return {}
        stored_message_ids = set(self._get_store_id_map_for_messages(active_replay_messages))
        occurrence_by_digest: dict[str, int] = {}
        ordinals: dict[str, set[int]] = {}
        for msg in active_replay_messages:
            text = text_content_for_pattern_matching(msg.get("content")) or ""
            digest = self._active_replay_placeholder_digest(text)
            if not digest or digest not in generated_hashes:
                continue
            occurrence_by_digest[digest] = occurrence_by_digest.get(digest, 0) + 1
            if id(msg) in stored_message_ids:
                continue
            ordinals.setdefault(digest, set()).add(occurrence_by_digest[digest])
        return ordinals

    def _generated_placeholder_digest_budget_for_active_replay(
        self,
        active_replay_messages: List[Dict[str, Any]],
    ) -> dict[str, int]:
        generated_hashes = self._load_generated_ignored_placeholder_hashes()
        if not generated_hashes or not active_replay_messages:
            return {}
        stored_message_ids = set(self._get_store_id_map_for_messages(active_replay_messages))
        budget: dict[str, int] = {}
        for msg in active_replay_messages:
            if id(msg) in stored_message_ids:
                continue
            text = text_content_for_pattern_matching(msg.get("content")) or ""
            digest = self._active_replay_placeholder_digest(text)
            if digest and digest in generated_hashes:
                budget[digest] = budget.get(digest, 0) + 1
        return budget


    def _remember_generated_ignored_placeholder_hash(self, digest: str) -> None:
        self._remember_hash_for_metadata_keys(
            digest,
            self._ignored_placeholder_metadata_keys(),
        )


