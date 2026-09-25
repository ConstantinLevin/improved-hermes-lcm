"""The store's daily backup (#6): one slot per store, replaced whole, checked first.

The slot is ``<store dir>/backups/lcm/<store stem>.daily.sqlite3``. It belongs to the
store, not to a Hermes home: two homes pointing at one store share it.

- **When.** At every ``on_session_start`` (a new agent, and a compaction's
  confirmation) the engine asks whether a backup is due. It is due when there is no
  slot, when the slot holds another store, or when the slot is older than a day and
  the store has changed since. Changed means the store file's modification time is
  later than the slot's; it is read with ``stat``, which never opens the file. After
  a failed backup the next attempt waits an hour.
- **How.** On a daemon thread of its own, so no agent creation or compaction waits
  for it. One process at a time: a non-blocking ``flock`` on a lock file beside the
  slot; a process that does not get it skips. The copy is taken through SQLite's
  backup API from a read-only connection of this thread, never as a file copy, in
  steps, so writers commit between them; SQLite restarts the copy when another
  connection writes, so the copy is one state of the store.
- **Checked before it replaces the slot.** ``PRAGMA integrity_check`` is ok, the
  identity row equals the store's, and the record's invariant (#29 W7) passes on the
  copy. Only then does the copy replace the slot, atomically; the slot is always the
  last copy that passed. A slot that holds another store (a store begun again after
  a refusal) is set aside under its store's uuid and never overwritten.
- **A failure keeps the old slot.** The temporary copy is removed, and a store event
  ``backup_failed`` names the cause (the store event logs a warning; the doctor lists
  it). A backup stopped because its engine was closed records ``backup_interrupted``.
- **Restore** is the owner's, by hand and offline (skill reference, diagnostics).
"""

from __future__ import annotations

import fcntl
import logging
import os
import sqlite3
import stat
import threading
import time
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote

from .db_bootstrap import STORE_FORMAT

logger = logging.getLogger(__name__)

SLOT_MAX_AGE_SECONDS = 24 * 3600
RETRY_AFTER_FAILURE_SECONDS = 3600
COPY_STEP_PAGES = 256
COPY_STEP_SLEEP_SECONDS = 0.05
COPY_TIME_LIMIT_SECONDS = 600


class BackupStopped(Exception):
    """The backup was stopped because its engine was closed."""


class BackupFailed(Exception):
    """The backup could not be taken, or its copy failed a check."""


def slot_path(db_path: str | Path) -> Path:
    db_path = Path(db_path)
    return db_path.parent / "backups" / "lcm" / f"{db_path.stem}.daily.sqlite3"


def _prepare_private_directory(path: Path) -> None:
    """Create the backup directory 0700, refusing anything but a real directory."""
    path.mkdir(parents=True, mode=0o700, exist_ok=True)
    expected = os.lstat(path)
    if stat.S_ISLNK(expected.st_mode) or not stat.S_ISDIR(expected.st_mode):
        raise BackupFailed(f"the backup directory is not a real directory: {path}")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(fd)
        if not stat.S_ISDIR(opened.st_mode) or (opened.st_dev, opened.st_ino) != (expected.st_dev, expected.st_ino):
            raise BackupFailed(f"the backup directory changed while it was checked: {path}")
        os.fchmod(fd, 0o700)
    finally:
        os.close(fd)


def _read_only(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{quote(str(path))}?mode=ro", uri=True, timeout=30.0)


def _identity(conn: sqlite3.Connection) -> Optional[tuple[str, str]]:
    try:
        rows = conn.execute("SELECT format, store_uuid FROM store_identity").fetchall()
    except sqlite3.Error:
        return None
    return (str(rows[0][0]), str(rows[0][1])) if len(rows) == 1 else None


def _fsync_path(path: Path, *, directory: bool = False) -> None:
    fd = os.open(path, os.O_RDONLY | (getattr(os, "O_DIRECTORY", 0) if directory else 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class DailyBackup:
    """The daily backup of one store, as one engine starts and stops it.

    It holds the store's record helper (for its events), never the engine, so an
    engine can be collected while its backup runs; the engine's finalizer stops it.
    """

    def __init__(self, db_path: str | Path, records: Any):
        self.db_path = Path(db_path)
        self.slot = slot_path(self.db_path)
        self._records = records
        self._stop = threading.Event()
        self._guard = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        # The uuid last read from the slot, keyed by the slot's modification time, so
        # the slot is opened only when it has changed.
        self._slot_identity: tuple[float, Optional[str]] = (-1.0, None)

    # --- When ---------------------------------------------------------------------

    def _slot_store_uuid(self, mtime: float) -> Optional[str]:
        if self._slot_identity[0] != mtime:
            conn = _read_only(self.slot)
            try:
                found = _identity(conn)
            finally:
                conn.close()
            self._slot_identity = (mtime, found[1] if found else None)
        return self._slot_identity[1]

    def _store_uuid(self) -> Optional[str]:
        rows = self._records._q("SELECT store_uuid FROM store_identity")
        return str(rows[0][0]) if rows else None

    def due(self) -> bool:
        now = time.time()
        try:
            slot = os.stat(self.slot)
        except FileNotFoundError:
            slot = None
        if slot is not None and self._slot_store_uuid(slot.st_mtime) == self._store_uuid():
            if now - slot.st_mtime < SLOT_MAX_AGE_SECONDS:
                return False
            if os.stat(self.db_path).st_mtime <= slot.st_mtime:
                return False  # nothing written since the slot was taken
        last_failure = self._records._q("SELECT MAX(at) FROM store_events WHERE kind = 'backup_failed'")[0][0]
        return not (last_failure and now - float(last_failure) < RETRY_AFTER_FAILURE_SECONDS)

    def start_if_due(self) -> Optional[threading.Thread]:
        """Start the backup on its own thread when one is due; never raises.

        The whole decision and the thread's start run under ``_guard``, which
        :meth:`stop` takes too: a stop waits for a start in progress, and once
        stopped the backup never starts again."""
        with self._guard:
            if self._stop.is_set() or (self._thread is not None and self._thread.is_alive()):
                return None
            try:
                if not self.due():
                    return None
            except Exception:
                logger.warning("LCM could not tell whether the daily backup of %s is due", self.db_path,
                               exc_info=True)
                return None
            thread = threading.Thread(target=self._run, name="lcm-daily-backup", daemon=True)
            self._thread = thread
            thread.start()
            return thread

    def stop(self) -> None:
        """Stop the backup for good and wait for a running one to end; the slot stays
        as it was. Taken under ``_guard``, so a start in progress finishes first (and
        its thread is started, so it can be joined), and no start follows."""
        with self._guard:
            self._stop.set()
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join()

    # --- How ----------------------------------------------------------------------

    def _run(self) -> None:
        try:
            self.take()
        except BackupStopped as exc:
            self._event("backup_interrupted", str(exc))
        except Exception as exc:
            self._event("backup_failed", f"{type(exc).__name__}: {exc}")

    def _event(self, kind: str, detail: str) -> None:
        try:
            self._records.event(kind, detail={"slot": str(self.slot), "cause": detail})
        except Exception:
            logger.warning("LCM could not record %s for the daily backup of %s: %s", kind, self.db_path, detail,
                           exc_info=True)

    def _check_stop(self) -> None:
        if self._stop.is_set():
            raise BackupStopped("the backup was stopped because its engine was closed")

    def take(self) -> Optional[Path]:
        """Take the backup now, if this process gets the slot's lock and it is still
        due. Returns the slot, or None when skipped. A stopped backup never takes."""
        self._check_stop()
        directory = self.slot.parent
        _prepare_private_directory(directory)
        lock_path = directory / f".{self.slot.name}.lock"
        lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
        try:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                logger.debug("LCM skips the daily backup of %s: another process is taking it", self.db_path)
                return None
            if not self.due():  # another process may have just taken it
                return None
            return self._take_locked()
        finally:
            os.close(lock_fd)

    def _take_locked(self) -> Path:
        tmp = self.slot.with_name(self.slot.name + ".tmp")
        try:
            tmp.unlink(missing_ok=True)  # a copy a killed process left behind
            os.close(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600))
            store_uuid = self._copy(tmp)
            self._check_stop()
            _fsync_path(tmp)
            self._check_copy(tmp, store_uuid)
            # An interrupted backup publishes nothing: checked again right before
            # each step that changes the slot.
            self._check_stop()
            self._set_aside_foreign_slot(store_uuid)
            self._check_stop()
            os.replace(tmp, self.slot)
            _fsync_path(self.slot.parent, directory=True)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        size = os.stat(self.slot).st_size
        logger.info("LCM took the daily backup of %s into %s (%d bytes)", self.db_path, self.slot, size)
        return self.slot

    def _copy(self, tmp: Path) -> str:
        """Copy the store into ``tmp`` through the backup API; returns its uuid."""
        source = _read_only(self.db_path)
        try:
            identity = _identity(source)
            if identity is None or identity[0] != STORE_FORMAT:
                raise BackupFailed(f"the store at {self.db_path} has no identity row of format {STORE_FORMAT}")
            deadline = time.monotonic() + COPY_TIME_LIMIT_SECONDS
            target = sqlite3.connect(str(tmp))
            try:
                def progress(_status: int, _remaining: int, _total: int) -> None:
                    # Called after each step, with the source's lock released: the
                    # pause lets a writer commit between steps.
                    self._check_stop()
                    if time.monotonic() > deadline:
                        raise BackupFailed(f"the copy did not finish within {COPY_TIME_LIMIT_SECONDS} s")
                    time.sleep(COPY_STEP_SLEEP_SECONDS)
                    self._check_stop()

                source.backup(target, pages=COPY_STEP_PAGES, progress=progress)
            finally:
                target.close()
            return identity[1]
        finally:
            source.close()

    def _check_copy(self, tmp: Path, store_uuid: str) -> None:
        self._check_stop()
        conn = _read_only(tmp)
        try:
            integrity = [str(row[0]) for row in conn.execute("PRAGMA integrity_check").fetchall()]
            if integrity != ["ok"]:
                raise BackupFailed(f"the copy fails integrity_check: {'; '.join(integrity[:3])}")
            if _identity(conn) != (STORE_FORMAT, store_uuid):
                raise BackupFailed("the copy's identity row is not the store's")
        finally:
            conn.close()
        self._check_stop()
        from .record_store import RecordStore

        copy_store = RecordStore(tmp)
        try:
            failing = [report for report in copy_store.check_invariant() if report["status"] != "pass"]
        finally:
            copy_store.close("the backup's check ended")
        self._check_stop()
        if failing:
            first = failing[0]
            problem = first["problems"][0] if first["problems"] else "a problem"
            raise BackupFailed(
                f"the copy fails the record's invariant in {len(failing)} session(s), "
                f"for example session {first['session']}: {problem}"
            )

    def _set_aside_foreign_slot(self, store_uuid: str) -> None:
        """A slot that holds another store is kept under that store's uuid, never
        overwritten."""
        if not self.slot.exists():
            return
        conn = _read_only(self.slot)
        try:
            found = _identity(conn)
        finally:
            conn.close()
        if found is not None and found[1] == store_uuid:
            return
        label = found[1] if found else "unknown"
        aside = self.slot.with_name(f"{self.db_path.stem}.daily.{label}.sqlite3")
        counter = 1
        while aside.exists():
            aside = self.slot.with_name(f"{self.db_path.stem}.daily.{label}.{counter}.sqlite3")
            counter += 1
        os.rename(self.slot, aside)
        logger.warning("LCM set the daily backup slot of another store aside, kept as %s", aside)

    # --- What the doctor shows ------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        try:
            slot = os.stat(self.slot)
        except FileNotFoundError:
            return {"slot": str(self.slot), "taken": False}
        return {
            "slot": str(self.slot),
            "taken": True,
            "taken_at": slot.st_mtime,
            "age_hours": round((time.time() - slot.st_mtime) / 3600, 1),
            "size_bytes": slot.st_size,
        }
