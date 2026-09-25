"""Backup operations for the LCM store.

``backup_database`` is the primitive behind ``/lcm backup``; it flushes the
engine's SQLite connections and snapshots the store to a timestamped file.
It is a pure function that
take the engine so the command layer (``command.py``) keeps only the text
formatting, and the connection handling lives in one place.
"""

from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import sqlite3
import stat
from typing import Any

from .sqlite_util import (
    _prepare_private_sqlite_file,
    _restrict_existing_sqlite_artifacts,
)


_FCHMOD = getattr(os, "fchmod", None)


def _prepare_private_backup_directory(path: Path) -> None:
    path.mkdir(parents=True, mode=0o700, exist_ok=True)
    expected = os.lstat(path)
    if stat.S_ISLNK(expected.st_mode) or not stat.S_ISDIR(expected.st_mode):
        raise OSError(f"backup directory is not a real directory: {path}")

    if os.name != "posix":  # pragma: no cover - Windows compatibility fallback
        path.chmod(0o700)
        return

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        opened = os.fstat(fd)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or (opened.st_dev, opened.st_ino) != (expected.st_dev, expected.st_ino)
        ):
            raise OSError(f"backup directory changed during validation: {path}")
        if callable(_FCHMOD):
            _FCHMOD(fd, 0o700)
        elif os.chmod in getattr(os, "supports_fd", ()):
            os.chmod(fd, 0o700)
        else:
            raise OSError("descriptor-based directory chmod is unavailable")
    finally:
        os.close(fd)

def flush_engine_connections(engine) -> None:
    """Commit pending writes on every SQLite connection the engine owns.

    Called by ``backup_database`` before it snapshots the store.
    """
    engine._store.commit()
    engine._dag._conn.commit()


def backup_database(engine) -> dict[str, Any]:
    db_path = Path(engine._store.db_path)
    if not db_path.exists():
        return {
            "ok": False,
            "db_path": db_path,
            "error": "database file does not exist",
        }

    backup_dir = engine.backup_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"{db_path.stem}-{timestamp}.sqlite3"

    try:
        _prepare_private_backup_directory(backup_dir)
        flush_engine_connections(engine)
        _prepare_private_sqlite_file(backup_path)

        dest = sqlite3.connect(str(backup_path))
        try:
            engine._store.backup(dest)
        finally:
            dest.close()
        _restrict_existing_sqlite_artifacts(backup_path)
    except (OSError, sqlite3.Error) as exc:
        return {
            "ok": False,
            "db_path": db_path,
            "error": str(exc),
        }

    backup_size = backup_path.stat().st_size if backup_path.exists() else 0
    return {
        "ok": True,
        "db_path": db_path,
        "backup_path": backup_path,
        "backup_size": backup_size,
    }
