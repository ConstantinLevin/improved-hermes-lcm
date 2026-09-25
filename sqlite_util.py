"""SQLite file helpers: create the store's file with a private mode."""

from __future__ import annotations

import errno
import os
from pathlib import Path
import stat


def _sqlite_artifact_error(path: Path, reason: str) -> OSError:
    return OSError(errno.EPERM, f"refusing SQLite artifact {path.name!r}: {reason}", str(path))


def _same_file_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _open_private_sqlite_directory(path: Path) -> int:
    directory = path.parent
    expected = os.stat(directory, follow_symlinks=False)
    if not stat.S_ISDIR(expected.st_mode):
        raise _sqlite_artifact_error(path, "parent is not a regular directory")
    if expected.st_mode & 0o022:
        raise _sqlite_artifact_error(path, "parent directory is writable by another user")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_BINARY", 0)
    directory_fd = os.open(directory, flags)
    opened = os.fstat(directory_fd)
    if not stat.S_ISDIR(opened.st_mode) or not _same_file_identity(expected, opened):
        os.close(directory_fd)
        raise _sqlite_artifact_error(path, "parent directory changed while opening")
    return directory_fd


def _create_private_sqlite_file(path: Path) -> bool:
    """Create a new database file with mode 0600 atomically, or leave an existing one alone.

    The final path is never opened by the plugin outside SQLite: closing any
    descriptor on a database releases the POSIX locks SQLite holds on it through
    every connection in the process, and another thread may open the path through
    SQLite the moment it exists. So a uniquely named temporary file is created in
    the same directory (``O_CREAT | O_EXCL``, 0600) and closed, then hard-linked
    to the final name, and the temporary name is removed. SQLite creates the
    ``-journal`` file with the database file's mode.

    Returns True when this call created the file. When the name already exists,
    it must be a regular file (symbolic links followed); anything else raises.
    """
    directory_fd = _open_private_sqlite_directory(path)
    temporary = f".{path.name}.{os.getpid()}.{os.urandom(8).hex()}.tmp"
    try:
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(temporary, flags, 0o600, dir_fd=directory_fd)
        os.close(fd)
        try:
            os.link(
                temporary,
                path.name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
            created = True
        except FileExistsError:
            created = False
        finally:
            os.unlink(temporary, dir_fd=directory_fd)
        if not created:
            try:
                existing = os.stat(path.name, dir_fd=directory_fd)
            except FileNotFoundError:
                existing = None
            if existing is None or not stat.S_ISREG(existing.st_mode):
                raise _sqlite_artifact_error(
                    path,
                    "the name is taken by something that is not a regular file "
                    "(for example a directory or a symbolic link to a missing file)",
                )
        return created
    finally:
        os.close(directory_fd)
