"""SQLite file helpers: create and restrict the store's files with private modes."""

from __future__ import annotations

import errno
import os
from pathlib import Path
import stat


_SQLITE_SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")


def _sqlite_artifact_error(path: Path, reason: str) -> OSError:
    return OSError(errno.EPERM, f"refusing SQLite artifact {path.name!r}: {reason}", str(path))


def _same_file_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _validate_sqlite_artifact(path: Path, file_stat: os.stat_result) -> None:
    if not stat.S_ISREG(file_stat.st_mode):
        raise _sqlite_artifact_error(path, "not a regular file")
    if file_stat.st_nlink != 1:
        raise _sqlite_artifact_error(path, "link count is not one")


def _require_sqlite_artifact_absent(path: Path, *, directory_fd: int) -> None:
    """Accept a vanished sidecar only while its directory entry stays absent."""
    try:
        current = os.stat(path.name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    _validate_sqlite_artifact(path, current)
    raise _sqlite_artifact_error(path, "directory entry changed while opening")


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


def _chmod_sqlite_artifact_at(
    path: Path,
    *,
    directory_fd: int,
    create: bool,
    allow_sidecar_disappearance: bool = False,
) -> bool:
    expected: os.stat_result | None
    try:
        expected = os.stat(path.name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        if not create:
            return False
        expected = None
    if expected is not None:
        _validate_sqlite_artifact(path, expected)

    flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
    if expected is None:
        flags |= os.O_CREAT | os.O_EXCL
    try:
        fd = os.open(path.name, flags, 0o600, dir_fd=directory_fd)
    except FileExistsError:
        if expected is not None:
            raise
        return _chmod_sqlite_artifact_at(
            path,
            directory_fd=directory_fd,
            create=False,
        )
    except FileNotFoundError:
        if not allow_sidecar_disappearance:
            raise
        _require_sqlite_artifact_absent(path, directory_fd=directory_fd)
        return False
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode):
            raise _sqlite_artifact_error(path, "not a regular file")
        if expected is not None and not _same_file_identity(expected, opened):
            raise _sqlite_artifact_error(path, "directory entry changed while opening")
        if opened.st_nlink == 0 and allow_sidecar_disappearance:
            _require_sqlite_artifact_absent(path, directory_fd=directory_fd)
            return False
        _validate_sqlite_artifact(path, opened)
        os.fchmod(fd, 0o600)
        restricted = os.fstat(fd)
        if restricted.st_nlink == 0 and allow_sidecar_disappearance:
            _require_sqlite_artifact_absent(path, directory_fd=directory_fd)
            return False
        if restricted.st_nlink != 1:
            raise _sqlite_artifact_error(path, "link count changed while restricting permissions")
    finally:
        os.close(fd)
    return True


def _restrict_existing_sqlite_artifacts(db_path: Path) -> None:
    """Restrict verified, single-link SQLite files without following links."""
    if os.name != "posix":  # pragma: no cover - Windows compatibility fallback
        for artifact in (
            db_path,
            *(db_path.with_name(db_path.name + suffix) for suffix in _SQLITE_SIDECAR_SUFFIXES),
        ):
            try:
                artifact.chmod(0o600)
            except FileNotFoundError:
                continue
        return

    directory_fd = _open_private_sqlite_directory(db_path)
    try:
        _chmod_sqlite_artifact_at(
            db_path,
            directory_fd=directory_fd,
            create=False,
        )
        for suffix in _SQLITE_SIDECAR_SUFFIXES:
            _chmod_sqlite_artifact_at(
                db_path.with_name(db_path.name + suffix),
                directory_fd=directory_fd,
                create=False,
                allow_sidecar_disappearance=True,
            )
    finally:
        os.close(directory_fd)


def _prepare_private_sqlite_file(path: Path) -> None:
    """Create or tighten one SQLite file and its existing sidecars safely."""
    if os.name != "posix":  # pragma: no cover - Windows compatibility fallback
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
        fd = os.open(path, flags, 0o600)
        try:
            if hasattr(os, "fchmod"):
                os.fchmod(fd, 0o600)
            else:
                path.chmod(0o600)
        finally:
            os.close(fd)
        _restrict_existing_sqlite_artifacts(path)
        return

    directory_fd = _open_private_sqlite_directory(path)
    try:
        _chmod_sqlite_artifact_at(path, directory_fd=directory_fd, create=True)
        for suffix in _SQLITE_SIDECAR_SUFFIXES:
            _chmod_sqlite_artifact_at(
                path.with_name(path.name + suffix),
                directory_fd=directory_fd,
                create=False,
                allow_sidecar_disappearance=True,
            )
    finally:
        os.close(directory_fd)


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
