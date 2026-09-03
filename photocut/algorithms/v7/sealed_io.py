"""Immutable, crash-durable JSON publication primitives."""
from __future__ import annotations
import errno, json, os, tempfile
from pathlib import Path
from typing import Any


_UNSUPPORTED_DIRECTORY_FSYNC = {errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP, errno.ENOSYS}


def _fsync_directory(path: Path) -> None:
    """Best-effort directory durability where the platform exposes it.

    Windows generally cannot open a directory as a file descriptor.  The file
    publication has already succeeded at this point, so only explicit
    unsupported-directory errors are ignored; permission and I/O failures on
    platforms that support directory fsync remain visible.
    """
    try:
        dfd = os.open(str(path), os.O_RDONLY)
    except OSError as exc:
        if os.name == "nt" or exc.errno in _UNSUPPORTED_DIRECTORY_FSYNC:
            return
        raise
    try:
        try:
            os.fsync(dfd)
        except OSError as exc:
            if os.name == "nt" or exc.errno in _UNSUPPORTED_DIRECTORY_FSYNC:
                return
            raise
    finally:
        os.close(dfd)

def write_json_new_fsync(path: str | os.PathLike[str], value: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()
    fd = None; tmp = None
    try:
        fd, tmp = tempfile.mkstemp(prefix=f".{p.name}.", dir=str(p.parent))
        try:
            view = memoryview(data)
            while view:
                n = os.write(fd, view)
                view = view[n:]
            os.fsync(fd)
        finally:
            os.close(fd); fd = None
        os.link(tmp, p)
        os.unlink(tmp); tmp = None
        _fsync_directory(p.parent)
    except Exception:
        if fd is not None:
            os.close(fd)
        # An interrupted publication must not leave a seemingly valid artifact.
        if tmp:
            try: os.unlink(tmp)
            except FileNotFoundError: pass
        raise
