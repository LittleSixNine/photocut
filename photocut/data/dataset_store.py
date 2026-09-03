import fcntl
import hashlib
import json
import os
import platform
import re
import secrets
import shutil
import stat
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from photocut.config import ALGORITHM_VERSION, DATASET_SCHEMA_VERSION


ARCHIVE_SPACE_RESERVE_BYTES = 1024 * 1024
_BATCH_ID_PATTERN = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{8}$")
_LOCK_REGISTRY_GUARD = threading.RLock()
_LOCK_OWNERS: dict[Path, tuple[int, object]] = {}
_UNSPECIFIED_JSONL_IDENTITY = object()


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_directory(directory: Path) -> None:
    missing = []
    current = directory
    while not current.exists():
        missing.append(current)
        parent = current.parent
        if parent == current:
            break
        current = parent

    created = []
    for current in reversed(missing):
        try:
            current.mkdir()
        except FileExistsError:
            continue
        created.append(current)

    for current in created:
        _fsync_directory(current)
        _fsync_directory(current.parent)


def _close_descriptor_quietly(descriptor: int) -> None:
    try:
        os.close(descriptor)
    except OSError:
        pass


def _remove_file_durably(path: Path) -> None:
    path.unlink()
    _fsync_directory(path.parent)


def _lexical_absolute_path(path: Path) -> Path:
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            info = current.lstat()
        except FileNotFoundError:
            break
        if stat.S_ISLNK(info.st_mode):
            raise ValueError(f"dataset path contains symlink: {current}")
    return absolute


def atomic_write_json(path: Path, value: Any) -> None:
    path = Path(path)
    _ensure_directory(path.parent)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temp_path = Path(temp_name)
    try:
        try:
            stream = os.fdopen(descriptor, "w", encoding="utf-8")
        except BaseException:
            _close_descriptor_quietly(descriptor)
            raise
        with stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _jsonl_identity(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _jsonl_leaf_error(path: Path, cause: OSError | None = None) -> ValueError:
    try:
        leaf = path.lstat()
    except OSError:
        message = "JSONL path changed during access"
    else:
        if stat.S_ISLNK(leaf.st_mode):
            message = "JSONL file must not be a symlink"
        elif not stat.S_ISREG(leaf.st_mode):
            message = "JSONL file must be a regular file"
        elif leaf.st_nlink != 1:
            message = "JSONL file must not have hard link aliases"
        else:
            message = "JSONL path changed during access"
    error = ValueError(message)
    if cause is not None:
        error.__cause__ = cause
    return error


def _validate_jsonl_identity(path: Path, descriptor: int) -> tuple[int, int, int, int, int]:
    opened = os.fstat(descriptor)
    if not stat.S_ISREG(opened.st_mode):
        raise ValueError("JSONL file must be a regular file")
    if opened.st_nlink != 1:
        if opened.st_nlink == 0:
            raise ValueError("JSONL opened file is detached from its path")
        raise ValueError("JSONL file must not have hard link aliases")
    try:
        leaf = path.lstat()
    except OSError as exc:
        raise _jsonl_leaf_error(path, exc)
    if stat.S_ISLNK(leaf.st_mode):
        raise ValueError("JSONL file must not be a symlink")
    if not stat.S_ISREG(leaf.st_mode):
        raise ValueError("JSONL file must be a regular file")
    if leaf.st_nlink != 1:
        raise ValueError("JSONL file must not have hard link aliases")
    if _jsonl_identity(leaf) != _jsonl_identity(opened):
        raise ValueError("JSONL path identity changed during access")
    return _jsonl_identity(opened)


def _read_jsonl_bytes(descriptor: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks = []
    while chunk := os.read(descriptor, 1024 * 1024):
        chunks.append(chunk)
    return b"".join(chunks)


def _parse_jsonl_bytes(data: bytes, *, require_terminal_newline: bool) -> list[dict]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"invalid JSONL encoding: {exc}") from exc
    events = []
    for line_number, line in enumerate(text.split("\n"), 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at line {line_number}: {exc}") from exc
        if not isinstance(event, dict):
            raise ValueError(
                f"invalid JSONL event at line {line_number}: expected object"
            )
        events.append(event)
    if require_terminal_newline and data and not data.endswith(b"\n"):
        raise ValueError("non-empty JSONL file must end with a terminal newline")
    return events


def _open_jsonl_for_append(path: Path) -> tuple[int, bool]:
    flags = os.O_RDWR | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0)
    while True:
        created = False
        try:
            descriptor = os.open(path, flags)
        except FileNotFoundError:
            try:
                descriptor = os.open(path, flags | os.O_CREAT | os.O_EXCL, 0o666)
                created = True
            except FileExistsError:
                continue
        except OSError as exc:
            raise _jsonl_leaf_error(path, exc)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
        except BaseException:
            _close_descriptor_quietly(descriptor)
            raise
        if os.fstat(descriptor).st_nlink == 0:
            os.close(descriptor)
            continue
        try:
            _validate_jsonl_identity(path, descriptor)
        except BaseException:
            _close_descriptor_quietly(descriptor)
            raise
        return descriptor, created


def _safe_cleanup_created_jsonl(path: Path, descriptor: int) -> BaseException | None:
    try:
        opened = os.fstat(descriptor)
        current = path.lstat()
        if (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino):
            return RuntimeError("created JSONL path identity changed before cleanup")
    except BaseException as exc:
        return exc

    cleanup_path = path.parent / f".{path.name}.{secrets.token_hex(16)}.cleanup"
    try:
        os.replace(path, cleanup_path)
        cleanup = cleanup_path.lstat()
        if (cleanup.st_dev, cleanup.st_ino) != (opened.st_dev, opened.st_ino):
            if not os.path.lexists(path):
                os.replace(cleanup_path, path)
                _fsync_directory(path.parent)
            return RuntimeError("created JSONL cleanup identity changed")
        os.unlink(cleanup_path)
        _fsync_directory(path.parent)
    except BaseException as exc:
        return exc
    return None


def append_jsonl_validated(
    path: Path,
    event: dict,
    validate_existing: Callable[[list[dict]], None],
    *,
    expected_identity=_UNSPECIFIED_JSONL_IDENTITY,
) -> None:
    path = Path(path)
    payload = (
        json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    _ensure_directory(path.parent)
    descriptor = None
    created = False
    original_size = None
    wrote_payload = False
    try:
        descriptor, created = _open_jsonl_for_append(path)
        before = _validate_jsonl_identity(path, descriptor)
        if expected_identity is None:
            if not created:
                raise ValueError("JSONL path changed from expected absence")
        elif (
            expected_identity is not _UNSPECIFIED_JSONL_IDENTITY
            and before != expected_identity
        ):
            raise ValueError("JSONL path identity changed before append")
        original_size = before[2]
        events = _parse_jsonl_bytes(
            _read_jsonl_bytes(descriptor), require_terminal_newline=True
        )
        after_read = _validate_jsonl_identity(path, descriptor)
        if before != after_read:
            raise ValueError("JSONL file changed during locked read")
        validate_existing(events)
        if after_read != _validate_jsonl_identity(path, descriptor):
            raise ValueError("JSONL file changed before append")

        written = 0
        while written < len(payload):
            count = os.write(descriptor, payload[written:])
            if count <= 0:
                raise OSError("JSONL append write made no progress")
            written += count
            wrote_payload = True
        os.fsync(descriptor)
        after_write = _validate_jsonl_identity(path, descriptor)
        if (
            after_write[:2] != before[:2]
            or after_write[2] != original_size + len(payload)
        ):
            raise ValueError("JSONL file changed after append")
        _fsync_directory(path.parent)
    except BaseException as append_error:
        rollback_error = None
        cleanup_error = None
        if descriptor is not None and original_size is not None:
            try:
                os.ftruncate(descriptor, original_size)
                os.fsync(descriptor)
            except BaseException as exc:
                rollback_error = exc
            if created and original_size == 0:
                cleanup_error = _safe_cleanup_created_jsonl(path, descriptor)
        if rollback_error is not None:
            raise RuntimeError(
                "JSONL append consistency failure: "
                f"rollback failed: {rollback_error}"
            ) from append_error
        if cleanup_error is not None:
            raise RuntimeError(
                "JSONL append consistency failure: "
                f"cleanup could not prove path ownership: {cleanup_error}"
            ) from append_error
        cleaned_created_file = created and original_size == 0
        if wrote_payload and not cleaned_created_file:
            try:
                _validate_jsonl_identity(path, descriptor)
            except BaseException as identity_error:
                raise RuntimeError(
                    "JSONL append consistency failure: "
                    f"path changed after write: {identity_error}"
                ) from append_error
        raise
    finally:
        if descriptor is not None:
            _close_descriptor_quietly(descriptor)


def append_jsonl(
    path: Path,
    event: dict,
    *,
    expected_identity=_UNSPECIFIED_JSONL_IDENTITY,
) -> None:
    append_jsonl_validated(
        path,
        event,
        lambda _events: None,
        expected_identity=expected_identity,
    )


def load_jsonl(path: Path, *, require_terminal_newline: bool = False) -> list[dict]:
    path = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        pre_open = path.lstat()
        if stat.S_ISLNK(pre_open.st_mode):
            raise ValueError("JSONL file must not be a symlink")
        if not stat.S_ISREG(pre_open.st_mode):
            raise ValueError("JSONL file must be a regular file")
        if pre_open.st_nlink != 1:
            raise ValueError("JSONL file must not have hard link aliases")
    except FileNotFoundError:
        return []
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        raise ValueError("JSONL path changed during access")
    except OSError as exc:
        raise _jsonl_leaf_error(path, exc)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_SH)
        before = _validate_jsonl_identity(path, descriptor)
        if _jsonl_identity(pre_open) != before:
            raise ValueError("JSONL path identity changed during open")
        events = _parse_jsonl_bytes(
            _read_jsonl_bytes(descriptor),
            require_terminal_newline=require_terminal_newline,
        )
        after = _validate_jsonl_identity(path, descriptor)
        if before != after:
            raise ValueError("JSONL file changed during locked read")
        return events
    finally:
        _close_descriptor_quietly(descriptor)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_image_extension(path: Path) -> str:
    with Path(path).open("rb") as stream:
        prefix = stream.read(8)
    if prefix.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if prefix == b"\x89PNG\r\n\x1a\n":
        return ".png"
    raise ValueError(f"unsupported image bytes: {path}")


@dataclass(frozen=True)
class ArchivedImage:
    image_id: str
    sha256: str
    object_path: Path
    source_path: Path
    byte_size: int
    width: int | None
    height: int | None


@dataclass(frozen=True)
class BatchPaths:
    batch_id: str
    batch_dir: Path
    manifest: Path
    run_in_progress: Path
    production_run: Path
    annotations: Path
    crops: Path
    lock: Path


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _json_copy(value: Any, field_name: str) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be JSON serializable") from exc


def _validate_source_record(source_record: Any) -> None:
    if not isinstance(source_record, dict):
        raise ValueError("manifest contains an invalid source record")
    for field in ("source_id", "source_filename", "source_path"):
        value = source_record.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError("manifest contains an invalid source record")


def _read_json_regular_file(path: Path, field_name: str) -> Any:
    """Read one stable, non-symlink JSON file without following replacements."""
    path = Path(path)
    descriptor = None
    try:
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode):
            raise ValueError
        descriptor = os.open(
            path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        )
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
        ):
            raise ValueError
        with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
            descriptor = None
            value = json.load(stream)
        after = path.lstat()
        if (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino):
            raise ValueError
        return value
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid {field_name}") from exc
    finally:
        if descriptor is not None:
            _close_descriptor_quietly(descriptor)


def _git_output(arguments: Sequence[str]) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=Path(__file__).resolve().parent,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.decode("utf-8", errors="replace")


def _runtime_snapshot(runtime: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(runtime, Mapping):
        raise ValueError("runtime must be a mapping")
    parameters = runtime.get("parameters", {})
    if not isinstance(parameters, Mapping):
        raise ValueError("runtime parameters must be a mapping")

    try:
        commit = _git_output(["rev-parse", "HEAD"]).strip()
        dirty = bool(_git_output(["status", "--porcelain"]).strip())
        diff = _git_output(["diff", "--binary", "HEAD"]).encode("utf-8")
        git_metadata = {
            "available": True,
            "commit": commit,
            "dirty": dirty,
            "code_diff_sha256": hashlib.sha256(diff).hexdigest(),
        }
    except (OSError, subprocess.SubprocessError) as exc:
        git_metadata = {
            "available": False,
            "commit": None,
            "dirty": None,
            "code_diff_sha256": None,
            "fallback": f"git metadata unavailable: {type(exc).__name__}",
        }

    try:
        import cv2

        opencv_version = cv2.__version__
    except ImportError:
        opencv_version = "unavailable"

    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "captured_at": _timestamp(),
        "algorithm_version": runtime.get("algorithm_version", ALGORITHM_VERSION),
        "parameters": _json_copy(dict(parameters), "runtime parameters"),
        "git": git_metadata,
        "environment": {
            "python": platform.python_version(),
            "opencv": opencv_version,
            "numpy": np.__version__,
            "os": platform.platform(),
        },
    }


def _atomic_write_json_new(path: Path, value: Any) -> None:
    """Durably publish a JSON document only when its target does not exist."""
    path = Path(path)
    _ensure_directory(path.parent)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp_path, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    temp_path.unlink(missing_ok=True)


class BatchLock:
    """Process-safe lock with persistent, inspectable metadata."""

    def __init__(
        self,
        paths: BatchPaths | Path,
        *,
        retain_on_error: bool = True,
        reclaim_existing: bool = False,
    ):
        self.path = Path(paths.lock if isinstance(paths, BatchPaths) else paths)
        self.retain_on_error = retain_on_error
        self.reclaim_existing = reclaim_existing
        self._acquired = False
        self._descriptor: int | None = None
        self._directory_descriptor: int | None = None
        self._token: object | None = None
        self._directory_identity: tuple[int, int] | None = None
        self._lock_identity: tuple[int, int] | None = None

    @staticmethod
    def _registry_path(path: Path) -> Path:
        return path.absolute()

    @classmethod
    def require_owner(cls, path: Path) -> None:
        with _LOCK_REGISTRY_GUARD:
            owner = _LOCK_OWNERS.get(cls._registry_path(path))
            if owner is None or owner[0] != threading.get_ident():
                raise RuntimeError("BatchLock must be held by the current thread")
        lock_path = Path(path)
        try:
            directory = lock_path.parent.lstat()
            lock = lock_path.lstat()
        except OSError as exc:
            raise RuntimeError("batch lock ownership changed") from exc
        _, token = owner
        state = getattr(token, "state", None)
        if state is None or (directory.st_dev, directory.st_ino) != state[0] or (
            lock.st_dev,
            lock.st_ino,
        ) != state[1]:
            raise RuntimeError("batch lock ownership changed")

    @staticmethod
    def _write_metadata(descriptor: int) -> None:
        payload = json.dumps(
            {"pid": os.getpid(), "created_at": _timestamp()}, ensure_ascii=False
        ).encode("utf-8") + b"\n"
        os.ftruncate(descriptor, 0)
        os.lseek(descriptor, 0, os.SEEK_SET)
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("failed to write batch lock")
            offset += written
        os.fsync(descriptor)

    def _reclaim_existing_lock(self) -> None:
        """Remove a valid stale lock while this process exclusively locks its directory."""
        try:
            before = self.path.lstat()
            if (
                not stat.S_ISREG(before.st_mode)
                or stat.S_ISLNK(before.st_mode)
                or before.st_nlink != 1
            ):
                raise ValueError
            descriptor = os.open(
                self.path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            )
            try:
                opened = os.fstat(descriptor)
                if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                    raise ValueError
                with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
                    descriptor = None
                    metadata = json.load(stream)
                after = self.path.lstat()
                if (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino):
                    raise ValueError
            finally:
                if descriptor is not None:
                    _close_descriptor_quietly(descriptor)
            if (
                not isinstance(metadata, dict)
                or set(metadata) != {"pid", "created_at"}
                or type(metadata["pid"]) is not int
                or metadata["pid"] <= 0
                or not isinstance(metadata["created_at"], str)
                or not metadata["created_at"]
            ):
                raise ValueError
            datetime.fromisoformat(metadata["created_at"])
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"cannot safely reclaim batch lock: {self.path}") from exc
        cleanup = self.path.parent / f".{self.path.name}.{secrets.token_hex(16)}.reclaim"
        os.replace(self.path, cleanup)
        moved = cleanup.lstat()
        if (moved.st_dev, moved.st_ino) != (before.st_dev, before.st_ino):
            raise RuntimeError("batch lock reclaim ownership changed")
        cleanup.unlink()
        _fsync_directory(self.path.parent)

    def __enter__(self) -> "BatchLock":
        _ensure_directory(self.path.parent)
        registry_path = self._registry_path(self.path)
        with _LOCK_REGISTRY_GUARD:
            if registry_path in _LOCK_OWNERS:
                raise RuntimeError(f"batch already locked: {self.path}")
        try:
            directory_descriptor = os.open(
                self.path.parent,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
            )
            directory_stat = os.fstat(directory_descriptor)
            if not stat.S_ISDIR(directory_stat.st_mode):
                raise RuntimeError(f"invalid batch directory: {self.path.parent}")
            try:
                fcntl.flock(directory_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f"batch already locked: {self.path}") from exc
            try:
                descriptor = os.open(
                    self.path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
                    0o600,
                )
            except FileExistsError as exc:
                if not self.reclaim_existing:
                    raise RuntimeError(f"batch already locked: {self.path}") from exc
                self._reclaim_existing_lock()
                descriptor = os.open(
                    self.path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
                    0o600,
                )
            self._write_metadata(descriptor)
            _fsync_directory(self.path.parent)
        except BaseException:
            if "descriptor" in locals():
                _close_descriptor_quietly(descriptor)
            if "directory_descriptor" in locals():
                _close_descriptor_quietly(directory_descriptor)
            raise

        lock_stat = os.fstat(descriptor)
        token = type("LockToken", (), {})()
        token.state = (
            (directory_stat.st_dev, directory_stat.st_ino),
            (lock_stat.st_dev, lock_stat.st_ino),
        )
        with _LOCK_REGISTRY_GUARD:
            if registry_path in _LOCK_OWNERS:
                _close_descriptor_quietly(descriptor)
                raise RuntimeError(f"batch already locked: {self.path}")
            _LOCK_OWNERS[registry_path] = (threading.get_ident(), token)
        self._descriptor = descriptor
        self._directory_descriptor = directory_descriptor
        self._token = token
        self._directory_identity, self._lock_identity = token.state
        self._acquired = True
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        if not self._acquired or self._descriptor is None or self._directory_descriptor is None:
            return False
        cleanup_error = None
        if exc_type is None or not self.retain_on_error:
            try:
                current = self.path.lstat()
                if (current.st_dev, current.st_ino) != self._lock_identity:
                    raise RuntimeError("batch lock ownership changed")
                cleanup = self.path.parent / f".{self.path.name}.{secrets.token_hex(16)}.cleanup"
                os.replace(self.path, cleanup)
                moved = cleanup.lstat()
                if (moved.st_dev, moved.st_ino) != self._lock_identity:
                    raise RuntimeError("batch lock cleanup ownership changed")
                cleanup.unlink()
                _fsync_directory(self.path.parent)
            except BaseException as exc:
                cleanup_error = exc
        with _LOCK_REGISTRY_GUARD:
            registry_path = self._registry_path(self.path)
            if _LOCK_OWNERS.get(registry_path) == (threading.get_ident(), self._token):
                del _LOCK_OWNERS[registry_path]
        try:
            fcntl.flock(self._directory_descriptor, fcntl.LOCK_UN)
        finally:
            _close_descriptor_quietly(self._descriptor)
            _close_descriptor_quietly(self._directory_descriptor)
        self._acquired = False
        self._descriptor = None
        self._directory_descriptor = None
        self._token = None
        if cleanup_error is not None:
            raise cleanup_error
        return False


class DatasetStore:
    def __init__(self, root: Path):
        self.root = _lexical_absolute_path(Path(root))
        self.objects_dir = self.root / "objects"
        self.batches_dir = self.root / "batches"
        self.experiments_dir = self.root / "experiments"

    def start_batch(self, runtime: Mapping[str, Any]) -> BatchPaths:
        self._prepare_directory(self.batches_dir)
        self._validate_store_directories()
        runtime_snapshot = _runtime_snapshot(runtime)
        for _ in range(128):
            batch_id = (
                f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-"
                f"{secrets.token_hex(4)}"
            )
            batch_dir = self.batches_dir / batch_id
            staging_dir = self.batches_dir / f".{batch_id}.{secrets.token_hex(4)}.staging"
            try:
                staging_dir.mkdir()
            except FileExistsError:
                continue
            _fsync_directory(self.batches_dir)
            staging_paths = self._paths_for_batch(batch_id, staging_dir)
            created_at = _timestamp()
            try:
                atomic_write_json(
                    staging_paths.manifest,
                    {
                        "schema_version": DATASET_SCHEMA_VERSION,
                        "batch_id": batch_id,
                        "created_at": created_at,
                        "images": [],
                    },
                )
                atomic_write_json(
                    staging_paths.run_in_progress,
                    {
                        "schema_version": DATASET_SCHEMA_VERSION,
                        "run_id": f"run_{batch_id}",
                        "batch_id": batch_id,
                        "status": "in_progress",
                        "started_at": created_at,
                        "runtime": runtime_snapshot,
                        "images": [],
                    },
                )
                if os.path.lexists(batch_dir):
                    shutil.rmtree(staging_dir)
                    _fsync_directory(self.batches_dir)
                    continue
                os.replace(staging_dir, batch_dir)
                _fsync_directory(self.batches_dir)
            except BaseException:
                try:
                    shutil.rmtree(staging_dir)
                    _fsync_directory(self.batches_dir)
                except OSError:
                    pass
                raise
            return self._paths_for_batch(batch_id, batch_dir)
        raise RuntimeError("could not allocate a unique batch id")

    @staticmethod
    def _paths_for_batch(batch_id: str, batch_dir: Path) -> BatchPaths:
        return BatchPaths(
                batch_id=batch_id,
                batch_dir=batch_dir,
                manifest=batch_dir / "manifest.json",
                run_in_progress=batch_dir / "production_run.inprogress.json",
                production_run=batch_dir / "production_run.json",
                annotations=batch_dir / "annotations.jsonl",
                crops=batch_dir / "crops.jsonl",
                lock=batch_dir / "batch.lock",
            )

    def _validate_batch_paths(self, paths: BatchPaths) -> None:
        if not isinstance(paths, BatchPaths):
            raise ValueError("invalid batch paths")
        if not _BATCH_ID_PATTERN.fullmatch(paths.batch_id):
            raise ValueError("invalid batch paths")
        expected = self._paths_for_batch(paths.batch_id, self.batches_dir / paths.batch_id)
        self._validate_store_directories()
        fields = (
            "batch_dir",
            "manifest",
            "run_in_progress",
            "production_run",
            "annotations",
            "crops",
            "lock",
        )
        if any(getattr(paths, field).absolute() != getattr(expected, field).absolute() for field in fields):
            raise ValueError("invalid batch paths")
        try:
            batch_stat = paths.batch_dir.lstat()
        except OSError as exc:
            raise ValueError("invalid batch paths") from exc
        if not stat.S_ISDIR(batch_stat.st_mode) or stat.S_ISLNK(batch_stat.st_mode):
            raise ValueError("invalid batch paths")
        for field in fields[1:]:
            path = getattr(paths, field)
            if not os.path.lexists(path):
                continue
            try:
                mode = path.lstat().st_mode
            except OSError as exc:
                raise ValueError("invalid batch paths") from exc
            if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
                raise ValueError("invalid batch paths")

    def batch_paths_from_candidate(self, batch_dir: Path) -> BatchPaths | None:
        """Return validated paths for a published batch, skipping unrelated entries."""
        batch_dir = Path(batch_dir)
        if batch_dir.name.startswith(".") or not _BATCH_ID_PATTERN.fullmatch(
            batch_dir.name
        ):
            return None
        expected = self.batches_dir / batch_dir.name
        if batch_dir.absolute() != expected.absolute():
            raise ValueError("invalid batch paths")
        paths = self._paths_for_batch(batch_dir.name, expected)
        self._validate_batch_paths(paths)
        return paths

    def validate_batch_paths(self, paths: BatchPaths) -> None:
        self._validate_batch_paths(paths)

    def load_in_progress_run(self, paths: BatchPaths) -> dict[str, Any]:
        self._validate_batch_paths(paths)
        run = _read_json_regular_file(paths.run_in_progress, "in-progress run")
        self._validate_batch_paths(paths)
        if not isinstance(run, dict):
            raise ValueError("invalid in-progress run")
        return run

    def load_finalized_run(self, paths: BatchPaths) -> dict[str, Any]:
        self._validate_batch_paths(paths)
        self._require_lock(paths)
        final = self._load_finalized_run(paths)
        self._validate_batch_paths(paths)
        self._require_lock(paths)
        return final

    def _validate_store_directories(self) -> None:
        _lexical_absolute_path(self.root)
        _lexical_absolute_path(self.batches_dir)
        for directory in (self.root, self.batches_dir):
            try:
                info = directory.lstat()
            except OSError as exc:
                raise ValueError("invalid dataset directory") from exc
            if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
                raise ValueError("invalid dataset directory")

    @staticmethod
    def _prepare_directory(directory: Path) -> None:
        _lexical_absolute_path(directory)
        _ensure_directory(directory)
        _lexical_absolute_path(directory)

    def _validate_object_directory(self, directory: Path) -> None:
        _lexical_absolute_path(directory)
        try:
            root_info = self.root.lstat()
            objects_info = self.objects_dir.lstat()
            leaf_info = directory.lstat()
        except OSError as exc:
            raise IOError("invalid dataset object directory") from exc
        for info in (root_info, objects_info, leaf_info):
            if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
                raise IOError("invalid dataset object directory")

    @staticmethod
    def _require_lock(paths: BatchPaths) -> None:
        BatchLock.require_owner(paths.lock)

    def register_archived_image(
        self,
        paths: BatchPaths,
        archived_image: ArchivedImage,
        source_id: str | None = None,
    ) -> None:
        self._validate_batch_paths(paths)
        self._require_lock(paths)
        if not isinstance(archived_image, ArchivedImage):
            raise TypeError("archived_image must be ArchivedImage")
        if (
            not isinstance(archived_image.sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", archived_image.sha256)
            or archived_image.image_id != f"sha256:{archived_image.sha256}"
        ):
            raise ValueError("invalid archived image identity")
        candidate = Path(archived_image.object_path)
        if candidate.suffix not in (".jpg", ".png"):
            raise ValueError("invalid archived object path")
        expected_object = (
            self.objects_dir
            / archived_image.sha256[:2]
            / f"{archived_image.sha256}{candidate.suffix}"
        )
        if candidate.absolute() != expected_object.absolute():
            raise ValueError("archived object path is outside dataset root")
        try:
            self._validate_object_directory(expected_object.parent)
            verified_size = self._verify_object(expected_object, archived_image.sha256)
        except IOError as exc:
            raise ValueError("invalid archived object") from exc
        if verified_size != archived_image.byte_size:
            raise ValueError("archived object byte size does not match")
        if canonical_image_extension(expected_object) != candidate.suffix:
            raise ValueError("archived object extension does not match bytes")
        object_relative = expected_object.relative_to(self.root)

        image_record = {
            "image_id": archived_image.image_id,
            "object_path": object_relative.as_posix(),
            "source_filename": archived_image.source_path.name,
            "source_path": str(archived_image.source_path.resolve()),
            "byte_size": archived_image.byte_size,
            "width": archived_image.width,
            "height": archived_image.height,
        }
        source_record = {
            "source_id": source_id,
            "source_filename": archived_image.source_path.name,
            "source_path": str(archived_image.source_path.resolve()),
        }
        if source_id is not None:
            _validate_source_record(source_record)
        manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
        images = manifest.get("images")
        if not isinstance(images, list):
            raise ValueError("manifest images must be a list")
        sources_by_id = {}
        for existing in images:
            if not isinstance(existing, dict):
                raise ValueError("manifest contains an invalid image record")
            if "sources" not in existing:
                continue
            existing_sources = existing["sources"]
            if not isinstance(existing_sources, list):
                raise ValueError("manifest image sources must be a list")
            for existing_source in existing_sources:
                _validate_source_record(existing_source)
                existing_source_id = existing_source["source_id"]
                if existing_source_id in sources_by_id:
                    raise ValueError(
                        f"duplicate source_id in manifest: {existing_source_id}"
                    )
                sources_by_id[existing_source_id] = (
                    existing.get("image_id"),
                    existing_source,
                )
        if source_id in sources_by_id:
            existing_image_id, existing_source = sources_by_id[source_id]
            if (
                existing_image_id != archived_image.image_id
                or existing_source != source_record
            ):
                raise ValueError(f"conflicting metadata for source {source_id}")

        for existing in images:
            if not isinstance(existing, dict):
                raise ValueError("manifest contains an invalid image record")
            if existing.get("image_id") != archived_image.image_id:
                continue
            comparable_keys = (
                "image_id", "object_path", "byte_size", "width", "height"
            )
            if any(existing.get(key) != image_record[key] for key in comparable_keys):
                raise ValueError(
                    f"conflicting metadata for archived image {archived_image.image_id}"
                )
            if source_id is None:
                if all(existing.get(key) == image_record[key] for key in image_record):
                    return
                raise ValueError(
                    f"conflicting metadata for archived image {archived_image.image_id}"
                )
            sources = existing.get("sources")
            if not isinstance(sources, list):
                raise ValueError("manifest image sources must be a list")
            for existing_source in sources:
                _validate_source_record(existing_source)
                if existing_source.get("source_id") != source_id:
                    continue
                if existing_source == source_record:
                    return
                raise ValueError(f"conflicting metadata for source {source_id}")
            sources.append(source_record)
            atomic_write_json(paths.manifest, manifest)
            return

        image_record["imported_at"] = _timestamp()
        if source_id is not None:
            image_record["sources"] = [source_record]
        images.append(image_record)
        atomic_write_json(paths.manifest, manifest)

    def update_run(self, paths: BatchPaths, image_result: Mapping[str, Any]) -> None:
        self._validate_batch_paths(paths)
        self._require_lock(paths)
        if paths.production_run.exists():
            raise RuntimeError("production run already finalized")
        if not paths.run_in_progress.exists():
            raise RuntimeError("production run is not in progress")
        if not isinstance(image_result, Mapping):
            raise ValueError("image result must be a mapping")
        result = _json_copy(dict(image_result), "image result")
        image_id = result.get("image_id")
        if not isinstance(image_id, str) or not image_id:
            raise ValueError("image result must include image_id")

        run = json.loads(paths.run_in_progress.read_text(encoding="utf-8"))
        if run.get("status") != "in_progress":
            raise RuntimeError("production run is not in progress")
        images = run.get("images")
        if not isinstance(images, list):
            raise ValueError("production run images must be a list")
        source_id = result.get("source_id")
        if source_id is not None and (not isinstance(source_id, str) or not source_id):
            raise ValueError("source_id must be a non-empty string")
        identity_field = "source_id" if source_id is not None else "image_id"
        identity = source_id if source_id is not None else image_id
        if any(
            isinstance(entry, dict) and entry.get(identity_field) == identity
            for entry in images
        ):
            raise ValueError(
                f"duplicate {identity_field} in production run: {identity}"
            )
        images.append(result)
        atomic_write_json(paths.run_in_progress, run)

    def finalize_run(self, paths: BatchPaths) -> None:
        self._validate_batch_paths(paths)
        self._require_lock(paths)
        if paths.production_run.exists():
            if not paths.run_in_progress.exists():
                final = self._load_finalized_run(paths)
                return
            final = self._load_finalized_run(paths)
            try:
                in_progress = json.loads(paths.run_in_progress.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError("invalid in-progress run") from exc
            if not self._final_matches_in_progress(final, in_progress):
                raise RuntimeError("existing finalized production run conflicts with in-progress run")
            _remove_file_durably(paths.run_in_progress)
            return
        if not paths.run_in_progress.exists():
            raise RuntimeError("production run is not in progress")

        run = json.loads(paths.run_in_progress.read_text(encoding="utf-8"))
        if run.get("status") != "in_progress":
            raise RuntimeError("production run is not in progress")
        run["status"] = "complete"
        run["completed_at"] = _timestamp()
        try:
            _atomic_write_json_new(paths.production_run, run)
        except FileExistsError as exc:
            final = self._load_finalized_run(paths)
            try:
                in_progress = json.loads(paths.run_in_progress.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as read_exc:
                raise ValueError("invalid in-progress run") from read_exc
            if not self._final_matches_in_progress(final, in_progress):
                raise RuntimeError(
                    "existing finalized production run conflicts with in-progress run"
                ) from exc
        _remove_file_durably(paths.run_in_progress)

    @staticmethod
    def _final_matches_in_progress(
        final: Mapping[str, Any], in_progress: Mapping[str, Any]
    ) -> bool:
        if final.get("status") != "complete" or in_progress.get("status") != "in_progress":
            return False
        normalized_final = dict(final)
        normalized_progress = dict(in_progress)
        normalized_final.pop("status", None)
        normalized_final.pop("completed_at", None)
        normalized_progress.pop("status", None)
        normalized_progress.pop("completed_at", None)
        return normalized_final == normalized_progress

    @staticmethod
    def _load_finalized_run(paths: BatchPaths) -> dict[str, Any]:
        final = _read_json_regular_file(paths.production_run, "finalized run")
        if (
            not isinstance(final, dict)
            or final.get("schema_version") != DATASET_SCHEMA_VERSION
            or final.get("batch_id") != paths.batch_id
            or final.get("run_id") != f"run_{paths.batch_id}"
            or final.get("status") != "complete"
            or not isinstance(final.get("runtime"), dict)
            or not isinstance(final.get("images"), list)
        ):
            raise ValueError("invalid finalized run")
        completed_at = final.get("completed_at")
        if not isinstance(completed_at, str) or not completed_at:
            raise ValueError("invalid finalized run")
        try:
            datetime.fromisoformat(completed_at)
        except ValueError as exc:
            raise ValueError("invalid finalized run") from exc
        return final

    def preflight_sources(self, sources: Sequence[Path]) -> None:
        source_paths = [Path(source) for source in sources]
        if not source_paths:
            return
        self._prepare_directory(self.root)
        required = 0
        seen = set()
        for source in source_paths:
            digest = sha256_file(source)
            extension = canonical_image_extension(source)
            object_path = self.objects_dir / digest[:2] / f"{digest}{extension}"
            if os.path.lexists(object_path):
                self._validate_object_directory(object_path.parent)
                self._verify_object(object_path, digest)
            elif digest not in seen:
                required += source.stat().st_size
            seen.add(digest)
        if not required:
            return
        reserve = ARCHIVE_SPACE_RESERVE_BYTES
        required_with_reserve = required + reserve
        if shutil.disk_usage(self.root).free <= required_with_reserve:
            raise OSError(
                "insufficient disk space: need "
                f"{required_with_reserve} bytes for dataset archive "
                f"({required} bytes plus {reserve} byte reserve)"
            )

    def archive_image(
        self, source: Path, image_size: tuple[int, int] | None
    ) -> ArchivedImage:
        source = Path(source).resolve()
        if image_size is None:
            width = height = None
        else:
            width, height = image_size
            if width <= 0 or height <= 0:
                raise ValueError("image_size must be positive")

        source_hash = sha256_file(source)
        extension = canonical_image_extension(source)
        object_dir = self.objects_dir / source_hash[:2]
        object_path = object_dir / f"{source_hash}{extension}"
        self._prepare_directory(object_dir)
        self._validate_object_directory(object_dir)

        if os.path.lexists(object_path):
            object_size = self._verify_object(object_path, source_hash)
        else:
            object_size = self._publish_object(source, source_hash, object_path)

        return ArchivedImage(
            image_id=f"sha256:{source_hash}",
            sha256=source_hash,
            object_path=object_path,
            source_path=source,
            byte_size=object_size,
            width=width,
            height=height,
        )

    @staticmethod
    def _verify_object(object_path: Path, expected_hash: str) -> int:
        try:
            lstat = object_path.lstat()
        except OSError as exc:
            raise IOError(f"invalid archived object {object_path}") from exc
        if not stat.S_ISREG(lstat.st_mode) or lstat.st_mode & 0o222:
            raise IOError(f"invalid archived object {object_path}")

        descriptor = None
        try:
            descriptor = os.open(
                object_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            )
            file_stat = os.fstat(descriptor)
            if (
                not stat.S_ISREG(file_stat.st_mode)
                or file_stat.st_mode & 0o222
                or (file_stat.st_dev, file_stat.st_ino) != (lstat.st_dev, lstat.st_ino)
            ):
                raise ValueError
            digest = hashlib.sha256()
            while chunk := os.read(descriptor, 1024 * 1024):
                digest.update(chunk)
            post_lstat = object_path.lstat()
            if (
                not stat.S_ISREG(post_lstat.st_mode)
                or post_lstat.st_mode & 0o222
                or post_lstat.st_size != file_stat.st_size
                or (post_lstat.st_dev, post_lstat.st_ino)
                != (file_stat.st_dev, file_stat.st_ino)
            ):
                raise ValueError
        except OSError as exc:
            raise IOError(f"invalid archived object {object_path}") from exc
        except ValueError as exc:
            raise IOError(f"invalid archived object {object_path}") from exc
        finally:
            if descriptor is not None:
                _close_descriptor_quietly(descriptor)
        if digest.hexdigest() != expected_hash:
            raise IOError(f"hash verification failed for archived object {object_path}")
        return file_stat.st_size

    def _publish_object(self, source: Path, source_hash: str, object_path: Path) -> int:
        descriptor, temp_name = tempfile.mkstemp(
            prefix=f".{source_hash}.", suffix=".tmp", dir=object_path.parent
        )
        temp_path = Path(temp_name)
        try:
            _close_descriptor_quietly(descriptor)
            shutil.copyfile(source, temp_path)
            if sha256_file(temp_path) != source_hash:
                raise IOError(f"hash verification failed for {source}")
            os.chmod(temp_path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
            with temp_path.open("rb") as stream:
                os.fsync(stream.fileno())
            try:
                os.link(temp_path, object_path)
            except FileExistsError:
                pass
            object_size = self._verify_object(object_path, source_hash)
            _fsync_directory(object_path.parent)
        except BaseException:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise
        temp_path.unlink(missing_ok=True)
        return object_size
