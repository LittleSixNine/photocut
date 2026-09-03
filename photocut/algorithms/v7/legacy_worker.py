"""Bounded legacy v5.2 worker for the V7 auto cascade."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import multiprocessing as mp
import queue
import time
from typing import Any, Callable, Mapping

import numpy as np


@dataclass(frozen=True)
class SourceSnapshot:
    image: np.ndarray
    source_sha256: str
    original_size: tuple[int, int]
    normalized_size: tuple[int, int]
    orientation_transform: str = "identity"
    image_sha256: str | None = None

    def __post_init__(self) -> None:
        image = np.asarray(self.image)
        if image.ndim not in (2, 3):
            raise ValueError("snapshot image must be 2D or 3D")
        if not isinstance(self.source_sha256, str) or len(self.source_sha256) != 64:
            raise ValueError("snapshot source_sha256 must be a sha256 hex digest")
        image_sha256 = hashlib.sha256(image.tobytes()).hexdigest()
        if self.image_sha256 is not None and self.image_sha256 != image_sha256:
            raise ValueError("snapshot image hash mismatch")
        object.__setattr__(self, "image_sha256", image_sha256)
        object.__setattr__(self, "image", np.array(image, copy=True))
        self.image.setflags(write=False)


def _default_legacy(
    snapshot: SourceSnapshot,
    *,
    shrink_min: int = 25,
    shrink_max: int = 70,
    params: Any = None,
) -> Mapping[str, Any]:
    from photocut.core import detect_and_save_corners

    kwargs = {"img": np.asarray(snapshot.image)}
    if params is not None:
        kwargs["params"] = params
    return detect_and_save_corners(
        "<v7-source-snapshot>", "", int(shrink_min), int(shrink_max), **kwargs,
    )


def _worker_entry(
    out_queue: Any,
    snapshot: SourceSnapshot,
    detector: Callable[[SourceSnapshot], Any] | None,
    shrink_min: int,
    shrink_max: int,
    params: Any,
) -> None:
    try:
        if detector is None:
            result = _default_legacy(
                snapshot, shrink_min=shrink_min, shrink_max=shrink_max, params=params,
            )
        else:
            result = detector(snapshot)
        out_queue.put({
            "status": "ok",
            "source_sha256": snapshot.source_sha256,
            "image_sha256": snapshot.image_sha256,
            "original_size": snapshot.original_size,
            "normalized_size": snapshot.normalized_size,
            "orientation_transform": snapshot.orientation_transform,
            "result": result,
        })
    except BaseException as exc:
        out_queue.put({"status": "error", "error": f"{type(exc).__name__}: {exc}"})


def _cancelled(token: Any) -> bool:
    if token is None:
        return False
    for attr in ("is_cancelled", "cancelled", "is_set"):
        value = getattr(token, attr, None)
        if value is not None:
            try:
                return bool(value() if callable(value) else value)
            except Exception:
                return False
    return bool(token) if isinstance(token, bool) else False


def run_legacy_worker(
    snapshot: SourceSnapshot,
    *,
    timeout_s: float,
    detector: Callable[[SourceSnapshot], Any] | None = None,
    cancellation_token: Any = None,
    shrink_min: int = 25,
    shrink_max: int = 70,
    params: Any = None,
) -> dict[str, Any]:
    """Run legacy detection once, terminating the worker at timeout/cancel."""
    if timeout_s <= 0:
        return {"status": "timeout", "error": "legacy_timeout"}
    if _cancelled(cancellation_token):
        return {"status": "cancelled", "error": "cancelled"}
    methods = mp.get_all_start_methods()
    # OpenCV may hold native locks after V7 feature extraction; for the real
    # legacy path always use spawn so a child never inherits those locks.
    # Injected test doubles may use fork on POSIX because local callables are
    # not pickleable under spawn.
    if detector is None:
        context = mp.get_context("spawn")
    else:
        context = mp.get_context("fork" if "fork" in methods else "spawn")
    out_queue = context.Queue(maxsize=1)
    process = context.Process(
        target=_worker_entry,
        args=(out_queue, snapshot, detector, int(shrink_min), int(shrink_max), params),
        daemon=True,
    )
    process.start()
    deadline = time.monotonic() + float(timeout_s)
    payload: dict[str, Any] | None = None
    try:
        while time.monotonic() < deadline:
            if _cancelled(cancellation_token):
                process.terminate(); process.join(timeout=0.5)
                return {"status": "cancelled", "error": "cancelled"}
            try:
                payload = out_queue.get(timeout=min(0.02, max(0.001, deadline - time.monotonic())))
                break
            except queue.Empty:
                if not process.is_alive():
                    break
        if payload is None:
            process.terminate(); process.join(timeout=0.5)
            return {"status": "timeout", "error": "legacy_timeout"}
        if payload.get("status") != "ok":
            return payload
        if (payload.get("source_sha256") != snapshot.source_sha256 or
                payload.get("image_sha256") != snapshot.image_sha256 or
                tuple(payload.get("original_size", ())) != snapshot.original_size or
                tuple(payload.get("normalized_size", ())) != snapshot.normalized_size or
                payload.get("orientation_transform") != snapshot.orientation_transform):
            return {"status": "error", "error": "legacy_snapshot_identity_mismatch"}
        return payload
    finally:
        if process.is_alive():
            process.terminate()
        process.join(timeout=0.5)
        out_queue.close()


__all__ = ["SourceSnapshot", "run_legacy_worker"]
