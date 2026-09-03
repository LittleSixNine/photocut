"""Bounded, cancellable coordinator for v7 GUI detection requests."""
from __future__ import annotations

import inspect
import queue
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Mapping


@dataclass(frozen=True)
class RequestIdentity:
    request_id: str
    image_id: str
    orientation_transform: str
    algorithm_version: str
    parameter_sha256: str
    mode: str

    def __post_init__(self):
        for field in ("request_id", "image_id", "orientation_transform", "algorithm_version", "parameter_sha256", "mode"):
            if not isinstance(getattr(self, field), str) or not getattr(self, field):
                raise ValueError(f"{field} must be non-empty")
        if self.mode not in {"safe", "aggressive"}:
            raise ValueError("mode must be safe or aggressive")
        if not re.fullmatch(r"[0-9a-fA-F]{64}", self.parameter_sha256):
            raise ValueError("parameter_sha256 must be a SHA-256 digest")

    @classmethod
    def from_value(cls, value: Any) -> "RequestIdentity":
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            data = dict(value)
            aliases = {"detection_id": "request_id"}
            for source, target in aliases.items():
                if target not in data and source in data:
                    data[target] = data[source]
            return cls(**{field: data[field] for field in (
                "request_id", "image_id", "orientation_transform", "algorithm_version", "parameter_sha256", "mode")})
        raise TypeError("request identity must be RequestIdentity or mapping")

    def key(self) -> tuple[str, ...]:
        return (self.request_id, self.image_id, self.orientation_transform,
                self.algorithm_version, self.parameter_sha256, self.mode)


class CancellationToken:
    def __init__(self, deadline: float | None = None):
        self.deadline = deadline
        self._event = threading.Event()
        self._reason: str | None = None

    @property
    def is_cancelled(self) -> bool:
        if self._event.is_set():
            return True
        if self.deadline is not None and time.monotonic() >= self.deadline:
            self.cancel("deadline_exceeded")
            return True
        return False

    @property
    def cancellation_reason(self) -> str | None:
        return self._reason

    def cancel(self, reason: str = "cancelled") -> None:
        if self._reason is None:
            self._reason = str(reason)
        self._event.set()


@dataclass
class _Task:
    identity: RequestIdentity
    image: Any
    token: CancellationToken
    submitted_at: float
    deadline: float | None


class DetectionCoordinator:
    def __init__(self, detector: Callable[..., Any], *, on_result: Callable[..., Any] | None = None,
                 worker_count: int = 1, max_queue: int = 2, cache_size: int = 16,
                 task_deadline_s: float | None = None):
        if not callable(detector):
            raise TypeError("detector must be callable")
        if isinstance(worker_count, bool) or not isinstance(worker_count, int) or worker_count < 1:
            raise ValueError("worker_count must be positive")
        if isinstance(max_queue, bool) or not isinstance(max_queue, int) or max_queue < 1:
            raise ValueError("max_queue must be positive")
        if isinstance(cache_size, bool) or not isinstance(cache_size, int) or cache_size < 1:
            raise ValueError("cache_size must be positive")
        self.detector = detector
        self.on_result = on_result
        self.task_deadline_s = task_deadline_s
        self._queue: queue.Queue[_Task | None] = queue.Queue(maxsize=max_queue)
        self._cache: OrderedDict[tuple[str, ...], Any] = OrderedDict()
        self._tasks: dict[tuple[str, ...], _Task] = {}
        self._metrics: list[dict[str, Any]] = []
        self._lock = threading.RLock()
        self._current: RequestIdentity | None = None
        self._closed = False
        self._cache_limit = cache_size
        self._workers = [threading.Thread(target=self._worker, name=f"photocut-v7-{i}", daemon=True)
                         for i in range(worker_count)]
        for worker in self._workers:
            worker.start()

    @property
    def current_identity(self) -> RequestIdentity | None:
        with self._lock:
            return self._current

    @property
    def cache_size(self) -> int:
        with self._lock:
            return len(self._cache)

    def cached_result(self, identity: RequestIdentity | Mapping[str, Any]) -> Any:
        normalized = RequestIdentity.from_value(identity)
        with self._lock:
            value = self._cache.get(normalized.key())
            if normalized.key() in self._cache:
                self._cache.move_to_end(normalized.key())
            return value

    def set_current(self, identity: RequestIdentity | Mapping[str, Any] | None) -> None:
        with self._lock:
            self._current = None if identity is None else RequestIdentity.from_value(identity)

    def submit(self, identity: RequestIdentity | Mapping[str, Any], image: Any,
               *, deadline_s: float | None = None) -> bool:
        normalized = RequestIdentity.from_value(identity)
        now = time.monotonic()
        duration = self.task_deadline_s if deadline_s is None else deadline_s
        deadline = now + float(duration) if duration is not None else None
        task = _Task(normalized, image, CancellationToken(deadline), now, deadline)
        with self._lock:
            if self._closed:
                return False
            self._tasks[normalized.key()] = task
        try:
            self._queue.put_nowait(task)
            return True
        except queue.Full:
            with self._lock:
                self._tasks.pop(normalized.key(), None)
                self._metrics.append({"request_id": normalized.request_id, "cancellation_reason": "queue_full", "queue_rejected": True})
            return False

    def navigate(self, identity: RequestIdentity | Mapping[str, Any], image: Any, **kwargs: Any) -> bool:
        normalized = RequestIdentity.from_value(identity)
        with self._lock:
            previous = self._current
            self._current = normalized
            if previous is not None and previous.key() != normalized.key():
                old = self._tasks.get(previous.key())
                if old is not None:
                    old.token.cancel("superseded")
        return self.submit(normalized, image, **kwargs)

    def change_mode(self, identity: RequestIdentity | Mapping[str, Any], image: Any, **kwargs: Any) -> bool:
        return self.navigate(identity, image, **kwargs)

    def replace_batch(self, identity: RequestIdentity | Mapping[str, Any] | None = None) -> None:
        self.cancel("batch_replaced")
        self.set_current(identity)

    def cancel(self, reason: str = "cancelled", identity: RequestIdentity | Mapping[str, Any] | None = None) -> None:
        target = RequestIdentity.from_value(identity).key() if identity is not None else None
        with self._lock:
            for key, task in list(self._tasks.items()):
                if target is None or key == target:
                    task.token.cancel(reason)

    def _call_detector(self, task: _Task) -> Any:
        kwargs = {"cancellation_token": task.token, "deadline": task.deadline, "request_identity": task.identity}
        try:
            signature = inspect.signature(self.detector)
            accepts_kwargs = any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values())
            if not accepts_kwargs:
                kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}
        except (TypeError, ValueError):
            pass
        return self.detector(task.image, **kwargs)

    def _worker(self) -> None:
        while True:
            task = self._queue.get()
            if task is None:
                self._queue.task_done()
                return
            started = time.monotonic()
            metric = {"request_id": task.identity.request_id,
                      "queue_ms": max(0.0, (started - task.submitted_at) * 1000.0),
                      "worker_ms": 0.0, "gui_delivery_ms": 0.0,
                      "cancellation_reason": None, "cache_size": 0}
            try:
                if task.token.is_cancelled:
                    metric["cancellation_reason"] = task.token.cancellation_reason or "cancelled"
                    continue
                result = self._call_detector(task)
                metric["worker_ms"] = max(0.0, (time.monotonic() - started) * 1000.0)
                if task.token.is_cancelled:
                    metric["cancellation_reason"] = task.token.cancellation_reason or "cancelled"
                    continue
                self.accept_result(task.identity, result, metric=metric)
            except Exception as exc:
                metric["worker_ms"] = max(0.0, (time.monotonic() - started) * 1000.0)
                metric["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                with self._lock:
                    self._tasks.pop(task.identity.key(), None)
                    self._metrics.append(metric)
                self._queue.task_done()

    def accept_result(self, identity: RequestIdentity | Mapping[str, Any], result: Any,
                      *, metric: dict[str, Any] | None = None) -> bool:
        normalized = RequestIdentity.from_value(identity)
        started = time.monotonic()
        with self._lock:
            current = self._current
            if current is not None and normalized.key() == current.key():
                delivered = False
                if self.on_result is not None:
                    self.on_result(normalized, result)
                    delivered = True
                if metric is not None:
                    metric["gui_delivery_ms"] = max(0.0, (time.monotonic() - started) * 1000.0)
                return delivered or self.on_result is None
            self._cache[normalized.key()] = result
            self._cache.move_to_end(normalized.key())
            while len(self._cache) > self._cache_limit:
                self._cache.popitem(last=False)
            if metric is not None:
                metric["cache_size"] = len(self._cache)
            return False

    def drain_for_test(self, timeout_s: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._queue.unfinished_tasks == 0:
                return True
            time.sleep(0.001)
        return self._queue.unfinished_tasks == 0

    def metrics(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(item) for item in self._metrics]

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for task in self._tasks.values():
                task.token.cancel("coordinator_closed")
        for _ in self._workers:
            while True:
                try:
                    self._queue.put(None, timeout=0.1)
                    break
                except queue.Full:
                    if all(not worker.is_alive() for worker in self._workers):
                        break
        for worker in self._workers:
            worker.join(timeout=2.0)


__all__ = ["RequestIdentity", "CancellationToken", "DetectionCoordinator"]
