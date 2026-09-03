"""Thread-safe, single-owner image feature cache for v7 providers."""
from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from types import MappingProxyType
from typing import Any, Mapping

import cv2
import numpy as np


class FeatureCancelled(RuntimeError):
    """Raised before or during feature construction when cancellation is set."""


CancellationError = FeatureCancelled


def _is_cancelled(token: Any) -> bool:
    if token is None:
        return False
    value = getattr(token, "is_cancelled", None)
    if callable(value):
        try:
            return bool(value())
        except TypeError:
            pass
    elif value is not None:
        return bool(value)
    value = getattr(token, "cancelled", False)
    return bool(value() if callable(value) else value)


def _parameter_hash(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if hasattr(value, "sha256") and callable(value.sha256):
        try:
            return str(value.sha256())
        except TypeError:
            pass
    if hasattr(value, "to_dict") and callable(value.to_dict):
        value = value.to_dict()
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    except Exception:
        encoded = repr(value)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _immutable_array(array: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(array)
    return np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(contiguous.shape)


class ImageFeatureContext:
    """Own an immutable normalized BGR image and lazily construct its features.

    ``scale`` can be an exact target long-edge in pixels (integer), a relative
    fraction in ``(0, 1]`` or ``None`` for original resolution.  The exact target
    edge, not a floating point scale token, participates in the cache key.
    """

    def __init__(self, normalized_bgr: np.ndarray | Any, cancellation_token: Any = None,
                 *, cancel_token: Any = None):
        if hasattr(normalized_bgr, "normalized_bgr"):
            normalized_bgr = normalized_bgr.normalized_bgr
        array = np.asarray(normalized_bgr)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError("normalized_bgr must be an HxWx3 array")
        if array.dtype != np.uint8:
            raise ValueError("normalized_bgr must use uint8 pixels")
        if array.shape[0] <= 0 or array.shape[1] <= 0:
            raise ValueError("normalized_bgr must be non-empty")
        owned = _immutable_array(array)
        self._bgr_source = owned
        self._token = cancellation_token if cancellation_token is not None else cancel_token
        self._cache: dict[tuple[str, int | None, str], Any] = {}
        self._lock = threading.RLock()
        self._closed = False
        self.build_counts: dict[str, int] = {}
        self.pixels_visited: int = 0
        self.timings_ms: dict[str, float] = {}

    @classmethod
    def from_loaded(cls, loaded: Any, cancellation_token: Any = None, *, cancel_token: Any = None) -> "ImageFeatureContext":
        image = getattr(loaded, "normalized_bgr", loaded)
        return cls(image, cancellation_token, cancel_token=cancel_token)

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def shape(self) -> tuple[int, int, int]:
        self._ensure_open()
        return self._bgr_source.shape

    @property
    def image(self) -> np.ndarray:
        return self.bgr()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("ImageFeatureContext is closed")

    def _check_cancel(self) -> None:
        if _is_cancelled(self._token):
            raise FeatureCancelled("feature construction cancelled")

    def _target_edge(self, scale: Any) -> int | None:
        self._ensure_open()
        source_edge = max(self._bgr_source.shape[:2])
        if scale is None:
            return source_edge
        if isinstance(scale, bool) or not isinstance(scale, (int, float, np.integer, np.floating)):
            raise ValueError("scale must be None, a positive edge, or a fraction")
        value = float(scale)
        if not math.isfinite(value) or value <= 0:
            raise ValueError("scale must be positive and finite")
        # Integer scales and the explicit ``target_edge`` keyword are absolute
        # edge lengths; only fractional floats request proportional scaling.
        if isinstance(scale, (int, np.integer)) and not isinstance(scale, bool):
            return int(scale)
        if value <= 1.0:
            edge = int(round(source_edge * value))
        else:
            edge = int(round(value))
        return max(1, edge)

    def _cache_get(self, name: str, scale: Any, parameter_hash: Any, builder):
        self._ensure_open()
        target = self._target_edge(scale)
        key = (name, target, _parameter_hash(parameter_hash))
        with self._lock:
            self._ensure_open(); self._check_cancel()
            existing = self._cache.get(key)
            if existing is not None:
                return existing
            started = time.perf_counter()
            self._check_cancel()
            value = builder(target, key[2])
            self._check_cancel()
            if isinstance(value, np.ndarray):
                value = _immutable_array(value)
                self.pixels_visited += int(value.shape[0] * value.shape[1])
            self._cache[key] = value
            self.build_counts[name] = self.build_counts.get(name, 0) + 1
            elapsed = (time.perf_counter() - started) * 1000.0
            self.timings_ms[name] = self.timings_ms.get(name, 0.0) + elapsed
            return value

    def _bgr_for_target(self, target: int | None) -> np.ndarray:
        if target is None or target == max(self._bgr_source.shape[:2]):
            return self._bgr_source
        height, width = self._bgr_source.shape[:2]
        if width >= height:
            new_width, new_height = target, max(1, int(round(height * target / width)))
        else:
            new_height, new_width = target, max(1, int(round(width * target / height)))
        return cv2.resize(self._bgr_source, (new_width, new_height), interpolation=cv2.INTER_AREA)

    def bgr(self, scale: Any = None, parameter_hash: Any = None, *, params_hash: Any = None,
            target_edge: Any = None) -> np.ndarray:
        if target_edge is not None:
            if scale is not None: raise ValueError("specify scale or target_edge, not both")
            scale = target_edge
        parameter_hash = parameter_hash if params_hash is None else params_hash
        # Original source is also cached, so repeated providers cannot mutate or
        # accidentally trigger a second construction path.
        return self._cache_get("bgr", scale, parameter_hash, lambda target, _: self._bgr_for_target(target))

    def gray(self, scale: Any = None, parameter_hash: Any = None, *, params_hash: Any = None,
             target_edge: Any = None) -> np.ndarray:
        if target_edge is not None:
            if scale is not None: raise ValueError("specify scale or target_edge, not both")
            scale = target_edge
        parameter_hash = parameter_hash if params_hash is None else params_hash
        return self._cache_get("gray", scale, parameter_hash,
                              lambda target, _: cv2.cvtColor(self.bgr(target, parameter_hash), cv2.COLOR_BGR2GRAY))

    def lab(self, scale: Any = None, parameter_hash: Any = None, *, params_hash: Any = None,
            target_edge: Any = None) -> np.ndarray:
        if target_edge is not None:
            if scale is not None: raise ValueError("specify scale or target_edge, not both")
            scale = target_edge
        parameter_hash = parameter_hash if params_hash is None else params_hash
        return self._cache_get("lab", scale, parameter_hash,
                              lambda target, _: cv2.cvtColor(self.bgr(target, parameter_hash), cv2.COLOR_BGR2LAB))

    def gradient(self, scale: Any = None, parameter_hash: Any = None, *, params_hash: Any = None,
                 target_edge: Any = None) -> np.ndarray:
        if target_edge is not None:
            if scale is not None: raise ValueError("specify scale or target_edge, not both")
            scale = target_edge
        parameter_hash = parameter_hash if params_hash is None else params_hash
        def build(target, _):
            gray = self.gray(target, parameter_hash)
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            return cv2.magnitude(gx, gy)
        return self._cache_get("gradient", scale, parameter_hash, build)

    def edges(self, scale: Any = None, parameter_hash: Any = None, *, params_hash: Any = None,
              low_threshold: int = 50, high_threshold: int = 150,
              target_edge: Any = None) -> np.ndarray:
        if target_edge is not None:
            if scale is not None: raise ValueError("specify scale or target_edge, not both")
            scale = target_edge
        parameter_hash = parameter_hash if params_hash is None else params_hash
        edge_params = (_parameter_hash(parameter_hash), float(low_threshold), float(high_threshold))
        return self._cache_get("edges", scale, edge_params,
                              lambda target, _: cv2.Canny(self.gray(target, parameter_hash), low_threshold, high_threshold))

    def background_stats(self, scale: Any = None, parameter_hash: Any = None, *, params_hash: Any = None,
                         border_fraction: float = 0.05, target_edge: Any = None) -> Mapping[str, float]:
        if target_edge is not None:
            if scale is not None: raise ValueError("specify scale or target_edge, not both")
            scale = target_edge
        parameter_hash = parameter_hash if params_hash is None else params_hash
        if not isinstance(border_fraction, (int, float)) or not 0 < border_fraction < 0.5:
            raise ValueError("border_fraction must be between 0 and 0.5")
        key_params = (_parameter_hash(parameter_hash), float(border_fraction))
        def build(target, _):
            image = self.bgr(target, parameter_hash)
            h, w = image.shape[:2]
            by, bx = max(1, int(round(h * border_fraction))), max(1, int(round(w * border_fraction)))
            border = np.concatenate((image[:by].reshape(-1, 3), image[-by:].reshape(-1, 3),
                                     image[:, :bx].reshape(-1, 3), image[:, -bx:].reshape(-1, 3)), axis=0)
            return MappingProxyType({"mean": float(border.mean()), "std": float(border.std()),
                                     "pixels": int(border.shape[0])})
        return self._cache_get("background_stats", scale, key_params, build)

    def original_resolution_features(self, parameter_hash: Any = None) -> Mapping[str, Any]:
        """Build and share refinement features once for all Top-K candidates."""
        value = self._cache_get("refinement", None, parameter_hash,
                                lambda target, ph: self._refinement_feature_bundle(ph))
        return value

    def _refinement_feature_bundle(self, parameter_hash: Any) -> Mapping[str, Any]:
        """Build original-resolution refinement arrays and scalar thresholds once."""
        gray = self.gray(None, parameter_hash)
        gradient = self.gradient(None, parameter_hash)
        return MappingProxyType({
            "gray": gray,
            "gradient": gradient,
            "edges": self.edges(None, parameter_hash),
            "gradient_mean": float(np.mean(gradient)),
            "gradient_p70": float(np.percentile(gradient, 70)),
        })

    refinement_features = original_resolution_features

    def close(self) -> None:
        with self._lock:
            self._cache.clear()
            self._closed = True
            self._bgr_source = np.empty((0, 0, 3), dtype=np.uint8)
            self._bgr_source.setflags(write=False)
