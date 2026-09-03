"""Small, read-only global mouse sampler used by the magnifier drag UI."""

from __future__ import annotations

import ctypes
import math
import sys
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PointerSample:
    x: float
    y: float
    left_down: bool


class PointerReadError(RuntimeError):
    """The operating system could not provide a trustworthy pointer sample."""


class _CGPoint(ctypes.Structure):
    _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]


class _CoreGraphicsNative:
    """ctypes bindings kept behind a tiny injectable interface for testing."""

    def __init__(self):
        self._graphics = ctypes.CDLL(
            "/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics"
        )
        self._foundation = ctypes.CDLL(
            "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
        )
        self._create_event = self._graphics.CGEventCreate
        self._create_event.argtypes = [ctypes.c_void_p]
        self._create_event.restype = ctypes.c_void_p
        self._get_location = self._graphics.CGEventGetLocation
        self._get_location.argtypes = [ctypes.c_void_p]
        self._get_location.restype = _CGPoint
        self._button_down = self._graphics.CGEventSourceButtonState
        self._button_down.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
        self._button_down.restype = ctypes.c_bool
        self._release = self._foundation.CFRelease
        self._release.argtypes = [ctypes.c_void_p]
        self._release.restype = None

    def create_event(self):
        return self._create_event(None)

    def get_location(self, event):
        return self._get_location(event)

    def button_down(self):
        # CombinedSessionState and kCGMouseButtonLeft are both zero.
        return bool(self._button_down(0, 0))

    def release(self, event):
        self._release(event)


class MacOSGlobalPointerReader:
    def __init__(self, native: Any | None = None):
        self._native = native or _CoreGraphicsNative()

    def sample(self) -> PointerSample:
        event = None
        try:
            event = self._native.create_event()
            if not event:
                raise PointerReadError("CGEventCreate returned no event")
            location = self._native.get_location(event)
            x = float(location.x)
            y = float(location.y)
            if not (math.isfinite(x) and math.isfinite(y)):
                raise PointerReadError("global pointer coordinates are not finite")
            left_down = bool(self._native.button_down())
            return PointerSample(x, y, left_down)
        except PointerReadError:
            raise
        except Exception as exc:
            raise PointerReadError(f"global pointer read failed: {exc}") from exc
        finally:
            if event:
                try:
                    self._native.release(event)
                except Exception as exc:
                    # A release failure is still a failed sample; never let a
                    # stale capture continue after the native object is unsafe.
                    if isinstance(exc, PointerReadError):
                        raise
                    raise PointerReadError(f"global pointer release failed: {exc}") from exc


def create_global_pointer_reader(*, platform: str | None = None):
    platform = sys.platform if platform is None else platform
    if platform != "darwin":
        return None
    try:
        return MacOSGlobalPointerReader()
    except Exception:
        return None
