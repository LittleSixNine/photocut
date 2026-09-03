"""Bounded, token-scoped binary images for the local confirmation web UI."""
from __future__ import annotations

import math
import threading
from collections import OrderedDict
from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType
from typing import Callable, Mapping

import cv2

from photocut.confirmation.model import render_magnifier_source
from photocut.cli import build_confirmation_preview


class StaleImageToken(ValueError):
    """The client requested media for an image that is no longer current."""


class PreviewEncodingError(RuntimeError):
    """An image could not be encoded into the route's fixed binary format."""

    def __init__(self, code: str):
        super().__init__(code.replace("_", " "))
        self.code = code


@dataclass(frozen=True)
class EncodedImage:
    body: bytes
    content_type: str
    image_token: str
    width: int
    height: int
    transform: Mapping[str, float | int]

    def __post_init__(self) -> None:
        if type(self.body) is not bytes:
            raise TypeError("encoded body must be bytes")
        if self.content_type not in {"image/jpeg", "image/png"}:
            raise ValueError("unsupported encoded image content type")
        if not isinstance(self.image_token, str) or not self.image_token:
            raise ValueError("image token must be non-empty")
        if type(self.width) is not int or type(self.height) is not int:
            raise TypeError("encoded image dimensions must be integers")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("encoded image dimensions must be positive")
        if not isinstance(self.transform, Mapping):
            raise TypeError("transform must be a mapping")
        canonical = {}
        for key, value in self.transform.items():
            if not isinstance(key, str):
                raise TypeError("transform keys must be strings")
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
            ):
                raise ValueError("transform values must be finite numbers")
            canonical[key] = value
        object.__setattr__(self, "transform", MappingProxyType(canonical))


class ByteBudgetCache:
    """A thread-safe LRU whose hard limit is encoded payload bytes."""

    def __init__(self, max_bytes: int):
        if type(max_bytes) is not int or max_bytes < 0:
            raise ValueError("cache byte budget must be a non-negative integer")
        self.max_bytes = max_bytes
        self._values = OrderedDict()
        self._total_bytes = 0
        self._peak_bytes = 0
        self._lock = threading.RLock()

    @staticmethod
    def _size(value) -> int:
        if isinstance(value, EncodedImage):
            return len(value.body)
        if type(value) is bytes:
            return len(value)
        raise TypeError("cache values must be bytes or EncodedImage")

    @property
    def total_bytes(self) -> int:
        with self._lock:
            return self._total_bytes

    @property
    def peak_bytes(self) -> int:
        with self._lock:
            return self._peak_bytes

    def __len__(self) -> int:
        with self._lock:
            return len(self._values)

    def get(self, key):
        with self._lock:
            value = self._values.get(key)
            if value is not None:
                self._values.move_to_end(key)
            return value

    def put(self, key, value) -> None:
        size = self._size(value)
        with self._lock:
            previous = self._values.pop(key, None)
            if previous is not None:
                self._total_bytes -= self._size(previous)
            if size > self.max_bytes:
                return
            self._values[key] = value
            self._total_bytes += size
            while self._total_bytes > self.max_bytes and self._values:
                _, removed = self._values.popitem(last=False)
                self._total_bytes -= self._size(removed)
            self._peak_bytes = max(self._peak_bytes, self._total_bytes)

    def remove_if(self, predicate: Callable[[object, object], bool]) -> None:
        with self._lock:
            for key in list(self._values):
                value = self._values[key]
                if predicate(key, value):
                    self._total_bytes -= self._size(self._values.pop(key))

    def clear(self) -> None:
        with self._lock:
            self._values.clear()
            self._total_bytes = 0

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._values)


@dataclass
class _EncodingFlight:
    event: threading.Event
    result: EncodedImage | None = None
    error: BaseException | None = None


class ConfirmationMediaService:
    """Encode current/next confirmation media without retaining full next images."""

    def __init__(
        self,
        session,
        preview_budget: int = 64 * 1024 * 1024,
        magnifier_budget: int = 8 * 1024 * 1024,
    ):
        self.session = session
        self.preview_cache = ByteBudgetCache(preview_budget)
        self.magnifier_cache = ByteBudgetCache(magnifier_budget)
        self.preview_encode_count = 0
        self.magnifier_encode_count = 0
        self._current_token: str | None = None
        self._state_lock = threading.RLock()
        self._flights: dict[tuple, _EncodingFlight] = {}

    @property
    def prefetched_full_image(self):
        return None

    @property
    def preview_cache_tokens(self) -> set[str]:
        return {key[0] for key in self.preview_cache.snapshot()}

    @property
    def has_observed_current(self) -> bool:
        with self._state_lock:
            return self._current_token is not None

    @property
    def inflight_count(self) -> int:
        with self._state_lock:
            return len(self._flights)

    @staticmethod
    def _validate_preview_geometry(
        css_width: int, css_height: int, dpr: float
    ) -> tuple[int, int]:
        if (
            type(css_width) is not int
            or type(css_height) is not int
            or not 1 <= css_width <= 4096
            or not 1 <= css_height <= 4096
        ):
            raise ValueError("preview CSS dimensions must be integers in 1..4096")
        if (
            isinstance(dpr, bool)
            or not isinstance(dpr, Real)
            or not math.isfinite(dpr)
            or not 0.5 <= dpr <= 4.0
        ):
            raise ValueError("device pixel ratio must be finite and in 0.5..4.0")
        return (
            min(4096, max(1, round(css_width * dpr))),
            min(4096, max(1, round(css_height * dpr))),
        )

    @staticmethod
    def _validate_token(value: object) -> str:
        if not isinstance(value, str) or not value:
            raise ValueError("image token must be non-empty")
        return value

    def _observe_current_locked(self, current_token: str) -> None:
        current_token = self._validate_token(current_token)
        if self._current_token == current_token:
            return
        self._current_token = current_token
        self.preview_cache.remove_if(
            lambda key, value: key[0] != current_token
        )
        self.magnifier_cache.clear()

    def sync_navigation(self) -> None:
        """Immediately discard media that cannot belong to the current item."""
        _, current_token = self.session.current_image()
        with self._state_lock:
            self._observe_current_locked(current_token)

    def _observe_current(self, current_token: str) -> None:
        with self._state_lock:
            self._observe_current_locked(current_token)

    def _fence_current(self, expected_token: str, message: str) -> None:
        _, actual_token = self.session.current_image()
        with self._state_lock:
            self._observe_current_locked(actual_token)
            if actual_token != expected_token:
                raise StaleImageToken(message)

    def _singleflight(
        self,
        *,
        namespace: str,
        key: tuple,
        cache: ByteBudgetCache,
        expected_current_token: str,
        producer: Callable[[], EncodedImage],
        publish: Callable[[EncodedImage], None],
    ) -> EncodedImage:
        flight_key = (namespace, *key)
        with self._state_lock:
            cached = cache.get(key)
            if cached is not None:
                return cached
            flight = self._flights.get(flight_key)
            owner = flight is None
            if owner:
                flight = _EncodingFlight(threading.Event())
                self._flights[flight_key] = flight

        if not owner:
            flight.event.wait()
            if flight.error is not None:
                raise flight.error
            if flight.result is None:
                raise RuntimeError("encoding flight completed without a result")
            return flight.result

        try:
            result = producer()
            self._fence_current(
                expected_current_token,
                f"current image changed during {namespace} render",
            )
            with self._state_lock:
                if self._current_token != expected_current_token:
                    raise StaleImageToken(
                        f"current image changed during {namespace} publish"
                    )
                publish(result)
                flight.result = result
            return result
        except BaseException as exc:
            flight.error = exc
            raise
        finally:
            with self._state_lock:
                if self._flights.get(flight_key) is flight:
                    self._flights.pop(flight_key)
                flight.event.set()

    def preview(
        self, image_token: str, css_width: int, css_height: int, dpr: float
    ) -> EncodedImage:
        physical_width, physical_height = self._validate_preview_geometry(
            css_width, css_height, dpr
        )
        image_token = self._validate_token(image_token)
        image, current_token = self.session.current_image()
        self._observe_current(current_token)
        if image_token != current_token:
            raise StaleImageToken("preview token is not current")
        result = self._encode_preview(
            image,
            image_token,
            physical_width,
            physical_height,
            expected_current_token=current_token,
        )
        self._fence_current(
            current_token, "current image changed during preview lookup"
        )
        return result

    def _encode_preview(
        self,
        image,
        image_token: str,
        physical_width: int,
        physical_height: int,
        *,
        expected_current_token: str,
    ) -> EncodedImage:
        key = (image_token, physical_width, physical_height)

        def produce() -> EncodedImage:
            preview = build_confirmation_preview(
                image,
                viewport_width=physical_width,
                viewport_height=physical_height,
            )
            try:
                ok, encoded = cv2.imencode(
                    ".jpg",
                    preview.scaled_image,
                    [cv2.IMWRITE_JPEG_QUALITY, 90],
                )
            except cv2.error as exc:
                raise PreviewEncodingError("preview_encoding_failed") from exc
            if not ok or encoded is None:
                raise PreviewEncodingError("preview_encoding_failed")
            transform = preview.transform
            return EncodedImage(
                body=encoded.tobytes(),
                content_type="image/jpeg",
                image_token=image_token,
                width=preview.scaled_width,
                height=preview.scaled_height,
                transform={
                    "scale": float(transform.scale),
                    "offset_x": int(transform.offset_x),
                    "offset_y": int(transform.offset_y),
                    "original_width": int(transform.original_width),
                    "original_height": int(transform.original_height),
                },
            )

        def publish(result: EncodedImage) -> None:
            self.preview_encode_count += 1
            # Keep one encoded representation for current and at most one next.
            self.preview_cache.remove_if(
                lambda existing_key, value: (
                    existing_key[0] == image_token and existing_key != key
                )
                or existing_key[0] not in {self._current_token, image_token}
            )
            self.preview_cache.put(key, result)

        return self._singleflight(
            namespace="preview",
            key=key,
            cache=self.preview_cache,
            expected_current_token=expected_current_token,
            producer=produce,
            publish=publish,
        )

    def prefetch_next(
        self, css_width: int, css_height: int, dpr: float
    ) -> EncodedImage:
        physical_width, physical_height = self._validate_preview_geometry(
            css_width, css_height, dpr
        )
        image = None
        try:
            _, current_token = self.session.current_image()
            self._observe_current(current_token)
            image, image_token = self.session.next_preview_source()
            image_token = self._validate_token(image_token)
            result = self._encode_preview(
                image,
                image_token,
                physical_width,
                physical_height,
                expected_current_token=current_token,
            )
            self._fence_current(
                current_token, "current image changed during prefetch lookup"
            )
            return result
        finally:
            if image is not None:
                del image

    def magnifier(
        self,
        image_token: str,
        corner_index: int,
        zoom: int,
        size: int,
        center: tuple[int, int] | None = None,
    ) -> EncodedImage:
        image_token = self._validate_token(image_token)
        if type(corner_index) is not int or corner_index not in range(4):
            raise ValueError("corner index must be integer 0..3")
        if type(zoom) is not int or zoom not in (2, 4, 8):
            raise ValueError("zoom must be one of 2, 4, 8")
        if (
            type(size) is not int
            or not 64 <= size <= 768
            or size % zoom
        ):
            raise ValueError("magnifier size must be 64..768 and divisible by zoom")
        image, current_token = self.session.current_image()
        self._observe_current(current_token)
        if image_token != current_token:
            raise StaleImageToken("magnifier token is not current")
        snapshot = self.session.snapshot()
        self._fence_current(
            current_token, "current image changed during magnifier snapshot"
        )
        corner = snapshot["editor"]["corners"][corner_index]
        if (
            not isinstance(corner, (list, tuple))
            or len(corner) != 2
            or any(type(value) is not int for value in corner)
        ):
            raise RuntimeError("invalid canonical corner")
        canonical_corner = (corner[0], corner[1])
        render_center = canonical_corner if center is None else center
        if (
            not isinstance(render_center, (list, tuple))
            or len(render_center) != 2
            or any(type(value) is not int for value in render_center)
            or not 0 <= render_center[0] < image.shape[1]
            or not 0 <= render_center[1] < image.shape[0]
        ):
            raise ValueError("magnifier center must be within the current image")
        render_center = (render_center[0], render_center[1])
        key = (
            image_token,
            corner_index,
            zoom,
            size,
            render_center,
        )

        def produce() -> EncodedImage:
            source = render_magnifier_source(
                image, render_center, zoom, viewport_size=size
            )
            try:
                ok, encoded = cv2.imencode(".png", source)
            except cv2.error as exc:
                raise PreviewEncodingError("magnifier_encoding_failed") from exc
            if not ok or encoded is None:
                raise PreviewEncodingError("magnifier_encoding_failed")
            return EncodedImage(
                body=encoded.tobytes(),
                content_type="image/png",
                image_token=image_token,
                width=int(source.shape[1]),
                height=int(source.shape[0]),
                transform={},
            )

        def publish(result: EncodedImage) -> None:
            self.magnifier_encode_count += 1
            self.magnifier_cache.put(key, result)

        result = self._singleflight(
            namespace="magnifier",
            key=key,
            cache=self.magnifier_cache,
            expected_current_token=current_token,
            producer=produce,
            publish=publish,
        )
        self._fence_current(
            current_token, "current image changed during magnifier lookup"
        )
        return result


__all__ = [
    "ByteBudgetCache",
    "ConfirmationMediaService",
    "EncodedImage",
    "PreviewEncodingError",
    "StaleImageToken",
]
