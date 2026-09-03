"""Explicit byte-oriented input normalization for v7.

The v7 detector operates in one coordinate space: the EXIF-transposed image in
BGR uint8 form.  Legacy PhotoCut loading is intentionally kept separate and is
only exposed through :class:`PairedInputViews` for comparison and fallback.
"""
from __future__ import annotations

import hashlib
import io
import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


_PIL_PIXEL_LIMIT_LOCK = threading.RLock()


class InputDecodeError(ValueError):
    """Raised for corrupt, unsupported, or over-budget image input."""


class InputCancelled(RuntimeError):
    """Raised when a caller cancellation token is set during decoding."""


CancellationError = InputCancelled


def _cancelled(token: Any) -> bool:
    if token is None:
        return False
    value = getattr(token, "is_cancelled", None)
    if callable(value):
        try:
            return bool(value())
        except TypeError:
            pass
    if value is not None and not callable(value):
        return bool(value)
    value = getattr(token, "cancelled", False)
    return bool(value() if callable(value) else value)


def _check_cancel(token: Any) -> None:
    if _cancelled(token):
        raise InputCancelled("image decode cancelled")


def _orientation_matrix(orientation: int, width: int, height: int) -> np.ndarray:
    """Map original image pixel coordinates to EXIF-normalized coordinates."""
    w, h = float(width), float(height)
    matrices = {
        1: ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        2: ((-1, 0, w - 1), (0, 1, 0), (0, 0, 1)),
        3: ((-1, 0, w - 1), (0, -1, h - 1), (0, 0, 1)),
        4: ((1, 0, 0), (0, -1, h - 1), (0, 0, 1)),
        5: ((0, 1, 0), (1, 0, 0), (0, 0, 1)),
        6: ((0, -1, h - 1), (1, 0, 0), (0, 0, 1)),
        7: ((0, -1, h - 1), (-1, 0, w - 1), (0, 0, 1)),
        8: ((0, 1, 0), (-1, 0, w - 1), (0, 0, 1)),
    }
    return np.asarray(matrices[orientation], dtype=np.float64)


def _as_tuple_matrix(matrix: np.ndarray) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(v) for v in row) for row in matrix)


def _immutable_array(array: np.ndarray) -> np.ndarray:
    """Back an ndarray by immutable bytes so writeability cannot be re-enabled."""
    contiguous = np.ascontiguousarray(array)
    return np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(contiguous.shape)


def _map_points(matrix: np.ndarray, points: Iterable[Sequence[float]]) -> tuple[tuple[float, float], ...]:
    result = []
    for point in points:
        if len(point) != 2 or not all(isinstance(v, (int, float, np.number)) and math.isfinite(float(v)) for v in point):
            raise ValueError("points must contain finite (x, y) coordinates")
        mapped = matrix @ np.array([float(point[0]), float(point[1]), 1.0])
        result.append((float(mapped[0] / mapped[2]), float(mapped[1] / mapped[2])))
    return tuple(result)


@dataclass(frozen=True)
class LoadedImage:
    source_sha256: str
    original_dtype: str
    original_channels: int
    exif_orientation: int
    normalized_bgr: np.ndarray
    forward_transform: tuple[tuple[float, ...], ...]
    inverse_transform: tuple[tuple[float, ...], ...]
    original_size: tuple[int, int]  # width, height
    normalized_size: tuple[int, int]  # width, height
    full_normalized_size: tuple[int, int] | None = None  # width, height
    analysis_to_full_transform: tuple[tuple[float, ...], ...] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )

    def __post_init__(self) -> None:
        if self.normalized_bgr.ndim != 3 or self.normalized_bgr.shape[2] != 3:
            raise ValueError("normalized_bgr must be HxWx3")
        object.__setattr__(self, "normalized_bgr", _immutable_array(self.normalized_bgr))
        full_size = self.normalized_size if self.full_normalized_size is None else self.full_normalized_size
        if (len(full_size) != 2 or any(
                not isinstance(value, int) or isinstance(value, bool) or value <= 0
                for value in full_size)):
            raise ValueError("full_normalized_size must contain positive integers")
        object.__setattr__(self, "full_normalized_size", tuple(full_size))
        transform = np.asarray(self.analysis_to_full_transform, dtype=np.float64)
        if transform.shape != (3, 3) or not np.isfinite(transform).all():
            raise ValueError("analysis_to_full_transform must be a finite 3x3 matrix")

    @property
    def image(self) -> np.ndarray:
        return self.normalized_bgr

    @property
    def bgr(self) -> np.ndarray:
        return self.normalized_bgr

    @property
    def dtype(self) -> str:
        return self.original_dtype

    @property
    def channels(self) -> int:
        return self.original_channels

    @property
    def orientation(self) -> int:
        return self.exif_orientation

    @property
    def original_bytes_sha256(self) -> str:
        return self.source_sha256

    @property
    def source_hash(self) -> str:
        return self.source_sha256

    @property
    def orientation_transform(self) -> str:
        return f"exif_{self.exif_orientation}"

    def map_original_to_normalized(self, points: Iterable[Sequence[float]]) -> tuple[tuple[float, float], ...]:
        return _map_points(np.asarray(self.forward_transform), points)

    def map_normalized_to_original(self, points: Iterable[Sequence[float]]) -> tuple[tuple[float, float], ...]:
        return _map_points(np.asarray(self.inverse_transform), points)

    def forward(self, points: Iterable[Sequence[float]]) -> tuple[tuple[float, float], ...]:
        return self.map_original_to_normalized(points)

    def inverse(self, points: Iterable[Sequence[float]]) -> tuple[tuple[float, float], ...]:
        return self.map_normalized_to_original(points)

    def map_analysis_to_full(self, points: Iterable[Sequence[float]]) -> tuple[tuple[float, float], ...]:
        """Map detector coordinates to full EXIF-normalized pixel coordinates."""
        return _map_points(np.asarray(self.analysis_to_full_transform), points)

    @property
    def is_analysis_preview(self) -> bool:
        return tuple(self.normalized_size) != tuple(self.full_normalized_size)


def _to_bgr(array: np.ndarray, channels: int) -> np.ndarray:
    if array.dtype.kind == "u" and array.dtype.itemsize == 2:
        # Only a lossless conversion is allowed.  Silent clipping of HDR/16-bit
        # values would change detector semantics and is therefore rejected.
        native = array.astype(np.uint16, copy=False)
        if native.size and int(native.max()) > 255:
            raise InputDecodeError("unsupported 16-bit conversion: values exceed uint8 range")
        array = native.astype(np.uint8)
    elif array.dtype != np.uint8:
        raise InputDecodeError(f"unsupported image dtype: {array.dtype}")
    if channels == 1:
        return cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
    if channels == 3:
        return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    if channels == 4:
        return cv2.cvtColor(array, cv2.COLOR_RGBA2BGR)
    raise InputDecodeError(f"unsupported channel count: {channels}")


def normalize_array(array: np.ndarray) -> np.ndarray:
    """Normalize an already-decoded HxW grayscale/RGB/RGBA array to BGR.

    This explicit seam is useful for TIFF readers and tests.  It accepts uint16
    only when conversion is lossless; values outside uint8 are rejected rather
    than clipped.
    """
    try:
        value = np.asarray(array)
    except Exception as exc:
        raise InputDecodeError("invalid image array") from exc
    if value.ndim == 2:
        channels = 1
    elif value.ndim == 3:
        channels = int(value.shape[2])
    else:
        raise InputDecodeError("unsupported channel count")
    try:
        return _immutable_array(_to_bgr(value, channels))
    except cv2.error as exc:
        raise InputDecodeError("invalid image array channels") from exc


def decode_bytes(data: bytes | bytearray | memoryview, *, max_pixels: int = 100_000_000,
                 max_pixel_budget: int | None = None, cancellation_token: Any = None,
                 cancel_token: Any = None) -> LoadedImage:
    """Decode bytes, apply EXIF orientation, and return immutable BGR pixels."""
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("image input must be bytes-like")
    raw = bytes(data)
    digest = hashlib.sha256(raw).hexdigest()  # deliberately before Pillow decode
    _check_cancel(cancellation_token if cancellation_token is not None else cancel_token)
    budget = max_pixel_budget if max_pixel_budget is not None else max_pixels
    if not isinstance(budget, int) or isinstance(budget, bool) or budget <= 0:
        raise ValueError("max_pixels must be a positive integer")
    try:
        with Image.open(io.BytesIO(raw)) as source:
            width, height = source.size
            # TIFF readers are permitted to expose an orientation-adjusted
            # ``size``.  The coordinate transform must use the stored raster
            # dimensions from tags 256/257 whenever available.
            raw_width, raw_height = width, height
            tags = getattr(source, "tag_v2", None)
            if tags is not None:
                try:
                    tagged_width, tagged_height = tags.get(256), tags.get(257)
                    if tagged_width is not None and tagged_height is not None:
                        tagged_width, tagged_height = int(tagged_width), int(tagged_height)
                        if tagged_width > 0 and tagged_height > 0:
                            raw_width, raw_height = tagged_width, tagged_height
                except (TypeError, ValueError, OverflowError):
                    pass
            if raw_width <= 0 or raw_height <= 0 or raw_width * raw_height > budget:
                raise InputDecodeError("image exceeds maximum pixel budget")
            _check_cancel(cancellation_token if cancellation_token is not None else cancel_token)
            orientation_value = source.getexif().get(274, 1)
            orientation = 1 if orientation_value is None else int(orientation_value)
            if orientation not in range(1, 9):
                raise InputDecodeError(f"unsupported EXIF orientation: {orientation}")
            mode = source.mode
            if mode not in {"L", "RGB", "RGBA", "I;16", "I;16B", "I;16L"}:
                raise InputDecodeError(f"unsupported image mode/channels: {mode}")
            transposed = ImageOps.exif_transpose(source)
            _check_cancel(cancellation_token if cancellation_token is not None else cancel_token)
            arr = np.asarray(transposed)
            if arr.ndim == 2:
                channels = 1
            elif arr.ndim == 3:
                channels = int(arr.shape[2])
            else:
                raise InputDecodeError("unsupported channel count")
            original_dtype = str(arr.dtype)
            bgr = _to_bgr(arr, channels)
            bgr = _immutable_array(bgr)
            norm_h, norm_w = bgr.shape[:2]
    except InputDecodeError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, TypeError, cv2.error) as exc:
        raise InputDecodeError(f"could not decode image bytes: {exc}") from exc
    forward = _orientation_matrix(orientation, raw_width, raw_height)
    inverse = np.linalg.inv(forward)
    # Ensure the matrix maps all stored-raster corners into the normalized
    # pixel domain; malformed TIFF metadata must fail closed.
    mapped_corners = _map_points(forward, ((0, 0), (raw_width - 1, 0),
                                           (raw_width - 1, raw_height - 1), (0, raw_height - 1)))
    if any(x < 0 or y < 0 or x > norm_w - 1 or y > norm_h - 1 for x, y in mapped_corners):
        raise InputDecodeError("EXIF orientation transform exceeds normalized image bounds")
    return LoadedImage(digest, original_dtype, channels, orientation, bgr,
                       _as_tuple_matrix(forward), _as_tuple_matrix(inverse),
                       (raw_width, raw_height), (norm_w, norm_h))


def _full_normalized_size(orientation: int, width: int, height: int) -> tuple[int, int]:
    return (height, width) if orientation in {5, 6, 7, 8} else (width, height)


def _endpoint_scale(source_size: int, target_size: int) -> float:
    if source_size <= 1 or target_size <= 1:
        return 1.0
    return float(target_size - 1) / float(source_size - 1)


def decode_bytes_for_analysis(
    data: bytes | bytearray | memoryview,
    *,
    max_pixels: int = 100_000_000,
    analysis_max_edge: int = 4096,
    cancellation_token: Any = None,
    cancel_token: Any = None,
) -> LoadedImage:
    """Decode an oversized JPEG at bounded resolution for detection.

    Small inputs retain the exact :func:`decode_bytes` path.  Oversized JPEGs
    use the decoder's native draft reduction before allocating pixels, then
    carry an explicit endpoint-preserving transform back to the full
    EXIF-normalized coordinate space.  TIFF and other formats still fail
    closed when over budget because Pillow cannot promise reduced decoding for
    them.
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("image input must be bytes-like")
    if (not isinstance(max_pixels, int) or isinstance(max_pixels, bool) or max_pixels <= 0 or
            not isinstance(analysis_max_edge, int) or isinstance(analysis_max_edge, bool) or
            analysis_max_edge < 2):
        raise ValueError("pixel budget and analysis_max_edge must be positive integers")
    raw = bytes(data)
    digest = hashlib.sha256(raw).hexdigest()
    token = cancellation_token if cancellation_token is not None else cancel_token
    _check_cancel(token)

    # Pillow raises its decompression-bomb exception while merely reading JPEG
    # metadata above ~178 MP.  The detector intentionally accepts that metadata
    # only inside this lock and immediately requests a bounded native draft;
    # the full raster is never materialized here.
    with _PIL_PIXEL_LIMIT_LOCK:
        previous_limit = Image.MAX_IMAGE_PIXELS
        try:
            Image.MAX_IMAGE_PIXELS = None
            with Image.open(io.BytesIO(raw)) as source:
                raw_width, raw_height = map(int, source.size)
                if raw_width <= 0 or raw_height <= 0:
                    raise InputDecodeError("invalid image dimensions")
                if raw_width * raw_height <= max_pixels:
                    # Close this metadata handle before the ordinary decoder.
                    use_exact_decoder = True
                else:
                    use_exact_decoder = False
                    if str(source.format or "").upper() not in {"JPEG", "JPG"}:
                        raise InputDecodeError(
                            "oversized input requires a JPEG reduced-resolution decoder"
                        )
                    orientation_value = source.getexif().get(274, 1)
                    orientation = 1 if orientation_value is None else int(orientation_value)
                    if orientation not in range(1, 9):
                        raise InputDecodeError(f"unsupported EXIF orientation: {orientation}")
                    original_mode = source.mode
                    if original_mode not in {"L", "RGB"}:
                        raise InputDecodeError(f"unsupported image mode/channels: {original_mode}")
                    ratio = min(1.0, analysis_max_edge / float(max(raw_width, raw_height)))
                    requested = (
                        max(2, int(round(raw_width * ratio))),
                        max(2, int(round(raw_height * ratio))),
                    )
                    source.draft("RGB" if original_mode == "RGB" else "L", requested)
                    _check_cancel(token)
                    transposed = ImageOps.exif_transpose(source)
                    if max(transposed.size) > analysis_max_edge:
                        shrink = analysis_max_edge / float(max(transposed.size))
                        resized = (
                            max(2, int(round(transposed.size[0] * shrink))),
                            max(2, int(round(transposed.size[1] * shrink))),
                        )
                        transposed = transposed.resize(resized, Image.Resampling.BOX)
                    arr = np.asarray(transposed)
                    channels = 1 if arr.ndim == 2 else int(arr.shape[2])
                    bgr = _immutable_array(_to_bgr(arr, channels))
                    norm_h, norm_w = bgr.shape[:2]
                    original_dtype = str(arr.dtype)
        except InputDecodeError:
            raise
        except (UnidentifiedImageError, OSError, ValueError, TypeError, cv2.error) as exc:
            raise InputDecodeError(f"could not decode image bytes: {exc}") from exc
        finally:
            Image.MAX_IMAGE_PIXELS = previous_limit

    if use_exact_decoder:
        return decode_bytes(raw, max_pixels=max_pixels, cancellation_token=token)

    full_norm_w, full_norm_h = _full_normalized_size(
        orientation, raw_width, raw_height
    )
    analysis_to_full = np.asarray((
        (_endpoint_scale(norm_w, full_norm_w), 0.0, 0.0),
        (0.0, _endpoint_scale(norm_h, full_norm_h), 0.0),
        (0.0, 0.0, 1.0),
    ), dtype=np.float64)
    full_to_analysis = np.linalg.inv(analysis_to_full)
    full_forward = _orientation_matrix(orientation, raw_width, raw_height)
    forward = full_to_analysis @ full_forward
    inverse = np.linalg.inv(forward)
    return LoadedImage(
        digest, original_dtype, channels, orientation, bgr,
        _as_tuple_matrix(forward), _as_tuple_matrix(inverse),
        (raw_width, raw_height), (norm_w, norm_h),
        (full_norm_w, full_norm_h), _as_tuple_matrix(analysis_to_full),
    )


def decode_full_bgr_bytes(
    data: bytes | bytearray | memoryview,
    *,
    exif_orientation: int,
    expected_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """Decode a verified snapshot at full resolution with explicit EXIF pose.

    This path is for one-at-a-time GUI display/crop after detection, not for
    feature extraction.  OpenCV is told to ignore embedded orientation and the
    already-audited orientation value is then applied deterministically.
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("image input must be bytes-like")
    if type(exif_orientation) is not int or exif_orientation not in range(1, 9):
        raise InputDecodeError("EXIF orientation must be an integer from 1 to 8")
    flags = int(cv2.IMREAD_COLOR) | int(getattr(cv2, "IMREAD_IGNORE_ORIENTATION", 128))
    image = cv2.imdecode(np.frombuffer(bytes(data), dtype=np.uint8), flags)
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise InputDecodeError("could not decode full-resolution snapshot")
    if exif_orientation == 2:
        image = cv2.flip(image, 1)
    elif exif_orientation == 3:
        image = cv2.flip(image, -1)
    elif exif_orientation == 4:
        image = cv2.flip(image, 0)
    elif exif_orientation == 5:
        image = cv2.transpose(image)
    elif exif_orientation == 6:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif exif_orientation == 7:
        image = cv2.flip(cv2.transpose(image), -1)
    elif exif_orientation == 8:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if expected_size is not None:
        expected = tuple(int(value) for value in expected_size)
        actual = (int(image.shape[1]), int(image.shape[0]))
        if actual != expected:
            raise InputDecodeError(
                f"normalized snapshot size mismatch: expected {expected}, got {actual}"
            )
    return _immutable_array(image)


def decode_path(path: str | Path, **kwargs: Any) -> LoadedImage:
    return decode_bytes(Path(path).read_bytes(), **kwargs)


@dataclass(frozen=True)
class PairedInputViews:
    legacy_array: np.ndarray
    v7_array: np.ndarray
    loaded: LoadedImage
    legacy_to_normalized_transform: tuple[tuple[float, ...], ...]
    normalized_to_legacy_transform: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "legacy_array", _immutable_array(np.asarray(self.legacy_array)))
        object.__setattr__(self, "v7_array", _immutable_array(np.asarray(self.v7_array)))

    @classmethod
    def from_path(cls, path: str | Path, **kwargs: Any) -> "PairedInputViews":
        path = Path(path)
        raw = path.read_bytes()
        loaded = decode_bytes(raw, **kwargs)
        # This import intentionally resolves the production loader at call time,
        # making the v5.2 view an exact characterization seam.
        from photocut import core as photocut
        legacy = photocut.load_image(str(path))
        if legacy is None:
            raise InputDecodeError(f"legacy photocut.load_image failed: {path}")
        legacy = np.asarray(legacy)
        if legacy.ndim < 2:
            raise InputDecodeError("legacy loader returned an invalid array")
        # OpenCV builds differ: some apply EXIF orientation by default while
        # others expose the untransposed raster. Infer the seam from shape.
        legacy_shape = tuple(int(v) for v in legacy.shape[:2])
        normalized_shape = (loaded.normalized_size[1], loaded.normalized_size[0])
        original_shape = (loaded.original_size[1], loaded.original_size[0])
        if legacy_shape == normalized_shape:
            matrix = np.eye(3, dtype=np.float64)
        elif legacy_shape == original_shape:
            matrix = np.asarray(loaded.forward_transform, dtype=np.float64)
        else:
            raise InputDecodeError(
                f"legacy image shape {legacy_shape} is neither normalized {normalized_shape} "
                f"nor original {original_shape}"
            )
        inverse = np.linalg.inv(matrix)
        return cls(_immutable_array(legacy), loaded.normalized_bgr, loaded,
                   _as_tuple_matrix(matrix), _as_tuple_matrix(inverse))

    @property
    def legacy(self) -> np.ndarray:
        return self.legacy_array

    @property
    def v7(self) -> np.ndarray:
        return self.v7_array

    @property
    def legacy_to_normalized(self) -> tuple[tuple[float, ...], ...]:
        return self.legacy_to_normalized_transform

    @property
    def normalized_to_legacy(self) -> tuple[tuple[float, ...], ...]:
        return self.normalized_to_legacy_transform

    @property
    def source_sha256(self) -> str:
        return self.loaded.source_sha256

    def map_legacy_to_normalized(self, points: Iterable[Sequence[float]]) -> tuple[tuple[float, float], ...]:
        return _map_points(np.asarray(self.legacy_to_normalized_transform), points)

    def map_normalized_to_legacy(self, points: Iterable[Sequence[float]]) -> tuple[tuple[float, float], ...]:
        return _map_points(np.asarray(self.normalized_to_legacy_transform), points)


def float_corners_to_display(corners: Iterable[Sequence[float]], image_size: Sequence[int]) -> tuple[tuple[int, int], ...]:
    """Convert float corners using one deterministic half-up rule."""
    if len(image_size) != 2:
        raise ValueError("image_size must be (width, height)")
    width, height = int(image_size[0]), int(image_size[1])
    if width <= 0 or height <= 0:
        raise ValueError("image_size must be positive")
    result = []
    for point in corners:
        if len(point) != 2 or not all(math.isfinite(float(v)) for v in point):
            raise ValueError("corners must contain finite points")
        x = min(width - 1, max(0, int(math.floor(float(point[0]) + 0.5))))
        y = min(height - 1, max(0, int(math.floor(float(point[1]) + 0.5))))
        result.append((x, y))
    return tuple(result)


# Concise aliases used by callers during the migration.
to_display_corners = float_corners_to_display
display_corners = float_corners_to_display
float_to_display = float_corners_to_display
decode_image_bytes = decode_bytes
normalize_image_array = normalize_array
