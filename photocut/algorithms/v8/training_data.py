"""Pure NumPy/OpenCV training-data contracts for the V8 mask baseline."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _quad(value: Any, image_size: tuple[int, int] | None = None) -> np.ndarray:
    try:
        points = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("corners must be a convex quad") from exc
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError("corners must be a convex quad")
    turns = []
    for index in range(4):
        a, b, c = points[index], points[(index + 1) % 4], points[(index + 2) % 4]
        first, second = b - a, c - b
        turns.append(float(first[0] * second[1] - first[1] * second[0]))
    if not (all(turn > 1e-9 for turn in turns) or all(turn < -1e-9 for turn in turns)):
        raise ValueError("corners must be a convex ordered quad")
    if image_size is not None:
        width, height = image_size
        if np.any(points[:, 0] < 0) or np.any(points[:, 1] < 0) or np.any(points[:, 0] > width - 1) or np.any(points[:, 1] > height - 1):
            raise ValueError("corners exceed mask bounds")
    return points


def rasterize_photo_mask(
    image_size: Sequence[int],
    corners: Sequence[Sequence[float]],
) -> np.ndarray:
    """Rasterize one TL/TR/BR/BL photo quad into a binary uint8 mask."""
    if len(image_size) != 2:
        raise ValueError("image_size must be (width, height)")
    width, height = image_size
    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
        raise ValueError("image_size must contain positive integers")
    points = _quad(corners, (width, height))
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.rint(points).astype(np.int32), 1, lineType=cv2.LINE_8)
    return mask


@dataclass(frozen=True)
class GroupSplit:
    seed: int
    calibration_fraction: float
    fit_groups: tuple[str, ...]
    calibration_groups: tuple[str, ...]
    fit_image_ids: tuple[str, ...]
    calibration_image_ids: tuple[str, ...]


@dataclass(frozen=True)
class AugmentationConfig:
    """One bounded recipe for scanner-white photo composites."""

    output_size: tuple[int, int] = (1024, 768)
    synthetic_per_source: int = 32
    max_rotation_deg: float = 25.0
    max_perspective: float = 0.08
    min_photo_scale: float = 0.58
    max_photo_scale: float = 0.88
    background_range: tuple[int, int] = (226, 255)
    noise_sigma_max: float = 2.5
    jpeg_quality_range: tuple[int, int] = (86, 100)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.output_size, (tuple, list))
            or len(self.output_size) != 2
            or any(type(value) is not int or value < 32 or value > 4096 for value in self.output_size)
        ):
            raise ValueError("output_size must contain bounded width and height")
        object.__setattr__(self, "output_size", tuple(self.output_size))
        if type(self.synthetic_per_source) is not int or not 0 <= self.synthetic_per_source <= 128:
            raise ValueError("synthetic_per_source must be in [0, 128]")
        for name, low, high in (
            ("max_rotation_deg", 0.0, 35.0),
            ("max_perspective", 0.0, 0.15),
            ("noise_sigma_max", 0.0, 8.0),
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)) or not low <= float(value) <= high:
                raise ValueError(f"{name} is out of bounds")
            object.__setattr__(self, name, float(value))
        for name in ("min_photo_scale", "max_photo_scale"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, float(value))
        if not 0.3 <= self.min_photo_scale <= self.max_photo_scale <= 0.95:
            raise ValueError("photo scale range is out of bounds")
        for name, low, high in (
            ("background_range", 180, 255),
            ("jpeg_quality_range", 60, 100),
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, (tuple, list))
                or len(value) != 2
                or any(type(item) is not int or not low <= item <= high for item in value)
                or value[0] > value[1]
            ):
                raise ValueError(f"{name} is out of bounds")
            object.__setattr__(self, name, tuple(value))

    def to_dict(self) -> dict[str, Any]:
        return {
            "recipe": "scanner_white_composite_v1",
            "output_size": list(self.output_size),
            "synthetic_per_source": self.synthetic_per_source,
            "max_rotation_deg": self.max_rotation_deg,
            "max_perspective": self.max_perspective,
            "min_photo_scale": self.min_photo_scale,
            "max_photo_scale": self.max_photo_scale,
            "background_range": list(self.background_range),
            "noise_sigma_max": self.noise_sigma_max,
            "jpeg_quality_range": list(self.jpeg_quality_range),
        }

    def sha256(self) -> str:
        return _sha256(_canonical(self.to_dict()))


@dataclass(frozen=True)
class AugmentedSample:
    image_bgr: np.ndarray
    mask: np.ndarray
    corners: tuple[tuple[float, float], ...]
    lineage: Mapping[str, Any]


def _extract_photo_patch(image_bgr: np.ndarray, corners: Any) -> np.ndarray:
    points = _quad(corners)
    top = np.linalg.norm(points[1] - points[0])
    bottom = np.linalg.norm(points[2] - points[3])
    left = np.linalg.norm(points[3] - points[0])
    right = np.linalg.norm(points[2] - points[1])
    width = max(2, int(round(max(top, bottom))))
    height = max(2, int(round(max(left, right))))
    target = np.asarray(((0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)), dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(points.astype(np.float32), target)
    return cv2.warpPerspective(
        image_bgr,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def _target_quad(rng: np.random.Generator, patch_size: tuple[int, int], config: AugmentationConfig) -> np.ndarray:
    output_width, output_height = config.output_size
    patch_width, patch_height = patch_size
    scale_fraction = float(rng.uniform(config.min_photo_scale, config.max_photo_scale))
    fit = min(
        output_width * scale_fraction / max(1.0, patch_width),
        output_height * scale_fraction / max(1.0, patch_height),
    )
    width = max(12.0, patch_width * fit)
    height = max(12.0, patch_height * fit)
    points = np.asarray((
        (-width / 2, -height / 2),
        (width / 2, -height / 2),
        (width / 2, height / 2),
        (-width / 2, height / 2),
    ), dtype=np.float64)
    angle = math.radians(float(rng.uniform(-config.max_rotation_deg, config.max_rotation_deg)))
    rotation = np.asarray(((math.cos(angle), -math.sin(angle)), (math.sin(angle), math.cos(angle))))
    points = points @ rotation.T
    jitter = config.max_perspective * min(width, height)
    points += rng.uniform(-jitter, jitter, size=(4, 2))

    margin = 3.0
    span = np.ptp(points, axis=0)
    allowed = np.asarray((output_width - 2 * margin, output_height - 2 * margin))
    if np.any(span > allowed):
        points *= float(np.min(allowed / np.maximum(span, 1e-9)))
    lower = np.asarray((margin, margin)) - points.min(axis=0)
    upper = np.asarray((output_width - 1 - margin, output_height - 1 - margin)) - points.max(axis=0)
    center = np.asarray((
        rng.uniform(lower[0], max(lower[0], upper[0])),
        rng.uniform(lower[1], max(lower[1], upper[1])),
    ))
    return points + center


def generate_synthetic_sample(
    image_bgr: np.ndarray,
    corners: Sequence[Sequence[float]],
    *,
    seed: int,
    source_image_id: str,
    config: AugmentationConfig = AugmentationConfig(),
) -> AugmentedSample:
    """Composite one confirmed physical photo onto a procedural scanner bed."""
    if type(seed) is not int:
        raise TypeError("seed must be an integer")
    if not isinstance(source_image_id, str) or not source_image_id:
        raise ValueError("source_image_id is required")
    if not isinstance(config, AugmentationConfig):
        raise TypeError("config must be AugmentationConfig")
    image = np.asarray(image_bgr)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("image_bgr must be an HxWx3 uint8 image")
    patch = _extract_photo_patch(image, corners)
    rng = np.random.default_rng(seed)
    output_width, output_height = config.output_size
    target = _target_quad(rng, (patch.shape[1], patch.shape[0]), config)
    target_tuple = tuple((float(point[0]), float(point[1])) for point in target)
    mask = rasterize_photo_mask(config.output_size, target_tuple)

    low, high = config.background_range
    neutral = float(rng.uniform(low, high))
    warmth = float(rng.uniform(-4.0, 4.0))
    base_bgr = np.asarray((neutral + warmth, neutral, neutral - warmth), dtype=np.float32)
    vertical = np.linspace(float(rng.uniform(-2, 2)), float(rng.uniform(-2, 2)), output_height, dtype=np.float32)
    background = np.broadcast_to(base_bgr, (output_height, output_width, 3)).copy()
    background += vertical[:, None, None]
    if config.noise_sigma_max:
        sigma = float(rng.uniform(0.0, config.noise_sigma_max))
        background += rng.normal(0.0, sigma, size=background.shape).astype(np.float32)
    else:
        sigma = 0.0

    shadow_scale = min(output_width, output_height)
    shadow_dx = int(round(rng.uniform(-0.012, 0.022) * shadow_scale))
    shadow_dy = int(round(rng.uniform(0.004, 0.026) * shadow_scale))
    shifted = cv2.warpAffine(
        mask.astype(np.float32),
        np.asarray(((1, 0, shadow_dx), (0, 1, shadow_dy)), dtype=np.float32),
        (output_width, output_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    blur_sigma = float(rng.uniform(0.8, 4.0))
    shadow = cv2.GaussianBlur(shifted, (0, 0), blur_sigma)
    shadow_strength = float(rng.uniform(5.0, 24.0))
    background -= shadow[:, :, None] * shadow_strength

    gain = float(rng.uniform(0.90, 1.10))
    channel_shift = np.asarray((rng.uniform(-4, 4), rng.uniform(-2, 2), rng.uniform(-4, 4)), dtype=np.float32)
    adjusted_patch = np.clip(patch.astype(np.float32) * gain + channel_shift, 0, 255).astype(np.uint8)
    source_quad = np.asarray(((0, 0), (patch.shape[1] - 1, 0), (patch.shape[1] - 1, patch.shape[0] - 1), (0, patch.shape[0] - 1)), dtype=np.float32)
    transform = cv2.getPerspectiveTransform(source_quad, target.astype(np.float32))
    warped = cv2.warpPerspective(
        adjusted_patch,
        transform,
        (output_width, output_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    composite = np.clip(background, 0, 255).astype(np.uint8)
    composite[mask == 1] = warped[mask == 1]
    quality = int(rng.integers(config.jpeg_quality_range[0], config.jpeg_quality_range[1] + 1))
    ok, encoded = cv2.imencode(".jpg", composite, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("failed to encode synthetic scanner sample")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise RuntimeError("failed to decode synthetic scanner sample")
    lineage = {
        "recipe": "scanner_white_composite_v1",
        "source_image_id": source_image_id,
        "seed": seed,
        "config_hash": config.sha256(),
        "target_corners": [list(point) for point in target_tuple],
        "background_bgr": [float(value) for value in base_bgr],
        "noise_sigma": sigma,
        "shadow_offset": [shadow_dx, shadow_dy],
        "shadow_blur_sigma": blur_sigma,
        "shadow_strength": shadow_strength,
        "photo_gain": gain,
        "photo_channel_shift": [float(value) for value in channel_shift],
        "jpeg_quality": quality,
    }
    return AugmentedSample(decoded, mask, target_tuple, lineage)


def build_augmentation_lineage(
    training_manifest: Mapping[str, Any],
    *,
    config: AugmentationConfig,
    seed: int,
) -> list[dict[str, Any]]:
    """Build deterministic real/synthetic lineage without materializing pixels."""
    if not isinstance(training_manifest, Mapping) or training_manifest.get("schema_version") != 1:
        raise ValueError("invalid V8 training manifest")
    if training_manifest.get("frozen_accessed") is not False:
        raise PermissionError("frozen data cannot enter V8 augmentation")
    if type(seed) is not int or not isinstance(config, AugmentationConfig):
        raise ValueError("augmentation seed and config are required")
    manifest_id = training_manifest.get("manifest_id")
    samples = training_manifest.get("samples")
    if not isinstance(manifest_id, str) or not isinstance(samples, list):
        raise ValueError("invalid V8 training manifest")
    result: list[dict[str, Any]] = []
    for sample in sorted(samples, key=lambda value: value.get("image_id", "")):
        image_id = sample.get("image_id")
        subset = sample.get("training_subset")
        if not isinstance(image_id, str) or subset not in {"fit", "calibration"}:
            raise ValueError("invalid V8 training sample lineage")
        real_payload = f"{manifest_id}:real:{image_id}".encode("utf-8")
        result.append({
            "sample_id": _sha256(real_payload),
            "kind": "real",
            "source_image_id": image_id,
            "training_subset": subset,
            "recipe": "confirmed_real_v1",
        })
        if subset != "fit":
            continue
        for index in range(config.synthetic_per_source):
            digest = hashlib.sha256(f"{seed}:{image_id}:{index}".encode("utf-8")).digest()
            sample_seed = int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)
            payload = f"{manifest_id}:synthetic:{image_id}:{index}:{sample_seed}:{config.sha256()}".encode("utf-8")
            result.append({
                "sample_id": _sha256(payload),
                "kind": "synthetic",
                "source_image_id": image_id,
                "training_subset": "fit",
                "recipe": "scanner_white_composite_v1",
                "synthetic_index": index,
                "seed": sample_seed,
                "config_hash": config.sha256(),
            })
    return result


def _validate_training_sample(sample: Mapping[str, Any]) -> None:
    if sample.get("split") != "train":
        raise PermissionError("V8 training accepts train samples only")
    if sample.get("eligible") is not True:
        raise PermissionError("V8 training accepts pre-sealed eligible samples only")
    for key in ("image_id", "origin_group_id", "population_id"):
        if not isinstance(sample.get(key), str) or not sample[key]:
            raise ValueError(f"training sample requires {key}")


def split_training_groups(
    samples: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    calibration_fraction: float,
) -> GroupSplit:
    """Deterministically split released train samples by origin group."""
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if not isinstance(calibration_fraction, (int, float)) or isinstance(calibration_fraction, bool) or not 0 <= calibration_fraction <= 1:
        raise ValueError("calibration_fraction must be in [0, 1]")
    if not samples:
        raise ValueError("training samples cannot be empty")
    for sample in samples:
        _validate_training_sample(sample)
    image_ids = [sample["image_id"] for sample in samples]
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("duplicate training image_id")

    groups = sorted({sample["origin_group_id"] for sample in samples})
    ranked = sorted(
        groups,
        key=lambda group: (hashlib.sha256(f"{seed}:{group}".encode()).hexdigest(), group),
    )
    calibration_count = int(round(len(groups) * float(calibration_fraction)))
    if len(groups) > 1 and 0 < calibration_fraction < 1:
        calibration_count = min(len(groups) - 1, max(1, calibration_count))
    calibration = frozenset(ranked[:calibration_count])
    fit = frozenset(groups) - calibration
    fit_ids = tuple(sorted(sample["image_id"] for sample in samples if sample["origin_group_id"] in fit))
    calibration_ids = tuple(sorted(sample["image_id"] for sample in samples if sample["origin_group_id"] in calibration))
    return GroupSplit(
        seed=seed,
        calibration_fraction=float(calibration_fraction),
        fit_groups=tuple(sorted(fit)),
        calibration_groups=tuple(sorted(calibration)),
        fit_image_ids=fit_ids,
        calibration_image_ids=calibration_ids,
    )


def _relative_path(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("training object path must be relative")
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or value.startswith("~") or Path(value).is_absolute():
        raise ValueError("training object path must be relative and contained")
    return path.as_posix()


def build_training_manifest(
    samples: Sequence[Mapping[str, Any]],
    *,
    split: GroupSplit,
    generator_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a deterministic lineage-only V8 manifest without private paths."""
    if not isinstance(split, GroupSplit):
        raise TypeError("split must be a GroupSplit")
    if not isinstance(generator_identity, Mapping) or not generator_identity:
        raise ValueError("generator_identity is required")
    if not samples:
        raise ValueError("training samples cannot be empty")
    for sample in samples:
        _validate_training_sample(sample)
    populations = {sample["population_id"] for sample in samples}
    if len(populations) != 1:
        raise ValueError("training samples must share one population_id")
    expected_ids = set(split.fit_image_ids) | set(split.calibration_image_ids)
    actual_ids = {sample["image_id"] for sample in samples}
    if actual_ids != expected_ids or set(split.fit_image_ids) & set(split.calibration_image_ids):
        raise ValueError("group split does not match training samples")

    fit_ids = set(split.fit_image_ids)
    rows = []
    for sample in sorted(samples, key=lambda value: value["image_id"]):
        row = {
            "image_id": sample["image_id"],
            "origin_group_id": sample["origin_group_id"],
            "training_subset": "fit" if sample["image_id"] in fit_ids else "calibration",
            "relative_object_path": _relative_path(sample.get("relative_object_path")),
            "expected_object_hash": sample["expected_object_hash"],
            "photo_truth": [list(point) for point in _quad(sample["photo_truth"])],
            "coordinate_provenance_id": sample["coordinate_provenance_id"],
        }
        for field in (
            "exif_orientation",
            "original_size",
            "full_normalized_size",
            "analysis_size",
            "analysis_photo_truth",
            "transform_hash",
        ):
            if field in sample:
                row[field] = sample[field]
        rows.append(row)
    split_value = {
        "seed": split.seed,
        "calibration_fraction": split.calibration_fraction,
        "fit_groups": list(split.fit_groups),
        "calibration_groups": list(split.calibration_groups),
        "fit_image_ids": list(split.fit_image_ids),
        "calibration_image_ids": list(split.calibration_image_ids),
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "population_id": next(iter(populations)),
        "generator_identity": dict(generator_identity),
        "group_split": split_value,
        "samples": rows,
        "frozen_accessed": False,
    }
    payload["manifest_id"] = _sha256(_canonical(payload))
    return payload


__all__ = [
    "AugmentationConfig",
    "AugmentedSample",
    "GroupSplit",
    "build_augmentation_lineage",
    "build_training_manifest",
    "generate_synthetic_sample",
    "rasterize_photo_mask",
    "split_training_groups",
]
