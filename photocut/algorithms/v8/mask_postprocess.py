"""Deterministic V8 foreground-mask validation and quadrilateral fitting."""
from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from photocut.algorithms.v7.geometry import GeometryError, order_quad, validate_quad
from photocut.algorithms.v7.model_inference import LetterboxTransform


class MaskPostprocessError(ValueError):
    pass


@dataclass(frozen=True)
class PhotoMaskCandidate:
    corners: tuple[tuple[float, float], ...]
    confidence: float
    evidence: Mapping[str, Any]


def _probability(outputs: Mapping[str, np.ndarray], transform: LetterboxTransform) -> np.ndarray:
    if not isinstance(outputs, Mapping) or set(outputs) != {"mask_logits"}:
        raise MaskPostprocessError("V8 mask output must contain mask_logits only")
    logits = np.asarray(outputs["mask_logits"])
    width, height = transform.model_size
    if (
        logits.shape != (1, 2, height, width)
        or not np.issubdtype(logits.dtype, np.number)
        or not np.isfinite(logits).all()
    ):
        raise MaskPostprocessError("mask_logits must be finite N,2,H,W values")
    difference = np.clip(logits[0, 1].astype(np.float64) - logits[0, 0].astype(np.float64), -60.0, 60.0)
    return (1.0 / (1.0 + np.exp(-difference))).astype(np.float32)


def restore_photo_mask_probability(
    outputs: Mapping[str, np.ndarray],
    transform: LetterboxTransform,
) -> np.ndarray:
    """Restore the foreground probability to original-image coordinates."""
    if not isinstance(transform, LetterboxTransform):
        raise MaskPostprocessError("transform must be a LetterboxTransform")
    probability = _probability(outputs, transform)
    original_width, original_height = transform.original_size
    content_width = max(1, int(round(original_width * transform.scale)))
    content_height = max(1, int(round(original_height * transform.scale)))
    content = probability[
        transform.pad_y:transform.pad_y + content_height,
        transform.pad_x:transform.pad_x + content_width,
    ]
    if content.shape != (content_height, content_width):
        raise MaskPostprocessError("V8 mask letterbox content is incomplete")
    restored = cv2.resize(
        content,
        (original_width, original_height),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32, copy=False)
    if (
        restored.shape != (original_height, original_width)
        or not np.isfinite(restored).all()
        or np.any(restored < 0.0)
        or np.any(restored > 1.0)
    ):
        raise MaskPostprocessError("restored V8 mask probability is invalid")
    return restored


def _polygon_mask(size: tuple[int, int], corners: np.ndarray) -> np.ndarray:
    width, height = size
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.rint(corners).astype(np.int32), 1, lineType=cv2.LINE_8)
    return mask


def _mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = int(np.logical_and(left, right).sum())
    union = int(np.logical_or(left, right).sum())
    return float(intersection / union) if union else 0.0


def _fit_quad(component: np.ndarray) -> tuple[np.ndarray, float, str]:
    contours, _hierarchy = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise MaskPostprocessError("foreground component has no contour")
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 4 or cv2.contourArea(contour) <= 1:
        raise MaskPostprocessError("foreground contour is degenerate")
    hull = cv2.convexHull(contour)
    perimeter = float(cv2.arcLength(hull, True))
    candidates: list[tuple[np.ndarray, str]] = []
    for epsilon_ratio in (0.002, 0.004, 0.007, 0.012, 0.02, 0.035, 0.05):
        approx = cv2.approxPolyDP(hull, epsilon_ratio * perimeter, True)
        if len(approx) == 4:
            candidates.append((approx[:, 0, :].astype(np.float64), f"approx_{epsilon_ratio:.3f}"))
    candidates.append((cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float64), "min_area_rect"))
    height, width = component.shape
    scored = []
    for points, method in candidates:
        try:
            ordered = np.asarray(order_quad(points), dtype=np.float64)
            validate_quad(
                ordered,
                (width, height),
                min_area_ratio=0.005,
                max_area_ratio=0.9999,
                min_edge_ratio=0.01,
            )
        except (GeometryError, TypeError, ValueError):
            continue
        score = _mask_iou(component, _polygon_mask((width, height), ordered))
        scored.append((score, method != "min_area_rect", ordered, method))
    if not scored:
        raise MaskPostprocessError("foreground cannot be fitted by a legal quadrilateral")
    score, _prefer_perspective, corners, method = max(scored, key=lambda item: (item[0], item[1]))
    return corners, float(score), method


def decode_photo_mask(
    outputs: Mapping[str, np.ndarray],
    transform: LetterboxTransform,
    *,
    image_size: Sequence[int],
    threshold: float = 0.5,
) -> PhotoMaskCandidate | None:
    """Decode one foreground mask into at most one legal physical-photo quad."""
    if not isinstance(transform, LetterboxTransform):
        raise MaskPostprocessError("transform must be a LetterboxTransform")
    if (
        not isinstance(image_size, (tuple, list))
        or len(image_size) != 2
        or any(type(value) is not int or value <= 0 for value in image_size)
    ):
        raise MaskPostprocessError("image_size must contain positive integers")
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool) or not 0.05 <= float(threshold) <= 0.95:
        raise MaskPostprocessError("threshold is out of bounds")
    probability = _probability(outputs, transform)
    binary = (probability >= float(threshold)).astype(np.uint8)
    original_width, original_height = transform.original_size
    content_width = max(1, int(round(original_width * transform.scale)))
    content_height = max(1, int(round(original_height * transform.scale)))
    content = np.zeros_like(binary)
    content[
        transform.pad_y:transform.pad_y + content_height,
        transform.pad_x:transform.pad_x + content_width,
    ] = 1
    binary &= content
    kernel = np.ones((3, 3), dtype=np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.int64)
    main_index = int(np.argmax(areas)) + 1
    main_area = int(areas[main_index - 1])
    valid_area = max(1, content_width * content_height)
    if main_area / valid_area < 0.01:
        return None
    component = (labels == main_index).astype(np.uint8)
    model_corners, polygon_iou, method = _fit_quad(component)
    try:
        original_corners = validate_quad(
            transform.to_original(model_corners),
            image_size,
            min_area_ratio=0.01,
            max_area_ratio=0.9999,
            min_edge_ratio=0.02,
        )
    except (GeometryError, TypeError, ValueError):
        return None
    main_probability = float(probability[component == 1].mean())
    component_ratio = float(main_area / max(1, int(areas.sum())))
    contour = cv2.morphologyEx(component, cv2.MORPH_GRADIENT, kernel).astype(bool)
    boundary_values = probability[contour]
    if boundary_values.size:
        clipped = np.clip(boundary_values.astype(np.float64), 1e-7, 1 - 1e-7)
        boundary_entropy = float(np.mean(-(clipped * np.log(clipped) + (1 - clipped) * np.log(1 - clipped))) / math.log(2))
    else:
        boundary_entropy = 1.0
    confidence = float(np.clip(main_probability * polygon_iou * component_ratio, 0.0, 1.0))
    distances = (
        model_corners[:, 0] - transform.pad_x,
        transform.pad_x + content_width - 1 - model_corners[:, 0],
        model_corners[:, 1] - transform.pad_y,
        transform.pad_y + content_height - 1 - model_corners[:, 1],
    )
    visible_margin = float(max(0.0, min(float(np.min(value)) for value in distances)) / max(1.0, min(content_width, content_height)))
    evidence = MappingProxyType({
        "threshold": float(threshold),
        "component_count": int(count - 1),
        "main_component_area": main_area,
        "main_component_ratio": component_ratio,
        "foreground_area_ratio": float(main_area / valid_area),
        "foreground_probability": main_probability,
        "polygon_mask_iou": polygon_iou,
        "quad_method": method,
        "boundary_entropy": boundary_entropy,
        "visible_margin": visible_margin,
        "model_corners": tuple(tuple(float(value) for value in point) for point in model_corners),
    })
    return PhotoMaskCandidate(original_corners, confidence, evidence)


__all__ = [
    "MaskPostprocessError",
    "PhotoMaskCandidate",
    "decode_photo_mask",
    "restore_photo_mask_probability",
]
