"""Bounded post-processing for optional document-quadrilateral models."""
from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from .geometry import GeometryError, order_quad, validate_quad
from .model_inference import LetterboxTransform


class ModelOutputError(ValueError):
    pass


@dataclass(frozen=True)
class ModelQuadProposal:
    head: str
    corners: tuple[tuple[float, float], ...]
    confidence: float
    evidence: Mapping[str, Any]


def _array(outputs: Mapping[str, np.ndarray], name: str, shape: tuple[int, ...]) -> np.ndarray:
    value = outputs.get(name)
    if not isinstance(value, np.ndarray) or value.shape != shape:
        raise ModelOutputError(f"{name} must have shape {shape}")
    if not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
        raise ModelOutputError(f"{name} must contain finite numeric data")
    return value.astype(np.float32, copy=False)


def _refined_peak(heatmap: np.ndarray) -> tuple[float, float, float, float]:
    flat_index = int(np.argmax(heatmap))
    y, x = np.unravel_index(flat_index, heatmap.shape)
    y0, y1 = max(0, y - 2), min(heatmap.shape[0], y + 3)
    x0, x1 = max(0, x - 2), min(heatmap.shape[1], x + 3)
    window = heatmap[y0:y1, x0:x1].astype(np.float64)
    weights = np.exp(np.clip(window - float(np.max(window)), -40.0, 0.0))
    total = float(weights.sum())
    yy, xx = np.mgrid[y0:y1, x0:x1]
    refined_x = float((weights * xx).sum() / total)
    refined_y = float((weights * yy).sum() / total)
    mean = float(np.mean(heatmap, dtype=np.float64))
    std = float(np.std(heatmap, dtype=np.float64))
    peak = float(heatmap[y, x])
    sigma = (peak - mean) / std if std > 1e-9 else 0.0
    return refined_x, refined_y, peak, sigma


def _legal(points: Sequence[Sequence[float]], image_size: Sequence[float]):
    try:
        return validate_quad(order_quad(points), image_size, min_area_ratio=0.005,
                             max_area_ratio=0.999, min_edge_ratio=0.01)
    except (GeometryError, TypeError, ValueError):
        return None


def _mask_quad(mask_logits: np.ndarray, transform: LetterboxTransform,
               image_size: Sequence[float]):
    mask = mask_logits[0, 0] > 0.0
    ys, xs = np.nonzero(mask)
    if len(xs) < 4:
        return None, float(mask.mean())
    points = np.column_stack(((xs.astype(np.float32) + 0.5) * 4.0,
                              (ys.astype(np.float32) + 0.5) * 4.0))
    rect = cv2.minAreaRect(points)
    model_quad = cv2.boxPoints(rect)
    original = transform.to_original(model_quad)
    return _legal(original, image_size), float(mask.mean())


def _corner_distance(a, b, image_size) -> float | None:
    if a is None or b is None:
        return None
    width, height = map(float, image_size)
    diagonal = max(1e-9, math.hypot(width, height))
    return max(math.hypot(ax - bx, ay - by) for (ax, ay), (bx, by) in zip(a, b)) / diagonal


def decode_docquad_outputs(outputs: Mapping[str, np.ndarray], transform: LetterboxTransform,
                           image_size: Sequence[float]) -> tuple[ModelQuadProposal, ...]:
    if not isinstance(outputs, Mapping):
        raise ModelOutputError("model outputs must be a mapping")
    heatmaps = _array(outputs, "corner_heatmaps", (1, 4, 64, 64))
    mask_logits = _array(outputs, "mask_logits", (1, 1, 64, 64))
    model_points = []
    peaks = []
    sigmas = []
    for channel in range(4):
        x, y, peak, sigma = _refined_peak(heatmaps[0, channel])
        model_points.append(((x + 0.5) * 4.0, (y + 0.5) * 4.0))
        peaks.append(peak)
        sigmas.append(sigma)
    corner_quad = _legal(transform.to_original(model_points), image_size)
    mask_quad, mask_area = _mask_quad(mask_logits, transform, image_size)
    distance = _corner_distance(corner_quad, mask_quad, image_size)
    diffuse = min(sigmas) < 5.0
    common = {
        "corner_peaks": tuple(float(value) for value in peaks),
        "corner_peak_sigmas": tuple(float(value) for value in sigmas),
        "min_peak_sigma": float(min(sigmas)),
        "diffuse_heatmaps": bool(diffuse),
        "mask_available": mask_quad is not None,
        "mask_area_fraction": float(mask_area),
        "corner_mask_distance": None if distance is None else float(distance),
    }
    proposals = []
    if corner_quad is not None:
        confidence = max(0.0, min(1.0, min(sigmas) / 12.0))
        proposals.append(ModelQuadProposal("corners", corner_quad, confidence,
                                           MappingProxyType(dict(common))))
    if mask_quad is not None:
        mask_confidence = max(0.0, min(1.0, mask_area * 4.0))
        proposals.append(ModelQuadProposal("mask", mask_quad, mask_confidence,
                                           MappingProxyType(dict(common))))
    return tuple(proposals[:2])


def decode_docaligner_outputs(outputs: Mapping[str, np.ndarray], transform: LetterboxTransform,
                              image_size: Sequence[float]) -> tuple[ModelQuadProposal, ...]:
    if not isinstance(outputs, Mapping):
        raise ModelOutputError("model outputs must be a mapping")
    heatmaps = outputs.get("heatmap")
    if (not isinstance(heatmaps, np.ndarray) or heatmaps.ndim != 4 or
            heatmaps.shape[0] != 1 or heatmaps.shape[1] != 4 or
            heatmaps.shape[2] < 2 or heatmaps.shape[3] < 2):
        raise ModelOutputError("heatmap must have shape [1,4,H,W]")
    if not np.issubdtype(heatmaps.dtype, np.number) or not np.isfinite(heatmaps).all():
        raise ModelOutputError("heatmap must contain finite numeric data")
    _, _, map_h, map_w = heatmaps.shape
    model_w, model_h = transform.model_size
    model_points = []
    peaks = []
    sigmas = []
    support_pixels = []
    for channel in range(4):
        heatmap = heatmaps[0, channel].astype(np.float32, copy=False)
        binary = (heatmap >= 0.3).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if count <= 1:
            return ()
        component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        component_mask = labels == component
        support = int(component_mask.sum())
        weights = np.where(component_mask, np.maximum(heatmap, 0.0), 0.0).astype(np.float64)
        total = float(weights.sum())
        if support <= 0 or total <= 1e-12:
            return ()
        yy, xx = np.indices(heatmap.shape)
        cx = float((weights * xx).sum() / total)
        cy = float((weights * yy).sum() / total)
        model_points.append(((cx + 0.5) * model_w / map_w,
                             (cy + 0.5) * model_h / map_h))
        peak = float(np.max(heatmap))
        mean = float(np.mean(heatmap, dtype=np.float64))
        std = float(np.std(heatmap, dtype=np.float64))
        peaks.append(peak)
        sigmas.append((peak - mean) / std if std > 1e-9 else 0.0)
        support_pixels.append(support)
    quad = _legal(transform.to_original(model_points), image_size)
    if quad is None:
        return ()
    confidence = max(0.0, min(1.0, min(sigmas) / 12.0))
    evidence = MappingProxyType({
        "corner_peaks": tuple(peaks),
        "corner_peak_sigmas": tuple(sigmas),
        "min_peak_sigma": float(min(sigmas)),
        "corner_support_pixels": tuple(support_pixels),
        "diffuse_heatmaps": bool(min(sigmas) < 5.0),
    })
    return (ModelQuadProposal("heatmaps", quad, confidence, evidence),)
