"""Deterministic Top-K quadrilateral edge refinement with safe rollback.

The refiner deliberately operates on the shared :class:`ImageFeatureContext`.
It samples a bounded, normalized band around each candidate side, keeps pixels
whose local gradient agrees with that side's normal, and performs a small fixed
number of trimmed ``cv2.fitLine`` passes.  Refinement is only adopted after all
four sides pass support, conditioning, displacement, geometry and evidence
gates.  A failed gate never leaks a partial quad to callers.
"""
from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
import time
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

import cv2
import numpy as np

from .features import FeatureCancelled, ImageFeatureContext
from .geometry import GeometryError, line_intersection, validate_quad
from .parameters import V7Parameters
from .types import CandidateAudit


_MAX_SAMPLES = 256
_MIN_SUPPORT = 8
_TRIM_PASSES = 3
_EVIDENCE_TOLERANCE = 0.05
_FIT_CONDITION_TOLERANCE = 1e-4


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    return value if math.isfinite(value) else float(default)


def _cancelled(token: Any) -> bool:
    if token is None:
        return False
    method = getattr(token, "is_cancelled", None)
    if method is not None:
        try:
            return bool(method() if callable(method) else method)
        except (TypeError, ValueError):
            return False
    value = getattr(token, "cancelled", None)
    if value is not None:
        try:
            return bool(value() if callable(value) else value)
        except (TypeError, ValueError):
            return False
    value = getattr(token, "is_set", None)
    if value is not None:
        try:
            return bool(value() if callable(value) else value)
        except (TypeError, ValueError):
            return False
    return bool(token) if isinstance(token, bool) else False


def _check(token: Any, deadline: Any) -> None:
    if _cancelled(token):
        raise FeatureCancelled("refinement cancelled")
    if deadline is None:
        return
    for attr in ("is_expired", "expired"):
        marker = getattr(deadline, attr, None)
        if marker is not None:
            try:
                if bool(marker() if callable(marker) else marker):
                    raise TimeoutError("refinement deadline exceeded")
                return
            except (TypeError, ValueError):
                pass
    value = deadline() if callable(deadline) else deadline
    if isinstance(value, bool):
        if value:
            raise TimeoutError("refinement deadline exceeded")
        return
    try:
        value = float(value)
    except (TypeError, ValueError):
        return
    if value <= 0 or value <= time.monotonic():
        raise TimeoutError("refinement deadline exceeded")


def _quad_from(value: Any) -> tuple[tuple[float, float], ...]:
    if isinstance(value, Mapping):
        for key in ("corners", "pre_topk_corners", "original_legal_corners"):
            if value.get(key) is not None:
                value = value[key]
                break
    if value is None:
        raise ValueError("candidate corners are required")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (tuple, list)) or len(value) != 4:
        raise ValueError("candidate corners must contain four points")
    return tuple((float(point[0]), float(point[1])) for point in value)


def _context(value: Any) -> ImageFeatureContext:
    if isinstance(value, ImageFeatureContext):
        return value
    if hasattr(value, "normalized_bgr"):
        return ImageFeatureContext.from_loaded(value)
    return ImageFeatureContext(np.asarray(value))


def _local_gradient(gray: np.ndarray, x: int, y: int) -> tuple[float, float, float]:
    """A bounded local derivative; no full-image feature is rebuilt here."""
    height, width = gray.shape[:2]
    x0, x1 = max(0, x - 1), min(width - 1, x + 1)
    y0, y1 = max(0, y - 1), min(height - 1, y + 1)
    gx = (float(gray[y, x1]) - float(gray[y, x0])) * 0.5
    gy = (float(gray[y1, x]) - float(gray[y0, x])) * 0.5
    magnitude = math.hypot(gx, gy)
    return gx, gy, magnitude


def _edge_points(
    gray: np.ndarray,
    gradient: np.ndarray,
    edge: Sequence[Sequence[float]],
    centroid: Sequence[float],
    params: V7Parameters,
    *,
    gradient_stats: tuple[float, float] | None = None,
    token: Any = None,
    deadline: Any = None,
) -> tuple[np.ndarray, Mapping[str, float]]:
    """Collect one deterministic best-supported point per longitudinal sample."""
    p0 = np.asarray(edge[0], dtype=float)
    p1 = np.asarray(edge[1], dtype=float)
    delta = p1 - p0
    length = float(np.linalg.norm(delta))
    if not math.isfinite(length) or length <= 1e-6:
        return np.empty((0, 2), dtype=np.float32), MappingProxyType({"support": 0.0, "support_count": 0, "samples": 0})
    tangent = delta / length
    normal = np.asarray((-tangent[1], tangent[0]), dtype=float)
    # Orienting the normal toward the interior is useful for ties, but both
    # gradient signs remain compatible (photographs may be lighter or darker).
    midpoint = (p0 + p1) * 0.5
    if float(np.dot(np.asarray(centroid, dtype=float) - midpoint, normal)) < 0:
        normal = -normal
    diagonal = math.hypot(gray.shape[1], gray.shape[0])
    band = max(2.0, float(params.edge_band) * diagonal)
    offsets = np.linspace(-band, band, int(max(5, min(25, round(band * 2.0 + 1.0)))), dtype=float)
    samples = int(max(24, min(_MAX_SAMPLES, round(length / max(1.0, diagonal / 120.0)))))
    samples = min(samples, _MAX_SAMPLES)
    if gradient_stats is None:
        mean_grad = float(np.mean(gradient))
        p70 = float(np.percentile(gradient, 70)) if gradient.size else 0.0
    else:
        mean_grad, p70 = (float(gradient_stats[0]), float(gradient_stats[1]))
    threshold = max(2.0, p70 * 0.42, mean_grad * 1.25)
    selected: list[tuple[float, float, float, float]] = []
    height, width = gray.shape[:2]
    for index in range(samples):
        _check(token, deadline)
        t = 0.0 if samples == 1 else index / float(samples - 1)
        base = p0 + delta * t
        best: tuple[float, float, float, float] | None = None
        for offset in offsets:
            point = base + normal * float(offset)
            x, y = int(round(point[0])), int(round(point[1]))
            if x < 1 or y < 1 or x >= width - 1 or y >= height - 1:
                continue
            gx, gy, magnitude = _local_gradient(gray, x, y)
            if magnitude < threshold:
                continue
            alignment = abs((gx * normal[0] + gy * normal[1]) / (magnitude + 1e-9))
            if alignment < 0.40:
                continue
            # Edge-map presence is a small tie-breaker, never a hard gate:
            # broken edges can still be recovered from the cached gradient.
            edge_bonus = 0.10 if magnitude > 0 and gradient[y, x] > p70 else 0.0
            quality = float(alignment * min(1.0, magnitude / (p70 + 1e-6)) + edge_bonus)
            candidate = (quality, float(magnitude), float(x), float(y))
            if best is None or candidate > best:
                best = candidate
        if best is not None:
            selected.append(best)
    # Sort by longitudinal position and then coordinates before fitting.  This
    # removes dependence on OpenCV's internal traversal order.
    selected.sort(key=lambda item: (round((item[2] - p0[0]) * tangent[0] + (item[3] - p0[1]) * tangent[1], 6), round(item[2], 4), round(item[3], 4)))
    points = np.asarray([(item[2], item[3]) for item in selected], dtype=np.float32)
    support_ratio = len(selected) / float(max(1, samples))
    mean_alignment = float(np.mean([item[0] for item in selected])) if selected else 0.0
    mean_magnitude = float(np.mean([item[1] for item in selected])) if selected else 0.0
    evidence = MappingProxyType({
        "support": float(np.clip(support_ratio, 0.0, 1.0)),
        "support_count": int(len(selected)),
        "samples": int(samples),
        "band_normalized": float(params.edge_band),
        "direction": float(np.clip(mean_alignment, 0.0, 1.0)),
        "gradient": float(np.clip(mean_magnitude / (p70 + 1e-6), 0.0, 1.0)),
    })
    return points, evidence


def _fit_line(points: np.ndarray, params: V7Parameters, diagonal: float) -> tuple[tuple[float, float, float, float] | None, Mapping[str, float], str | None]:
    if len(points) < _MIN_SUPPORT:
        return None, MappingProxyType({"support": float(len(points)), "residual": float("inf")}), "insufficient_support"
    work = np.asarray(points, dtype=np.float32)
    threshold = max(1.0, float(params.refinement_residual) * diagonal)
    residual_value = float("inf")
    for _ in range(_TRIM_PASSES):
        if len(work) < _MIN_SUPPORT:
            return None, MappingProxyType({"support": float(len(work)), "residual": residual_value}), "insufficient_support"
        try:
            fit = cv2.fitLine(work, cv2.DIST_L2, 0.0, 0.01, 0.01)
        except (cv2.error, ValueError, TypeError):
            return None, MappingProxyType({"support": float(len(work)), "residual": residual_value}), "fit_failure"
        values = np.asarray(fit, dtype=float).reshape(-1)
        if len(values) != 4 or not np.all(np.isfinite(values)):
            return None, MappingProxyType({"support": float(len(work)), "residual": residual_value}), "fit_failure"
        vx, vy, x0, y0 = (float(v) for v in values)
        norm = math.hypot(vx, vy)
        if norm <= 1e-9:
            return None, MappingProxyType({"support": float(len(work)), "residual": residual_value}), "fit_failure"
        vx, vy = vx / norm, vy / norm
        distances = np.abs((work[:, 0] - x0) * vy - (work[:, 1] - y0) * vx)
        residual_value = float(np.median(distances)) if len(distances) else float("inf")
        keep = distances <= threshold
        if int(np.count_nonzero(keep)) < _MIN_SUPPORT:
            return None, MappingProxyType({"support": float(np.count_nonzero(keep)), "residual": residual_value}), "insufficient_support"
        new_work = work[keep]
        if len(new_work) == len(work):
            work = new_work
            break
        work = new_work
    return (vx, vy, x0, y0), MappingProxyType({"support": float(len(work)), "residual": float(residual_value)}), None


def _aggregate(edge_evidence: Sequence[Mapping[str, Any]]) -> float:
    values = []
    for edge in edge_evidence:
        support = _finite(edge.get("support"))
        direction = _finite(edge.get("direction"), 0.0)
        gradient = _finite(edge.get("gradient"), 0.0)
        values.append(float(np.clip(support, 0.0, 1.0) * 0.5 + np.clip(direction, 0.0, 1.0) * 0.25 + np.clip(gradient, 0.0, 1.0) * 0.25))
    return float(np.mean(values)) if values else 0.0


def _max_edge_ratio_ok(quad: Sequence[Sequence[float]], image_size: Sequence[float], maximum: float) -> bool:
    diagonal = math.hypot(float(image_size[0]), float(image_size[1]))
    if diagonal <= 0 or not math.isfinite(diagonal):
        return False
    lengths = [math.hypot(float(quad[(index + 1) % 4][0]) - float(quad[index][0]), float(quad[(index + 1) % 4][1]) - float(quad[index][1])) for index in range(4)]
    return bool(lengths) and max(lengths) / diagonal <= float(maximum) + 1e-12


@dataclass(frozen=True)
class RefinementResult:
    original_corners: tuple[tuple[float, float], ...]
    proposed_corners: tuple[tuple[float, float], ...] | None
    adopted_corners: tuple[tuple[float, float], ...]
    adopted: bool
    risks: tuple[str, ...] = ()
    edge_evidence: tuple[Mapping[str, Any], ...] = ()
    proposed_edge_evidence: tuple[Mapping[str, Any], ...] = ()
    max_normalized_shift: float = 0.0
    evidence: Mapping[str, Any] = MappingProxyType({})
    candidate_id: str = "refined"
    sources: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "risks", tuple(str(v) for v in self.risks))
        object.__setattr__(self, "edge_evidence", tuple(MappingProxyType(dict(v)) for v in self.edge_evidence))
        object.__setattr__(self, "proposed_edge_evidence", tuple(MappingProxyType(dict(v)) for v in self.proposed_edge_evidence))
        object.__setattr__(self, "evidence", MappingProxyType(dict(self.evidence)))
        object.__setattr__(self, "sources", tuple(str(v) for v in self.sources))

    @property
    def original(self):
        return self.original_corners

    @property
    def proposed(self):
        return self.proposed_corners

    @property
    def adopted_refined_corners(self):
        return self.adopted_corners if self.adopted else None

    def to_audit(self, candidate_id: str | None = None, sources: Sequence[str] | None = None) -> CandidateAudit:
        cid = str(candidate_id or self.candidate_id)
        src = tuple(str(v) for v in (sources if sources is not None else self.sources)) or ("refinement",)
        decisions = self.risks or ("none",)
        evidence = {
            "refinement": dict(self.evidence),
            "edge_support": tuple(dict(v) for v in self.edge_evidence),
            "proposed_edge_support": tuple(dict(v) for v in self.proposed_edge_evidence),
        }
        return CandidateAudit(
            candidate_id=cid, sources=src,
            original_legal_corners=self.original_corners,
            pre_topk_corners=self.original_corners,
            proposed_refined_corners=self.proposed_corners,
            adopted_refined_corners=self.adopted_corners if self.adopted else self.original_corners,
            pre_truncation_risk_decisions=decisions,
            pre_truncation_risk_evidence=evidence,
            stage_scores={"refinement": dict(self.evidence)}, stage_ranks={},
        )


def refine_quad(
    context_or_image: Any,
    candidate_or_corners: Any,
    params: V7Parameters | None = None,
    cancellation_token: Any = None,
    deadline: Any = None,
    *,
    image_size: Sequence[float] | None = None,
    cancel_token: Any = None,
) -> RefinementResult:
    """Refine one Top-K candidate and safely return original/proposed/adopted quads."""
    params = params or V7Parameters()
    token = cancellation_token if cancellation_token is not None else cancel_token
    candidate_id = str(candidate_or_corners.get("candidate_id", candidate_or_corners.get("id", "refined"))) if isinstance(candidate_or_corners, Mapping) else "refined"
    sources_value = candidate_or_corners.get("sources", candidate_or_corners.get("source", ())) if isinstance(candidate_or_corners, Mapping) else ()
    if isinstance(sources_value, str):
        sources_value = (sources_value,)
    sources = tuple(str(v) for v in (sources_value or ()))
    context = _context(context_or_image)
    raw = _quad_from(candidate_or_corners)
    source_h, source_w = context.shape[:2]
    size = (float(image_size[0]), float(image_size[1])) if image_size is not None else (float(source_w), float(source_h))
    try:
        original = validate_quad(raw, size, min_area_ratio=params.min_area_ratio, max_area_ratio=params.max_area_ratio, min_edge_ratio=params.min_edge_ratio)
        if not _max_edge_ratio_ok(original, size, params.max_edge_ratio):
            raise GeometryError("quadrilateral has a too-long normalized edge")
    except (GeometryError, ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"candidate corners must be legal before refinement: {exc}") from exc
    risks: list[str] = []
    try:
        _check(token, deadline)
        features = context.original_resolution_features(params.sha256())
        gray = np.asarray(features["gray"])
        gradient = np.asarray(features["gradient"])
        mean_value = features.get("gradient_mean")
        p70_value = features.get("gradient_p70")
        if mean_value is None:
            mean_value = np.mean(gradient)
        if p70_value is None:
            p70_value = np.percentile(gradient, 70)
        gradient_stats = (float(mean_value), float(p70_value))
        centroid = tuple(np.mean(np.asarray(original, dtype=float), axis=0))
        baseline_edges: list[Mapping[str, Any]] = []
        points_per_edge: list[np.ndarray] = []
        for index in range(4):
            _check(token, deadline)
            points, evidence = _edge_points(gray, gradient, (original[index], original[(index + 1) % 4]), centroid, params, gradient_stats=gradient_stats, token=token, deadline=deadline)
            baseline_edges.append(dict(evidence))
            points_per_edge.append(points)
        fits: list[tuple[float, float, float, float]] = []
        fit_evidence: list[Mapping[str, Any]] = []
        diagonal = math.hypot(size[0], size[1])
        for points in points_per_edge:
            _check(token, deadline)
            fit, evidence, reason = _fit_line(points, params, diagonal)
            fit_evidence.append(dict(evidence))
            if reason is not None:
                risks.append(reason)
            if fit is None:
                fits = []
                break
            fits.append(fit)
        for index, fitted in enumerate(fit_evidence):
            if index < len(baseline_edges):
                residual = _finite(fitted.get("residual"), -1.0)
                baseline_edges[index] = {**dict(baseline_edges[index]), "fit_support": _finite(fitted.get("support")), "fit_residual": residual}
        proposed: tuple[tuple[float, float], ...] | None = None
        proposed_edges: list[Mapping[str, Any]] = []
        if len(fits) == 4:
            intersections: list[tuple[float, float]] = []
            for index in range(4):
                _check(token, deadline)
                previous, current = fits[(index - 1) % 4], fits[index]
                try:
                    # A normalized determinant test gives a clear, stable gate
                    # before the geometry helper computes a potentially huge
                    # intersection for almost parallel fits.
                    determinant = abs(previous[0] * current[1] - previous[1] * current[0])
                    if determinant <= _FIT_CONDITION_TOLERANCE:
                        raise GeometryError("near-parallel adjacent fits")
                    intersections.append(line_intersection((previous[2], previous[3]), (previous[2] + previous[0], previous[3] + previous[1]), (current[2], current[3]), (current[2] + current[0], current[3] + current[1]), tolerance=_FIT_CONDITION_TOLERANCE))
                except (GeometryError, ValueError, TypeError, OverflowError):
                    risks.append("near_parallel_intersection")
                    intersections = []
                    break
            if len(intersections) == 4:
                try:
                    proposed = validate_quad(intersections, size, min_area_ratio=params.min_area_ratio, max_area_ratio=params.max_area_ratio, min_edge_ratio=params.min_edge_ratio)
                    if not _max_edge_ratio_ok(proposed, size, params.max_edge_ratio):
                        raise GeometryError("quadrilateral has a too-long normalized edge")
                except (GeometryError, ValueError, TypeError, OverflowError):
                    risks.append("geometry_invalid")
                    proposed = None
        if proposed is not None:
            proposed_centroid = tuple(np.mean(np.asarray(proposed, dtype=float), axis=0))
            for index in range(4):
                _check(token, deadline)
                _, evidence = _edge_points(gray, gradient, (proposed[index], proposed[(index + 1) % 4]), proposed_centroid, params, gradient_stats=gradient_stats, token=token, deadline=deadline)
                proposed_edges.append(dict(evidence))
            shift = max(float(np.linalg.norm(np.asarray(a) - np.asarray(b))) / max(diagonal, 1e-9) for a, b in zip(original, proposed))
        else:
            shift = 0.0
        baseline_score = _aggregate(baseline_edges)
        proposed_score = _aggregate(proposed_edges) if proposed_edges else 0.0
        adopted = proposed is not None
        if proposed is None:
            adopted = False
        if proposed is not None and shift > float(params.max_refinement_shift) + 1e-12:
            risks.append("excessive_shift")
            adopted = False
        if proposed is not None and proposed_score + _EVIDENCE_TOLERANCE < baseline_score:
            risks.append("edge_evidence_degraded")
            adopted = False
        if any(float(edge.get("support", 0.0)) < (_MIN_SUPPORT / max(1.0, float(edge.get("samples", _MIN_SUPPORT)))) for edge in baseline_edges):
            if "insufficient_support" not in risks:
                risks.append("insufficient_support")
            adopted = False
        adopted_corners = proposed if adopted and proposed is not None else original
        evidence = {
            "adopted": bool(adopted), "baseline_edge_score": float(np.clip(baseline_score, 0.0, 1.0)),
            "proposed_edge_score": float(np.clip(proposed_score, 0.0, 1.0)),
            "max_normalized_shift": float(np.clip(shift, 0.0, 1.0)),
            "edge_evidence_tolerance": float(_EVIDENCE_TOLERANCE),
            "risk_reasons": tuple(dict.fromkeys(risks)),
            "fit_count": int(len(fits)),
        }
        return RefinementResult(original, proposed, adopted_corners, adopted, tuple(dict.fromkeys(risks)), tuple(baseline_edges), tuple(proposed_edges), float(shift), evidence, candidate_id, sources)
    except FeatureCancelled:
        return RefinementResult(original, None, original, False, ("refinement_cancelled",), (), (), 0.0, {"adopted": False, "risk_reasons": ("refinement_cancelled",)}, candidate_id, sources)
    except TimeoutError:
        return RefinementResult(original, None, original, False, ("refinement_timeout",), (), (), 0.0, {"adopted": False, "risk_reasons": ("refinement_timeout",)}, candidate_id, sources)


def refine_candidate(
    candidate_or_context: Any,
    context_or_candidate: Any,
    params: V7Parameters | None = None,
    **kwargs: Any,
) -> RefinementResult:
    """Compatibility wrapper accepting either ``(context, candidate)`` or
    ``(candidate, context)``.  The canonical implementation is ``refine_quad``.
    """
    looks_like_context = isinstance(candidate_or_context, ImageFeatureContext) or hasattr(candidate_or_context, "normalized_bgr")
    if isinstance(candidate_or_context, np.ndarray):
        looks_like_context = candidate_or_context.ndim == 3 and candidate_or_context.shape[2] == 3
    if looks_like_context:
        return refine_quad(candidate_or_context, context_or_candidate, params, **kwargs)
    return refine_quad(context_or_candidate, candidate_or_context, params, **kwargs)


refine_edges = refine_candidate


def refine_topk(
    context_or_image: Any,
    candidates: Sequence[Any],
    params: V7Parameters | None = None,
    cancellation_token: Any = None,
    deadline: Any = None,
    *,
    cancel_token: Any = None,
) -> tuple[RefinementResult, ...]:
    """Refine at most ``params.refined_budget`` candidates in stable order."""
    params = params or V7Parameters()
    output: list[RefinementResult] = []
    for candidate in itertools.islice(candidates, int(params.refined_budget)):
        result = refine_quad(context_or_image, candidate, params, cancellation_token=cancellation_token, cancel_token=cancel_token, deadline=deadline)
        output.append(result)
    return tuple(output)


refine_topk_candidates = refine_topk


__all__ = ["RefinementResult", "refine_quad", "refine_candidate", "refine_edges", "refine_topk", "refine_topk_candidates"]
