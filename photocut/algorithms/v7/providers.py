"""Deterministic, budgeted region candidate providers for v7.

Providers deliberately operate on :class:`ImageFeatureContext` rather than an
image array.  The context owns colour conversion and resizing; this module only
consumes cached features and converts each work-scale contour to original
coordinates once.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import time
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence

import cv2
import numpy as np

from .features import FeatureCancelled, ImageFeatureContext
from .geometry import (
    GeometryError, line_intersection, normalized_corner_distance, order_quad,
    to_original_scale, validate_quad,
)
from .parameters import V7Parameters
from .types import ProviderResult, ProviderStatus


def _cancelled(token: Any) -> bool:
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
    if value:
        return bool(value() if callable(value) else value)
    value = getattr(token, "is_set", False)
    return bool(value() if callable(value) else value)


def _expired(deadline: Any) -> bool:
    if deadline is None:
        return False
    for attr in ("is_expired", "expired"):
        marker = getattr(deadline, attr, None)
        if marker is not None:
            try:
                return bool(marker() if callable(marker) else marker)
            except TypeError:
                pass
    value = deadline() if callable(deadline) else deadline
    if isinstance(value, bool):
        return value
    # Deadlines are monotonic absolute timestamps.  Zero/negative is an
    # already-expired deadline; callers that need a duration must convert it
    # with ``time.monotonic() + duration`` before invoking a provider.
    try:
        value = float(value)
    except (TypeError, ValueError):
        return False
    if value <= 0:
        return True
    now = time.monotonic()
    return value <= now


class ProviderBudgetExceeded(RuntimeError):
    pass


class ProviderDeadlineExceeded(RuntimeError):
    pass


class _Budget:
    def __init__(self, provider: str, params: V7Parameters, token: Any, deadline: Any):
        self.provider = provider
        self.limit = int(params.provider_work_limits.get(provider, 120_000))
        self.token = token
        self.deadline = deadline
        self.used = 0

    def check(self, units: int = 1) -> None:
        units = max(0, int(units))
        if _cancelled(self.token):
            raise FeatureCancelled("provider cancelled")
        if _expired(self.deadline):
            raise ProviderDeadlineExceeded("provider deadline exceeded")
        if self.used + units > self.limit:
            self.used = self.limit
            raise ProviderBudgetExceeded("provider work budget exhausted")
        self.used += units


def _stable_id(provider: str, source: str, scale: Any, corners: Sequence[Sequence[float]], variant: Any = None) -> str:
    payload = json.dumps([provider, source, scale, variant, [[round(float(x), 6), round(float(y), 6)] for x, y in corners]],
                         sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _candidate(provider: str, source: str, corners: Sequence[Sequence[float]], *, scale: Any,
               evidence: Mapping[str, Any], score: float = 0.0, variant: Any = None) -> Mapping[str, Any]:
    quad = tuple((float(x), float(y)) for x, y in corners)
    cid = _stable_id(provider, source, scale, quad, variant)
    ev = dict(evidence)
    return MappingProxyType({
        "id": cid, "candidate_id": cid, "corners": quad,
        "sources": (source,), "source": source, "evidence": MappingProxyType(ev),
        "generation_scale": scale, "score": float(max(0.0, min(1.0, score))),
    })


def _finish(name: str, started: float, budget: _Budget, candidates: Sequence[Mapping[str, Any]],
            status: ProviderStatus = ProviderStatus.SUCCESS, **kwargs: Any) -> ProviderResult:
    elapsed = (time.perf_counter() - started) * 1000.0
    return ProviderResult(name, tuple(candidates), status=status, elapsed_ms=elapsed,
                          work_consumed=min(budget.used, budget.limit), work_limit=budget.limit,
                          **kwargs)


class RegionProvider(Protocol):
    name: str
    def provide(self, context: ImageFeatureContext, params: V7Parameters,
                cancellation_token: Any = None, deadline: Any = None) -> ProviderResult: ...


# Public spelling used by the design document; RegionProvider remains the more
# descriptive alias for downstream code that has already imported it.
CandidateProvider = RegionProvider


def _run(provider: Any, context: ImageFeatureContext, params: V7Parameters,
         cancellation_token: Any, deadline: Any) -> ProviderResult:
    started = time.perf_counter()
    if deadline is None:
        timeout_ms = int(params.provider_timeout_ms.get(provider.name, 250))
        deadline = time.monotonic() + timeout_ms / 1000.0
    budget = _Budget(provider.name, params, cancellation_token, deadline)
    try:
        candidates = provider._generate(context, params, budget)
        budget.check(0)
        candidates = sorted(candidates, key=lambda c: (str(c["id"]), str(c.get("source", ""))))
        budget.check(0)
        status = ProviderStatus.SUCCESS if candidates else ProviderStatus.NO_CANDIDATE
        ambiguity = "none" if len(candidates) == 1 else ("multiple_candidates" if candidates else "no_primary")
        diagnostics = {"candidate_count": len(candidates), "primary_ambiguity": ambiguity}
        diagnostics.update(getattr(provider, "_last_diagnostics", {}) or {})
        return _finish(provider.name, started, budget, candidates, status,
                       diagnostics=diagnostics)
    except FeatureCancelled:
        return _finish(provider.name, started, budget, (), ProviderStatus.CANCELLED,
                       error_code="cancelled")
    except ProviderDeadlineExceeded:
        return _finish(provider.name, started, budget, (), ProviderStatus.TIMEOUT,
                       timeout_code="provider_deadline")
    except ProviderBudgetExceeded:
        return _finish(provider.name, started, budget, (), ProviderStatus.BUDGET_EXHAUSTED,
                       error_code="work_limit")
    except Exception as exc:  # provider isolation is part of the public contract
        return _finish(provider.name, started, budget, (), ProviderStatus.ERROR,
                       error_code=type(exc).__name__, diagnostics={"message": str(exc)})


def _scales(context: ImageFeatureContext, params: V7Parameters) -> tuple[int, ...]:
    edge = max(context.shape[:2])
    # Preserve deterministic order while avoiding duplicate cache work when a
    # small image already has an edge below the configured target.
    return tuple(dict.fromkeys(max(1, min(edge, int(s))) for s in params.work_edges))


def _border_connected(mask: np.ndarray) -> np.ndarray:
    """Keep only mask components connected to the image perimeter."""
    count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    if count <= 1:
        return np.zeros_like(mask, dtype=np.uint8)
    perimeter = np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1]))
    border_labels = np.unique(perimeter)
    border_labels = border_labels[border_labels != 0]
    if not len(border_labels):
        return np.zeros_like(mask, dtype=np.uint8)
    return np.isin(labels, border_labels).astype(np.uint8)


def _white_lab_reference(block: np.ndarray) -> np.ndarray | None:
    if not block.size:
        return None
    reference = np.median(block.reshape(-1, 3).astype(np.float32), axis=0)
    chroma = math.hypot(float(reference[1]) - 128.0, float(reference[2]) - 128.0)
    return reference if float(reference[0]) >= 190.0 and chroma <= 32.0 else None


class BackgroundDifferenceProvider:
    name = "background"

    def provide(self, context: ImageFeatureContext, params: V7Parameters,
                cancellation_token: Any = None, deadline: Any = None) -> ProviderResult:
        return _run(self, context, params, cancellation_token, deadline)

    generate = provide
    run = provide

    def _border_connected_generate(self, context: ImageFeatureContext,
                                   params: V7Parameters, budget: _Budget):
        scales = _scales(context, params)
        if not scales:
            return (), {"border_connected_raw_candidates": 0,
                        "border_connected_clusters": 0}
        scale = scales[0]
        budget.check(1)
        lab = context.lab(scale, params.sha256())
        height, width = lab.shape[:2]
        short = min(height, width)
        patch = max(3, int(round(short * 0.018)))
        band = max(2, int(round(short * 0.012)))
        blocks = (
            ("corner_tl", lab[:patch, :patch]),
            ("corner_tr", lab[:patch, -patch:]),
            ("corner_br", lab[-patch:, -patch:]),
            ("corner_bl", lab[-patch:, :patch]),
            ("edge_top", lab[:band, :]),
            ("edge_bottom", lab[-band:, :]),
            ("edge_left", lab[:, :band]),
            ("edge_right", lab[:, -band:]),
        )
        references: list[tuple[str, np.ndarray]] = []
        seen_references = set()
        for name, block in blocks:
            reference = _white_lab_reference(block)
            if reference is None:
                continue
            key = tuple(int(round(float(value) / 3.0)) for value in reference)
            if key in seen_references:
                continue
            seen_references.add(key)
            references.append((name, reference))
        if not references:
            return (), {"border_connected_references": 0,
                        "border_connected_raw_candidates": 0,
                        "border_connected_clusters": 0}

        kernel_size = max(3, int(round(short * 0.004)) | 1)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        image_area = float(width * height)
        original_size = (context.shape[1], context.shape[0])
        raw: list[dict[str, Any]] = []
        labf = lab.astype(np.float32)
        for reference_name, reference in references:
            budget.check(max(1, int(math.ceil(lab.size / 768.0))))
            delta = labf - reference[None, None, :]
            distance = np.sqrt(
                (delta[..., 0] * 0.55) ** 2 + delta[..., 1] ** 2 + delta[..., 2] ** 2
            )
            for threshold in (6.0, 10.0, 16.0, 24.0):
                budget.check(max(1, int(math.ceil(distance.size / 1024.0))))
                connected = _border_connected(distance <= threshold)
                foreground = (1 - connected) * 255
                foreground = cv2.morphologyEx(
                    foreground, cv2.MORPH_CLOSE, kernel
                )
                contours, _ = cv2.findContours(
                    foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                background_ratio = float(np.count_nonzero(connected)) / image_area
                ordered_contours = sorted(
                    contours,
                    key=lambda contour: (
                        -cv2.contourArea(contour),
                        tuple(int(value) for value in contour.reshape(-1, 2)[0]),
                    ),
                )[:2]
                for contour_index, contour in enumerate(ordered_contours):
                    budget.check(1)
                    area_ratio = float(cv2.contourArea(contour)) / image_area
                    if not max(0.18, params.min_area_ratio) <= area_ratio <= params.max_area_ratio:
                        continue
                    perimeter = float(cv2.arcLength(contour, True))
                    variants = []
                    for epsilon_ratio in (0.008, 0.015):
                        approx = cv2.approxPolyDP(
                            contour, max(1.0, epsilon_ratio * perimeter), True
                        ).reshape(-1, 2)
                        if len(approx) == 4:
                            variants.append((f"approx_{epsilon_ratio}", approx))
                    variants.append((
                        "min_area_rect",
                        cv2.boxPoints(cv2.minAreaRect(cv2.convexHull(contour))),
                    ))
                    for variant, points in variants:
                        try:
                            work_quad = validate_quad(
                                order_quad([(float(x), float(y)) for x, y in points]),
                                (width, height),
                                min_area_ratio=max(0.18, params.min_area_ratio),
                                max_area_ratio=params.max_area_ratio,
                                min_edge_ratio=params.min_edge_ratio,
                            )
                        except (GeometryError, TypeError, ValueError):
                            continue
                        raw.append({
                            "corners": to_original_scale(
                                work_quad, (width, height), original_size
                            ),
                            "reference": reference_name,
                            "threshold": threshold,
                            "variant": variant,
                            "contour_index": contour_index,
                            "area_ratio": area_ratio,
                            "background_ratio": background_ratio,
                        })

        raw.sort(key=lambda item: (
            item["reference"], item["threshold"], item["variant"],
            item["contour_index"],
            tuple(round(value, 5) for point in item["corners"] for value in point),
        ))
        clusters: list[list[dict[str, Any]]] = []
        for item in raw:
            match = None
            for cluster in clusters:
                try:
                    distance = normalized_corner_distance(
                        item["corners"], cluster[0]["corners"], original_size
                    )
                except (GeometryError, TypeError, ValueError):
                    continue
                if distance <= params.dedup_distance:
                    match = cluster
                    break
            if match is None:
                clusters.append([item])
            else:
                match.append(item)

        stable = []
        for cluster in clusters:
            references_supported = sorted({item["reference"] for item in cluster})
            thresholds_supported = sorted({float(item["threshold"]) for item in cluster})
            representative = min(
                cluster,
                key=lambda candidate: (
                    sum(normalized_corner_distance(
                        candidate["corners"], other["corners"], original_size
                    ) for other in cluster),
                    candidate["reference"], candidate["threshold"],
                    candidate["variant"], candidate["contour_index"],
                ),
            )
            stable.append((
                len(references_supported), len(thresholds_supported), len(cluster),
                representative, references_supported, thresholds_supported,
            ))
        stable.sort(key=lambda item: (
            -item[0], -item[1], -item[2], -float(item[3]["area_ratio"]),
            item[3]["reference"], item[3]["threshold"], item[3]["variant"],
        ))

        candidates = []
        for cluster_index, (reference_support, threshold_support, support_count,
                            representative, reference_names, thresholds) in enumerate(stable[:3]):
            stability = min(1.0, 0.15 * reference_support +
                            0.12 * threshold_support + 0.04 * support_count)
            candidates.append(_candidate(
                self.name, "background:border_connected", representative["corners"],
                scale=scale,
                variant=(cluster_index, tuple(reference_names), tuple(thresholds)),
                score=stability,
                evidence={
                    "mask": "border_connected_lab",
                    "supplemental_only": True,
                    "reference_support": int(reference_support),
                    "threshold_support": int(threshold_support),
                    "raw_support_count": int(support_count),
                    "references": tuple(reference_names),
                    "thresholds": tuple(thresholds),
                    "area_ratio": float(representative["area_ratio"]),
                    "background_ratio": float(representative["background_ratio"]),
                    "representative_variant": representative["variant"],
                },
            ))
        return tuple(candidates), {
            "border_connected_references": len(references),
            "border_connected_raw_candidates": len(raw),
            "border_connected_clusters": len(clusters),
            "border_connected_emitted": len(candidates),
        }

    def _generate(self, context: ImageFeatureContext, params: V7Parameters, budget: _Budget):
        output: list[Mapping[str, Any]] = []
        for scale in _scales(context, params):
            budget.check(1)
            lab = context.lab(scale, params.sha256())
            budget.check(0)
            h, w = lab.shape[:2]
            # Several non-corner bands make the estimate robust to a dark/bright
            # corner mark.  Width is scale-normalized and never copied from a
            # fixed pixel constant.
            band = max(1, int(round(min(h, w) * 0.025)))
            side = max(1, int(round(min(h, w) * 0.18)))
            pieces = (lab[:band, side:w-side], lab[-band:, side:w-side],
                      lab[side:h-side, :band], lab[side:h-side, -band:])
            valid_pieces = [p.reshape(-1, 3) for p in pieces if p.size]
            if not valid_pieces:
                continue
            samples = np.concatenate(valid_pieces, axis=0)
            median = np.median(samples.astype(np.float32), axis=0)
            mad = np.median(np.abs(samples.astype(np.float32) - median), axis=0)
            # A fixed, parameterized robust threshold: channel noise gets 5 MAD
            # and a small floor so perfectly uniform scanner paper still works.
            threshold = np.maximum(10.0, 5.0 * 1.4826 * mad + 4.0)
            kernel_size = max(3, int(round(min(h, w) * 0.006)) | 1)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            labf = lab.astype(np.float32)
            channel_distance = np.abs(labf - median) / np.maximum(threshold, 1e-6)
            distances = np.max(channel_distance, axis=2)
            chroma_distance = np.linalg.norm(labf[..., 1:] - median[1:], axis=2) / max(1e-6, float(np.linalg.norm(threshold[1:])))
            masks = (
                ("lab_max", distances > 1.0),
                ("l_channel", channel_distance[..., 0] > 1.0),
                ("chroma", chroma_distance > 1.0),
            )
            # The Lab traversal is shared by all fixed mask variants.  Charge
            # pixels once per scale; each variant then pays only a small,
            # deterministic morphology/contour bookkeeping unit.
            budget.check(max(1, int(math.ceil(lab.size / 256.0))))
            for mask_variant, raw_mask in masks:
                budget.check(2)
                mask = raw_mask.astype(np.uint8) * 255
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                budget.check(0)
                if not contours:
                    continue
                image_area = float(w * h)
                work_size = (w, h)
                original_size = (context.shape[1], context.shape[0])
                for index, contour in enumerate(sorted(contours, key=lambda c: (-cv2.contourArea(c), tuple(c.reshape(-1, 2)[0])))):
                    budget.check(1)
                    area = float(cv2.contourArea(contour)) / image_area
                    if area < params.min_area_ratio or area > params.max_area_ratio:
                        continue
                    epsilon = max(1.0, 0.015 * cv2.arcLength(contour, True))
                    approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
                    if len(approx) != 4:
                        hull = cv2.convexHull(contour)
                        rect = cv2.minAreaRect(hull)
                        approx = cv2.boxPoints(rect)
                        source = "background:min_area_rect"
                    else:
                        source = "background:mask_quad"
                    try:
                        quad_work = validate_quad(order_quad([(float(x), float(y)) for x, y in approx]), (w, h),
                                                  min_area_ratio=params.min_area_ratio,
                                                  max_area_ratio=params.max_area_ratio,
                                                  min_edge_ratio=params.min_edge_ratio)
                        quad = to_original_scale(quad_work, work_size, original_size)
                        border_touch = sum(x <= 1 or y <= 1 or x >= w - 2 or y >= h - 2 for x, y in quad_work)
                        output.append(_candidate(self.name, source, quad, scale=scale, variant=mask_variant,
                                                 score=min(1.0, area), evidence={
                                                     "mask": "lab_median_mad", "mask_variant": mask_variant,
                                                     "area_ratio": area,
                                                     "background_median_lab": tuple(float(x) for x in median),
                                                     "background_mad_lab": tuple(float(x) for x in mad),
                                                     "border_touch_count": int(border_touch),
                                                 }))
                    except (GeometryError, ValueError, TypeError):
                        continue
        self._last_diagnostics = {}
        return output


class WhiteBorderProvider(BackgroundDifferenceProvider):
    """High-recall white-scanner candidates isolated from the V7 primary pool."""

    name = "white_border"

    def _generate(self, context: ImageFeatureContext, params: V7Parameters,
                  budget: _Budget):
        enabled = params.scene_profile == "scanner_white"
        diagnostics = {
            "border_connected_enabled": enabled,
            "border_connected_references": 0,
            "border_connected_raw_candidates": 0,
            "border_connected_clusters": 0,
            "border_connected_emitted": 0,
        }
        if not enabled:
            self._last_diagnostics = diagnostics
            return ()
        candidates, details = self._border_connected_generate(
            context, params, budget
        )
        diagnostics.update(details)
        self._last_diagnostics = diagnostics
        return candidates


class ContourProvider:
    name = "contour"

    def provide(self, context: ImageFeatureContext, params: V7Parameters,
                cancellation_token: Any = None, deadline: Any = None) -> ProviderResult:
        return _run(self, context, params, cancellation_token, deadline)

    generate = provide
    run = provide

    def _generate(self, context: ImageFeatureContext, params: V7Parameters, budget: _Budget):
        output: list[Mapping[str, Any]] = []
        for scale in _scales(context, params):
            budget.check(1)
            edges = context.edges(scale, params.sha256(), low_threshold=35, high_threshold=130)
            budget.check(0)
            budget.check(max(1, int(math.ceil(edges.size / 256.0))))
            h, w = edges.shape[:2]
            # External contours capture the photo silhouette; CCOMP adds selected
            # nested scanner frames without exploding all hierarchy levels.
            contours, hierarchy = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            budget.check(0)
            if hierarchy is None:
                continue
            image_area = float(w * h)
            original_size = (context.shape[1], context.shape[0])
            indexed_contours = sorted(enumerate(contours), key=lambda item: (
                -cv2.contourArea(item[1]), tuple(item[1].reshape(-1, 2)[0])))
            for idx, contour in indexed_contours:
                budget.check(1)
                area = float(cv2.contourArea(contour)) / image_area
                if area < params.min_area_ratio or area > params.max_area_ratio:
                    continue
                perimeter = float(cv2.arcLength(contour, True))
                if perimeter <= 0:
                    continue
                for epsilon_ratio in params.contour_epsilon_ratios:
                    budget.check(1)
                    approx = cv2.approxPolyDP(contour, max(0.5, epsilon_ratio * perimeter), True).reshape(-1, 2)
                    source = "contour:external_quad" if len(approx) == 4 else "contour:hierarchy_quad"
                    quad_work = None
                    if len(approx) == 4:
                        try:
                            quad_work = validate_quad(order_quad([(float(x), float(y)) for x, y in approx]), (w, h),
                                                      min_area_ratio=params.min_area_ratio,
                                                      max_area_ratio=params.max_area_ratio,
                                                      min_edge_ratio=params.min_edge_ratio)
                        except (GeometryError, ValueError, TypeError):
                            quad_work = None
                    if quad_work is None:
                        # A raw approximation can be concave, self-intersecting,
                        # degenerate, too small, or too close to an image edge.
                        # Treat it as weak and use the bounded hull supplement.
                        budget.check(1)
                        hull = cv2.convexHull(contour)
                        if len(hull) < 4:
                            continue
                        approx_fallback = cv2.boxPoints(cv2.minAreaRect(hull))
                        source = "contour:convex_hull_rect"
                        try:
                            quad_work = validate_quad(order_quad([(float(x), float(y)) for x, y in approx_fallback]), (w, h),
                                                      min_area_ratio=params.min_area_ratio,
                                                      max_area_ratio=params.max_area_ratio,
                                                      min_edge_ratio=params.min_edge_ratio)
                        except (GeometryError, ValueError, TypeError):
                            continue
                    quad = to_original_scale(quad_work, (w, h), original_size)
                    output.append(_candidate(self.name, source, quad, scale=scale, variant=float(epsilon_ratio),
                                             score=min(1.0, area), evidence={
                                                 "area_ratio": area,
                                                 "epsilon_ratio": float(epsilon_ratio),
                                                 "retrieval": "ccomp",
                                                 "hierarchy_depth": int(hierarchy[0, idx, 3]),
                                             }))
        # Multiple epsilon approximations often identify the same quad.  Keep
        # every deterministic source candidate; reducer-level dedupe can audit
        # the agreement rather than losing evidence here.
        return output


@dataclass(frozen=True)
class _DetectedLine:
    """Canonical, numeric-only representation of one LSD segment."""

    p1: tuple[float, float]
    p2: tuple[float, float]
    angle: float
    length: float
    offset: float
    projection: tuple[float, float]
    quality: float


def _angle_mod_pi(angle: float) -> float:
    angle = float(angle) % math.pi
    return angle if angle >= 0.0 else angle + math.pi


def _angle_distance(a: float, b: float) -> float:
    delta = abs(float(a) - float(b)) % math.pi
    return min(delta, math.pi - delta)


def _line_sort_key(line: _DetectedLine) -> tuple[float, ...]:
    return (round(line.angle, 10), round(line.offset, 6), round(line.projection[0], 6),
            round(line.projection[1], 6), round(line.p1[0], 6), round(line.p1[1], 6),
            round(line.p2[0], 6), round(line.p2[1], 6))


class LineProvider:
    """Bounded arbitrary-angle line-segment provider.

    LSD output is converted to canonical lines, clustered on a fixed circular
    angle grid, merged by normal residual/projection gap, and paired only with
    an approximately perpendicular family.  Every potentially unbounded loop
    is explicitly charged to the provider work budget.
    """

    name = "lines"
    max_raw_segments = 256
    max_merged_lines = 64
    max_families = 36
    max_pairs = 256
    max_candidates = 64
    angle_bins = 36
    # Perspective can rotate the two opposite side edges by ~15 degrees on a
    # strongly trapezoidal scan; the fixed circular bin still keeps unrelated
    # texture lines out while admitting that spread.
    cluster_angle = math.radians(19.0)

    def provide(self, context: ImageFeatureContext, params: V7Parameters,
                cancellation_token: Any = None, deadline: Any = None) -> ProviderResult:
        return _run(self, context, params, cancellation_token, deadline)

    generate = provide
    run = provide

    @staticmethod
    def _canonical_segment(values: Sequence[float], shape: tuple[int, int], quality: float = 1.0) -> _DetectedLine | None:
        x1, y1, x2, y2 = (float(v) for v in values)
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
            return None
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        height, width = shape
        min_len = max(6.0, 0.035 * math.hypot(width, height))
        if length < min_len:
            return None
        angle = _angle_mod_pi(math.atan2(dy, dx))
        ux, uy = math.cos(angle), math.sin(angle)
        # The normal is chosen consistently for angle modulo pi.  This makes
        # opposite endpoint order and LSD's orientation irrelevant.
        nx, ny = -uy, ux
        projection = tuple(sorted((x1 * ux + y1 * uy, x2 * ux + y2 * uy)))
        offset = ((x1 + x2) * 0.5) * nx + ((y1 + y2) * 0.5) * ny
        if not math.isfinite(offset):
            return None
        p_a = (ux * projection[0] + nx * offset, uy * projection[0] + ny * offset)
        p_b = (ux * projection[1] + nx * offset, uy * projection[1] + ny * offset)
        # Quality is deliberately a soft gate: LSD's width/precision values
        # vary across OpenCV builds, while very short segments are rejected by
        # length above.  A non-positive confidence remains weak evidence.
        quality = max(0.0, min(1.0, float(quality)))
        if quality <= 0.02:
            return None
        return _DetectedLine(p_a, p_b, angle, float(length), float(offset),
                             (float(projection[0]), float(projection[1])), quality)

    def _generate(self, context: ImageFeatureContext, params: V7Parameters, budget: _Budget):
        output: list[Mapping[str, Any]] = []
        raw_count = merged_count = family_count = pair_count = 0
        for scale in _scales(context, params):
            budget.check(1)
            gray = context.gray(scale, params.sha256())
            # Charge one deterministic hash/cache unit and one unit per fixed
            # pixel block.  This also bounds callers that supply huge images.
            budget.check(1 + max(1, int(math.ceil(gray.size / 256.0))))
            height, width = gray.shape[:2]
            detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
            budget.check(1)
            detected = detector.detect(gray)
            budget.check(1)
            raw = detected[0] if isinstance(detected, tuple) else detected
            widths = detected[1] if isinstance(detected, tuple) and len(detected) > 1 else None
            precisions = detected[2] if isinstance(detected, tuple) and len(detected) > 2 else None
            if raw is None:
                continue
            records: list[_DetectedLine] = []
            raw_array = np.asarray(raw).reshape(-1, 4)
            # Sorting before truncation means a noisy scene cannot alter which
            # bounded subset is retained merely through detector traversal order.
            retained_limit = max(0, self.max_raw_segments - raw_count)
            for index, row in enumerate(raw_array):
                budget.check(1)
                values = tuple(float(v) for v in row)
                quality = 1.0
                if widths is not None:
                    try:
                        quality *= max(0.0, min(1.0, 1.0 / (1.0 + abs(float(np.asarray(widths).reshape(-1)[index])))))
                    except (IndexError, TypeError, ValueError):
                        pass
                if precisions is not None:
                    try:
                        quality *= max(0.0, min(1.0, float(np.asarray(precisions).reshape(-1)[index]) * 10.0))
                    except (IndexError, TypeError, ValueError):
                        pass
                segment = self._canonical_segment(values, (height, width), quality)
                if segment is not None:
                    records.append(segment)
                    # Keep only the numerically smallest bounded prefix while
                    # extracting, avoiding an unbounded retained-segment list
                    # on adversarial line grids.
                    if len(records) > retained_limit:
                        records.sort(key=_line_sort_key)
                        del records[retained_limit:]
            records.sort(key=_line_sort_key)
            records = records[: max(0, self.max_raw_segments - raw_count)]
            raw_count += len(records)
            budget.check(1)

            # Fixed circular angle bins provide stable clustering while the
            # tolerance permits perspective-induced changes between opposite
            # edges.  Families are sorted by angle and then offset.
            families: list[dict[str, Any]] = []
            for record in records:
                budget.check(1)
                bucket = int(round(record.angle / math.pi * self.angle_bins)) % self.angle_bins
                assigned = None
                best_distance = float("inf")
                for family_index, family in enumerate(families):
                    distance = _angle_distance(record.angle, family["angle"])
                    if distance <= self.cluster_angle and (distance, family_index) < (best_distance, family_index):
                        assigned, best_distance = family_index, distance
                if assigned is None:
                    families.append({"angle": record.angle, "bucket": bucket, "records": [record]})
                else:
                    families[assigned]["records"].append(record)
            families.sort(key=lambda family: (int(family["bucket"]), float(family["angle"]),
                                              _line_sort_key(family["records"][0])))
            families = families[: max(0, self.max_families - family_count)]
            family_count += len(families)
            budget.check(1)

            diagonal = math.hypot(width, height)
            residual_limit = max(2.0, 0.006 * diagonal)
            gap_limit = max(4.0, 0.025 * diagonal)
            merged_families: list[dict[str, Any]] = []
            for family in families:
                budget.check(1)
                records_in_family = sorted(family["records"], key=lambda line: (round(line.offset, 6), _line_sort_key(line)))
                merged: list[dict[str, Any]] = []
                for record in records_in_family:
                    budget.check(1)
                    if not merged:
                        merged.append({"angle": record.angle, "offset": record.offset,
                                       "lo": record.projection[0], "hi": record.projection[1],
                                       "quality": record.quality, "length": record.length})
                        continue
                    previous = merged[-1]
                    gap = record.projection[0] - previous["hi"]
                    if (abs(record.offset - previous["offset"]) <= residual_limit and
                            gap <= gap_limit):
                        weight = previous["length"] + record.length
                        previous["offset"] = (previous["offset"] * previous["length"] + record.offset * record.length) / weight
                        previous["angle"] = (previous["angle"] * previous["length"] + record.angle * record.length) / weight
                        previous["lo"] = min(previous["lo"], record.projection[0])
                        previous["hi"] = max(previous["hi"], record.projection[1])
                        previous["quality"] = max(previous["quality"], record.quality)
                        previous["length"] = weight
                    else:
                        merged.append({"angle": record.angle, "offset": record.offset,
                                       "lo": record.projection[0], "hi": record.projection[1],
                                       "quality": record.quality, "length": record.length})
                canonical = []
                for item in merged:
                    budget.check(1)
                    # Keep each merged line's weighted angle.  Opposite edges
                    # in a perspective trapezoid may differ by several degrees;
                    # forcing the family anchor angle would move intersections
                    # substantially even though clustering remains correct.
                    ux, uy = math.cos(float(item["angle"])), math.sin(float(item["angle"]))
                    nx, ny = -uy, ux
                    canonical.append({**item,
                                      "p1": (ux * item["lo"] + nx * item["offset"], uy * item["lo"] + ny * item["offset"]),
                                      "p2": (ux * item["hi"] + nx * item["offset"], uy * item["hi"] + ny * item["offset"])})
                canonical.sort(key=lambda item: (round(float(item["offset"]), 6), round(float(item["lo"]), 6), round(float(item["hi"]), 6)))
                remaining_lines = max(0, self.max_merged_lines - merged_count)
                canonical = canonical[:remaining_lines]
                merged_families.append({**family, "lines": canonical})
                merged_count += len(canonical)
            merged_families = [family for family in merged_families if len(family["lines"]) >= 2]
            merged_families.sort(key=lambda family: (int(family["bucket"]), float(family["angle"])))

            def line_pairs(lines: Sequence[Mapping[str, Any]]) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
                pairs = []
                for i in range(len(lines)):
                    for j in range(i + 1, len(lines)):
                        budget.check(1)
                        separation = abs(float(lines[i]["offset"]) - float(lines[j]["offset"]))
                        pairs.append((separation, i, j))
                pairs.sort(key=lambda item: (-round(item[0], 6), item[1], item[2]))
                return [(lines[i], lines[j]) for _, i, j in pairs[:16]]

            for family_index, family_a in enumerate(merged_families):
                for family_b in merged_families[family_index + 1:]:
                    budget.check(1)
                    family_angle_delta = _angle_distance(float(family_a["angle"]), float(family_b["angle"]))
                    if abs(family_angle_delta - math.pi / 2.0) > math.radians(25.0):
                        continue
                    pairs_a = line_pairs(family_a["lines"])
                    pairs_b = line_pairs(family_b["lines"])
                    for pair_a in pairs_a:
                        for pair_b in pairs_b:
                            budget.check(1)
                            if pair_count >= self.max_pairs:
                                break
                            sep_a = abs(float(pair_a[0]["offset"]) - float(pair_a[1]["offset"]))
                            sep_b = abs(float(pair_b[0]["offset"]) - float(pair_b[1]["offset"]))
                            if min(sep_a, sep_b) < max(8.0, params.min_edge_ratio * diagonal):
                                continue
                            pair_count += 1
                            try:
                                corners = (
                                    line_intersection(pair_a[0]["p1"], pair_a[0]["p2"], pair_b[0]["p1"], pair_b[0]["p2"]),
                                    line_intersection(pair_a[0]["p1"], pair_a[0]["p2"], pair_b[1]["p1"], pair_b[1]["p2"]),
                                    line_intersection(pair_a[1]["p1"], pair_a[1]["p2"], pair_b[1]["p1"], pair_b[1]["p2"]),
                                    line_intersection(pair_a[1]["p1"], pair_a[1]["p2"], pair_b[0]["p1"], pair_b[0]["p2"]),
                                )
                                quad_work = validate_quad(order_quad(corners), (width, height),
                                                          min_area_ratio=params.min_area_ratio,
                                                          max_area_ratio=params.max_area_ratio,
                                                          min_edge_ratio=params.min_edge_ratio)
                            except (GeometryError, ValueError, TypeError, OverflowError):
                                continue
                            quad = to_original_scale(quad_work, (width, height), (context.shape[1], context.shape[0]))
                            area = abs(sum(quad_work[i][0] * quad_work[(i + 1) % 4][1] -
                                           quad_work[(i + 1) % 4][0] * quad_work[i][1] for i in range(4))) * 0.5 / (width * height)
                            angle_score = max(0.0, 1.0 - abs(family_angle_delta - math.pi / 2.0) / math.radians(25.0))
                            quality_score = (float(pair_a[0]["quality"]) + float(pair_a[1]["quality"]) +
                                             float(pair_b[0]["quality"]) + float(pair_b[1]["quality"])) / 4.0
                            output.append(_candidate(self.name, "lines:lsd", quad, scale=scale,
                                                     variant=(family_index, int(family_b["bucket"]), round(sep_a, 5), round(sep_b, 5)),
                                                     score=min(1.0, 0.5 * area + 0.35 * angle_score + 0.15 * quality_score),
                                                     evidence={"detector": "createLineSegmentDetector", "angle_modulo": "pi",
                                                               "family_angle": float(family_a["angle"]), "cross_family_angle": float(family_b["angle"]),
                                                               "separation": (float(sep_a), float(sep_b)), "area_ratio": float(area)}))
                        if pair_count >= self.max_pairs:
                            break
                    if pair_count >= self.max_pairs:
                        break
                if pair_count >= self.max_pairs:
                    break
            output.sort(key=lambda candidate: (str(candidate["id"]), str(candidate["source"])))
            output = output[: self.max_candidates]
        self._last_diagnostics = {"raw_segments": int(raw_count), "retained_segments": int(raw_count),
                                  "merged_lines": int(merged_count), "families": int(family_count),
                                  "direction_families": int(family_count), "pairs": int(pair_count),
                                  "line_pairs": int(pair_count), "candidate_count": len(output)}
        return output


def generate_background_candidates(context: ImageFeatureContext, params: V7Parameters,
                                   cancellation_token: Any = None, deadline: Any = None) -> ProviderResult:
    return BackgroundDifferenceProvider().provide(context, params, cancellation_token, deadline)


def generate_contour_candidates(context: ImageFeatureContext, params: V7Parameters,
                                cancellation_token: Any = None, deadline: Any = None) -> ProviderResult:
    return ContourProvider().provide(context, params, cancellation_token, deadline)


def generate_line_candidates(context: ImageFeatureContext, params: V7Parameters,
                             cancellation_token: Any = None, deadline: Any = None) -> ProviderResult:
    return LineProvider().provide(context, params, cancellation_token, deadline)


BackgroundProvider = BackgroundDifferenceProvider
BackgroundMaskProvider = BackgroundDifferenceProvider
MultiScaleContourProvider = ContourProvider
ContourCandidateProvider = ContourProvider
LineSegmentProvider = LineProvider
LineCandidateProvider = LineProvider
LinesProvider = LineProvider

__all__ = ["RegionProvider", "CandidateProvider", "BackgroundDifferenceProvider", "BackgroundProvider",
           "BackgroundMaskProvider", "ContourProvider", "MultiScaleContourProvider", "ContourCandidateProvider",
           "LineProvider", "LineSegmentProvider", "LineCandidateProvider", "LinesProvider", "WhiteBorderProvider",
           "generate_background_candidates", "generate_contour_candidates", "generate_line_candidates"]
