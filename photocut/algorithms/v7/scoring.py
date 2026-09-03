"""Bounded, truth-independent v7 candidate scoring and audit.

The scorer intentionally keeps image work in :class:`ImageFeatureContext`.  A
single cached Lab/gradient image is sampled for every fused candidate; only the
small Top-K set receives the more detailed four-edge pass.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from .features import ImageFeatureContext
from .geometry import validate_quad, quad_area_ratio
from .reducer import select_topk
from .types import CandidateAudit


_COMPONENTS = (
    "geometry", "gradient_continuity", "gradient_direction", "inside_outside_lab",
    "inside_outside_texture", "outside_background_consistency", "internal_content_coverage",
    "provider_consensus", "candidate_margin", "nesting", "frame_extent",
)


def _clip(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    return float(np.clip(value if math.isfinite(value) else default, 0.0, 1.0))


def _quad(item: Mapping[str, Any]) -> tuple[tuple[float, float], ...]:
    corners = item.get("corners", item.get("original_legal_corners"))
    return tuple((float(p[0]), float(p[1])) for p in corners)


def _area(q: Sequence[Sequence[float]]) -> float:
    return abs(sum(q[i][0] * q[(i + 1) % 4][1] - q[(i + 1) % 4][0] * q[i][1] for i in range(4))) / 2.0


def _bbox(q: Sequence[Sequence[float]]) -> tuple[float, float, float, float]:
    xs, ys = zip(*q)
    return min(xs), min(ys), max(xs), max(ys)


def _inside(points: np.ndarray, q: Sequence[Sequence[float]]) -> np.ndarray:
    # Convex TL/TR/BR/BL quads: all cross products have one sign.  This is
    # vectorized and avoids a per-candidate resize or polygon rasterization.
    vertices = np.asarray(q, dtype=np.float32)
    a = vertices
    b = np.roll(vertices, -1, axis=0)
    cross = (b[:, 0, None] - a[:, 0, None]) * (points[None, :, 1] - a[:, 1, None]) - (b[:, 1, None] - a[:, 1, None]) * (points[None, :, 0] - a[:, 0, None])
    return np.all(cross >= -1e-3, axis=0) | np.all(cross <= 1e-3, axis=0)


class _FeatureView:
    def __init__(self, value: Any, image_size: Sequence[float] | None, scale: Any):
        if isinstance(value, ImageFeatureContext):
            self.context = value
        elif hasattr(value, "normalized_bgr"):
            self.context = ImageFeatureContext.from_loaded(value)
        else:
            self.context = ImageFeatureContext(np.asarray(value))
        source_h, source_w = self.context.shape[:2]
        self.source_size = (float(image_size[0]), float(image_size[1])) if image_size is not None else (float(source_w), float(source_h))
        self.scale = scale
        self.bgr = self.context.bgr(scale)
        self.lab = self.context.lab(scale)
        self.gradient = self.context.gradient(scale)
        self.height, self.width = self.gradient.shape[:2]
        # Global scalar thresholds are computed once per image scale; candidate
        # loops only gather a handful of cached pixels.
        self.gradient_mean = float(np.mean(self.gradient))
        self.gradient_p85 = float(np.percentile(self.gradient, 85))
        self.gradient_p60 = float(np.percentile(self.gradient, 60))
        band = max(2, int(round(min(self.height, self.width) * .025)))
        lab_pieces = (
            self.lab[:band], self.lab[-band:], self.lab[:, :band], self.lab[:, -band:],
        )
        gradient_pieces = (
            self.gradient[:band], self.gradient[-band:],
            self.gradient[:, :band], self.gradient[:, -band:],
        )
        border_lab = np.concatenate(tuple(
            piece.reshape(-1, 3)[::max(1, piece.size // 15_000)]
            for piece in lab_pieces
        )).astype(np.float32)
        border_gradient = np.concatenate(tuple(
            piece.reshape(-1)[::max(1, piece.size // 5_000)]
            for piece in gradient_pieces
        )).astype(np.float32)
        self.border_lab_median = np.median(border_lab, axis=0)
        self.border_gradient_median = float(np.median(border_gradient))
        self.border_gradient_mad = float(np.median(np.abs(
            border_gradient - self.border_gradient_median
        )))

    def points(self, q: Sequence[Sequence[float]], count: int = 1800) -> tuple[np.ndarray, np.ndarray]:
        sx, sy = self.width / max(self.source_size[0], 1.0), self.height / max(self.source_size[1], 1.0)
        x0, y0, x1, y1 = _bbox(q)
        nx = max(5, min(48, int(math.sqrt(count * max(x1 - x0, 1) / max(y1 - y0, 1)))))
        ny = max(5, min(48, int(count / max(nx, 1))))
        xs = np.linspace(max(0, x0), min(self.source_size[0] - 1, x1), nx)
        ys = np.linspace(max(0, y0), min(self.source_size[1] - 1, y1), ny)
        xx, yy = np.meshgrid(xs, ys)
        points = np.column_stack((xx.ravel(), yy.ravel()))
        inside = _inside(points, q)
        return points, inside

    def sample(self, array: np.ndarray, points: np.ndarray) -> np.ndarray:
        sx, sy = self.width / max(self.source_size[0], 1.0), self.height / max(self.source_size[1], 1.0)
        xi = np.clip(np.rint(points[:, 0] * sx).astype(int), 0, self.width - 1)
        yi = np.clip(np.rint(points[:, 1] * sy).astype(int), 0, self.height - 1)
        return array[yi, xi]


def _edge_pass(view: _FeatureView, q: Sequence[Sequence[float]], *, samples: int = 20) -> tuple[np.ndarray, np.ndarray]:
    vals, dirs = [], []
    sx, sy = view.width / max(view.source_size[0], 1), view.height / max(view.source_size[1], 1)
    for i in range(4):
        a, b = np.asarray(q[i]), np.asarray(q[(i + 1) % 4])
        t = np.linspace(0, 1, samples)
        line = a[None, :] * (1 - t[:, None]) + b[None, :] * t[:, None]
        # A pair of offset samples estimates whether the gradient is aligned to
        # the candidate side, without running another Sobel/Canny operation.
        tangent = b - a
        length = float(np.linalg.norm(tangent)) or 1.0
        normal = np.asarray((-tangent[1], tangent[0])) / length
        g = view.sample(view.gradient, line)
        left = view.sample(view.gradient, line + normal[None, :] * 2.0)
        right = view.sample(view.gradient, line - normal[None, :] * 2.0)
        mean = float(np.mean(g))
        continuity = math.exp(-float(np.std(g)) / (mean + 1e-3))
        direction = _clip(mean / (float(np.mean(left + right)) + mean + 1e-3) * 2.0)
        vals.append(_clip(continuity)); dirs.append(direction)
    return np.asarray(vals), np.asarray(dirs)


def _scanner_boundary_pass(
    view: _FeatureView, q: Sequence[Sequence[float]], *, samples: int = 40,
) -> tuple[float, tuple[float, ...], Mapping[str, tuple[float, ...]]]:
    """Score four observable inside/outside strips against the scanner frame."""
    centroid = np.mean(np.asarray(q, dtype=float), axis=0)
    source_w, source_h = view.source_size
    source_per_work_pixel = max(
        source_w / max(view.width, 1), source_h / max(view.height, 1),
    )
    near = 2.5 * source_per_work_pixel
    far = 6.0 * source_per_work_pixel
    edge_scores: list[float] = []
    lab_scores: list[float] = []
    texture_scores: list[float] = []
    frame_scores: list[float] = []
    coverage_scores: list[float] = []
    for index in range(4):
        a = np.asarray(q[index], dtype=float)
        b = np.asarray(q[(index + 1) % 4], dtype=float)
        tangent = b - a
        length = float(np.linalg.norm(tangent))
        if length <= 1e-6:
            edge_scores.append(0.0)
            lab_scores.append(0.0)
            texture_scores.append(0.0)
            frame_scores.append(0.0)
            coverage_scores.append(0.0)
            continue
        normal = np.asarray((-tangent[1], tangent[0]), dtype=float) / length
        midpoint = (a + b) * .5
        if float(np.dot(centroid - midpoint, normal)) < 0:
            normal = -normal
        t = np.linspace(.04, .96, samples)
        line = a[None, :] * (1.0 - t[:, None]) + b[None, :] * t[:, None]
        inner_points = np.concatenate((
            line + normal[None, :] * near,
            line + normal[None, :] * far,
        ))
        outer_points = np.concatenate((
            line - normal[None, :] * near,
            line - normal[None, :] * far,
        ))
        valid = (
            (inner_points[:, 0] >= 0) & (inner_points[:, 0] < source_w) &
            (inner_points[:, 1] >= 0) & (inner_points[:, 1] < source_h) &
            (outer_points[:, 0] >= 0) & (outer_points[:, 0] < source_w) &
            (outer_points[:, 1] >= 0) & (outer_points[:, 1] < source_h)
        )
        if int(np.count_nonzero(valid)) < 12:
            edge_scores.append(0.0)
            lab_scores.append(0.0)
            texture_scores.append(0.0)
            frame_scores.append(0.0)
            coverage_scores.append(0.0)
            continue
        inner_points = inner_points[valid]
        outer_points = outer_points[valid]
        inner_lab = view.sample(view.lab, inner_points).astype(np.float32)
        outer_lab = view.sample(view.lab, outer_points).astype(np.float32)
        inner_gradient = view.sample(view.gradient, inner_points).astype(np.float32)
        outer_gradient = view.sample(view.gradient, outer_points).astype(np.float32)

        lab_delta = np.abs(inner_lab - outer_lab)
        lab_delta[:, 0] *= .55
        lab_score = _clip(float(np.median(np.linalg.norm(lab_delta, axis=1))) / 28.0)
        texture_score = _clip(
            abs(float(np.median(inner_gradient)) - float(np.median(outer_gradient))) /
            (view.gradient_p85 + 1e-3)
        )
        outer_delta = np.abs(outer_lab - view.border_lab_median[None, :])
        outer_delta[:, 0] *= .55
        frame_lab = math.exp(-float(np.median(np.linalg.norm(outer_delta, axis=1))) / 22.0)
        outer_median = float(np.median(outer_gradient))
        outer_mad = float(np.median(np.abs(outer_gradient - outer_median)))
        gradient_scale = view.border_gradient_mad + 12.0
        frame_texture = math.exp(-(
            abs(outer_median - view.border_gradient_median) +
            abs(outer_mad - view.border_gradient_mad)
        ) / gradient_scale)
        frame_score = _clip(.45 * frame_lab + .55 * frame_texture)

        edge_gradient = np.maximum.reduce([
            view.sample(view.gradient, line),
            view.sample(view.gradient, line + normal[None, :] * source_per_work_pixel),
            view.sample(view.gradient, line - normal[None, :] * source_per_work_pixel),
        ]).astype(np.float32)
        coverage = _clip(float(np.mean(edge_gradient > view.gradient_p60)))
        strength = _clip(float(np.median(edge_gradient)) / (view.gradient_p85 + 1e-3))
        score = _clip(
            .30 * coverage + .20 * strength + .20 * lab_score +
            .10 * texture_score + .20 * frame_score
        )
        edge_scores.append(score)
        lab_scores.append(lab_score)
        texture_scores.append(texture_score)
        frame_scores.append(frame_score)
        coverage_scores.append(coverage)
    boundary = _clip(.60 * min(edge_scores, default=0.0) +
                     .40 * float(np.mean(edge_scores) if edge_scores else 0.0))
    evidence = {
        "scanner_boundary_lab_per_edge": tuple(lab_scores),
        "scanner_boundary_texture_per_edge": tuple(texture_scores),
        "scanner_frame_similarity_per_edge": tuple(frame_scores),
        "scanner_edge_coverage_per_edge": tuple(coverage_scores),
    }
    return boundary, tuple(edge_scores), evidence


def _cheap_edge_proxy(view: _FeatureView, q: Sequence[Sequence[float]]) -> tuple[np.ndarray, np.ndarray]:
    """Sample four midpoints from the cached gradient only.

    This is deliberately not the full edge pass: no line bands, offsets or
    extra image operators are run for the forty-candidate audit stage.
    """
    mids = np.asarray([((q[i][0] + q[(i + 1) % 4][0]) / 2.0,
                       (q[i][1] + q[(i + 1) % 4][1]) / 2.0) for i in range(4)], dtype=float)
    values = view.sample(view.gradient, mids).astype(float)
    scale = view.gradient_p85 + 1e-3
    cont = np.clip(values / scale, 0.0, 1.0)
    direction = np.clip(values / (view.gradient_mean + scale * .15 + 1e-3), 0.0, 1.0)
    return cont, direction


def _aggregate_pre_score(scores: Mapping[str, Any]) -> float:
    geometry = _clip(scores.get("geometry")); edge_score = _clip(scores.get("edge_score"))
    lab_score = _clip(scores.get("inside_outside_lab")); texture = _clip(scores.get("inside_outside_texture"))
    background = _clip(scores.get("outside_background_consistency")); content = _clip(scores.get("internal_content_coverage"))
    provider = _clip(scores.get("provider_consensus")); model_agreement = _clip(scores.get("model_classical_agreement")); margin = _clip(scores.get("candidate_margin"), .5); nesting = _clip(scores.get("nesting"), .5)
    extent = _clip(scores.get("frame_extent"), .5)
    # Strong internal lines can produce excellent local edge scores. A scan
    # photo normally occupies most of the image, so extent is an explicit
    # tie-breaker against small interior rectangles.
    # Edge strength is retained as a hard quality gate below, rather than as
    # the main ranking signal: internal photo details can be stronger than the
    # true scan boundary. Extent therefore carries the largest explicit
    # ranking weight for scan-like inputs.
    # Independent-provider agreement is a strong reliability signal: a quad
    # supported by (for example) both the background and line providers should
    # beat an otherwise similar single-provider proposal.  The weight remains
    # bounded and is applied only after geometry/edge evidence, so agreement
    # cannot rescue a weak or malformed edge.
    weighted = (.24 * geometry + .09 * lab_score + .07 * texture + .08 * background + .07 * content + .04 * provider + .02 * model_agreement + .04 * margin + .05 * nesting + .30 * extent)
    gated = min(weighted, .45 * edge_score + .55 * weighted)
    # After the weak-edge gate, use a small consensus tie-break only when the
    # candidate still has a usable edge signal.  This fixes near-ties between
    # one-provider and independently corroborated quads without allowing a
    # consensus label to rescue an obviously weak boundary.
    if edge_score >= 0.25 and provider >= 0.5:
        gated += 0.05 * provider
    return _clip(gated)


def _component_scores(candidate: Mapping[str, Any], view: _FeatureView, image_size: Sequence[float], *, nested: float = .5,
                      margin: float = .5, full: bool = False,
                      scanner_boundary: bool = False) -> dict[str, float]:
    q = _quad(candidate)
    w, h = float(image_size[0]), float(image_size[1])
    area_ratio = _clip(_area(q) / max(w * h, 1.0))
    bbox_ratio = _clip(((max(p[0] for p in q) - min(p[0] for p in q) + 1.0) *
                        (max(p[1] for p in q) - min(p[1] for p in q) + 1.0)) / max(w * h, 1.0))
    frame_extent = _clip((bbox_ratio - .35) / .60)
    edges = [math.hypot(q[(i + 1) % 4][0] - q[i][0], q[(i + 1) % 4][1] - q[i][1]) for i in range(4)]
    edge_regular = _clip(1.0 - np.std(edges) / (np.mean(edges) + 1e-6) * 2.0)
    geometry = _clip(.65 * min(1.0, area_ratio * 5.0) + .35 * edge_regular)
    edge_cont, edge_dir = _edge_pass(view, q, samples=32) if full else _cheap_edge_proxy(view, q)
    edge_scores = np.minimum(edge_cont, edge_dir)
    edge_score = _clip(.55 * float(np.min(edge_scores)) + .45 * float(np.mean(edge_scores)))
    points, mask = view.points(q, count=3200 if full else 900)
    inside_values = view.sample(view.lab, points[mask]) if np.any(mask) else np.empty((0, 3))
    outside_values = view.sample(view.lab, points[~mask]) if np.any(~mask) else np.empty((0, 3))
    outside_known = len(outside_values) >= 8 and min(_bbox(q)) > 1 and _bbox(q)[2] < w - 2 and _bbox(q)[3] < h - 2
    if outside_known and len(inside_values) and len(outside_values):
        lab_delta = float(np.mean(np.abs(inside_values.mean(axis=0) - outside_values.mean(axis=0))))
        inside_texture, outside_texture = float(np.std(view.sample(view.gradient, points[mask]))), float(np.std(view.sample(view.gradient, points[~mask])))
        lab_score = _clip(lab_delta / 55.0)
        texture_score = _clip(abs(inside_texture - outside_texture) / 60.0)
        background = _clip(math.exp(-float(np.std(outside_values)) / 45.0))
    else:
        lab_score = texture_score = .5
        background = .5
    content = _clip(float(np.mean(view.sample(view.gradient, points[mask]) > view.gradient_p60)) if np.any(mask) else 0.0)
    providers = candidate.get("providers", candidate.get("sources", candidate.get("source", ())))
    if isinstance(providers, str): providers = (providers,)
    provider_names = {str(value).split(":", 1)[0].lower() for value in providers} if providers else set()
    provider_consensus = _clip((len(provider_names) if provider_names else 0) / 3.0)
    has_model = bool(provider_names & {"docquadnet", "docaligner_heatmap"})
    has_classical = bool(provider_names & {"background", "contour", "lines", "shape"})
    model_classical_agreement = 1.0 if has_model and has_classical else 0.0
    scores = {
        "geometry": geometry, "gradient_continuity": float(np.mean(edge_cont)), "gradient_direction": float(np.mean(edge_dir)),
        "inside_outside_lab": lab_score, "inside_outside_texture": texture_score,
        "outside_background_consistency": background, "internal_content_coverage": content,
        "provider_consensus": provider_consensus, "candidate_margin": _clip(margin), "nesting": _clip(nested),
        "frame_extent": frame_extent, "model_classical_agreement": model_classical_agreement,
        "model_classical_agreement_weight": 0.02,
        "edge_score": edge_score,
        "gradient_continuity_per_edge": tuple(float(_clip(v)) for v in edge_cont),
        "gradient_direction_per_edge": tuple(float(_clip(v)) for v in edge_dir),
        "edge_scores": tuple(float(_clip(v)) for v in edge_scores),
    }
    if scanner_boundary:
        boundary, scanner_edges, scanner_evidence = _scanner_boundary_pass(view, q)
        scores.update({
            "scanner_boundary_score": boundary,
            "scanner_boundary_per_edge": tuple(
                float(_clip(v)) for v in scanner_edges
            ),
            **scanner_evidence,
        })
    # Stable descriptive aliases make the contract convenient for callers
    # while retaining the canonical component names above.
    scores.update({"geometry_score": geometry, "lab_difference": lab_score,
                   "texture_difference": texture_score, "background_consistency": background,
                   "content_coverage": content, "nesting_score": _clip(nested)})
    # A minimum-edge gate is deliberate: averaging four good sides must not
    # conceal one absent/weak side.
    scores["pre_score"] = _aggregate_pre_score(scores)
    return scores


def score_components(candidate: Mapping[str, Any], image_or_features: Any, image_size: Sequence[float] | None = None, *, scale: Any = None,
                     nested: float = .5, margin: float = .5) -> Mapping[str, float]:
    """Return bounded component evidence for one candidate."""
    view = image_or_features if isinstance(image_or_features, _FeatureView) else _FeatureView(image_or_features, image_size, scale)
    size = image_size if image_size is not None else view.source_size
    return _component_scores(
        candidate, view, size, nested=nested, margin=margin, full=True,
        scanner_boundary=True,
    )


def score_scanner_boundaries(
    candidates: Iterable[Mapping[str, Any]], image_or_features: Any,
    image_size: Sequence[float] | None = None, *, scale: Any = 800,
) -> Mapping[str, Mapping[str, Any]]:
    """Return one cached, low-resolution four-strip score per candidate."""
    view = image_or_features if isinstance(image_or_features, _FeatureView) else _FeatureView(
        image_or_features, image_size, scale,
    )
    output = {}
    for item in candidates:
        candidate_id = str(item.get("candidate_id", item.get("id", "")))
        if not candidate_id:
            continue
        boundary, edges, evidence = _scanner_boundary_pass(view, _quad(item))
        output[candidate_id] = {
            "scanner_boundary_score": boundary,
            "scanner_boundary_per_edge": tuple(float(_clip(value)) for value in edges),
            **evidence,
        }
    return output


@dataclass(frozen=True)
class ScoreResult:
    candidates: tuple[Mapping[str, Any], ...]
    audits: tuple[CandidateAudit, ...]
    selected: tuple[Mapping[str, Any], ...]
    truncation_trace: tuple[Mapping[str, Any], ...]

    @property
    def candidate_audits(self):
        return self.audits

    @property
    def topk(self):
        return self.selected

    @property
    def trace(self):
        return self.truncation_trace

    def __iter__(self):
        yield self.selected
        yield self.audits


def _nested_info(items: Sequence[Mapping[str, Any]], scores: Sequence[Mapping[str, float]], i: int, size: Sequence[float]) -> tuple[bool, float]:
    parent = _bbox(_quad(items[i])); parent_area = _area(_quad(items[i])); best = .0; found = False
    for j, item in enumerate(items):
        if i == j: continue
        child = _bbox(_quad(item))
        if _area(_quad(item)) >= parent_area * .90: continue
        child_corners = np.asarray(_quad(item), dtype=float)
        # Bounding boxes alone make a rotated diagonal candidate appear nested
        # even when its corners cross the parent's polygon.  Require all child
        # vertices to be inside the convex parent quad.
        polygon_contained = bool(np.all(_inside(child_corners, _quad(items[i]))))
        if polygon_contained and child[0] >= parent[0] and child[1] >= parent[1] and child[2] <= parent[2] and child[3] <= parent[3]:
            found = True; best = max(best, float(scores[j].get("pre_score", 0.0)))
    return found, best


def _audit_for(item: Mapping[str, Any], score: Mapping[str, float], nested_exists: bool, nested_score: float, image_size: Sequence[float]) -> CandidateAudit:
    q = _quad(item); w, h = float(image_size[0]), float(image_size[1]); x0, y0, x1, y1 = _bbox(q)
    proximity = _clip(1.0 - min(x0, y0, w - 1 - x1, h - 1 - y1) / max(.08 * min(w, h), 1.0))
    outside = x0 > 2 and y0 > 2 and x1 < w - 3 and y1 < h - 3
    # A border-touching candidate normally has no observable background.  An
    # outer scanner frame is the narrow exception: it is approximately the
    # whole image, contains another candidate, and either has unusually sparse
    # interior content or a distinct edge/frame cue.  Keep this cue independent
    # of outside-background consistency, whose ``.5`` value means unknown.
    area_ratio = _clip(_area(q) / max(w * h, 1.0))
    # Coordinates are pixel-inclusive (a quad from 0..w-1 spans all w
    # columns), so include both endpoints when reporting image coverage.
    width_ratio = _clip((x1 - x0 + 1.0) / max(w, 1.0))
    height_ratio = _clip((y1 - y0 + 1.0) / max(h, 1.0))
    near_full_border = proximity >= .75 and width_ratio >= .90 and height_ratio >= .90
    content = _clip(score.get("internal_content_coverage", .5))
    edge_score = _clip(score.get("edge_score", .5))
    edge_direction = _clip(score.get("gradient_direction", .5))
    sparse_content = content <= .08
    frame_cue = edge_score <= .10 and edge_direction <= .15
    decisions: list[str] = []
    if not outside:
        decisions.append("outside_background_unverifiable")
    support = (nested_exists and nested_score >= .15 and
               _clip(score.get("outside_background_consistency", .5)) >= .35 and
               _clip(score.get("internal_content_coverage", .5)) >= .05)
    # Border-touching candidates have no observable outside background. This
    # is explicitly unknown evidence, never proof of an outer frame.
    if outside and support and proximity >= .75:
        decisions.append("suspected_outer_frame")
    scanner_frame = (not outside and near_full_border and nested_exists and
                     (sparse_content or frame_cue))
    if scanner_frame:
        decisions.append("suspected_outer_frame")
    if not decisions:
        decisions.append("none")
    evidence = {"image_border_proximity": proximity, "nested_candidate_exists": bool(nested_exists),
                "nested_candidate_score": _clip(nested_score), "outside_background_observability": "observed" if outside else "unknown",
                "outside_background_observable": bool(outside),
                "background_uniformity": _clip(score.get("outside_background_consistency", .5)),
                "content_coverage": content, "candidate_area_ratio": area_ratio,
                "candidate_width_ratio": width_ratio, "candidate_height_ratio": height_ratio,
                "near_full_border": bool(near_full_border), "scanner_frame_cue": bool(frame_cue),
                "scanner_frame_sparse_content": bool(sparse_content)}
    sources = item.get("sources", item.get("providers", item.get("source", ())))
    if isinstance(sources, str): sources = (sources,)
    return CandidateAudit(candidate_id=str(item.get("candidate_id", item.get("id", ""))), sources=tuple(str(s) for s in sources) or ("unknown",),
                          original_legal_corners=q, pre_topk_corners=q,
                          pre_truncation_risk_decisions=tuple(decisions), pre_truncation_risk_evidence=evidence,
                          stage_scores={"pre_score": _clip(score.get("pre_score")), "components": dict(score)}, stage_ranks={})


def score_candidates(fused_candidates: Iterable[Mapping[str, Any]], image_or_features: Any, image_size: Sequence[float] | None = None,
                     *, top_k: int = 5, scale: Any = None, params: Any = None) -> ScoreResult:
    """Audit all fused candidates, select Top-K, and fully score only Top-K."""
    # Refinement/full edge work is budgeted at five candidates by contract.
    # Callers may request a smaller set, but never expand the expensive pass.
    top_k = min(5, int(top_k))
    items = [dict(item) for item in fused_candidates]
    if not items:
        return ScoreResult((), (), (), ())
    view = image_or_features if isinstance(image_or_features, _FeatureView) else _FeatureView(image_or_features, image_size, scale)
    size = image_size if image_size is not None else view.source_size
    # Cheap pre-score for all candidates; no per-candidate feature construction.
    cheap = [_component_scores(item, view, size, full=False) for item in items]
    # Candidate margin is truth-independent separation from the next-best
    # candidate.  It is computed before nesting and included in the cheap rank.
    raw_scores = [float(s.get("pre_score", 0.0)) for s in cheap]
    for i, score in enumerate(raw_scores):
        others = [other for j, other in enumerate(raw_scores) if j != i]
        second = max(others, default=0.0)
        cheap[i]["candidate_margin"] = _clip(.5 + (score - second) * 5.0)
        cheap[i]["pre_score"] = _aggregate_pre_score(cheap[i])
    for i, item in enumerate(items):
        item["component_scores"] = dict(cheap[i]); item["pre_score"] = float(cheap[i]["pre_score"]); item["cheap_score"] = item["pre_score"]
    audits0 = []
    for i, item in enumerate(items):
        exists, ns = _nested_info(items, cheap, i, size)
        cheap[i]["nesting"] = _clip(ns if exists else .5)
        cheap[i]["pre_score"] = _aggregate_pre_score(cheap[i])
        audits0.append(_audit_for(item, cheap[i], exists, ns, size))
        item["audit"] = audits0[-1].to_dict()
        # Reducer precondition uses this short marker while the complete,
        # production evidence remains in ``CandidateAudit`` above.
        item["outer_risk"] = audits0[-1].pre_truncation_risk_decisions
    # Nesting evidence changes the cheap rank; keep the reducer-facing fields
    # synchronized with the production component/audit record.
    for i, item in enumerate(items):
        item["component_scores"] = dict(cheap[i])
        item["pre_score"] = float(cheap[i]["pre_score"])
        item["cheap_score"] = item["pre_score"]
        item["audit"]["stage_scores"]["pre_score"] = item["pre_score"]
    selection = select_topk(items, image_size=size, top_k=top_k)
    selected_ids = {str(item.get("candidate_id", item.get("id", ""))) for item in selection.candidates}
    ranked_ids = [str(item.get("candidate_id", item.get("id", ""))) for item in sorted(items, key=lambda x: (-float(x.get("pre_score", 0.0)), str(x.get("candidate_id", ""))))]
    rank_by_id = {cid: rank for rank, cid in enumerate(ranked_ids, 1)}
    selected_rank_by_id = {str(item.get("candidate_id", item.get("id", ""))): rank for rank, item in enumerate(selection.candidates, 1)}
    audits, output = [], []
    for i, (item, audit) in enumerate(zip(items, audits0)):
        cid = str(item.get("candidate_id", item.get("id", "")))
        if cid in selected_ids:
            full = _component_scores(item, view, size, nested=cheap[i].get("nesting", .5), margin=cheap[i].get("candidate_margin", .5), full=True)
            item["component_scores"] = dict(full); item["score"] = float(full["pre_score"]); item["stage_scores"] = dict(full)
            audit = CandidateAudit.from_dict({**audit.to_dict(), "stage_scores": {"pre_score": item["pre_score"], "full_score": item["score"], "components": dict(full)}, "stage_ranks": {"pre_score": rank_by_id.get(cid), "selected": selected_rank_by_id.get(cid)}, "truncation_stage": "selected", "truncation_reason": None})
        else:
            trace = next((event for event in selection.truncation_trace if str(event.get("candidate_id")) == cid and event.get("stage") == "topk"), None)
            reason = str(trace.get("reason", "topk_limit")) if trace else "topk_limit"
            audit = CandidateAudit.from_dict({**audit.to_dict(), "stage_ranks": {"pre_score": rank_by_id.get(cid)}, "truncation_stage": "topk", "truncation_reason": reason})
            item["score"] = item["pre_score"]
        item["audit"] = audit.to_dict(); item["truncation_stage"] = audit.truncation_stage; item["truncation_reason"] = audit.truncation_reason
        audits.append(audit); output.append(item)
    output_by_id = {str(item.get("candidate_id", item.get("id", ""))): item for item in output}
    selected = tuple(output_by_id[cid] for cid in selected_rank_by_id if cid in output_by_id)
    audit_by_id = {audit.candidate_id: audit for audit in audits}
    trace = []
    for event in selection.truncation_trace:
        event = dict(event)
        audit = audit_by_id.get(str(event.get("candidate_id", "")))
        if audit is not None:
            # Keep the trace tied to the same production audit contract as the
            # selected result; consumers never need to inspect debug payloads.
            event["candidate_audit"] = audit.to_dict()
            event["pre_truncation_risk_decisions"] = audit.to_dict()["pre_truncation_risk_decisions"]
            event["pre_truncation_risk_evidence"] = audit.to_dict()["pre_truncation_risk_evidence"]
            event["stage_scores"] = audit.to_dict()["stage_scores"]
        trace.append(event)
    return ScoreResult(tuple(output), tuple(audits), selected, tuple(trace))


score_fused_candidates = score_candidates
audit_candidates = score_candidates
score_candidate = score_components
compute_component_scores = score_components
component_scores = score_components
rank_candidates = score_candidates
score_fused = score_candidates

__all__ = ["ScoreResult", "score_components", "score_candidate", "compute_component_scores", "score_candidates",
           "component_scores", "score_fused_candidates", "score_fused", "audit_candidates", "rank_candidates",
           "score_scanner_boundaries"]
