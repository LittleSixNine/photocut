"""White-scanner-bed exterior evidence and bounded joint candidate ranking.

The selector looks beyond each proposed photo side.  A physical photo edge
should lead into a long, stable region whose appearance agrees with the
scanner bed visible at the image frame.  An internal print or white-border
edge usually encounters another transition before reaching that bed.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np

from photocut.algorithms.v7.geometry import validate_quad


@dataclass(frozen=True)
class ScannerExteriorConfig:
    work_max_edge: int = 1200
    samples_per_side: int = 64
    distance_samples: int = 12
    outward_distance_ratio: float = 0.06
    frame_ratio: float = 0.02

    def __post_init__(self) -> None:
        for name, lower, upper in (
            ("work_max_edge", 64, 2048),
            ("samples_per_side", 16, 128),
            ("distance_samples", 4, 32),
        ):
            value = getattr(self, name)
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError(f"{name} is outside its bounded range")
        for name, lower, upper in (
            ("outward_distance_ratio", 0.01, 0.15),
            ("frame_ratio", 0.005, 0.05),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be finite")
            if not math.isfinite(float(value)) or not lower <= float(value) <= upper:
                raise ValueError(f"{name} is outside its bounded range")


@dataclass(frozen=True)
class ScannerSelectorConfig:
    """Sealed ranking, rescue and refusal thresholds for scanner-white V8."""

    exterior_weight: float = 0.10
    area_weight: float = 0.68
    prior_weight: float = 0.22
    v7_edge_min_disagreement: float = 0.02
    v7_edge_agreement_distance: float = 0.02
    v7_edge_agreement_ratio: float = 4.0
    v7_mask_agreement_distance: float = 0.01
    v7_mask_outlier_ratio: float = 1.25
    v7_mask_exterior_advantage: float = 0.05
    edge_mask_agreement_distance: float = 0.03
    edge_mask_outlier_ratio: float = 3.0
    edge_mask_max_prior: float = 0.10
    edge_support_exterior_tolerance: float = 0.10
    minimum_unopposed_exterior_score: float = 0.50
    edge_cluster_distance: float = 0.012
    minimum_automatic_score_margin: float = 0.02
    minimum_edge_seed_support: int = 2
    minimum_edge_source_group_support: int = 2
    minimum_automatic_valid_sides: int = 2
    manual_conflict_distance: float = 0.05

    def __post_init__(self) -> None:
        for name in ("exterior_weight", "area_weight", "prior_weight"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be finite")
            if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} is outside its bounded range")
        if not math.isclose(
            float(self.exterior_weight + self.area_weight + self.prior_weight),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("selector ranking weights must sum to one")
        for name, lower, upper in (
            ("v7_edge_min_disagreement", 0.001, 0.15),
            ("v7_edge_agreement_distance", 0.001, 0.15),
            ("v7_mask_agreement_distance", 0.001, 0.15),
            ("v7_mask_exterior_advantage", 0.0, 0.5),
            ("edge_mask_agreement_distance", 0.001, 0.15),
            ("edge_mask_max_prior", 0.0, 0.5),
            ("edge_support_exterior_tolerance", 0.0, 0.5),
            ("minimum_unopposed_exterior_score", 0.0, 1.0),
            ("edge_cluster_distance", 0.001, 0.05),
            ("minimum_automatic_score_margin", 0.0, 0.25),
            ("manual_conflict_distance", 0.01, 0.15),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be finite")
            if not math.isfinite(float(value)) or not lower <= float(value) <= upper:
                raise ValueError(f"{name} is outside its bounded range")
        for name in ("v7_edge_agreement_ratio", "v7_mask_outlier_ratio", "edge_mask_outlier_ratio"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be finite")
            if not math.isfinite(float(value)) or not 1.0 <= float(value) <= 10.0:
                raise ValueError(f"{name} is outside its bounded range")
        for name, upper in (
            ("minimum_edge_seed_support", 8),
            ("minimum_edge_source_group_support", 8),
            ("minimum_automatic_valid_sides", 4),
        ):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= upper:
                raise ValueError(f"{name} must be in [1, {upper}]")


@dataclass(frozen=True)
class ScannerExteriorEvidence:
    score: float
    side_scores: tuple[float, ...]
    side_bed_scores: tuple[float, ...]
    side_connected_scores: tuple[float, ...]
    side_stability_scores: tuple[float, ...]
    side_coverages: tuple[float, ...]
    valid_side_count: int
    side_score_min: float
    side_score_mean: float
    side_stability_mean: float
    bed_lab: tuple[float, float, float]
    bed_scale: float


@dataclass(frozen=True)
class RankedScannerCandidate:
    candidate_id: str
    corners: tuple[tuple[float, float], ...]
    score: float
    prior_score_normalized: float
    area_ratio: float
    evidence: ScannerExteriorEvidence
    seed_support_count: int = 1
    seed_source_group_count: int = 1
    seed_candidate_ids: tuple[str, ...] = ()
    seed_source_groups: tuple[str, ...] = ()


@dataclass(frozen=True)
class ScannerSelectionResult:
    selected: RankedScannerCandidate
    reason: str
    base_edge_candidate_id: str | None
    competing_edge_candidate_id: str | None = None
    competing_edge_distance: float | None = None
    edge_score_margin: float | None = None
    selected_edge_cluster_size: int = 0
    selected_edge_cluster_seed_support_count: int = 0
    selected_edge_cluster_source_group_count: int = 0


@dataclass(frozen=True)
class _PreparedScannerExterior:
    width: int
    height: int
    scale: float
    work_width: int
    work_height: int
    lab: np.ndarray
    bed_lab: np.ndarray
    bed_scale: float
    distances: np.ndarray
    config: ScannerExteriorConfig


def _sample(lab: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = lab.shape[:2]
    points = np.asarray(points, dtype=np.float32)
    valid = (
        (points[:, 0] >= 0.0)
        & (points[:, 0] <= width - 1.0)
        & (points[:, 1] >= 0.0)
        & (points[:, 1] <= height - 1.0)
    )
    values = cv2.remap(
        lab,
        points[:, 0].reshape(-1, 1),
        points[:, 1].reshape(-1, 1),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    ).reshape(-1, 3)
    return values.astype(np.float32, copy=False), valid


def _scanner_bed_reference(lab: np.ndarray, frame_ratio: float) -> tuple[np.ndarray, float]:
    height, width = lab.shape[:2]
    thickness = max(2, int(round(min(width, height) * frame_ratio)))
    frame = np.concatenate(
        (
            lab[:thickness].reshape(-1, 3),
            lab[-thickness:].reshape(-1, 3),
            lab[:, :thickness].reshape(-1, 3),
            lab[:, -thickness:].reshape(-1, 3),
        ),
        axis=0,
    ).astype(np.float32, copy=False)
    light_cut = float(np.percentile(frame[:, 0], 65.0))
    chroma = np.linalg.norm(frame[:, 1:] - 128.0, axis=1)
    chroma_cut = float(np.percentile(chroma, 75.0))
    likely_bed = frame[(frame[:, 0] >= light_cut) & (chroma <= chroma_cut)]
    if len(likely_bed) < 32:
        likely_bed = frame[frame[:, 0] >= light_cut]
    reference = np.median(likely_bed, axis=0).astype(np.float32)
    distances = np.linalg.norm(likely_bed - reference, axis=1)
    spread = float(np.percentile(distances, 80.0)) if len(distances) else 0.0
    scale = float(np.clip(6.0 + 2.5 * spread, 8.0, 30.0))
    return reference, scale


def _side_evidence(
    lab: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    centroid: np.ndarray,
    bed_lab: np.ndarray,
    bed_scale: float,
    distances: np.ndarray,
    samples_per_side: int,
) -> tuple[float, float, float, float, float]:
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length < 4.0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    direction /= length
    normal = np.asarray((direction[1], -direction[0]), dtype=np.float64)
    if float(np.dot(normal, centroid - (start + end) * 0.5)) > 0.0:
        normal *= -1.0
    positions = np.linspace(0.08, 0.92, samples_per_side, dtype=np.float64)[:, None]
    base = start + positions * (end - start)
    sampled = []
    valid_rows = []
    for distance in distances:
        values, valid = _sample(lab, base + float(distance) * normal)
        sampled.append(values)
        valid_rows.append(valid)
    values = np.stack(sampled, axis=0)
    valid = np.stack(valid_rows, axis=0)
    minimum_valid_distances = max(3, len(distances) // 3)
    ray_valid = np.count_nonzero(valid, axis=0) >= minimum_valid_distances
    coverage = float(np.mean(ray_valid))
    if np.count_nonzero(ray_valid) < max(6, samples_per_side // 5):
        return 0.0, 0.0, 0.0, coverage, 0.0

    color_distance = np.linalg.norm(values - bed_lab[None, None, :], axis=2)
    similarity = np.exp(-0.5 * np.square(color_distance / bed_scale))
    similarity[~valid] = np.nan
    distance_scores = []
    for index in range(len(distances)):
        available = np.isfinite(similarity[index])
        if np.count_nonzero(available) >= max(6, samples_per_side // 5):
            distance_scores.append(float(np.nanmedian(similarity[index])))
    bed_score = float(np.percentile(distance_scores, 25.0)) if distance_scores else 0.0

    ray_scores = []
    for index in np.flatnonzero(ray_valid):
        available = similarity[:, index]
        available = available[np.isfinite(available)]
        ray_scores.append(float(np.percentile(available, 25.0)))
    connected_score = float(np.mean(ray_scores)) if ray_scores else 0.0

    transitions = []
    for index in range(len(distances) - 1):
        pair_valid = valid[index] & valid[index + 1]
        if np.count_nonzero(pair_valid) >= max(6, samples_per_side // 5):
            delta = np.linalg.norm(values[index + 1] - values[index], axis=1)
            transitions.append(float(np.median(delta[pair_valid])))
    strongest_late_transition = max(transitions, default=0.0)
    stability_score = float(math.exp(-strongest_late_transition / 12.0))
    score = float(np.clip(
        0.45 * bed_score + 0.35 * connected_score + 0.20 * stability_score,
        0.0,
        1.0,
    ))
    return score, bed_score, connected_score, coverage, stability_score


def _prepare_scanner_exterior(
    image_bgr: np.ndarray,
    config: ScannerExteriorConfig,
) -> _PreparedScannerExterior:
    image = np.asarray(image_bgr)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8 or image.size == 0:
        raise ValueError("image_bgr must be a non-empty uint8 BGR image")
    height, width = image.shape[:2]
    scale = min(1.0, float(config.work_max_edge) / max(width, height))
    work_width = max(2, int(round(width * scale)))
    work_height = max(2, int(round(height * scale)))
    work = image if scale == 1.0 else cv2.resize(
        image, (work_width, work_height), interpolation=cv2.INTER_AREA
    )
    lab = cv2.cvtColor(work, cv2.COLOR_BGR2LAB).astype(np.float32)
    bed_lab, bed_scale = _scanner_bed_reference(lab, float(config.frame_ratio))
    diagonal = math.hypot(work_width, work_height)
    first_distance = max(2.0, 0.003 * diagonal)
    last_distance = max(first_distance * 2.0, float(config.outward_distance_ratio) * diagonal)
    distances = np.linspace(
        first_distance, last_distance, config.distance_samples, dtype=np.float64
    )
    return _PreparedScannerExterior(
        width=width,
        height=height,
        scale=scale,
        work_width=work_width,
        work_height=work_height,
        lab=lab,
        bed_lab=bed_lab,
        bed_scale=bed_scale,
        distances=distances,
        config=config,
    )


def _evaluate_prepared_scanner_exterior(
    prepared: _PreparedScannerExterior,
    corners: Sequence[Sequence[float]],
) -> ScannerExteriorEvidence:
    legal = validate_quad(
        corners,
        (prepared.width, prepared.height),
        min_area_ratio=0.001,
        max_area_ratio=0.9999,
    )
    proposed = np.asarray(legal, dtype=np.float64) * prepared.scale
    centroid = np.mean(proposed, axis=0)
    config = prepared.config
    side_values = [
        _side_evidence(
            prepared.lab,
            proposed[index],
            proposed[(index + 1) % 4],
            centroid,
            prepared.bed_lab,
            prepared.bed_scale,
            prepared.distances,
            config.samples_per_side,
        )
        for index in range(4)
    ]
    side_scores = tuple(float(value[0]) for value in side_values)
    side_bed = tuple(float(value[1]) for value in side_values)
    side_connected = tuple(float(value[2]) for value in side_values)
    side_coverages = tuple(float(value[3]) for value in side_values)
    side_stability = tuple(float(value[4]) for value in side_values)
    valid_indices = [index for index, coverage in enumerate(side_coverages) if coverage >= 0.2]
    valid_scores = [side_scores[index] for index in valid_indices]
    valid_stability = [side_stability[index] for index in valid_indices]
    if len(valid_scores) >= 2:
        score_min = min(valid_scores)
        score_mean = float(np.mean(valid_scores))
        score = float(np.clip(0.55 * score_mean + 0.45 * score_min, 0.0, 1.0))
        stability_mean = float(np.mean(valid_stability))
    else:
        score_min = score_mean = stability_mean = score = 0.0
    return ScannerExteriorEvidence(
        score=score,
        side_scores=side_scores,
        side_bed_scores=side_bed,
        side_connected_scores=side_connected,
        side_stability_scores=side_stability,
        side_coverages=side_coverages,
        valid_side_count=len(valid_indices),
        side_score_min=float(score_min),
        side_score_mean=float(score_mean),
        side_stability_mean=float(stability_mean),
        bed_lab=tuple(float(value) for value in prepared.bed_lab),
        bed_scale=float(prepared.bed_scale),
    )


def evaluate_scanner_exterior(
    image_bgr: np.ndarray,
    corners: Sequence[Sequence[float]],
    config: ScannerExteriorConfig | None = None,
) -> ScannerExteriorEvidence:
    """Measure whether all available candidate sides lead into scanner bed."""
    config = config or ScannerExteriorConfig()
    if not isinstance(config, ScannerExteriorConfig):
        raise TypeError("config must be ScannerExteriorConfig")
    return _evaluate_prepared_scanner_exterior(
        _prepare_scanner_exterior(image_bgr, config),
        corners,
    )


def _string_id_tuple(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)):
        raise ValueError(f"{name} must be a sequence of strings")
    result = tuple(value)
    if any(not isinstance(item, str) or not item for item in result):
        raise ValueError(f"{name} must contain non-empty strings")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _candidate_input(
    candidate: Mapping[str, Any],
) -> tuple[str, Any, float, int, int, tuple[str, ...], tuple[str, ...]]:
    candidate_id = candidate.get("candidate_id")
    if not isinstance(candidate_id, str) or not candidate_id:
        raise ValueError("candidate_id is required")
    if "corners" not in candidate:
        raise ValueError("candidate corners are required")
    prior = candidate.get("prior_score", 0.0)
    if isinstance(prior, bool) or not isinstance(prior, (int, float)) or not math.isfinite(float(prior)):
        raise ValueError("prior_score must be finite")
    seed_support = candidate.get("seed_support_count", 1)
    seed_source_groups = candidate.get("seed_source_group_count", 1)
    if type(seed_support) is not int or seed_support < 1:
        raise ValueError("seed_support_count must be a positive integer")
    if type(seed_source_groups) is not int or seed_source_groups < 1:
        raise ValueError("seed_source_group_count must be a positive integer")
    raw_seed_ids = candidate.get("seed_candidate_ids")
    seed_ids = (
        tuple(f"{candidate_id}:implicit-seed:{index}" for index in range(seed_support))
        if raw_seed_ids is None
        else _string_id_tuple(raw_seed_ids, "seed_candidate_ids")
    )
    raw_source_groups = candidate.get("seed_source_groups")
    source_groups = (
        tuple(f"{candidate_id}:implicit-source:{index}" for index in range(seed_source_groups))
        if raw_source_groups is None
        else _string_id_tuple(raw_source_groups, "seed_source_groups")
    )
    if len(seed_ids) != seed_support:
        raise ValueError("seed_candidate_ids count does not match seed_support_count")
    if len(source_groups) != seed_source_groups:
        raise ValueError("seed_source_groups count does not match seed_source_group_count")
    return (
        candidate_id,
        candidate["corners"],
        float(prior),
        seed_support,
        seed_source_groups,
        seed_ids,
        source_groups,
    )


def rank_scanner_candidates(
    image_bgr: np.ndarray,
    candidates: Iterable[Mapping[str, Any]],
    config: ScannerExteriorConfig | None = None,
    *,
    selector_config: ScannerSelectorConfig | None = None,
) -> tuple[RankedScannerCandidate, ...]:
    """Rank legal proposals with joint four-side exterior evidence."""
    selector_config = selector_config or ScannerSelectorConfig()
    if not isinstance(selector_config, ScannerSelectorConfig):
        raise TypeError("selector_config must be ScannerSelectorConfig")
    materialized = tuple(candidates)
    if not materialized:
        return ()
    exterior_config = config or ScannerExteriorConfig()
    if not isinstance(exterior_config, ScannerExteriorConfig):
        raise TypeError("config must be ScannerExteriorConfig")
    prepared_exterior = _prepare_scanner_exterior(image_bgr, exterior_config)
    parsed = []
    seen: set[str] = set()
    width = prepared_exterior.width
    height = prepared_exterior.height
    for candidate in materialized:
        if not isinstance(candidate, Mapping):
            raise TypeError("scanner candidate must be a mapping")
        (
            candidate_id,
            corners,
            prior,
            seed_support,
            seed_source_groups,
            seed_ids,
            source_groups,
        ) = _candidate_input(candidate)
        if candidate_id in seen:
            raise ValueError("candidate_id must be unique")
        seen.add(candidate_id)
        legal = validate_quad(corners, (width, height), min_area_ratio=0.001, max_area_ratio=0.9999)
        parsed.append((
            candidate_id,
            legal,
            prior,
            seed_support,
            seed_source_groups,
            seed_ids,
            source_groups,
        ))
    priors = np.asarray([item[2] for item in parsed], dtype=np.float64)
    span = float(np.max(priors) - np.min(priors))
    normalized = np.full(len(parsed), 0.5, dtype=np.float64) if span < 1e-12 else (
        (priors - float(np.min(priors))) / span
    )
    ranked = []
    image_area = float(width * height)
    for index, (
        candidate_id,
        legal,
        _prior,
        seed_support,
        seed_source_groups,
        seed_ids,
        source_groups,
    ) in enumerate(parsed):
        evidence = _evaluate_prepared_scanner_exterior(prepared_exterior, legal)
        area_ratio = float(abs(cv2.contourArea(np.asarray(legal, dtype=np.float32))) / image_area)
        score = float(np.clip(
            float(selector_config.exterior_weight) * evidence.score
            + float(selector_config.area_weight) * area_ratio
            + float(selector_config.prior_weight) * float(normalized[index]),
            0.0,
            1.0,
        ))
        ranked.append(RankedScannerCandidate(
            candidate_id=candidate_id,
            corners=tuple(tuple(float(value) for value in point) for point in legal),
            score=score,
            prior_score_normalized=float(normalized[index]),
            area_ratio=area_ratio,
            evidence=evidence,
            seed_support_count=seed_support,
            seed_source_group_count=seed_source_groups,
            seed_candidate_ids=seed_ids,
            seed_source_groups=source_groups,
        ))
    ranked.sort(key=lambda item: (-item.score, item.candidate_id))
    return tuple(ranked)


def _quad_distance(
    left: Sequence[Sequence[float]],
    right: Sequence[Sequence[float]],
    image_size: tuple[int, int],
) -> float:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    return float(
        np.max(np.linalg.norm(left_array - right_array, axis=1))
        / math.hypot(*image_size)
    )


def _rank_external_candidate(
    image_bgr: np.ndarray,
    candidate: Mapping[str, Any] | None,
    config: ScannerExteriorConfig | None,
    selector_config: ScannerSelectorConfig | None,
) -> RankedScannerCandidate | None:
    if candidate is None:
        return None
    prepared = dict(candidate)
    prepared.setdefault("prior_score", 0.5)
    return rank_scanner_candidates(
        image_bgr, (prepared,), config, selector_config=selector_config
    )[0]


def _complete_link_cluster(
    anchor: RankedScannerCandidate,
    candidates: Sequence[RankedScannerCandidate],
    image_size: tuple[int, int],
    maximum_distance: float,
) -> tuple[RankedScannerCandidate, ...]:
    cluster = [anchor]
    for candidate in candidates:
        if candidate.candidate_id == anchor.candidate_id:
            continue
        if all(
            _quad_distance(candidate.corners, member.corners, image_size)
            <= maximum_distance
            for member in cluster
        ):
            cluster.append(candidate)
    return tuple(cluster)


def _complete_link_partition(
    candidates: Sequence[RankedScannerCandidate],
    image_size: tuple[int, int],
    maximum_distance: float,
) -> tuple[tuple[RankedScannerCandidate, ...], ...]:
    remaining = list(candidates)
    clusters = []
    while remaining:
        anchor = remaining.pop(0)
        cluster = [anchor]
        deferred = []
        for candidate in remaining:
            if all(
                _quad_distance(candidate.corners, member.corners, image_size)
                <= maximum_distance
                for member in cluster
            ):
                cluster.append(candidate)
            else:
                deferred.append(candidate)
        clusters.append(tuple(cluster))
        remaining = deferred
    return tuple(clusters)


def _cluster_support(
    cluster: Sequence[RankedScannerCandidate],
) -> tuple[int, int]:
    seed_ids = {seed_id for item in cluster for seed_id in item.seed_candidate_ids}
    source_groups = {source for item in cluster for source in item.seed_source_groups}
    return len(seed_ids), len(source_groups)


def select_scanner_candidate(
    image_bgr: np.ndarray,
    edge_candidates: Iterable[Mapping[str, Any]],
    *,
    v7_candidate: Mapping[str, Any] | None = None,
    mask_candidate: Mapping[str, Any] | None = None,
    config: ScannerExteriorConfig | None = None,
    selector_config: ScannerSelectorConfig | None = None,
) -> ScannerSelectionResult:
    """Select one scanner-white proposal with conservative consensus rescues.

    Edge hypotheses remain the primary route.  V7 and mask proposals can
    replace that result only when independent geometry agrees and the
    disagreement pattern is strong enough to identify the outlier.
    """
    selector_config = selector_config or ScannerSelectorConfig()
    if not isinstance(selector_config, ScannerSelectorConfig):
        raise TypeError("selector_config must be ScannerSelectorConfig")
    ranked_edges = rank_scanner_candidates(
        image_bgr, edge_candidates, config, selector_config=selector_config
    )
    ranked_v7 = _rank_external_candidate(
        image_bgr, v7_candidate, config, selector_config
    )
    ranked_mask = _rank_external_candidate(
        image_bgr, mask_candidate, config, selector_config
    )
    if not ranked_edges:
        fallback = ranked_v7 or ranked_mask
        if fallback is None:
            raise ValueError("at least one scanner candidate is required")
        reason = "v7_only" if ranked_v7 is not None else "mask_only"
        return ScannerSelectionResult(fallback, reason, None)

    base = ranked_edges[0]
    selected = base
    reason = "edge_base"
    width = int(np.asarray(image_bgr).shape[1])
    height = int(np.asarray(image_bgr).shape[0])
    image_size = (width, height)

    supported_edges = tuple(
        item for item in ranked_edges
        if item.seed_support_count >= selector_config.minimum_edge_seed_support
        and item.seed_source_group_count >= selector_config.minimum_edge_source_group_support
        and item.evidence.score + selector_config.edge_support_exterior_tolerance
        >= base.evidence.score
    )
    if (
        supported_edges
        and (
            base.seed_support_count < selector_config.minimum_edge_seed_support
            or base.seed_source_group_count < selector_config.minimum_edge_source_group_support
        )
    ):
        selected = min(
            supported_edges,
            key=lambda item: (
                -item.seed_source_group_count,
                -item.seed_support_count,
                -item.evidence.score,
                -item.score,
                item.candidate_id,
            ),
        )
        reason = "edge_seed_support_rescue"

    edge_clusters = _complete_link_partition(
        ranked_edges,
        image_size,
        float(selector_config.edge_cluster_distance),
    )
    base_cluster = next(
        cluster for cluster in edge_clusters
        if any(item.candidate_id == base.candidate_id for item in cluster)
    )
    base_cluster_support = _cluster_support(base_cluster)
    supported_clusters = []
    for cluster in edge_clusters:
        representative = cluster[0]
        seed_support, source_support = _cluster_support(cluster)
        if (
            seed_support >= selector_config.minimum_edge_seed_support
            and source_support >= selector_config.minimum_edge_source_group_support
            and representative.evidence.score
            + selector_config.edge_support_exterior_tolerance
            >= base.evidence.score
        ):
            supported_clusters.append((cluster, seed_support, source_support))
    if (
        selected.candidate_id == base.candidate_id
        and (
            base_cluster_support[0] < selector_config.minimum_edge_seed_support
            or base_cluster_support[1]
            < selector_config.minimum_edge_source_group_support
        )
        and supported_clusters
    ):
        cluster, _seed_support, _source_support = min(
            supported_clusters,
            key=lambda item: (
                -item[2],
                -item[1],
                -item[0][0].evidence.score,
                -item[0][0].score,
                item[0][0].candidate_id,
            ),
        )
        selected = cluster[0]
        reason = "edge_cluster_support_rescue"

    if ranked_v7 is not None:
        base_v7_distance = _quad_distance(selected.corners, ranked_v7.corners, image_size)
        nearest = min(
            ranked_edges,
            key=lambda item: (
                _quad_distance(item.corners, ranked_v7.corners, image_size),
                item.candidate_id,
            ),
        )
        nearest_distance = _quad_distance(nearest.corners, ranked_v7.corners, image_size)
        if (
            base_v7_distance >= float(selector_config.v7_edge_min_disagreement)
            and nearest_distance <= float(selector_config.v7_edge_agreement_distance)
            and base_v7_distance >= float(selector_config.v7_edge_agreement_ratio) * max(nearest_distance, 1e-6)
        ):
            selected = nearest
            reason = "v7_edge_agreement_rescue"

    if ranked_v7 is not None and ranked_mask is not None:
        edge_v7_distance = _quad_distance(selected.corners, ranked_v7.corners, image_size)
        edge_mask_distance = _quad_distance(selected.corners, ranked_mask.corners, image_size)
        v7_mask_distance = _quad_distance(ranked_v7.corners, ranked_mask.corners, image_size)
        if (
            v7_mask_distance <= float(selector_config.v7_mask_agreement_distance)
            and min(edge_v7_distance, edge_mask_distance)
            >= float(selector_config.v7_mask_outlier_ratio) * max(v7_mask_distance, 1e-6)
            and ranked_v7.evidence.score - selected.evidence.score
            >= float(selector_config.v7_mask_exterior_advantage)
        ):
            selected = ranked_v7
            reason = "v7_mask_consensus_rescue"
        elif (
            edge_mask_distance <= float(selector_config.edge_mask_agreement_distance)
            and min(edge_v7_distance, v7_mask_distance)
            >= float(selector_config.edge_mask_outlier_ratio) * max(edge_mask_distance, 1e-6)
            and selected.prior_score_normalized <= float(selector_config.edge_mask_max_prior)
        ):
            selected = ranked_mask
            reason = "edge_mask_consensus_rescue"

    selected_cluster = _complete_link_cluster(
        selected,
        ranked_edges,
        image_size,
        float(selector_config.edge_cluster_distance),
    )
    selected_cluster_ids = {item.candidate_id for item in selected_cluster}
    competing = next(
        (
            item for item in ranked_edges
            if item.candidate_id not in selected_cluster_ids
        ),
        None,
    )
    competing_distance = (
        None
        if competing is None
        else _quad_distance(competing.corners, selected.corners, image_size)
    )
    score_margin = None if competing is None else float(selected.score - competing.score)
    cluster_seed_ids = {
        seed_id for item in selected_cluster for seed_id in item.seed_candidate_ids
    }
    cluster_source_groups = {
        source for item in selected_cluster for source in item.seed_source_groups
    }
    return ScannerSelectionResult(
        selected,
        reason,
        base.candidate_id,
        None if competing is None else competing.candidate_id,
        competing_distance,
        score_margin,
        len(selected_cluster),
        len(cluster_seed_ids),
        len(cluster_source_groups),
    )


__all__ = [
    "RankedScannerCandidate",
    "ScannerExteriorConfig",
    "ScannerExteriorEvidence",
    "ScannerSelectionResult",
    "ScannerSelectorConfig",
    "evaluate_scanner_exterior",
    "rank_scanner_candidates",
    "select_scanner_candidate",
]
