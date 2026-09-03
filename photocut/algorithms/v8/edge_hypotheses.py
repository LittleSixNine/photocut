"""Bounded multi-hypothesis side search for scanner-white photographs.

Unlike a conventional refinement pass, this module does not force each side
to one locally strongest edge.  It retains a few spatially distinct,
continuous side hypotheses and emits their legal quadrilateral combinations
for later V7 scoring or candidate-only oracle evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json
import math
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np

from photocut.algorithms.v7.geometry import GeometryError, line_intersection, validate_quad


@dataclass(frozen=True)
class EdgeHypothesisConfig:
    work_max_edge: int = 1600
    search_band_ratio: float = 0.04
    samples_per_edge: int = 128
    hypotheses_per_edge: int = 4
    minimum_continuity: float = 1.5
    minimum_offset_separation_ratio: float = 0.0015
    fit_window_ratio: float = 0.002
    max_corner_shift_ratio: float = 0.065
    max_quad_hypotheses: int = 256

    def __post_init__(self) -> None:
        for name, lower, upper in (
            ("work_max_edge", 64, 4096),
            ("samples_per_edge", 32, 256),
            ("hypotheses_per_edge", 1, 4),
            ("max_quad_hypotheses", 1, 256),
        ):
            value = getattr(self, name)
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError(f"{name} is outside its bounded range")
        for name, lower, upper in (
            ("search_band_ratio", 0.003, 0.04),
            ("minimum_continuity", 0.0, 100.0),
            ("minimum_offset_separation_ratio", 0.0005, 0.01),
            ("fit_window_ratio", 0.0005, 0.01),
            ("max_corner_shift_ratio", 0.01, 0.10),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise TypeError(f"{name} must be finite")
            if not lower <= float(value) <= upper:
                raise ValueError(f"{name} is outside its bounded range")
        if self.max_quad_hypotheses > self.hypotheses_per_edge ** 4:
            raise ValueError("max_quad_hypotheses exceeds the side-combination bound")


@dataclass(frozen=True)
class EdgeQuadHypothesis:
    candidate_id: str
    corners: tuple[tuple[float, float], ...]
    score: float
    offsets_px: tuple[float, ...]


@dataclass(frozen=True)
class EdgeHypothesisResult:
    candidates: tuple[EdgeQuadHypothesis, ...]
    edge_evidence: tuple[Mapping[str, Any], ...]
    config: EdgeHypothesisConfig

    def __post_init__(self) -> None:
        object.__setattr__(self, "edge_evidence", tuple(MappingProxyType(dict(value)) for value in self.edge_evidence))


def aggregate_edge_hypotheses(
    observations: Iterable[tuple[str, Sequence[str], EdgeQuadHypothesis]],
    *,
    budget: int,
) -> tuple[dict[str, Any], ...]:
    """Deduplicate edge geometry while retaining canonical seed provenance."""
    if type(budget) is not int or not 1 <= budget <= 256:
        raise ValueError("budget must be an integer in [1, 256]")
    aggregated: dict[str, dict[str, Any]] = {}
    for seed_id, raw_sources, candidate in observations:
        if not isinstance(seed_id, str) or not seed_id:
            raise ValueError("edge observation seed_id is required")
        if not isinstance(raw_sources, (tuple, list)):
            raise ValueError("edge observation sources must be a sequence")
        sources = tuple(raw_sources)
        if any(not isinstance(source, str) or not source for source in sources):
            raise ValueError("edge observation sources must be non-empty strings")
        if not isinstance(candidate, EdgeQuadHypothesis):
            raise TypeError("edge observation candidate must be EdgeQuadHypothesis")
        entry = aggregated.setdefault(candidate.candidate_id, {
            "candidate": candidate,
            "seed_candidate_ids": set(),
            "seed_source_groups": set(),
        })
        if entry["candidate"].corners != candidate.corners:
            raise ValueError("edge candidate ID collision has different geometry")
        if candidate.score > entry["candidate"].score:
            entry["candidate"] = candidate
        entry["seed_candidate_ids"].add(seed_id)
        entry["seed_source_groups"].update(sources)
    ordered = sorted(
        aggregated.values(),
        key=lambda item: (-item["candidate"].score, item["candidate"].candidate_id),
    )
    output = []
    for entry in ordered[:budget]:
        candidate = entry["candidate"]
        seed_ids = tuple(sorted(entry["seed_candidate_ids"]))
        source_groups = tuple(sorted(entry["seed_source_groups"]))
        output.append({
            "candidate_id": candidate.candidate_id,
            "corners": candidate.corners,
            "prior_score": candidate.score,
            "seed_candidate_ids": seed_ids,
            "seed_source_groups": source_groups,
            "seed_support_count": len(seed_ids),
            "seed_source_group_count": max(1, len(source_groups)),
        })
    return tuple(output)


def _sample(image: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    points = np.asarray(points, dtype=np.float32)
    valid = (
        (points[:, 0] >= 0.0)
        & (points[:, 0] <= width - 1.0)
        & (points[:, 1] >= 0.0)
        & (points[:, 1] <= height - 1.0)
    )
    values = cv2.remap(
        image,
        points[:, 0].reshape(-1, 1),
        points[:, 1].reshape(-1, 1),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    ).reshape(-1, image.shape[2])
    return values.astype(np.float32, copy=False), valid


def _profile(
    lab: np.ndarray,
    base: np.ndarray,
    normal: np.ndarray,
    offsets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    continuity = np.zeros(len(offsets), dtype=np.float64)
    white_bed = np.zeros(len(offsets), dtype=np.float64)
    valid_fraction = np.zeros(len(offsets), dtype=np.float64)
    score = np.zeros(len(offsets), dtype=np.float64)
    for index, offset in enumerate(offsets):
        center = base + float(offset) * normal
        differences = []
        validity = []
        for distance in (1.0, 2.0, 4.0):
            inside, inside_valid = _sample(lab, center - distance * normal)
            outside, outside_valid = _sample(lab, center + distance * normal)
            valid = inside_valid & outside_valid
            difference = np.linalg.norm(outside - inside, axis=1)
            difference[~valid] = np.nan
            differences.append(difference)
            validity.append(valid)
        stacked = np.vstack(differences)
        has_value = np.isfinite(stacked).any(axis=0)
        per_sample = np.max(np.where(np.isfinite(stacked), stacked, -np.inf), axis=0)
        per_sample[~has_value] = np.nan
        valid_fraction[index] = float(np.mean(has_value))
        if np.count_nonzero(has_value) < max(8, len(base) // 3):
            continue
        continuity[index] = float(np.nanpercentile(per_sample, 35.0))
        bed_values = []
        bed_valid = []
        for distance in (6.0, 10.0, 16.0):
            values, valid = _sample(lab, center + distance * normal)
            bed_values.append(values)
            bed_valid.append(valid)
        bed = np.vstack(bed_values)
        bed_mask = np.concatenate(bed_valid)
        if np.count_nonzero(bed_mask) >= max(8, len(base) // 2):
            observed = bed[bed_mask]
            light = float(np.median(observed[:, 0]) / 255.0)
            neutral = float(1.0 - min(1.0, np.median(np.linalg.norm(observed[:, 1:] - 128.0, axis=1)) / 72.0))
            uniform = float(math.exp(-float(np.std(observed[:, 0])) / 32.0))
            white_bed[index] = float(np.clip(0.55 * light + 0.25 * neutral + 0.20 * uniform, 0.0, 1.0))
        score[index] = continuity[index] * (0.55 + 0.45 * white_bed[index])
    return continuity, white_bed, valid_fraction, score


def _local_maxima(values: np.ndarray) -> list[int]:
    return [
        index
        for index in range(1, len(values) - 1)
        if values[index] >= values[index - 1] and values[index] > values[index + 1]
    ]


def _select_positions(
    offsets: np.ndarray,
    continuity: np.ndarray,
    valid_fraction: np.ndarray,
    score: np.ndarray,
    *,
    config: EdgeHypothesisConfig,
    diagonal: float,
) -> list[int]:
    positions = [
        index for index in _local_maxima(score)
        if continuity[index] >= float(config.minimum_continuity)
        and valid_fraction[index] >= 0.55
    ]
    positions.sort(key=lambda index: (score[index], continuity[index], -abs(offsets[index])), reverse=True)
    selected: list[int] = []
    separation = float(config.minimum_offset_separation_ratio) * diagonal
    for position in positions:
        if all(abs(float(offsets[position] - offsets[other])) >= separation for other in selected):
            selected.append(position)
        if len(selected) == config.hypotheses_per_edge:
            break
    return selected


def _fit_line(
    lab: np.ndarray,
    base: np.ndarray,
    normal: np.ndarray,
    selected_offset: float,
    fit_window: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    deltas = np.linspace(-fit_window, fit_window, max(7, int(math.ceil(2.0 * fit_window)) + 1))
    values = np.full((len(deltas), len(base)), -1.0, dtype=np.float32)
    for row, delta in enumerate(deltas):
        center = base + (selected_offset + float(delta)) * normal
        inside, inside_valid = _sample(lab, center - normal)
        outside, outside_valid = _sample(lab, center + normal)
        valid = inside_valid & outside_valid
        values[row, valid] = np.linalg.norm(outside[valid] - inside[valid], axis=1)
    best = np.argmax(values, axis=0)
    valid = values[best, np.arange(len(base))] >= 0.0
    points = base[valid] + (selected_offset + deltas[best[valid]])[:, None] * normal
    if len(points) < 8:
        raise GeometryError("too few points for edge hypothesis")
    fit = cv2.fitLine(points.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).reshape(-1)
    direction = np.asarray((float(fit[0]), float(fit[1])), dtype=np.float64)
    anchor = np.asarray((float(fit[2]), float(fit[3])), dtype=np.float64)
    length = float(np.linalg.norm(direction))
    if not np.isfinite(direction).all() or length < 1e-9:
        raise GeometryError("edge hypothesis line is degenerate")
    direction /= length
    return tuple(anchor - direction), tuple(anchor + direction)


def _candidate_id(corners: Sequence[Sequence[float]]) -> str:
    payload = [[round(float(x), 6), round(float(y), 6)] for x, y in corners]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode("utf-8")).hexdigest()[:20]


def _is_near_duplicate(
    proposed: np.ndarray,
    accepted: Sequence[np.ndarray],
    distance: float,
) -> bool:
    """Match the prior max-per-corner rule without a Python candidate loop."""
    if not accepted:
        return False
    stacked = np.stack(accepted, axis=0)
    per_candidate = np.max(np.linalg.norm(stacked - proposed[None, :, :], axis=2), axis=1)
    return bool(np.any(per_candidate <= distance))


def generate_edge_hypotheses(
    image_bgr: np.ndarray,
    coarse_corners: Sequence[Sequence[float]],
    config: EdgeHypothesisConfig | None = None,
) -> EdgeHypothesisResult:
    config = config or EdgeHypothesisConfig()
    if not isinstance(config, EdgeHypothesisConfig):
        raise TypeError("config must be EdgeHypothesisConfig")
    image = np.asarray(image_bgr)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8 or image.size == 0:
        raise ValueError("image_bgr must be a non-empty uint8 BGR image")
    height, width = image.shape[:2]
    original = validate_quad(coarse_corners, (width, height), min_area_ratio=0.001, max_area_ratio=0.9999)
    scale = min(1.0, float(config.work_max_edge) / max(width, height))
    work_width = max(2, int(round(width * scale)))
    work_height = max(2, int(round(height * scale)))
    work = image if scale == 1.0 else cv2.resize(image, (work_width, work_height), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(work, cv2.COLOR_BGR2LAB).astype(np.float32)
    corners = np.asarray(original, dtype=np.float64) * scale
    centroid = np.mean(corners, axis=0)
    diagonal = math.hypot(work_width, work_height)
    band = max(3.0, float(config.search_band_ratio) * diagonal)
    step = max(0.5, diagonal / 1800.0)
    offsets = np.arange(-band, band + 0.5 * step, step, dtype=np.float64)
    sample_t = np.linspace(0.06, 0.94, config.samples_per_edge, dtype=np.float64)[:, None]
    fit_window = max(1.5, float(config.fit_window_ratio) * diagonal)
    all_lines: list[list[tuple[tuple[float, float], tuple[float, float]]]] = []
    all_scores: list[list[float]] = []
    all_offsets: list[list[float]] = []
    edge_evidence: list[Mapping[str, Any]] = []
    for index in range(4):
        start, end = corners[index], corners[(index + 1) % 4]
        direction = end - start
        length = float(np.linalg.norm(direction))
        if length < 4.0:
            return EdgeHypothesisResult((), tuple(edge_evidence), config)
        direction /= length
        normal = np.asarray((direction[1], -direction[0]), dtype=np.float64)
        if float(np.dot(normal, centroid - (start + end) * 0.5)) > 0.0:
            normal *= -1.0
        base = start + sample_t * (end - start)
        continuity, white_bed, valid_fraction, profile_score = _profile(lab, base, normal, offsets)
        selected = _select_positions(
            offsets, continuity, valid_fraction, profile_score,
            config=config, diagonal=diagonal,
        )
        lines = []
        scores = []
        selected_offsets = []
        successful_positions = []
        for position in selected:
            try:
                lines.append(_fit_line(lab, base, normal, float(offsets[position]), fit_window))
            except (GeometryError, cv2.error, ValueError):
                continue
            scores.append(float(profile_score[position]))
            selected_offsets.append(float(offsets[position]))
            successful_positions.append(position)
        if not lines:
            return EdgeHypothesisResult((), tuple(edge_evidence), config)
        all_lines.append(lines)
        all_scores.append(scores)
        all_offsets.append(selected_offsets)
        edge_evidence.append({
            "edge_index": index,
            "offsets_px": tuple(value / scale for value in selected_offsets),
            "continuity": tuple(float(continuity[position]) for position in successful_positions),
            "white_bed": tuple(float(white_bed[position]) for position in successful_positions),
        })
    combinations = list(itertools.product(*(range(len(lines)) for lines in all_lines)))
    combinations.sort(
        key=lambda combination: sum(all_scores[edge][choice] for edge, choice in enumerate(combination)),
        reverse=True,
    )
    output: list[EdgeQuadHypothesis] = []
    output_arrays: list[np.ndarray] = []
    original_array = np.asarray(original, dtype=np.float64)
    source_diagonal = math.hypot(width, height)
    dedup_distance = 0.0005 * source_diagonal
    for combination in combinations:
        try:
            chosen = [all_lines[edge][choice] for edge, choice in enumerate(combination)]
            intersections = [
                line_intersection(*chosen[(edge - 1) % 4], *chosen[edge], tolerance=1e-6)
                for edge in range(4)
            ]
            proposed = validate_quad(
                np.asarray(intersections, dtype=np.float64) / scale,
                (width, height),
                min_area_ratio=0.001,
                max_area_ratio=0.9999,
            )
        except (GeometryError, TypeError, ValueError, cv2.error):
            continue
        proposed_array = np.asarray(proposed, dtype=np.float64)
        if float(np.max(np.linalg.norm(proposed_array - original_array, axis=1))) > float(config.max_corner_shift_ratio) * source_diagonal:
            continue
        if _is_near_duplicate(proposed_array, output_arrays, dedup_distance):
            continue
        raw_score = sum(all_scores[edge][choice] for edge, choice in enumerate(combination)) / 4.0
        chosen_offsets = tuple(all_offsets[edge][choice] / scale for edge, choice in enumerate(combination))
        output.append(EdgeQuadHypothesis(_candidate_id(proposed), proposed, float(raw_score), chosen_offsets))
        output_arrays.append(proposed_array)
        if len(output) == config.max_quad_hypotheses:
            break
    return EdgeHypothesisResult(tuple(output), tuple(edge_evidence), config)


__all__ = [
    "EdgeHypothesisConfig",
    "EdgeHypothesisResult",
    "EdgeQuadHypothesis",
    "aggregate_edge_hypotheses",
    "generate_edge_hypotheses",
]
