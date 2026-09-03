"""Bounded, truth-free V8.2 boundary evidence and refinement primitives."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from photocut.algorithms.v7.geometry import GeometryError, validate_quad

EDGE_NAMES = ("top", "right", "bottom", "left")
FEATURE_NAMES = (
    "score", "raw_prior_log", "prior_score_normalized", "area_ratio",
    "exterior_score", "valid_side_ratio", "seed_support_log",
    "seed_source_group_ratio", "side_score", "side_bed_score",
    "side_connected_score", "side_stability_score", "side_coverage",
    "source_edge", "source_v7_current", "source_v7_seed", "source_mask",
    "source_rank_fraction", "distance_to_edge", "distance_to_v7_current",
    "distance_to_v7_seed", "distance_to_mask",
    "agreement_source_fraction_005", "agreement_source_fraction_010",
    "agreement_source_fraction_020",
)

BOUNDARY_VIEW_NAMES = (
    "identity",
    "horizontal_flip",
    "area_downscale_075",
    "gamma_085",
    "gamma_115",
    "gaussian_blur",
)

BOUNDARY_RUNTIME_FEATURE_NAMES = (
    *FEATURE_NAMES,
    *(f"edge_{name}" for name in EDGE_NAMES),
    "profile_peak_offset",
    "profile_peak_width",
    "profile_entropy",
    "profile_second_peak_ratio",
    "profile_support_q10",
    "profile_support_q50",
    "profile_support_q90",
    "profile_maximum_gap_ratio",
    "profile_valid_coverage",
    "refinement_residual_q50",
    "refinement_residual_q90",
    "refinement_residual_q95",
    "refinement_inlier_ratio",
    "refinement_orientation_change",
    "refinement_parent_shift",
    "refinement_accepted",
    "candidate_uses_refined_line",
    "consistency_peak_offset_dispersion",
    "consistency_line_angle_dispersion",
    "consistency_maximum_endpoint_deviation",
    "consistency_view_agreement_ratio",
    "consistency_view_failure_ratio",
)

MASK_BOUNDARY_FEATURE_NAMES = (
    "mask_peak_offset",
    "mask_zero_crossing_width",
    "mask_entropy",
    "mask_inside_foreground",
    "mask_outside_background",
    "mask_transition_strength",
    "mask_valid_coverage",
    "mask_lab_disagreement",
)

MASK_BOUNDARY_ABSOLUTE_FEATURE_NAMES = (
    "mask_absolute_peak_offset",
    *MASK_BOUNDARY_FEATURE_NAMES[1:],
)

MASK_BOUNDARY_SIGNED_ABSOLUTE_FEATURE_NAMES = (
    "mask_peak_offset",
    "mask_absolute_peak_offset",
    *MASK_BOUNDARY_FEATURE_NAMES[1:],
)

FORBIDDEN_TRUTH_FEATURE_FIELDS = (
    "truth",
    "analysis_photo_truth",
    "photo_truth",
    "strict",
    "automatic_strict",
    "recommended_strict",
    "v7_strict",
    "mask_strict",
    "union_strict",
    "edge_first_strict_rank",
    "edge_strict_candidate",
    "catastrophic",
    "automatic_catastrophic",
    "recommended_catastrophic",
    "max_normalized_corner_error",
    "mean_normalized_corner_error",
    "recommended_max_normalized_corner_error",
    "recommended_mean_normalized_corner_error",
)

@dataclass(frozen=True)
class BoundaryProfileConfig:
    work_max_edge: int = 1600
    longitudinal_samples: int = 64
    offset_samples: int = 81
    normal_distance_samples: int = 3
    search_band_ratio: float = 0.04
    maximum_normal_distance_ratio: float = 0.003
    frame_ratio: float = 0.02
    second_peak_separation_ratio: float = 0.008
    maximum_sample_evaluations: int = 65_536
    minimum_refinement_coverage: float = 0.50
    maximum_refinement_entropy: float = 0.80
    maximum_refinement_second_peak_ratio: float = 0.10
    maximum_refinement_gap_ratio: float = 0.60
    maximum_refinement_shift_ratio: float = 0.04
    maximum_orientation_change_degrees: float = 12.0
    minimum_refinement_inlier_ratio: float = 0.50
    refinement_huber_iterations: int = 4
    maximum_consistency_endpoint_deviation_ratio: float = 0.005
    adaptive_consistency_candidate_margin: float = 0.08
    adaptive_consistency_entropy: float = 0.72
    adaptive_consistency_second_peak_ratio: float = 0.25

    def __post_init__(self) -> None:
        for name, lower, upper in (
            ("work_max_edge", 64, 4096),
            ("longitudinal_samples", 16, 128),
            ("offset_samples", 17, 129),
            ("normal_distance_samples", 2, 5),
            ("maximum_sample_evaluations", 4096, 131_072),
            ("refinement_huber_iterations", 1, 8),
        ):
            value = getattr(self, name)
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError(f"{name} is outside its bounded range")
        if self.offset_samples % 2 != 1:
            raise ValueError("offset_samples must be odd so zero is represented")
        for name, lower, upper in (
            ("search_band_ratio", 0.003, 0.04),
            ("maximum_normal_distance_ratio", 0.0005, 0.01),
            ("frame_ratio", 0.005, 0.05),
            ("second_peak_separation_ratio", 0.002, 0.02),
            ("minimum_refinement_coverage", 0.10, 1.0),
            ("maximum_refinement_entropy", 0.10, 1.0),
            ("maximum_refinement_second_peak_ratio", 0.01, 1.0),
            ("maximum_refinement_gap_ratio", 0.05, 1.0),
            ("maximum_refinement_shift_ratio", 0.001, 0.065),
            ("maximum_orientation_change_degrees", 0.5, 30.0),
            ("minimum_refinement_inlier_ratio", 0.10, 1.0),
            ("maximum_consistency_endpoint_deviation_ratio", 0.001, 0.03),
            ("adaptive_consistency_candidate_margin", 0.001, 0.50),
            ("adaptive_consistency_entropy", 0.10, 1.0),
            ("adaptive_consistency_second_peak_ratio", 0.01, 1.0),
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise TypeError(f"{name} must be finite")
            if not lower <= float(value) <= upper:
                raise ValueError(f"{name} is outside its bounded range")
        evaluations = (
            self.longitudinal_samples
            * self.offset_samples
            * self.normal_distance_samples
            * 2
        )
        if evaluations > self.maximum_sample_evaluations:
            raise ValueError("boundary profile exceeds the declared work budget")

@dataclass(frozen=True)
class BoundaryProfileEvidence:
    offsets: tuple[float, ...]
    likelihood: tuple[float, ...]
    peak_offset: float
    peak_width: float
    entropy: float
    second_peak_ratio: float
    support_q10: float
    support_q50: float
    support_q90: float
    maximum_gap_ratio: float
    valid_coverage: float

    def __post_init__(self) -> None:
        if not 17 <= len(self.offsets) <= 129 or len(self.offsets) != len(
            self.likelihood
        ):
            raise ValueError("boundary profile arrays exceed their bounded schema")
        if any(not math.isfinite(value) for value in self.offsets):
            raise ValueError("boundary profile offsets must be finite")
        if any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in self.likelihood
        ):
            raise ValueError("boundary profile likelihood must be finite and bounded")
        total = sum(self.likelihood)
        if not (math.isclose(total, 0.0, abs_tol=1e-12) or math.isclose(
            total, 1.0, abs_tol=1e-9
        )):
            raise ValueError("boundary profile likelihood must be normalized")
        if (
            not math.isfinite(self.peak_offset)
            or not math.isfinite(self.peak_width)
            or self.peak_width < 0.0
        ):
            raise ValueError("boundary profile peak statistics must be finite")
        for name in (
            "entropy",
            "second_peak_ratio",
            "support_q10",
            "support_q50",
            "support_q90",
            "maximum_gap_ratio",
            "valid_coverage",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and bounded")

@dataclass(frozen=True)
class BoundaryLineRefinement:
    edge_name: str
    accepted: bool
    refined_corners: tuple[tuple[float, float], ...]
    line_start: tuple[float, float]
    line_end: tuple[float, float]
    profile: BoundaryProfileEvidence
    residual_q50: float
    residual_q90: float
    residual_q95: float
    inlier_ratio: float
    orientation_change_degrees: float
    parent_shift: float
    support_count: int
    refusal_reason: str | None

    def __post_init__(self) -> None:
        if self.edge_name not in EDGE_NAMES:
            raise ValueError("unsupported edge name")
        _validate_quad(self.refined_corners)
        for point in (self.line_start, self.line_end):
            if len(point) != 2 or any(not math.isfinite(value) for value in point):
                raise ValueError("refined line must be finite")
        for name in (
            "residual_q50",
            "residual_q90",
            "residual_q95",
            "inlier_ratio",
            "orientation_change_degrees",
            "parent_shift",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.inlier_ratio > 1.0:
            raise ValueError("inlier_ratio must be bounded")
        if type(self.support_count) is not int or self.support_count < 0:
            raise ValueError("support_count must be a non-negative integer")
        if self.accepted is (self.refusal_reason is not None):
            raise ValueError("refinement acceptance and refusal reason disagree")

@dataclass(frozen=True)
class BoundaryViewMapping:
    view_name: str
    source_size: tuple[int, int]
    view_size: tuple[int, int]
    forward_matrix: tuple[tuple[float, float, float], ...]
    inverse_matrix: tuple[tuple[float, float, float], ...]
    canonical_source_indices: tuple[int, int, int, int]
    source_edge_to_view: tuple[str, str, str, str]

    def __post_init__(self) -> None:
        if self.view_name not in BOUNDARY_VIEW_NAMES:
            raise ValueError("unsupported boundary view")
        if any(
            type(value) is not int or value < 2
            for value in (*self.source_size, *self.view_size)
        ):
            raise ValueError("boundary view dimensions must be positive integers")
        for matrix in (self.forward_matrix, self.inverse_matrix):
            parsed = np.asarray(matrix, dtype=np.float64)
            if parsed.shape != (3, 3) or not np.isfinite(parsed).all():
                raise ValueError("boundary view matrix must be finite 3x3")
        if sorted(self.canonical_source_indices) != [0, 1, 2, 3]:
            raise ValueError("boundary view corner permutation is invalid")
        if sorted(self.source_edge_to_view) != sorted(EDGE_NAMES):
            raise ValueError("boundary view edge mapping is invalid")

@dataclass(frozen=True)
class TransformationConsistencyEvidence:
    view_names: tuple[str, ...]
    peak_offset_dispersion: float
    line_angle_dispersion: float
    maximum_endpoint_deviation: float
    view_agreement_count: int
    failure_count: int
    view_agreement_ratio: float

    def __post_init__(self) -> None:
        if (
            not self.view_names
            or len(set(self.view_names)) != len(self.view_names)
            or any(name not in BOUNDARY_VIEW_NAMES for name in self.view_names)
        ):
            raise ValueError("consistency view names are invalid")
        for name in (
            "peak_offset_dispersion",
            "line_angle_dispersion",
            "maximum_endpoint_deviation",
            "view_agreement_ratio",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and bounded")
        for name in ("view_agreement_count", "failure_count"):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value <= len(self.view_names):
                raise ValueError(f"{name} must be a bounded integer")
        if self.view_agreement_count + self.failure_count > len(self.view_names):
            raise ValueError("agreement and failure counts exceed evaluated views")

    def runtime_features(self) -> dict[str, float]:
        return {
            "consistency_peak_offset_dispersion": self.peak_offset_dispersion,
            "consistency_line_angle_dispersion": self.line_angle_dispersion,
            "consistency_maximum_endpoint_deviation": (
                self.maximum_endpoint_deviation
            ),
            "consistency_view_agreement_ratio": self.view_agreement_ratio,
            "consistency_view_failure_ratio": (
                self.failure_count / len(self.view_names)
            ),
        }

@dataclass(frozen=True)
class MaskBoundaryEvidence:
    peak_offset: float
    zero_crossing_width: float
    entropy: float
    inside_foreground: float
    outside_background: float
    transition_strength: float
    valid_coverage: float
    lab_disagreement: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.peak_offset):
            raise ValueError("mask peak offset must be finite")
        for name in (
            "zero_crossing_width",
            "entropy",
            "inside_foreground",
            "outside_background",
            "transition_strength",
            "valid_coverage",
            "lab_disagreement",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and bounded")

    def runtime_features(self) -> dict[str, float]:
        return {
            "mask_peak_offset": self.peak_offset,
            "mask_zero_crossing_width": self.zero_crossing_width,
            "mask_entropy": self.entropy,
            "mask_inside_foreground": self.inside_foreground,
            "mask_outside_background": self.outside_background,
            "mask_transition_strength": self.transition_strength,
            "mask_valid_coverage": self.valid_coverage,
            "mask_lab_disagreement": self.lab_disagreement,
        }

def mask_boundary_feature_vector(evidence: Mapping[str, Any]) -> tuple[float, ...]:
    """Serialize only the frozen, truth-free mask witness schema."""
    if not isinstance(evidence, Mapping):
        raise TypeError("mask boundary evidence must be a mapping")
    forbidden = _find_forbidden_truth_fields(evidence, prefix="mask")
    if forbidden:
        raise ValueError(
            "forbidden truth-derived field in mask evidence: " + forbidden[0]
        )
    if set(evidence) != set(MASK_BOUNDARY_FEATURE_NAMES):
        raise ValueError("mask boundary evidence fields differ from schema")
    values = tuple(float(evidence[name]) for name in MASK_BOUNDARY_FEATURE_NAMES)
    if any(not math.isfinite(value) for value in values):
        raise ValueError("mask boundary evidence must be finite")
    if not -0.04 <= values[0] <= 0.04 or any(
        not 0.0 <= value <= 1.0 for value in values[1:]
    ):
        raise ValueError("mask boundary evidence is outside bounded ranges")
    return values

def mask_boundary_ranker_feature_vector(
    evidence: Mapping[str, Any],
    *,
    peak_mode: str = "signed",
) -> tuple[float, ...]:
    """Apply a declared peak-distance representation to cached mask evidence."""
    values = mask_boundary_feature_vector(evidence)
    if peak_mode == "signed":
        return values
    if peak_mode == "absolute":
        return (abs(values[0]), *values[1:])
    if peak_mode == "signed_absolute":
        return (values[0], abs(values[0]), *values[1:])
    raise ValueError("unsupported mask peak feature mode")

def _mask_boundary_ranker_feature_names(peak_mode: str) -> tuple[str, ...]:
    if peak_mode == "signed":
        return MASK_BOUNDARY_FEATURE_NAMES
    if peak_mode == "absolute":
        return MASK_BOUNDARY_ABSOLUTE_FEATURE_NAMES
    if peak_mode == "signed_absolute":
        return MASK_BOUNDARY_SIGNED_ABSOLUTE_FEATURE_NAMES
    raise ValueError("unsupported mask peak feature mode")

def _is_forbidden_truth_field(name: str) -> bool:
    lowered = name.lower()
    return (
        lowered in FORBIDDEN_TRUTH_FEATURE_FIELDS
        or lowered.endswith("_strict")
        or lowered.endswith("_catastrophic")
        or "normalized_corner_error" in lowered
        or lowered.endswith("_truth")
        or "strict_candidate" in lowered
        or "first_strict_rank" in lowered
    )

def _find_forbidden_truth_fields(
    value: Any, *, prefix: str = ""
) -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key)
            path = f"{prefix}.{key}" if prefix else key
            if _is_forbidden_truth_field(key):
                found.append(path)
            found.extend(_find_forbidden_truth_fields(item, prefix=path))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            path = f"{prefix}[{index}]"
            found.extend(_find_forbidden_truth_fields(item, prefix=path))
    return found

def boundary_runtime_feature_vector(evidence: Mapping[str, Any]) -> tuple[float, ...]:
    """Serialize only the explicit truth-free scalar whitelist."""
    if not isinstance(evidence, Mapping):
        raise TypeError("boundary runtime evidence must be a mapping")
    forbidden = _find_forbidden_truth_fields(evidence)
    if forbidden:
        raise ValueError(
            "forbidden truth-derived field in runtime evidence: " + forbidden[0]
        )
    result = []
    for name in BOUNDARY_RUNTIME_FEATURE_NAMES:
        value = evidence.get(name, 0.0)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"runtime feature must be numeric: {name}")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"runtime feature must be finite: {name}")
        result.append(numeric)
    return tuple(result)

def _sample_lab(
    lab: np.ndarray, points: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    height, width = lab.shape[:2]
    coordinates = np.asarray(points, dtype=np.float32)
    valid = (
        (coordinates[:, 0] >= 0.0)
        & (coordinates[:, 0] <= width - 1.0)
        & (coordinates[:, 1] >= 0.0)
        & (coordinates[:, 1] <= height - 1.0)
    )
    values = cv2.remap(
        lab,
        coordinates[:, 0].reshape(-1, 1),
        coordinates[:, 1].reshape(-1, 1),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    ).reshape(-1, 3)
    return values.astype(np.float64, copy=False), valid

def _scanner_bed_reference(
    lab: np.ndarray, frame_ratio: float
) -> tuple[np.ndarray, float]:
    """Match the robust scanner-frame convention used by scanner_selector."""
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
    ).astype(np.float64, copy=False)
    light_cut = float(np.percentile(frame[:, 0], 65.0))
    chroma = np.linalg.norm(frame[:, 1:] - 128.0, axis=1)
    chroma_cut = float(np.percentile(chroma, 75.0))
    likely_bed = frame[(frame[:, 0] >= light_cut) & (chroma <= chroma_cut)]
    if len(likely_bed) < 32:
        likely_bed = frame[frame[:, 0] >= light_cut]
    if not len(likely_bed):
        raise ValueError("scanner-bed frame reference has no valid support")
    reference = np.median(likely_bed, axis=0)
    distances = np.linalg.norm(likely_bed - reference, axis=1)
    spread = float(np.percentile(distances, 80.0)) if len(distances) else 0.0
    scale = float(np.clip(6.0 + 2.5 * spread, 8.0, 30.0))
    return reference, scale

def _longest_false_gap(values: np.ndarray) -> float:
    longest = 0
    current = 0
    for supported in values.tolist():
        if supported:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest / len(values) if len(values) else 1.0

def _local_peak_indices(values: np.ndarray) -> list[int]:
    peaks = []
    index = 0
    while index < len(values):
        end = index
        while end + 1 < len(values) and math.isclose(
            float(values[end + 1]),
            float(values[index]),
            rel_tol=1e-10,
            abs_tol=1e-15,
        ):
            end += 1
        value = float(values[index])
        left = float(values[index - 1]) if index > 0 else -1.0
        right = float(values[end + 1]) if end + 1 < len(values) else -1.0
        if value >= left and value >= right and (value > left or value > right):
            peaks.append((index + end) // 2)
        index = end + 1
    return peaks

def _empty_boundary_profile(offsets: np.ndarray) -> BoundaryProfileEvidence:
    return BoundaryProfileEvidence(
        offsets=tuple(float(value) for value in offsets),
        likelihood=tuple(0.0 for _ in offsets),
        peak_offset=0.0,
        peak_width=float(offsets[-1] - offsets[0]),
        entropy=1.0,
        second_peak_ratio=1.0,
        support_q10=0.0,
        support_q50=0.0,
        support_q90=0.0,
        maximum_gap_ratio=1.0,
        valid_coverage=0.0,
    )

def _directional_responses(
    lab: np.ndarray,
    center: np.ndarray,
    normal: np.ndarray,
    distances: np.ndarray,
    bed_lab: np.ndarray,
    bed_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    distance_responses = []
    distance_polarities = []
    distance_validity = []
    for distance in distances:
        inside, inside_valid = _sample_lab(
            lab, center - float(distance) * normal
        )
        outside, outside_valid = _sample_lab(
            lab, center + float(distance) * normal
        )
        valid = inside_valid & outside_valid
        contrast = np.clip(
            np.linalg.norm(outside - inside, axis=1) / 64.0, 0.0, 1.0
        )
        inside_bed_distance = np.linalg.norm(inside - bed_lab, axis=1)
        outside_bed_distance = np.linalg.norm(outside - bed_lab, axis=1)
        outside_similarity = np.exp(
            -0.5 * np.square(outside_bed_distance / bed_scale)
        )
        inside_non_bed = 1.0 - np.exp(
            -0.5 * np.square(inside_bed_distance / bed_scale)
        )
        polarity = np.clip(
            (inside_bed_distance - outside_bed_distance)
            / max(1e-9, 2.0 * bed_scale),
            0.0,
            1.0,
        )
        response = (
            contrast
            * (0.35 + 0.65 * outside_similarity)
            * (0.25 + 0.75 * inside_non_bed)
            * (0.40 + 0.60 * polarity)
        )
        response[~valid] = 0.0
        polarity[~valid] = 0.0
        distance_responses.append(response)
        distance_polarities.append(polarity)
        distance_validity.append(valid)
    return (
        np.stack(distance_responses, axis=0),
        np.stack(distance_polarities, axis=0),
        np.stack(distance_validity, axis=0),
    )

def evaluate_boundary_profile(
    image_bgr: np.ndarray,
    corners: Sequence[Sequence[int | float]],
    edge_name: str,
    config: BoundaryProfileConfig,
) -> BoundaryProfileEvidence:
    """Evaluate a bounded directional photo-to-scanner-bed offset distribution."""
    if not isinstance(config, BoundaryProfileConfig):
        raise TypeError("config must be BoundaryProfileConfig")
    image = np.asarray(image_bgr)
    if (
        image.ndim != 3
        or image.shape[2] != 3
        or image.dtype != np.uint8
        or image.size == 0
    ):
        raise ValueError("image_bgr must be a non-empty uint8 BGR image")
    if edge_name not in EDGE_NAMES:
        raise ValueError("unsupported edge name")
    height, width = image.shape[:2]
    try:
        ordered = validate_quad(
            corners,
            (width, height),
            min_area_ratio=0.0001,
            max_area_ratio=0.9999,
        )
    except (GeometryError, TypeError, ValueError) as exc:
        message = "quadrilateral must contain finite legal geometry"
        if any(
            isinstance(value, (int, float)) and not math.isfinite(float(value))
            for point in corners
            if isinstance(point, (list, tuple))
            for value in point
        ):
            message = "quadrilateral coordinates must be finite"
        raise ValueError(message) from exc
    scale = min(1.0, float(config.work_max_edge) / max(width, height))
    work_width = max(2, int(round(width * scale)))
    work_height = max(2, int(round(height * scale)))
    work = (
        image
        if scale == 1.0
        else cv2.resize(
            image, (work_width, work_height), interpolation=cv2.INTER_AREA
        )
    )
    lab = cv2.cvtColor(work, cv2.COLOR_BGR2LAB).astype(np.float64)
    bed_lab, bed_scale = _scanner_bed_reference(lab, float(config.frame_ratio))
    work_corners = np.asarray(ordered, dtype=np.float64) * scale
    centroid = np.mean(work_corners, axis=0)
    edge_index = EDGE_NAMES.index(edge_name)
    start = work_corners[edge_index]
    end = work_corners[(edge_index + 1) % 4]
    direction = end - start
    edge_length = float(np.linalg.norm(direction))
    if not math.isfinite(edge_length) or edge_length < 4.0:
        raise ValueError("quadrilateral edge is too short for a bounded profile")
    direction /= edge_length
    normal = np.asarray((direction[1], -direction[0]), dtype=np.float64)
    if float(np.dot(normal, centroid - (start + end) * 0.5)) > 0.0:
        normal *= -1.0
    longitudinal = np.linspace(
        0.06,
        0.94,
        config.longitudinal_samples,
        dtype=np.float64,
    )[:, None]
    base = start + longitudinal * (end - start)
    diagonal = math.hypot(work_width, work_height)
    offset_ratios = np.linspace(
        -float(config.search_band_ratio),
        float(config.search_band_ratio),
        config.offset_samples,
        dtype=np.float64,
    )
    offsets_px = offset_ratios * diagonal
    first_distance = 1.0
    last_distance = max(
        4.0, float(config.maximum_normal_distance_ratio) * diagonal
    )
    distances = np.linspace(
        first_distance,
        last_distance,
        config.normal_distance_samples,
        dtype=np.float64,
    )

    raw_scores = np.zeros(config.offset_samples, dtype=np.float64)
    per_position_scores = np.zeros(
        (config.offset_samples, config.longitudinal_samples), dtype=np.float64
    )
    per_position_valid = np.zeros(
        (config.offset_samples, config.longitudinal_samples), dtype=bool
    )
    pair_coverages = np.zeros(config.offset_samples, dtype=np.float64)
    for offset_index, offset in enumerate(offsets_px):
        center = base + float(offset) * normal
        responses, polarities, valid_pairs = _directional_responses(
            lab, center, normal, distances, bed_lab, bed_scale
        )
        valid_positions = np.any(valid_pairs, axis=0)
        pair_coverages[offset_index] = float(np.mean(valid_pairs))
        per_position_valid[offset_index] = valid_positions
        for position in np.flatnonzero(valid_positions):
            available = valid_pairs[:, position]
            weights = 1.0 / distances[available]
            weights /= np.sum(weights)
            per_position_scores[offset_index, position] = float(
                np.dot(weights, responses[available, position])
            )
        if np.count_nonzero(valid_positions) < max(
            8, config.longitudinal_samples // 4
        ):
            continue
        supported_values = per_position_scores[offset_index, valid_positions]
        q35 = float(np.quantile(supported_values, 0.35))
        q50 = float(np.quantile(supported_values, 0.50))
        polarity_consistency = float(
            np.mean(np.max(polarities[:, valid_positions], axis=0) >= 0.15)
        )
        raw_scores[offset_index] = (
            (0.65 * q35 + 0.35 * q50)
            * math.sqrt(pair_coverages[offset_index])
            * (0.60 + 0.40 * polarity_consistency)
        )

    smoothed = np.convolve(
        raw_scores, np.asarray((0.20, 0.60, 0.20)), mode="same"
    )
    strongest = float(np.max(smoothed))
    if not math.isfinite(strongest) or strongest <= 1e-12:
        return _empty_boundary_profile(offset_ratios)
    sharpened = np.square(np.maximum(smoothed - 0.05 * strongest, 0.0))
    total = float(np.sum(sharpened))
    if not math.isfinite(total) or total <= 1e-15:
        return _empty_boundary_profile(offset_ratios)
    likelihood = sharpened / total
    peak_index = min(
        range(len(likelihood)),
        key=lambda index: (
            -float(likelihood[index]),
            abs(float(offset_ratios[index])),
            -float(offset_ratios[index]),
        ),
    )
    half_height = 0.5 * float(likelihood[peak_index])
    lower = peak_index
    upper = peak_index
    while lower > 0 and likelihood[lower - 1] >= half_height:
        lower -= 1
    while upper + 1 < len(likelihood) and likelihood[upper + 1] >= half_height:
        upper += 1
    peak_width = float(offset_ratios[upper] - offset_ratios[lower])
    positive = likelihood[likelihood > 0.0]
    entropy = float(
        -np.sum(positive * np.log(positive)) / math.log(len(likelihood))
    )
    separation_steps = max(
        2,
        int(
            math.ceil(
                float(config.second_peak_separation_ratio)
                / float(offset_ratios[1] - offset_ratios[0])
            )
        ),
    )
    peaks = sorted(
        _local_peak_indices(likelihood),
        key=lambda index: (
            -float(likelihood[index]),
            abs(float(offset_ratios[index])),
            -float(offset_ratios[index]),
        ),
    )
    second = next(
        (
            index
            for index in peaks
            if abs(index - peak_index) >= separation_steps
        ),
        None,
    )
    second_peak_ratio = (
        0.0
        if second is None
        else float(likelihood[second] / likelihood[peak_index])
    )
    position_values = per_position_scores[peak_index]
    valid_positions = per_position_valid[peak_index]
    quantile_values = position_values[valid_positions]
    if len(quantile_values):
        support_q10, support_q50, support_q90 = (
            float(value)
            for value in np.quantile(quantile_values, (0.10, 0.50, 0.90))
        )
    else:
        support_q10 = support_q50 = support_q90 = 0.0
    support_threshold = max(0.12, 0.35 * support_q50)
    longitudinal_support = valid_positions & (
        position_values >= support_threshold
    )
    return BoundaryProfileEvidence(
        offsets=tuple(float(value) for value in offset_ratios),
        likelihood=tuple(float(value) for value in likelihood),
        peak_offset=float(offset_ratios[peak_index]),
        peak_width=peak_width,
        entropy=float(np.clip(entropy, 0.0, 1.0)),
        second_peak_ratio=float(np.clip(second_peak_ratio, 0.0, 1.0)),
        support_q10=float(np.clip(support_q10, 0.0, 1.0)),
        support_q50=float(np.clip(support_q50, 0.0, 1.0)),
        support_q90=float(np.clip(support_q90, 0.0, 1.0)),
        maximum_gap_ratio=float(
            np.clip(_longest_false_gap(longitudinal_support), 0.0, 1.0)
        ),
        valid_coverage=float(np.clip(pair_coverages[peak_index], 0.0, 1.0)),
    )

def _refinement_refusal(
    *,
    edge_name: str,
    parent_corners: Sequence[Sequence[int | float]],
    profile: BoundaryProfileEvidence,
    reason: str,
    residuals: tuple[float, float, float] = (0.0, 0.0, 0.0),
    inlier_ratio: float = 0.0,
    orientation_change_degrees: float = 0.0,
    parent_shift: float = 0.0,
    support_count: int = 0,
) -> BoundaryLineRefinement:
    corners = tuple(
        (float(point[0]), float(point[1])) for point in parent_corners
    )
    edge_index = EDGE_NAMES.index(edge_name)
    return BoundaryLineRefinement(
        edge_name=edge_name,
        accepted=False,
        refined_corners=corners,
        line_start=corners[edge_index],
        line_end=corners[(edge_index + 1) % 4],
        profile=profile,
        residual_q50=residuals[0],
        residual_q90=residuals[1],
        residual_q95=residuals[2],
        inlier_ratio=inlier_ratio,
        orientation_change_degrees=orientation_change_degrees,
        parent_shift=parent_shift,
        support_count=support_count,
        refusal_reason=reason,
    )

def _weighted_robust_tls(
    points: np.ndarray,
    base_weights: np.ndarray,
    parent_direction: np.ndarray,
    *,
    iterations: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    weights = np.asarray(base_weights, dtype=np.float64).copy()
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2 or points.shape != (len(points), 2):
        raise ValueError("robust line fit requires two-dimensional points")
    if (
        len(weights) != len(points)
        or not np.isfinite(points).all()
        or not np.isfinite(weights).all()
        or np.any(weights <= 0.0)
    ):
        raise ValueError("robust line fit inputs must be finite and positive")
    direction = np.asarray(parent_direction, dtype=np.float64)
    anchor = np.average(points, axis=0, weights=weights)
    residuals = np.zeros(len(points), dtype=np.float64)
    for _ in range(iterations):
        weight_sum = float(np.sum(weights))
        if weight_sum <= 1e-12:
            raise ValueError("robust line fit lost all support")
        anchor = np.sum(points * weights[:, None], axis=0) / weight_sum
        centered = points - anchor
        covariance = (centered * weights[:, None]).T @ centered / weight_sum
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        direction = eigenvectors[:, int(np.argmax(eigenvalues))]
        if float(np.dot(direction, parent_direction)) < 0.0:
            direction *= -1.0
        direction /= np.linalg.norm(direction)
        normal = np.asarray((direction[1], -direction[0]), dtype=np.float64)
        residuals = np.abs(centered @ normal)
        median = float(np.median(residuals))
        scale = max(0.25, 1.4826 * median)
        cutoff = 1.5 * scale
        robust = np.ones_like(residuals)
        mask = residuals > cutoff
        robust[mask] = cutoff / residuals[mask]
        weights = base_weights * robust
    return anchor, direction, residuals, weights

def refine_boundary_line(
    image_bgr: np.ndarray,
    corners: Sequence[Sequence[int | float]],
    edge_name: str,
    config: BoundaryProfileConfig,
) -> BoundaryLineRefinement:
    """Fit one deterministic scanner-aware line or explicitly keep its parent."""
    profile = evaluate_boundary_profile(image_bgr, corners, edge_name, config)
    image = np.asarray(image_bgr)
    height, width = image.shape[:2]
    ordered = validate_quad(
        corners,
        (width, height),
        min_area_ratio=0.0001,
        max_area_ratio=0.9999,
    )
    if (
        not any(profile.likelihood)
        or profile.valid_coverage < float(config.minimum_refinement_coverage)
        or profile.support_q50 < 0.08
        or profile.maximum_gap_ratio > float(config.maximum_refinement_gap_ratio)
    ):
        return _refinement_refusal(
            edge_name=edge_name,
            parent_corners=ordered,
            profile=profile,
            reason="insufficient_profile_support",
        )
    if (
        profile.entropy > float(config.maximum_refinement_entropy)
        or profile.second_peak_ratio
        > float(config.maximum_refinement_second_peak_ratio)
    ):
        return _refinement_refusal(
            edge_name=edge_name,
            parent_corners=ordered,
            profile=profile,
            reason="ambiguous_profile",
        )

    scale = min(1.0, float(config.work_max_edge) / max(width, height))
    work_width = max(2, int(round(width * scale)))
    work_height = max(2, int(round(height * scale)))
    work = (
        image
        if scale == 1.0
        else cv2.resize(
            image, (work_width, work_height), interpolation=cv2.INTER_AREA
        )
    )
    lab = cv2.cvtColor(work, cv2.COLOR_BGR2LAB).astype(np.float64)
    bed_lab, bed_scale = _scanner_bed_reference(lab, float(config.frame_ratio))
    work_corners = np.asarray(ordered, dtype=np.float64) * scale
    edge_index = EDGE_NAMES.index(edge_name)
    start = work_corners[edge_index]
    end = work_corners[(edge_index + 1) % 4]
    parent_direction = end - start
    parent_length = float(np.linalg.norm(parent_direction))
    parent_direction /= parent_length
    centroid = np.mean(work_corners, axis=0)
    normal = np.asarray(
        (parent_direction[1], -parent_direction[0]), dtype=np.float64
    )
    if float(np.dot(normal, centroid - (start + end) * 0.5)) > 0.0:
        normal *= -1.0
    longitudinal = np.linspace(
        0.06,
        0.94,
        config.longitudinal_samples,
        dtype=np.float64,
    )[:, None]
    base = start + longitudinal * (end - start)
    diagonal = math.hypot(work_width, work_height)
    offset_ratios = np.asarray(profile.offsets, dtype=np.float64)
    offsets_px = offset_ratios * diagonal
    last_distance = max(
        4.0, float(config.maximum_normal_distance_ratio) * diagonal
    )
    distances = np.linspace(
        1.0,
        last_distance,
        config.normal_distance_samples,
        dtype=np.float64,
    )
    local_scores = np.zeros(
        (config.offset_samples, config.longitudinal_samples), dtype=np.float64
    )
    local_valid = np.zeros_like(local_scores, dtype=bool)
    distance_weights = 1.0 / distances
    for offset_index, offset in enumerate(offsets_px):
        responses, _, valid_pairs = _directional_responses(
            lab,
            base + float(offset) * normal,
            normal,
            distances,
            bed_lab,
            bed_scale,
        )
        for position in range(config.longitudinal_samples):
            available = valid_pairs[:, position]
            if not np.any(available):
                continue
            weights = distance_weights[available]
            weights /= np.sum(weights)
            local_scores[offset_index, position] = float(
                np.dot(weights, responses[available, position])
            )
            local_valid[offset_index, position] = True
    likelihood = np.asarray(profile.likelihood, dtype=np.float64)
    likelihood_scale = likelihood / max(float(np.max(likelihood)), 1e-12)
    ranked_scores = local_scores * (0.20 + 0.80 * likelihood_scale[:, None])
    separation_steps = max(
        2,
        int(
            math.ceil(
                float(config.second_peak_separation_ratio)
                / float(offset_ratios[1] - offset_ratios[0])
            )
        ),
    )
    points = []
    base_weights = []
    minimum_local_score = max(0.08, 0.20 * profile.support_q50)
    for position in range(config.longitudinal_samples):
        available_indices = np.flatnonzero(local_valid[:, position])
        if not len(available_indices):
            continue
        best = min(
            available_indices.tolist(),
            key=lambda index: (
                -float(ranked_scores[index, position]),
                abs(float(offset_ratios[index] - profile.peak_offset)),
                abs(float(offset_ratios[index])),
            ),
        )
        best_score = float(local_scores[best, position])
        if best_score < minimum_local_score:
            continue
        separated = [
            index
            for index in available_indices
            if abs(int(index) - int(best)) >= separation_steps
        ]
        second_score = max(
            (float(ranked_scores[index, position]) for index in separated),
            default=0.0,
        )
        best_ranked = float(ranked_scores[best, position])
        sharpness = float(
            np.clip(
                (best_ranked - second_score) / max(best_ranked, 1e-12),
                0.0,
                1.0,
            )
        )
        points.append(base[position] + offsets_px[best] * normal)
        base_weights.append(best_score * (0.25 + 0.75 * sharpness))
    minimum_support = max(8, config.longitudinal_samples // 3)
    if len(points) < minimum_support:
        return _refinement_refusal(
            edge_name=edge_name,
            parent_corners=ordered,
            profile=profile,
            reason="insufficient_profile_support",
            support_count=len(points),
        )
    points_array = np.asarray(points, dtype=np.float64)
    weights_array = np.asarray(base_weights, dtype=np.float64)
    try:
        anchor, direction, residuals_px, _ = _weighted_robust_tls(
            points_array,
            weights_array,
            parent_direction,
            iterations=config.refinement_huber_iterations,
        )
    except (FloatingPointError, ValueError, np.linalg.LinAlgError):
        return _refinement_refusal(
            edge_name=edge_name,
            parent_corners=ordered,
            profile=profile,
            reason="robust_fit_failed",
            support_count=len(points),
        )
    residual_quantiles_px = tuple(
        float(value) for value in np.quantile(residuals_px, (0.50, 0.90, 0.95))
    )
    residual_quantiles = tuple(value / diagonal for value in residual_quantiles_px)
    inlier_cutoff = max(0.75, 2.5 * residual_quantiles_px[0])
    inlier_ratio = float(np.mean(residuals_px <= inlier_cutoff))
    cosine = float(np.clip(abs(np.dot(direction, parent_direction)), 0.0, 1.0))
    orientation_change = math.degrees(math.acos(cosine))
    projected = []
    for endpoint in (start, end):
        projected.append(anchor + np.dot(endpoint - anchor, direction) * direction)
    projected_source = np.asarray(projected, dtype=np.float64) / scale
    parent_source = np.asarray((ordered[edge_index], ordered[(edge_index + 1) % 4]))
    source_diagonal = math.hypot(width, height)
    parent_shift = float(
        np.max(np.linalg.norm(projected_source - parent_source, axis=1))
        / source_diagonal
    )
    refusal_kwargs = {
        "edge_name": edge_name,
        "parent_corners": ordered,
        "profile": profile,
        "residuals": residual_quantiles,
        "inlier_ratio": inlier_ratio,
        "orientation_change_degrees": orientation_change,
        "parent_shift": parent_shift,
        "support_count": len(points),
    }
    if parent_shift > float(config.maximum_refinement_shift_ratio):
        return _refinement_refusal(
            **refusal_kwargs, reason="maximum_shift_exceeded"
        )
    if orientation_change > float(config.maximum_orientation_change_degrees):
        return _refinement_refusal(
            **refusal_kwargs, reason="maximum_orientation_change_exceeded"
        )
    if inlier_ratio < float(config.minimum_refinement_inlier_ratio):
        return _refinement_refusal(
            **refusal_kwargs, reason="insufficient_fit_inliers"
        )
    refined = np.asarray(ordered, dtype=np.float64)
    refined[edge_index] = projected_source[0]
    refined[(edge_index + 1) % 4] = projected_source[1]
    refined_corners = tuple(
        (float(point[0]), float(point[1])) for point in refined
    )
    return BoundaryLineRefinement(
        edge_name=edge_name,
        accepted=True,
        refined_corners=refined_corners,
        line_start=refined_corners[edge_index],
        line_end=refined_corners[(edge_index + 1) % 4],
        profile=profile,
        residual_q50=residual_quantiles[0],
        residual_q90=residual_quantiles[1],
        residual_q95=residual_quantiles[2],
        inlier_ratio=inlier_ratio,
        orientation_change_degrees=orientation_change,
        parent_shift=parent_shift,
        support_count=len(points),
        refusal_reason=None,
    )

def _homogeneous_points(
    points: Sequence[Sequence[int | float]],
    matrix: Sequence[Sequence[int | float]],
) -> np.ndarray:
    parsed = np.asarray(points, dtype=np.float64)
    transform = np.asarray(matrix, dtype=np.float64)
    if parsed.shape != (4, 2) or not np.isfinite(parsed).all():
        raise ValueError("boundary view corners must be a finite quadrilateral")
    if transform.shape != (3, 3) or not np.isfinite(transform).all():
        raise ValueError("boundary view matrix must be finite 3x3")
    homogeneous = np.column_stack((parsed, np.ones(4, dtype=np.float64)))
    projected = homogeneous @ transform.T
    if np.any(np.abs(projected[:, 2]) < 1e-12):
        raise ValueError("boundary view transform projects a point to infinity")
    return projected[:, :2] / projected[:, 2, None]

def _matrix_tuple(matrix: np.ndarray) -> tuple[tuple[float, float, float], ...]:
    return tuple(tuple(float(value) for value in row) for row in matrix)

def transform_boundary_view(
    image_bgr: np.ndarray,
    corners: Sequence[Sequence[int | float]],
    view_name: str,
) -> tuple[np.ndarray, tuple[tuple[float, float], ...], BoundaryViewMapping]:
    """Apply one fixed diagnostic view and preserve an exact coordinate inverse."""
    if view_name not in BOUNDARY_VIEW_NAMES:
        raise ValueError("unsupported boundary view")
    image = np.asarray(image_bgr)
    if (
        image.ndim != 3
        or image.shape[2] != 3
        or image.dtype != np.uint8
        or image.size == 0
    ):
        raise ValueError("image_bgr must be a non-empty uint8 BGR image")
    height, width = image.shape[:2]
    ordered = validate_quad(
        corners,
        (width, height),
        min_area_ratio=0.0001,
        max_area_ratio=0.9999,
    )
    forward = np.eye(3, dtype=np.float64)
    permutation = (0, 1, 2, 3)
    edge_mapping = tuple(EDGE_NAMES)
    view = image.copy()
    view_width, view_height = width, height
    if view_name == "horizontal_flip":
        view = cv2.flip(image, 1)
        forward[0, 0] = -1.0
        forward[0, 2] = width - 1.0
        permutation = (1, 0, 3, 2)
        edge_mapping = ("top", "left", "bottom", "right")
    elif view_name == "area_downscale_075":
        view_width = max(2, int(round(width * 0.75)))
        view_height = max(2, int(round(height * 0.75)))
        scale_x = (view_width - 1.0) / (width - 1.0)
        scale_y = (view_height - 1.0) / (height - 1.0)
        forward[0, 0] = scale_x
        forward[1, 1] = scale_y
        view = cv2.resize(
            image,
            (view_width, view_height),
            interpolation=cv2.INTER_AREA,
        )
    elif view_name in {"gamma_085", "gamma_115"}:
        gamma = 0.85 if view_name == "gamma_085" else 1.15
        values = np.arange(256, dtype=np.float64) / 255.0
        lookup = np.rint(np.power(values, gamma) * 255.0).astype(np.uint8)
        view = cv2.LUT(image, lookup)
    elif view_name == "gaussian_blur":
        view = cv2.GaussianBlur(
            image,
            (5, 5),
            sigmaX=1.0,
            sigmaY=1.0,
            borderType=cv2.BORDER_REPLICATE,
        )
    inverse = np.linalg.inv(forward)
    source = np.asarray(ordered, dtype=np.float64)
    transformed = _homogeneous_points(source[list(permutation)], forward)
    mapping = BoundaryViewMapping(
        view_name=view_name,
        source_size=(width, height),
        view_size=(view_width, view_height),
        forward_matrix=_matrix_tuple(forward),
        inverse_matrix=_matrix_tuple(inverse),
        canonical_source_indices=permutation,
        source_edge_to_view=edge_mapping,
    )
    return (
        view,
        tuple((float(point[0]), float(point[1])) for point in transformed),
        mapping,
    )

def map_corners_from_boundary_view(
    corners: Sequence[Sequence[int | float]],
    mapping: BoundaryViewMapping,
) -> tuple[tuple[float, float], ...]:
    """Map canonical view corners back to the source canonical ordering."""
    if not isinstance(mapping, BoundaryViewMapping):
        raise TypeError("mapping must be BoundaryViewMapping")
    restored_view_order = _homogeneous_points(corners, mapping.inverse_matrix)
    restored = np.zeros((4, 2), dtype=np.float64)
    for view_index, source_index in enumerate(mapping.canonical_source_indices):
        restored[source_index] = restored_view_order[view_index]
    return tuple((float(point[0]), float(point[1])) for point in restored)

def _undirected_angle_dispersion(angles: Sequence[float]) -> float:
    if len(angles) < 2:
        return 0.0
    maximum = 0.0
    for left, right in combinations(angles, 2):
        difference = abs(float(left) - float(right)) % math.pi
        maximum = max(maximum, min(difference, math.pi - difference))
    return float(np.clip(maximum / (math.pi / 2.0), 0.0, 1.0))

def evaluate_transformation_consistency(
    image_bgr: np.ndarray,
    corners: Sequence[Sequence[int | float]],
    edge_name: str,
    config: BoundaryProfileConfig,
    *,
    view_names: Sequence[str] = BOUNDARY_VIEW_NAMES,
) -> TransformationConsistencyEvidence:
    """Measure refinement stability across a fixed, bounded diagnostic view set."""
    names = tuple(view_names)
    if (
        not names
        or len(set(names)) != len(names)
        or any(name not in BOUNDARY_VIEW_NAMES for name in names)
    ):
        raise ValueError("consistency views must be unique supported names")
    if edge_name not in EDGE_NAMES:
        raise ValueError("unsupported edge name")
    if not isinstance(config, BoundaryProfileConfig):
        raise TypeError("config must be BoundaryProfileConfig")
    image = np.asarray(image_bgr)
    height, width = image.shape[:2]
    source_diagonal = math.hypot(width, height)
    edge_index = EDGE_NAMES.index(edge_name)
    successful: list[dict[str, Any]] = []
    for name in names:
        try:
            view, view_corners, mapping = transform_boundary_view(
                image, corners, name
            )
        except GeometryError:
            # A selected edge option can extend a fitted line just outside the
            # decoded image.  Consistency is a safety witness, so invalid view
            # geometry must count as disagreement instead of aborting the run.
            continue
        view_edge = mapping.source_edge_to_view[edge_index]
        refinement = refine_boundary_line(
            view,
            view_corners,
            view_edge,
            config,
        )
        if not refinement.accepted:
            continue
        restored = map_corners_from_boundary_view(
            refinement.refined_corners,
            mapping,
        )
        endpoints = np.asarray(
            (restored[edge_index], restored[(edge_index + 1) % 4]),
            dtype=np.float64,
        )
        direction = endpoints[1] - endpoints[0]
        if not np.isfinite(endpoints).all() or np.linalg.norm(direction) <= 1e-9:
            continue
        successful.append({
            "view_name": name,
            "peak_offset": float(refinement.profile.peak_offset),
            "angle": math.atan2(float(direction[1]), float(direction[0])),
            "endpoints": endpoints,
        })
    failure_count = len(names) - len(successful)
    failure_ratio = failure_count / len(names)
    peak_values = [item["peak_offset"] for item in successful]
    raw_peak_dispersion = (
        (max(peak_values) - min(peak_values))
        / max(1e-12, 2.0 * float(config.search_band_ratio))
        if len(peak_values) >= 2
        else 0.0
    )
    raw_angle_dispersion = _undirected_angle_dispersion(
        [item["angle"] for item in successful]
    )
    reference = next(
        (item for item in successful if item["view_name"] == "identity"),
        None,
    )
    deviations: list[float] = []
    if reference is not None:
        reference_endpoints = reference["endpoints"]
        for item in successful:
            deviations.append(float(
                np.max(
                    np.linalg.norm(
                        item["endpoints"] - reference_endpoints,
                        axis=1,
                    )
                )
                / source_diagonal
            ))
    raw_endpoint_deviation = max(deviations, default=0.0)
    if failure_count:
        raw_peak_dispersion = max(raw_peak_dispersion, failure_ratio)
        raw_angle_dispersion = max(raw_angle_dispersion, failure_ratio)
        raw_endpoint_deviation = max(raw_endpoint_deviation, failure_ratio)
    agreement_count = (
        sum(
            value
            <= float(config.maximum_consistency_endpoint_deviation_ratio)
            for value in deviations
        )
        if reference is not None
        else 0
    )
    return TransformationConsistencyEvidence(
        view_names=names,
        peak_offset_dispersion=float(np.clip(raw_peak_dispersion, 0.0, 1.0)),
        line_angle_dispersion=float(np.clip(raw_angle_dispersion, 0.0, 1.0)),
        maximum_endpoint_deviation=float(
            np.clip(raw_endpoint_deviation, 0.0, 1.0)
        ),
        view_agreement_count=agreement_count,
        failure_count=failure_count,
        view_agreement_ratio=agreement_count / len(names),
    )

def _sample_probability(
    probability: np.ndarray,
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = probability.shape
    coordinates = np.asarray(points, dtype=np.float32)
    valid = (
        (coordinates[:, 0] >= 0.0)
        & (coordinates[:, 0] <= width - 1.0)
        & (coordinates[:, 1] >= 0.0)
        & (coordinates[:, 1] <= height - 1.0)
    )
    values = cv2.remap(
        probability,
        coordinates[:, 0].reshape(-1, 1),
        coordinates[:, 1].reshape(-1, 1),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    ).reshape(-1)
    return values.astype(np.float64, copy=False), valid

def evaluate_mask_boundary_profile(
    mask_probability: np.ndarray,
    corners: Sequence[Sequence[int | float]],
    edge_name: str,
    config: BoundaryProfileConfig,
    *,
    lab_peak_offset: float | None = None,
) -> MaskBoundaryEvidence:
    """Measure a bounded foreground-to-background mask transition on one edge."""
    probability = np.asarray(mask_probability)
    if (
        probability.ndim != 2
        or not np.issubdtype(probability.dtype, np.number)
        or probability.size == 0
        or not np.isfinite(probability).all()
        or np.any(probability < 0.0)
        or np.any(probability > 1.0)
    ):
        raise ValueError("mask probability must be a finite 2D array in [0, 1]")
    if edge_name not in EDGE_NAMES:
        raise ValueError("unsupported edge name")
    if not isinstance(config, BoundaryProfileConfig):
        raise TypeError("config must be BoundaryProfileConfig")
    if lab_peak_offset is not None and (
        isinstance(lab_peak_offset, bool)
        or not isinstance(lab_peak_offset, (int, float))
        or not math.isfinite(float(lab_peak_offset))
    ):
        raise ValueError("lab_peak_offset must be finite when provided")
    height, width = probability.shape
    raw_corners = np.asarray(corners, dtype=np.float64)
    if raw_corners.shape != (4, 2) or not np.isfinite(raw_corners).all():
        raise ValueError("mask witness quadrilateral must be finite 4x2 geometry")
    clipped_corners = raw_corners.copy()
    clipped_corners[:, 0] = np.clip(clipped_corners[:, 0], 0.0, width - 1.0)
    clipped_corners[:, 1] = np.clip(clipped_corners[:, 1], 0.0, height - 1.0)
    try:
        ordered = np.asarray(
            validate_quad(
                clipped_corners,
                (width, height),
                min_area_ratio=0.0001,
                max_area_ratio=0.9999,
            ),
            dtype=np.float64,
        )
    except (GeometryError, TypeError, ValueError) as exc:
        raise ValueError("mask witness quadrilateral is not legal after clipping") from exc
    edge_index = EDGE_NAMES.index(edge_name)
    start = ordered[edge_index]
    end = ordered[(edge_index + 1) % 4]
    direction = end - start
    direction /= np.linalg.norm(direction)
    normal = np.asarray((direction[1], -direction[0]), dtype=np.float64)
    centroid = np.mean(ordered, axis=0)
    if float(np.dot(normal, centroid - (start + end) * 0.5)) > 0.0:
        normal *= -1.0
    longitudinal = np.linspace(
        0.06, 0.94, config.longitudinal_samples, dtype=np.float64
    )[:, None]
    base = start + longitudinal * (end - start)
    diagonal = math.hypot(width, height)
    offsets = np.linspace(
        -float(config.search_band_ratio),
        float(config.search_band_ratio),
        config.offset_samples,
        dtype=np.float64,
    )
    distances = np.linspace(
        1.0,
        max(2.0, float(config.maximum_normal_distance_ratio) * diagonal),
        config.normal_distance_samples,
        dtype=np.float64,
    )
    responses = []
    coverage = []
    foreground_values = []
    background_values = []
    for offset in offsets:
        center = base + float(offset * diagonal) * normal
        distance_responses = []
        distance_valid = []
        distance_inside = []
        distance_outside = []
        for distance in distances:
            inside, inside_valid = _sample_probability(
                probability, center - float(distance) * normal
            )
            outside, outside_valid = _sample_probability(
                probability, center + float(distance) * normal
            )
            valid = inside_valid & outside_valid
            polarity = np.clip(inside - outside, 0.0, 1.0)
            response = polarity * (0.5 * (inside + (1.0 - outside)))
            distance_responses.append(response)
            distance_valid.append(valid)
            distance_inside.append(inside)
            distance_outside.append(outside)
        stacked = np.stack(distance_responses)
        valid = np.stack(distance_valid)
        if np.any(valid):
            responses.append(float(np.mean(stacked[valid])))
            foreground_values.append(float(np.mean(np.stack(distance_inside)[valid])))
            background_values.append(float(np.mean(1.0 - np.stack(distance_outside)[valid])))
        else:
            responses.append(0.0)
            foreground_values.append(0.0)
            background_values.append(0.0)
        coverage.append(float(np.mean(valid)))
    raw = np.asarray(responses, dtype=np.float64)
    total = float(np.sum(raw))
    if total <= 1e-12:
        return MaskBoundaryEvidence(
            peak_offset=0.0,
            zero_crossing_width=1.0,
            entropy=1.0,
            inside_foreground=0.0,
            outside_background=0.0,
            transition_strength=0.0,
            valid_coverage=float(np.clip(max(coverage, default=0.0), 0.0, 1.0)),
            lab_disagreement=1.0 if lab_peak_offset is not None else 0.0,
        )
    likelihood = raw / total
    peak_index = min(
        range(len(offsets)),
        key=lambda index: (-raw[index], abs(offsets[index]), index),
    )
    peak_offset = float(offsets[peak_index])
    entropy = float(
        -np.sum(likelihood * np.log(np.maximum(likelihood, 1e-15)))
        / math.log(len(likelihood))
    )
    variance = float(np.sum(likelihood * np.square(offsets - peak_offset)))
    width = math.sqrt(max(0.0, variance)) / max(
        1e-12, float(config.search_band_ratio)
    )
    disagreement = (
        abs(peak_offset - float(lab_peak_offset))
        / max(1e-12, 2.0 * float(config.search_band_ratio))
        if lab_peak_offset is not None
        else 0.0
    )
    return MaskBoundaryEvidence(
        peak_offset=peak_offset,
        zero_crossing_width=float(np.clip(width, 0.0, 1.0)),
        entropy=float(np.clip(entropy, 0.0, 1.0)),
        inside_foreground=float(
            np.clip(foreground_values[peak_index], 0.0, 1.0)
        ),
        outside_background=float(
            np.clip(background_values[peak_index], 0.0, 1.0)
        ),
        transition_strength=float(np.clip(raw[peak_index], 0.0, 1.0)),
        valid_coverage=float(np.clip(coverage[peak_index], 0.0, 1.0)),
        lab_disagreement=float(np.clip(disagreement, 0.0, 1.0)),
    )

def select_adaptive_consistency_views(
    identity_profile: BoundaryProfileEvidence,
    *,
    candidate_margin: float,
    config: BoundaryProfileConfig,
) -> tuple[str, ...]:
    """Activate diagnostic transforms only when identity evidence is ambiguous."""
    if not isinstance(identity_profile, BoundaryProfileEvidence):
        raise TypeError("identity_profile must be BoundaryProfileEvidence")
    if not isinstance(config, BoundaryProfileConfig):
        raise TypeError("config must be BoundaryProfileConfig")
    if (
        isinstance(candidate_margin, bool)
        or not isinstance(candidate_margin, (int, float))
        or not math.isfinite(float(candidate_margin))
        or float(candidate_margin) < 0.0
    ):
        raise ValueError("candidate_margin must be finite and non-negative")
    ambiguous = (
        float(candidate_margin)
        <= float(config.adaptive_consistency_candidate_margin)
        or identity_profile.entropy >= float(config.adaptive_consistency_entropy)
        or identity_profile.second_peak_ratio
        >= float(config.adaptive_consistency_second_peak_ratio)
        or identity_profile.valid_coverage
        < float(config.minimum_refinement_coverage)
    )
    return BOUNDARY_VIEW_NAMES if ambiguous else ("identity",)

def consistency_work_summary(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, int | float]:
    extras = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("consistency work row must be an object")
        all_count = row.get("all_view_count")
        adaptive_count = row.get("adaptive_view_count")
        if (
            type(all_count) is not int
            or type(adaptive_count) is not int
            or not 1 <= adaptive_count <= all_count <= len(BOUNDARY_VIEW_NAMES)
        ):
            raise ValueError("consistency work counts are invalid")
        extras.append(adaptive_count - 1)
    if not extras:
        raise ValueError("consistency work rows are empty")
    values = np.asarray(extras, dtype=np.float64)
    return {
        "sample_count": len(extras),
        "adaptive_extra_view_activation_rate": float(np.mean(values > 0.0)),
        "adaptive_extra_work_mean": float(np.mean(values)),
        "adaptive_extra_work_p95": float(
            np.quantile(values, 0.95, method="higher")
        ),
        "adaptive_extra_work_max": int(np.max(values)),
    }

def _validate_quad(corners: Any) -> None:
    if not isinstance(corners, (list, tuple)) or len(corners) != 4:
        raise ValueError("geometry must be a finite quadrilateral")
    for point in corners:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError("geometry must be a finite quadrilateral")
        for value in point:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("geometry must be a finite quadrilateral")
            if not math.isfinite(float(value)):
                raise ValueError("geometry must be a finite quadrilateral")

__all__ = [
    "BOUNDARY_RUNTIME_FEATURE_NAMES", "BOUNDARY_VIEW_NAMES",
    "MASK_BOUNDARY_ABSOLUTE_FEATURE_NAMES", "MASK_BOUNDARY_FEATURE_NAMES",
    "BoundaryLineRefinement", "BoundaryProfileConfig",
    "BoundaryProfileEvidence", "BoundaryViewMapping",
    "MaskBoundaryEvidence", "TransformationConsistencyEvidence",
    "boundary_runtime_feature_vector", "consistency_work_summary",
    "evaluate_boundary_profile", "evaluate_mask_boundary_profile",
    "evaluate_transformation_consistency", "map_corners_from_boundary_view",
    "mask_boundary_feature_vector", "mask_boundary_ranker_feature_vector",
    "refine_boundary_line", "select_adaptive_consistency_views",
    "transform_boundary_view",
]
