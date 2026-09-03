"""Truth-free, fail-closed inference for the explicit V8.2 research gate."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from .boundary_evidence import (
    BOUNDARY_RUNTIME_FEATURE_NAMES,
    BOUNDARY_VIEW_NAMES,
    EDGE_NAMES,
    FEATURE_NAMES,
    MASK_BOUNDARY_ABSOLUTE_FEATURE_NAMES,
    BoundaryLineRefinement,
    BoundaryProfileConfig,
    BoundaryProfileEvidence,
    _find_forbidden_truth_fields,
    boundary_runtime_feature_vector,
    evaluate_mask_boundary_profile,
    evaluate_transformation_consistency,
    mask_boundary_ranker_feature_vector,
    refine_boundary_line,
)


SOURCE_NAMES = ("edge", "v7_current", "v7_seed", "mask")
EDGE_CORNERS = ((0, 1), (1, 2), (2, 3), (3, 0))


_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_AUTOMATIC_ACTIONS = frozenset({"automatic", "v7_fallback"})
_MANUAL_ACTIONS = frozenset({"manual_review", "rescan_required"})

RELATIVE_ADOPTION_FEATURE_NAMES = (
    "proposal_edge_risk_min",
    "proposal_edge_risk_mean",
    "proposal_edge_risk_max",
    "proposal_edge_risk_spread",
    "proposal_upper_risk_max",
    "proposal_edge_margin_min",
    "proposal_edge_margin_mean",
    "proposal_edge_margin_max",
    "proposal_cross_source_support_min",
    "proposal_cross_source_support_mean",
    "profile_entropy_mean",
    "profile_entropy_max",
    "profile_second_peak_mean",
    "profile_second_peak_max",
    "profile_maximum_gap_max",
    "profile_valid_coverage_min",
    "profile_support_q10_min",
    "profile_absolute_peak_offset_max",
    "refinement_acceptance_ratio",
    "refined_edge_ratio",
    "refinement_residual_q95_max",
    "refinement_inlier_ratio_min",
    "refinement_orientation_change_max",
    "refinement_parent_shift_max",
    "geometry_legal_min_parent_anchor",
    "geometry_legal_both_parent_anchor",
    "geometry_max_min_parent_displacement",
    "current_proposal_corner_distance_mean",
    "current_proposal_corner_distance_max",
    "adaptive_consistency_endpoint_max",
    "adaptive_consistency_agreement_min",
    "adaptive_consistency_agreement_mean",
    "adaptive_consistency_failure_max",
    "adaptive_consistency_failure_mean",
    "all_view_consistency_endpoint_max",
    "all_view_consistency_agreement_min",
    "all_view_consistency_agreement_mean",
    "all_view_consistency_failure_max",
    "all_view_consistency_failure_mean",
    "status_automatic",
    "status_manual_review",
    "reason_edge_base",
    "reason_v7_edge_unresolved_conflict",
    "reason_v7_edge_agreement_rescue",
    "reason_strong_three_way_conflict",
    "reason_other",
)


def _finite_vector(value: Any, *, label: str, length: int) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{label} width differs from its feature schema")
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if any(not math.isfinite(item) for item in result):
        raise ValueError(f"{label} must be finite")
    return result


def _validate_linear_model(
    model: Any,
    *,
    kind: str,
    expected_feature_names: Sequence[str] | None = None,
) -> Mapping[str, Any]:
    if not isinstance(model, Mapping) or model.get("kind") != kind:
        raise ValueError("linear model kind differs from the sealed schema")
    names = model.get("feature_names")
    if (
        not isinstance(names, list)
        or not names
        or len(set(names)) != len(names)
        or any(not isinstance(name, str) or not name for name in names)
    ):
        raise ValueError("linear model feature schema is invalid")
    if expected_feature_names is not None and tuple(names) != tuple(
        expected_feature_names
    ):
        raise ValueError("linear model feature schema differs from the gate")
    mean = _finite_vector(model.get("mean"), label="model mean", length=len(names))
    scale = _finite_vector(model.get("scale"), label="model scale", length=len(names))
    _finite_vector(
        model.get("coefficients"),
        label="model coefficients",
        length=len(names),
    )
    if any(value <= 0.0 for value in scale):
        raise ValueError("model scale must be finite and positive")
    for name in ("intercept", "regularization"):
        value = model.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"model {name} must be finite")
    if kind == "relative_linear_logistic":
        temperature = model.get("temperature", 1.0)
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(float(temperature))
            or float(temperature) <= 0.0
        ):
            raise ValueError("relative model temperature must be finite and positive")
    return MappingProxyType(dict(model))


def predict_boundary_risk(
    model: Mapping[str, Any],
    features: Sequence[int | float],
) -> float:
    """Evaluate the fixed pairwise ranker with strict tensor validation."""
    parsed = _validate_linear_model(model, kind="boundary_pairwise_linear_logistic")
    names = parsed["feature_names"]
    values = np.asarray(
        _finite_vector(features, label="boundary features", length=len(names)),
        dtype=np.float64,
    )
    mean = np.asarray(parsed["mean"], dtype=np.float64)
    scale = np.asarray(parsed["scale"], dtype=np.float64)
    coefficients = np.asarray(parsed["coefficients"], dtype=np.float64)
    result = float((values - mean) / scale @ coefficients + parsed["intercept"])
    if not math.isfinite(result):
        raise ValueError("boundary ranker produced a non-finite score")
    return result


def predict_relative_probability(
    model: Mapping[str, Any],
    features: Sequence[int | float],
) -> float:
    parsed = _validate_linear_model(model, kind="relative_linear_logistic")
    names = parsed["feature_names"]
    values = np.asarray(
        _finite_vector(features, label="relative features", length=len(names)),
        dtype=np.float64,
    )
    mean = np.asarray(parsed["mean"], dtype=np.float64)
    scale = np.asarray(parsed["scale"], dtype=np.float64)
    coefficients = np.asarray(parsed["coefficients"], dtype=np.float64)
    logit = (
        float(coefficients @ ((values - mean) / scale))
        + float(parsed["intercept"])
    ) / float(parsed.get("temperature", 1.0))
    bounded = max(-60.0, min(60.0, logit))
    return 1.0 / (1.0 + math.exp(-bounded))


def _thresholds(value: Any) -> Mapping[str, float]:
    names = (
        "replace_better_min",
        "replace_harmful_max",
        "rescue_safe_min",
        "rescue_harmful_max",
        "minimum_view_agreement_ratio",
    )
    if not isinstance(value, Mapping) or set(value) != set(names):
        raise ValueError("relative gate thresholds differ from the sealed schema")
    parsed = {}
    for name in names:
        raw = value[name]
        if (
            isinstance(raw, bool)
            or not isinstance(raw, (int, float))
            or not math.isfinite(float(raw))
        ):
            raise ValueError("relative gate thresholds must be finite")
        parsed[name] = float(raw)
    if not 0.0 <= parsed["minimum_view_agreement_ratio"] <= 1.0:
        raise ValueError("minimum view agreement must be in [0, 1]")
    return MappingProxyType(parsed)


@dataclass(frozen=True)
class BoundaryQualityArtifact:
    artifact_type: str
    proposal_model: Mapping[str, Any]
    proposal_config: Mapping[str, Any]
    boundary_profile_config: BoundaryProfileConfig
    relative_gate: Mapping[str, Any]
    inputs: Mapping[str, str]
    source_sha256: str | None = None

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
        *,
        source_sha256: str | None = None,
    ) -> "BoundaryQualityArtifact":
        if not isinstance(value, Mapping):
            raise TypeError("boundary-quality artifact must be an object")
        if value.get("schema_version") != 1:
            raise ValueError("unsupported boundary-quality artifact schema")
        artifact_type = value.get("artifact_type")
        if artifact_type != "v8.2_mask_absolute_manual_rescue_final_v6":
            raise ValueError("unsupported boundary-quality artifact type")
        if value.get("production_integration") is not False:
            raise ValueError("boundary-quality production integration must remain false")
        proposal = value.get("proposal")
        gate = value.get("relative_gate")
        inputs = value.get("inputs")
        if not isinstance(proposal, Mapping) or not isinstance(gate, Mapping):
            raise ValueError("boundary-quality artifact payload is incomplete")
        proposal_model = _validate_linear_model(
            proposal.get("model"), kind="boundary_pairwise_linear_logistic"
        )
        configuration = proposal.get("configuration")
        expected_configuration_fields = {
            "allow_refinement",
            "boundary_profile",
            "epochs",
            "feature_mode",
            "feature_names",
            "mask_peak_mode",
            "objective_mode",
            "option_limit",
            "pair_limit_per_group",
            "regularization",
            "shortlist_limit",
        }
        if (
            not isinstance(configuration, Mapping)
            or set(configuration) != expected_configuration_fields
            or configuration.get("allow_refinement") is not True
            or configuration.get("feature_mode") != "full"
            or configuration.get("mask_peak_mode") != "absolute"
            or configuration.get("objective_mode") != "minimax"
            or configuration.get("option_limit") != 6
            or configuration.get("shortlist_limit") != 12
            or tuple(configuration.get("feature_names", ()))
            != tuple(proposal_model["feature_names"])
            or tuple(proposal_model["feature_names"])
            != (
                *BOUNDARY_RUNTIME_FEATURE_NAMES,
                *MASK_BOUNDARY_ABSOLUTE_FEATURE_NAMES,
            )
        ):
            raise ValueError("boundary proposal configuration differs from the sealed schema")
        profile_value = configuration.get("boundary_profile")
        if not isinstance(profile_value, Mapping):
            raise ValueError("boundary profile configuration is missing")
        try:
            profile_config = BoundaryProfileConfig(**dict(profile_value))
        except (TypeError, ValueError) as exc:
            raise ValueError("boundary profile configuration is invalid") from exc
        names = gate.get("feature_names")
        models = gate.get("models")
        if (
            gate.get("schema_version") != 1
            or gate.get("kind") != "v8.2_final_relative_gate"
            or gate.get("action_policy") != "manual_rescue_coverage"
            or not isinstance(names, list)
            or not names
            or any(name not in RELATIVE_ADOPTION_FEATURE_NAMES for name in names)
            or not isinstance(models, Mapping)
            or set(models) != {
                "proposal_better_max_error",
                "harmful",
                "proposal_strict",
            }
        ):
            raise ValueError("relative gate payload differs from the sealed schema")
        parsed_models = {
            label: _validate_linear_model(
                models[label],
                kind="relative_linear_logistic",
                expected_feature_names=names,
            )
            for label in sorted(models)
        }
        parsed_gate = dict(gate)
        parsed_gate["models"] = MappingProxyType(parsed_models)
        parsed_gate["thresholds"] = _thresholds(gate.get("thresholds"))
        if not isinstance(inputs, Mapping) or not inputs or any(
            not isinstance(key, str)
            or not isinstance(raw, str)
            or not _SHA256_RE.fullmatch(raw)
            for key, raw in inputs.items()
        ):
            raise ValueError("boundary-quality input identities are invalid")
        if source_sha256 is not None and not _SHA256_RE.fullmatch(source_sha256):
            raise ValueError("boundary-quality source SHA-256 is invalid")
        return cls(
            artifact_type=artifact_type,
            proposal_model=proposal_model,
            proposal_config=MappingProxyType(dict(configuration)),
            boundary_profile_config=profile_config,
            relative_gate=MappingProxyType(parsed_gate),
            inputs=MappingProxyType(dict(inputs)),
            source_sha256=source_sha256,
        )


def load_boundary_quality_artifact(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> BoundaryQualityArtifact:
    artifact_path = Path(path)
    if artifact_path.is_symlink() or not artifact_path.is_file():
        raise ValueError("boundary-quality artifact must be a regular file")
    raw = artifact_path.read_bytes()
    identity = "sha256:" + hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and identity != expected_sha256:
        raise ValueError("boundary-quality artifact SHA-256 mismatch")
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("boundary-quality artifact is invalid JSON") from exc
    return BoundaryQualityArtifact.from_dict(value, source_sha256=identity)


def _choose_action(
    *,
    decision_status: str,
    better_probability: float,
    harmful_probability: float,
    safe_probability: float,
    view_agreement_ratio: float,
    thresholds: Mapping[str, float],
) -> str:
    fallback = "keep_manual" if decision_status in _MANUAL_ACTIONS else "keep_current"
    if view_agreement_ratio < thresholds["minimum_view_agreement_ratio"]:
        return fallback
    if decision_status in _AUTOMATIC_ACTIONS:
        if (
            better_probability >= thresholds["replace_better_min"]
            and harmful_probability <= thresholds["replace_harmful_max"]
        ):
            return "replace_automatic"
        return "keep_current"
    if decision_status in _MANUAL_ACTIONS:
        if (
            safe_probability >= thresholds["rescue_safe_min"]
            and harmful_probability <= thresholds["rescue_harmful_max"]
        ):
            return "rescue_manual"
        return "keep_manual"
    return "keep_current"


def apply_relative_gate(
    record: Mapping[str, Any],
    artifact: BoundaryQualityArtifact,
) -> dict[str, Any]:
    """Apply the frozen relative head to one truth-free runtime record."""
    if not isinstance(artifact, BoundaryQualityArtifact):
        raise TypeError("artifact must be BoundaryQualityArtifact")
    if not isinstance(record, Mapping):
        raise TypeError("relative gate record must be an object")
    features = record.get("runtime_features")
    decision_status = record.get("decision_status")
    if (
        not isinstance(features, Mapping)
        or set(features) != set(RELATIVE_ADOPTION_FEATURE_NAMES)
        or not isinstance(decision_status, str)
    ):
        raise ValueError("relative gate runtime feature schema is invalid")
    try:
        runtime_values = {name: float(features[name]) for name in features}
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("relative gate runtime features must be numeric") from exc
    if any(not math.isfinite(value) for value in runtime_values.values()):
        raise ValueError("relative gate runtime features must be finite")
    gate = artifact.relative_gate
    names = gate["feature_names"]
    vector = tuple(runtime_values[name] for name in names)
    models = gate["models"]
    better = predict_relative_probability(
        models["proposal_better_max_error"], vector
    )
    harmful = predict_relative_probability(models["harmful"], vector)
    safe = predict_relative_probability(models["proposal_strict"], vector)
    agreement = runtime_values["adaptive_consistency_agreement_min"]
    if not 0.0 <= agreement <= 1.0:
        raise ValueError("relative gate view agreement must be in [0, 1]")
    action = _choose_action(
        decision_status=decision_status,
        better_probability=better,
        harmful_probability=harmful,
        safe_probability=safe,
        view_agreement_ratio=agreement,
        thresholds=gate["thresholds"],
    )
    return {
        "better_probability": better,
        "harmful_probability": harmful,
        "safe_probability": safe,
        "view_agreement_ratio": agreement,
        "action": action,
    }


def _quad(value: Any, *, label: str) -> tuple[tuple[float, float], ...]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"{label} must contain four corners")
    result = []
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError(f"{label} contains an invalid point")
        x, y = point
        if (
            isinstance(x, bool)
            or isinstance(y, bool)
            or not isinstance(x, (int, float))
            or not isinstance(y, (int, float))
            or not math.isfinite(float(x))
            or not math.isfinite(float(y))
        ):
            raise ValueError(f"{label} contains a non-finite point")
        result.append((float(x), float(y)))
    return tuple(result)

def _image_diagonal(image_size: Sequence[int | float]) -> float:
    if not isinstance(image_size, (list, tuple)) or len(image_size) != 2:
        raise ValueError("image_size must be [width, height]")
    width, height = image_size
    if (
        isinstance(width, bool)
        or isinstance(height, bool)
        or not isinstance(width, (int, float))
        or not isinstance(height, (int, float))
        or float(width) <= 0
        or float(height) <= 0
    ):
        raise ValueError("image_size must be positive")
    return math.hypot(float(width), float(height))

def _point_line_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.hypot(dx, dy)
    if length <= 1e-12:
        raise ValueError("edge endpoints must be distinct")
    return abs(dx * (start[1] - point[1]) - (start[0] - point[0]) * dy) / length

def edge_line_errors(
    corners: Any,
    truth: Any,
    image_size: Sequence[int | float],
) -> dict[str, float]:
    """Return symmetric per-edge normal-distance errors, diagonal-normalized.

    Each edge compares both candidate endpoints to the infinite truth line and
    both truth endpoints to the infinite candidate line.  Along-edge extent is
    intentionally ignored because adjacent selected lines determine the final
    intersections in a future structured selector.
    """
    candidate_quad = _quad(corners, label="candidate corners")
    truth_quad = _quad(truth, label="truth corners")
    diagonal = _image_diagonal(image_size)
    result: dict[str, float] = {}
    for name, (first, second) in zip(EDGE_NAMES, EDGE_CORNERS, strict=True):
        candidate_edge = (candidate_quad[first], candidate_quad[second])
        truth_edge = (truth_quad[first], truth_quad[second])
        distances = (
            _point_line_distance(candidate_edge[0], *truth_edge),
            _point_line_distance(candidate_edge[1], *truth_edge),
            _point_line_distance(truth_edge[0], *candidate_edge),
            _point_line_distance(truth_edge[1], *candidate_edge),
        )
        result[name] = max(distances) / diagonal
    return result

def _line_intersection(
    first_start: tuple[float, float],
    first_end: tuple[float, float],
    second_start: tuple[float, float],
    second_end: tuple[float, float],
) -> tuple[float, float] | None:
    first_dx = first_end[0] - first_start[0]
    first_dy = first_end[1] - first_start[1]
    second_dx = second_end[0] - second_start[0]
    second_dy = second_end[1] - second_start[1]
    denominator = first_dx * second_dy - first_dy * second_dx
    scale = max(
        math.hypot(first_dx, first_dy) * math.hypot(second_dx, second_dy),
        1.0,
    )
    if abs(denominator) <= 1e-9 * scale:
        return None
    offset_x = second_start[0] - first_start[0]
    offset_y = second_start[1] - first_start[1]
    first_t = (offset_x * second_dy - offset_y * second_dx) / denominator
    return (
        first_start[0] + first_t * first_dx,
        first_start[1] + first_t * first_dy,
    )

def recombine_edge_lines(
    edge_candidates: Mapping[str, Mapping[str, Any]],
) -> list[list[float]] | None:
    """Intersect independently selected top/right/bottom/left candidate lines."""
    if set(edge_candidates) != set(EDGE_NAMES):
        raise ValueError("edge candidate mapping must contain top/right/bottom/left")
    quads = {
        name: _quad(edge_candidates[name].get("corners"), label=f"{name} corners")
        for name in EDGE_NAMES
    }
    top = (quads["top"][0], quads["top"][1])
    right = (quads["right"][1], quads["right"][2])
    bottom = (quads["bottom"][2], quads["bottom"][3])
    left = (quads["left"][3], quads["left"][0])
    intersections = (
        _line_intersection(*top, *left),
        _line_intersection(*top, *right),
        _line_intersection(*bottom, *right),
        _line_intersection(*bottom, *left),
    )
    if any(point is None for point in intersections):
        return None
    return [[float(point[0]), float(point[1])] for point in intersections if point]

def _quad_geometry(
    corners: Any,
    image_size: Sequence[int | float],
    edge_candidates: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    quad = _quad(corners, label="recombined corners")
    width, height = (float(image_size[0]), float(image_size[1]))
    diagonal = _image_diagonal(image_size)
    crosses = []
    angles = []
    for index in range(4):
        current = quad[index]
        next_point = quad[(index + 1) % 4]
        next_next = quad[(index + 2) % 4]
        crosses.append(
            (next_point[0] - current[0]) * (next_next[1] - next_point[1])
            - (next_point[1] - current[1]) * (next_next[0] - next_point[0])
        )
        previous = quad[(index - 1) % 4]
        first = (previous[0] - current[0], previous[1] - current[1])
        second = (next_point[0] - current[0], next_point[1] - current[1])
        denominator = math.hypot(*first) * math.hypot(*second)
        if denominator <= 1e-12:
            angles.append(0.0)
        else:
            cosine = max(-1.0, min(1.0, (first[0] * second[0] + first[1] * second[1]) / denominator))
            angles.append(math.degrees(math.acos(cosine)))
    convex = all(value > 1e-9 for value in crosses) or all(
        value < -1e-9 for value in crosses
    )
    twice_area = abs(sum(
        quad[index][0] * quad[(index + 1) % 4][1]
        - quad[(index + 1) % 4][0] * quad[index][1]
        for index in range(4)
    ))
    area_ratio = 0.5 * twice_area / (width * height)
    margin = 0.065 * diagonal
    within_extended_bounds = all(
        -margin <= point[0] <= width - 1.0 + margin
        and -margin <= point[1] <= height - 1.0 + margin
        for point in quad
    )
    adjacent_sources = (
        ("top", "left"),
        ("top", "right"),
        ("right", "bottom"),
        ("bottom", "left"),
    )
    minimum_parent_displacements = []
    all_parent_displacements = []
    for corner_index, names in enumerate(adjacent_sources):
        distances = []
        for name in names:
            parent_quad = _quad(
                edge_candidates[name].get("corners"), label=f"{name} parent corners"
            )
            distance = math.hypot(
                quad[corner_index][0] - parent_quad[corner_index][0],
                quad[corner_index][1] - parent_quad[corner_index][1],
            ) / diagonal
            distances.append(distance)
            all_parent_displacements.append(distance)
        minimum_parent_displacements.append(min(distances))
    angle_legal = min(angles) >= 15.0 and max(angles) <= 165.0
    base_legal = (
        convex
        and within_extended_bounds
        and 0.015 <= area_ratio <= 1.05
        and angle_legal
    )
    max_min_parent = max(minimum_parent_displacements)
    max_all_parent = max(all_parent_displacements)
    return {
        "convex": convex,
        "within_extended_bounds": within_extended_bounds,
        "area_ratio": area_ratio,
        "minimum_interior_angle_degrees": min(angles),
        "maximum_interior_angle_degrees": max(angles),
        "max_min_parent_corner_displacement": max_min_parent,
        "max_all_parent_corner_displacement": max_all_parent,
        "legal_min_parent_anchor": base_legal and max_min_parent <= 0.065,
        "legal_both_parent_anchor": base_legal and max_all_parent <= 0.065,
    }

def quad_geometry(
    corners: Any,
    image_size: Sequence[int | float],
    edge_candidates: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Public truth-independent geometry constraints for offline prototypes."""
    return _quad_geometry(corners, image_size, edge_candidates)

def candidate_key(candidate: Mapping[str, Any]) -> str:
    source = candidate.get("analysis_source")
    candidate_id = candidate.get("candidate_id")
    if source not in SOURCE_NAMES or not isinstance(candidate_id, str) or not candidate_id:
        raise ValueError("candidate lacks a supported runtime identity")
    return f"{source}:{candidate_id}"

def _number(value: Any, *, default: float = 0.0, limit: float = 20.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    numeric = float(value)
    if not math.isfinite(numeric):
        return default
    return max(-limit, min(limit, numeric))

def _side_value(candidate: Mapping[str, Any], field: str, edge_index: int) -> float:
    values = candidate.get(field)
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        return 0.0
    return _number(values[edge_index])

def _source_rank_fraction(
    candidate: Mapping[str, Any], pool: Sequence[Mapping[str, Any]]
) -> float:
    source = candidate.get("analysis_source")
    same_source = sorted(
        (value for value in pool if value.get("analysis_source") == source),
        key=lambda value: (-_number(value.get("score")), candidate_key(value)),
    )
    if len(same_source) <= 1:
        return 0.0
    key = candidate_key(candidate)
    index = next(
        (position for position, value in enumerate(same_source) if candidate_key(value) == key),
        None,
    )
    if index is None:
        raise ValueError("candidate is absent from its feature pool")
    return index / (len(same_source) - 1)

def _cross_source_distances(
    candidate: Mapping[str, Any],
    pool: Sequence[Mapping[str, Any]],
    edge_name: str,
    image_size: Sequence[int | float],
) -> tuple[list[float], list[float]]:
    key = candidate_key(candidate)
    by_source: dict[str, list[float]] = {source: [] for source in SOURCE_NAMES}
    for other in pool:
        if candidate_key(other) == key:
            continue
        distance = edge_line_errors(
            candidate.get("corners"), other.get("corners"), image_size
        )[edge_name]
        by_source[str(other["analysis_source"])].append(distance)
    minimums = [
        min(by_source[source], default=0.1) / 0.1 for source in SOURCE_NAMES
    ]
    agreement = []
    for threshold in (0.005, 0.01, 0.02):
        agreeing_sources = sum(
            bool(values) and min(values) <= threshold
            for source, values in by_source.items()
            if source != candidate.get("analysis_source")
        )
        agreement.append(agreeing_sources / len(SOURCE_NAMES))
    return minimums, agreement

def runtime_feature_vector(
    candidate: Mapping[str, Any],
    pool: Sequence[Mapping[str, Any]],
    edge_name: str,
    image_size: Sequence[int | float],
) -> tuple[float, ...]:
    """Build an explicit truth-free candidate-edge feature vector."""
    if edge_name not in EDGE_NAMES:
        raise ValueError("unsupported edge name")
    candidate_key(candidate)
    edge_index = EDGE_NAMES.index(edge_name)
    raw_prior = _number(candidate.get("raw_prior_score"))
    distances, agreement = _cross_source_distances(
        candidate, pool, edge_name, image_size
    )
    source = candidate.get("analysis_source")
    features = (
        _number(candidate.get("score")),
        math.copysign(math.log1p(abs(raw_prior)), raw_prior),
        _number(candidate.get("prior_score_normalized")),
        _number(candidate.get("area_ratio")),
        _number(candidate.get("exterior_score")),
        _number(candidate.get("valid_side_count")) / 4.0,
        math.log1p(max(0.0, _number(candidate.get("seed_support_count")))),
        _number(candidate.get("seed_source_group_count")) / 4.0,
        _side_value(candidate, "side_scores", edge_index),
        _side_value(candidate, "side_bed_scores", edge_index),
        _side_value(candidate, "side_connected_scores", edge_index),
        _side_value(candidate, "side_stability_scores", edge_index),
        _side_value(candidate, "side_coverages", edge_index),
        *(1.0 if source == expected else 0.0 for expected in SOURCE_NAMES),
        _source_rank_fraction(candidate, pool),
        *distances,
        *agreement,
    )
    if len(features) != len(FEATURE_NAMES):
        raise RuntimeError("runtime feature schema length mismatch")
    return tuple(float(value) for value in features)

def build_boundary_candidate_feature(
    candidate: Mapping[str, Any],
    pool: Sequence[Mapping[str, Any]],
    edge_name: str,
    image_size: Sequence[int | float],
    *,
    profile: BoundaryProfileEvidence,
    refinement: BoundaryLineRefinement,
    consistency: Mapping[str, Any] | None = None,
    use_refined_line: bool = False,
) -> tuple[float, ...]:
    """Combine existing metadata with profile/refinement truth-free evidence."""
    forbidden = _find_forbidden_truth_fields(candidate, prefix="candidate")
    for index, other in enumerate(pool):
        forbidden.extend(
            _find_forbidden_truth_fields(other, prefix=f"pool[{index}]")
        )
    if forbidden:
        raise ValueError(
            "forbidden truth-derived field in runtime evidence: " + forbidden[0]
        )
    if not isinstance(profile, BoundaryProfileEvidence):
        raise TypeError("profile must be BoundaryProfileEvidence")
    if not isinstance(refinement, BoundaryLineRefinement):
        raise TypeError("refinement must be BoundaryLineRefinement")
    if refinement.edge_name != edge_name:
        raise ValueError("refinement edge differs from requested feature edge")
    if type(use_refined_line) is not bool:
        raise TypeError("use_refined_line must be boolean")
    base = runtime_feature_vector(candidate, pool, edge_name, image_size)
    values = dict(zip(FEATURE_NAMES, base, strict=True))
    values.update({f"edge_{name}": float(name == edge_name) for name in EDGE_NAMES})
    values.update({
        "profile_peak_offset": profile.peak_offset,
        "profile_peak_width": profile.peak_width,
        "profile_entropy": profile.entropy,
        "profile_second_peak_ratio": profile.second_peak_ratio,
        "profile_support_q10": profile.support_q10,
        "profile_support_q50": profile.support_q50,
        "profile_support_q90": profile.support_q90,
        "profile_maximum_gap_ratio": profile.maximum_gap_ratio,
        "profile_valid_coverage": profile.valid_coverage,
        "refinement_residual_q50": refinement.residual_q50,
        "refinement_residual_q90": refinement.residual_q90,
        "refinement_residual_q95": refinement.residual_q95,
        "refinement_inlier_ratio": refinement.inlier_ratio,
        "refinement_orientation_change": (
            refinement.orientation_change_degrees / 30.0
        ),
        "refinement_parent_shift": refinement.parent_shift,
        "refinement_accepted": float(refinement.accepted),
        "candidate_uses_refined_line": float(use_refined_line),
    })
    consistency = {} if consistency is None else consistency
    if not isinstance(consistency, Mapping):
        raise TypeError("consistency evidence must be a mapping")
    forbidden = _find_forbidden_truth_fields(consistency, prefix="consistency")
    if forbidden:
        raise ValueError(
            "forbidden truth-derived field in runtime evidence: " + forbidden[0]
        )
    for name in (
        "consistency_peak_offset_dispersion",
        "consistency_line_angle_dispersion",
        "consistency_maximum_endpoint_deviation",
        "consistency_view_agreement_ratio",
        "consistency_view_failure_ratio",
    ):
        values[name] = consistency.get(name, 0.0)
    return boundary_runtime_feature_vector(values)

def structured_combination_key(
    edge_risks: Sequence[int | float],
    upper_corner_risks: Sequence[int | float],
    geometry: Mapping[str, Any],
    cross_source_support: Sequence[int | float],
    identities: Sequence[str],
    *,
    objective_mode: str,
) -> tuple[Any, ...]:
    if objective_mode not in {"minimax", "sum"}:
        raise ValueError("objective_mode must be minimax or sum")
    if not all(
        len(values) == 4
        for values in (edge_risks, upper_corner_risks, cross_source_support, identities)
    ):
        raise ValueError("structured objective requires four edges")
    risks = tuple(float(value) for value in edge_risks)
    upper = tuple(float(value) for value in upper_corner_risks)
    support = tuple(float(value) for value in cross_source_support)
    if any(not math.isfinite(value) for value in risks + upper + support):
        raise ValueError("structured objective values must be finite")
    if geometry.get("legal_min_parent_anchor") is not True:
        raise ValueError("structured objective requires legal geometry")
    geometry_key = (
        0 if geometry.get("legal_both_parent_anchor") is True else 1,
        float(geometry.get("max_min_parent_corner_displacement", 1.0)),
    )
    stable_ids = tuple(str(value) for value in identities)
    if objective_mode == "sum":
        return (
            sum(risks),
            max(risks),
            max(upper),
            *geometry_key,
            -min(support),
            stable_ids,
        )
    return (
        max(risks),
        max(upper),
        *geometry_key,
        -min(support),
        sum(risks),
        stable_ids,
    )

def select_structured_boundary_candidate(
    ranked_options: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    image_size: Sequence[int | float],
    option_limit: int = 6,
    objective_mode: str = "minimax",
) -> dict[str, Any] | None:
    """Enumerate a bounded legal top-k and select by sum or minimax ablation."""
    if set(ranked_options) != set(EDGE_NAMES):
        raise ValueError("ranked options must contain top/right/bottom/left")
    if type(option_limit) is not int or not 1 <= option_limit <= 8:
        raise ValueError("option_limit must be in [1, 8]")
    options = {}
    for edge_name in EDGE_NAMES:
        parsed = []
        for option in ranked_options[edge_name]:
            if not isinstance(option, Mapping) or not isinstance(
                option.get("candidate"), Mapping
            ):
                raise ValueError("structured option lacks a candidate")
            identity = candidate_key(option["candidate"])
            risk = float(option.get("predicted_risk"))
            upper = float(option.get("predicted_upper_corner_risk"))
            support = float(option.get("cross_source_support"))
            if any(not math.isfinite(value) for value in (risk, upper, support)):
                raise ValueError("structured option values must be finite")
            parsed.append((risk, identity, option["candidate"], upper, support))
        parsed.sort(key=lambda value: (value[0], value[1]))
        if not parsed:
            return None
        options[edge_name] = parsed[:option_limit]
    best = None
    for combination in product(*(options[name] for name in EDGE_NAMES)):
        selected = {
            edge_name: combination[index][2]
            for index, edge_name in enumerate(EDGE_NAMES)
        }
        corners = recombine_edge_lines(selected)
        if corners is None:
            continue
        parent_anchors = {
            edge_name: (
                {**candidate, "corners": candidate["parent_corners"]}
                if candidate.get("parent_corners") is not None
                else candidate
            )
            for edge_name, candidate in selected.items()
        }
        geometry = quad_geometry(corners, image_size, parent_anchors)
        if geometry["legal_min_parent_anchor"] is not True:
            continue
        risks = [value[0] for value in combination]
        identities = [value[1] for value in combination]
        upper = [value[3] for value in combination]
        support = [value[4] for value in combination]
        key = structured_combination_key(
            risks,
            upper,
            geometry,
            support,
            identities,
            objective_mode=objective_mode,
        )
        value = (key, corners, geometry, identities, risks, upper, support)
        if best is None or value[0] < best[0]:
            best = value
    if best is None:
        return None
    key, corners, geometry, identities, risks, upper, support = best
    return {
        "corners": corners,
        "edge_candidates": dict(zip(EDGE_NAMES, identities, strict=True)),
        "edge_risks": dict(zip(EDGE_NAMES, risks, strict=True)),
        "upper_corner_risks": dict(zip(EDGE_NAMES, upper, strict=True)),
        "cross_source_support": dict(zip(EDGE_NAMES, support, strict=True)),
        "geometry": geometry,
        "objective_mode": objective_mode,
        "objective_key": key,
    }

def _profile_cache_summary(profile: BoundaryProfileEvidence) -> dict[str, float]:
    return {
        name: float(getattr(profile, name))
        for name in (
            "peak_offset",
            "peak_width",
            "entropy",
            "second_peak_ratio",
            "support_q10",
            "support_q50",
            "support_q90",
            "maximum_gap_ratio",
            "valid_coverage",
        )
    }

def _refinement_cache_summary(
    refinement: BoundaryLineRefinement,
) -> dict[str, Any]:
    return {
        "accepted": refinement.accepted,
        "residual_q50": refinement.residual_q50,
        "residual_q90": refinement.residual_q90,
        "residual_q95": refinement.residual_q95,
        "inlier_ratio": refinement.inlier_ratio,
        "orientation_change_degrees": refinement.orientation_change_degrees,
        "parent_shift": refinement.parent_shift,
        "support_count": refinement.support_count,
        "refusal_reason": refinement.refusal_reason,
    }

def _raw_boundary_risk(
    profile: BoundaryProfileEvidence,
    refinement: BoundaryLineRefinement,
    *,
    use_refined_line: bool,
    config: BoundaryProfileConfig,
) -> float:
    ambiguity = (
        1.50 * profile.entropy
        + 1.75 * profile.second_peak_ratio
        + 1.25 * profile.maximum_gap_ratio
        + 1.50 * (1.0 - profile.valid_coverage)
        + 0.50 * (1.0 - profile.support_q50)
    )
    offset = abs(profile.peak_offset) / max(1e-12, config.search_band_ratio)
    if use_refined_line:
        fit = (
            4.0 * refinement.residual_q95
            + refinement.orientation_change_degrees / 30.0
            + refinement.parent_shift
            + (1.0 - refinement.inlier_ratio)
        )
        return float(ambiguity + 0.35 * offset + fit)
    return float(ambiguity + 1.50 * offset)

def _masked_boundary_features(
    features: Sequence[int | float],
    *,
    feature_mode: str,
) -> tuple[float, ...]:
    values = tuple(float(value) for value in features)
    if len(values) != len(BOUNDARY_RUNTIME_FEATURE_NAMES) or any(
        not math.isfinite(value) for value in values
    ):
        raise ValueError("cached boundary feature vector is invalid")
    if feature_mode not in {"metadata", "profile", "full"}:
        raise ValueError("unsupported boundary feature mode")
    if feature_mode == "full":
        return values
    keep = set(FEATURE_NAMES) | {f"edge_{name}" for name in EDGE_NAMES}
    if feature_mode == "profile":
        keep.update(
            name for name in BOUNDARY_RUNTIME_FEATURE_NAMES
            if name.startswith("profile_")
        )
    return tuple(
        value if name in keep else 0.0
        for name, value in zip(BOUNDARY_RUNTIME_FEATURE_NAMES, values, strict=True)
    )

def _mask_option_features(
    mask_cache_row: Mapping[str, Any],
    edge_name: str,
    option_id: str,
    *,
    peak_mode: str,
) -> tuple[float, ...]:
    option = next(
        (
            value
            for value in mask_cache_row["edges"][edge_name]
            if value.get("option_id") == option_id
        ),
        None,
    )
    if option is None:
        raise ValueError("mask witness option is missing")
    return mask_boundary_ranker_feature_vector(
        option.get("features"), peak_mode=peak_mode
    )

def _boundary_option_features(
    option: Mapping[str, Any],
    *,
    edge_name: str,
    feature_mode: str,
    mask_cache_row: Mapping[str, Any] | None,
    mask_peak_mode: str = "signed",
) -> tuple[float, ...]:
    base = _masked_boundary_features(
        option["features"], feature_mode=feature_mode
    )
    if mask_cache_row is None:
        return base
    return base + _mask_option_features(
        mask_cache_row,
        edge_name,
        str(option["option_id"]),
        peak_mode=mask_peak_mode,
    )

def _boundary_option_upper_adjustment(option: Mapping[str, Any]) -> float:
    profile = option["profile"]
    refinement = option["refinement"]
    return float(
        0.20 * float(profile["entropy"])
        + 0.30 * float(profile["second_peak_ratio"])
        + 0.20 * float(profile["maximum_gap_ratio"])
        + 8.0 * float(refinement["residual_q95"])
        + 0.50 * float(refinement["parent_shift"])
        + 0.10 * float(option["uses_refined_line"])
    )

def _build_boundary_proposal(
    cache_row: Mapping[str, Any],
    *,
    image_size: Sequence[int | float],
    objective_mode: str,
    model: Mapping[str, Any] | None,
    feature_mode: str = "full",
    allow_refinement: bool = True,
    option_limit: int = 6,
    mask_cache_row: Mapping[str, Any] | None = None,
    mask_peak_mode: str = "signed",
) -> dict[str, Any] | None:
    ranked_options = {}
    option_by_id = {edge_name: {} for edge_name in EDGE_NAMES}
    scored_by_id = {edge_name: {} for edge_name in EDGE_NAMES}
    edge_margins = {}
    for edge_name in EDGE_NAMES:
        values = []
        for option in cache_row["edges"][edge_name]:
            if option["uses_refined_line"] and not allow_refinement:
                continue
            risk = (
                float(option["raw_boundary_risk"])
                if model is None
                else predict_boundary_risk(
                    model,
                    _boundary_option_features(
                        option,
                        edge_name=edge_name,
                        feature_mode=feature_mode,
                        mask_cache_row=mask_cache_row,
                        mask_peak_mode=mask_peak_mode,
                    ),
                )
            )
            scored = {
                "candidate": option["candidate"],
                "predicted_risk": risk,
                "predicted_upper_corner_risk": (
                    risk + _boundary_option_upper_adjustment(option)
                ),
                "cross_source_support": option["cross_source_support"],
            }
            values.append(scored)
            option_by_id[edge_name][option["option_id"]] = option
            scored_by_id[edge_name][option["option_id"]] = scored
        values.sort(key=lambda value: (
            float(value["predicted_risk"]),
            candidate_key(value["candidate"]),
        ))
        if not values:
            return None
        edge_margins[edge_name] = (
            float(values[1]["predicted_risk"] - values[0]["predicted_risk"])
            if len(values) >= 2
            else 20.0
        )
        ranked_options[edge_name] = values
    selected = select_structured_boundary_candidate(
        ranked_options,
        image_size=image_size,
        option_limit=option_limit,
        objective_mode=objective_mode,
    )
    if selected is None:
        fallback = {}
        for edge_name in EDGE_NAMES:
            current_option = next(
                (
                    option
                    for option in cache_row["edges"][edge_name]
                    if option["is_current"] is True
                ),
                None,
            )
            if current_option is None:
                return None
            fallback[edge_name] = [scored_by_id[edge_name][current_option["option_id"]]]
        selected = select_structured_boundary_candidate(
            fallback,
            image_size=image_size,
            option_limit=1,
            objective_mode=objective_mode,
        )
        if selected is None:
            return None
        selected["fallback_current"] = True
    selected_options = {
        edge_name: option_by_id[edge_name][selected["edge_candidates"][edge_name]]
        for edge_name in EDGE_NAMES
    }
    source_names = [
        str(selected_options[name]["candidate"]["analysis_source"])
        for name in EDGE_NAMES
    ]
    selected.update({
        "edge_margins": edge_margins,
        "edge_profiles": {
            name: selected_options[name]["profile"] for name in EDGE_NAMES
        },
        "edge_refinements": {
            name: selected_options[name]["refinement"] for name in EDGE_NAMES
        },
        "uses_refined_edge_count": sum(
            option["uses_refined_line"] for option in selected_options.values()
        ),
        "source_combination": "+".join(source_names),
    })
    return selected


def _runtime_candidate_shortlist(
    pool: Sequence[Mapping[str, Any]],
    current_candidate: Mapping[str, Any],
    *,
    limit: int,
    per_source_limit: int = 2,
) -> list[Mapping[str, Any]]:
    if type(limit) is not int or not 4 <= limit <= 32:
        raise ValueError("candidate shortlist limit must be in [4, 32]")
    if type(per_source_limit) is not int or not 1 <= per_source_limit <= 4:
        raise ValueError("per-source shortlist limit must be in [1, 4]")
    materialized = list(pool)
    if not materialized:
        raise ValueError("runtime candidate pool is empty")

    def score(candidate: Mapping[str, Any]) -> float:
        value = candidate.get("score")
        return (
            float(value)
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            else -20.0
        )

    ranked = sorted(
        materialized,
        key=lambda value: (-score(value), candidate_key(value)),
    )
    current_key = candidate_key(current_candidate)
    by_key = {candidate_key(value): value for value in materialized}
    if len(by_key) != len(materialized) or current_key not in by_key:
        raise ValueError("runtime candidate identities are duplicate or omit current")
    selected = []
    seen = set()
    for candidate in (by_key[current_key], *ranked[:limit]):
        identity = candidate_key(candidate)
        if identity not in seen:
            selected.append(candidate)
            seen.add(identity)
    for source_name in SOURCE_NAMES:
        for candidate in (
            value
            for value in ranked
            if value.get("analysis_source") == source_name
        ):
            identity = candidate_key(candidate)
            if identity not in seen:
                selected.append(candidate)
                seen.add(identity)
            if sum(
                value.get("analysis_source") == source_name
                for value in selected
            ) >= per_source_limit:
                break
    return selected


def serialize_ranked_candidate(
    candidate: Any,
    *,
    raw_prior_score: int | float,
    analysis_source: str,
) -> dict[str, Any]:
    """Serialize scanner ranking evidence while excluding all truth labels."""
    from .scanner_selector import RankedScannerCandidate

    if not isinstance(candidate, RankedScannerCandidate):
        raise TypeError("candidate must be RankedScannerCandidate")
    if analysis_source not in SOURCE_NAMES:
        raise ValueError("unsupported runtime candidate source")
    if (
        isinstance(raw_prior_score, bool)
        or not isinstance(raw_prior_score, (int, float))
        or not math.isfinite(float(raw_prior_score))
    ):
        raise ValueError("raw prior score must be finite")
    evidence = candidate.evidence
    return {
        "candidate_id": candidate.candidate_id,
        "corners": candidate.corners,
        "score": candidate.score,
        "raw_prior_score": float(raw_prior_score),
        "prior_score_normalized": candidate.prior_score_normalized,
        "area_ratio": candidate.area_ratio,
        "seed_candidate_ids": candidate.seed_candidate_ids,
        "seed_source_groups": candidate.seed_source_groups,
        "seed_support_count": candidate.seed_support_count,
        "seed_source_group_count": candidate.seed_source_group_count,
        "exterior_score": evidence.score,
        "side_scores": evidence.side_scores,
        "side_bed_scores": evidence.side_bed_scores,
        "side_connected_scores": evidence.side_connected_scores,
        "side_stability_scores": evidence.side_stability_scores,
        "side_coverages": evidence.side_coverages,
        "valid_side_count": evidence.valid_side_count,
        "side_score_min": evidence.side_score_min,
        "side_score_mean": evidence.side_score_mean,
        "side_stability_mean": evidence.side_stability_mean,
        "analysis_source": analysis_source,
    }


def build_runtime_boundary_evidence(
    image_bgr: np.ndarray,
    *,
    pool: Sequence[Mapping[str, Any]],
    current_candidate: Mapping[str, Any],
    mask_probability: np.ndarray,
    image_size: Sequence[int | float],
    config: BoundaryProfileConfig,
    shortlist_limit: int = 12,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the scalar boundary and mask evidence rows without persistence."""
    image = np.asarray(image_bgr)
    probability = np.asarray(mask_probability)
    if (
        image.ndim != 3
        or image.shape[2] != 3
        or image.dtype != np.uint8
        or tuple(map(int, image_size)) != (image.shape[1], image.shape[0])
    ):
        raise ValueError("runtime boundary image differs from analysis size")
    if (
        probability.shape != image.shape[:2]
        or not np.issubdtype(probability.dtype, np.floating)
        or not np.isfinite(probability).all()
        or np.any(probability < 0.0)
        or np.any(probability > 1.0)
    ):
        raise ValueError("runtime mask probability must match the analysis image")
    if not isinstance(config, BoundaryProfileConfig):
        raise TypeError("config must be BoundaryProfileConfig")
    runtime_pool = [dict(value) for value in pool]
    shortlist = _runtime_candidate_shortlist(
        runtime_pool,
        current_candidate,
        limit=shortlist_limit,
    )
    current_key = candidate_key(current_candidate)
    boundary_edges: dict[str, list[dict[str, Any]]] = {}
    mask_edges: dict[str, list[dict[str, Any]]] = {}
    refinement_evaluation_count = 0
    for edge_name in EDGE_NAMES:
        options = []
        mask_options = []
        for runtime_candidate in shortlist:
            identity = candidate_key(runtime_candidate)
            refinement = refine_boundary_line(
                image,
                runtime_candidate["corners"],
                edge_name,
                config,
            )
            refinement_evaluation_count += 1
            profile_summary = _profile_cache_summary(refinement.profile)
            refinement_summary = _refinement_cache_summary(refinement)
            variants = [(False, runtime_candidate["corners"])]
            if refinement.accepted:
                variants.append((True, refinement.refined_corners))
            for use_refined_line, corners in variants:
                option_candidate_id = str(runtime_candidate["candidate_id"])
                if use_refined_line:
                    option_candidate_id += "|boundary_refined"
                option_candidate = {
                    "analysis_source": runtime_candidate["analysis_source"],
                    "candidate_id": option_candidate_id,
                    "corners": [list(point) for point in corners],
                    "parent_corners": [
                        list(point) for point in runtime_candidate["corners"]
                    ],
                }
                feature = build_boundary_candidate_feature(
                    runtime_candidate,
                    runtime_pool,
                    edge_name,
                    image_size,
                    profile=refinement.profile,
                    refinement=refinement,
                    use_refined_line=use_refined_line,
                )
                feature_by_name = dict(
                    zip(BOUNDARY_RUNTIME_FEATURE_NAMES, feature, strict=True)
                )
                option_id = candidate_key(option_candidate)
                options.append({
                    "option_id": option_id,
                    "parent_candidate_id": identity,
                    "candidate": option_candidate,
                    "features": list(feature),
                    "profile": profile_summary,
                    "refinement": refinement_summary,
                    "uses_refined_line": use_refined_line,
                    "parent_is_current": identity == current_key,
                    "is_current": identity == current_key and not use_refined_line,
                    "raw_boundary_risk": _raw_boundary_risk(
                        refinement.profile,
                        refinement,
                        use_refined_line=use_refined_line,
                        config=config,
                    ),
                    "cross_source_support": 4.0
                    * feature_by_name["agreement_source_fraction_005"],
                })
                mask_profile = evaluate_mask_boundary_profile(
                    probability,
                    option_candidate["corners"],
                    edge_name,
                    config,
                    lab_peak_offset=float(profile_summary["peak_offset"]),
                )
                mask_options.append({
                    "option_id": option_id,
                    "features": mask_profile.runtime_features(),
                })
        options.sort(key=lambda value: value["option_id"])
        mask_options.sort(key=lambda value: value["option_id"])
        if (
            len({value["option_id"] for value in options}) != len(options)
            or [value["option_id"] for value in options]
            != [value["option_id"] for value in mask_options]
        ):
            raise RuntimeError("runtime boundary option identities are invalid")
        boundary_edges[edge_name] = options
        mask_edges[edge_name] = mask_options
    return (
        {
            "schema_version": 1,
            "analysis_size": [int(image.shape[1]), int(image.shape[0])],
            "refinement_evaluation_count": refinement_evaluation_count,
            "edges": boundary_edges,
        },
        {
            "schema_version": 1,
            "inference_count": 1,
            "persisted_full_logits": False,
            "persisted_source_image": False,
            "edges": mask_edges,
        },
    )


def build_runtime_proposal_consistency(
    image_bgr: np.ndarray,
    *,
    proposal: Mapping[str, Any],
    boundary_evidence: Mapping[str, Any],
    config: BoundaryProfileConfig,
) -> dict[str, Any]:
    """Evaluate fixed all-view and ambiguity-triggered proposal stability."""
    if set(boundary_evidence.get("edges", {})) != set(EDGE_NAMES):
        raise ValueError("runtime boundary evidence is incomplete")
    edges = {}
    all_view_count = 0
    adaptive_view_count = 0
    for edge_name in EDGE_NAMES:
        option_id = proposal["edge_candidates"][edge_name]
        option = next(
            (
                value
                for value in boundary_evidence["edges"][edge_name]
                if value.get("option_id") == option_id
            ),
            None,
        )
        if option is None:
            raise ValueError("selected runtime boundary option is missing")
        corners = option["candidate"]["corners"]
        all_evidence = evaluate_transformation_consistency(
            image_bgr,
            corners,
            edge_name,
            config,
            view_names=BOUNDARY_VIEW_NAMES,
        )
        profile = option["profile"]
        ambiguous = (
            float(proposal["edge_margins"][edge_name])
            <= float(config.adaptive_consistency_candidate_margin)
            or float(profile["entropy"])
            >= float(config.adaptive_consistency_entropy)
            or float(profile["second_peak_ratio"])
            >= float(config.adaptive_consistency_second_peak_ratio)
            or float(profile["valid_coverage"])
            < float(config.minimum_refinement_coverage)
        )
        adaptive_names = BOUNDARY_VIEW_NAMES if ambiguous else ("identity",)
        adaptive_evidence = (
            all_evidence
            if adaptive_names == BOUNDARY_VIEW_NAMES
            else evaluate_transformation_consistency(
                image_bgr,
                corners,
                edge_name,
                config,
                view_names=adaptive_names,
            )
        )
        all_view_count += len(BOUNDARY_VIEW_NAMES)
        adaptive_view_count += len(adaptive_names)
        edges[edge_name] = {
            "option_id": option_id,
            "adaptive_view_names": list(adaptive_names),
            "adaptive": adaptive_evidence.runtime_features(),
            "all_views": all_evidence.runtime_features(),
        }
    return {
        "schema_version": 1,
        "all_view_count": all_view_count,
        "adaptive_view_count": adaptive_view_count,
        "edges": edges,
    }


def build_relative_runtime_features(
    *,
    current_corners: Sequence[Sequence[int | float]],
    proposal: Mapping[str, Any],
    consistency: Mapping[str, Any],
    image_size: Sequence[int | float],
    decision_status: str,
    decision_reason: str | None,
) -> dict[str, float]:
    """Aggregate the final proposal using only evidence available at runtime."""
    if set(consistency.get("edges", {})) != set(EDGE_NAMES):
        raise ValueError("proposal consistency evidence is incomplete")
    risks = [float(proposal["edge_risks"][name]) for name in EDGE_NAMES]
    upper = [float(proposal["upper_corner_risks"][name]) for name in EDGE_NAMES]
    margins = [float(proposal["edge_margins"][name]) for name in EDGE_NAMES]
    support = [float(proposal["cross_source_support"][name]) for name in EDGE_NAMES]
    profiles = [proposal["edge_profiles"][name] for name in EDGE_NAMES]
    refinements = [proposal["edge_refinements"][name] for name in EDGE_NAMES]
    current = np.asarray(_quad(current_corners, label="current corners"), dtype=np.float64)
    proposed = np.asarray(_quad(proposal.get("corners"), label="proposal corners"), dtype=np.float64)
    diagonal = _image_diagonal(image_size)
    corner_distances = np.linalg.norm(proposed - current, axis=1) / diagonal
    adaptive = [consistency["edges"][name]["adaptive"] for name in EDGE_NAMES]
    all_views = [consistency["edges"][name]["all_views"] for name in EDGE_NAMES]
    geometry = proposal.get("geometry")
    geometry = geometry if isinstance(geometry, Mapping) else {}
    status = str(decision_status)
    reason = str(decision_reason)
    known_reasons = (
        "edge_base",
        "v7_edge_unresolved_conflict",
        "v7_edge_agreement_rescue",
        "strong_three_way_conflict",
    )
    values = {
        "proposal_edge_risk_min": min(risks),
        "proposal_edge_risk_mean": float(np.mean(risks)),
        "proposal_edge_risk_max": max(risks),
        "proposal_edge_risk_spread": max(risks) - min(risks),
        "proposal_upper_risk_max": max(upper),
        "proposal_edge_margin_min": min(margins),
        "proposal_edge_margin_mean": float(np.mean(margins)),
        "proposal_edge_margin_max": max(margins),
        "proposal_cross_source_support_min": min(support),
        "proposal_cross_source_support_mean": float(np.mean(support)),
        "profile_entropy_mean": float(np.mean([value["entropy"] for value in profiles])),
        "profile_entropy_max": max(float(value["entropy"]) for value in profiles),
        "profile_second_peak_mean": float(np.mean([value["second_peak_ratio"] for value in profiles])),
        "profile_second_peak_max": max(float(value["second_peak_ratio"]) for value in profiles),
        "profile_maximum_gap_max": max(float(value["maximum_gap_ratio"]) for value in profiles),
        "profile_valid_coverage_min": min(float(value["valid_coverage"]) for value in profiles),
        "profile_support_q10_min": min(float(value["support_q10"]) for value in profiles),
        "profile_absolute_peak_offset_max": max(abs(float(value["peak_offset"])) for value in profiles),
        "refinement_acceptance_ratio": float(np.mean([value["accepted"] for value in refinements])),
        "refined_edge_ratio": float(proposal["uses_refined_edge_count"]) / 4.0,
        "refinement_residual_q95_max": max(float(value["residual_q95"]) for value in refinements),
        "refinement_inlier_ratio_min": min(float(value["inlier_ratio"]) for value in refinements),
        "refinement_orientation_change_max": max(float(value["orientation_change_degrees"]) for value in refinements) / 30.0,
        "refinement_parent_shift_max": max(float(value["parent_shift"]) for value in refinements),
        "geometry_legal_min_parent_anchor": float(geometry.get("legal_min_parent_anchor") is True),
        "geometry_legal_both_parent_anchor": float(geometry.get("legal_both_parent_anchor") is True),
        "geometry_max_min_parent_displacement": float(geometry.get("max_min_parent_corner_displacement", 1.0)),
        "current_proposal_corner_distance_mean": float(np.mean(corner_distances)),
        "current_proposal_corner_distance_max": float(np.max(corner_distances)),
        "adaptive_consistency_endpoint_max": max(float(value["consistency_maximum_endpoint_deviation"]) for value in adaptive),
        "adaptive_consistency_agreement_min": min(float(value["consistency_view_agreement_ratio"]) for value in adaptive),
        "adaptive_consistency_agreement_mean": float(np.mean([value["consistency_view_agreement_ratio"] for value in adaptive])),
        "adaptive_consistency_failure_max": max(float(value["consistency_view_failure_ratio"]) for value in adaptive),
        "adaptive_consistency_failure_mean": float(np.mean([value["consistency_view_failure_ratio"] for value in adaptive])),
        "all_view_consistency_endpoint_max": max(float(value["consistency_maximum_endpoint_deviation"]) for value in all_views),
        "all_view_consistency_agreement_min": min(float(value["consistency_view_agreement_ratio"]) for value in all_views),
        "all_view_consistency_agreement_mean": float(np.mean([value["consistency_view_agreement_ratio"] for value in all_views])),
        "all_view_consistency_failure_max": max(float(value["consistency_view_failure_ratio"]) for value in all_views),
        "all_view_consistency_failure_mean": float(np.mean([value["consistency_view_failure_ratio"] for value in all_views])),
        "status_automatic": float(status in _AUTOMATIC_ACTIONS),
        "status_manual_review": float(status in _MANUAL_ACTIONS),
        **{f"reason_{name}": float(reason == name) for name in known_reasons},
        "reason_other": float(reason not in known_reasons),
    }
    if set(values) != set(RELATIVE_ADOPTION_FEATURE_NAMES):
        raise RuntimeError("relative proposal feature schema mismatch")
    if any(not math.isfinite(float(value)) for value in values.values()):
        raise ValueError("relative proposal runtime features must be finite")
    return {name: float(values[name]) for name in RELATIVE_ADOPTION_FEATURE_NAMES}


def evaluate_boundary_quality(
    image_bgr: np.ndarray,
    *,
    pool: Sequence[Mapping[str, Any]],
    current_candidate: Mapping[str, Any],
    mask_probability: np.ndarray,
    image_size: Sequence[int | float],
    decision_status: str,
    decision_reason: str | None,
    artifact: BoundaryQualityArtifact,
) -> dict[str, Any]:
    """Run the sealed V8.2 proposal and conservative relative adoption gate."""
    if not isinstance(artifact, BoundaryQualityArtifact):
        raise TypeError("artifact must be BoundaryQualityArtifact")
    config = artifact.proposal_config
    boundary, mask = build_runtime_boundary_evidence(
        image_bgr,
        pool=pool,
        current_candidate=current_candidate,
        mask_probability=mask_probability,
        image_size=image_size,
        config=artifact.boundary_profile_config,
        shortlist_limit=int(config["shortlist_limit"]),
    )
    proposal = _build_boundary_proposal(
        boundary,
        image_size=image_size,
        objective_mode=str(config["objective_mode"]),
        model=artifact.proposal_model,
        feature_mode=str(config["feature_mode"]),
        allow_refinement=bool(config["allow_refinement"]),
        option_limit=int(config["option_limit"]),
        mask_cache_row=mask,
        mask_peak_mode=str(config["mask_peak_mode"]),
    )
    if proposal is None:
        raise ValueError("runtime boundary proposal is unavailable")
    consistency = build_runtime_proposal_consistency(
        image_bgr,
        proposal=proposal,
        boundary_evidence=boundary,
        config=artifact.boundary_profile_config,
    )
    runtime_features = build_relative_runtime_features(
        current_corners=current_candidate["corners"],
        proposal=proposal,
        consistency=consistency,
        image_size=image_size,
        decision_status=decision_status,
        decision_reason=decision_reason,
    )
    gate = apply_relative_gate(
        {
            "decision_status": decision_status,
            "runtime_features": runtime_features,
        },
        artifact,
    )
    action = str(gate["action"])
    adopted = action in {"replace_automatic", "rescue_manual"}
    selected_corners = (
        proposal["corners"] if adopted else current_candidate["corners"]
    )
    return {
        "action": action,
        "adopted_proposal": adopted,
        "selected_corners": [list(point) for point in selected_corners],
        "proposal": proposal,
        "relative_gate": gate,
        "runtime_features": runtime_features,
        "consistency": consistency,
        "boundary_evaluation_count": int(
            boundary["refinement_evaluation_count"]
        ),
        "mask_inference_count": 1,
    }

__all__ = [
    "BoundaryQualityArtifact",
    "RELATIVE_ADOPTION_FEATURE_NAMES",
    "apply_relative_gate",
    "build_runtime_boundary_evidence",
    "build_runtime_proposal_consistency",
    "build_relative_runtime_features",
    "evaluate_boundary_quality",
    "load_boundary_quality_artifact",
    "predict_boundary_risk",
    "predict_relative_probability",
    "serialize_ranked_candidate",
]
