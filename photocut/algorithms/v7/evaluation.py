"""Pure paired evaluation metrics and auditable statistical checks."""
from __future__ import annotations

import math
import random
from typing import Any, Iterable, Mapping, Sequence


def _points(value: Any) -> tuple[tuple[float, float], ...] | None:
    try:
        points = tuple((float(p[0]), float(p[1])) for p in value)
        return points if len(points) == 4 else None
    except (TypeError, ValueError, IndexError):
        return None


def _diagonal(image_size: Sequence[int] | None, truth: Sequence[Sequence[float]] | None = None) -> float:
    if image_size and len(image_size) == 2:
        return math.hypot(float(image_size[0]), float(image_size[1]))
    if truth:
        xs, ys = zip(*truth)
        return math.hypot(max(xs) - min(xs), max(ys) - min(ys))
    return 1.0


def jitter_threshold(image_size: Sequence[int] | None = None, truth: Any = None) -> float:
    return max(2.0, 0.0005 * _diagonal(image_size, _points(truth)))


def corner_errors(predicted: Any, truth: Any, *, image_size: Sequence[int] | None = None) -> dict[str, Any]:
    pred, target = _points(predicted), _points(truth)
    if pred is None or target is None:
        return {"valid": False, "per_corner_px": (), "p95_px": None, "normalized_per_corner": (), "jitter_threshold_px": jitter_threshold(image_size, target)}
    errors = tuple(math.hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(pred, target))
    diagonal = _diagonal(image_size, target)
    ordered = sorted(errors)
    p95 = ordered[min(len(ordered) - 1, math.ceil(.95 * len(ordered)) - 1)]
    threshold = jitter_threshold(image_size, target)
    return {
        "valid": True,
        "per_corner_px": errors,
        "normalized_per_corner": tuple(error / max(diagonal, 1e-9) for error in errors),
        "p95_px": p95,
        "jitter_threshold_px": threshold,
        "jittered_corner_count": sum(error > threshold for error in errors),
    }


def moved_corner_count(original: Any, final: Any, *, image_size: Sequence[int] | None = None) -> int:
    result = corner_errors(final, original, image_size=image_size)
    return int(sum(error > result["jitter_threshold_px"] for error in result.get("per_corner_px", ()))) if result.get("valid") else 0


def is_outer_frame_match(predicted: Any, outer_truth: Any, *, tolerance_px: float = 2.0) -> bool:
    pred, truth = _points(predicted), _points(outer_truth)
    if pred is None or truth is None:
        return False
    return all(math.hypot(a[0] - b[0], a[1] - b[1]) <= tolerance_px for a, b in zip(pred, truth))


def metric_dictionary(record: Mapping[str, Any], *, image_size: Sequence[int] | None = None) -> dict[str, Any]:
    truth = record.get("truth", record.get("photo_truth"))
    top1 = record.get("top1_corners", record.get("corners"))
    errors = corner_errors(top1, truth, image_size=image_size)
    result = dict(errors)
    result.update({
        "image_id": record.get("image_id"),
        "origin_group_id": record.get("origin_group_id"),
        "status": record.get("status", record.get("detection_status")),
        "direct_top1": record.get("operation", "prediction") in {"prediction", "direct_top1"},
        "fallback": record.get("status") == "v52_fallback",
        "failed": record.get("status") in {"error", "cancelled"},
        "outer_frame_error": is_outer_frame_match(top1, record.get("outer_frame_truth")),
        "top5_recall": bool(record.get("top5_contains_truth", False)),
    })
    return result


def paired_binary_rate(v7: Iterable[bool], v52: Iterable[bool]) -> dict[str, Any]:
    left, right = list(v7), list(v52)
    if len(left) != len(right) or not left:
        return {"passed": False, "reason": "insufficient_pairs", "estimate": None}
    improved = sum(a and not b for a, b in zip(left, right))
    worsened = sum(b and not a for a, b in zip(left, right))
    estimate = sum(left) / len(left) - sum(right) / len(right)
    # Conservative exact discordant-pair lower bound without assuming
    # independence.  It is deliberately reported as a gate result, not a
    # claim of significance when the pilot is underpowered.
    lower = estimate - 1.96 * math.sqrt(max(0.0, (improved + worsened) / len(left) ** 2))
    return {"passed": lower > 0, "estimate": estimate, "ci_lower": lower, "improved": improved, "worsened": worsened, "n": len(left)}


def cluster_bootstrap_delta(v7: Sequence[float], v52: Sequence[float], *, seed: int = 0, repetitions: int = 2000) -> dict[str, Any]:
    if len(v7) != len(v52) or not v7:
        return {"passed": False, "reason": "insufficient_power_model", "estimate": None}
    delta = sum(v7) / len(v7) - sum(v52) / len(v52)
    rng = random.Random(seed)
    draws = []
    for _ in range(max(100, repetitions)):
        indices = [rng.randrange(len(v7)) for _ in v7]
        draws.append(sum(v7[i] - v52[i] for i in indices) / len(indices))
    lower = sorted(draws)[max(0, int(.05 * len(draws)))]
    return {"passed": lower >= 0, "estimate": delta, "ci_lower": lower, "n": len(v7), "seed": seed, "repetitions": repetitions}


__all__ = ["jitter_threshold", "corner_errors", "moved_corner_count", "is_outer_frame_match", "metric_dictionary", "paired_binary_rate", "cluster_bootstrap_delta"]
