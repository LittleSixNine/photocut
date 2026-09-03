"""Immutable, JSON-safe contracts for the v7 detector pipeline.

These objects are deliberately small production records.  Evaluation metrics and
ground-truth comparisons belong to the offline evaluator, never to candidate
audits emitted by a detector run.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import json
import re
from types import MappingProxyType
from typing import Any, Mapping


class DetectionStatus(str, Enum):
    V7_RECOMMENDED = "v7_recommended"
    V7_LOW_CONFIDENCE = "v7_low_confidence"
    V52_FALLBACK = "v52_fallback"
    NO_PRIMARY_PHOTO = "no_primary_photo"
    CANCELLED = "cancelled"
    ERROR = "error"


class ProviderStatus(str, Enum):
    SUCCESS = "success"
    NO_CANDIDATE = "no_candidate"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    BUDGET_EXHAUSTED = "budget_exhausted"
    ERROR = "error"


_MODES = {"safe", "aggressive"}
_HASH_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_TRUTH_KEYS = {
    "tp", "tn", "fp", "fn", "truepositive", "truenegative", "falsepositive",
    "falsenegative", "precision", "recall", "f1", "f1score", "groundtruth",
    "truth", "iou", "intersectionoverunion", "errormetric", "truthderived", "accuracy",
}


def _plain_key(key: Any) -> str:
    return str(key).replace("_", "").replace("-", "").lower()


def _is_truth_metric_key(key: Any) -> bool:
    """Reject metric names and common TP/FN-style variants recursively."""
    normalized = _plain_key(key)
    if normalized in _TRUTH_KEYS or "error" in normalized:
        return True
    # Count/rate/score suffixes are commonly appended to confusion-matrix
    # abbreviations (tp_count, fn_rate, candidate_fp, etc.).
    if normalized.startswith(("tp", "fp", "tn", "fn")) or normalized.endswith(("tp", "fp", "tn", "fn")):
        return True
    return any(token in normalized for token in (
        "truepositive", "truenegative", "falsepositive", "falsenegative",
        "precision", "recall", "f1", "accuracy", "groundtruth", "truth",
        "intersectionoverunion", "iou",
    ))


def _finite_number(value: Any, name: str) -> float:
    # NumPy scalar values intentionally stay an optional dependency: scalar
    # ``item()`` normalization gives us native Python values without importing
    # NumPy, while preserving rejection of numpy.bool_ after unwrapping.
    if not isinstance(value, (bool, int, float)):
        item = getattr(value, "item", None)
        if callable(item):
            try:
                value = item()
            except Exception:
                pass
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _nonnegative_integer(value: Any, name: str) -> int:
    if not isinstance(value, (bool, int)):
        item = getattr(value, "item", None)
        if callable(item):
            try:
                value = item()
            except Exception:
                pass
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _immutable(value: Any, *, reject_truth: bool = False) -> Any:
    """Copy nested values into immutable containers and normalize NumPy scalars."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        out = {}
        for key, item in value.items():
            if reject_truth and _is_truth_metric_key(key):
                raise ValueError(f"truth-derived metric is not allowed: {key}")
            string_key = str(key)
            if string_key in out:
                raise ValueError(f"mapping keys collide after string normalization: {string_key}")
            out[string_key] = _immutable(item, reject_truth=reject_truth)
        return MappingProxyType(out)
    if isinstance(value, (tuple, list, set, frozenset)):
        items = tuple(_immutable(item, reject_truth=reject_truth) for item in value)
        if isinstance(value, (set, frozenset)):
            return tuple(sorted(items, key=lambda item: json.dumps(_jsonable(item), sort_keys=True, separators=(",", ":"), ensure_ascii=False)))
        return items
    # numpy scalar types expose item(); avoid importing NumPy in this core module.
    if hasattr(value, "item") and callable(value.item):
        try:
            return _immutable(value.item(), reject_truth=reject_truth)
        except (TypeError, ValueError):
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("non-finite value is not JSON-safe")
        return value
    # ndarray-like values are accepted as a convenience, but must be finite and
    # reduced to regular nested lists/tuples by tolist().
    if hasattr(value, "tolist") and callable(value.tolist):
        return _immutable(value.tolist(), reject_truth=reject_truth)
    raise TypeError(f"unsupported value type: {type(value).__name__}")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "item") and callable(value.item):
        return _jsonable(value.item())
    if hasattr(value, "tolist") and callable(value.tolist):
        return _jsonable(value.tolist())
    return value


def _corners(value: Any, name: str, *, required: bool = False):
    if value is None:
        if required:
            raise ValueError(f"{name} is required")
        return None
    if hasattr(value, "tolist") and callable(value.tolist):
        value = value.tolist()
    if not isinstance(value, (tuple, list)) or len(value) != 4:
        raise ValueError(f"{name} must contain four corners")
    out = []
    for point in value:
        if hasattr(point, "tolist") and callable(point.tolist):
            point = point.tolist()
        if not isinstance(point, (tuple, list)) or len(point) != 2:
            raise ValueError(f"{name} must use (x, y) points")
        out.append((_finite_number(point[0], name), _finite_number(point[1], name)))
    return tuple(out)


@dataclass(frozen=True)
class DetectionIdentity:
    request_id: str
    image_id: str
    orientation_transform: str
    algorithm_version: str
    parameter_sha256: str
    mode: str

    def __post_init__(self):
        for name in ("request_id", "image_id", "orientation_transform", "algorithm_version"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be non-empty")
        if not isinstance(self.parameter_sha256, str) or not _HASH_RE.fullmatch(self.parameter_sha256):
            raise ValueError("parameter_sha256 must be a hexadecimal SHA-256 digest")
        if self.mode not in _MODES:
            raise ValueError("mode must be safe or aggressive")

    def to_dict(self) -> dict[str, Any]:
        return {"request_id": self.request_id, "image_id": self.image_id,
                "orientation_transform": self.orientation_transform,
                "algorithm_version": self.algorithm_version,
                "parameter_sha256": self.parameter_sha256, "mode": self.mode}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DetectionIdentity":
        return cls(**dict(value))


@dataclass(frozen=True)
class CandidateAudit:
    candidate_id: str
    sources: tuple[str, ...]
    original_legal_corners: tuple[tuple[float, float], ...]
    pre_topk_corners: tuple[tuple[float, float], ...] | None = None
    proposed_refined_corners: tuple[tuple[float, float], ...] | None = None
    adopted_refined_corners: tuple[tuple[float, float], ...] | None = None
    pre_truncation_risk_decisions: Any = ()
    pre_truncation_risk_evidence: Mapping[str, Any] = MappingProxyType({})
    stage_scores: Mapping[str, Any] = MappingProxyType({})
    stage_ranks: Mapping[str, Any] = MappingProxyType({})
    truncation_stage: str | None = None
    truncation_reason: str | None = None

    def __post_init__(self):
        if not isinstance(self.candidate_id, str) or not self.candidate_id.strip():
            raise ValueError("candidate_id must be non-empty")
        if not isinstance(self.sources, (tuple, list)) or any(not isinstance(s, str) or not s for s in self.sources):
            raise TypeError("sources must be strings")
        object.__setattr__(self, "sources", tuple(self.sources))
        object.__setattr__(self, "original_legal_corners", _corners(self.original_legal_corners, "original_legal_corners", required=True))
        for field in ("pre_topk_corners", "proposed_refined_corners", "adopted_refined_corners"):
            object.__setattr__(self, field, _corners(getattr(self, field), field))
        object.__setattr__(self, "pre_truncation_risk_decisions", _immutable(self.pre_truncation_risk_decisions, reject_truth=True))
        for field in ("pre_truncation_risk_evidence", "stage_scores", "stage_ranks"):
            val = getattr(self, field)
            if not isinstance(val, Mapping):
                raise TypeError(f"{field} must be a mapping")
            object.__setattr__(self, field, _immutable(val, reject_truth=True))
        if self.truncation_stage is not None and not isinstance(self.truncation_stage, str):
            raise TypeError("truncation_stage must be a string or None")
        if self.truncation_reason is not None and not isinstance(self.truncation_reason, str):
            raise TypeError("truncation_reason must be a string or None")

    def to_dict(self) -> dict[str, Any]:
        return {"candidate_id": self.candidate_id, "sources": _jsonable(self.sources),
                "original_legal_corners": _jsonable(self.original_legal_corners),
                "pre_topk_corners": _jsonable(self.pre_topk_corners),
                "proposed_refined_corners": _jsonable(self.proposed_refined_corners),
                "adopted_refined_corners": _jsonable(self.adopted_refined_corners),
                "pre_truncation_risk_decisions": _jsonable(self.pre_truncation_risk_decisions),
                "pre_truncation_risk_evidence": _jsonable(self.pre_truncation_risk_evidence),
                "stage_scores": _jsonable(self.stage_scores), "stage_ranks": _jsonable(self.stage_ranks),
                "truncation_stage": self.truncation_stage, "truncation_reason": self.truncation_reason}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CandidateAudit":
        return cls(**dict(value))


@dataclass(frozen=True)
class DetectionResult:
    identity: DetectionIdentity
    status: DetectionStatus
    corners: tuple[tuple[float, float], ...] | None = None
    alternate_corners: tuple[tuple[float, float], ...] | None = None
    overall_confidence: float | None = None
    edge_confidences: tuple[float, ...] = ()
    corner_confidences: tuple[float, ...] = ()
    risks: tuple[str, ...] = ()
    top1_sources: tuple[str, ...] = ()
    alternate_sources: tuple[str, ...] = ()
    candidate_audit: tuple[CandidateAudit, ...] = ()
    timings_ms: Mapping[str, float] = MappingProxyType({})
    debug: Mapping[str, Any] = MappingProxyType({})
    error: str | None = None

    def __post_init__(self):
        if not isinstance(self.identity, DetectionIdentity):
            raise TypeError("identity must be DetectionIdentity")
        status = self.status if isinstance(self.status, DetectionStatus) else DetectionStatus(self.status)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "corners", _corners(self.corners, "corners"))
        object.__setattr__(self, "alternate_corners", _corners(self.alternate_corners, "alternate_corners"))
        if self.overall_confidence is not None:
            confidence = _finite_number(self.overall_confidence, "overall_confidence")
            if not 0 <= confidence <= 1: raise ValueError("overall_confidence must be in [0, 1]")
            object.__setattr__(self, "overall_confidence", confidence)
        for field in ("edge_confidences", "corner_confidences"):
            values = tuple(_finite_number(v, field) for v in getattr(self, field))
            if any(v < 0 or v > 1 for v in values): raise ValueError(f"{field} must be in [0, 1]")
            object.__setattr__(self, field, values)
        for field in ("risks", "top1_sources", "alternate_sources"):
            values = tuple(getattr(self, field))
            if any(not isinstance(v, str) for v in values): raise TypeError(f"{field} must contain strings")
            object.__setattr__(self, field, values)
        audits = tuple(a if isinstance(a, CandidateAudit) else CandidateAudit.from_dict(a) for a in self.candidate_audit)
        object.__setattr__(self, "candidate_audit", audits)
        object.__setattr__(self, "timings_ms", _immutable(self.timings_ms))
        object.__setattr__(self, "debug", _immutable(self.debug))
        if self.error is not None and not isinstance(self.error, str): raise TypeError("error must be string or None")

    def to_dict(self) -> dict[str, Any]:
        return {"identity": self.identity.to_dict(), "status": self.status.value,
                "corners": _jsonable(self.corners), "alternate_corners": _jsonable(self.alternate_corners),
                "overall_confidence": self.overall_confidence,
                "edge_confidences": _jsonable(self.edge_confidences), "corner_confidences": _jsonable(self.corner_confidences),
                "risks": _jsonable(self.risks), "top1_sources": _jsonable(self.top1_sources),
                "alternate_sources": _jsonable(self.alternate_sources),
                "candidate_audit": [a.to_dict() for a in self.candidate_audit],
                "timings_ms": _jsonable(self.timings_ms), "debug": _jsonable(self.debug), "error": self.error}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DetectionResult":
        data = dict(value)
        data["identity"] = DetectionIdentity.from_dict(data["identity"])
        data["status"] = DetectionStatus(data["status"])
        data["candidate_audit"] = tuple(CandidateAudit.from_dict(a) for a in data.get("candidate_audit", ()))
        return cls(**data)


@dataclass(frozen=True)
class ProviderResult:
    provider: str
    candidates: tuple[Any, ...] = ()
    status: ProviderStatus = ProviderStatus.NO_CANDIDATE
    elapsed_ms: float = 0.0
    work_consumed: int = 0
    work_limit: int = 0
    timeout_code: str | None = None
    error_code: str | None = None
    diagnostics: Mapping[str, Any] = MappingProxyType({})

    def __post_init__(self):
        if not isinstance(self.provider, str) or not self.provider.strip(): raise ValueError("provider must be non-empty")
        object.__setattr__(self, "status", self.status if isinstance(self.status, ProviderStatus) else ProviderStatus(self.status))
        object.__setattr__(self, "elapsed_ms", _finite_number(self.elapsed_ms, "elapsed_ms"))
        if self.elapsed_ms < 0:
            raise ValueError("elapsed_ms must be non-negative")
        for field in ("work_consumed", "work_limit"):
            value = getattr(self, field)
            object.__setattr__(self, field, _nonnegative_integer(value, field))
        if self.work_limit > 0 and self.work_consumed > self.work_limit:
            raise ValueError("work_consumed cannot exceed work_limit")
        object.__setattr__(self, "candidates", _immutable(self.candidates))
        object.__setattr__(self, "diagnostics", _immutable(self.diagnostics))
        for field in ("timeout_code", "error_code"):
            value = getattr(self, field)
            if value is not None and not isinstance(value, str): raise TypeError(f"{field} must be string or None")

    def to_dict(self) -> dict[str, Any]:
        return {"provider": self.provider, "candidates": _jsonable(self.candidates), "status": self.status.value,
                "elapsed_ms": self.elapsed_ms, "work_consumed": self.work_consumed, "work_limit": self.work_limit,
                "timeout_code": self.timeout_code, "error_code": self.error_code, "diagnostics": _jsonable(self.diagnostics)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProviderResult":
        data = dict(value)
        data["status"] = ProviderStatus(data["status"])
        return cls(**data)
