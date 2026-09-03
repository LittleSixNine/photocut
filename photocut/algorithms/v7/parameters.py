"""Validated, immutable v7 detector parameters."""
from __future__ import annotations

from dataclasses import dataclass, fields, replace as dc_replace
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Mapping


_MODES = {"safe", "aggressive"}
_SCENE_PROFILES = {"scanner_white", "generic_single"}


def _finite(value: Any, name: str, *, low: float | None = None, high: float | None = None) -> float:
    if not isinstance(value, (bool, int, float)):
        item = getattr(value, "item", None)
        if callable(item):
            try:
                value = item()
            except Exception:
                pass
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if low is not None and value < low:
        raise ValueError(f"{name} is below its lower bound")
    if high is not None and value > high:
        raise ValueError(f"{name} is above its upper bound")
    return value


def _integer(value: Any, name: str, *, low: int = 0, high: int | None = None) -> int:
    if not isinstance(value, (bool, int)):
        item = getattr(value, "item", None)
        if callable(item):
            try:
                value = item()
            except Exception:
                pass
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < low or high is not None and value > high:
        raise ValueError(f"{name} is out of bounds")
    return value


def _mapping(value: Mapping[str, Any], name: str, *, integer_values: bool = False,
             value_low: float | None = None, value_high: float | None = None) -> MappingProxyType:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    out = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key.strip():
            raise TypeError(f"{name} keys must be non-empty strings")
        out[key] = (_integer(item, f"{name}[{key!r}]", low=1, high=10_000_000)
                    if integer_values else _finite(item, f"{name}[{key!r}]", low=value_low, high=value_high))
    return MappingProxyType(out)


def _tuple_of_numbers(value: Any, name: str, *, length: int | None = None, low: float = 0.0, high: float | None = None) -> tuple[float, ...]:
    if not isinstance(value, (tuple, list)) or length is not None and len(value) != length:
        raise ValueError(f"{name} has invalid length")
    return tuple(_finite(item, name, low=low, high=high) for item in value)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


@dataclass(frozen=True)
class V7Parameters:
    """All bounded values that can affect detector output or resource use."""

    mode: str = "safe"
    scene_profile: str = "scanner_white"
    work_edges: tuple[int, int] = (800, 1600)
    max_input_pixels: int = 80_000_000
    provider_work_limits: Mapping[str, int] = MappingProxyType({
        "background": 120_000, "contour": 120_000, "lines": 120_000, "shape": 80_000,
        "white_border": 120_000, "docquadnet": 120_000,
        "docaligner_heatmap": 120_000,
    })
    provider_timeout_ms: Mapping[str, int] = MappingProxyType({
        "background": 250, "contour": 250, "lines": 300, "shape": 200,
        "white_border": 250, "docquadnet": 250, "docaligner_heatmap": 250,
    })
    fused_budget: int = 40
    refined_budget: int = 5
    contour_epsilon_ratios: tuple[float, ...] = (0.008, 0.015, 0.025)
    min_area_ratio: float = 0.015
    max_area_ratio: float = 0.995
    min_edge_ratio: float = 0.03
    max_edge_ratio: float = 1.5
    edge_band: float = 0.012
    side_offsets: tuple[float, float] = (0.004, 0.02)
    dedup_distance: float = 0.015
    refinement_residual: float = 0.008
    max_refinement_shift: float = 0.025
    safe_threshold: float = 0.78
    aggressive_threshold: float = 0.62
    risk_thresholds: Mapping[str, float] = MappingProxyType({
        "weak_edge": 0.35, "outer_frame": 0.60, "ambiguity": 0.08,
    })

    def __post_init__(self):
        if self.mode not in _MODES:
            raise ValueError("mode must be safe or aggressive")
        if self.scene_profile not in _SCENE_PROFILES:
            raise ValueError("scene_profile must be scanner_white or generic_single")
        if not isinstance(self.work_edges, (tuple, list)) or len(self.work_edges) != 2:
            raise ValueError("work_edges must contain two widths")
        # Work scales are intentionally bounded: the low/high pair is the
        # detector's deterministic 800..1600px envelope, not an arbitrary
        # caller-controlled resize budget.
        edges = tuple(_integer(v, "work_edges", low=800, high=1600) for v in self.work_edges)
        if edges[0] > edges[1]:
            raise ValueError("work_edges must be ascending")
        object.__setattr__(self, "work_edges", edges)
        object.__setattr__(self, "max_input_pixels", _integer(self.max_input_pixels, "max_input_pixels", low=1, high=2_000_000_000))
        object.__setattr__(self, "provider_work_limits", _mapping(self.provider_work_limits, "provider_work_limits", integer_values=True))
        object.__setattr__(self, "provider_timeout_ms", _mapping(self.provider_timeout_ms, "provider_timeout_ms", integer_values=True))
        object.__setattr__(self, "fused_budget", _integer(self.fused_budget, "fused_budget", low=1, high=10_000))
        object.__setattr__(self, "refined_budget", _integer(self.refined_budget, "refined_budget", low=1, high=self.fused_budget))
        object.__setattr__(self, "contour_epsilon_ratios", _tuple_of_numbers(self.contour_epsilon_ratios, "contour_epsilon_ratios", low=1e-9, high=0.2))
        object.__setattr__(self, "min_area_ratio", _finite(self.min_area_ratio, "min_area_ratio", low=1e-9, high=1.0))
        object.__setattr__(self, "max_area_ratio", _finite(self.max_area_ratio, "max_area_ratio", low=1e-9, high=1.0))
        if self.min_area_ratio >= self.max_area_ratio:
            raise ValueError("min_area_ratio must be smaller than max_area_ratio")
        object.__setattr__(self, "min_edge_ratio", _finite(self.min_edge_ratio, "min_edge_ratio", low=1e-9, high=1.0))
        object.__setattr__(self, "max_edge_ratio", _finite(self.max_edge_ratio, "max_edge_ratio", low=1e-9, high=4.0))
        if self.min_edge_ratio >= self.max_edge_ratio:
            raise ValueError("min_edge_ratio must be smaller than max_edge_ratio")
        for field_name in ("edge_band", "dedup_distance", "refinement_residual", "max_refinement_shift"):
            object.__setattr__(self, field_name, _finite(getattr(self, field_name), field_name, low=1e-9, high=1.0))
        object.__setattr__(self, "side_offsets", _tuple_of_numbers(self.side_offsets, "side_offsets", length=2, low=0.0, high=1.0))
        object.__setattr__(self, "safe_threshold", _finite(self.safe_threshold, "safe_threshold", low=0.0, high=1.0))
        object.__setattr__(self, "aggressive_threshold", _finite(self.aggressive_threshold, "aggressive_threshold", low=0.0, high=1.0))
        if self.aggressive_threshold > self.safe_threshold:
            raise ValueError("aggressive_threshold cannot exceed safe_threshold")
        object.__setattr__(self, "risk_thresholds", _mapping(self.risk_thresholds, "risk_thresholds", value_low=0.0, value_high=1.0))

    def to_dict(self) -> dict[str, Any]:
        return {field.name: _plain(getattr(self, field.name)) for field in fields(self)}

    def sha256(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def replace(self, **changes: Any) -> "V7Parameters":
        known = {field.name for field in fields(self)}
        unknown = set(changes) - known
        if unknown:
            raise TypeError(f"unknown parameter field(s): {', '.join(sorted(unknown))}")
        return dc_replace(self, **changes)
