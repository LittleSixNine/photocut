"""Fixed, canonical parameters for the scanner-white V8 release candidate."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import hashlib
import json
import math
from typing import Any, Mapping

from photocut.config import V7_ALGORITHM_VERSION
from photocut.algorithms.v7.parameters import V7Parameters

from .edge_hypotheses import EdgeHypothesisConfig
from .scanner_selector import ScannerExteriorConfig, ScannerSelectorConfig
from .seed_selection import EdgeSeedSelectionConfig


CANDIDATE_POOL_DEFINITION = "v71-audit-plus-background-top1-seeded-edge-k4-dedup-plus-mask-v2"
LEGACY_EVIDENCE_SCHEMA = "v8-scanner-evidence-v1"
EVIDENCE_SCHEMA = "v8-scanner-evidence-v2"
_LEGACY_SELECTOR_FIELDS = frozenset({
    "exterior_weight",
    "area_weight",
    "prior_weight",
    "v7_edge_min_disagreement",
    "v7_edge_agreement_distance",
    "v7_edge_agreement_ratio",
    "v7_mask_agreement_distance",
    "v7_mask_outlier_ratio",
    "v7_mask_exterior_advantage",
    "edge_mask_agreement_distance",
    "edge_mask_outlier_ratio",
    "edge_mask_max_prior",
    "minimum_automatic_valid_sides",
    "manual_conflict_distance",
})
_CURRENT_SELECTOR_FIELDS = frozenset(
    item.name for item in fields(ScannerSelectorConfig)
)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value)).hexdigest()


@dataclass(frozen=True)
class V8Parameters:
    schema_version: int = 2
    scene_profile: str = "scanner_white"
    candidate_pool_definition: str = CANDIDATE_POOL_DEFINITION
    evidence_schema: str = EVIDENCE_SCHEMA
    edge_budget: int = 32
    mask_threshold: float = 0.2
    mask_min_foreground_probability: float = 0.75
    mask_min_component_ratio: float = 0.80
    mask_min_polygon_iou: float = 0.75
    mask_max_boundary_entropy: float = 0.85
    mask_max_refinement_shift: float = 0.065
    v7_algorithm_version: str = V7_ALGORITHM_VERSION
    edge_config: EdgeHypothesisConfig = field(default_factory=EdgeHypothesisConfig)
    edge_seed_config: EdgeSeedSelectionConfig = field(default_factory=EdgeSeedSelectionConfig)
    exterior_config: ScannerExteriorConfig = field(default_factory=ScannerExteriorConfig)
    selector_config: ScannerSelectorConfig = field(default_factory=ScannerSelectorConfig)

    def __post_init__(self) -> None:
        if self.schema_version not in {1, 2}:
            raise ValueError("unsupported V8 parameter schema")
        if self.scene_profile != "scanner_white":
            raise ValueError("V8 parameters are restricted to scanner_white")
        if self.candidate_pool_definition != CANDIDATE_POOL_DEFINITION:
            raise ValueError("candidate pool definition is not the sealed V8 pool")
        expected_evidence_schema = (
            LEGACY_EVIDENCE_SCHEMA if self.schema_version == 1 else EVIDENCE_SCHEMA
        )
        if self.evidence_schema != expected_evidence_schema:
            raise ValueError("unsupported V8 evidence schema")
        if self.v7_algorithm_version != V7_ALGORITHM_VERSION:
            raise ValueError("V8 candidate generation requires the current V7 identity")
        if type(self.edge_budget) is not int or not 1 <= self.edge_budget <= 256:
            raise ValueError("edge_budget must be an integer in [1, 256]")
        if (
            isinstance(self.mask_threshold, bool)
            or not isinstance(self.mask_threshold, (int, float))
            or not math.isfinite(float(self.mask_threshold))
            or not 0.05 <= float(self.mask_threshold) <= 0.95
        ):
            raise ValueError("mask_threshold is outside its bounded range")
        for name, lower, upper in (
            ("mask_min_foreground_probability", 0.5, 1.0),
            ("mask_min_component_ratio", 0.5, 1.0),
            ("mask_min_polygon_iou", 0.5, 1.0),
            ("mask_max_boundary_entropy", 0.0, 1.0),
            ("mask_max_refinement_shift", 0.0, 0.10),
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not lower <= float(value) <= upper
            ):
                raise ValueError(f"{name} is outside its bounded range")
        for name, expected in (
            ("edge_config", EdgeHypothesisConfig),
            ("edge_seed_config", EdgeSeedSelectionConfig),
            ("exterior_config", ScannerExteriorConfig),
            ("selector_config", ScannerSelectorConfig),
        ):
            if not isinstance(getattr(self, name), expected):
                raise TypeError(f"{name} must be {expected.__name__}")

    def to_dict(self) -> dict[str, Any]:
        selector_config = asdict(self.selector_config)
        if self.schema_version == 1:
            selector_config = {
                name: selector_config[name]
                for name in sorted(_LEGACY_SELECTOR_FIELDS)
            }
        return {
            "schema_version": self.schema_version,
            "scene_profile": self.scene_profile,
            "candidate_pool_definition": self.candidate_pool_definition,
            "evidence_schema": self.evidence_schema,
            "edge_budget": self.edge_budget,
            "mask_threshold": float(self.mask_threshold),
            "mask_min_foreground_probability": float(self.mask_min_foreground_probability),
            "mask_min_component_ratio": float(self.mask_min_component_ratio),
            "mask_min_polygon_iou": float(self.mask_min_polygon_iou),
            "mask_max_boundary_entropy": float(self.mask_max_boundary_entropy),
            "mask_max_refinement_shift": float(self.mask_max_refinement_shift),
            "v7_algorithm_version": self.v7_algorithm_version,
            "v7_parameter_sha256": V7Parameters(
                scene_profile=self.scene_profile
            ).sha256(),
            "edge_config": asdict(self.edge_config),
            "edge_seed_config": asdict(self.edge_seed_config),
            "exterior_config": asdict(self.exterior_config),
            "selector_config": selector_config,
        }

    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "V8Parameters":
        if not isinstance(value, Mapping):
            raise TypeError("V8 parameters must be an object")
        expected = {
            "schema_version", "scene_profile", "candidate_pool_definition",
            "evidence_schema", "edge_budget", "mask_threshold",
            "mask_min_foreground_probability", "mask_min_component_ratio",
            "mask_min_polygon_iou", "mask_max_boundary_entropy",
            "mask_max_refinement_shift",
            "v7_algorithm_version", "v7_parameter_sha256", "edge_config",
            "edge_seed_config", "exterior_config", "selector_config",
        }
        if set(value) != expected:
            raise ValueError("V8 parameter fields differ from the sealed schema")
        schema_version = value["schema_version"]
        if schema_version not in {1, 2}:
            raise ValueError("unsupported V8 parameter schema")
        selector_value = value["selector_config"]
        if not isinstance(selector_value, Mapping):
            raise TypeError("selector_config must be an object")
        expected_selector_fields = (
            _LEGACY_SELECTOR_FIELDS
            if schema_version == 1
            else _CURRENT_SELECTOR_FIELDS
        )
        if set(selector_value) != expected_selector_fields:
            raise ValueError("selector config fields differ from the sealed schema")
        result = cls(
            schema_version=schema_version,
            scene_profile=value["scene_profile"],
            candidate_pool_definition=value["candidate_pool_definition"],
            evidence_schema=value["evidence_schema"],
            edge_budget=value["edge_budget"],
            mask_threshold=value["mask_threshold"],
            mask_min_foreground_probability=value["mask_min_foreground_probability"],
            mask_min_component_ratio=value["mask_min_component_ratio"],
            mask_min_polygon_iou=value["mask_min_polygon_iou"],
            mask_max_boundary_entropy=value["mask_max_boundary_entropy"],
            mask_max_refinement_shift=value["mask_max_refinement_shift"],
            v7_algorithm_version=value["v7_algorithm_version"],
            edge_config=EdgeHypothesisConfig(**dict(value["edge_config"])),
            edge_seed_config=EdgeSeedSelectionConfig(**dict(value["edge_seed_config"])),
            exterior_config=ScannerExteriorConfig(**dict(value["exterior_config"])),
            selector_config=ScannerSelectorConfig(**dict(selector_value)),
        )
        if value["v7_parameter_sha256"] != result.to_dict()["v7_parameter_sha256"]:
            raise ValueError("V7 parameter identity mismatch")
        return result


__all__ = [
    "CANDIDATE_POOL_DEFINITION",
    "EVIDENCE_SCHEMA",
    "LEGACY_EVIDENCE_SCHEMA",
    "V8Parameters",
    "canonical_json",
    "canonical_sha256",
]
