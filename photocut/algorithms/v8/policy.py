"""Immutable V8 research policy identity and tamper-evident JSON loading."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Mapping

from photocut.algorithms.v7.sealed_io import write_json_new_fsync

from . import (
    LEGACY_V8_ALGORITHM_VERSION,
    PRACTICAL_V8_ALGORITHM_VERSION,
    PRODUCTION_PROMOTION,
    V8_ALGORITHM_VERSION,
)
from .parameters import V8Parameters, canonical_sha256


_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
LEGACY_V8_POLICY_ID = "v8-auto-v1"
V8_POLICY_ID = "v8-auto-v2"
V82_POLICY_ID = "v8-auto-v3"


@dataclass(frozen=True)
class V8Policy:
    model_id: str
    model_sha256: str
    model_manifest_sha256: str
    runtime_config_sha256: str
    calibration_population_sha256: str
    calibration_run_sha256: str
    validated_core_sha256: str
    boundary_quality_artifact_sha256: str | None = None
    parameters: V8Parameters = field(default_factory=V8Parameters)
    schema_version: int = 2
    policy_id: str = V8_POLICY_ID
    algorithm_version: str = PRACTICAL_V8_ALGORITHM_VERSION
    production_promotion: bool = PRODUCTION_PROMOTION

    def __post_init__(self) -> None:
        if self.schema_version not in {1, 2, 3}:
            raise ValueError("unsupported V8 policy schema")
        expected_identity = {
            1: (LEGACY_V8_POLICY_ID, LEGACY_V8_ALGORITHM_VERSION, 1),
            2: (V8_POLICY_ID, PRACTICAL_V8_ALGORITHM_VERSION, 2),
            3: (V82_POLICY_ID, V8_ALGORITHM_VERSION, 2),
        }[self.schema_version]
        if (
            self.policy_id,
            self.algorithm_version,
            self.parameters.schema_version,
        ) != expected_identity:
            raise ValueError("V8 policy identity or parameter schema mismatch")
        if self.production_promotion is not False:
            raise ValueError("V8 production promotion must remain false")
        if not isinstance(self.parameters, V8Parameters):
            raise TypeError("parameters must be V8Parameters")
        if self.parameters.scene_profile != "scanner_white":
            raise ValueError("V8 policy is restricted to scanner_white")
        for name in (
            "model_id", "model_sha256", "model_manifest_sha256",
            "runtime_config_sha256", "calibration_population_sha256",
            "calibration_run_sha256", "validated_core_sha256",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
                raise ValueError(f"{name} must be a canonical SHA-256 identity")
        if self.schema_version == 3:
            if (
                not isinstance(self.boundary_quality_artifact_sha256, str)
                or not _SHA256_RE.fullmatch(
                    self.boundary_quality_artifact_sha256
                )
            ):
                raise ValueError(
                    "boundary-quality artifact must be a canonical SHA-256 identity"
                )
        elif self.boundary_quality_artifact_sha256 is not None:
            raise ValueError(
                "boundary-quality artifact is only valid for V8.2 policy schema"
            )

    def payload(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "policy_id": self.policy_id,
            "algorithm_version": self.algorithm_version,
            "production_promotion": self.production_promotion,
            "model_id": self.model_id,
            "model_sha256": self.model_sha256,
            "model_manifest_sha256": self.model_manifest_sha256,
            "runtime_config_sha256": self.runtime_config_sha256,
            "calibration_population_sha256": self.calibration_population_sha256,
            "calibration_run_sha256": self.calibration_run_sha256,
            "validated_core_sha256": self.validated_core_sha256,
            "parameters": self.parameters.to_dict(),
        }
        if self.schema_version == 3:
            payload["boundary_quality_artifact_sha256"] = (
                self.boundary_quality_artifact_sha256
            )
        return payload

    @property
    def policy_sha256(self) -> str:
        return canonical_sha256(self.payload())

    def to_dict(self) -> dict[str, Any]:
        return {**self.payload(), "policy_sha256": self.policy_sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "V8Policy":
        if not isinstance(value, Mapping):
            raise TypeError("V8 policy must be an object")
        expected = {
            "schema_version", "policy_id", "algorithm_version",
            "production_promotion", "model_id", "model_sha256",
            "model_manifest_sha256", "runtime_config_sha256",
            "calibration_population_sha256", "calibration_run_sha256",
            "validated_core_sha256", "parameters", "policy_sha256",
        }
        if value.get("schema_version") == 3:
            expected.add("boundary_quality_artifact_sha256")
        if set(value) != expected:
            raise ValueError("V8 policy fields differ from the sealed schema")
        recorded_hash = value["policy_sha256"]
        raw_payload = {key: value[key] for key in expected - {"policy_sha256"}}
        if recorded_hash != canonical_sha256(raw_payload):
            raise ValueError("V8 policy SHA-256 mismatch")
        kwargs = {key: value[key] for key in expected - {"policy_sha256", "parameters"}}
        policy = cls(parameters=V8Parameters.from_dict(value["parameters"]), **kwargs)
        if recorded_hash != policy.policy_sha256:
            raise ValueError("V8 policy SHA-256 mismatch")
        return policy


def load_v8_policy(path: str | Path) -> V8Policy:
    try:
        value = json.loads(Path(path).read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid V8 policy JSON") from exc
    return V8Policy.from_dict(value)


def write_v8_policy(path: str | Path, policy: V8Policy) -> None:
    if not isinstance(policy, V8Policy):
        raise TypeError("policy must be V8Policy")
    write_json_new_fsync(path, policy.to_dict())


__all__ = [
    "LEGACY_V8_POLICY_ID",
    "V8_POLICY_ID",
    "V82_POLICY_ID",
    "V8Policy",
    "load_v8_policy",
    "write_v8_policy",
]
