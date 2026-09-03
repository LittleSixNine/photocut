"""Fail-closed conversion of confirmed truth into full EXIF-normalized space."""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from photocut.algorithms.v7.input import LoadedImage


COORDINATE_SPACES = frozenset({"stored_raster_original", "full_exif_normalized"})
CORNER_ORDER = "full_normalized_tl_tr_br_bl"


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _points(value: Any, name: str) -> tuple[tuple[float, float], ...]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"{name} must contain four corners")
    try:
        points = tuple((float(point[0]), float(point[1])) for point in value)
    except (TypeError, ValueError, IndexError, OverflowError) as exc:
        raise ValueError(f"{name} must contain finite corners") from exc
    if any(not math.isfinite(x) or not math.isfinite(y) for x, y in points):
        raise ValueError(f"{name} must contain finite corners")
    return points


def _map(matrix: np.ndarray, points: Iterable[Sequence[float]]) -> tuple[tuple[float, float], ...]:
    result = []
    for x, y in _points(tuple(points), "corners"):
        mapped = matrix @ np.asarray((x, y, 1.0), dtype=np.float64)
        if not np.isfinite(mapped).all() or abs(float(mapped[2])) <= 1e-12:
            raise ValueError("coordinate transform produced an invalid point")
        result.append((float(mapped[0] / mapped[2]), float(mapped[1] / mapped[2])))
    return tuple(result)


def _transform_identity(loaded: LoadedImage) -> dict[str, Any]:
    return {
        "source_sha256": "sha256:" + loaded.source_sha256.removeprefix("sha256:"),
        "exif_orientation": loaded.exif_orientation,
        "original_size": list(loaded.original_size),
        "full_normalized_size": list(loaded.full_normalized_size),
        "analysis_size": list(loaded.normalized_size),
        "forward_transform": [list(row) for row in loaded.forward_transform],
        "inverse_transform": [list(row) for row in loaded.inverse_transform],
        "analysis_to_full_transform": [list(row) for row in loaded.analysis_to_full_transform],
    }


def make_coordinate_provenance(
    annotation: Mapping[str, Any],
    loaded: LoadedImage,
    *,
    coordinate_space: str,
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Seal an explicit coordinate interpretation backed by durable evidence.

    The caller must derive ``coordinate_space`` from an import, batch or
    confirmation record.  This function records and seals that decision; it
    never guesses from corner bounds or image orientation.
    """
    annotation_id = annotation.get("annotation_id")
    image_id = annotation.get("image_id")
    if not isinstance(annotation_id, str) or not annotation_id or not isinstance(image_id, str) or not image_id:
        raise ValueError("annotation identity is required")
    _points(annotation.get("boundary_corners"), "annotation boundary_corners")
    if coordinate_space not in COORDINATE_SPACES:
        raise ValueError("unsupported coordinate provenance space")
    if not isinstance(evidence, Mapping):
        raise ValueError("coordinate provenance evidence is required")
    if not isinstance(evidence.get("kind"), str) or not evidence["kind"].strip():
        raise ValueError("coordinate provenance evidence kind is required")
    if not isinstance(evidence.get("identity"), str) or not evidence["identity"].strip():
        raise ValueError("coordinate provenance evidence identity is required")

    transform = _transform_identity(loaded)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "annotation_id": annotation_id,
        "image_id": image_id,
        "coordinate_space": coordinate_space,
        "corner_order": CORNER_ORDER,
        "evidence": dict(evidence),
        "evidence_hash": _sha256(_canonical(dict(evidence))),
        "source_sha256": transform["source_sha256"],
        "exif_orientation": transform["exif_orientation"],
        "original_size": transform["original_size"],
        "full_normalized_size": transform["full_normalized_size"],
        "analysis_size": transform["analysis_size"],
        "transform_hash": _sha256(_canonical(transform)),
    }
    payload["provenance_id"] = _sha256(_canonical(payload))
    return payload


def derive_coordinate_decision(
    annotation: Mapping[str, Any],
    loaded: LoadedImage,
    *,
    run_identity: Mapping[str, Any],
    run_image: Mapping[str, Any],
    stored_coordinate_rule: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Derive a coordinate decision from durable import/run identities.

    Three forms of evidence are accepted: an explicit reviewed import rule,
    a V7 run that records both stored and normalized identities, or a clean
    legacy run whose GUI raster dimensions equal the decoded full-normalized
    dimensions.  Bounds and corner values are never used to infer semantics.
    """
    annotation_id = annotation.get("annotation_id")
    image_id = annotation.get("image_id")
    run_id = annotation.get("run_id")
    if (
        not isinstance(annotation_id, str)
        or not annotation_id
        or not isinstance(image_id, str)
        or not image_id
        or not isinstance(run_id, str)
        or not run_id
        or run_identity.get("run_id") != run_id
        or run_image.get("run_id", run_identity.get("run_id")) != run_id
        or run_image.get("image_id") != image_id
    ):
        raise ValueError("coordinate evidence identity mismatch")

    batch_id = run_identity.get("batch_id")
    if stored_coordinate_rule is not None:
        provenance = annotation.get("provenance")
        if (
            not isinstance(provenance, Mapping)
            or run_image.get("legacy_import") is not True
            or stored_coordinate_rule.get("coordinate_space") != "stored_raster_original"
            or stored_coordinate_rule.get("batch_id") != batch_id
            or not isinstance(stored_coordinate_rule.get("evidence_document_hash"), str)
            or not str(stored_coordinate_rule["evidence_document_hash"]).startswith("sha256:")
        ):
            raise ValueError("ambiguous stored-raster coordinate evidence")
        identity = {
            "annotation_id": annotation_id,
            "batch_id": batch_id,
            "run_id": run_id,
            "annotation_provenance": dict(provenance),
            "evidence_document_hash": stored_coordinate_rule["evidence_document_hash"],
            "run_artifact_hash": run_identity.get("run_artifact_hash"),
        }
        return {
            "coordinate_space": "stored_raster_original",
            "evidence": {
                "kind": "reviewed_legacy_import_transform",
                "identity": _sha256(_canonical(identity)),
            },
        }

    legacy = run_image.get("legacy_info")
    if isinstance(legacy, Mapping):
        detection = legacy.get("detection_identity")
        if (
            list(loaded.original_size) == legacy.get("source_original_size")
            and list(loaded.full_normalized_size) == legacy.get("normalized_size")
            and isinstance(detection, Mapping)
            and detection.get("orientation_transform") == loaded.orientation_transform
        ):
            identity = {
                "annotation_id": annotation_id,
                "batch_id": batch_id,
                "run_id": run_id,
                "source_original_size": legacy["source_original_size"],
                "normalized_size": legacy["normalized_size"],
                "orientation_transform": detection["orientation_transform"],
                "run_artifact_hash": run_identity.get("run_artifact_hash"),
            }
            return {
                "coordinate_space": "full_exif_normalized",
                "evidence": {
                    "kind": "v7_normalized_confirmation_transaction",
                    "identity": _sha256(_canonical(identity)),
                },
            }

        runtime = run_identity.get("runtime")
        git = runtime.get("git") if isinstance(runtime, Mapping) else None
        environment = runtime.get("environment") if isinstance(runtime, Mapping) else None
        if (
            legacy.get("original_size") == list(loaded.full_normalized_size)
            and isinstance(git, Mapping)
            and git.get("available") is True
            and git.get("dirty") is False
            and isinstance(git.get("commit"), str)
            and git["commit"]
            and isinstance(environment, Mapping)
            and isinstance(environment.get("opencv"), str)
            and environment["opencv"]
        ):
            identity = {
                "annotation_id": annotation_id,
                "batch_id": batch_id,
                "run_id": run_id,
                "recorded_gui_size": legacy["original_size"],
                "full_normalized_size": list(loaded.full_normalized_size),
                "git_commit": git["commit"],
                "opencv": environment["opencv"],
                "run_artifact_hash": run_identity.get("run_artifact_hash"),
            }
            return {
                "coordinate_space": "full_exif_normalized",
                "evidence": {
                    "kind": "clean_legacy_gui_normalized_raster",
                    "identity": _sha256(_canonical(identity)),
                },
            }

    raise ValueError("ambiguous coordinate provenance")


def _validate_provenance(
    annotation: Mapping[str, Any],
    loaded: LoadedImage,
    provenance: Mapping[str, Any] | None,
) -> str:
    if not isinstance(provenance, Mapping):
        raise ValueError("coordinate provenance is required")
    if provenance.get("schema_version") != 1:
        raise ValueError("unsupported coordinate provenance schema")
    recorded_id = provenance.get("provenance_id")
    unsigned = {key: value for key, value in provenance.items() if key != "provenance_id"}
    if recorded_id != _sha256(_canonical(unsigned)):
        raise ValueError("coordinate provenance hash mismatch")
    if provenance.get("annotation_id") != annotation.get("annotation_id") or provenance.get("image_id") != annotation.get("image_id"):
        raise ValueError("coordinate provenance annotation identity mismatch")
    coordinate_space = provenance.get("coordinate_space")
    if coordinate_space not in COORDINATE_SPACES or provenance.get("corner_order") != CORNER_ORDER:
        raise ValueError("invalid coordinate provenance semantics")
    evidence = provenance.get("evidence")
    if not isinstance(evidence, Mapping) or provenance.get("evidence_hash") != _sha256(_canonical(dict(evidence))):
        raise ValueError("coordinate provenance evidence mismatch")

    transform = _transform_identity(loaded)
    expected = {
        "source_sha256": transform["source_sha256"],
        "exif_orientation": transform["exif_orientation"],
        "original_size": transform["original_size"],
        "full_normalized_size": transform["full_normalized_size"],
        "analysis_size": transform["analysis_size"],
        "transform_hash": _sha256(_canonical(transform)),
    }
    if any(provenance.get(key) != value for key, value in expected.items()):
        raise ValueError("coordinate provenance transform identity mismatch")
    return str(coordinate_space)


def _validate_full_quad(points: tuple[tuple[float, float], ...], size: tuple[int, int]) -> None:
    width, height = size
    if any(x < 0 or y < 0 or x > width - 1 or y > height - 1 for x, y in points):
        raise ValueError("full-normalized truth exceeds image bounds")
    turns = []
    for index in range(4):
        a, b, c = points[index], points[(index + 1) % 4], points[(index + 2) % 4]
        turns.append((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]))
    if not (all(turn > 1e-9 for turn in turns) or all(turn < -1e-9 for turn in turns)):
        raise ValueError("full-normalized truth must be a convex ordered quad")


def resolve_full_normalized_truth(
    annotation: Mapping[str, Any],
    loaded: LoadedImage,
    provenance: Mapping[str, Any] | None,
) -> tuple[tuple[float, float], ...]:
    """Resolve truth only when its historical coordinate meaning is sealed."""
    coordinate_space = _validate_provenance(annotation, loaded, provenance)
    source = _points(annotation.get("boundary_corners"), "annotation boundary_corners")
    if coordinate_space == "stored_raster_original":
        # For bounded oversized-image decoding, ``forward_transform`` maps
        # stored raster coordinates into the analysis preview.  Lift that
        # result through the explicit endpoint transform before publishing
        # full-normalized truth.
        result = loaded.map_analysis_to_full(loaded.map_original_to_normalized(source))
    else:
        result = source
    _validate_full_quad(result, loaded.full_normalized_size)
    return result


def map_full_normalized_to_analysis(
    loaded: LoadedImage,
    points: Iterable[Sequence[float]],
) -> tuple[tuple[float, float], ...]:
    """Map full EXIF-normalized truth into the detector/training tensor."""
    matrix = np.asarray(loaded.analysis_to_full_transform, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("invalid analysis-to-full transform")
    return _map(np.linalg.inv(matrix), points)


__all__ = [
    "COORDINATE_SPACES",
    "CORNER_ORDER",
    "derive_coordinate_decision",
    "make_coordinate_provenance",
    "map_full_normalized_to_analysis",
    "resolve_full_normalized_truth",
]
