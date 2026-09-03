"""Blinded prospective study assignment and event contracts."""
from __future__ import annotations

import copy
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def assignment_hash(assignment: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(_canonical(assignment).encode()).hexdigest()


def _stratum(sample: Mapping[str, Any]) -> str:
    labels = sample.get("audited_labels", {})
    values = labels.get("slices", ()) if isinstance(labels, Mapping) else labels
    return "|".join(sorted(str(value) for value in values)) or "unlabeled"


def assign_study_arms(samples: Iterable[Mapping[str, Any]], *, seed: int,
                      assignment_id: str | None = None) -> dict[str, Any]:
    """Assign one hidden arm per origin group, balanced within slice strata."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    records = [dict(sample) for sample in samples]
    groups: dict[str, dict[str, Any]] = {}
    for sample in records:
        image_id = str(sample.get("image_id", ""))
        group_id = str(sample.get("origin_group_id", image_id))
        if not image_id or not group_id:
            raise ValueError("samples require image_id and origin_group_id")
        existing = groups.get(group_id)
        if existing is not None and existing["stratum"] != _stratum(sample):
            raise ValueError("origin group crosses strata")
        groups.setdefault(group_id, {"stratum": _stratum(sample), "samples": []})["samples"].append(sample)
    rng = random.Random(seed)
    by_stratum: dict[str, list[str]] = {}
    for group_id, data in groups.items():
        by_stratum.setdefault(data["stratum"], []).append(group_id)
    arms: dict[str, str] = {}
    for stratum, group_ids in sorted(by_stratum.items()):
        shuffled = sorted(group_ids)
        rng.shuffle(shuffled)
        for index, group_id in enumerate(shuffled):
            arms[group_id] = "arm_a" if index % 2 == 0 else "arm_b"
    assignment_id = assignment_id or f"study_{seed}_{hashlib.sha256(str(seed).encode()).hexdigest()[:12]}"
    assignments = []
    for sample in sorted(records, key=lambda item: str(item["image_id"])):
        group_id = str(sample["origin_group_id"])
        assignments.append({
            "image_id": str(sample["image_id"]),
            "origin_group_id": group_id,
            "stratum": groups[group_id]["stratum"],
            "arm_pseudonym": arms[group_id],
        })
    return {"schema_version": 1, "assignment_id": assignment_id, "seed": seed,
            "arm_labels": {"arm_a": "hidden", "arm_b": "hidden"},
            "assignments": assignments}


def blinded_view(assignment: Mapping[str, Any], *, initial_corners: Any = None,
                 risks: Any = None) -> dict[str, Any]:
    """Return only screen-safe data; detector names/status are intentionally omitted."""
    allowed = {"image_id", "origin_group_id", "arm_pseudonym", "stratum"}
    view = {key: copy.deepcopy(assignment[key]) for key in allowed if key in assignment}
    if initial_corners is not None:
        view["initial_corners"] = copy.deepcopy(initial_corners)
    elif "initial_corners" in assignment:
        view["initial_corners"] = copy.deepcopy(assignment["initial_corners"])
    if risks is not None:
        view["risks"] = copy.deepcopy(risks)
    elif "risks" in assignment:
        view["risks"] = copy.deepcopy(assignment["risks"])
    return view


def build_study_event(*, assignment_id: str, arm_pseudonym: str, image_id: str,
                      initial_corners: Any, operation: str, duration_ms: int,
                      jitter_threshold_px: float, final_corners: Any = None,
                      final_truth: Any = None, risks: Any = (),
                      evidence_type: str = "observational") -> dict[str, Any]:
    if operation not in {"direct_top1", "accepted_alternate", "dragged", "fallback", "skipped", "failed", "cancelled"}:
        raise ValueError("invalid study operation")
    if not isinstance(assignment_id, str) or not assignment_id or not isinstance(arm_pseudonym, str) or not arm_pseudonym:
        raise ValueError("assignment identity is required")
    if type(duration_ms) is not int or duration_ms < 0:
        raise ValueError("duration_ms must be non-negative integer")
    event = {
        "schema_version": 1, "event_type": "study", "event_id": "",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "assignment_id": assignment_id, "arm_pseudonym": arm_pseudonym,
        "image_id": image_id, "initial_corners": copy.deepcopy(initial_corners),
        "operation": operation, "duration_ms": duration_ms,
        "jitter_threshold_px": float(jitter_threshold_px),
        "final_corners": copy.deepcopy(final_corners), "final_truth": copy.deepcopy(final_truth),
        "risks": list(risks), "evidence_type": evidence_type,
    }
    event["event_id"] = hashlib.sha256(_canonical({k: v for k, v in event.items() if k != "event_id"}).encode()).hexdigest()
    return event


def write_assignment_new(path: str | Path, assignment: Mapping[str, Any]) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(assignment, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)


__all__ = ["assign_study_arms", "assignment_hash", "blinded_view", "build_study_event", "write_assignment_new"]
