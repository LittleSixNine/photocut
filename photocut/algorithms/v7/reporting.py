"""Crash-durable shadow evaluation artifacts, independent of production data."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

from .sealed_io import write_json_new_fsync


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def build_run_identity(manifest_hash: str, split: str, evaluated_ids: Iterable[str],
                       v7_parameter_hash: str, git_commit: str, analysis_hash: str,
                       environment_fingerprint: str, origin_group_hash: str | None = None) -> dict[str, Any]:
    if split not in {"train", "validation"}:
        raise PermissionError("shadow evaluation accepts only train/validation")
    ids = sorted(str(value) for value in evaluated_ids)
    payload = {"manifest_hash": manifest_hash, "split": split, "evaluated_ids": ids,
               "v7_parameter_hash": v7_parameter_hash, "git_commit": git_commit,
               "analysis_hash": analysis_hash, "environment_fingerprint": environment_fingerprint,
               "origin_group_hash": origin_group_hash}
    run_id = hashlib.sha256(_canonical(payload).encode()).hexdigest()
    return {**payload, "run_id": run_id}


def write_evaluation_report(output_dir: str | Path, *, run_identity: Mapping[str, Any],
                            paired_results: Iterable[Mapping[str, Any]], metrics: Mapping[str, Any],
                            slices: Mapping[str, Any], gate: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    paired = [dict(item) for item in paired_results]
    report = {"run_id": run_identity.get("run_id"), "metrics": dict(metrics), "slices": dict(slices), "gate": dict(gate)}
    write_json_new_fsync(root / "run.json", dict(run_identity))
    write_jsonl_new_fsync(root / "paired_results.jsonl", paired)
    write_json_new_fsync(root / "metrics.json", dict(metrics))
    write_json_new_fsync(root / "slices.json", dict(slices))
    write_json_new_fsync(root / "gate.json", dict(gate))
    (root / "error_overlays").mkdir(exist_ok=True)
    write_json_new_fsync(root / "diagnostics.json", {"feature_cache": {}, "source_mutations": 0})
    _write_text_fsync(root / "report.md", "# v7 evaluation\n\n" + json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def write_jsonl_new_fsync(path: str | Path, rows: list[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in rows).encode()
    fd, temp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        os.write(fd, payload)
        os.fsync(fd)
        os.close(fd)
        os.replace(temp, path)
        temp = None
    finally:
        if fd is not None:
            try: os.close(fd)
            except OSError: pass
        if temp:
            try: os.unlink(temp)
            except OSError: pass


def _write_text_fsync(path: Path, text: str) -> None:
    data = text.encode("utf-8")
    fd, temp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        os.write(fd, data); os.fsync(fd); os.close(fd); fd = None
        os.replace(temp, path); temp = None
    finally:
        if fd is not None:
            try: os.close(fd)
            except OSError: pass
        if temp:
            try: os.unlink(temp)
            except OSError: pass


__all__ = ["build_run_identity", "write_evaluation_report", "write_jsonl_new_fsync"]
