"""Read-only joined view of archived objects, manifests, runs and latest annotations."""
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from photocut.data.annotation_store import AnnotationStore
from photocut.data.dataset_store import DatasetStore, atomic_write_json


SPLIT_VERSION = 1


@dataclass(frozen=True)
class DatasetSample:
    image_id: str
    object_path: Path
    boundary_corners: tuple
    annotation_id: str


def assign_split(image_id: str, salt: str, split_version: int = SPLIT_VERSION) -> str:
    digest = hashlib.sha256(
        f"{split_version}:{salt}:{image_id}".encode("utf-8")
    ).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "validation"
    return "frozen_test"


def load_json_from_path(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def load_or_create_split(
    experiment_dir: Path, samples: Sequence[DatasetSample], salt: str
) -> dict:
    split_path = experiment_dir / "split.json"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    existing = {}
    if split_path.exists():
        raw = load_json_from_path(split_path)
        if raw.get("schema_version") != 1:
            raise ValueError("Unsupported split schema version")
        if raw.get("salt") != salt:
            raise ValueError("Split salt has changed")
        existing = raw["assignments"]
        for image_id, set_name in existing.items():
            if set_name not in ("train", "validation", "frozen_test"):
                raise ValueError(f"Unknown set name {set_name!r} for {image_id}")

    assignments = dict(existing)
    for sample in samples:
        if sample.image_id not in assignments:
            assignments[sample.image_id] = assign_split(sample.image_id, salt)

    atomic_write_json(
        split_path,
        {
            "schema_version": 1,
            "split_version": SPLIT_VERSION,
            "salt": salt,
            "assignments": assignments,
        },
    )
    return {
        "schema_version": 1,
        "split_version": SPLIT_VERSION,
        "salt": salt,
        "assignments": assignments,
    }


def load_confirmed_samples(dataset_root: Path) -> list[DatasetSample]:
    dataset_root = Path(dataset_root).resolve()
    batches_dir = dataset_root / "batches"
    if not batches_dir.is_dir():
        return []

    latest_by_image: dict[str, tuple] = {}

    for batch_dir in sorted(batches_dir.iterdir()):
        manifest_path = batch_dir / "manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = load_json_from_path(manifest_path)
        if manifest.get("schema_version") != 1:
            continue

        annotation_path = batch_dir / "annotations.jsonl"
        if not annotation_path.is_file():
            continue
        store = AnnotationStore(annotation_path)
        annotations = store.latest_by_image()

        for img_entry in manifest.get("images", []):
            image_id = img_entry["image_id"]
            if image_id not in annotations:
                continue
            ann = annotations[image_id]
            object_path = _resolve_object_path(
                dataset_root, batch_dir, img_entry["object_path"]
            )
            if object_path is None:
                continue
            if _sha256_file(object_path) != image_id:
                continue
            boundary = ann.get("boundary_corners")
            if not boundary or len(boundary) != 4:
                continue
            try:
                bc = tuple((int(p[0]), int(p[1])) for p in boundary)
            except (TypeError, IndexError, ValueError):
                continue
            latest_by_image[image_id] = (object_path, bc, ann["annotation_id"])

    return [
        DatasetSample(
            image_id=img_id,
            object_path=obj_path,
            boundary_corners=bc,
            annotation_id=ann_id,
        )
        for img_id, (obj_path, bc, ann_id) in sorted(latest_by_image.items())
    ]


def _resolve_object_path(
    dataset_root: Path, batch_dir: Path, object_path: str
) -> Path | None:
    existing_candidates: list[Path] = []
    for candidate in (
        (batch_dir / object_path).resolve(),
        (dataset_root / object_path).resolve(),
    ):
        try:
            candidate.relative_to(dataset_root)
        except ValueError:
            continue
        if candidate.is_file():
            existing_candidates.append(candidate)
    if len(existing_candidates) == 1:
        return existing_candidates[0]
    return None


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"
