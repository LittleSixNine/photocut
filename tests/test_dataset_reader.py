"""Tests for dataset_reader: stable splits and confirmed sample joining."""
import hashlib
import os
import tempfile
import unittest
from pathlib import Path

from photocut.data.annotation_store import AnnotationStore
from photocut.confirmation.model import build_annotation_event
from photocut.data.dataset_store import DatasetStore, atomic_write_json
from photocut.data.dataset_reader import (
    DatasetSample,
    assign_split,
    load_confirmed_samples,
    load_or_create_split,
)


class StableSplitTests(unittest.TestCase):
    def test_assignment_is_stable_and_uses_three_named_sets(self):
        assignments = {
            assign_split(f"sha256:{index:064x}", "photocut-v1")
            for index in range(1000)
        }
        self.assertEqual({"train", "validation", "frozen_test"}, assignments)
        self.assertEqual(
            assign_split("sha256:" + "a" * 64, "photocut-v1"),
            assign_split("sha256:" + "a" * 64, "photocut-v1"),
        )

    def test_existing_split_never_reassigns_old_images(self):
        with tempfile.TemporaryDirectory(prefix="dsr_") as tmp:
            experiment_dir = Path(tmp) / "experiment"
            boundary = ((0, 0), (9, 0), (9, 9), (0, 9))
            first_samples = [
                DatasetSample(
                    image_id=f"sha256:{index:064x}",
                    object_path=Path(f"object-{index}"),
                    boundary_corners=boundary,
                    annotation_id=f"ann_{index}",
                )
                for index in range(20)
            ]
            first = load_or_create_split(
                experiment_dir, first_samples, salt="photocut-v1"
            )
            expanded_samples = first_samples + [
                DatasetSample(
                    image_id=f"sha256:{index:064x}",
                    object_path=Path(f"object-{index}"),
                    boundary_corners=boundary,
                    annotation_id=f"ann_{index}",
                )
                for index in range(20, 30)
            ]
            expanded = load_or_create_split(
                experiment_dir, expanded_samples, salt="photocut-v1"
            )
            for sample in first_samples:
                self.assertEqual(
                    first["assignments"][sample.image_id],
                    expanded["assignments"][sample.image_id],
                )
            self.assertEqual(30, len(expanded["assignments"]))

    def test_duplicate_image_across_batches_produces_one_latest_sample(self):
        with tempfile.TemporaryDirectory(
            prefix="dsr_", dir=Path(tempfile.gettempdir()).resolve()
        ) as tmp:
            dataset_root = Path(tmp) / "datasets"
            source = Path(tmp) / "scan.jpg"
            source.write_bytes(b"\xff\xd8\xff\xe0fixture\xff\xd9")
            dataset_store = DatasetStore(dataset_root)
            archived = dataset_store.archive_image(source, image_size=(10, 10))
            algorithm = [[1, 1], [8, 1], [8, 8], [1, 8]]
            newest_boundary = [[2, 1], [8, 1], [8, 8], [1, 8]]

            for index, boundary in enumerate((algorithm, newest_boundary), 1):
                batch_dir = dataset_root / "batches" / f"batch-{index}"
                batch_dir.mkdir(parents=True)
                atomic_write_json(
                    batch_dir / "manifest.json",
                    {
                        "schema_version": 1,
                        "batch_id": f"batch-{index}",
                        "images": [{
                            "image_id": archived.image_id,
                            "object_path": os.path.relpath(
                                archived.object_path, batch_dir
                            ),
                            "source_filename": "scan.jpg",
                        }],
                    },
                )
                event = build_annotation_event(
                    image_id=archived.image_id,
                    run_id=f"run-{index}",
                    algorithm_boundary_corners=algorithm,
                    boundary_corners=boundary,
                    adjusted_corner_indices=set() if index == 1 else {0},
                    confirmation_duration_ms=1000,
                )
                event["confirmed_at"] = f"2026-07-2{index}T10:00:00+08:00"
                AnnotationStore(batch_dir / "annotations.jsonl").append(event)

            samples = load_confirmed_samples(dataset_root)
            self.assertEqual(1, len(samples))
            self.assertEqual(
                tuple(map(tuple, newest_boundary)), samples[0].boundary_corners
            )


class ConfirmedSamplePathTests(unittest.TestCase):
    def _write_confirmed_batch(self, dataset_root, image_id, object_path):
        batch_dir = dataset_root / "batches" / "batch-1"
        batch_dir.mkdir(parents=True, exist_ok=True)
        boundary = [[1, 1], [8, 1], [8, 8], [1, 8]]
        atomic_write_json(
            batch_dir / "manifest.json",
            {
                "schema_version": 1,
                "batch_id": "batch-1",
                "images": [{"image_id": image_id, "object_path": object_path}],
            },
        )
        AnnotationStore(batch_dir / "annotations.jsonl").append(
            build_annotation_event(
                image_id=image_id,
                run_id="run-1",
                algorithm_boundary_corners=boundary,
                boundary_corners=boundary,
                adjusted_corner_indices=set(),
                confirmation_duration_ms=1000,
            )
        )

    def test_path_with_dataset_root_prefix_but_outside_root_is_ignored(self):
        with tempfile.TemporaryDirectory(prefix="dsr_") as tmp:
            root = Path(tmp)
            dataset_root = root / "datasets"
            outside_path = root / "datasets-escape" / "scan.jpg"
            outside_path.parent.mkdir()
            contents = b"outside-object"
            outside_path.write_bytes(contents)
            image_id = "sha256:" + hashlib.sha256(contents).hexdigest()
            self._write_confirmed_batch(
                dataset_root, image_id, "../../../datasets-escape/scan.jpg"
            )

            self.assertEqual([], load_confirmed_samples(dataset_root))

    def test_symlinked_object_that_escapes_dataset_root_is_ignored(self):
        with tempfile.TemporaryDirectory(prefix="dsr_") as tmp:
            root = Path(tmp)
            dataset_root = root / "datasets"
            outside_path = root / "datasets-escape" / "scan.jpg"
            outside_path.parent.mkdir()
            contents = b"outside-object"
            outside_path.write_bytes(contents)
            image_id = "sha256:" + hashlib.sha256(contents).hexdigest()
            objects_dir = dataset_root / "objects"
            objects_dir.mkdir(parents=True)
            (objects_dir / "escape.jpg").symlink_to(outside_path)
            self._write_confirmed_batch(dataset_root, image_id, "objects/escape.jpg")

            self.assertEqual([], load_confirmed_samples(dataset_root))

    def test_ambiguous_legacy_and_canonical_object_paths_are_ignored(self):
        with tempfile.TemporaryDirectory(prefix="dsr_") as tmp:
            dataset_root = Path(tmp) / "datasets"
            contents = b"same-object-content"
            image_id = "sha256:" + hashlib.sha256(contents).hexdigest()
            batch_object = (
                dataset_root / "batches" / "batch-1" / "objects" / "scan.jpg"
            )
            canonical_object = dataset_root / "objects" / "scan.jpg"
            batch_object.parent.mkdir(parents=True)
            canonical_object.parent.mkdir(parents=True)
            batch_object.write_bytes(contents)
            canonical_object.write_bytes(contents)
            self._write_confirmed_batch(dataset_root, image_id, "objects/scan.jpg")

            self.assertEqual([], load_confirmed_samples(dataset_root))

    def test_missing_object_or_annotations_are_ignored(self):
        with tempfile.TemporaryDirectory(prefix="dsr_") as tmp:
            dataset_root = Path(tmp) / "datasets"
            image_id = "sha256:" + "0" * 64
            batch_dir = dataset_root / "batches" / "batch-1"
            batch_dir.mkdir(parents=True)
            atomic_write_json(
                batch_dir / "manifest.json",
                {
                    "schema_version": 1,
                    "batch_id": "batch-1",
                    "images": [
                        {"image_id": image_id, "object_path": "objects/missing.jpg"}
                    ],
                },
            )

            self.assertEqual([], load_confirmed_samples(dataset_root))

            self._write_confirmed_batch(dataset_root, image_id, "objects/missing.jpg")
            self.assertEqual([], load_confirmed_samples(dataset_root))
