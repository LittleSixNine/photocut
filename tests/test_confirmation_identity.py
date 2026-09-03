import json
import tempfile
import unittest
from pathlib import Path

from photocut.confirmation.identity import match_finalized_detection
from photocut.data.dataset_store import BatchLock, DatasetStore, atomic_write_json
from photocut import cli as photocut_cli

class ConfirmationIdentityTests(unittest.TestCase):
    def setUp(self):
        self.corners = [[1, 2], [90, 2], [90, 70], [1, 70]]
        self.entry = {
            "batch_id": "batch-a", "run_id": "run-a",
            "image_id": "sha256:image", "source_id": "source-a",
            "detection_id": "det-a", "algorithm_version": "7.0",
            "algorithm_boundary_corners": self.corners,
        }
        self.reference = {"batch_id": "batch-a", "run_id": "run-a"}
        self.finalized = {
            "status": "complete", "batch_id": "batch-a", "run_id": "run-a",
            "images": [{
                **self.entry,
                "detector_requested": "auto",
                "detector_used": "v7",
            }],
        }

    def test_exact_finalized_per_image_identity_is_returned(self):
        evidence = match_finalized_detection(
            self.entry, self.reference, self.finalized
        )
        self.assertEqual("7.0", evidence.algorithm_version)
        self.assertEqual("auto", evidence.detector_requested)
        self.assertEqual("v7", evidence.detector_used)
        self.assertEqual(tuple(map(tuple, self.corners)), evidence.algorithm_boundary_corners)

    def test_same_path_and_content_from_another_run_is_rejected(self):
        for field, value in (("batch_id", "batch-b"), ("run_id", "run-b")):
            with self.subTest(field=field):
                changed = dict(self.entry, **{field: value})
                self.assertIsNone(match_finalized_detection(
                    changed, self.reference, self.finalized
                ))
        forged_envelope = dict(self.finalized, run_id="run-b")
        self.assertIsNone(match_finalized_detection(
            self.entry, self.reference, forged_envelope
        ))

    def test_mutable_algorithm_baseline_tamper_is_rejected(self):
        changed = dict(self.entry, algorithm_boundary_corners=[[0, 0]] * 4)
        self.assertIsNone(match_finalized_detection(
            changed, self.reference, self.finalized
        ))

    def test_reference_run_identity_cannot_be_forged(self):
        forged = dict(self.reference, run_id="run-b")
        self.assertIsNone(match_finalized_detection(
            self.entry, forged, self.finalized
        ))

    def test_duplicate_per_image_result_and_unfinalized_run_fail_closed(self):
        duplicate = dict(self.finalized, images=self.finalized["images"] * 2)
        self.assertIsNone(match_finalized_detection(
            self.entry, self.reference, duplicate
        ))
        pending = dict(self.finalized, status="in_progress")
        self.assertIsNone(match_finalized_detection(
            self.entry, self.reference, pending
        ))


class FinalizedEvidenceLoaderTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.root = Path(self.temp_dir.name)
        self.output = self.root / "output"
        self.output.mkdir()
        self.dataset_root = self.root / "dataset"
        self.store = DatasetStore(self.dataset_root)
        self.paths = self.store.start_batch(runtime={"algorithm_version": "7.0", "parameters": {}})
        self.corners = [[1, 2], [90, 2], [90, 70], [1, 70]]
        self.entry = {
            "batch_id": self.paths.batch_id,
            "run_id": f"run_{self.paths.batch_id}",
            "image_id": "sha256:image",
            "source_id": "source-a",
            "detection_id": "det-a",
            "algorithm_boundary_corners": self.corners,
        }
        with BatchLock(self.paths):
            run = json.loads(self.paths.run_in_progress.read_text(encoding="utf-8"))
            run["images"] = [{
                **self.entry,
                "algorithm_version": "7.0",
                "detector_requested": "auto",
                "detector_used": "v7",
            }]
            atomic_write_json(self.paths.run_in_progress, run)
            self.store.finalize_run(self.paths)
        atomic_write_json(
            self.output / ".photocut_batch.json",
            {
                "schema_version": 1,
                "batch_id": self.paths.batch_id,
                "run_id": f"run_{self.paths.batch_id}",
                "dataset_root": str(self.dataset_root),
            },
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_loader_returns_only_matching_finalized_row(self):
        evidence = photocut_cli._load_finalized_detection_evidence(
            self.entry, self.output, self.root
        )
        self.assertEqual("7.0", evidence.algorithm_version)
        self.assertEqual("v7", evidence.detector_used)

    def test_loader_rejects_mutable_baseline_tamper(self):
        changed = dict(self.entry, algorithm_boundary_corners=[[0, 0]] * 4)
        self.assertIsNone(photocut_cli._load_finalized_detection_evidence(
            changed, self.output, self.root
        ))

    def test_loader_rejects_symlinked_reference(self):
        outside = self.root / "reference.json"
        outside.write_bytes((self.output / ".photocut_batch.json").read_bytes())
        (self.output / ".photocut_batch.json").unlink()
        (self.output / ".photocut_batch.json").symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "regular file"):
            photocut_cli._load_finalized_detection_evidence(
                self.entry, self.output, self.root
            )

    def test_loader_ignores_inprogress_when_finalized_run_is_absent(self):
        self.paths.production_run.unlink()
        self.paths.run_in_progress.write_text("{}", encoding="utf-8")
        self.assertIsNone(photocut_cli._load_finalized_detection_evidence(
            self.entry, self.output, self.root
        ))


if __name__ == "__main__":
    unittest.main()
