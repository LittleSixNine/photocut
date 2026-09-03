import tempfile
import unittest
from pathlib import Path

from photocut.data.annotation_store import AnnotationStore
from photocut.confirmation.model import build_annotation_event
from photocut.algorithms.v7.confirmation_transaction import (
    ConfirmationConflictError,
    V7ConfirmationTransaction,
    can_crop_v7,
)
from photocut.algorithms.v7.feedback_store import FeedbackStore


class ConfirmationTransactionTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.feedback = FeedbackStore(root / "v7_feedback.jsonl")
        self.annotations = AnnotationStore(root / "annotations.jsonl")
        self.identity = {
            "detection_id": "det-1",
            "image_id": "sha256:source",
            "source_hash": "sha256:source",
            "orientation_transform": "exif_1",
            "algorithm_version": "7.0",
            "parameter_sha256": "a" * 64,
            "mode": "safe",
        }
        self.algorithm = [[1, 1], [19, 1], [19, 19], [1, 19]]
        self.boundary = [[2, 2], [18, 2], [18, 18], [2, 18]]

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_commit_requires_feedback_binding_before_gui_or_crop(self):
        transaction = V7ConfirmationTransaction(self.feedback, self.annotations, self.identity)
        self.assertFalse(can_crop_v7(self.feedback, self.identity, "ann_missing", self.boundary))
        annotation = transaction.commit(self.algorithm, self.boundary, [0, 1, 2, 3], run_id="run-1")
        self.assertTrue(can_crop_v7(self.feedback, self.identity, annotation["annotation_id"], self.boundary))
        phases = [event["transaction_phase"] for event in self.feedback.events() if event["event_type"] == "confirmation"]
        self.assertEqual(["PREPARED", "ANNOTATION_COMMITTED", "FEEDBACK_COMMITTED"], phases)

    def test_recovery_finishes_matching_annotation_and_aborts_missing_one(self):
        transaction = V7ConfirmationTransaction(self.feedback, self.annotations, self.identity)
        prepared = transaction.prepare(self.algorithm, self.boundary, [0, 1, 2, 3], run_id="run-1")
        recovered = transaction.recover(prepared["annotation_id"])
        self.assertEqual("ABORTED", recovered["transaction_phase"])
        self.assertFalse(can_crop_v7(self.feedback, self.identity, prepared["annotation_id"], self.boundary))

        annotation_event = build_annotation_event(
            "sha256:source", "run-1", self.algorithm, self.boundary, [0, 1, 2, 3], 0,
        )
        self.annotations.append_idempotent(annotation_event)
        prepared = transaction.prepare(self.algorithm, self.boundary, [0, 1, 2, 3], run_id="run-1", annotation_id=annotation_event["annotation_id"])
        recovered = transaction.recover(annotation_event["annotation_id"])
        self.assertIn(recovered["transaction_phase"], {"ANNOTATION_COMMITTED", "FEEDBACK_COMMITTED"})

    def test_conflicting_source_or_corners_blocks_recovery(self):
        transaction = V7ConfirmationTransaction(self.feedback, self.annotations, self.identity)
        prepared = transaction.prepare(self.algorithm, self.boundary, [0, 1, 2, 3], run_id="run-1")
        annotation = build_annotation_event(
            "sha256:other", "run-1", self.algorithm, self.boundary, [0, 1, 2, 3], 0,
        )
        self.annotations.append_idempotent(annotation)
        with self.assertRaises(ConfirmationConflictError):
            transaction.recover(prepared["annotation_id"])


if __name__ == "__main__":
    unittest.main()
