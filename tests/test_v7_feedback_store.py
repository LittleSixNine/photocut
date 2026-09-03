import json
import tempfile
import unittest
from pathlib import Path

from photocut.algorithms.v7.feedback_store import (
    FeedbackStore,
    build_prediction_event,
    deterministic_event_id,
)
from photocut.algorithms.v7.parameters import V7Parameters
from photocut.algorithms.v7.types import DetectionIdentity, DetectionResult, DetectionStatus


class FeedbackStoreTests(unittest.TestCase):
    def result(self):
        params = V7Parameters()
        identity = DetectionIdentity("det-1", "sha256:source", "exif_1", "7.0", params.sha256(), "safe")
        return DetectionResult(
            identity=identity,
            status=DetectionStatus.V7_RECOMMENDED,
            corners=((1.25, 2.5), (19.0, 2.5), (19.0, 17.0), (1.25, 17.0)),
            alternate_corners=((2.0, 3.0), (18.0, 3.0), (18.0, 16.0), (2.0, 16.0)),
            overall_confidence=.91,
            edge_confidences=(.9, .91, .92, .93),
            corner_confidences=(.9, .91, .92, .93),
            top1_sources=("contour", "lines"),
            alternate_sources=("background",),
            risks=("weak_edge",),
            timings_ms={"total": 12.5},
        )

    def test_prediction_event_contains_identity_and_audit_fields(self):
        event = build_prediction_event(self.result(), displayed_corners=((1, 3), (19, 3), (19, 17), (1, 17)))
        self.assertEqual(1, event["schema_version"])
        self.assertEqual("prediction", event["event_type"])
        self.assertEqual("det-1", event["detection_id"])
        self.assertEqual("exif_1", event["orientation_transform"])
        self.assertEqual("7.0", event["algorithm_version"])
        self.assertEqual("safe", event["mode"])
        self.assertEqual("v7_recommended", event["status"])
        self.assertEqual([[1, 3], [19, 3], [19, 17], [1, 17]], event["displayed_corners"])
        self.assertEqual([], event["adjusted_corner_indices"])
        self.assertTrue(event["event_id"])

    def test_retry_is_idempotent_and_conflicting_duplicate_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            store = FeedbackStore(Path(directory) / "v7_feedback.jsonl")
            event = build_prediction_event(self.result())
            first = store.append_idempotent(event)
            second = store.append_idempotent(dict(event))
            self.assertEqual(first, second)
            self.assertEqual(1, len(store.events()))
            conflict = dict(event)
            conflict["status"] = "v7_low_confidence"
            with self.assertRaises(ValueError):
                store.append_idempotent(conflict)

    def test_event_id_is_stable_for_same_coordinate_snapshot(self):
        event = build_prediction_event(self.result())
        self.assertEqual(event["event_id"], deterministic_event_id(event))
        changed = dict(event)
        changed["displayed_corners"] = [[0, 0], [1, 0], [1, 1], [0, 1]]
        self.assertNotEqual(event["event_id"], deterministic_event_id(changed))

    def test_store_is_append_only_jsonl_and_reopens(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "v7_feedback.jsonl"
            store = FeedbackStore(path)
            store.append_idempotent(build_prediction_event(self.result()))
            self.assertTrue(path.exists())
            self.assertTrue(path.read_text(encoding="utf-8").endswith("\n"))
            self.assertEqual(1, len(FeedbackStore(path).events()))


if __name__ == "__main__":
    unittest.main()
