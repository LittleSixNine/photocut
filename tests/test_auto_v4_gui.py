import tempfile
import unittest
from pathlib import Path

from photocut.data.annotation_store import AnnotationStore
from photocut.confirmation.identity import FinalizedDetectionEvidence
from photocut.cli import (
    AutoV4ConfirmationViewModel,
    _v7_view_model_for_entry,
    commit_confirmation,
)


PRIMARY = [[10, 10], [90, 10], [90, 90], [10, 90]]
ALTERNATE = [[12, 10], [92, 10], [92, 90], [12, 90]]
LEGACY = [[11, 10], [91, 10], [91, 90], [11, 90]]


def _entry(**overrides):
    value = {
        "filename": "scan.jpg",
        "image_id": "sha256:image",
        "run_id": "run-1",
        "algorithm_version": "8.2",
        "algorithm_boundary_corners": PRIMARY,
        "boundary_corners": PRIMARY,
        "corners": PRIMARY,
        "detector_requested": "auto",
        "detector_used": "manual_review",
        "auto_cascade_version": "auto-v4",
        "auto_v4_status": "manual_review",
        "confirmation_primary_candidate_id": "v8:draft",
        "confirmation_selected_candidate_id": "v8:draft",
        "confirmation_selected_algorithm_version": "8.2",
        "candidate_audit": [
            {
                "candidate_id": "edge-primary",
                "stage_ranks": {"selected": 1},
                "adopted_refined_corners": PRIMARY,
                "sources": ["background:min_area_rect"],
            },
            {
                "candidate_id": "edge-alternate",
                "stage_ranks": {"selected": 2},
                "adopted_refined_corners": ALTERNATE,
                "sources": ["edge:hough"],
            },
        ],
        "v52_corners": LEGACY,
    }
    value.update(overrides)
    return value


class AutoV4GuiTests(unittest.TestCase):
    def test_factory_uses_auto_v4_model_and_deduplicates_primary_geometry(self):
        model = _v7_view_model_for_entry(_entry(), (100, 100))

        self.assertIsInstance(model, AutoV4ConfirmationViewModel)
        self.assertEqual("v8:draft", model.selection)
        self.assertEqual(2, len(model._candidates))
        self.assertEqual("v7:edge-alternate", model.toggle_alternate())
        self.assertEqual(ALTERNATE, model.algorithm_corners)

    def test_automatic_v8_exposes_only_its_primary_candidate(self):
        model = AutoV4ConfirmationViewModel(
            _entry(auto_v4_status="automatic"), image_size=(100, 100)
        )

        self.assertEqual(1, len(model._candidates))
        self.assertFalse(model.can_toggle_alternate)
        self.assertTrue(model.can_toggle_v52)
        self.assertEqual("v8:draft", model.toggle_alternate())

    def test_b_toggles_legacy_and_c_from_legacy_advances_ranked_candidate(self):
        entry = _entry()
        model = AutoV4ConfirmationViewModel(entry, image_size=(100, 100))

        self.assertEqual("v52:gui", model.toggle_v52())
        self.assertEqual("5.2", entry["confirmation_selected_algorithm_version"])
        self.assertEqual("v8:draft", model.toggle_v52())
        model.toggle_v52()
        self.assertEqual("v7:edge-alternate", model.toggle_alternate())
        self.assertEqual("7.1", entry["confirmation_selected_algorithm_version"])

    def test_drag_blocks_switch_until_reset_and_reset_returns_primary(self):
        model = AutoV4ConfirmationViewModel(_entry(), image_size=(100, 100))
        model.toggle_alternate()
        model.select_corner(0)
        model.move_selected(1, 0)

        with self.assertRaisesRegex(ValueError, "reset"):
            model.toggle_v52()
        model.reset()
        self.assertEqual("v8:draft", model.selection)
        self.assertEqual(PRIMARY, model.algorithm_corners)

    def test_no_legal_primary_cannot_be_confirmed_but_can_be_skipped(self):
        model = AutoV4ConfirmationViewModel(
            _entry(
                algorithm_boundary_corners=[],
                corners=[],
                candidate_audit=[],
                v52_corners=None,
            ),
            image_size=(100, 100),
        )

        self.assertFalse(model.can_confirm)
        with self.assertRaises(ValueError):
            model.confirm()
        self.assertEqual("skipped", model.skip("no_candidate"))

    def test_annotation_records_primary_and_selected_candidate_identity(self):
        entry = _entry()
        model = AutoV4ConfirmationViewModel(entry, image_size=(100, 100))
        model.toggle_alternate()
        evidence = FinalizedDetectionEvidence(
            tuple(map(tuple, PRIMARY)), "8.2", "det-1", "auto", "manual_review"
        )
        with tempfile.TemporaryDirectory() as directory:
            store = AnnotationStore(Path(directory) / "annotations.jsonl")
            event = commit_confirmation(
                entry,
                model.state,
                store,
                1,
                finalized_evidence=evidence,
            )

        self.assertEqual("v8:draft", event["confirmation_primary_candidate_id"])
        self.assertEqual("v7:edge-alternate", event["selected_candidate_id"])
        self.assertEqual("7.1", event["selected_candidate_algorithm_version"])
        self.assertEqual("7.1", event["algorithm_version"])
        self.assertEqual(ALTERNATE, event["algorithm_boundary_corners"])


if __name__ == "__main__":
    unittest.main()
