import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from photocut.data.annotation_store import AnnotationStore
from photocut.confirmation.model import ConfirmationState
from photocut.confirmation.identity import FinalizedDetectionEvidence
from photocut.confirmation.version import OPENCV_GUI_VERSION
from photocut import load_corners_info, reconcile_confirmation_events, save_corners_info
from photocut.cli import (
    commit_confirmation,
    confirmation_persistence_error_message,
    _gui_identity_entry,
    save_confirmation_draft,
    select_confirmation_entries,
)
from photocut.confirmation import backend as confirmation_backend

class ConfirmIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.corners = [[10, 10], [90, 10], [90, 70], [10, 70]]
        self.entry = {
            "filename": "image.jpg",
            "image_id": "sha256:image",
            "run_id": "run_batch",
            "algorithm_boundary_corners": [point[:] for point in self.corners],
            "boundary_corners": [point[:] for point in self.corners],
            "preview_size": [100, 80],
            "confirmed": False,
        }

    def tearDown(self):
        self.temp_dir.cleanup()

    def state(self):
        return ConfirmationState(
            algorithm_corners=[point[:] for point in self.corners],
            work_corners=[point[:] for point in self.corners],
            image_size=(100, 80),
        )

    def test_commit_persists_accepted_event_before_mutating_entry(self):
        store = AnnotationStore(self.root / "annotations.jsonl")

        event = commit_confirmation(self.entry, self.state(), store, duration_ms=1500)

        self.assertTrue(self.entry["confirmed"])
        self.assertEqual(event["boundary_corners"], self.entry["boundary_corners"])
        self.assertEqual(event["annotation_id"], self.entry["annotation_id"])
        self.assertEqual("accepted", event["confirmation"])
        self.assertEqual(2, event["schema_version"])
        self.assertEqual(OPENCV_GUI_VERSION, event["gui_version"])
        self.assertEqual(1, len(store.events()))

    def test_cli_reexports_backend_persistence_helpers_without_duplicate_implementations(self):
        from photocut import cli as photocut_cli
        helper_names = (
            "save_manual_boundary",
            "reset_confirmation_boundary",
            "save_confirmation_draft",
            "commit_confirmation",
            "_gui_identity_entry",
            "confirmation_persistence_error_message",
            "select_confirmation_entries",
            "_annotation_store_from_reference",
            "_load_finalized_detection_evidence",
            "_load_archived_v7_image",
            "_entry_uses_normalized_snapshot",
            "_load_entry_image",
        )
        for name in helper_names:
            with self.subTest(name=name):
                self.assertIs(
                    getattr(photocut_cli, name), getattr(confirmation_backend, name)
                )

    def test_commit_uses_only_finalized_evidence_for_algorithm_identity(self):
        store = AnnotationStore(self.root / "annotations.jsonl")
        evidence = FinalizedDetectionEvidence(
            tuple(map(tuple, self.corners)), "7.0", "det-1", "auto", "v7"
        )

        event = commit_confirmation(
            dict(self.entry, algorithm_version="5.2"),
            self.state(),
            store,
            duration_ms=1,
            finalized_evidence=evidence,
        )

        self.assertEqual("7.0", event["algorithm_version"])
        self.assertEqual("det-1", event["detection_id"])
        self.assertEqual("auto", event["detector_requested"])
        self.assertEqual("v7", event["detector_used"])

    def test_commit_accepts_detection_evidence_alias(self):
        store = AnnotationStore(self.root / "annotations.jsonl")
        evidence = FinalizedDetectionEvidence(
            tuple(map(tuple, self.corners)), "7.0", "det-1", "auto", "v7"
        )
        event = commit_confirmation(
            self.entry,
            self.state(),
            store,
            duration_ms=1,
            detection_evidence=evidence,
        )
        self.assertEqual("7.0", event["algorithm_version"])

    def test_commit_accepts_explicit_gui_version(self):
        store = AnnotationStore(self.root / "annotations.jsonl")
        event = commit_confirmation(
            self.entry,
            self.state(),
            store,
            duration_ms=1,
            detection_evidence=None,
            gui_version="1.0.1",
        )
        self.assertEqual("1.0.1", event["gui_version"])

    def test_invalid_gui_version_rejects_before_append_and_mutation(self):
        store = AnnotationStore(self.root / "annotations.jsonl")
        with self.assertRaisesRegex(ValueError, "gui_version"):
            commit_confirmation(
                self.entry,
                self.state(),
                store,
                duration_ms=1,
                detection_evidence=None,
                gui_version=" 1.0",
            )
        self.assertFalse(self.entry["confirmed"])
        self.assertFalse(store.path.exists())

    def test_commit_rejects_conflicting_evidence_aliases(self):
        store = AnnotationStore(self.root / "annotations.jsonl")
        evidence = FinalizedDetectionEvidence(
            tuple(map(tuple, self.corners)), "7.0", "det-1", "auto", "v7"
        )
        with self.assertRaisesRegex(ValueError, "only one"):
            commit_confirmation(
                self.entry,
                self.state(),
                store,
                duration_ms=1,
                finalized_evidence=evidence,
                detection_evidence=evidence,
            )

    def test_commit_binds_event_baseline_to_finalized_evidence(self):
        store = AnnotationStore(self.root / "annotations.jsonl")
        evidence = FinalizedDetectionEvidence(
            ((11, 10), (90, 10), (90, 70), (10, 70)),
            "7.0", "det-1", "auto", "v7",
        )

        event = commit_confirmation(
            self.entry,
            self.state(),
            store,
            duration_ms=1,
            finalized_evidence=evidence,
        )
        self.assertEqual(evidence.algorithm_boundary_corners,
                         tuple(map(tuple, event["algorithm_boundary_corners"])))
        self.assertEqual("7.0", event["algorithm_version"])

    def test_revision_without_finalized_evidence_keeps_formal_algorithm_baseline(self):
        store = AnnotationStore(self.root / "annotations.jsonl")
        first = commit_confirmation(self.entry, self.state(), store, duration_ms=1)
        previous = store.events()[0]
        state = self.state()
        state.algorithm_corners = [[8, 8], [92, 8], [92, 72], [8, 72]]
        event = commit_confirmation(
            dict(self.entry, annotation_id=first["annotation_id"]),
            state,
            store,
            duration_ms=1,
            previous_annotation=previous,
        )
        self.assertEqual(previous["algorithm_boundary_corners"], event["algorithm_boundary_corners"])

    def test_commit_omits_unverified_algorithm_identity(self):
        store = AnnotationStore(self.root / "annotations.jsonl")
        event = commit_confirmation(
            dict(self.entry, algorithm_version="5.2"),
            self.state(),
            store,
            duration_ms=1,
            finalized_evidence=None,
        )

        self.assertEqual(2, event["schema_version"])
        self.assertEqual(OPENCV_GUI_VERSION, event["gui_version"])
        self.assertNotIn("algorithm_version", event)
        self.assertNotIn("detector_used", event)

    def test_sidebar_identity_does_not_trust_mutable_algorithm_version(self):
        identity = _gui_identity_entry(
            {
                "algorithm_version": "5.2",
                "detector_requested": "auto",
                "detector_used": "v5.2",
            },
            None,
        )

        self.assertIsNone(identity["algorithm_version"])
        self.assertIsNone(identity["detector_requested"])
        self.assertIsNone(identity["detector_used"])

    def test_failed_event_append_leaves_entry_unconfirmed(self):
        store = AnnotationStore(self.root / "annotations.jsonl")

        with patch("photocut.data.annotation_store.append_jsonl_validated", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                commit_confirmation(self.entry, self.state(), store, duration_ms=1500)

        self.assertFalse(self.entry["confirmed"])
        self.assertNotIn("annotation_id", self.entry)
        self.assertFalse((self.root / "annotations.jsonl").exists())

    def test_adjusted_confirmation_and_revision_supersede_previous_annotation(self):
        store = AnnotationStore(self.root / "annotations.jsonl")
        first = commit_confirmation(self.entry, self.state(), store, duration_ms=1)
        self.entry["confirmed"] = False
        state = self.state()
        state.selected = 1
        state.move_selected(2, 0)

        revision = commit_confirmation(self.entry, state, store, duration_ms=2)

        self.assertEqual("adjusted", revision["confirmation"])
        self.assertEqual([1], revision["adjusted_corner_indices"])
        self.assertEqual(first["annotation_id"], revision["supersedes_annotation_id"])
        self.assertEqual(revision["annotation_id"], store.latest_by_image()["sha256:image"]["annotation_id"])

    def test_default_selection_skips_confirmed_entries(self):
        entries = [
            {"filename": "approved.jpg", "confirmed": True, "annotation_id": "ann_1", "image_id": "sha256:approved"},
            {"filename": "new.jpg", "confirmed": False, "image_id": "sha256:new"},
        ]

        selected = select_confirmation_entries(entries, revise_filenames=())

        self.assertEqual(["new.jpg"], [entry["filename"] for entry in selected])

    def test_explicit_revision_selects_only_named_confirmed_entries(self):
        entries = [
            {"filename": "approved.jpg", "confirmed": True, "annotation_id": "ann_1", "image_id": "sha256:approved"},
            {"filename": "other-approved.jpg", "confirmed": True, "annotation_id": "ann_2", "image_id": "sha256:other"},
            {"filename": "new.jpg", "confirmed": False, "image_id": "sha256:new"},
        ]

        selected = select_confirmation_entries(entries, revise_filenames=("approved.jpg",))

        self.assertEqual(["approved.jpg", "new.jpg"], [entry["filename"] for entry in selected])

    def test_explicit_revision_skips_annotation_owned_by_another_image(self):
        store = AnnotationStore(self.root / "annotations.jsonl")
        event = commit_confirmation(self.entry, self.state(), store, duration_ms=1)
        cross_image_entry = dict(self.entry, image_id="sha256:other")

        with patch("builtins.print") as printed:
            selected = select_confirmation_entries(
                [cross_image_entry],
                revise_filenames=("image.jpg",),
                annotation_store=store,
            )

        self.assertEqual([], selected)
        self.assertIn("属于另一张图片", printed.call_args[0][0])
        self.assertEqual(event["annotation_id"], cross_image_entry["annotation_id"])

    def test_explicit_revision_skips_stale_annotation_head(self):
        store = AnnotationStore(self.root / "annotations.jsonl")
        first = commit_confirmation(self.entry, self.state(), store, duration_ms=1)
        current = dict(self.entry)
        commit_confirmation(current, self.state(), store, duration_ms=2)
        stale_entry = dict(self.entry, annotation_id=first["annotation_id"])

        with patch("builtins.print") as printed:
            selected = select_confirmation_entries(
                [stale_entry],
                revise_filenames=("image.jpg",),
                annotation_store=store,
            )

        self.assertEqual([], selected)
        self.assertIn("不是当前正式 head", printed.call_args[0][0])

    def test_confirmation_storage_conflict_message_is_recoverable(self):
        message = confirmation_persistence_error_message(
            "image.jpg",
            ValueError("revision must supersede the current active head without branching"),
        )

        self.assertIn("未提交", message)
        self.assertIn("存储冲突", message)
        self.assertIn("重新打开确认模式", message)

    def test_explicit_revision_requires_known_formal_annotation_and_source_identity(self):
        entries = [{"filename": "legacy.jpg", "confirmed": True}]

        with self.assertRaisesRegex(ValueError, "no formal annotation"):
            select_confirmation_entries(entries, revise_filenames=("legacy.jpg",))
        with self.assertRaisesRegex(ValueError, "unknown revision filenames"):
            select_confirmation_entries(entries, revise_filenames=("missing.jpg",))
        with self.assertRaisesRegex(ValueError, "source image identity"):
            select_confirmation_entries(
                [{"filename": "missing-id.jpg", "confirmed": True, "annotation_id": "ann_1"}],
                revise_filenames=("missing-id.jpg",),
            )

    def test_retrying_same_revision_is_idempotent_and_keeps_history(self):
        store = AnnotationStore(self.root / "annotations.jsonl")
        first = commit_confirmation(self.entry, self.state(), store, duration_ms=1)
        revision_entry = dict(self.entry, annotation_id=first["annotation_id"])
        state = self.state()
        state.selected = 1
        state.move_selected(2, 0)

        revised = commit_confirmation(revision_entry, state, store, duration_ms=2)
        retried = commit_confirmation(
            dict(self.entry, annotation_id=first["annotation_id"]), state, store, duration_ms=2
        )

        self.assertEqual(first["annotation_id"], revised["supersedes_annotation_id"])
        self.assertEqual(revised["annotation_id"], retried["annotation_id"])
        self.assertEqual(2, len(store.events()))
        self.assertEqual(revised["annotation_id"], store.latest_by_image()["sha256:image"]["annotation_id"])

    def test_draft_navigation_persists_without_creating_annotation_event(self):
        store = AnnotationStore(self.root / "annotations.jsonl")
        state = self.state()
        state.selected = 0
        state.move_selected(3, 4)

        save_confirmation_draft(self.entry, state)
        save_corners_info(str(self.root / "corners_info.json"), [self.entry])

        restored = load_corners_info(str(self.root / "corners_info.json"))[0]
        self.assertFalse(restored["confirmed"])
        self.assertEqual([[13, 14], [90, 10], [90, 70], [10, 70]], restored["boundary_corners"])
        self.assertEqual([0], restored["draft_adjusted_corner_indices"])
        self.assertEqual([], store.events())

    def test_restart_recovers_durable_event_after_mutable_save_failure(self):
        store = AnnotationStore(self.root / "annotations.jsonl")
        event = commit_confirmation(self.entry, self.state(), store, duration_ms=1)
        stale = dict(self.entry, confirmed=False)
        stale.pop("annotation_id")
        stale["boundary_corners"] = [[0, 0]] * 4
        save_corners_info(str(self.root / "corners_info.json"), [stale])

        entries = load_corners_info(str(self.root / "corners_info.json"))
        reconcile_confirmation_events(entries, store)
        save_corners_info(str(self.root / "corners_info.json"), entries)

        recovered = json.loads((self.root / "corners_info.json").read_text())[0]
        self.assertTrue(recovered["confirmed"])
        self.assertEqual(event["annotation_id"], recovered["annotation_id"])
        self.assertEqual(event["boundary_corners"], recovered["boundary_corners"])


if __name__ == "__main__":
    unittest.main()
