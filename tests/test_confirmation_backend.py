import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from photocut.data.annotation_store import AnnotationStore
from photocut.confirmation import backend as confirmation_backend
from photocut.confirmation.backend import (
    PhotoCutConfirmationBackend,
    commit_confirmation,
)
from photocut.confirmation.controller import ConfirmationAction, ConfirmationSessionController
from photocut.confirmation.web.media import ConfirmationMediaService
from photocut.confirmation.version import OPENCV_GUI_VERSION


def _entry(**overrides):
    value = {
        "filename": "scan.jpg",
        "image_id": "sha256:image",
        "run_id": "run-1",
        "detector_used": "v5.2",
        "algorithm_boundary_corners": [[10, 10], [90, 10], [90, 70], [10, 70]],
        "boundary_corners": [[10, 10], [90, 10], [90, 70], [10, 70]],
        "corners": [[10, 10], [90, 10], [90, 70], [10, 70]],
    }
    value.update(overrides)
    return value


class PhotoCutConfirmationBackendTests(unittest.TestCase):
    def backend(self, root, entry, **overrides):
        values = {
            "corners_info": [entry],
            "entries": [entry],
            "input_dir": root,
            "output_dir": root,
            "json_path": root / "corners_info.json",
            "annotation_store": Mock(),
            "project_root": root,
            "image_loader": lambda entry, source, output: np.zeros(
                (80, 100, 3), dtype=np.uint8
            ),
            "evidence_loader": lambda entry, output, project: None,
            "commit_entry": Mock(return_value={"annotation_id": "ann-1"}),
            "save_view": Mock(),
            "clock": lambda: 1.0,
        }
        values.update(overrides)
        return PhotoCutConfirmationBackend(**values)

    def test_commit_appends_annotation_before_mutable_view(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scan.jpg").write_bytes(b"fixture")
            calls = []
            entry = _entry(confirmed=False)
            backend = self.backend(
                root,
                entry,
                commit_entry=lambda *args, **kwargs: calls.append("append")
                or {"annotation_id": "ann-1"},
                save_view=lambda path, rows: calls.append("view"),
            )

            item = backend.load(0)
            backend.commit(item, 12, "2.0")

            self.assertEqual(["append", "view"], calls)

    def test_revision_checkpoint_does_not_overwrite_formal_head(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scan.jpg").write_bytes(b"fixture")
            entry = _entry(confirmed=True, annotation_id="ann-existing")
            store = Mock()
            store.events.return_value = [
                {"annotation_id": "ann-existing", "image_id": "sha256:image"}
            ]
            backend = self.backend(
                root,
                entry,
                annotation_store=store,
                commit_entry=commit_confirmation,
            )

            item = backend.load(0)
            item.editor.select_corner(0)
            item.editor.move_selected(5, 0)
            backend.checkpoint(item)

            self.assertEqual("ann-existing", item.entry["annotation_id"])
            self.assertTrue(item.entry["confirmed"])

    def test_preview_does_not_read_annotation_or_change_current_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("scan.jpg", "next.jpg"):
                (root / name).write_bytes(b"fixture")
            entries = [_entry(), _entry(filename="next.jpg", image_id="sha256:next")]
            store = Mock()
            evidence = Mock()
            backend = PhotoCutConfirmationBackend(
                corners_info=entries,
                entries=entries,
                input_dir=root,
                output_dir=root,
                json_path=root / "corners_info.json",
                annotation_store=store,
                project_root=root,
                image_loader=lambda entry, source, output: np.zeros(
                    (80, 100, 3), dtype=np.uint8
                ),
                evidence_loader=evidence,
                commit_entry=Mock(),
                save_view=Mock(),
                clock=lambda: 1.0,
            )

            image, token = backend.load_preview(1)

            self.assertEqual((80, 100, 3), image.shape)
            self.assertIsInstance(token, str)
            self.assertNotEqual("", token)
            evidence.assert_not_called()
            store.events.assert_not_called()
            self.assertNotIn("current_item", backend.__dict__)

    def test_preview_and_loaded_entry_share_stable_opaque_session_token(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("scan.jpg", "next.jpg"):
                (root / name).write_bytes(b"fixture")
            entries = [_entry(), _entry(filename="next.jpg", image_id="sha256:next")]
            backend = PhotoCutConfirmationBackend(
                corners_info=entries,
                entries=entries,
                input_dir=root,
                output_dir=root,
                json_path=root / "corners_info.json",
                annotation_store=Mock(),
                project_root=root,
                image_loader=lambda entry, source, output: np.zeros(
                    (80, 100, 3), dtype=np.uint8
                ),
                evidence_loader=lambda entry, output, project: None,
                commit_entry=Mock(),
                save_view=Mock(),
                clock=lambda: 1.0,
            )
            session = ConfirmationSessionController(backend, gui_version="2.0")
            media = ConfirmationMediaService(session)

            prefetched = media.prefetch_next(1200, 800, 1.0)
            encode_count = media.preview_encode_count
            session.dispatch(ConfirmationAction("next", 0, "next", {}))
            media.sync_navigation()
            current_token = session.snapshot()["image"]["token"]
            promoted = media.preview(current_token, 1200, 800, 1.0)

            self.assertEqual(prefetched.image_token, current_token)
            self.assertEqual(prefetched, promoted)
            self.assertEqual(encode_count, media.preview_encode_count)
            self.assertNotEqual(session.current_item.image_token, backend._image_token(0))
            for token in (backend._image_token(0), backend._image_token(1)):
                self.assertNotIn("scan", token)
                self.assertNotIn("next", token)
                self.assertNotIn("sha256", token)

    def test_load_rejects_missing_or_undecodable_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = _entry()
            backend = self.backend(root, entry)
            with self.assertRaisesRegex(ValueError, "source image"):
                backend.load(0)

            (root / "scan.jpg").write_bytes(b"fixture")
            backend = self.backend(root, entry, image_loader=lambda *args: None)
            with self.assertRaisesRegex(ValueError, "load image"):
                backend.load(0)

    def test_migrated_image_loader_calls_local_archived_helper_directly(self):
        archived = np.zeros((80, 100, 3), dtype=np.uint8)
        entry = _entry(
            detector_requested="auto",
            source_sha256="a" * 64,
            normalized_orientation="exif_1",
            normalized_size=[100, 80],
        )
        with patch(
            "photocut.confirmation.backend._load_archived_v7_image", return_value=archived
        ) as loader, patch(
            "photocut.cli._load_archived_v7_image",
            side_effect=AssertionError("must not route through CLI re-export"),
        ):
            result = confirmation_backend._load_entry_image(
                entry, Path("unused.jpg"), output_dir=Path("output")
            )

        self.assertIs(result, archived)
        loader.assert_called_once_with(entry, Path("output"))

    def test_constructor_resolves_replaceable_helpers_at_instance_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scan.jpg").write_bytes(b"fixture")
            entry = _entry(confirmed=False)
            image_loader = Mock(return_value=np.zeros((80, 100, 3), dtype=np.uint8))
            evidence_loader = Mock(return_value=None)
            commit_entry = Mock(return_value={"annotation_id": "late"})
            with patch("photocut.confirmation.backend._load_entry_image", image_loader), patch(
                "photocut.confirmation.backend._load_finalized_detection_evidence",
                evidence_loader,
            ), patch("photocut.confirmation.backend.commit_confirmation", commit_entry):
                backend = PhotoCutConfirmationBackend(
                    corners_info=[entry],
                    entries=[entry],
                    input_dir=root,
                    output_dir=root,
                    json_path=root / "corners_info.json",
                    annotation_store=Mock(),
                    project_root=root,
                    save_view=Mock(),
                )
                item = backend.load(0)
                backend.commit(item, 1, "2.0")

            image_loader.assert_called_once()
            evidence_loader.assert_called_once()
            commit_entry.assert_called_once()

    def test_formal_append_survives_mutable_view_save_failure_and_action_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scan.jpg").write_bytes(b"fixture")
            entry = _entry(confirmed=False)
            store = AnnotationStore(root / "annotations.jsonl")
            backend = self.backend(
                root,
                entry,
                annotation_store=store,
                commit_entry=commit_confirmation,
                save_view=Mock(side_effect=OSError("view unavailable")),
            )
            session = ConfirmationSessionController(
                backend, gui_version=OPENCV_GUI_VERSION
            )
            action = ConfirmationAction("confirm-once", 0, "confirm", {})

            first = session.dispatch(action)
            replay = session.dispatch(action)

            self.assertEqual(first, replay)
            self.assertEqual("completed", first["session"]["status"])
            self.assertEqual(1, len(store.events()))
            event = store.events()[0]
            self.assertTrue(entry["confirmed"])
            self.assertEqual(event["annotation_id"], entry["annotation_id"])
            backend.save_view.assert_called_once()
            self.assertEqual(
                event["annotation_id"], store.latest_by_image()[entry["image_id"]]["annotation_id"]
            )
            stale_view = _entry(confirmed=False)
            confirmation_backend.reconcile_confirmation_events([stale_view], store)
            self.assertTrue(stale_view["confirmed"])
            self.assertEqual(event["annotation_id"], stale_view["annotation_id"])

    def test_shared_session_preserves_candidate_move_skip_and_annotation_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scan.jpg").write_bytes(b"fixture")

            def session_for(entry, store=None):
                backend = self.backend(
                    root,
                    entry,
                    annotation_store=store or Mock(),
                    commit_entry=commit_confirmation,
                )
                return ConfirmationSessionController(
                    backend, gui_version=OPENCV_GUI_VERSION
                )

            auto_entry = _entry(
                confirmed=False,
                detector_requested="auto",
                detector_used="manual_review",
                auto_cascade_version="auto-v4",
                auto_v4_status="manual_review",
                algorithm_version="8.2",
                confirmation_primary_candidate_id="v8:primary",
                confirmation_selected_candidate_id="v8:primary",
                confirmation_selected_algorithm_version="8.2",
                candidate_audit=[{
                    "candidate_id": "alternate",
                    "stage_ranks": {"selected": 2},
                    "adopted_refined_corners": [
                        [12, 10], [92, 10], [92, 70], [12, 70]
                    ],
                    "sources": ["edge:hough"],
                }],
                v52_corners=[[11, 10], [91, 10], [91, 70], [11, 70]],
            )

            candidate_session = session_for(dict(auto_entry))
            candidate = candidate_session.dispatch(
                ConfirmationAction("candidate", 0, "candidate", {})
            )
            self.assertEqual("v7:alternate", candidate["editor"]["candidate"]["id"])
            self.assertEqual("7.1", candidate["editor"]["candidate"]["algorithm_version"])
            self.assertEqual([12, 10], candidate["editor"]["corners"][0])

            v52_session = session_for(dict(auto_entry))
            v52 = v52_session.dispatch(ConfirmationAction("v52", 0, "v52", {}))
            self.assertEqual("v52:gui", v52["editor"]["candidate"]["id"])
            self.assertEqual("5.2", v52["editor"]["candidate"]["algorithm_version"])
            self.assertEqual([11, 10], v52["editor"]["corners"][0])

            move_session = session_for(dict(auto_entry))
            move_session.dispatch(
                ConfirmationAction("select", 0, "select_corner", {"index": 0})
            )
            moved = move_session.dispatch(
                ConfirmationAction("move", 1, "move", {"dx": 50, "dy": 0})
            )
            self.assertEqual([60, 10], moved["editor"]["corners"][0])
            reset = move_session.dispatch(
                ConfirmationAction("reset", 2, "reset", {})
            )
            self.assertEqual([10, 10], reset["editor"]["corners"][0])
            self.assertEqual("v8:primary", reset["editor"]["candidate"]["id"])

            skipped_entry = dict(auto_entry)
            skip_session = session_for(skipped_entry)
            skip_session.dispatch(
                ConfirmationAction("skip", 0, "skip", {"reason": "user_skip"})
            )
            self.assertEqual("skipped", skipped_entry["v7_operation"])
            self.assertEqual("user_skip", skipped_entry["v7_skip_reason"])

            confirmed_entry = dict(auto_entry)
            store = AnnotationStore(root / "annotations.jsonl")
            confirm_session = session_for(confirmed_entry, store)
            confirm_session.dispatch(
                ConfirmationAction("confirm", 0, "confirm", {})
            )
            event = store.events()[0]
            self.assertEqual(OPENCV_GUI_VERSION, event["gui_version"])
            self.assertEqual("v8:primary", event["selected_candidate_id"])
            self.assertEqual("8.2", event["selected_candidate_algorithm_version"])


if __name__ == "__main__":
    unittest.main()
