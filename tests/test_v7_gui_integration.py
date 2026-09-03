import copy
import tempfile
import unittest
from unittest.mock import Mock, patch
from pathlib import Path

import numpy as np

from photocut.data.annotation_store import AnnotationStore
from photocut.confirmation import backend as confirmation_backend
from photocut import cli as photocut_cli
from photocut.confirmation.backend import PhotoCutConfirmationBackend, commit_confirmation
from photocut.confirmation.controller import ConfirmationAction, ConfirmationSessionController
from photocut.confirmation.version import OPENCV_GUI_VERSION
from photocut.confirmation.pointer import PointerSample


class V7GuiIntegrationTests(unittest.TestCase):
    def entry(self, status="v7_recommended"):
        return {
            "detector": "v7",
            "detection_status": status,
            "algorithm_boundary_corners": [[2, 2], [18, 2], [18, 18], [2, 18]],
            "boundary_corners": [[2, 2], [18, 2], [18, 18], [2, 18]],
            "alternate_corners": [[3, 3], [17, 3], [17, 17], [3, 17]],
            "candidate_sources": ["contour"],
            "alternate_sources": ["lines"],
            "confidences": [.9, .8, .7, .6],
            "risks": ["weak_edge"],
        }

    def auto_entry(self, filename="scan.jpg", image_id="sha256:image"):
        return {
            "filename": filename,
            "image_id": image_id,
            "run_id": "run-1",
            "detector_requested": "auto",
            "detector_used": "manual_review",
            "auto_cascade_version": "auto-v4",
            "auto_v4_status": "manual_review",
            "algorithm_version": "8.2",
            "confirmation_primary_candidate_id": "v8:primary",
            "confirmation_selected_candidate_id": "v8:primary",
            "confirmation_selected_algorithm_version": "8.2",
            "algorithm_boundary_corners": [[10, 10], [90, 10], [90, 70], [10, 70]],
            "boundary_corners": [[10, 10], [90, 10], [90, 70], [10, 70]],
            "corners": [[10, 10], [90, 10], [90, 70], [10, 70]],
            "candidate_audit": [{
                "candidate_id": "alternate",
                "stage_ranks": {"selected": 2},
                "adopted_refined_corners": [[12, 10], [92, 10], [92, 70], [12, 70]],
                "sources": ["edge:hough"],
            }],
            "v52_corners": [[11, 10], [91, 10], [91, 70], [11, 70]],
        }

    def real_session(self, root, entries, *, store=None, save_view=None):
        for entry in entries:
            (root / entry["filename"]).write_bytes(b"fixture")
        store = store or Mock()
        save_view = save_view or Mock()
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
            evidence_loader=lambda entry, output, project: None,
            commit_entry=commit_confirmation,
            save_view=save_view,
            clock=lambda: 1.0,
        )
        return (
            ConfirmationSessionController(backend, gui_version=OPENCV_GUI_VERSION),
            save_view,
        )

    def dispatch_native(self, session, key, key_ascii=None, navigate=0):
        key_ascii = key if key_ascii is None and 0 <= key <= 255 else (key_ascii or 0)
        kind, payload = photocut_cli.opencv_confirmation_action(
            key, key_ascii, navigate=navigate
        )
        return photocut_cli.dispatch_opencv_confirmation_action(
            session, kind, payload
        )

    def test_cli_builds_v7_model_only_for_v7_entries(self):
        self.assertIsNone(photocut_cli._v7_view_model_for_entry({"detector": "v5.2"}, (20, 20)))
        model = photocut_cli._v7_view_model_for_entry(self.entry(), (20, 20))
        self.assertEqual("recommended", model.status_label)
        model.toggle_alternate()
        self.assertEqual("alternate", model.selection)

    def test_gui_identity_wrapper_keeps_stable_window_key(self):
        with patch("photocut.cli.set_native_gui_title", return_value="title") as title:
            rendered = photocut_cli.update_confirmation_gui_identity(
                photocut_cli.cv2, "PhotoCut", {"algorithm_version": "7.0"}
            )
        self.assertEqual("title", rendered[0])
        title.assert_called_once_with(photocut_cli.cv2, "PhotoCut", "7.0")

    def test_gui_identity_wrapper_marks_schema_v1_previous_gui_unknown(self):
        title, lines = photocut_cli.update_confirmation_gui_identity(
            Mock(), "PhotoCut", {"algorithm_version": "7.0"},
            previous_annotation={"schema_version": 1},
        )
        self.assertEqual("PhotoCut GUI 1.0 · Algorithm 7.0", title)
        self.assertIn("Previous GUI: legacy / unknown", lines)

    def test_confirm_resolves_relative_dataset_root_from_project_root(self):
        entry = {"filename": "scan.jpg", "confirmed": False}
        args = type("Args", (), {
            "input": "/tmp/source/input",
            "output": "/tmp/output",
            "revise_confirmed": (),
        })()
        store = object()
        with patch("photocut.confirmation.backend.load_corners_info", return_value=[entry]), patch(
            "photocut.cli.find_images", return_value=[]
        ), patch(
            "photocut.cli.normalize_entry_relative_paths",
            return_value=([(entry, "scan.jpg")], []),
        ), patch(
            "photocut.confirmation.backend._annotation_store_from_reference", return_value=store
        ) as resolve, patch(
            "photocut.confirmation.backend.reconcile_confirmation_events"
        ), patch(
            "photocut.confirmation.backend.select_confirmation_entries", return_value=[]
        ), patch("photocut.confirmation.backend.save_corners_info"):
            photocut_cli.confirm_command(args)

        resolve.assert_called_once_with(
            args.output, Path(photocut_cli.__file__).resolve().parents[1]
        )

    def test_cli_model_blocks_error_and_no_primary_confirmation(self):
        for status in ("error", "no_primary_photo"):
            model = photocut_cli._v7_view_model_for_entry(self.entry(status), (20, 20))
            self.assertFalse(model.can_confirm)
            with self.assertRaises(ValueError):
                model.confirm()

    def test_s_skip_is_explicit_and_records_reason_in_model(self):
        model = photocut_cli._v7_view_model_for_entry(self.entry("v7_low_confidence"), (20, 20))
        self.assertEqual("skipped", model.skip("user_skip"))
        self.assertEqual("user_skip", model.skip_reason)
        self.assertFalse(model.can_confirm)

    def test_confirmation_preview_resizes_full_image_only_once(self):
        image = np.zeros((200, 400, 3), dtype=np.uint8)
        resized = np.zeros((100, 200, 3), dtype=np.uint8)
        with patch("photocut.cli.cv2.resize", return_value=resized) as resize:
            cache = photocut_cli.build_confirmation_preview(
                image, viewport_width=200, viewport_height=120
            )
            for _ in range(5):
                self.assertIs(cache.scaled_image, resized)

        resize.assert_called_once_with(image, (200, 100))
        self.assertEqual((200, 100), (cache.scaled_width, cache.scaled_height))
        self.assertEqual((200, 100), photocut_cli.original_to_display(
            (400, 200), cache.transform
        ))

    def test_preview_failure_note_never_confirms_entry(self):
        entry = {"filename": "huge.jpg", "confirmed": True}
        photocut_cli.record_preview_render_error(entry, MemoryError("preview"))
        self.assertFalse(entry["confirmed"])
        self.assertEqual("preview_render_error", entry["preview_render_error"]["code"])
        self.assertIn("MemoryError", entry["preview_render_error"]["message"])

    def test_draft_tracker_never_persists_on_idle_frames(self):
        tracker = photocut_cli.ConfirmationDraftTracker(
            [[2, 2], [18, 2], [18, 18], [2, 18]]
        )
        for _ in range(100):
            tracker.observe([[2, 2], [18, 2], [18, 18], [2, 18]])
        self.assertFalse(tracker.dirty)

        tracker.observe([[3, 2], [18, 2], [18, 18], [2, 18]])
        self.assertTrue(tracker.dirty)
        tracker.checkpoint()
        self.assertFalse(tracker.dirty)

    def test_gui_refresh_is_capped_to_leave_time_for_window_events(self):
        self.assertGreaterEqual(photocut_cli.GUI_FRAME_DELAY_MS, 16)
        self.assertLessEqual(photocut_cli.GUI_FRAME_DELAY_MS, 50)

    def test_idle_drag_and_arrow_frames_do_not_write_json(self):
        state = photocut_cli.ConfirmationState(
            [[2, 2], [18, 2], [18, 18], [2, 18]],
            [[2, 2], [18, 2], [18, 18], [2, 18]],
            (20, 20),
        )
        state.selected = 0
        tracker = photocut_cli.ConfirmationDraftTracker(state.work_corners)
        with patch("photocut.cli.save_corners_info") as save:
            for _ in range(20):
                photocut_cli.observe_confirmation_frame(tracker, state, None)
            state.move_selected(1, 0)  # mouse/arrow paths share the same state mutation.
            for _ in range(20):
                photocut_cli.observe_confirmation_frame(tracker, state, None)

        self.assertTrue(tracker.dirty)
        save.assert_not_called()

    def test_auto_v52_reopens_verified_normalized_snapshot(self):
        archived = np.zeros((20, 30, 3), dtype=np.uint8)
        entry = {
            "detector_requested": "auto",
            "detector_used": "v5.2",
            "source_sha256": "a" * 64,
            "normalized_orientation": "exif_8",
            "normalized_size": [30, 20],
        }
        with patch(
            "photocut.confirmation.backend._load_archived_v7_image", return_value=archived
        ) as verified, patch("photocut.cli.load_image") as legacy:
            result = photocut_cli._load_entry_image(
                entry, Path("changed.jpg"), output_dir=Path("output")
            )

        self.assertIs(result, archived)
        verified.assert_called_once()
        legacy.assert_not_called()

    def test_native_window_close_dispatches_quit_once_and_missing_probe_is_safe(self):
        class ClosedCv2:
            WND_PROP_VISIBLE = 4

            def getWindowProperty(self, window_name, property_id):
                self.call = (window_name, property_id)
                return 0.0

        session = Mock()
        session.status = "active"
        session.snapshot.return_value = {"revision": 0}

        def stop(action):
            session.status = "quit"
            return {"session": {"status": "quit"}}

        session.dispatch.side_effect = stop
        cv2_module = ClosedCv2()

        self.assertTrue(
            photocut_cli.checkpoint_closed_opencv_window(
                session, cv2_module, "PhotoCut"
            )
        )
        self.assertTrue(
            photocut_cli.checkpoint_closed_opencv_window(
                session, cv2_module, "PhotoCut"
            )
        )
        session.dispatch.assert_called_once()
        action = session.dispatch.call_args.args[0]
        self.assertEqual("quit", action.kind)
        self.assertEqual(("PhotoCut", 4), cv2_module.call)

        open_session = Mock(status="active")
        self.assertFalse(
            photocut_cli.checkpoint_closed_opencv_window(
                open_session, object(), "PhotoCut"
            )
        )
        open_session.dispatch.assert_not_called()

    def test_native_window_probe_error_preserves_normal_quit_path(self):
        class FailingCv2:
            WND_PROP_VISIBLE = 4

            def getWindowProperty(self, window_name, property_id):
                raise RuntimeError("window probe unavailable")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session, save_view = self.real_session(
                root, [self.auto_entry()]
            )

            self.assertFalse(
                photocut_cli.checkpoint_closed_opencv_window(
                    session, FailingCv2(), "PhotoCut"
                )
            )
            self.assertEqual("active", session.status)
            save_view.assert_not_called()

            terminal = self.dispatch_native(session, ord("q"))
            self.assertEqual("quit", terminal["session"]["status"])
            self.assertEqual("quit", session.status)
            save_view.assert_called_once()

    def test_global_capture_moves_corner_outside_and_stops_on_release(self):
        class Reader:
            def __init__(self):
                self.samples = iter((
                    PointerSample(100, 100, True),
                    PointerSample(140, 100, True),
                    PointerSample(600, 100, False),
                    PointerSample(700, 100, True),
                ))

            def sample(self):
                return next(self.samples)

        state = photocut_cli.ConfirmationState(
            [[10, 10], [90, 10], [90, 70], [10, 70]],
            [[10, 10], [90, 10], [90, 70], [10, 70]],
            (100, 80), selected=0,
        )
        mouse_state = {
            "state": state,
            "global_pointer_reader": Reader(),
            "global_magnifier_capture": photocut_cli.MagnifierDragCapture(),
            "global_magnifier_capture_active": False,
            "magnifier_dragging": False,
        }

        self.assertTrue(photocut_cli.begin_magnifier_drag(mouse_state, 100, 100))
        self.assertTrue(photocut_cli.poll_magnifier_drag(mouse_state))
        self.assertEqual([0, 10], state.work_corners[0])
        self.assertTrue(photocut_cli.poll_magnifier_drag(mouse_state))
        self.assertFalse(mouse_state["magnifier_dragging"])
        corner_after_release = state.work_corners[0][:]
        self.assertFalse(photocut_cli.poll_magnifier_drag(mouse_state))
        self.assertEqual(corner_after_release, state.work_corners[0])

    def test_global_capture_reader_error_clears_drag_state(self):
        class BrokenReader:
            def sample(self):
                raise RuntimeError("pointer unavailable")

        state = photocut_cli.ConfirmationState(
            [[10, 10], [90, 10], [90, 70], [10, 70]],
            [[10, 10], [90, 10], [90, 70], [10, 70]],
            (100, 80), selected=0,
        )
        capture = photocut_cli.MagnifierDragCapture()
        mouse_state = {
            "state": state,
            "global_pointer_reader": BrokenReader(),
            "global_magnifier_capture": capture,
            "global_magnifier_capture_active": False,
            "magnifier_dragging": False,
        }
        with patch.object(mouse_state["global_pointer_reader"], "sample", return_value=PointerSample(100, 100, True)):
            self.assertTrue(photocut_cli.begin_magnifier_drag(mouse_state, 100, 100))
        self.assertFalse(photocut_cli.poll_magnifier_drag(mouse_state))
        self.assertFalse(capture.active)
        self.assertFalse(mouse_state["magnifier_dragging"])

    def test_missing_global_reader_falls_back_to_window_local_drag(self):
        mouse_state = {
            "global_pointer_reader": None,
            "global_magnifier_capture": photocut_cli.MagnifierDragCapture(),
            "global_magnifier_capture_active": False,
            "magnifier_dragging": False,
            "magnifier_drag_anchor": (0, 0),
        }
        self.assertFalse(photocut_cli.begin_magnifier_drag(mouse_state, 12, 34))
        self.assertTrue(mouse_state["magnifier_dragging"])
        self.assertEqual((12, 34), mouse_state["magnifier_drag_anchor"])

    def test_global_reader_released_before_start_does_not_fallback_or_stick(self):
        class ReleasedReader:
            def sample(self):
                return PointerSample(12, 34, False)

        mouse_state = {
            "global_pointer_reader": ReleasedReader(),
            "global_magnifier_capture": photocut_cli.MagnifierDragCapture(),
            "global_magnifier_capture_active": False,
            "magnifier_dragging": False,
        }
        self.assertFalse(photocut_cli.begin_magnifier_drag(mouse_state, 12, 34))
        self.assertFalse(mouse_state["magnifier_dragging"])
        self.assertFalse(mouse_state["global_magnifier_capture_active"])

    def test_explicit_gui_checkpoints_cancel_active_capture(self):
        capture = photocut_cli.MagnifierDragCapture()
        capture.begin(PointerSample(10, 10, True))
        mouse_state = {
            "global_magnifier_capture": capture,
            "global_magnifier_capture_active": True,
            "magnifier_dragging": True,
        }
        photocut_cli.cancel_magnifier_drag(mouse_state)
        self.assertFalse(capture.active)
        self.assertFalse(mouse_state["global_magnifier_capture_active"])
        self.assertFalse(mouse_state["magnifier_dragging"])

    def test_repeated_global_poll_does_not_double_apply_one_pointer_position(self):
        class Reader:
            def __init__(self):
                self.samples = iter((
                    PointerSample(100, 100, True),
                    PointerSample(140, 100, True),
                    PointerSample(140, 100, True),
                ))

            def sample(self):
                return next(self.samples)

        state = photocut_cli.ConfirmationState(
            [[10, 10], [90, 10], [90, 70], [10, 70]],
            [[10, 10], [90, 10], [90, 70], [10, 70]],
            (100, 80), selected=0,
        )
        mouse_state = {
            "state": state,
            "global_pointer_reader": Reader(),
            "global_magnifier_capture": photocut_cli.MagnifierDragCapture(),
            "global_magnifier_capture_active": False,
            "magnifier_dragging": False,
        }
        photocut_cli.begin_magnifier_drag(mouse_state, 100, 100)
        photocut_cli.poll_magnifier_drag(mouse_state)
        photocut_cli.poll_magnifier_drag(mouse_state)
        self.assertEqual([0, 10], state.work_corners[0])

    def test_wasd_coarse_key_mapping_is_lowercase_only(self):
        self.assertEqual((0, -50), photocut_cli.coarse_move_delta(ord("w")))
        self.assertEqual((-50, 0), photocut_cli.coarse_move_delta(ord("a")))
        self.assertEqual((0, 50), photocut_cli.coarse_move_delta(ord("s")))
        self.assertEqual((50, 0), photocut_cli.coarse_move_delta(ord("d")))
        for key in "WASD":
            with self.subTest(key=key):
                self.assertIsNone(photocut_cli.coarse_move_delta(ord(key)))

    def test_v7_candidate_key_mapping_preserves_existing_controls(self):
        self.assertEqual("alternate", photocut_cli.v7_candidate_key_action(ord("C")))
        self.assertEqual("v5.2", photocut_cli.v7_candidate_key_action(ord("B")))
        self.assertEqual("v5.2", photocut_cli.v7_candidate_key_action(ord("b")))
        self.assertEqual("skip", photocut_cli.v7_candidate_key_action(ord("X")))
        self.assertIsNone(photocut_cli.v7_candidate_key_action(ord("A")))
        self.assertIsNone(photocut_cli.v7_candidate_key_action(ord("S")))
        self.assertIsNone(photocut_cli.v7_candidate_key_action(ord("a")))
        self.assertIsNone(photocut_cli.v7_candidate_key_action(ord("s")))

    def test_opencv_key_action_table_routes_existing_controls_to_session_actions(self):
        cases = (
            (ord("C"), ord("C"), 0, ("candidate", {})),
            (ord("B"), ord("B"), 0, ("v52", {})),
            (ord("b"), ord("b"), 0, ("v52", {})),
            (ord("R"), ord("R"), 0, ("reset", {})),
            (ord("X"), ord("X"), 0, ("skip", {"reason": "user_skip"})),
            (ord("w"), ord("w"), 0, ("move", {"dx": 0, "dy": -50})),
            (ord("a"), ord("a"), 0, ("move", {"dx": -50, "dy": 0})),
            (ord("s"), ord("s"), 0, ("move", {"dx": 0, "dy": 50})),
            (ord("d"), ord("d"), 0, ("move", {"dx": 50, "dy": 0})),
            (65361, 0, 0, ("move", {"dx": -5, "dy": 0})),
            (65363, 0, 0, ("move", {"dx": 5, "dy": 0})),
            (65362, 0, 0, ("move", {"dx": 0, "dy": -5})),
            (65364, 0, 0, ("move", {"dx": 0, "dy": 5})),
            (32, 32, 0, ("confirm", {})),
            (0, 0, -1, ("previous", {})),
            (0, 0, 1, ("next", {})),
            (ord("p"), ord("p"), 0, ("pause", {})),
            (ord("q"), ord("q"), 0, ("quit", {})),
        )
        for key, key_ascii, navigate, expected in cases:
            with self.subTest(expected=expected):
                action = photocut_cli.opencv_confirmation_action(
                    key, key_ascii, navigate=navigate
                )
                self.assertEqual(expected, action)

    def test_opencv_dispatch_constructs_revisioned_confirmation_action(self):
        session = Mock()
        session.snapshot.return_value = {"revision": 7}
        session.dispatch.return_value = {"revision": 8}

        result = photocut_cli.dispatch_opencv_confirmation_action(
            session, "move", {"dx": 5, "dy": 0}
        )

        self.assertEqual({"revision": 8}, result)
        dispatched = session.dispatch.call_args.args[0]
        self.assertIsInstance(dispatched, ConfirmationAction)
        self.assertEqual(7, dispatched.expected_revision)
        self.assertEqual("move", dispatched.kind)
        self.assertEqual({"dx": 5, "dy": 0}, dispatched.payload)

    def test_real_session_candidate_reset_skip_and_move_equivalence_table(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = self.auto_entry()

            for key in ("C", "B", "b"):
                with self.subTest(key=key):
                    expected_entry = copy.deepcopy(base)
                    expected = photocut_cli._v7_view_model_for_entry(
                        expected_entry, (100, 80)
                    )
                    if key == "C":
                        expected.toggle_alternate()
                    else:
                        expected.toggle_v52()
                    actual_entry = copy.deepcopy(base)
                    session, save = self.real_session(root, [actual_entry])
                    snapshot = self.dispatch_native(session, ord(key))
                    self.assertEqual(expected.work_corners, snapshot["editor"]["corners"])
                    self.assertEqual(expected.selection, snapshot["editor"]["candidate"]["id"])
                    self.assertEqual(
                        expected_entry["confirmation_selected_algorithm_version"],
                        snapshot["editor"]["candidate"]["algorithm_version"],
                    )
                    save.assert_not_called()

            movement_cases = (
                (ord("w"), ord("w"), (0, -50)),
                (ord("a"), ord("a"), (-50, 0)),
                (ord("s"), ord("s"), (0, 50)),
                (ord("d"), ord("d"), (50, 0)),
                (65361, 0, (-5, 0)),
                (65363, 0, (5, 0)),
                (65362, 0, (0, -5)),
                (65364, 0, (0, 5)),
            )
            for key, key_ascii, delta in movement_cases:
                with self.subTest(delta=delta):
                    expected = photocut_cli.ConfirmationState(
                        copy.deepcopy(base["algorithm_boundary_corners"]),
                        copy.deepcopy(base["boundary_corners"]),
                        (100, 80),
                        selected=0,
                    )
                    expected.move_selected(*delta)
                    session, save = self.real_session(root, [copy.deepcopy(base)])
                    photocut_cli.dispatch_opencv_confirmation_action(
                        session, "select_corner", {"index": 0}
                    )
                    snapshot = self.dispatch_native(session, key, key_ascii)
                    self.assertEqual(expected.work_corners, snapshot["editor"]["corners"])
                    save.assert_not_called()

            reset_entry = copy.deepcopy(base)
            expected_model = photocut_cli._v7_view_model_for_entry(
                copy.deepcopy(base), (100, 80)
            )
            expected_model.select_corner(0)
            expected_model.move_selected(5, 0)
            expected_model.reset()
            session, save = self.real_session(root, [reset_entry])
            photocut_cli.dispatch_opencv_confirmation_action(
                session, "select_corner", {"index": 0}
            )
            photocut_cli.dispatch_opencv_confirmation_action(
                session, "move", {"dx": 5, "dy": 0}
            )
            reset = self.dispatch_native(session, ord("R"))
            self.assertEqual(expected_model.work_corners, reset["editor"]["corners"])
            self.assertEqual(expected_model.selection, reset["editor"]["candidate"]["id"])
            save.assert_called_once()

            skip_entry = copy.deepcopy(base)
            expected_skip = photocut_cli._v7_view_model_for_entry(
                copy.deepcopy(base), (100, 80)
            )
            expected_skip.skip("user_skip")
            session, save = self.real_session(root, [skip_entry])
            skipped = self.dispatch_native(session, ord("X"))
            self.assertEqual("completed", skipped["session"]["status"])
            self.assertEqual(expected_skip.skip_reason, skip_entry["v7_skip_reason"])
            self.assertEqual("skipped", skip_entry["v7_operation"])
            save.assert_called_once()

    def test_real_session_terminal_navigation_and_mouse_action_equivalence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = self.auto_entry()

            entries = [
                copy.deepcopy(base),
                self.auto_entry("next.jpg", "sha256:next"),
            ]
            session, save = self.real_session(root, entries)
            next_snapshot = self.dispatch_native(session, 0, 0, navigate=1)
            self.assertEqual(1, next_snapshot["progress"]["index"])
            previous_snapshot = self.dispatch_native(session, 0, 0, navigate=-1)
            self.assertEqual(0, previous_snapshot["progress"]["index"])
            self.assertEqual(2, save.call_count)

            for key, status in ((ord("p"), "pause"), (ord("q"), "quit")):
                with self.subTest(status=status):
                    session, save = self.real_session(root, [copy.deepcopy(base)])
                    terminal = self.dispatch_native(session, key)
                    self.assertEqual(status, terminal["session"]["status"])
                    save.assert_called_once()

            mouse_entry = copy.deepcopy(base)
            session, save = self.real_session(root, [mouse_entry])
            expected = photocut_cli.ConfirmationState(
                copy.deepcopy(base["algorithm_boundary_corners"]),
                copy.deepcopy(base["boundary_corners"]),
                (100, 80),
                selected=0,
            )
            expected.move_selected(15, 12)
            mouse = photocut_cli.dispatch_opencv_confirmation_action(
                session, "set_corner", {"index": 0, "x": 25, "y": 22}
            )
            self.assertEqual(expected.work_corners, mouse["editor"]["corners"])
            save.assert_not_called()
            photocut_cli.dispatch_opencv_confirmation_action(session, "pause", {})
            save.assert_called_once()

            confirmed_entry = copy.deepcopy(base)
            store = AnnotationStore(root / "annotations.jsonl")
            session, save = self.real_session(
                root, [confirmed_entry], store=store
            )
            confirmed = self.dispatch_native(session, 32, 32)
            self.assertEqual("completed", confirmed["session"]["status"])
            event = store.events()[0]
            self.assertEqual(OPENCV_GUI_VERSION, event["gui_version"])
            self.assertEqual("v8:primary", event["selected_candidate_id"])
            self.assertEqual("8.2", event["selected_candidate_algorithm_version"])
            save.assert_called_once()

    def test_wasd_moves_selected_corner_fifty_pixels(self):
        expected = {
            "w": [10, 0], "a": [0, 10], "s": [10, 60], "d": [60, 10],
        }
        for key, corner in expected.items():
            with self.subTest(key=key):
                state = photocut_cli.ConfirmationState(
                    [[10, 10], [90, 10], [90, 70], [10, 70]],
                    [[10, 10], [90, 10], [90, 70], [10, 70]],
                    (100, 80), selected=0,
                )
                self.assertTrue(photocut_cli.apply_coarse_corner_key(state, ord(key)))
                self.assertEqual(corner, state.work_corners[0])

    def test_wasd_ignores_unselected_and_clamps_boundaries(self):
        state = photocut_cli.ConfirmationState(
            [[0, 0], [90, 10], [90, 70], [10, 70]],
            [[0, 0], [90, 10], [90, 70], [10, 70]],
            (100, 80), selected=-1,
        )
        self.assertFalse(photocut_cli.apply_coarse_corner_key(state, ord("d")))
        self.assertEqual([0, 0], state.work_corners[0])
        state.selected = 0
        self.assertTrue(photocut_cli.apply_coarse_corner_key(state, ord("a")))
        self.assertEqual([0, 0], state.work_corners[0])

    def test_wasd_cancels_mouse_capture_without_writing_json(self):
        state = photocut_cli.ConfirmationState(
            [[10, 10], [90, 10], [90, 70], [10, 70]],
            [[10, 10], [90, 10], [90, 70], [10, 70]],
            (100, 80), selected=0,
        )
        capture = photocut_cli.MagnifierDragCapture()
        capture.begin(PointerSample(10, 10, True))
        mouse_state = {
            "global_magnifier_capture": capture,
            "global_magnifier_capture_active": True,
            "magnifier_dragging": True,
        }
        with patch("photocut.cli.save_corners_info") as save:
            self.assertTrue(photocut_cli.apply_coarse_corner_key(state, ord("d"), mouse_state))
        self.assertFalse(capture.active)
        self.assertFalse(mouse_state["global_magnifier_capture_active"])
        self.assertEqual([60, 10], state.work_corners[0])
        save.assert_not_called()

    def test_arrow_delta_moves_five_pixels_and_cancels_capture(self):
        state = photocut_cli.ConfirmationState(
            [[10, 10], [90, 10], [90, 70], [10, 70]],
            [[10, 10], [90, 10], [90, 70], [10, 70]],
            (100, 80), selected=0,
        )
        capture = photocut_cli.MagnifierDragCapture()
        capture.begin(PointerSample(10, 10, True))
        mouse_state = {
            "global_magnifier_capture": capture,
            "global_magnifier_capture_active": True,
            "magnifier_dragging": True,
        }
        self.assertTrue(photocut_cli.apply_corner_delta(state, (5, 0), mouse_state))
        self.assertEqual([15, 10], state.work_corners[0])
        self.assertFalse(capture.active)


if __name__ == "__main__":
    unittest.main()
