import gc
import threading
import time
import unittest
import weakref

import numpy as np

from photocut.confirmation.controller import (
    ConfirmationAction,
    ConfirmationActionError,
    ConfirmationEntryController,
    ConfirmationSessionController,
    ConfirmationSessionError,
    LoadedConfirmationItem,
)


def _auto_v4_entry(**overrides):
    value = {
        "filename": "scan.jpg", "image_id": "sha256:image", "run_id": "run-1",
        "algorithm_version": "8.2", "detector_requested": "auto",
        "detector_used": "manual_review", "auto_cascade_version": "auto-v4",
        "auto_v4_status": "manual_review",
        "confirmation_primary_candidate_id": "v8:draft",
        "confirmation_selected_candidate_id": "v8:draft",
        "confirmation_selected_algorithm_version": "8.2",
        "algorithm_boundary_corners": [[10, 10], [90, 10], [90, 70], [10, 70]],
        "boundary_corners": [[10, 10], [90, 10], [90, 70], [10, 70]],
        "corners": [[10, 10], [90, 10], [90, 70], [10, 70]],
        "candidate_audit": [{
            "candidate_id": "edge-alternate", "stage_ranks": {"selected": 2},
            "adopted_refined_corners": [[12, 10], [92, 10], [92, 70], [12, 70]],
            "sources": ["edge:hough"],
        }],
        "v52_corners": [[11, 10], [91, 10], [91, 70], [11, 70]],
    }
    value.update(overrides)
    return value


def _loaded_item(filename):
    entry = _auto_v4_entry(filename=filename)
    editor = ConfirmationEntryController(entry, (100, 80))
    return LoadedConfirmationItem(
        entry=entry,
        image=np.zeros((80, 100, 3), dtype=np.uint8),
        image_token=f"current-{filename}",
        display_path=f"/fixtures/{filename}",
        editor=editor,
        finalized_evidence=None,
        previous_annotation=None,
        started_at=0.0,
        is_revision=False,
    )


class MemoryBackend:
    def __init__(self, items):
        self.items = items
        self.commit_count = 0
        self.checkpoints = []
        self.loaded_indices = []
        self.preview_indices = []
        self.fail_commit = False
        self.fail_load_indices = set()
        self.fail_load_after_commit = False
        self.skip_count = 0

    def count(self):
        return len(self.items)

    def load(self, index):
        self.loaded_indices.append(index)
        if index in self.fail_load_indices or (
            self.fail_load_after_commit and self.commit_count
        ):
            raise RuntimeError("load failed")
        return self.items[index]

    def load_preview(self, index):
        self.preview_indices.append(index)
        item = self.items[index]
        return item.image.copy(), f"preview-{index}"

    def checkpoint(self, item):
        self.checkpoints.append(item.entry["filename"])

    def commit(self, item, duration_ms, gui_version):
        if self.fail_commit:
            raise RuntimeError("commit failed")
        self.commit_count += 1
        return {"annotation_id": f"ann-{self.commit_count}", "gui_version": gui_version}

    def skip(self, item, reason):
        self.skip_count += 1
        item.entry["v7_operation"] = "skipped"
        item.entry["v7_skip_reason"] = reason


class ReleasingMemoryBackend(MemoryBackend):
    """A backend whose loaded items become controller-owned references."""

    def __init__(self, items):
        super().__init__(items)
        self.items = {index: item for index, item in enumerate(items)}
        self.total = len(items)

    def count(self):
        return self.total

    def load(self, index):
        self.loaded_indices.append(index)
        return self.items.pop(index)


class ConfirmationEntryControllerTests(unittest.TestCase):
    def test_auto_v4_restores_saved_draft_and_keeps_candidate_switch_locked(self):
        entry = _auto_v4_entry(
            boundary_corners=[[13, 14], [90, 10], [90, 70], [10, 70]],
            corners=[[13, 14], [90, 10], [90, 70], [10, 70]],
            manual_corners=[[13, 14], [90, 10], [90, 70], [10, 70]],
            manually_adjusted=True,
            draft_adjusted_corner_indices=[0],
        )

        editor = ConfirmationEntryController(entry, (100, 80))

        self.assertEqual([13, 14], editor.snapshot()["corners"][0])
        self.assertTrue(editor.snapshot()["dirty"])
        self.assertEqual([0], editor.snapshot()["adjusted_corner_indices"])
        self.assertFalse(editor.snapshot()["capabilities"]["candidate"])
        self.assertFalse(editor.snapshot()["capabilities"]["v52"])
        with self.assertRaises(ConfirmationActionError) as raised:
            editor.toggle_candidate()
        self.assertEqual("reset_required", raised.exception.code)

        editor.reset()
        self.assertEqual([10, 10], editor.snapshot()["corners"][0])
        self.assertFalse(editor.snapshot()["dirty"])

    def test_drag_is_clamped_and_blocks_candidate_switch_until_reset(self):
        editor = ConfirmationEntryController(_auto_v4_entry(), (100, 80))
        editor.select_corner(0)
        editor.set_selected_corner(-20, 15)

        self.assertEqual([0, 15], editor.snapshot()["corners"][0])
        self.assertTrue(editor.snapshot()["dirty"])
        with self.assertRaisesRegex(ConfirmationActionError, "reset"):
            editor.toggle_candidate()

        editor.reset()
        self.assertEqual("v8:draft", editor.snapshot()["candidate"]["id"])
        self.assertFalse(editor.snapshot()["dirty"])

    def test_move_and_zoom_keep_existing_pixel_contract(self):
        editor = ConfirmationEntryController(_auto_v4_entry(), (100, 80))
        editor.select_corner(0)
        editor.move_selected(50, 0)
        editor.move_selected(-5, 0)
        editor.set_zoom(16)

        self.assertEqual([55, 10], editor.snapshot()["corners"][0])
        self.assertEqual(16, editor.snapshot()["zoom"])

    def test_snapshot_exposes_fixed_capabilities_and_candidate_identity(self):
        editor = ConfirmationEntryController(_auto_v4_entry(), (100, 80))

        snapshot = editor.snapshot()

        self.assertEqual(
            {"confirm", "candidate", "v52", "reset", "skip", "move"},
            set(snapshot["capabilities"]),
        )
        self.assertTrue(snapshot["capabilities"]["candidate"])
        self.assertTrue(snapshot["capabilities"]["v52"])
        self.assertEqual("v8:draft", snapshot["candidate"]["id"])
        self.assertEqual("8.2", snapshot["candidate"]["algorithm_version"])

    def test_no_legal_legacy_draft_cannot_confirm_or_move(self):
        editor = ConfirmationEntryController(
            _auto_v4_entry(
                auto_cascade_version=None,
                detector_used="v5.2",
                algorithm_boundary_corners=[[1, 1]] * 4,
                boundary_corners=[[1, 1]] * 4,
                corners=[[1, 1]] * 4,
            ),
            (100, 80),
        )

        self.assertFalse(editor.snapshot()["capabilities"]["confirm"])
        self.assertFalse(editor.snapshot()["capabilities"]["move"])
        with self.assertRaisesRegex(ConfirmationActionError, "confirmed") as raised:
            editor.confirm()
        self.assertEqual("confirm_unavailable", raised.exception.code)

    def test_move_back_to_algorithm_position_stays_locked_until_reset(self):
        editor = ConfirmationEntryController(_auto_v4_entry(), (100, 80))
        editor.select_corner(0)
        editor.move_selected(1, 0)
        editor.move_selected(-1, 0)

        self.assertFalse(editor.snapshot()["dirty"])
        self.assertFalse(editor.snapshot()["capabilities"]["candidate"])
        self.assertFalse(editor.snapshot()["capabilities"]["v52"])
        for action in (editor.toggle_candidate, editor.toggle_v52):
            with self.subTest(action=action.__name__):
                with self.assertRaises(ConfirmationActionError) as raised:
                    action()
                self.assertEqual("reset_required", raised.exception.code)

        editor.reset()
        self.assertTrue(editor.snapshot()["capabilities"]["candidate"])
        self.assertTrue(editor.snapshot()["capabilities"]["v52"])
        editor.toggle_candidate()
        self.assertEqual("v7:edge-alternate", editor.snapshot()["candidate"]["id"])

    def test_v7_v52_toggle_updates_candidate_algorithm_version(self):
        editor = ConfirmationEntryController(
            _auto_v4_entry(
                auto_cascade_version=None,
                detector_used="v7",
                detection_status="v7_recommended",
                algorithm_version="7.1",
                confirmation_selected_algorithm_version="7.1",
            ),
            (100, 80),
        )

        editor.toggle_v52()

        self.assertEqual("v52", editor.snapshot()["candidate"]["id"])
        self.assertEqual("5.2", editor.snapshot()["candidate"]["algorithm_version"])


class ConfirmationSessionControllerTests(unittest.TestCase):
    def test_web_starts_each_entry_at_two_x_without_changing_opencv_default(self):
        web_backend = MemoryBackend([_loaded_item("a.jpg"), _loaded_item("b.jpg")])
        web = ConfirmationSessionController(web_backend, gui_version="2.0")
        self.assertEqual(2, web.snapshot()["editor"]["zoom"])

        web.dispatch(ConfirmationAction("next", 0, "next", {}))
        self.assertEqual(2, web.snapshot()["editor"]["zoom"])

        opencv = ConfirmationSessionController(
            MemoryBackend([_loaded_item("legacy.jpg")]), gui_version="1.0"
        )
        self.assertEqual(4, opencv.snapshot()["editor"]["zoom"])
        opencv.dispatch(ConfirmationAction("zoom-16", 0, "set_zoom", {"zoom": 16}))
        self.assertEqual(16, opencv.snapshot()["editor"]["zoom"])

    def test_duplicate_action_id_returns_same_result_without_second_commit(self):
        backend = MemoryBackend([_loaded_item("a.jpg"), _loaded_item("b.jpg")])
        session = ConfirmationSessionController(backend, gui_version="2.0")
        revision = session.snapshot()["revision"]
        action = ConfirmationAction("act-1", revision, "confirm", {})

        first = session.dispatch(action)
        second = session.dispatch(action)

        self.assertEqual(first, second)
        self.assertEqual(1, backend.commit_count)
        self.assertEqual(1, first["progress"]["index"])

    def test_stale_action_fails_closed(self):
        backend = MemoryBackend([_loaded_item("a.jpg")])
        session = ConfirmationSessionController(backend, gui_version="2.0")
        session.dispatch(ConfirmationAction("act-1", 0, "select_corner", {"index": 0}))

        with self.assertRaisesRegex(ConfirmationSessionError, "stale") as raised:
            session.dispatch(ConfirmationAction("act-2", 0, "move", {"dx": 5, "dy": 0}))

        self.assertEqual("stale_revision", raised.exception.code)
        self.assertEqual("action", raised.exception.scope)
        self.assertEqual(1, session.snapshot()["revision"])
        self.assertEqual([10, 10], session.snapshot()["editor"]["corners"][0])

    def test_conflicting_reuse_of_action_id_is_rejected(self):
        backend = MemoryBackend([_loaded_item("a.jpg")])
        session = ConfirmationSessionController(backend, gui_version="2.0")
        session.dispatch(ConfirmationAction("act-1", 0, "select_corner", {"index": 0}))

        with self.assertRaises(ConfirmationSessionError) as raised:
            session.dispatch(ConfirmationAction("act-1", 1, "move", {"dx": 1, "dy": 0}))

        self.assertEqual("action_id_conflict", raised.exception.code)

    def test_navigation_checkpoints_current_item_and_preview_does_not_load_editor(self):
        backend = MemoryBackend([_loaded_item("a.jpg"), _loaded_item("b.jpg")])
        session = ConfirmationSessionController(backend, gui_version="2.0")

        preview, token = session.next_preview_source()
        self.assertEqual("preview-1", token)
        self.assertEqual([0], backend.loaded_indices)
        self.assertEqual([1], backend.preview_indices)
        self.assertEqual(0, session.snapshot()["progress"]["index"])
        del preview

        result = session.dispatch(ConfirmationAction("next", 0, "next", {}))

        self.assertEqual(["a.jpg"], backend.checkpoints)
        self.assertEqual([0, 1], backend.loaded_indices)
        self.assertEqual("b.jpg", session.current_item.entry["filename"])
        self.assertEqual(1, result["revision"])

    def test_blocked_preview_decode_does_not_block_navigation_and_returns_stale(self):
        class BlockingPreviewBackend(MemoryBackend):
            def __init__(self, items):
                super().__init__(items)
                self.preview_started = threading.Event()
                self.preview_release = threading.Event()

            def load_preview(self, index):
                self.preview_indices.append(index)
                self.preview_started.set()
                if not self.preview_release.wait(2):
                    raise AssertionError("preview release timed out")
                item = self.items[index]
                return item.image.copy(), item.image_token

        backend = BlockingPreviewBackend([_loaded_item("a.jpg"), _loaded_item("b.jpg")])
        session = ConfirmationSessionController(backend, gui_version="2.0")
        result = {}

        def load_preview():
            try:
                result["value"] = session.next_preview_source()
            except Exception as exc:
                result["error"] = exc

        thread = threading.Thread(target=load_preview)
        thread.start()
        self.assertTrue(backend.preview_started.wait(1))
        started = time.perf_counter()
        navigated = session.dispatch(ConfirmationAction("next", 0, "next", {}))
        elapsed = time.perf_counter() - started
        backend.preview_release.set()
        thread.join(2)

        self.assertLess(elapsed, 0.25)
        self.assertFalse(thread.is_alive())
        self.assertEqual("b.jpg", navigated["image"]["filename"])
        self.assertNotIn("value", result)
        self.assertIsInstance(result.get("error"), ConfirmationSessionError)
        self.assertEqual("stale_preview", result["error"].code)
        self.assertEqual("image", result["error"].scope)

    def test_commit_failure_leaves_current_item_and_revision_retryable(self):
        backend = MemoryBackend([_loaded_item("a.jpg"), _loaded_item("b.jpg")])
        backend.fail_commit = True
        session = ConfirmationSessionController(backend, gui_version="2.0")
        action = ConfirmationAction("confirm-1", 0, "confirm", {})

        with self.assertRaisesRegex(RuntimeError, "commit failed"):
            session.dispatch(action)

        self.assertEqual(0, session.snapshot()["revision"])
        self.assertEqual("a.jpg", session.current_item.entry["filename"])
        self.assertEqual(0, backend.commit_count)
        backend.fail_commit = False
        result = session.dispatch(action)
        self.assertEqual(1, backend.commit_count)
        self.assertEqual(1, result["revision"])
        self.assertEqual("b.jpg", result["image"]["filename"])

    def test_snapshot_and_technical_details_use_fixed_dtos(self):
        item = _loaded_item("a.jpg")
        item.entry["source_sha256"] = "source-sha"
        item.entry["candidate_audit"].append({"candidate_id": "extra"})
        item.previous_annotation = {"gui_version": "1.0"}
        backend = MemoryBackend([item])
        session = ConfirmationSessionController(backend, gui_version="2.0")

        snapshot = session.snapshot()
        details = session.technical_details()

        self.assertEqual(
            {"revision", "session", "progress", "image", "editor", "identity", "storage", "error"},
            set(snapshot),
        )
        self.assertEqual({"status": "active", "readonly": False}, snapshot["session"])
        self.assertEqual("current-a.jpg", snapshot["image"]["token"])
        self.assertEqual({"path", "source_sha256", "detection_id", "previous_gui", "candidate_audit"}, set(details))
        self.assertEqual("/fixtures/a.jpg", details["path"])
        self.assertEqual("source-sha", details["source_sha256"])
        self.assertEqual("1.0", details["previous_gui"])
        details["candidate_audit"].append({"candidate_id": "mutated"})
        self.assertEqual(2, len(item.entry["candidate_audit"]))

    def test_invalid_action_payload_and_end_preview_have_scoped_errors(self):
        backend = MemoryBackend([_loaded_item("a.jpg")])
        session = ConfirmationSessionController(backend, gui_version="2.0")

        with self.assertRaises(ConfirmationSessionError) as action_error:
            session.dispatch(ConfirmationAction("bad", 0, "move", {"dx": 1}))
        with self.assertRaises(ConfirmationSessionError) as image_error:
            session.next_preview_source()

        self.assertEqual("invalid_action", action_error.exception.code)
        self.assertEqual("action", action_error.exception.scope)
        self.assertEqual("no_next_image", image_error.exception.code)
        self.assertEqual("image", image_error.exception.scope)

    def test_confirm_preloads_next_before_commit_and_load_failure_is_retryable(self):
        backend = MemoryBackend([_loaded_item("a.jpg"), _loaded_item("b.jpg")])
        backend.fail_load_indices.add(1)
        session = ConfirmationSessionController(backend, gui_version="2.0")
        action = ConfirmationAction("confirm-1", 0, "confirm", {})

        with self.assertRaisesRegex(RuntimeError, "load failed"):
            session.dispatch(action)

        self.assertEqual(0, backend.commit_count)
        self.assertEqual(0, session.revision)
        self.assertEqual("a.jpg", session.current_item.entry["filename"])
        backend.fail_load_indices.clear()
        result = session.dispatch(action)
        self.assertEqual(1, backend.commit_count)
        self.assertEqual("b.jpg", result["image"]["filename"])

    def test_confirm_never_loads_next_after_successful_commit(self):
        backend = MemoryBackend([_loaded_item("a.jpg"), _loaded_item("b.jpg")])
        backend.fail_load_after_commit = True
        session = ConfirmationSessionController(backend, gui_version="2.0")

        result = session.dispatch(ConfirmationAction("confirm-1", 0, "confirm", {}))

        self.assertEqual(1, backend.commit_count)
        self.assertEqual(1, result["revision"])
        self.assertEqual("b.jpg", result["image"]["filename"])
        self.assertEqual([0, 1], backend.loaded_indices)

    def test_skip_preloads_next_before_external_write(self):
        backend = MemoryBackend([_loaded_item("a.jpg"), _loaded_item("b.jpg")])
        backend.fail_load_indices.add(1)
        session = ConfirmationSessionController(backend, gui_version="2.0")

        with self.assertRaisesRegex(RuntimeError, "load failed"):
            session.dispatch(ConfirmationAction("skip-1", 0, "skip", {"reason": "blur"}))

        self.assertEqual(0, backend.skip_count)
        self.assertEqual(0, session.revision)
        self.assertEqual("a.jpg", session.current_item.entry["filename"])

    def test_navigation_load_failure_does_not_advance_session(self):
        backend = MemoryBackend([_loaded_item("a.jpg"), _loaded_item("b.jpg")])
        backend.fail_load_indices.add(1)
        session = ConfirmationSessionController(backend, gui_version="2.0")

        with self.assertRaisesRegex(RuntimeError, "load failed"):
            session.dispatch(ConfirmationAction("next-1", 0, "next", {}))

        self.assertEqual(0, session.revision)
        self.assertEqual(0, session.index)
        self.assertEqual("a.jpg", session.current_item.entry["filename"])
        self.assertEqual(["a.jpg"], backend.checkpoints)

    def test_stopped_sessions_reject_new_actions_but_replay_cached_result(self):
        for terminal_kind in ("pause", "quit"):
            with self.subTest(terminal_kind=terminal_kind):
                backend = MemoryBackend([_loaded_item("a.jpg")])
                session = ConfirmationSessionController(backend, gui_version="2.0")
                terminal = ConfirmationAction("terminal", 0, terminal_kind, {})
                first = session.dispatch(terminal)

                self.assertEqual(first, session.dispatch(terminal))
                with self.assertRaises(ConfirmationSessionError) as raised:
                    session.dispatch(ConfirmationAction("new", 1, "select_corner", {"index": 0}))

                self.assertEqual("session_stopped", raised.exception.code)
                self.assertEqual("session", raised.exception.scope)
                self.assertEqual(1, session.revision)

    def test_completed_session_rejects_new_actions_but_replays_cached_result(self):
        backend = MemoryBackend([_loaded_item("a.jpg")])
        session = ConfirmationSessionController(backend, gui_version="2.0")
        terminal = ConfirmationAction("confirm-1", 0, "confirm", {})
        first = session.dispatch(terminal)

        self.assertEqual("completed", first["session"]["status"])
        self.assertEqual(first, session.dispatch(terminal))
        with self.assertRaises(ConfirmationSessionError) as raised:
            session.dispatch(ConfirmationAction("new", 1, "reset", {}))

        self.assertEqual("session_stopped", raised.exception.code)
        self.assertEqual("session", raised.exception.scope)

    def test_invalid_payload_values_are_atomic_and_action_scoped(self):
        invalid_actions = (
            ConfirmationAction("bool-index", 0, "select_corner", {"index": True}),
            ConfirmationAction("bad-index", 0, "set_corner", {"index": 4, "x": 11, "y": 11}),
            ConfirmationAction("string-delta", 0, "move", {"dx": "1", "dy": 0}),
            ConfirmationAction("bad-zoom", 0, "set_zoom", {"zoom": 5}),
            ConfirmationAction("blank-reason", 0, "skip", {"reason": "  "}),
        )
        for action in invalid_actions:
            with self.subTest(action_id=action.action_id):
                backend = MemoryBackend([_loaded_item("a.jpg")])
                session = ConfirmationSessionController(backend, gui_version="2.0")
                before = session.snapshot()

                with self.assertRaises(ConfirmationSessionError) as raised:
                    session.dispatch(action)

                self.assertEqual("invalid_action", raised.exception.code)
                self.assertEqual("action", raised.exception.scope)
                self.assertEqual(before, session.snapshot())
                self.assertEqual(0, backend.skip_count)

    def test_action_cache_evicts_oldest_entry_after_256_successes(self):
        backend = MemoryBackend([_loaded_item("a.jpg")])
        session = ConfirmationSessionController(backend, gui_version="2.0")
        for revision in range(257):
            session.dispatch(
                ConfirmationAction(f"edit-{revision}", revision, "select_corner", {"index": revision % 4})
            )

        self.assertEqual(256, len(session.results))
        self.assertNotIn("edit-0", session.results)
        with self.assertRaises(ConfirmationSessionError) as raised:
            session.dispatch(ConfirmationAction("edit-0", 0, "select_corner", {"index": 0}))
        self.assertEqual("stale_revision", raised.exception.code)

    def test_navigation_releases_controller_reference_to_previous_item(self):
        first = _loaded_item("a.jpg")
        previous_ref = weakref.ref(first)
        backend = ReleasingMemoryBackend([first, _loaded_item("b.jpg")])
        session = ConfirmationSessionController(backend, gui_version="2.0")
        del first

        session.dispatch(ConfirmationAction("next-1", 0, "next", {}))
        gc.collect()

        self.assertEqual("b.jpg", session.current_item.entry["filename"])
        self.assertIsNone(previous_ref())

    def test_current_item_is_read_only(self):
        session = ConfirmationSessionController(MemoryBackend([_loaded_item("a.jpg")]), gui_version="2.0")

        with self.assertRaises(AttributeError):
            session.current_item = _loaded_item("replacement.jpg")
