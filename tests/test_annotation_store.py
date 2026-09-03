import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from photocut.data.annotation_store import AnnotationStore
from photocut.confirmation.model import build_annotation_event
from photocut.data.dataset_store import append_jsonl_validated, load_jsonl


class AnnotationTestCase(unittest.TestCase):
    def setUp(self):
        self.algorithm = [[10, 10], [110, 10], [110, 70], [10, 70]]

    def event(
        self,
        annotation_id,
        *,
        image_id="sha256:abc",
        run_id="run_1",
        boundary=None,
        adjusted=(),
        supersedes=None,
    ):
        event = build_annotation_event(
            image_id=image_id,
            run_id=run_id,
            algorithm_boundary_corners=self.algorithm,
            boundary_corners=boundary or self.algorithm,
            adjusted_corner_indices=adjusted,
            confirmation_duration_ms=1200,
            supersedes_annotation_id=supersedes,
        )
        event["annotation_id"] = annotation_id
        return event

    def write_events(self, path, events):
        path.write_text(
            "".join(json.dumps(event) + "\n" for event in events),
            encoding="utf-8",
        )

    def assert_append_rejected_without_change(self, path, event, message):
        before = path.read_bytes() if path.exists() else None
        with self.assertRaisesRegex(ValueError, message):
            AnnotationStore(path).append(event)
        after = path.read_bytes() if path.exists() else None
        self.assertEqual(before, after)


class AnnotationEventTests(AnnotationTestCase):
    def test_direct_acceptance_is_complete_zero_error_truth(self):
        event = build_annotation_event(
            image_id="sha256:abc",
            run_id="run_1",
            algorithm_boundary_corners=self.algorithm,
            boundary_corners=self.algorithm,
            adjusted_corner_indices=set(),
            confirmation_duration_ms=1200,
        )

        self.assertEqual(1, event["schema_version"])
        self.assertTrue(event["annotation_id"].startswith("ann_"))
        self.assertEqual("accepted", event["confirmation"])
        self.assertEqual([], event["adjusted_corner_indices"])
        self.assertEqual([0.0, 0.0, 0.0, 0.0], event["corner_errors_px"])
        self.assertEqual(1200, event["confirmation_duration_ms"])
        self.assertIn("+", event["confirmed_at"])

    def test_adjustment_derives_confirmation_indices_and_euclidean_errors(self):
        boundary = [point[:] for point in self.algorithm]
        boundary[0] = [7, 6]
        boundary[3] = [11, 70]

        event = build_annotation_event(
            "sha256:abc", "run_1", self.algorithm, boundary, [3, 0], 3
        )

        self.assertEqual("adjusted", event["confirmation"])
        self.assertEqual([0, 3], event["adjusted_corner_indices"])
        self.assertEqual([5.0, 0.0, 0.0, 1.0], event["corner_errors_px"])

    def test_boundary_fields_require_exactly_four_two_integer_points(self):
        invalid_corners = (
            self.algorithm[:3],
            self.algorithm + [[0, 0]],
            [[10], *self.algorithm[1:]],
            [[10, 10, 10], *self.algorithm[1:]],
            [[True, 10], *self.algorithm[1:]],
            [[10.0, 10], *self.algorithm[1:]],
        )

        for field in ("algorithm_boundary_corners", "boundary_corners"):
            for corners in invalid_corners:
                arguments = {
                    "image_id": "sha256:abc",
                    "run_id": "run_1",
                    "algorithm_boundary_corners": self.algorithm,
                    "boundary_corners": self.algorithm,
                    "adjusted_corner_indices": [],
                    "confirmation_duration_ms": 1,
                }
                arguments[field] = corners
                with self.subTest(field=field, corners=corners):
                    with self.assertRaisesRegex(ValueError, field):
                        build_annotation_event(**arguments)

    def test_image_and_run_ids_must_be_non_empty_strings(self):
        for field, value in (
            ("image_id", ""),
            ("image_id", "   "),
            ("image_id", 1),
            ("run_id", ""),
            ("run_id", None),
        ):
            arguments = {
                "image_id": "sha256:abc",
                "run_id": "run_1",
                "algorithm_boundary_corners": self.algorithm,
                "boundary_corners": self.algorithm,
                "adjusted_corner_indices": [],
                "confirmation_duration_ms": 1,
            }
            arguments[field] = value
            with self.subTest(field=field, value=value):
                with self.assertRaisesRegex(ValueError, field):
                    build_annotation_event(**arguments)

    def test_duration_must_be_a_non_negative_integer_without_coercion(self):
        for duration in (-1, True, 1.0, "1"):
            with self.subTest(duration=duration):
                with self.assertRaisesRegex(ValueError, "confirmation_duration_ms"):
                    build_annotation_event(
                        "sha256:abc",
                        "run_1",
                        self.algorithm,
                        self.algorithm,
                        [],
                        duration,
                    )

    def test_adjusted_indices_must_be_unique_integers_in_range(self):
        for adjusted in ([0, 0], [True], [1.0], [-1], [4]):
            with self.subTest(adjusted=adjusted):
                with self.assertRaisesRegex(ValueError, "adjusted_corner_indices"):
                    build_annotation_event(
                        "sha256:abc",
                        "run_1",
                        self.algorithm,
                        self.algorithm,
                        adjusted,
                        1,
                    )

    def test_adjusted_indices_must_exactly_match_changed_points(self):
        boundary = [point[:] for point in self.algorithm]
        boundary[0] = [9, 9]

        for adjusted in ([], [0, 1]):
            with self.subTest(adjusted=adjusted):
                with self.assertRaisesRegex(ValueError, "adjusted_corner_indices"):
                    build_annotation_event(
                        "sha256:abc",
                        "run_1",
                        self.algorithm,
                        boundary,
                        adjusted,
                        1,
                    )

    def test_supersedes_id_is_omitted_or_a_non_empty_string(self):
        event = build_annotation_event(
            "sha256:abc", "run_1", self.algorithm, self.algorithm, [], 1
        )
        self.assertNotIn("supersedes_annotation_id", event)

        for supersedes in ("", "  ", 1):
            with self.subTest(supersedes=supersedes):
                with self.assertRaisesRegex(ValueError, "supersedes_annotation_id"):
                    build_annotation_event(
                        "sha256:abc",
                        "run_1",
                        self.algorithm,
                        self.algorithm,
                        [],
                        1,
                        supersedes,
                    )

    def test_event_owns_independent_corner_copies(self):
        shared = [point[:] for point in self.algorithm]
        event = build_annotation_event(
            "sha256:abc", "run_1", shared, shared, [], 1
        )

        shared[0][0] = -1
        event["algorithm_boundary_corners"][1][0] = -2

        self.assertEqual([10, 10], event["algorithm_boundary_corners"][0])
        self.assertEqual([110, 10], event["boundary_corners"][1])


class AnnotationSchemaV2Tests(AnnotationTestCase):
    def setUp(self):
        super().setUp()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def v2_event(
        self,
        *,
        gui_version="1.0",
        algorithm_version="7.0",
        supersedes=None,
    ):
        return build_annotation_event(
            image_id="sha256:abc",
            run_id="run_1",
            algorithm_boundary_corners=self.algorithm,
            boundary_corners=self.algorithm,
            adjusted_corner_indices=[],
            confirmation_duration_ms=1,
            supersedes_annotation_id=supersedes,
            schema_version=2,
            gui_version=gui_version,
            algorithm_version=algorithm_version,
            detection_id="det-1",
            detector_requested="auto",
            detector_used="v7",
        )

    def test_mixed_v1_v2_history_is_read_without_rewriting_v1_bytes(self):
        path = self.root / "annotations.jsonl"
        first = self.event("ann-v1")
        self.write_events(path, [first])
        before = path.read_bytes()
        second = self.v2_event(supersedes="ann-v1")

        AnnotationStore(path).append_idempotent(second)

        events = AnnotationStore(path).events()
        self.assertEqual([1, 2], [event["schema_version"] for event in events])
        self.assertTrue(path.read_bytes().startswith(before))
        self.assertEqual("1.0", events[1]["gui_version"])
        self.assertEqual("7.0", events[1]["algorithm_version"])

    def test_v2_requires_gui_version_but_allows_unknown_algorithm(self):
        event = self.v2_event(algorithm_version=None)
        self.assertNotIn("algorithm_version", event)
        for bad in (None, "", " 1.0", "v1.0", "1." + "2" * 31):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(ValueError, "gui_version"):
                    self.v2_event(gui_version=bad)

    def test_schema_v1_shape_is_unchanged_and_rejects_tool_identity(self):
        event = build_annotation_event(
            "sha256:abc", "run_1", self.algorithm, self.algorithm, [], 1
        )
        self.assertEqual(
            {
                "schema_version",
                "annotation_id",
                "image_id",
                "run_id",
                "algorithm_boundary_corners",
                "boundary_corners",
                "confirmation",
                "adjusted_corner_indices",
                "corner_errors_px",
                "confirmation_duration_ms",
                "confirmed_at",
            },
            set(event),
        )
        with self.assertRaisesRegex(ValueError, "schema v1"):
            build_annotation_event(
                "sha256:abc",
                "run_1",
                self.algorithm,
                self.algorithm,
                [],
                1,
                gui_version="1.0",
            )

    def test_builder_rejects_unsupported_schema_versions(self):
        for bad in (True, 0, 3, "2"):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(ValueError, "schema_version"):
                    build_annotation_event(
                        "sha256:abc",
                        "run_1",
                        self.algorithm,
                        self.algorithm,
                        [],
                        1,
                        schema_version=bad,
                    )

    def test_v2_store_rejects_invalid_identity_and_unexpected_fields(self):
        path = self.root / "annotations.jsonl"
        self.write_events(path, [self.event("ann-v1")])
        for field, value, message in (
            ("algorithm_version", " 7.0", "algorithm_version"),
            ("detection_id", " ", "detection_id"),
            ("detector_requested", 1, "detector_requested"),
            ("detector_used", "", "detector_used"),
            ("provenance", {}, "schema"),
            ("forged", True, "schema"),
        ):
            event = self.v2_event(supersedes="ann-v1")
            event[field] = value
            with self.subTest(field=field):
                self.assert_append_rejected_without_change(path, event, message)

    def test_revision_identity_includes_schema_and_tool_identity(self):
        first = self.event("ann-v1")
        one = self.v2_event(gui_version="1.0", supersedes=first["annotation_id"])
        same = self.v2_event(gui_version="1.0", supersedes=first["annotation_id"])
        changed = self.v2_event(
            gui_version="1.0.1", supersedes=first["annotation_id"]
        )
        v1 = build_annotation_event(
            "sha256:abc",
            "run_1",
            self.algorithm,
            self.algorithm,
            [],
            1,
            first["annotation_id"],
        )

        self.assertEqual(one["annotation_id"], same["annotation_id"])
        self.assertNotEqual(one["annotation_id"], changed["annotation_id"])
        self.assertNotEqual(one["annotation_id"], v1["annotation_id"])

    def test_idempotent_retry_ignores_only_volatile_fields(self):
        path = self.root / "annotations.jsonl"
        first = self.event("ann-v1")
        self.write_events(path, [first])
        original = self.v2_event(supersedes="ann-v1")
        store = AnnotationStore(path)
        recorded = store.append_idempotent(original)
        before_retry = path.read_bytes()
        retry = self.v2_event(supersedes="ann-v1")
        retry["confirmed_at"] = "2099-01-01T00:00:00+00:00"
        retry["confirmation_duration_ms"] = 999

        self.assertEqual(recorded, store.append_idempotent(retry))
        self.assertEqual(before_retry, path.read_bytes())

        changed = self.v2_event(gui_version="1.0.1", supersedes="ann-v1")
        changed["annotation_id"] = original["annotation_id"]
        self.assert_append_rejected_without_change(path, changed, "duplicate")


class AnnotationStoreTests(AnnotationTestCase):
    def test_append_round_trips_unicode_line_separators_inside_fields(self):
        for separator in ("\u0085", "\u2028", "\u2029"):
            with self.subTest(separator=hex(ord(separator))):
                with tempfile.TemporaryDirectory() as temp_dir:
                    path = Path(temp_dir) / "annotations.jsonl"
                    store = AnnotationStore(path)
                    event = self.event(
                        "ann_1", run_id=f"before{separator}after"
                    )

                    store.append(event)

                    self.assertEqual(1, path.read_bytes().count(b"\n"))
                    self.assertEqual([event], store.events())

    def test_append_delegates_to_durable_jsonl_helper(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "annotations.jsonl"
            event = self.event("ann_1")

            with patch(
                "photocut.data.annotation_store.append_jsonl_validated",
                wraps=append_jsonl_validated,
            ) as durable:
                AnnotationStore(path).append(event)

            durable.assert_called_once()
            self.assertEqual(path, durable.call_args.args[0])

    def test_append_stores_a_copy_and_events_returns_fresh_deep_copies(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = AnnotationStore(Path(temp_dir) / "annotations.jsonl")
            event = self.event("ann_1")
            store.append(event)

            event["boundary_corners"][0][0] = -1
            first_read = store.events()
            first_read[0]["boundary_corners"][0][0] = -2

            self.assertEqual(10, store.events()[0]["boundary_corners"][0][0])

    def test_append_partial_write_preserves_readable_annotation_history(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "annotations.jsonl"
            store = AnnotationStore(path)
            first = self.event("ann_1", image_id="sha256:one")
            store.append(first)
            original = path.read_bytes()
            real_write = os.write
            write_calls = 0

            def partial_write_then_fail(descriptor, data):
                nonlocal write_calls
                write_calls += 1
                if write_calls == 1:
                    return real_write(descriptor, data[:7])
                raise OSError("annotation append failed")

            with patch(
                "photocut.data.dataset_store.os.write", side_effect=partial_write_then_fail
            ):
                with self.assertRaisesRegex(OSError, "annotation append failed"):
                    store.append(self.event("ann_2", image_id="sha256:two"))

            self.assertEqual(original, path.read_bytes())
            self.assertEqual([first], store.events())

    def test_read_rejects_forged_inode_swapped_in_and_restored_during_read(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = root / "annotations.jsonl"
            backup = root / "original.jsonl"
            forged = root / "forged.jsonl"
            legitimate = self.event("ann_legitimate")
            forged_event = self.event(
                "ann_forged", image_id="sha256:forged"
            )
            self.write_events(path, [legitimate])
            self.write_events(forged, [forged_event])
            original_bytes = path.read_bytes()
            forged_bytes = forged.read_bytes()
            real_read = os.read

            def swap_in_forged_inode():
                os.replace(path, backup)
                os.replace(forged, path)

            def restore_original_inode():
                os.replace(path, forged)
                os.replace(backup, path)

            def replace_during_descriptor_read(descriptor, size):
                if not backup.exists():
                    swap_in_forged_inode()
                return real_read(descriptor, size)

            try:
                with patch(
                    "photocut.data.dataset_store.os.read",
                    side_effect=replace_during_descriptor_read,
                ):
                    with self.assertRaisesRegex(ValueError, "changed|identity"):
                        AnnotationStore(path).events()
            finally:
                if backup.exists():
                    restore_original_inode()

            self.assertEqual(original_bytes, path.read_bytes())
            self.assertEqual(forged_bytes, forged.read_bytes())

    def test_append_identity_cannot_be_redirected_to_symlink_after_validation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = root / "annotations.jsonl"
            backup = root / "original.jsonl"
            unrelated = root / "unrelated.jsonl"
            store = AnnotationStore(path)
            store.append(self.event("ann_1", image_id="sha256:one"))
            original_bytes = path.read_bytes()
            unrelated.write_bytes(b"unrelated bytes\n")
            unrelated_bytes = unrelated.read_bytes()
            real_append = append_jsonl_validated

            def redirect_append(*arguments, **keywords):
                os.replace(path, backup)
                path.symlink_to(unrelated)
                try:
                    return real_append(*arguments, **keywords)
                finally:
                    path.unlink()
                    os.replace(backup, path)

            with patch(
                "photocut.data.annotation_store.append_jsonl_validated",
                side_effect=redirect_append,
            ):
                with self.assertRaisesRegex(ValueError, "changed|symlink"):
                    store.append(
                        self.event("ann_2", image_id="sha256:two")
                    )

            self.assertEqual(original_bytes, path.read_bytes())
            self.assertEqual(unrelated_bytes, unrelated.read_bytes())

    def test_latest_by_image_returns_only_active_revision_and_a_copy(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = AnnotationStore(Path(temp_dir) / "annotations.jsonl")
            first = self.event("ann_1")
            boundary = [point[:] for point in self.algorithm]
            boundary[0] = [9, 9]
            revision = self.event(
                "ann_2", boundary=boundary, adjusted=[0], supersedes="ann_1"
            )
            store.append(first)
            store.append(revision)

            latest = store.latest_by_image()
            latest["sha256:abc"]["boundary_corners"][0][0] = -1

            self.assertEqual(
                "ann_2",
                store.latest_by_image()["sha256:abc"]["annotation_id"],
            )
            self.assertEqual(2, len(store.events()))

    def test_revise_constructs_without_appending_and_preserves_detection_identity(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "annotations.jsonl"
            store = AnnotationStore(path)
            previous = self.event("ann_1")
            store.append(previous)
            before = path.read_bytes()
            boundary = [point[:] for point in self.algorithm]
            boundary[2] = [109, 69]

            revision = store.revise(previous, boundary, [2], 800)

            self.assertEqual(before, path.read_bytes())
            self.assertEqual(previous["image_id"], revision["image_id"])
            self.assertEqual(previous["run_id"], revision["run_id"])
            self.assertEqual(
                previous["algorithm_boundary_corners"],
                revision["algorithm_boundary_corners"],
            )
            self.assertEqual("ann_1", revision["supersedes_annotation_id"])
            self.assertNotEqual("ann_1", revision["annotation_id"])
            store.append(revision)
            self.assertEqual(2, len(store.events()))

    def test_revise_validates_previous_and_new_adjusted_indices(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = AnnotationStore(Path(temp_dir) / "annotations.jsonl")
            previous = self.event("ann_1")
            boundary = [point[:] for point in self.algorithm]
            boundary[0] = [9, 9]

            with self.assertRaisesRegex(ValueError, "adjusted_corner_indices"):
                store.revise(previous, boundary, [], 1)

            previous["confirmation"] = "adjusted"
            with self.assertRaisesRegex(ValueError, "confirmation"):
                store.revise(previous, self.algorithm, [], 1)

    def test_event_schema_rejects_missing_or_unexpected_fields_before_write(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "annotations.jsonl"
            for mutation in ("missing", "extra"):
                event = self.event(f"ann_{mutation}")
                if mutation == "missing":
                    del event["run_id"]
                else:
                    event["forged"] = True
                with self.subTest(mutation=mutation):
                    self.assert_append_rejected_without_change(path, event, "schema")

    def test_confirmation_and_errors_cannot_be_forged(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "annotations.jsonl"
            for field, value in (
                ("confirmation", "adjusted"),
                ("corner_errors_px", [1.0, 0.0, 0.0, 0.0]),
            ):
                event = self.event(f"ann_{field}")
                event[field] = value
                with self.subTest(field=field):
                    self.assert_append_rejected_without_change(path, event, field)

    def test_duplicate_annotation_id_is_rejected_without_changing_bytes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "annotations.jsonl"
            first = self.event("ann_1")
            self.write_events(path, [first])
            self.assert_append_rejected_without_change(
                path, self.event("ann_1"), "duplicate"
            )

    def test_unknown_supersedes_is_rejected_without_changing_bytes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "annotations.jsonl"
            event = self.event("ann_2", supersedes="ann_missing")
            self.assert_append_rejected_without_change(path, event, "unknown")

    def test_cross_image_supersedes_is_rejected_without_changing_bytes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "annotations.jsonl"
            self.write_events(path, [self.event("ann_1")])
            revision = self.event(
                "ann_2", image_id="sha256:other", supersedes="ann_1"
            )
            self.assert_append_rejected_without_change(path, revision, "another image")

    def test_revision_must_supersede_current_active_head(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "annotations.jsonl"
            first = self.event("ann_1")
            second = self.event("ann_2", supersedes="ann_1")
            self.write_events(path, [first, second])

            self.assert_append_rejected_without_change(
                path,
                self.event("ann_3", supersedes="ann_1"),
                "current active head",
            )

    def test_second_independent_head_is_rejected_without_changing_bytes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "annotations.jsonl"
            self.write_events(path, [self.event("ann_1")])
            self.assert_append_rejected_without_change(
                path, self.event("ann_2"), "multiple active heads"
            )

    def test_read_rejects_branching_history(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "annotations.jsonl"
            self.write_events(
                path,
                [
                    self.event("ann_1"),
                    self.event("ann_2", supersedes="ann_1"),
                    self.event("ann_3", supersedes="ann_1"),
                ],
            )

            with self.assertRaisesRegex(ValueError, "current active head"):
                AnnotationStore(path).events()
            self.assert_append_rejected_without_change(
                path,
                self.event("ann_new", image_id="sha256:new"),
                "current active head",
            )

    def test_read_rejects_cycle_explicitly(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "annotations.jsonl"
            first = self.event("ann_1", supersedes="ann_2")
            second = self.event("ann_2", supersedes="ann_1")
            self.write_events(path, [first, second])

            with self.assertRaisesRegex(ValueError, "cycle"):
                AnnotationStore(path).latest_by_image()
            self.assert_append_rejected_without_change(
                path,
                self.event("ann_new", image_id="sha256:new"),
                "cycle",
            )

    def test_read_rejects_multiple_active_heads(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "annotations.jsonl"
            self.write_events(path, [self.event("ann_1"), self.event("ann_2")])

            with self.assertRaisesRegex(ValueError, "multiple active heads"):
                AnnotationStore(path).latest_by_image()
            self.assert_append_rejected_without_change(
                path,
                self.event("ann_new", image_id="sha256:new"),
                "multiple active heads",
            )

    def test_long_cycle_is_rejected_explicitly_without_recursion_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "annotations.jsonl"
            event_count = sys.getrecursionlimit() + 100
            events = [
                self.event(
                    f"ann_{index}",
                    supersedes=(
                        f"ann_{event_count - 1}"
                        if index == 0
                        else f"ann_{index - 1}"
                    ),
                )
                for index in range(event_count)
            ]
            self.write_events(path, events)

            with self.assertRaisesRegex(ValueError, "cycle"):
                AnnotationStore(path).events()

    def test_read_rejects_duplicate_ids_and_invalid_existing_schema(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "annotations.jsonl"
            duplicate = self.event("ann_1")
            self.write_events(path, [duplicate, duplicate])
            with self.assertRaisesRegex(ValueError, "duplicate"):
                AnnotationStore(path).events()

            invalid = self.event("ann_2")
            invalid["confirmation_duration_ms"] = 1.0
            self.write_events(path, [invalid])
            with self.assertRaisesRegex(ValueError, "confirmation_duration_ms"):
                AnnotationStore(path).events()

    def test_append_rejects_corrupt_or_truncated_jsonl_without_changing_bytes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "annotations.jsonl"
            for contents in (b'{"annotation_id":', b"42\n"):
                path.write_bytes(contents)
                with self.subTest(contents=contents):
                    self.assert_append_rejected_without_change(
                        path, self.event("ann_new"), "invalid JSONL"
                    )

    def test_append_rejects_complete_last_event_without_terminal_newline(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "annotations.jsonl"
            original = json.dumps(self.event("ann_1")).encode("utf-8")
            path.write_bytes(original)

            self.assert_append_rejected_without_change(
                path,
                self.event("ann_2", image_id="sha256:other"),
                "terminal newline",
            )
            self.assertEqual(original, path.read_bytes())

    def test_append_accepts_empty_file_and_newline_terminated_history(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "annotations.jsonl"
            path.touch()
            store = AnnotationStore(path)

            store.append(self.event("ann_1", image_id="sha256:one"))
            store.append(self.event("ann_2", image_id="sha256:two"))

            self.assertTrue(path.read_bytes().endswith(b"\n"))
            self.assertEqual(2, len(store.events()))

    def test_leaf_symlink_is_rejected_for_reads_and_concurrent_append(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target = root / "target.jsonl"
            target.touch()
            alias = root / "annotations.jsonl"
            alias.symlink_to(target)
            direct_store = AnnotationStore(target)
            alias_store = AnnotationStore(alias)
            barrier = threading.Barrier(2)
            outcomes = []

            def append_after_barrier(label, store, event):
                barrier.wait()
                try:
                    store.append(event)
                except ValueError as exc:
                    outcomes.append((label, "error", str(exc)))
                else:
                    outcomes.append((label, "success", None))

            threads = [
                threading.Thread(
                    target=append_after_barrier,
                    args=(
                        "direct",
                        direct_store,
                        self.event("ann_direct", image_id="sha256:direct"),
                    ),
                ),
                threading.Thread(
                    target=append_after_barrier,
                    args=(
                        "alias",
                        alias_store,
                        self.event("ann_alias", image_id="sha256:alias"),
                    ),
                ),
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            self.assertEqual(
                [("alias", "error")],
                [(label, result) for label, result, _ in outcomes if result == "error"],
            )
            self.assertIn(
                "symlink",
                next(
                    message
                    for label, _, message in outcomes
                    if label == "alias"
                ),
            )
            self.assertEqual(
                ["ann_direct"],
                [event["annotation_id"] for event in direct_store.events()],
            )
            target_bytes = target.read_bytes()
            for operation in (alias_store.events, alias_store.latest_by_image):
                with self.subTest(operation=operation.__name__):
                    with self.assertRaisesRegex(ValueError, "symlink"):
                        operation()
                    self.assertEqual(target_bytes, target.read_bytes())

    def test_parent_symlink_alias_uses_one_lock_before_and_after_file_creation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            real_parent = root / "real"
            real_parent.mkdir()
            alias_parent = root / "alias"
            alias_parent.symlink_to(real_parent, target_is_directory=True)
            alias_store = AnnotationStore(alias_parent / "annotations.jsonl")
            (real_parent / "annotations.jsonl").touch()
            direct_store = AnnotationStore(real_parent / "annotations.jsonl")
            active = 0
            maximum_active = 0
            guard = threading.Lock()

            def slow_append(*arguments):
                nonlocal active, maximum_active
                with guard:
                    active += 1
                    maximum_active = max(maximum_active, active)
                time.sleep(0.03)
                try:
                    append_jsonl_validated(*arguments)
                finally:
                    with guard:
                        active -= 1

            with patch(
                "photocut.data.annotation_store.append_jsonl_validated",
                side_effect=slow_append,
            ):
                threads = [
                    threading.Thread(
                        target=store.append,
                        args=(self.event(f"ann_{index}", image_id=f"sha256:{index}"),),
                    )
                    for index, store in enumerate((alias_store, direct_store), 1)
                ]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()

            self.assertEqual(1, maximum_active)
            self.assertEqual(2, len(direct_store.events()))

    def test_multiple_hard_links_are_rejected_without_changing_bytes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = root / "annotations.jsonl"
            self.write_events(path, [self.event("ann_1")])
            alias = root / "alias.jsonl"
            os.link(path, alias)
            original = path.read_bytes()

            for operation in (
                AnnotationStore(path).events,
                lambda: AnnotationStore(alias).append(
                    self.event("ann_2", image_id="sha256:other")
                ),
            ):
                with self.subTest(operation=operation):
                    with self.assertRaisesRegex(ValueError, "hard link"):
                        operation()
                    self.assertEqual(original, path.read_bytes())

    def test_same_process_lock_serializes_check_and_append_across_instances(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "annotations.jsonl"
            stores = (AnnotationStore(path), AnnotationStore(path))
            events = (
                self.event("ann_1", image_id="sha256:one"),
                self.event("ann_2", image_id="sha256:two"),
            )
            active = 0
            maximum_active = 0
            guard = threading.Lock()

            def slow_append(*arguments):
                nonlocal active, maximum_active
                with guard:
                    active += 1
                    maximum_active = max(maximum_active, active)
                time.sleep(0.03)
                try:
                    append_jsonl_validated(*arguments)
                finally:
                    with guard:
                        active -= 1

            with patch(
                "photocut.data.annotation_store.append_jsonl_validated",
                side_effect=slow_append,
            ):
                threads = [
                    threading.Thread(target=store.append, args=(event,))
                    for store, event in zip(stores, events)
                ]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()

            self.assertEqual(1, maximum_active)
            self.assertEqual(2, len(stores[0].events()))

    def test_concurrent_independent_heads_allow_exactly_one_append(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "annotations.jsonl"
            stores = (AnnotationStore(path), AnnotationStore(path))
            events = (self.event("ann_1"), self.event("ann_2"))
            barrier = threading.Barrier(2)
            errors = []

            def append_after_barrier(store, event):
                barrier.wait()
                try:
                    store.append(event)
                except ValueError as exc:
                    errors.append(str(exc))

            threads = [
                threading.Thread(target=append_after_barrier, args=(store, event))
                for store, event in zip(stores, events)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            self.assertEqual(1, len(errors))
            self.assertIn("multiple active heads", errors[0])
            self.assertEqual(1, len(stores[0].events()))

    def test_history_validation_and_append_share_one_file_transaction(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "annotations.jsonl"
            stores = (AnnotationStore(path), AnnotationStore(path))
            stores[0]._lock = threading.RLock()
            stores[1]._lock = threading.RLock()
            barrier = threading.Barrier(2)
            errors = []
            real_append = append_jsonl_validated

            def coordinated_append(*arguments):
                barrier.wait()
                real_append(*arguments)

            def append(store, event):
                try:
                    store.append(event)
                except ValueError as exc:
                    errors.append(str(exc))

            with patch(
                "photocut.data.annotation_store.append_jsonl_validated",
                side_effect=coordinated_append,
            ):
                threads = [
                    threading.Thread(target=append, args=(store, event))
                    for store, event in zip(
                        stores, (self.event("ann_1"), self.event("ann_2"))
                    )
                ]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()

            self.assertEqual(1, len(errors))
            self.assertIn("multiple active heads", errors[0])
            self.assertEqual(1, len(stores[0].events()))

    def test_symlink_directory_aliases_share_one_physical_path_lock(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            real_directory = root / "real"
            alias_directory = root / "alias"
            real_directory.mkdir()
            alias_directory.symlink_to(real_directory, target_is_directory=True)
            stores = (
                AnnotationStore(real_directory / "annotations.jsonl"),
                AnnotationStore(alias_directory / "annotations.jsonl"),
            )
            events = (self.event("ann_1"), self.event("ann_2"))
            start = threading.Barrier(2)
            guard = threading.Lock()
            active = 0
            maximum_active = 0
            append_calls = 0
            errors = []

            def slow_append(*arguments):
                nonlocal active, maximum_active, append_calls
                with guard:
                    active += 1
                    append_calls += 1
                    maximum_active = max(maximum_active, active)
                time.sleep(0.03)
                try:
                    append_jsonl_validated(*arguments)
                finally:
                    with guard:
                        active -= 1

            def append_after_barrier(store, event):
                start.wait()
                try:
                    store.append(event)
                except BaseException as exc:
                    errors.append(exc)

            with patch(
                "photocut.data.annotation_store.append_jsonl_validated",
                side_effect=slow_append,
            ):
                threads = [
                    threading.Thread(target=append_after_barrier, args=pair)
                    for pair in zip(stores, events)
                ]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()

            self.assertEqual(2, append_calls)
            self.assertEqual(1, maximum_active)
            self.assertEqual(1, len(errors))
            self.assertIsInstance(errors[0], ValueError)
            self.assertIn("multiple active heads", str(errors[0]))
            self.assertEqual(1, len(stores[0].events()))


if __name__ == "__main__":
    unittest.main()
