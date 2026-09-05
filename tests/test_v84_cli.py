"""V8.4 workflow contracts using synthetic photos and injected inference."""
import argparse
import copy
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from photocut import core
from photocut import cli
from photocut.confirmation.controller import _v7_view_model_for_entry, candidate_snapshot
from photocut.confirmation.backend import _entry_uses_normalized_snapshot
from photocut.detection_parameters import DEFAULT_DETECTION_PARAMETERS
from photocut.algorithms.v7.input import LoadedImage


class V84CliTests(unittest.TestCase):
    def loaded(self):
        image = np.zeros((50, 100, 3), dtype=np.uint8)
        return LoadedImage(
            "a" * 64, "uint8", 3, 1, image,
            ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.)),
            ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.)),
            (400, 200), (100, 50), (400, 200),
            ((399./99., 0., 0.), (0., 199./49., 0.), (0., 0., 1.)),
        )

    def runtime(self, status="candidate_requires_confirmation"):
        return SimpleNamespace(model_sha256="sha256:" + "b" * 64, predict=Mock(return_value={
            "status": status,
            "analysis_corners": np.array([[0., 0.], [99., 0.], [99., 49.], [0., 49.]])
            if status == "candidate_requires_confirmation" else None,
        }))

    def entry(self, status="candidate_requires_confirmation"):
        return core.detect_and_save_corners_v84(
            "scan.jpg", "unused", runtime=self.runtime(status), loaded_input=self.loaded(),
        )

    def test_candidate_coordinates_identity_gui_and_json(self):
        entry = self.entry()
        self.assertEqual([[0., 0.], [399., 0.], [399., 199.], [0., 199.]], entry["corners"])
        self.assertFalse(entry["confirmed"])
        self.assertTrue(entry["success"])
        self.assertEqual("8.4", entry["algorithm_version"])
        self.assertTrue(_entry_uses_normalized_snapshot(entry))
        model = _v7_view_model_for_entry(entry, (400, 200))
        self.assertTrue(model.can_confirm)
        self.assertEqual("8.4", candidate_snapshot(model, entry)["algorithm_version"])
        self.assertIsNone(entry["alternate_corners"])
        json.dumps(entry, allow_nan=False)

    def test_invalid_and_blank_have_no_confirmable_coordinates(self):
        for status in ("invalid_geometry", "no_boundary_evidence"):
            with self.subTest(status=status):
                entry = self.entry(status)
                self.assertFalse(entry["success"])
                self.assertEqual([], entry["corners"])
                model = _v7_view_model_for_entry(entry, (400, 200))
                self.assertFalse(model.can_confirm)
                with self.assertRaises(ValueError):
                    model.confirm()

    def test_previous_approved_unadjusted_boundary_is_preserved(self):
        entry = self.entry()
        entry["image_id"] = "sha256:" + "a" * 64
        previous = copy.deepcopy(entry)
        previous.update(confirmed=True, annotation_id="annotation-1", manually_adjusted=False)
        previous["corners"] = previous["boundary_corners"] = [[5, 5], [390, 5], [390, 190], [5, 190]]
        cli._preserve_confirmed_detection(previous, entry, detector="v8.4")
        self.assertTrue(entry["confirmed"])
        self.assertEqual(previous["corners"], entry["corners"])
        # A previous legacy flag alone cannot approve a new prediction.
        previous.pop("annotation_id")
        entry = self.entry()
        cli._preserve_confirmed_detection(previous, entry, detector="v8.4")
        self.assertFalse(entry["confirmed"])

    def test_decode_failure_and_runtime_record_keep_v84_identity(self):
        runtime = self.runtime()
        entry = cli._decode_failure_info(Path("bad.jpg"), "s", "sha256:" + "a" * 64,
            "batch", "run", detector="v8.4", v8_runtime=runtime)
        self.assertEqual("8.4", entry["algorithm_version"])
        self.assertEqual(runtime.model_sha256, cli._result_from_info(entry)["model_sha256"])
        args = argparse.Namespace(detector="v8.4", output="out", shrink_min=25, shrink_max=70)
        record = cli._runtime_for_detection(args, [], DEFAULT_DETECTION_PARAMETERS, v8_runtime=runtime)
        self.assertEqual("8.4", record["algorithm_version"])
        self.assertEqual(runtime.model_sha256, record["parameters"]["model_sha256"])
        self.assertNotIn("auto_cascade_version", record["parameters"])

    def test_default_dispatch_and_explicit_generic_rollback(self):
        with patch.object(sys, "argv", ["photocut", "input", "--detect"]), patch.object(cli, "detect_command") as run:
            cli.main()
        self.assertEqual("v8.4", run.call_args.args[0].detector)
        with patch.object(sys, "argv", ["photocut", "input", "--detect", "--scene-profile", "generic_single"]), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            cli.main()
        with patch.object(sys, "argv", ["photocut", "input", "--detect", "--detector", "auto", "--scene-profile", "generic_single"]), patch.object(cli, "detect_command") as run:
            cli.main()
        self.assertEqual("auto", run.call_args.args[0].detector)

    def test_resume_refuses_detector_or_model_change_and_preserves_finalized_run(self):
        from photocut.data.dataset_store import DatasetStore, BatchLock
        for changed_field in ("detector", "model"):
            with self.subTest(changed_field=changed_field), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp).resolve()
                args = argparse.Namespace(detector="v8.4", output=str(root / "out"), shrink_min=25, shrink_max=70)
                requested = cli._runtime_for_detection(args, [], DEFAULT_DETECTION_PARAMETERS, v8_runtime=self.runtime())
                old = copy.deepcopy(requested)
                if changed_field == "detector":
                    old["algorithm_version"] = "auto-v4"
                    old["parameters"]["detector"] = "auto"
                    old["parameters"]["detector_requested"] = "auto"
                else:
                    old["parameters"]["model_sha256"] = "sha256:" + "c" * 64
                store = DatasetStore(root / "dataset")
                batch = store.start_batch(runtime=old)
                original = batch.run_in_progress.read_bytes()
                with self.assertRaisesRegex(RuntimeError, "refuse resume"):
                    cli._resume_or_start_batch(store, requested, allow_final_repair=True)
                self.assertEqual(original, batch.run_in_progress.read_bytes())
                same, final = cli._resume_or_start_batch(store, old, allow_final_repair=True)
                self.assertEqual(batch.batch_id, same.batch_id)
                self.assertIsNone(final)
                with BatchLock(batch):
                    store.finalize_run(batch)
                original_final = batch.production_run.read_bytes()
                new, final = cli._resume_or_start_batch(store, requested, allow_final_repair=True)
                self.assertNotEqual(batch.batch_id, new.batch_id)
                self.assertIsNone(final)
                self.assertEqual(original_final, batch.production_run.read_bytes())

    def test_finalized_float_evidence_rejects_subpixel_and_model_tampering(self):
        from photocut.confirmation.identity import match_finalized_detection
        entry = self.entry()
        entry.update(batch_id="b", run_id="r", image_id="i", source_id="s")
        entry["algorithm_boundary_corners"][0] = [0.25, 0.25]
        reference = {"batch_id": "b", "run_id": "r"}
        run = {"status": "complete", "batch_id": "b", "run_id": "r", "images": [copy.deepcopy(entry)]}
        evidence = match_finalized_detection(entry, reference, run)
        self.assertEqual("8.4", evidence.algorithm_version)
        changed = copy.deepcopy(entry)
        changed["algorithm_boundary_corners"][0][0] += .01
        self.assertIsNone(match_finalized_detection(changed, reference, run))
        changed = dict(entry, model_sha256="sha256:" + "c" * 64)
        self.assertIsNone(match_finalized_detection(changed, reference, run))
        for points in ([], [[float("nan"), 0.]] * 4, [[True, 0.]] * 4):
            invalid = copy.deepcopy(entry)
            invalid["algorithm_boundary_corners"] = points
            invalid_run = {**run, "images": [copy.deepcopy(invalid)]}
            self.assertIsNone(match_finalized_detection(invalid, reference, invalid_run))

    def test_archived_detection_confirmation_and_crop_record_v84(self):
        import cv2
        from contextlib import ExitStack
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve(); source = root / "input"; source.mkdir(); output = root / "out"
            Image.fromarray(np.arange(15000, dtype=np.uint8).reshape(50, 100, 3)).save(source / "scan.png")
            args = argparse.Namespace(input=str(source), output=str(output), shrink_min=25,
                shrink_max=70, detector="v8.4", v7_mode=None, scene_profile=None,
                no_dataset_archive=False, dataset_root=root / "dataset", skip="", inset=0,
                confirm_ui="opencv", revise_confirmed=[])
            with patch('photocut.algorithms.v8_4.runtime.V84Runtime', return_value=self.runtime()), redirect_stdout(io.StringIO()):
                cli.detect_command(args)
            reference = json.loads((output / ".photocut_batch.json").read_text())
            batch = root / "dataset" / "batches" / reference["batch_id"]
            production = json.loads((batch / "production_run.json").read_text())
            self.assertIn('"algorithm_version": "8.4"', json.dumps(production))
            with ExitStack() as stack:
                stack.enter_context(redirect_stdout(io.StringIO()))
                stack.enter_context(patch.object(cli, "check_gui_and_prompt", return_value=True))
                stack.enter_context(patch.object(cli, "set_native_gui_title"))
                for name in ("namedWindow", "imshow", "setMouseCallback", "destroyAllWindows"):
                    stack.enter_context(patch.object(cv2, name))
                stack.enter_context(patch.object(cv2, "waitKey", return_value=0))
                stack.enter_context(patch.object(cv2, "waitKeyEx", return_value=32))
                cli.confirm_command(args)
            annotations = [json.loads(line) for line in (batch / "annotations.jsonl").read_text().splitlines()]
            self.assertEqual(1, len(annotations))
            self.assertEqual("8.4", annotations[0]["algorithm_version"])
            self.assertEqual("v8.4", annotations[0]["detector_used"])
            with redirect_stdout(io.StringIO()):
                cli.crop_command(args)
            crops = [json.loads(line) for line in (batch / "crops.jsonl").read_text().splitlines()]
            self.assertEqual(1, len(crops))
            entries = json.loads((output / "corners_info.json").read_text())
            self.assertTrue(entries[0]["confirmed"])

    def test_old_confirmed_batch_can_be_revised_after_v84_redetection(self):
        import cv2
        from photocut.confirmation.backend import _OriginAnnotationStore
        from photocut.data.annotation_store import AnnotationStore
        from photocut.data.dataset_reader import load_confirmed_samples
        from contextlib import ExitStack
        def confirm(args, keys=(32,)):
            with ExitStack() as stack:
                stack.enter_context(redirect_stdout(io.StringIO()))
                stack.enter_context(patch.object(cli, "check_gui_and_prompt", return_value=True))
                stack.enter_context(patch.object(cli, "set_native_gui_title"))
                for name in ("namedWindow", "imshow", "setMouseCallback", "destroyAllWindows"):
                    stack.enter_context(patch.object(cv2, name))
                stack.enter_context(patch.object(cv2, "waitKey", return_value=0))
                stack.enter_context(patch.object(cv2, "waitKeyEx", side_effect=iter(keys)))
                cli.confirm_command(args)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve(); source = root / "input"; source.mkdir(); output = root / "out"
            Image.fromarray(np.arange(15000, dtype=np.uint8).reshape(50, 100, 3)).save(source / "scan.png")
            args = argparse.Namespace(input=str(source), output=str(output), shrink_min=25,
                shrink_max=70, detector="v5.2", v7_mode=None, scene_profile=None,
                no_dataset_archive=False, dataset_root=root / "dataset", skip="", inset=0,
                confirm_ui="opencv", revise_confirmed=[])
            old_corners = [[4, 4], [95, 4], [95, 45], [4, 45]]
            with patch.object(core, "detect_corners_detailed", return_value=(old_corners, [1.] * 4, [])), redirect_stdout(io.StringIO()):
                cli.detect_command(args)
            confirm(args)
            old_reference = json.loads((output / ".photocut_batch.json").read_text())
            old_batch = root / "dataset" / "batches" / old_reference["batch_id"]
            old_annotations = (old_batch / "annotations.jsonl").read_bytes()
            old_run = (old_batch / "production_run.json").read_bytes()
            old_event = json.loads(old_annotations.splitlines()[-1])
            self.assertEqual(1, len(load_confirmed_samples(root / "dataset")))
            args.detector = "v8.4"
            with patch('photocut.algorithms.v8_4.runtime.V84Runtime', return_value=self.runtime()), redirect_stdout(io.StringIO()):
                cli.detect_command(args)
            new_reference = json.loads((output / ".photocut_batch.json").read_text())
            new_batch = root / "dataset" / "batches" / new_reference["batch_id"]
            self.assertNotEqual(old_batch, new_batch)
            entry = json.loads((output / "corners_info.json").read_text())[0]
            self.assertEqual(old_corners, entry["boundary_corners"])
            self.assertTrue(entry["confirmed"])
            self.assertEqual("8.4", entry["algorithm_version"])
            self.assertNotEqual(old_corners, entry["algorithm_boundary_corners"])
            self.assertEqual(old_reference["batch_id"], entry["confirmation_origin_batch_id"])
            self.assertEqual(1, len(load_confirmed_samples(root / "dataset")))
            # No duplicated truth is introduced, and inherited crops cite the old head.
            with redirect_stdout(io.StringIO()):
                cli.crop_command(args)
            initial_crop = json.loads((new_batch / "crops.jsonl").read_text().splitlines()[0])
            self.assertEqual(old_event["annotation_id"], initial_crop["annotation_id"])
            self.assertEqual(old_corners, initial_crop["boundary_corners"])
            current_store = AnnotationStore(new_batch / "annotations.jsonl")
            for invalid_origin in ("../escape", "20260101-000000-deadbeef", "/tmp/escape"):
                bad = dict(entry, confirmation_origin_batch_id=invalid_origin)
                with self.assertRaises((ValueError, OSError)):
                    _OriginAnnotationStore(current_store, [bad])
            bad = dict(entry, image_id="sha256:" + "d" * 64)
            with self.assertRaisesRegex(ValueError, "matching image annotation"):
                _OriginAnnotationStore(current_store, [bad])
            self.assertFalse((root / "dataset" / "batches" / "20260101-000000-deadbeef").exists())
            self.assertFalse((root / "dataset" / "escape").exists())
            args.revise_confirmed = ["scan.png"]
            confirm(args, keys=(ord("1"), ord("d"), 32))
            events = [json.loads(line) for line in (old_batch / "annotations.jsonl").read_text().splitlines()]
            self.assertEqual(2, len(events))
            self.assertEqual(old_event, events[0])
            self.assertEqual(old_event["annotation_id"], events[1]["supersedes_annotation_id"])
            self.assertEqual("8.4", events[1]["algorithm_version"])
            self.assertTrue((old_batch / "annotations.jsonl").read_bytes().startswith(old_annotations))
            self.assertFalse((new_batch / "annotations.jsonl").exists())
            samples = load_confirmed_samples(root / "dataset")
            self.assertEqual(1, len(samples))
            self.assertEqual(events[1]["annotation_id"], samples[0].annotation_id)
            self.assertEqual(events[1]["boundary_corners"], [list(point) for point in samples[0].boundary_corners])
            self.assertNotEqual(old_corners, events[1]["boundary_corners"])
            # Reopening repairs a stale mutable view from the same origin head.
            stale = json.loads((output / "corners_info.json").read_text())
            stale[0].update(confirmed=False, annotation_id=old_event["annotation_id"], boundary_corners=old_corners, corners=old_corners)
            (output / "corners_info.json").write_text(json.dumps(stale))
            args.revise_confirmed = []
            confirm(args)
            repaired = json.loads((output / "corners_info.json").read_text())[0]
            self.assertTrue(repaired["confirmed"])
            self.assertEqual(events[1]["annotation_id"], repaired["annotation_id"])
            self.assertEqual(events[1]["boundary_corners"], repaired["boundary_corners"])
            with redirect_stdout(io.StringIO()):
                cli.crop_command(args)
            crops = [json.loads(line) for line in (new_batch / "crops.jsonl").read_text().splitlines()]
            self.assertEqual(events[1]["annotation_id"], crops[-1]["annotation_id"])
            self.assertEqual(events[1]["boundary_corners"], crops[-1]["boundary_corners"])
            all_ids = [json.loads(line)["annotation_id"] for path in (root / "dataset" / "batches").glob("*/annotations.jsonl") for line in path.read_text().splitlines()]
            self.assertEqual(len(all_ids), len(set(all_ids)))
            self.assertEqual(2, len((old_batch / "annotations.jsonl").read_text().splitlines()))
            self.assertEqual(old_run, (old_batch / "production_run.json").read_bytes())

    def test_one_runtime_for_batch_and_unconfirmed_crop_is_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve(); source = root / "input"; source.mkdir(); output = root / "out"
            for i in range(2):
                Image.fromarray(np.arange(15000, dtype=np.uint8).reshape(50, 100, 3)).save(source / f"{i}.png")
            (source / "broken.jpg").write_bytes(b"invalid")
            args = argparse.Namespace(input=str(source), output=str(output), shrink_min=25,
                shrink_max=70, detector="v8.4", v7_mode=None, scene_profile=None,
                no_dataset_archive=True)
            runtime = self.runtime()
            with patch("photocut.algorithms.v8_4.runtime.V84Runtime", return_value=runtime) as factory, \
                    patch.object(cli, "_v8_runtime_from_args") as old_v8, \
                    patch.object(cli, "detect_and_save_corners_auto") as old_auto, redirect_stdout(io.StringIO()):
                cli.detect_command(args)
            factory.assert_called_once_with()
            self.assertEqual(2, runtime.predict.call_count)
            old_v8.assert_not_called(); old_auto.assert_not_called()
            entries = json.loads((output / "corners_info.json").read_text())
            self.assertEqual(3, len(entries))
            self.assertTrue(all(e["algorithm_version"] == "8.4" and not e["confirmed"] for e in entries))
            args.inset = 0; args.skip = ""
            with patch.object(cli, "crop_image") as crop, redirect_stdout(io.StringIO()):
                cli.crop_command(args)
            crop.assert_not_called()


if __name__ == "__main__":
    unittest.main()
