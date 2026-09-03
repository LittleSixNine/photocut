import contextlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from photocut import cli as photocut_cli
from photocut.data.dataset_store import load_jsonl


class DetectDatasetIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.root = Path(self.temp_dir.name)
        self.input_dir = self.root / "input"
        self.output_dir = self.root / "output"
        self.dataset_root = self.root / "project" / ".photocut" / "internal" / "datasets"
        self.input_dir.mkdir(parents=True)
        self.output_dir.mkdir()
        for filename, value in (("first.jpg", 40), ("second.jpg", 180)):
            image = np.full((20, 20, 3), value, dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(self.input_dir / filename), image))

    def tearDown(self):
        self.temp_dir.cleanup()

    def args(self, **changes):
        values = {
            "input": str(self.input_dir),
            "output": str(self.output_dir),
            "shrink_min": 25,
            "shrink_max": 70,
            "dataset_root": str(self.dataset_root),
            "no_dataset_archive": False,
        }
        values.update(changes)
        return Namespace(**values)

    @staticmethod
    def detailed(*_args, **_kwargs):
        return (
            [[2, 3], [17, 3], [17, 16], [2, 16]],
            [0.91, 0.82, 0.73, 0.64],
            [
                {"corner": index, "score": score}
                for index, score in enumerate((91, 82, 73, 64))
            ],
        )

    def test_detect_archives_two_real_jpegs_and_finalizes_one_run(self):
        with patch("photocut.core.detect_corners_detailed", side_effect=self.detailed):
            photocut_cli.detect_command(self.args())

        corners_info = json.loads((self.output_dir / "corners_info.json").read_text())
        reference = json.loads((self.output_dir / ".photocut_batch.json").read_text())
        batch_dir = self.dataset_root / "batches" / reference["batch_id"]
        production_run = json.loads((batch_dir / "production_run.json").read_text())
        self.assertEqual(2, len(corners_info))
        self.assertTrue(all(entry["image_id"].startswith("sha256:") for entry in corners_info))
        self.assertTrue(reference["batch_id"])
        self.assertTrue((batch_dir / "manifest.json").exists())
        self.assertFalse((batch_dir / "production_run.inprogress.json").exists())
        self.assertEqual(2, len(production_run["images"]))
        self.assertTrue(all(result["algorithm_boundary_corners"] for result in production_run["images"]))
        self.assertTrue(
            all(
                result["confidences"] == [0.91, 0.82, 0.73, 0.64]
                for result in production_run["images"]
            )
        )
        self.assertEqual(
            [
                {"corner": index, "score": score}
                for index, score in enumerate((91, 82, 73, 64))
            ],
            production_run["images"][0]["legacy_info"]["detection_debug"],
        )
        self.assertTrue(all("detection_duration_ms" in result for result in production_run["images"]))
        self.assertEqual(
            {
                "command": "detect",
                "output_identity": str(self.output_dir.resolve()),
                "shrink_min": 25,
                "shrink_max": 70,
                "effective_detector_parameters": photocut_cli.DEFAULT_DETECTION_PARAMETERS.to_dict(),
                "input_images": photocut_cli._source_records(
                    [self.input_dir / "first.jpg", self.input_dir / "second.jpg"]
                ),
            },
            production_run["runtime"]["parameters"],
        )

    def test_detect_then_crop_uses_an_external_absolute_dataset_root(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd().parent) as external_dir:
            external_dataset = Path(external_dir) / "datasets"
            with patch("photocut.core.detect_corners_detailed", side_effect=self.detailed):
                photocut_cli.detect_command(self.args(dataset_root=str(external_dataset)))

            reference = json.loads((self.output_dir / ".photocut_batch.json").read_text())
            self.assertEqual(str(external_dataset.resolve()), reference["dataset_root"])
            corners_path = self.output_dir / "corners_info.json"
            corners_info = json.loads(corners_path.read_text())
            corners_info[0]["annotation_id"] = "ann_external_dataset"
            corners_info[0]["confirmed"] = True
            corners_path.write_text(json.dumps(corners_info), encoding="utf-8")
            cropped = self.output_dir / "裁切成品" / "first.jpg"

            def successful_crop(*_args):
                cropped.parent.mkdir(exist_ok=True)
                cropped.write_bytes(b"crop")
                return True, str(cropped)

            with patch("photocut.cli.crop_image", side_effect=successful_crop):
                photocut_cli.crop_command(
                    Namespace(
                        input=str(self.input_dir), output=str(self.output_dir), skip="", inset=5.0
                    )
                )

            events = load_jsonl(external_dataset / "batches" / reference["batch_id"] / "crops.jsonl")
            self.assertEqual(1, len(events))
            self.assertEqual("ann_external_dataset", events[0]["annotation_id"])

    def test_detect_records_and_passes_custom_effective_parameters(self):
        params = photocut_cli.effective_detection_parameters(211)
        with patch("photocut.core.detect_corners_detailed", side_effect=self.detailed) as detect:
            photocut_cli.detect_command(self.args(threshold=211))

        production_run = json.loads(next((self.dataset_root / "batches").glob("*/production_run.json")).read_text())
        self.assertEqual(
            params.to_dict(),
            production_run["runtime"]["parameters"]["effective_detector_parameters"],
        )
        self.assertEqual(params, detect.call_args.kwargs["params"])

    def test_preflight_failure_aborts_before_detection_or_legacy_output(self):
        with patch.object(
            photocut_cli.DatasetStore, "preflight_sources", side_effect=OSError("no space")
        ), patch("photocut.core.detect_corners_detailed") as detect:
            with self.assertRaisesRegex(OSError, "no space"):
                photocut_cli.detect_command(self.args())

        detect.assert_not_called()
        self.assertFalse((self.output_dir / "corners_info.json").exists())

    def test_explicit_no_archive_keeps_legacy_output_and_warns(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout), patch(
            "photocut.core.detect_corners_detailed", side_effect=self.detailed
        ):
            photocut_cli.detect_command(self.args(no_dataset_archive=True))

        self.assertTrue((self.output_dir / "corners_info.json").exists())
        self.assertFalse((self.output_dir / ".photocut_batch.json").exists())
        self.assertFalse(self.dataset_root.exists())
        self.assertIn("本批次不会进入长期数据集", stdout.getvalue())

    def test_clean_inprogress_run_resumes_same_batch_and_skips_recorded_image(self):
        calls = []

        def interrupted(*_args, **_kwargs):
            calls.append("detected")
            if len(calls) == 2:
                raise KeyboardInterrupt("simulated interruption")
            return self.detailed()

        with patch("photocut.core.detect_corners_detailed", side_effect=interrupted):
            with self.assertRaises(KeyboardInterrupt):
                photocut_cli.detect_command(self.args())

        batch_dirs = list((self.dataset_root / "batches").iterdir())
        self.assertEqual(1, len(batch_dirs))
        self.assertTrue((batch_dirs[0] / "production_run.inprogress.json").exists())

        with patch("photocut.core.detect_corners_detailed", side_effect=self.detailed) as detect:
            photocut_cli.detect_command(self.args())

        reference = json.loads((self.output_dir / ".photocut_batch.json").read_text())
        self.assertEqual(batch_dirs[0].name, reference["batch_id"])
        self.assertEqual(1, detect.call_count)
        run = json.loads((batch_dirs[0] / "production_run.json").read_text())
        self.assertEqual(2, len(run["images"]))

    def test_input_order_or_content_mismatch_refuses_inprogress_resume(self):
        with patch("photocut.core.detect_corners_detailed", side_effect=KeyboardInterrupt("stop")):
            with self.assertRaises(KeyboardInterrupt):
                photocut_cli.detect_command(self.args())

        original_batches = list((self.dataset_root / "batches").iterdir())
        replacement = np.full((20, 20, 3), 99, dtype=np.uint8)
        self.assertTrue(cv2.imwrite(str(self.input_dir / "second.jpg"), replacement))
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            photocut_cli.detect_command(self.args())

        self.assertEqual(original_batches, list((self.dataset_root / "batches").iterdir()))

    def test_stale_lock_reports_manual_recovery_without_deleting_lock(self):
        with patch("photocut.core.detect_corners_detailed", side_effect=KeyboardInterrupt("stop")):
            with self.assertRaises(KeyboardInterrupt):
                photocut_cli.detect_command(self.args())

        batch_dir = next((self.dataset_root / "batches").iterdir())
        lock = batch_dir / "batch.lock"
        lock.write_text('{"pid":0}\n', encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "manual recovery"):
            photocut_cli.detect_command(self.args())
        self.assertTrue(lock.exists())

    def test_resume_requires_exact_runtime_and_does_not_mix_same_output(self):
        with patch("photocut.core.detect_corners_detailed", side_effect=KeyboardInterrupt("stop")):
            with self.assertRaises(KeyboardInterrupt):
                photocut_cli.detect_command(self.args())

        batches = list((self.dataset_root / "batches").iterdir())
        with patch("photocut.core.detect_corners_detailed") as detect:
            with self.assertRaisesRegex(RuntimeError, "runtime does not match"):
                photocut_cli.detect_command(self.args(shrink_min=26))
        detect.assert_not_called()
        self.assertEqual(batches, list((self.dataset_root / "batches").iterdir()))

    def test_unrelated_output_inprogress_batch_does_not_block(self):
        other_output = self.root / "other-output"
        other_output.mkdir()
        with patch("photocut.core.detect_corners_detailed", side_effect=KeyboardInterrupt("stop")):
            with self.assertRaises(KeyboardInterrupt):
                photocut_cli.detect_command(self.args(output=str(other_output)))

        with patch("photocut.core.detect_corners_detailed", side_effect=self.detailed):
            photocut_cli.detect_command(self.args())

        self.assertTrue((self.output_dir / ".photocut_batch.json").exists())
        self.assertEqual(2, len(list((self.dataset_root / "batches").iterdir())))

    def test_duplicate_content_has_one_manifest_image_and_two_source_results(self):
        (self.input_dir / "second.jpg").write_bytes((self.input_dir / "first.jpg").read_bytes())

        with patch("photocut.core.detect_corners_detailed", side_effect=self.detailed) as detect:
            photocut_cli.detect_command(self.args())

        reference = json.loads((self.output_dir / ".photocut_batch.json").read_text())
        batch_dir = self.dataset_root / "batches" / reference["batch_id"]
        manifest = json.loads((batch_dir / "manifest.json").read_text())
        run = json.loads((batch_dir / "production_run.json").read_text())
        self.assertEqual(1, len(manifest["images"]))
        self.assertEqual(
            {"first.jpg", "second.jpg"},
            {
                source["source_filename"]
                for source in manifest["images"][0]["sources"]
            },
        )
        self.assertEqual(2, len(run["images"]))
        self.assertEqual(2, len({result["source_id"] for result in run["images"]}))
        self.assertEqual(1, len({result["image_id"] for result in run["images"]}))
        self.assertEqual(2, detect.call_count)

    def test_resume_rebuilds_legacy_entry_from_authoritative_run(self):
        previous = {
            "filename": "first.jpg",
            "confirmed": True,
            "manually_adjusted": True,
            "boundary_corners": [[4, 4], [15, 4], [15, 15], [4, 15]],
            "corners": [[4, 4], [15, 4], [15, 15], [4, 15]],
            "preview_corners": [[4, 4], [15, 4], [15, 15], [4, 15]],
            "manual_corners": [[4, 4], [15, 4], [15, 15], [4, 15]],
            "manual_preview_corners": [[4, 4], [15, 4], [15, 15], [4, 15]],
            "confirm_timestamp": "2026-07-24T00:00:00+08:00",
        }
        (self.output_dir / "corners_info.json").write_text(
            json.dumps([previous]), encoding="utf-8"
        )
        with patch("photocut.core.detect_corners_detailed", side_effect=self.detailed), patch(
            "photocut.cli.save_corners_info", side_effect=OSError("corners write failed")
        ):
            with self.assertRaisesRegex(OSError, "corners write failed"):
                photocut_cli.detect_command(self.args())

        batch_dir = next((self.dataset_root / "batches").iterdir())
        stored = json.loads(
            (batch_dir / "production_run.inprogress.json").read_text()
        )["images"][0]["legacy_info"]
        stale = dict(previous, boundary_corners=[[0, 0]] * 4, confirmed=False)
        (self.output_dir / "corners_info.json").write_text(json.dumps([stale]), encoding="utf-8")
        with patch("photocut.core.detect_corners_detailed", side_effect=self.detailed) as detect:
            photocut_cli.detect_command(self.args())

        restored = json.loads((self.output_dir / "corners_info.json").read_text())[0]
        self.assertEqual(stored, restored)
        self.assertTrue(restored["confirmed"])
        self.assertTrue(restored["manually_adjusted"])
        self.assertEqual(1, detect.call_count)

    def test_missing_no_archive_attribute_defaults_to_archival_enabled(self):
        args = self.args()
        del args.no_dataset_archive
        with patch("photocut.core.detect_corners_detailed", side_effect=self.detailed):
            photocut_cli.detect_command(args)
        self.assertTrue((self.output_dir / ".photocut_batch.json").exists())

    def test_undecodable_jpeg_is_archived_registered_and_recorded_as_failure(self):
        shutil.rmtree(self.input_dir)
        self.input_dir.mkdir()
        source = self.input_dir / "broken.jpg"
        source.write_bytes(b"\xff\xd8\xffnot-a-decodable-jpeg")

        with patch("photocut.core.detect_corners_detailed") as detect:
            photocut_cli.detect_command(self.args())

        detect.assert_not_called()
        reference = json.loads((self.output_dir / ".photocut_batch.json").read_text())
        batch_dir = self.dataset_root / "batches" / reference["batch_id"]
        manifest = json.loads((batch_dir / "manifest.json").read_text())
        run = json.loads((batch_dir / "production_run.json").read_text())
        self.assertEqual(1, len(manifest["images"]))
        self.assertIsNone(manifest["images"][0]["width"])
        self.assertIsNone(manifest["images"][0]["height"])
        self.assertTrue((self.dataset_root / manifest["images"][0]["object_path"]).exists())
        self.assertFalse(run["images"][0]["success"])
        self.assertEqual(manifest["images"][0]["image_id"], run["images"][0]["image_id"])
        self.assertEqual(
            manifest["images"][0]["sources"][0]["source_id"],
            run["images"][0]["source_id"],
        )

    def test_missing_reference_repairs_matching_final_without_redetection_or_new_batch(self):
        real_reference_write = photocut_cli.atomic_write_json

        def fail_reference(path, value):
            if Path(path).name == ".photocut_batch.json":
                raise OSError("reference write failed")
            return real_reference_write(path, value)

        with patch("photocut.core.detect_corners_detailed", side_effect=self.detailed), patch(
            "photocut.cli.atomic_write_json", side_effect=fail_reference
        ):
            with self.assertRaisesRegex(OSError, "reference write failed"):
                photocut_cli.detect_command(self.args())

        batches = list((self.dataset_root / "batches").iterdir())
        (self.output_dir / "corners_info.json").unlink()
        with patch("photocut.core.detect_corners_detailed") as detect:
            photocut_cli.detect_command(self.args())

        detect.assert_not_called()
        self.assertEqual(batches, list((self.dataset_root / "batches").iterdir()))
        self.assertTrue((self.output_dir / ".photocut_batch.json").exists())
        self.assertEqual(
            2,
            len(json.loads((self.output_dir / "corners_info.json").read_text())),
        )

    def _finalized_batch_without_reference(self):
        with patch("photocut.core.detect_corners_detailed", side_effect=self.detailed):
            photocut_cli.detect_command(self.args())
        reference_path = self.output_dir / ".photocut_batch.json"
        reference = json.loads(reference_path.read_text())
        reference_path.unlink()
        corners_path = self.output_dir / "corners_info.json"
        corners_path.write_text('[{"filename":"sentinel.jpg"}]\n', encoding="utf-8")
        return self.dataset_root / "batches" / reference["batch_id"], corners_path

    def _assert_unsafe_finalized_candidate_is_rejected(self, batch_dir, corners_path):
        original_corners = corners_path.read_bytes()
        with patch("photocut.core.detect_corners_detailed") as detect:
            with self.assertRaisesRegex(RuntimeError, "invalid batch"):
                photocut_cli.detect_command(self.args())
        detect.assert_not_called()
        self.assertFalse((self.output_dir / ".photocut_batch.json").exists())
        self.assertEqual(original_corners, corners_path.read_bytes())
        self.assertTrue(batch_dir.parent.exists())

    def test_finalized_repair_rejects_symlinked_batch_directory(self):
        batch_dir, corners_path = self._finalized_batch_without_reference()
        outside = self.root / "outside-batch"
        batch_dir.rename(outside)
        before = {
            path.relative_to(outside).as_posix(): path.read_bytes()
            for path in outside.rglob("*")
            if path.is_file()
        }
        batch_dir.symlink_to(outside, target_is_directory=True)

        self._assert_unsafe_finalized_candidate_is_rejected(batch_dir, corners_path)
        self.assertEqual(
            before,
            {
                path.relative_to(outside).as_posix(): path.read_bytes()
                for path in outside.rglob("*")
                if path.is_file()
            },
        )

    def test_finalized_repair_rejects_symlinked_batch_metadata(self):
        for filename in ("manifest.json", "production_run.json"):
            with self.subTest(filename=filename):
                batch_dir, corners_path = self._finalized_batch_without_reference()
                metadata = batch_dir / filename
                outside = self.root / f"outside-{filename}"
                metadata.rename(outside)
                before = outside.read_bytes()
                metadata.symlink_to(outside)

                self._assert_unsafe_finalized_candidate_is_rejected(
                    batch_dir, corners_path
                )
                self.assertEqual(before, outside.read_bytes())

                shutil.rmtree(batch_dir)
                corners_path.unlink(missing_ok=True)

    def test_finalized_repair_rejects_dangling_inprogress_symlink(self):
        batch_dir, corners_path = self._finalized_batch_without_reference()
        outside = self.root / "outside-inprogress.json"
        (batch_dir / "production_run.inprogress.json").symlink_to(outside)

        self._assert_unsafe_finalized_candidate_is_rejected(batch_dir, corners_path)
        self.assertFalse(outside.exists())

    def test_finalized_repair_skips_unrelated_hidden_and_staging_entries(self):
        batch_dir, _ = self._finalized_batch_without_reference()
        outside = self.root / "outside-unrelated"
        outside.mkdir()
        before = list(outside.iterdir())
        for name in (".hidden", ".candidate.staging", "notes"):
            (batch_dir.parent / name).symlink_to(outside, target_is_directory=True)

        with patch("photocut.core.detect_corners_detailed") as detect:
            photocut_cli.detect_command(self.args())

        detect.assert_not_called()
        self.assertTrue((self.output_dir / ".photocut_batch.json").exists())
        self.assertEqual(before, list(outside.iterdir()))

    def test_existing_reference_makes_normal_redetection_start_a_new_batch(self):
        with patch("photocut.core.detect_corners_detailed", side_effect=self.detailed):
            photocut_cli.detect_command(self.args())
        first_reference = json.loads((self.output_dir / ".photocut_batch.json").read_text())

        with patch("photocut.core.detect_corners_detailed", side_effect=self.detailed) as detect:
            photocut_cli.detect_command(self.args())

        second_reference = json.loads((self.output_dir / ".photocut_batch.json").read_text())
        self.assertNotEqual(first_reference["batch_id"], second_reference["batch_id"])
        self.assertEqual(2, detect.call_count)
        self.assertEqual(2, len(list((self.dataset_root / "batches").iterdir())))

    def test_multiple_matching_finalized_batches_refuse_reference_repair(self):
        with patch("photocut.core.detect_corners_detailed", side_effect=self.detailed):
            photocut_cli.detect_command(self.args())
        reference = json.loads((self.output_dir / ".photocut_batch.json").read_text())
        original = self.dataset_root / "batches" / reference["batch_id"]
        duplicate_id = reference["batch_id"][:-8] + "deadbeef"
        duplicate = self.dataset_root / "batches" / duplicate_id
        shutil.copytree(original, duplicate)
        for name in ("manifest.json", "production_run.json"):
            path = duplicate / name
            value = json.loads(path.read_text())
            value["batch_id"] = duplicate_id
            if "run_id" in value:
                value["run_id"] = f"run_{duplicate_id}"
            for result in value.get("images", []):
                result["batch_id"] = duplicate_id
                result["run_id"] = f"run_{duplicate_id}"
                if isinstance(result.get("legacy_info"), dict):
                    result["legacy_info"]["batch_id"] = duplicate_id
                    result["legacy_info"]["run_id"] = f"run_{duplicate_id}"
            path.write_text(json.dumps(value), encoding="utf-8")
        (self.output_dir / ".photocut_batch.json").unlink()

        with patch("photocut.core.detect_corners_detailed") as detect:
            with self.assertRaisesRegex(RuntimeError, "multiple matching finalized"):
                photocut_cli.detect_command(self.args())
        detect.assert_not_called()

    def test_cli_accepts_no_archive_before_and_after_detect_forms(self):
        forms = (
            [str(self.input_dir), "--no-dataset-archive", "detect"],
            [str(self.input_dir), "detect", "--no-dataset-archive"],
            [str(self.input_dir), "--no-dataset-archive", "--detect"],
            [str(self.input_dir), "--detect", "--no-dataset-archive"],
        )
        for argv in forms:
            with self.subTest(argv=argv), patch.object(
                sys, "argv", ["photocut_cli.py", *argv]
            ), patch("photocut.cli.detect_command") as command:
                photocut_cli.main()
            self.assertIs(command.call_args.args[0].no_dataset_archive, True)


if __name__ == "__main__":
    unittest.main()
