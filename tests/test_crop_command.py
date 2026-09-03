import contextlib
import io
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np

import photocut
from photocut import cli as photocut_cli

class CropCommandTests(unittest.TestCase):
    def test_crop_uses_computed_inset_and_records_both_coordinate_sets(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            output_dir.mkdir()
            (input_dir / "scan.jpg").write_bytes(b"fixture")
            boundary = [[10, 10], [110, 10], [110, 70], [10, 70]]
            info_path = output_dir / "corners_info.json"
            info_path.write_text(
                json.dumps(
                    [
                        {
                            "filename": "scan.jpg",
                            "boundary_corners": boundary,
                            "corners": boundary,
                            "original_size": [120, 80],
                            "success": True,
                            "confirmed": True,
                        }
                    ]
                ),
                encoding="utf-8",
            )
            args = Namespace(
                input=str(input_dir), output=str(output_dir), skip="", inset=5.0
            )

            with patch("photocut.cli.crop_image", return_value=(True, "done.jpg")) as crop:
                photocut_cli.crop_command(args)

            self.assertEqual(
                [[15.0, 15.0], [105.0, 15.0], [105.0, 65.0], [15.0, 65.0]],
                crop.call_args.args[1],
            )
            saved = json.loads(info_path.read_text(encoding="utf-8"))
            self.assertEqual(boundary, saved[0]["boundary_corners"])
            self.assertEqual(5.0, saved[0]["inset"]["distance_px"])
            self.assertEqual("parallel_edge_offset", saved[0]["inset"]["method"])

    def test_manual_confirmation_boundary_is_used_for_crop_without_overwriting_algorithm(self):
        algorithm = [[10, 10], [110, 10], [110, 70], [10, 70]]
        adjusted = [[20, 20], [100, 20], [100, 60], [20, 60]]
        entry = {
            "filename": "scan.jpg",
            "algorithm_boundary_corners": algorithm,
            "algorithm_corners": algorithm,
            "boundary_corners": algorithm,
            "corners": algorithm,
            "original_size": [120, 80],
            "success": True,
            "confirmed": True,
        }
        photocut_cli.save_manual_boundary(entry, adjusted, adjusted)

        self.assertEqual(algorithm, entry["algorithm_boundary_corners"])
        self.assertEqual(adjusted, entry["boundary_corners"])
        self.assertEqual(adjusted, entry["corners"])

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            output_dir.mkdir()
            (input_dir / "scan.jpg").write_bytes(b"fixture")
            (output_dir / "corners_info.json").write_text(
                json.dumps([entry]), encoding="utf-8"
            )
            args = Namespace(
                input=str(input_dir), output=str(output_dir), skip="", inset=5.0
            )

            with patch("photocut.cli.crop_image", return_value=(True, "done.jpg")) as crop:
                photocut_cli.crop_command(args)

        self.assertEqual(
            [[25.0, 25.0], [95.0, 25.0], [95.0, 55.0], [25.0, 55.0]],
            crop.call_args.args[1],
        )

    def test_redetect_preserves_confirmed_manual_boundary_for_crop(self):
        automatic_a = [[10, 10], [110, 10], [110, 70], [10, 70]]
        manual = [[20, 20], [100, 20], [100, 60], [20, 60]]
        automatic_b = [[15, 15], [105, 15], [105, 65], [15, 65]]
        manual_preview = [[2, 2], [10, 2], [10, 6], [2, 6]]

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            output_dir.mkdir()
            (input_dir / "scan.jpg").write_bytes(b"fixture")
            info_path = output_dir / "corners_info.json"
            info_path.write_text(
                json.dumps(
                    [
                        {
                            "filename": "scan.jpg",
                            "algorithm_boundary_corners": automatic_a,
                            "algorithm_corners": automatic_a,
                            "algorithm_preview_corners": automatic_a,
                            "boundary_corners": manual,
                            "corners": manual,
                            "preview_corners": manual_preview,
                            "manual_corners": manual,
                            "manual_preview_corners": manual_preview,
                            "original_size": [120, 80],
                            "success": True,
                            "confirmed": True,
                            "manually_adjusted": True,
                        }
                    ]
                ),
                encoding="utf-8",
            )
            redetection = {
                "filename": "scan.jpg",
                "algorithm_boundary_corners": automatic_b,
                "algorithm_corners": automatic_b,
                "algorithm_preview_corners": automatic_b,
                "boundary_corners": automatic_b,
                "corners": automatic_b,
                "preview_corners": automatic_b,
                "manual_corners": None,
                "manual_preview_corners": None,
                "original_size": [120, 80],
                "success": True,
                "confirmed": False,
                "manually_adjusted": False,
            }
            detect_args = Namespace(
                input=str(input_dir),
                output=str(output_dir),
                shrink_min=25,
                shrink_max=70,
                no_dataset_archive=True,
            )

            with patch("photocut.cli.detect_and_save_corners", return_value=redetection):
                photocut_cli.detect_command(detect_args)

            saved = json.loads(info_path.read_text(encoding="utf-8"))[0]
            self.assertEqual(automatic_b, saved["algorithm_boundary_corners"])
            self.assertEqual(manual, saved["boundary_corners"])
            self.assertEqual(manual, saved["corners"])
            self.assertEqual(manual_preview, saved["preview_corners"])
            self.assertEqual(manual, saved["manual_corners"])
            self.assertEqual(manual_preview, saved["manual_preview_corners"])
            self.assertTrue(saved["confirmed"])
            self.assertTrue(saved["manually_adjusted"])

            crop_args = Namespace(
                input=str(input_dir), output=str(output_dir), skip="", inset=5.0
            )
            with patch("photocut.cli.crop_image", return_value=(True, "done.jpg")) as crop:
                photocut_cli.crop_command(crop_args)

        self.assertEqual(
            [[25.0, 25.0], [95.0, 25.0], [95.0, 55.0], [25.0, 55.0]],
            crop.call_args.args[1],
        )

    def test_redetect_replaces_unconfirmed_automatic_boundary(self):
        automatic_a = [[10, 10], [110, 10], [110, 70], [10, 70]]
        automatic_b = [[15, 15], [105, 15], [105, 65], [15, 65]]

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            output_dir.mkdir()
            (input_dir / "scan.jpg").write_bytes(b"fixture")
            info_path = output_dir / "corners_info.json"
            info_path.write_text(
                json.dumps(
                    [
                        {
                            "filename": "scan.jpg",
                            "boundary_corners": automatic_a,
                            "corners": automatic_a,
                            "original_size": [120, 80],
                            "success": True,
                            "confirmed": False,
                            "manually_adjusted": False,
                        }
                    ]
                ),
                encoding="utf-8",
            )
            redetection = {
                "filename": "scan.jpg",
                "algorithm_boundary_corners": automatic_b,
                "algorithm_corners": automatic_b,
                "algorithm_preview_corners": automatic_b,
                "boundary_corners": automatic_b,
                "corners": automatic_b,
                "preview_corners": automatic_b,
                "manual_corners": None,
                "manual_preview_corners": None,
                "original_size": [120, 80],
                "success": True,
                "confirmed": False,
                "manually_adjusted": False,
            }
            args = Namespace(
                input=str(input_dir),
                output=str(output_dir),
                shrink_min=25,
                shrink_max=70,
                no_dataset_archive=True,
            )

            with patch("photocut.cli.detect_and_save_corners", return_value=redetection):
                photocut_cli.detect_command(args)

            saved = json.loads(info_path.read_text(encoding="utf-8"))[0]

        self.assertEqual(automatic_b, saved["boundary_corners"])
        self.assertEqual(automatic_b, saved["corners"])
        self.assertFalse(saved["confirmed"])

    def test_reset_confirmation_boundary_restores_algorithm_boundary(self):
        algorithm = [[10, 10], [110, 10], [110, 70], [10, 70]]
        adjusted = [[20, 20], [100, 20], [100, 60], [20, 60]]
        entry = {
            "algorithm_boundary_corners": algorithm,
            "algorithm_corners": algorithm,
            "algorithm_preview_corners": algorithm,
            "boundary_corners": adjusted,
            "corners": adjusted,
            "preview_corners": adjusted,
            "manual_corners": adjusted,
            "manual_preview_corners": adjusted,
        }

        photocut_cli.reset_confirmation_boundary(entry)

        self.assertEqual(algorithm, entry["algorithm_boundary_corners"])
        self.assertEqual(algorithm, entry["boundary_corners"])
        self.assertEqual(algorithm, entry["corners"])
        self.assertEqual(algorithm, entry["preview_corners"])
        self.assertIsNone(entry["manual_corners"])
        self.assertIsNone(entry["manual_preview_corners"])

    def test_crop_cli_accepts_default_and_positive_inset_for_both_entry_points(self):
        cases = [
            (["input", "--crop"], 50.0),
            (["input", "--crop", "--inset", "7.5"], 7.5),
            (["input", "crop"], 50.0),
            (["input", "crop", "--inset", "7.5"], 7.5),
        ]
        for argv, expected_inset in cases:
            with self.subTest(argv=argv), patch.object(
                sys, "argv", ["photocut_cli.py", *argv]
            ), patch("photocut.cli.crop_command") as crop:
                photocut_cli.main()
                self.assertEqual(expected_inset, crop.call_args.args[0].inset)

    def test_crop_cli_rejects_negative_inset_for_both_entry_points(self):
        for argv in (
            ["input", "--crop", "--inset", "-1"],
            ["input", "crop", "--inset", "-1"],
        ):
            with self.subTest(argv=argv), patch.object(
                sys, "argv", ["photocut_cli.py", *argv]
            ), contextlib.redirect_stderr(io.StringIO()) as stderr:
                with self.assertRaises(SystemExit) as error:
                    photocut_cli.main()

            self.assertEqual(2, error.exception.code)
            self.assertIn("--inset must be non-negative", stderr.getvalue())

    def test_detect_result_stores_boundary_and_crop_coordinate_fields(self):
        image = np.zeros((80, 120, 3), dtype=np.uint8)
        boundary = [[10, 10], [110, 10], [110, 70], [10, 70]]
        with patch("photocut.core.load_image", return_value=image), patch(
            "photocut.core.detect_corners_detailed",
            return_value=(boundary, [0.9] * 4, [{"call": 1}] * 4),
        ):
            info = photocut.detect_and_save_corners("scan.jpg", "unused")

        self.assertEqual(boundary, info["algorithm_boundary_corners"])
        self.assertEqual(boundary, info["boundary_corners"])
        self.assertIsNone(info["crop_corners"])
        self.assertIsNone(info["inset"])

    def test_loading_legacy_record_adds_new_fields_only_in_memory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            info_path = Path(temp_dir) / "corners_info.json"
            legacy = [{"filename": "scan.jpg", "corners": [[1, 1]] * 4}]
            info_path.write_text(json.dumps(legacy), encoding="utf-8")

            loaded = photocut.load_corners_info(str(info_path))

            self.assertEqual(legacy[0]["corners"], loaded[0]["boundary_corners"])
            self.assertEqual(legacy[0]["corners"], loaded[0]["algorithm_boundary_corners"])
            self.assertIsNone(loaded[0]["crop_corners"])
            self.assertIsNone(loaded[0]["inset"])
            self.assertEqual(legacy, json.loads(info_path.read_text(encoding="utf-8")))


if __name__ == "__main__":
    unittest.main()
