import copy
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from photocut.confirmation.model import build_crop_event
from photocut.data.crop_event_store import CropEventStore
from photocut.data.dataset_store import DatasetStore, atomic_write_json, load_jsonl
from photocut import cli as photocut_cli

class CropEventIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.boundary = [[10, 10], [110, 10], [110, 70], [10, 70]]
        self.crop = [[15.0, 15.0], [105.0, 15.0], [105.0, 65.0], [15.0, 65.0]]
        self.entry = {
            "image_id": "sha256:fixture",
            "annotation_id": "ann_fixture",
            "boundary_corners": self.boundary,
            "crop_corners": self.crop,
        }
        self.inset = {"method": "parallel_edge_offset", "distance_px": 5.0}

    def _event(self, root):
        output = root / "crop.jpg"
        output.write_bytes(b"crop")
        return build_crop_event(self.entry, output, self.inset)

    def test_builds_complete_crop_provenance_event(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            event = self._event(Path(temp_dir))

        self.assertEqual(1, event["schema_version"])
        self.assertTrue(event["crop_id"].startswith("crop_"))
        self.assertEqual("sha256:fixture", event["image_id"])
        self.assertEqual("ann_fixture", event["annotation_id"])
        self.assertEqual(self.boundary, event["boundary_corners"])
        self.assertEqual(self.crop, event["crop_corners"])
        self.assertEqual(self.inset, event["inset"])
        self.assertIn("+", event["created_at"])

    def test_rejects_missing_annotation_invalid_geometry_inset_and_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            cases = (
                (dict(self.entry, annotation_id=""), self.inset, root / "crop.jpg"),
                (dict(self.entry, crop_corners=self.crop[:3]), self.inset, root / "crop.jpg"),
                (self.entry, {"method": "parallel_edge_offset", "distance_px": -1}, root / "crop.jpg"),
                (self.entry, self.inset, root / "missing.jpg"),
            )
            for entry, inset, output in cases:
                with self.subTest(entry=entry, inset=inset, output=output):
                    with self.assertRaises(ValueError):
                        build_crop_event(entry, output, inset)

    def test_identical_rerun_returns_existing_event_without_duplicate_line(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = root / "crops.jsonl"
            store = CropEventStore(path)
            event = self._event(root)

            first = store.append_idempotent(event)
            second = store.append_idempotent(dict(event, crop_id="crop_other"))

            self.assertEqual(first, second)
            self.assertEqual(1, len(load_jsonl(path)))

    def test_changed_inset_appends_a_new_crop_event(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = root / "crops.jsonl"
            store = CropEventStore(path)
            event = self._event(root)
            store.append_idempotent(event)
            changed = copy.deepcopy(event)
            changed["inset"]["distance_px"] = 8.0

            store.append_idempotent(changed)

            self.assertEqual(2, len(load_jsonl(path)))

    def test_partial_append_failure_preserves_existing_history_for_retry(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = root / "crops.jsonl"
            store = CropEventStore(path)
            first = self._event(root)
            store.append_idempotent(first)
            before = path.read_bytes()
            path.write_bytes(before + b'{"incomplete"')

            with self.assertRaisesRegex(ValueError, "invalid JSONL"):
                store.append_idempotent(self._event(root))

            self.assertEqual(before + b'{"incomplete"', path.read_bytes())

    def test_malformed_existing_jsonl_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = root / "crops.jsonl"
            original = b"not json\n"
            path.write_bytes(original)

            with self.assertRaisesRegex(ValueError, "invalid JSONL"):
                CropEventStore(path).append_idempotent(self._event(root))

            self.assertEqual(original, path.read_bytes())

    def test_events_and_append_both_reject_complete_log_without_terminal_newline(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = root / "crops.jsonl"
            event = self._event(root)
            path.write_bytes(json.dumps(event).encode("utf-8"))

            with self.assertRaisesRegex(ValueError, "terminal newline"):
                CropEventStore(path).events()
            with self.assertRaisesRegex(ValueError, "terminal newline"):
                CropEventStore(path).append_idempotent(event)

    def test_events_reject_forged_deterministic_ids_without_rewriting_history(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = root / "crops.jsonl"
            forged = self._event(root)
            forged["event_id"] = forged["crop_id"] = "crop_forged"
            original = (json.dumps(forged) + "\n").encode("utf-8")
            path.write_bytes(original)

            with self.assertRaisesRegex(ValueError, "deterministic"):
                CropEventStore(path).events()
            self.assertEqual(original, path.read_bytes())

    def test_crop_uses_detection_project_root_for_relative_dataset_reference(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            project = root / "project"
            input_dir = root / "input"
            output_dir = root / "output"
            project.mkdir()
            input_dir.mkdir()
            output_dir.mkdir()
            (input_dir / "scan.jpg").write_bytes(b"source")
            dataset = DatasetStore(project / ".photocut" / "internal" / "datasets")
            batch = dataset.start_batch({"parameters": {}})
            atomic_write_json(output_dir / ".photocut_batch.json", {
                "schema_version": 1, "batch_id": batch.batch_id,
                "dataset_root": ".photocut/internal/datasets",
            })
            (output_dir / "corners_info.json").write_text(json.dumps([{
                "filename": "scan.jpg", "image_id": "sha256:fixture",
                "annotation_id": "ann_fixture", "boundary_corners": self.boundary,
                "corners": self.boundary, "original_size": [120, 80],
                "success": True, "confirmed": True,
            }]), encoding="utf-8")
            args = Namespace(input=str(input_dir), output=str(output_dir), skip="", inset=5.0)
            output = output_dir / "裁切成品" / "scan.jpg"

            def successful_crop(*_args):
                output.parent.mkdir(exist_ok=True)
                output.write_bytes(b"crop")
                return True, str(output)

            with patch.object(
                photocut_cli,
                "__file__",
                str(project / "photocut" / "cli.py"),
            ), patch(
                "photocut.cli.crop_image", side_effect=successful_crop
            ):
                photocut_cli.crop_command(args)

            self.assertEqual(1, len(load_jsonl(batch.crops)))

    def test_crop_rejects_symlinked_output_and_escaping_dataset_reference(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            input_dir = root / "input"
            output_dir = root / "output"
            external = root / "external"
            input_dir.mkdir()
            output_dir.mkdir()
            external.mkdir()
            (input_dir / "scan.jpg").write_bytes(b"source")
            (output_dir / "裁切成品").symlink_to(external, target_is_directory=True)
            (output_dir / "corners_info.json").write_text(json.dumps([{
                "filename": "scan.jpg", "image_id": "sha256:fixture",
                "annotation_id": "ann_fixture", "boundary_corners": self.boundary,
                "corners": self.boundary, "original_size": [120, 80],
                "success": True, "confirmed": True,
            }]), encoding="utf-8")
            args = Namespace(input=str(input_dir), output=str(output_dir), skip="", inset=5.0)

            with patch("photocut.cli.crop_image") as crop:
                photocut_cli.crop_command(args)

            crop.assert_not_called()
            self.assertEqual([], list(external.iterdir()))

    def test_crop_reference_rejects_parent_traversal_before_external_event_write(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            project = root / "project"
            output_dir = root / "output"
            project.mkdir()
            output_dir.mkdir()
            external = DatasetStore(root / "external" / "datasets")
            batch = external.start_batch({"parameters": {}})
            atomic_write_json(output_dir / ".photocut_batch.json", {
                "schema_version": 1, "batch_id": batch.batch_id,
                "dataset_root": "../external/datasets",
            })

            with self.assertRaisesRegex(ValueError, "dataset_root"):
                photocut_cli.resolve_crop_event_store(output_dir, project)
            self.assertFalse(batch.crops.exists())

    def test_crop_reference_rejects_absolute_missing_or_symlinked_dataset_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            project = root / "project"
            output_dir = root / "output"
            project.mkdir()
            output_dir.mkdir()
            dataset = DatasetStore(root / "external" / "datasets")
            batch = dataset.start_batch({"parameters": {}})
            for name, dataset_root in (
                ("missing", root / "missing"),
                ("symlink", root / "datasets-link"),
            ):
                with self.subTest(name=name):
                    if name == "symlink":
                        dataset_root.symlink_to(dataset.root, target_is_directory=True)
                    atomic_write_json(output_dir / ".photocut_batch.json", {
                        "schema_version": 1, "batch_id": batch.batch_id,
                        "dataset_root": str(dataset_root),
                    })
                    with self.assertRaisesRegex(ValueError, "dataset_root"):
                        photocut_cli.resolve_crop_event_store(output_dir, project)
            self.assertFalse(batch.crops.exists())

    def test_production_crop_rerun_records_one_event_and_failed_attempt_is_append_only(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            output_dir.mkdir()
            (input_dir / "scan.jpg").write_bytes(b"source")
            project = root / "project"
            project.mkdir()
            dataset = DatasetStore(project / "dataset")
            batch = dataset.start_batch({"parameters": {}})
            atomic_write_json(
                output_dir / ".photocut_batch.json",
                {"schema_version": 1, "batch_id": batch.batch_id, "dataset_root": "dataset"},
            )
            (output_dir / "corners_info.json").write_text(json.dumps([{
                "filename": "scan.jpg", "image_id": "sha256:fixture",
                "annotation_id": "ann_fixture", "boundary_corners": self.boundary,
                "corners": self.boundary, "original_size": [120, 80],
                "success": True, "confirmed": True,
            }]), encoding="utf-8")
            args = Namespace(input=str(input_dir), output=str(output_dir), skip="", inset=5.0)
            output = output_dir / "裁切成品" / "scan.jpg"

            def successful_crop(*_args):
                output.parent.mkdir(exist_ok=True)
                output.write_bytes(b"crop")
                return True, str(output)

            with patch.object(
                photocut_cli,
                "__file__",
                str(project / "photocut" / "cli.py"),
            ), patch(
                "photocut.cli.crop_image", side_effect=successful_crop
            ):
                photocut_cli.crop_command(args)
                photocut_cli.crop_command(args)

            events = load_jsonl(batch.crops)
            self.assertEqual(1, len(events))
            self.assertEqual("succeeded", events[0]["status"])

            with patch.object(
                photocut_cli,
                "__file__",
                str(project / "photocut" / "cli.py"),
            ), patch(
                "photocut.cli.crop_image", return_value=(False, None)
            ):
                photocut_cli.crop_command(args)

            events = load_jsonl(batch.crops)
            self.assertEqual(2, len(events))
            self.assertEqual("failed", events[1]["status"])
            self.assertIn("failure", events[1]["error"])


if __name__ == "__main__":
    unittest.main()
