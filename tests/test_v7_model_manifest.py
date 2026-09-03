import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from photocut.algorithms.v7.model_manifest import (
    ModelArtifactError,
    ModelManifestError,
    load_model_manifest,
    verify_model_artifact,
)


def _manifest(model_bytes=b"model", **changes):
    data = {
        "schema_version": 1,
        "model_id": "docquadnet-256-test",
        "adapter": "docquadnet",
        "source_repository": "https://github.com/example/project",
        "source_revision": "0123456789abcdef",
        "license": "Apache-2.0",
        "notice_path": "NOTICE",
        "model_filename": "model.ort",
        "model_size": len(model_bytes),
        "model_sha256": "sha256:" + hashlib.sha256(model_bytes).hexdigest(),
        "input_name": "input",
        "input_size": [256, 256],
        "output_names": ["corner_heatmaps", "mask_logits"],
        "redistributable": True,
    }
    data.update(changes)
    return data


class ModelManifestTests(unittest.TestCase):
    def _write(self, root, data):
        path = Path(root) / "manifest.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_loads_strict_immutable_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = load_model_manifest(self._write(tmp, _manifest()))
        self.assertEqual("docquadnet", manifest.adapter)
        self.assertEqual((256, 256), manifest.input_size)
        self.assertEqual(("corner_heatmaps", "mask_logits"), manifest.output_names)
        with self.assertRaises((AttributeError, TypeError)):
            manifest.model_id = "changed"

    def test_rejects_unknown_missing_and_invalid_fields(self):
        cases = []
        unknown = _manifest(extra="leak")
        cases.append(unknown)
        missing = _manifest()
        missing.pop("source_revision")
        cases.append(missing)
        cases.extend([
            _manifest(schema_version=True),
            _manifest(adapter="unknown"),
            _manifest(model_sha256="sha256:" + "G" * 64),
            _manifest(model_size=0),
            _manifest(input_size=[256, 0]),
            _manifest(output_names=[]),
            _manifest(output_names=["same", "same"]),
            _manifest(redistributable="yes"),
        ])
        with tempfile.TemporaryDirectory() as tmp:
            for index, data in enumerate(cases):
                with self.subTest(index=index):
                    with self.assertRaises(ModelManifestError):
                        load_model_manifest(self._write(tmp, data))

    def test_accepts_docaligner_adapter_with_one_heatmap_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = load_model_manifest(self._write(tmp, _manifest(
                model_id="docaligner-lcnet",
                adapter="docaligner_heatmap",
                model_filename="model.onnx",
                output_names=["heatmap"],
            )))
        self.assertEqual("docaligner_heatmap", manifest.adapter)

    def test_accepts_v8_photo_mask_adapter_without_changing_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = load_model_manifest(self._write(tmp, _manifest(
                model_id="photocut-v8-mask",
                adapter="photo_mask_v8",
                model_filename="model.onnx",
                input_size=[768, 768],
                output_names=["mask_logits"],
            )))
        self.assertEqual("photo_mask_v8", manifest.adapter)
        self.assertEqual((768, 768), manifest.input_size)

    def test_verifies_exact_regular_file(self):
        payload = b"verified model bytes"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model.ort"
            model.write_bytes(payload)
            manifest = load_model_manifest(self._write(root, _manifest(payload)))
            self.assertEqual(model.resolve(), verify_model_artifact(manifest, model))

    def test_rejects_missing_size_hash_filename_and_symlink(self):
        payload = b"verified model bytes"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model.ort"
            model.write_bytes(payload)
            cases = [
                (_manifest(payload, model_filename="different.ort"), model, "filename"),
                (_manifest(payload, model_size=len(payload) + 1), model, "size"),
                (_manifest(payload, model_sha256="sha256:" + "0" * 64), model, "hash"),
                (_manifest(payload), root / "missing.ort", "missing"),
            ]
            for index, (data, candidate, message) in enumerate(cases):
                with self.subTest(index=index):
                    manifest = load_model_manifest(self._write(root, data))
                    with self.assertRaisesRegex(ModelArtifactError, message):
                        verify_model_artifact(manifest, candidate)

            link = root / "linked.ort"
            try:
                link.symlink_to(model)
            except (OSError, NotImplementedError):
                return
            manifest = load_model_manifest(self._write(root, _manifest(
                payload, model_filename="linked.ort"
            )))
            with self.assertRaisesRegex(ModelArtifactError, "symlink"):
                verify_model_artifact(manifest, link)


if __name__ == "__main__":
    unittest.main()
