import json
import tempfile
import unittest
from pathlib import Path

from photocut.algorithms.v7.reporting import build_run_identity, write_evaluation_report


class ReportingTests(unittest.TestCase):
    def test_identity_changes_when_split_or_parameter_changes(self):
        first = build_run_identity("sha256:manifest", "train", ["i1"], "a" * 64, "commit", "analysis", "env")
        second = build_run_identity("sha256:manifest", "validation", ["i1"], "a" * 64, "commit", "analysis", "env")
        self.assertNotEqual(first["run_id"], second["run_id"])
        self.assertEqual(64, len(first["run_id"]))

    def test_report_writes_jsonl_and_markdown_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            report = write_evaluation_report(
                directory,
                run_identity={"run_id": "r1"},
                paired_results=[{"image_id": "i1"}], metrics={"p95": 1},
                slices={"rotated": {"n": 1}}, gate={"passed": True},
            )
            self.assertEqual("r1", report["run_id"])
            self.assertTrue((Path(directory) / "run.json").exists())
            self.assertTrue((Path(directory) / "paired_results.jsonl").exists())
            self.assertIn("v7 evaluation", (Path(directory) / "report.md").read_text())


if __name__ == "__main__":
    unittest.main()
