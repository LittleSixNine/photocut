import copy
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from photocut import cli as photocut_cli
from photocut.algorithms.v8.code_seal import compute_validated_core_sha256
from photocut.algorithms.v8.cascade import V8CascadeStatus


PROJECT_ROOT = Path(photocut_cli.__file__).resolve().parent
POLICY_PATH = PROJECT_ROOT / photocut_cli.AUTO_V4_POLICY_RELATIVE_PATH


class AutoV4PolicyTests(unittest.TestCase):
    def setUp(self):
        self.policy = photocut_cli._load_auto_v4_policy(POLICY_PATH)
        self.loaded = SimpleNamespace(
            source_sha256="a" * 64,
            original_size=(100, 100),
            normalized_size=(100, 100),
            full_normalized_size=(100, 100),
            orientation_transform="identity",
            normalized_bgr=np.zeros((100, 100, 3), dtype=np.uint8),
        )
        self.v8_runtime = SimpleNamespace(
            policy=SimpleNamespace(
                algorithm_version="8.2", policy_sha256="sha256:" + "1" * 64
            ),
            mask_provider=object(),
            runtime_config={"schema_version": 1},
            runtime_config_sha256="sha256:" + "2" * 64,
            model_manifest_sha256="sha256:" + "3" * 64,
            boundary_quality_artifact=None,
        )
        self.auto_runtime = photocut_cli.AutoV4Runtime(
            self.policy, "implicit_default", "v8", "v8", False, None,
            self.v8_runtime,
        )

    @staticmethod
    def _core(*, corners=None, status="manual_review", version="8.2"):
        corners = corners or [[10, 10], [90, 10], [90, 90], [10, 90]]
        return {
            "filename": "scan.jpg",
            "algorithm_version": version,
            "algorithm_boundary_corners": copy.deepcopy(corners),
            "boundary_corners": copy.deepcopy(corners),
            "algorithm_corners": copy.deepcopy(corners),
            "corners": copy.deepcopy(corners),
            "original_size": [100, 100],
            "preview_size": [100, 100],
            "success": bool(corners),
            "confirmed": False,
            "detector": "v8" if version == "8.2" else "v7",
            "detector_requested": "v8" if version == "8.2" else "v7",
            "detector_used": "manual_review" if status == "manual_review" else "v8",
            "detection_status": status,
            "risks": [],
        }

    @staticmethod
    def _dormant(status, core, reason):
        return SimpleNamespace(
            status=status,
            core_payload=core,
            audit_envelope={
                "requested_detector": "v8",
                "v8_provider_status": "success",
                "v8_fallback_reason": reason if status is V8CascadeStatus.V7_FALLBACK else None,
                "v8_decision_status": status.value,
                "v8_selection_reason": reason,
                "auto_v4_observed_v7_calls": 1,
            },
        )

    def _run(self, dormant):
        with patch(
            "photocut.cli.detect_and_save_corners_v8_dormant",
            return_value=dormant,
        ):
            return photocut_cli._detect_auto_v4_v8(
                img_path=Path("scan.jpg"),
                output_dir="unused",
                auto_runtime=self.auto_runtime,
                loaded_input=self.loaded,
                relative_path="scan.jpg",
                request_id="source-1",
                image_id="sha256:" + "a" * 64,
                shrink_min=25,
                shrink_max=70,
                params=photocut_cli.DEFAULT_DETECTION_PARAMETERS,
            )

    def test_tracked_policy_is_canonical_and_fail_closed(self):
        policy = photocut_cli._load_auto_v4_policy(POLICY_PATH)

        self.assertEqual(1, policy.schema_version)
        self.assertEqual("auto-v4", policy.cascade_version)
        self.assertEqual("v8", policy.scanner_white_default_engine)
        self.assertEqual(3.0, policy.v52_worker_timeout_s)
        self.assertFalse(policy.v52_witness_auto_accept)
        self.assertLessEqual(policy.max_mean_normalized_corner_distance, 0.005)
        self.assertGreaterEqual(policy.minimum_polygon_iou, 0.98)
        self.assertTrue(policy.witness_reason_allowlist)
        self.assertFalse(
            set(policy.witness_reason_allowlist)
            & set(policy.semantic_terminal_reasons)
        )
        self.assertRegex(policy.policy_sha256, r"^sha256:[0-9a-f]{64}$")

    def test_policy_rejects_unknown_fields_and_loose_geometry(self):
        base = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
        mutations = []
        unknown = copy.deepcopy(base)
        unknown["future_switch"] = True
        mutations.append(unknown)
        loose_distance = copy.deepcopy(base)
        loose_distance["agreement"]["max_mean_normalized_corner_distance"] = 0.02
        mutations.append(loose_distance)
        loose_iou = copy.deepcopy(base)
        loose_iou["agreement"]["minimum_polygon_iou"] = 0.90
        mutations.append(loose_iou)
        overlap = copy.deepcopy(base)
        overlap["witness_reason_allowlist"].append("cancelled")
        mutations.append(overlap)

        for payload in mutations:
            with self.subTest(payload=payload), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "policy.json"
                path.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaises(ValueError):
                    photocut_cli._load_auto_v4_policy(path)

    def test_policy_rejects_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            link = Path(directory) / "policy.json"
            link.symlink_to(POLICY_PATH)
            with self.assertRaises(ValueError):
                photocut_cli._load_auto_v4_policy(link)

    def test_auto_policy_does_not_change_v8_validated_core(self):
        self.assertEqual(
            photocut_cli._v8_runtime_from_args(SimpleNamespace()).policy.validated_core_sha256,
            compute_validated_core_sha256(PROJECT_ROOT),
        )

    def test_v8_automatic_is_projected_without_legacy_or_core_mutation(self):
        core = self._core(status="v8_recommended")
        dormant = self._dormant(V8CascadeStatus.AUTOMATIC, core, "edge_base")
        original = copy.deepcopy(core)
        with patch("photocut.cli._run_auto_v4_legacy") as legacy:
            info = self._run(dormant)

        legacy.assert_not_called()
        self.assertEqual(original, core)
        self.assertEqual(original, info["cascade_v8_result"]["core_payload"])
        self.assertEqual("auto", info["detector_requested"])
        self.assertEqual("v8", info["detector_used"])
        self.assertEqual("automatic", info["auto_v4_status"])
        self.assertEqual({"v7": 1, "v8": 1, "v5.2": 0}, info["cascade_calls"])

    def test_allowlisted_manual_runs_one_legacy_but_disabled_witness_stays_manual(self):
        core = self._core()
        dormant = self._dormant(
            V8CascadeStatus.MANUAL_REVIEW,
            core,
            "weak_unopposed_edge_evidence",
        )
        legacy = {
            "status": "ok",
            "result": {
                "success": True,
                "corners": [[10, 10], [90, 10], [90, 90], [10, 90]],
                "confidences": [0.9, 0.9, 0.9, 0.9],
            },
        }
        with patch("photocut.cli._run_auto_v4_legacy", return_value=legacy) as run:
            info = self._run(dormant)

        run.assert_called_once()
        self.assertEqual("manual_review", info["detector_used"])
        self.assertEqual("disabled_safe_proposal", info["auto_v4_witness"]["action"])
        self.assertEqual(1, info["cascade_calls"]["v5.2"])
        self.assertEqual(legacy["result"]["corners"], info["v52_corners"])

    def test_non_allowlisted_manual_and_semantic_terminal_do_not_run_legacy(self):
        cases = (
            self._dormant(
                V8CascadeStatus.MANUAL_REVIEW,
                self._core(),
                "strong_three_way_conflict",
            ),
            self._dormant(
                V8CascadeStatus.V7_FALLBACK,
                {**self._core(version="7.1"), "detection_status": "no_primary_photo", "corners": []},
                "no_primary_photo",
            ),
        )
        for dormant in cases:
            with self.subTest(status=dormant.status), patch(
                "photocut.cli._run_auto_v4_legacy"
            ) as legacy:
                info = self._run(dormant)
            legacy.assert_not_called()
            self.assertEqual("manual_review", info["detector_used"])
            self.assertEqual(0, info["cascade_calls"]["v5.2"])

    def test_v7_fallback_is_always_manual_and_legacy_is_only_gui_assistance(self):
        core = self._core(version="7.1")
        dormant = self._dormant(
            V8CascadeStatus.V7_FALLBACK, core, "mask_provider_error"
        )
        legacy = {
            "status": "ok",
            "result": {
                "success": True,
                "corners": [[11, 11], [89, 11], [89, 89], [11, 89]],
                "confidences": [0.95] * 4,
            },
        }
        with patch("photocut.cli._run_auto_v4_legacy", return_value=legacy):
            info = self._run(dormant)

        self.assertEqual("manual_review", info["detector_used"])
        self.assertEqual("7.1", info["algorithm_version"])
        self.assertEqual(core["corners"], info["corners"])
        self.assertEqual(legacy["result"]["corners"], info["v52_corners"])
        self.assertEqual("not_applicable_fallback", info["auto_v4_witness"]["action"])

    def test_fallback_with_only_legal_v52_draft_keeps_manual_identity(self):
        core = self._core(corners=[], version="7.1")
        core.update({
            "success": False,
            "corners": [],
            "boundary_corners": [],
            "algorithm_boundary_corners": [],
            "algorithm_corners": [],
        })
        dormant = self._dormant(
            V8CascadeStatus.V7_FALLBACK, core, "no_v8_candidate"
        )
        legacy_corners = [[12, 12], [88, 12], [88, 88], [12, 88]]
        with patch(
            "photocut.cli._run_auto_v4_legacy",
            return_value={
                "status": "ok",
                "result": {"success": True, "corners": legacy_corners, "confidences": [0.9] * 4},
            },
        ):
            info = self._run(dormant)

        self.assertEqual("manual_review", info["detector_used"])
        self.assertEqual("5.2", info["algorithm_version"])
        self.assertEqual(legacy_corners, info["corners"])
        self.assertEqual("v52:fallback_gui_only", info["confirmation_primary_candidate_id"])

    def test_enabled_witness_can_only_accept_current_v8_corners(self):
        enabled = replace(self.policy, v52_witness_auto_accept=True)
        self.auto_runtime = replace(self.auto_runtime, policy=enabled)
        core = self._core()
        dormant = self._dormant(
            V8CascadeStatus.MANUAL_REVIEW, core, "weak_unopposed_edge_evidence"
        )
        with patch(
            "photocut.cli._run_auto_v4_legacy",
            return_value={
                "status": "ok",
                "result": {"success": True, "corners": copy.deepcopy(core["corners"]), "confidences": [0.9] * 4},
            },
        ):
            info = self._run(dormant)

        self.assertEqual("accept_current_v8", info["auto_v4_witness"]["action"])
        self.assertEqual("v8", info["detector_used"])
        self.assertEqual(core["corners"], info["corners"])


if __name__ == "__main__":
    unittest.main()
