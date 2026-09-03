import io
import tempfile
import unittest
from pathlib import Path


IMAGE_SIZE = (100, 100)
V7_SQUARE = [[10, 10], [90, 10], [90, 90], [10, 90]]
V7_SHIFTED = [[12, 10], [92, 10], [92, 90], [12, 90]]
LEGACY_SQUARE = [[10, 10], [90, 10], [90, 90], [10, 90]]
LEGACY_FAR = [[30, 30], [70, 30], [70, 70], [30, 70]]


def candidate_audit(candidate_id, corners, score, rank, *, boundary_score=None):
    components = {}
    if boundary_score is not None:
        components["scanner_boundary_score"] = boundary_score
    return {
        "candidate_id": candidate_id,
        "sources": ["background:mask_quad"],
        "original_legal_corners": corners,
        "pre_topk_corners": corners,
        "proposed_refined_corners": None,
        "adopted_refined_corners": corners,
        "pre_truncation_risk_decisions": ["none"],
        "pre_truncation_risk_evidence": {},
        "stage_scores": {"full_score": score, "components": components},
        "stage_ranks": {"selected": rank},
        "truncation_stage": "selected",
        "truncation_reason": None,
    }


def v7_info(confidence, *, corners=V7_SQUARE, risks=(), status="v7_recommended",
            audits=(), scene_profile=None):
    result = {
        "detector": "v7",
        "detection_status": status,
        "overall_confidence": confidence,
        "corners": corners,
        "risks": list(risks),
        "candidate_audit": list(audits),
        "success": True,
    }
    if scene_profile is not None:
        result["scene_profile"] = scene_profile
    return result


def legacy_info(*, corners=LEGACY_SQUARE, confidences=(0.85, 0.82, 0.84, 0.81), success=True):
    return {
        "detector": "v5.2",
        "corners": corners,
        "confidences": list(confidences),
        "success": success,
    }


class V7AutoCascadeTests(unittest.TestCase):
    def test_oversized_jpeg_uses_bounded_analysis_image_with_full_coordinate_mapping(self):
        from PIL import Image
        import photocut.algorithms.v7.input as v7_input

        self.assertTrue(
            hasattr(v7_input, "decode_bytes_for_analysis"),
            "bounded analysis decoder is required for oversized scans",
        )
        image = Image.new("RGB", (400, 200), (120, 80, 40))
        exif = image.getexif()
        exif[274] = 8
        payload = io.BytesIO()
        image.save(payload, format="JPEG", exif=exif)

        loaded = v7_input.decode_bytes_for_analysis(
            payload.getvalue(), max_pixels=10_000, analysis_max_edge=100
        )

        self.assertLessEqual(max(loaded.normalized_size), 100)
        self.assertEqual((200, 400), loaded.full_normalized_size)
        mapped = loaded.map_analysis_to_full(
            ((0, 0), (loaded.normalized_size[0] - 1, loaded.normalized_size[1] - 1))
        )
        self.assertAlmostEqual(0.0, mapped[0][0])
        self.assertAlmostEqual(0.0, mapped[0][1])
        self.assertAlmostEqual(199.0, mapped[1][0])
        self.assertAlmostEqual(399.0, mapped[1][1])

    def test_auto_decode_failure_still_returns_a_persistable_entry(self):
        from unittest.mock import patch
        from photocut import detect_and_save_corners_auto
        from photocut.algorithms.v7.input import InputDecodeError

        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "oversized.jpg"
            source.write_bytes(b"fixture")
            with patch("photocut.algorithms.v7.input.decode_bytes_for_analysis", side_effect=InputDecodeError("too large")):
                result = detect_and_save_corners_auto(str(source), temp_dir)

        required = {
            "algorithm_version", "algorithm_boundary_corners", "boundary_corners",
            "corners", "success", "confirmed", "detector_requested", "detector_used",
        }
        self.assertEqual(set(), required - set(result))

    def test_auto_maps_bounded_detection_back_to_full_normalized_pixels(self):
        import numpy as np
        from unittest.mock import patch
        from photocut import detect_and_save_corners_auto
        from photocut.algorithms.v7.input import LoadedImage

        preview = np.zeros((50, 100, 3), dtype=np.uint8)
        loaded = LoadedImage(
            "a" * 64, "uint8", 3, 1, preview,
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            (400, 200), (100, 50), (400, 200),
            ((399.0 / 99.0, 0.0, 0.0), (0.0, 199.0 / 49.0, 0.0), (0.0, 0.0, 1.0)),
        )
        info = v7_info(0.91, corners=[[0, 0], [99, 0], [99, 49], [0, 49]])
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "large.jpg"
            source.write_bytes(b"fixture")
            with patch("photocut.algorithms.v7.input.decode_bytes_for_analysis", return_value=loaded), patch(
                "photocut.core.detect_and_save_corners_v7", return_value=info
            ):
                result = detect_and_save_corners_auto(str(source), temp_dir)

        self.assertEqual([[0.0, 0.0], [399.0, 0.0], [399.0, 199.0], [0.0, 199.0]], result["corners"])
        self.assertEqual([400, 200], result["original_size"])
        self.assertEqual([100, 50], result["analysis_size"])
        self.assertEqual([400, 200], result["normalized_size"])

    def test_large_auto_manual_review_maps_v7_corners_to_full_space_once(self):
        import numpy as np
        from unittest.mock import patch
        from photocut import detect_and_save_corners_auto
        from photocut.algorithms.v7.input import LoadedImage
        from photocut.algorithms.v7.parameters import V7Parameters
        from photocut.algorithms.v7.types import DetectionIdentity, DetectionResult, DetectionStatus

        preview = np.zeros((50, 100, 3), dtype=np.uint8)
        loaded = LoadedImage(
            "a" * 64, "uint8", 3, 8, preview,
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            (200, 400), (100, 50), (400, 200),
            ((399.0 / 99.0, 0.0, 0.0), (0.0, 199.0 / 49.0, 0.0), (0.0, 0.0, 1.0)),
        )
        params = V7Parameters(mode="safe")
        v7_result = DetectionResult(
            identity=DetectionIdentity(
                "request-1", "image-1", "exif_8", "7.0", params.sha256(), "safe"
            ),
            status=DetectionStatus.V7_LOW_CONFIDENCE,
            corners=((10.0, 10.0), (90.0, 10.0), (90.0, 40.0), (10.0, 40.0)),
            overall_confidence=0.70,
            edge_confidences=(0.70, 0.70, 0.70, 0.70),
            corner_confidences=(0.70, 0.70, 0.70, 0.70),
            risks=("insufficient_support",),
        )
        legacy = legacy_info(
            corners=[[30, 15], [70, 15], [70, 35], [30, 35]],
            confidences=(0.55, 0.52, 0.54, 0.51),
        )

        with patch("photocut.algorithms.v7.detector.detect_corners_v7", return_value=v7_result), patch(
            "photocut.algorithms.v7.legacy_worker.run_legacy_worker",
            return_value={"status": "ok", "result": legacy},
        ):
            result = detect_and_save_corners_auto(
                "missing.jpg", "unused", loaded_input=loaded,
            )

        expected = loaded.map_analysis_to_full(v7_result.corners)
        self.assertEqual("manual_review", result["detector_used"])
        self.assertEqual([list(point) for point in expected], result["corners"])
        self.assertTrue(all(
            0 <= x < loaded.full_normalized_size[0]
            and 0 <= y < loaded.full_normalized_size[1]
            for x, y in result["corners"]
        ))

    def test_full_snapshot_decoder_applies_stored_orientation_without_pillow_budget(self):
        import cv2
        import numpy as np
        import photocut.algorithms.v7.input as v7_input

        self.assertTrue(hasattr(v7_input, "decode_full_bgr_bytes"))
        raw = np.zeros((2, 4, 3), dtype=np.uint8)
        raw[0, 0] = (10, 20, 30)
        ok, encoded = cv2.imencode(".png", raw)
        self.assertTrue(ok)
        normalized = v7_input.decode_full_bgr_bytes(
            encoded.tobytes(), exif_orientation=8, expected_size=(2, 4)
        )
        self.assertEqual((4, 2, 3), normalized.shape)
        self.assertEqual([10, 20, 30], normalized[3, 0].tolist())
        self.assertFalse(normalized.flags.writeable)

    def test_high_confidence_v7_is_accepted_without_legacy(self):
        from photocut.algorithms.v7.cascade import decide_auto

        decision = decide_auto(v7_info(0.91), image_size=IMAGE_SIZE)
        self.assertEqual("accept_v7", decision.action)
        self.assertEqual("v7", decision.detector_used)
        self.assertFalse(decision.needs_legacy)

    def test_medium_v7_requests_one_legacy_cross_check(self):
        from photocut.algorithms.v7.cascade import decide_auto

        decision = decide_auto(v7_info(0.70), image_size=IMAGE_SIZE)
        self.assertEqual("run_v5_2", decision.action)
        self.assertTrue(decision.needs_legacy)

    def test_agreement_prefers_refined_v7(self):
        from photocut.algorithms.v7.cascade import decide_auto

        decision = decide_auto(v7_info(0.70, corners=V7_SHIFTED), image_size=IMAGE_SIZE,
                               legacy=legacy_info(corners=LEGACY_SQUARE))
        self.assertEqual("accept_v7", decision.action)
        self.assertEqual("v7", decision.detector_used)
        self.assertIn("cross_algorithm_agreement", decision.reasons)

    def test_agreement_can_select_highest_scored_v7_top5_candidate(self):
        from photocut.algorithms.v7.cascade import decide_auto

        lower = candidate_audit("lower", [[11, 10], [91, 10], [91, 90], [11, 90]], .61, 2)
        higher = candidate_audit("higher", LEGACY_SQUARE, .69, 3)
        v7 = v7_info(.70, corners=LEGACY_FAR, audits=(lower, higher))

        decision = decide_auto(
            v7, image_size=IMAGE_SIZE, legacy=legacy_info(corners=LEGACY_SQUARE)
        )

        self.assertEqual("accept_v7", decision.action)
        self.assertEqual("v7", decision.detector_used)
        self.assertEqual(LEGACY_SQUARE, decision.selected["corners"])
        self.assertEqual("higher", decision.selected["cascade_v7_candidate_id"])
        self.assertEqual(3, decision.selected["cascade_v7_candidate_rank"])
        self.assertIn("cross_algorithm_agreement_topk", decision.reasons)

    def test_scanner_supplement_is_checked_after_regular_top5(self):
        from photocut.algorithms.v7.cascade import decide_auto

        regular = candidate_audit("regular", LEGACY_FAR, .70, 2)
        supplement = candidate_audit(
            "white", LEGACY_SQUARE, .99, 6, boundary_score=.70,
        )
        supplement["sources"] = ["background:border_connected"]
        supplement["pre_truncation_risk_evidence"] = {
            "supplemental_only": True,
        }
        v7 = v7_info(
            .50, corners=LEGACY_FAR, audits=(regular, supplement),
            scene_profile="scanner_white",
        )

        decision = decide_auto(
            v7, image_size=IMAGE_SIZE, legacy=legacy_info(corners=LEGACY_SQUARE)
        )

        self.assertEqual("v7", decision.detector_used)
        self.assertEqual(
            ("cross_algorithm_agreement_supplement",), decision.reasons
        )
        self.assertEqual("white", decision.selected["cascade_v7_candidate_id"])
        self.assertEqual(6, decision.selected["cascade_v7_candidate_rank"])

    def test_scanner_white_does_not_select_candidate_without_boundary_evidence(self):
        from photocut.algorithms.v7.cascade import decide_auto

        alternate = candidate_audit("alternate", LEGACY_SQUARE, .99, 2)
        decision = decide_auto(
            v7_info(
                .50, corners=LEGACY_FAR, audits=(alternate,),
                scene_profile="scanner_white",
            ),
            image_size=IMAGE_SIZE,
            legacy=legacy_info(corners=LEGACY_SQUARE),
        )

        self.assertEqual("manual_review", decision.action)
        self.assertEqual("manual_review", decision.detector_used)

    def test_primary_v7_is_kept_when_it_already_agrees(self):
        from photocut.algorithms.v7.cascade import decide_auto

        tempting = candidate_audit("tempting", LEGACY_SQUARE, .99, 2)
        v7 = v7_info(.70, corners=V7_SHIFTED, audits=(tempting,))

        decision = decide_auto(
            v7, image_size=IMAGE_SIZE, legacy=legacy_info(corners=LEGACY_SQUARE)
        )

        self.assertEqual(V7_SHIFTED, decision.selected["corners"])
        self.assertNotIn("cascade_v7_candidate_id", decision.selected)
        self.assertEqual(("cross_algorithm_agreement",), decision.reasons)

    def test_strong_legacy_can_take_over_weak_risky_v7(self):
        from photocut.algorithms.v7.cascade import decide_auto

        decision = decide_auto(v7_info(0.44, corners=V7_SHIFTED, risks=("suspected_outer_frame",), status="v7_low_confidence"),
                               image_size=IMAGE_SIZE, legacy=legacy_info(corners=LEGACY_FAR))
        self.assertEqual("accept_v5_2", decision.action)
        self.assertEqual("v5.2", decision.detector_used)

    def test_scanner_white_never_accepts_unverified_strong_legacy(self):
        from photocut.algorithms.v7.cascade import decide_auto

        decision = decide_auto(
            v7_info(
                0.44, corners=V7_SHIFTED,
                risks=("suspected_outer_frame",), status="v7_low_confidence",
                scene_profile="scanner_white",
            ),
            image_size=IMAGE_SIZE,
            legacy=legacy_info(corners=LEGACY_FAR),
        )

        self.assertEqual("manual_review", decision.action)
        self.assertEqual("manual_review", decision.detector_used)

    def test_scanner_white_ranks_all_agreeing_candidates_by_boundary_evidence(self):
        from photocut.algorithms.v7.cascade import decide_auto

        regular = candidate_audit(
            "regular", LEGACY_SQUARE, .95, 2, boundary_score=.20,
        )
        supplement = candidate_audit(
            "white", V7_SHIFTED, .60, 6, boundary_score=.85,
        )
        supplement["sources"] = ["background:border_connected"]
        supplement["pre_truncation_risk_evidence"] = {"supplemental_only": True}
        v7 = v7_info(
            .50, corners=LEGACY_SQUARE, audits=(regular, supplement),
            scene_profile="scanner_white",
        )

        decision = decide_auto(
            v7, image_size=IMAGE_SIZE, legacy=legacy_info(corners=LEGACY_SQUARE)
        )

        self.assertEqual("v7", decision.detector_used)
        self.assertEqual(V7_SHIFTED, decision.selected["corners"])
        self.assertEqual("white", decision.selected["cascade_v7_candidate_id"])
        self.assertEqual(
            ("cross_algorithm_agreement_supplement",), decision.reasons
        )

    def test_scanner_white_rejects_regular_switch_with_large_boundary_drop(self):
        from photocut.algorithms.v7.cascade import decide_auto

        primary = candidate_audit(
            "primary", LEGACY_FAR, .90, 1, boundary_score=.80,
        )
        alternate = candidate_audit(
            "alternate", LEGACY_SQUARE, .70, 2, boundary_score=.30,
        )
        decision = decide_auto(
            v7_info(
                .50, corners=LEGACY_FAR, audits=(primary, alternate),
                scene_profile="scanner_white",
            ),
            image_size=IMAGE_SIZE,
            legacy=legacy_info(corners=LEGACY_SQUARE),
        )

        self.assertEqual("manual_review", decision.action)
        self.assertEqual(
            ("scanner_boundary_conflict",), decision.reasons
        )

    def test_scanner_white_keeps_agreeing_primary_on_small_boundary_gain(self):
        from photocut.algorithms.v7.cascade import decide_auto

        primary = candidate_audit(
            "primary", LEGACY_SQUARE, .80, 1, boundary_score=.60,
        )
        supplement = candidate_audit(
            "supplement", V7_SHIFTED, .70, 6, boundary_score=.64,
        )
        supplement["sources"] = ["background:border_connected"]
        supplement["pre_truncation_risk_evidence"] = {"supplemental_only": True}
        decision = decide_auto(
            v7_info(
                .50, corners=LEGACY_SQUARE, audits=(primary, supplement),
                scene_profile="scanner_white",
            ),
            image_size=IMAGE_SIZE,
            legacy=legacy_info(corners=LEGACY_SQUARE),
        )

        self.assertEqual("accept_v7", decision.action)
        self.assertEqual(LEGACY_SQUARE, decision.selected["corners"])
        self.assertEqual(("cross_algorithm_agreement",), decision.reasons)

    def test_conflict_without_strong_evidence_requires_manual_review(self):
        from photocut.algorithms.v7.cascade import decide_auto

        decision = decide_auto(v7_info(0.70, corners=V7_SHIFTED), image_size=IMAGE_SIZE,
                               legacy=legacy_info(corners=LEGACY_FAR, confidences=(0.55, 0.52, 0.54, 0.51)))
        self.assertEqual("manual_review", decision.action)
        self.assertEqual("manual_review", decision.detector_used)

    def test_terminal_no_photo_and_cancel_do_not_request_legacy(self):
        from photocut.algorithms.v7.cascade import decide_auto

        no_photo = decide_auto(v7_info(0.0, corners=None, status="no_primary_photo"), image_size=IMAGE_SIZE)
        cancelled = decide_auto(v7_info(0.0, corners=None, status="cancelled"), image_size=IMAGE_SIZE)
        self.assertEqual(("manual_review", False), (no_photo.action, no_photo.needs_legacy))
        self.assertEqual(("manual_review", False), (cancelled.action, cancelled.needs_legacy))

    def test_auto_adapter_uses_v7_once_for_high_confidence(self):
        import numpy as np
        from unittest.mock import patch
        from photocut import detect_and_save_corners_auto

        v7 = v7_info(0.91)
        with patch("photocut.core.detect_and_save_corners_v7", return_value=v7) as v7_call, patch(
            "photocut.algorithms.v7.legacy_worker.run_legacy_worker"
        ) as worker:
            result = detect_and_save_corners_auto("missing.jpg", "unused", img=np.zeros((100, 100, 3), dtype=np.uint8))
        self.assertEqual("v7", result["detector_used"])
        v7_call.assert_called_once()
        worker.assert_not_called()

    def test_auto_adapter_calls_legacy_once_for_medium_confidence(self):
        import numpy as np
        from unittest.mock import patch
        from photocut import detect_and_save_corners_auto

        v7 = v7_info(0.70, corners=V7_SHIFTED)
        legacy = legacy_info(corners=LEGACY_SQUARE)
        with patch("photocut.core.detect_and_save_corners_v7", return_value=v7), patch(
            "photocut.algorithms.v7.legacy_worker.run_legacy_worker", return_value={"status": "ok", "result": legacy}
        ) as worker:
            result = detect_and_save_corners_auto("missing.jpg", "unused", img=np.zeros((100, 100, 3), dtype=np.uint8))
        self.assertEqual("v7", result["detector_used"])
        self.assertEqual({"v7": 1, "v5.2": 1}, result["cascade_calls"])
        worker.assert_called_once()

    def test_auto_adapter_uses_selected_top5_candidate(self):
        import numpy as np
        from unittest.mock import patch
        from photocut import detect_and_save_corners_auto

        alternate = candidate_audit("alternate", LEGACY_SQUARE, .68, 2)
        v7 = v7_info(.70, corners=LEGACY_FAR, audits=(alternate,))
        legacy = legacy_info(corners=LEGACY_SQUARE)
        with patch("photocut.core.detect_and_save_corners_v7", return_value=v7), patch(
            "photocut.algorithms.v7.legacy_worker.run_legacy_worker", return_value={"status": "ok", "result": legacy}
        ):
            result = detect_and_save_corners_auto(
                "missing.jpg", "unused", img=np.zeros((100, 100, 3), dtype=np.uint8)
            )

        self.assertEqual("v7", result["detector_used"])
        self.assertEqual(LEGACY_SQUARE, result["corners"])
        self.assertEqual("alternate", result["cascade_v7_candidate_id"])
        self.assertEqual(2, result["cascade_v7_candidate_rank"])

    def test_manual_cross_check_keeps_v7_draft_for_human_adjustment(self):
        import numpy as np
        from unittest.mock import patch
        from photocut import detect_and_save_corners_auto

        v7 = v7_info(0.70, corners=V7_SHIFTED)
        weak_legacy = legacy_info(
            corners=LEGACY_FAR, confidences=(0.55, 0.52, 0.54, 0.51)
        )
        with patch("photocut.core.detect_and_save_corners_v7", return_value=v7), patch(
            "photocut.algorithms.v7.legacy_worker.run_legacy_worker",
            return_value={"status": "ok", "result": weak_legacy},
        ):
            result = detect_and_save_corners_auto(
                "missing.jpg", "unused",
                img=np.zeros((100, 100, 3), dtype=np.uint8),
            )

        self.assertEqual("manual_review", result["detector_used"])
        self.assertEqual(V7_SHIFTED, result["corners"])
        self.assertFalse(result["confirmed"])

    def test_manual_cross_check_uses_legacy_draft_when_v7_has_no_polygon(self):
        import numpy as np
        from unittest.mock import patch
        from photocut import detect_and_save_corners_auto

        failed_v7 = {
            "detector": "v7",
            "detection_status": "error",
            "corners": [],
            "boundary_corners": [],
            "algorithm_boundary_corners": [],
            "risks": ["unrecoverable_error"],
            "success": False,
            "confirmed": False,
        }
        weak_legacy = legacy_info(
            corners=LEGACY_FAR, confidences=(0.55, 0.52, 0.54, 0.51)
        )
        with patch("photocut.core.detect_and_save_corners_v7", return_value=failed_v7), patch(
            "photocut.algorithms.v7.legacy_worker.run_legacy_worker",
            return_value={"status": "ok", "result": weak_legacy},
        ):
            result = detect_and_save_corners_auto(
                "missing.jpg", "unused",
                img=np.zeros((100, 100, 3), dtype=np.uint8),
            )

        self.assertEqual("manual_review", result["detector_used"])
        self.assertEqual(LEGACY_FAR, result["corners"])
        self.assertTrue(result["success"])
        self.assertFalse(result["confirmed"])


if __name__ == "__main__":
    unittest.main()
