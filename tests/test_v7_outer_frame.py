import numpy as np

from photocut.algorithms.v7.scoring import score_candidates


def q(x, y, w, h):
    return ((x, y), (x + w, y), (x + w, y + h), (x, y + h))


def scanner_scene():
    image = np.full((200, 240, 3), 175, np.uint8)
    image[20:180, 25:215] = 242
    image[45:155, 55:190] = (50, 100, 180)
    return image


def test_scanner_outer_frame_without_outside_pixels_is_suspected_when_inner_content_is_sparse():
    image = scanner_scene()
    candidates = [
        {"candidate_id": "outer", "corners": q(0, 0, 239, 199), "sources": ("contour",)},
        {"candidate_id": "photo", "corners": q(55, 45, 135, 110), "sources": ("lines",)},
    ]
    result = score_candidates(candidates, image, image_size=(240, 200), top_k=2)
    outer = next(a for a in result.audits if a.candidate_id == "outer")
    assert "suspected_outer_frame" in outer.pre_truncation_risk_decisions
    assert "outside_background_unverifiable" in outer.pre_truncation_risk_decisions
    assert outer.pre_truncation_risk_evidence["nested_candidate_exists"] is True


def test_textured_border_photo_with_nested_detail_stays_unknown():
    rng = np.random.default_rng(7)
    image = np.full((120, 160, 3), 120, np.uint8)
    image[:, :140] = rng.integers(20, 235, size=(120, 140, 3), dtype=np.uint8)
    image[35:85, 35:105] = 40
    result = score_candidates([
        {"candidate_id": "photo", "corners": q(0, 0, 140, 119), "sources": ("contour",)},
        {"candidate_id": "internal", "corners": q(35, 35, 70, 50), "sources": ("lines",)},
    ], image, image_size=(160, 120), top_k=2)
    photo = next(a for a in result.audits if a.candidate_id == "photo")
    assert photo.pre_truncation_risk_decisions == ("outside_background_unverifiable",)


def test_true_photo_touching_border_is_unknown_without_outside_pixels():
    image = np.full((120, 160, 3), 120, np.uint8)
    image[0:120, 0:140] = 210
    result = score_candidates([{"candidate_id": "touch", "corners": q(0, 0, 140, 119), "sources": ("contour",)}], image,
                              image_size=(160, 120), top_k=1)
    audit = result.audits[0]
    assert "outside_background_unverifiable" in audit.pre_truncation_risk_decisions
    assert "suspected_outer_frame" not in audit.pre_truncation_risk_decisions


def test_true_border_photo_with_internal_nested_rectangle_is_not_outer_rejected():
    image = np.full((120, 160, 3), 120, np.uint8)
    image[:, :140] = 210
    image[35:85, 35:105] = 40
    result = score_candidates([
        {"candidate_id": "photo", "corners": q(0, 0, 140, 119), "sources": ("contour",)},
        {"candidate_id": "internal", "corners": q(35, 35, 70, 50), "sources": ("lines",)},
    ], image, image_size=(160, 120), top_k=2)
    photo = next(a for a in result.audits if a.candidate_id == "photo")
    assert "outside_background_unverifiable" in photo.pre_truncation_risk_decisions
    assert "suspected_outer_frame" not in photo.pre_truncation_risk_decisions
