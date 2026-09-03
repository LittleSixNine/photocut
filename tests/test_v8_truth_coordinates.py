import io

import numpy as np
import pytest
from PIL import Image

from photocut.algorithms.v7.input import LoadedImage, decode_bytes, decode_bytes_for_analysis
from photocut.algorithms.v8.truth_coordinates import (
    derive_coordinate_decision,
    make_coordinate_provenance,
    map_full_normalized_to_analysis,
    resolve_full_normalized_truth,
)


def _jpeg(orientation: int) -> bytes:
    image = Image.new("RGB", (8, 6), (230, 230, 230))
    exif = image.getexif()
    exif[274] = orientation
    output = io.BytesIO()
    image.save(output, format="JPEG", exif=exif.tobytes())
    return output.getvalue()


def _annotation(image_id: str, corners) -> dict:
    return {
        "annotation_id": "ann-1",
        "image_id": image_id,
        "boundary_corners": [list(point) for point in corners],
    }


@pytest.mark.parametrize("orientation", range(1, 9))
def test_stored_raster_truth_maps_to_full_exif_normalized_for_all_orientations(orientation):
    loaded = decode_bytes(_jpeg(orientation))
    width, height = loaded.full_normalized_size
    expected = (
        (1.0, 1.0),
        (width - 2.0, 1.0),
        (width - 2.0, height - 2.0),
        (1.0, height - 2.0),
    )
    stored = loaded.map_normalized_to_original(expected)
    annotation = _annotation("image-1", stored)
    provenance = make_coordinate_provenance(
        annotation,
        loaded,
        coordinate_space="stored_raster_original",
        evidence={"kind": "album03_import", "identity": "review-log-sha256"},
    )

    actual = resolve_full_normalized_truth(annotation, loaded, provenance)

    np.testing.assert_allclose(actual, expected, atol=1e-9)
    assert provenance["corner_order"] == "full_normalized_tl_tr_br_bl"
    assert provenance["exif_orientation"] == orientation


def test_full_normalized_truth_is_kept_without_reinterpreting_it():
    loaded = decode_bytes(_jpeg(6))
    truth = ((1, 1), (4, 1), (4, 6), (1, 6))
    annotation = _annotation("image-1", truth)
    provenance = make_coordinate_provenance(
        annotation,
        loaded,
        coordinate_space="full_exif_normalized",
        evidence={"kind": "confirmation_transaction", "identity": "run-1"},
    )

    assert resolve_full_normalized_truth(annotation, loaded, provenance) == tuple(
        (float(x), float(y)) for x, y in truth
    )


def test_stored_raster_truth_on_analysis_preview_resolves_to_full_not_analysis_space():
    source = _jpeg(8)
    exact = decode_bytes(source)
    preview = decode_bytes_for_analysis(source, max_pixels=10, analysis_max_edge=4)
    assert preview.is_analysis_preview
    width, height = exact.full_normalized_size
    expected = ((1.0, 1.0), (width - 2.0, 1.0), (width - 2.0, height - 2.0), (1.0, height - 2.0))
    stored = exact.map_normalized_to_original(expected)
    annotation = _annotation("image-1", stored)
    provenance = make_coordinate_provenance(
        annotation,
        preview,
        coordinate_space="stored_raster_original",
        evidence={"kind": "album03_import", "identity": "review-log-sha256"},
    )

    actual = resolve_full_normalized_truth(annotation, preview, provenance)

    np.testing.assert_allclose(actual, expected, atol=1e-9)


def test_missing_or_conflicting_provenance_fails_closed():
    loaded = decode_bytes(_jpeg(1))
    annotation = _annotation("image-1", ((1, 1), (6, 1), (6, 4), (1, 4)))

    with pytest.raises(ValueError, match="provenance"):
        resolve_full_normalized_truth(annotation, loaded, None)

    provenance = make_coordinate_provenance(
        annotation,
        loaded,
        coordinate_space="full_exif_normalized",
        evidence={"kind": "confirmation_transaction", "identity": "run-1"},
    )
    tampered = dict(provenance)
    tampered["coordinate_space"] = "stored_raster_original"
    with pytest.raises(ValueError, match="provenance"):
        resolve_full_normalized_truth(annotation, loaded, tampered)

    wrong_annotation = dict(annotation, annotation_id="ann-2")
    with pytest.raises(ValueError, match="annotation"):
        resolve_full_normalized_truth(wrong_annotation, loaded, provenance)


def test_full_normalized_analysis_round_trip_uses_explicit_endpoint_transform():
    analysis_to_full = (
        (2.25, 0.0, 0.0),
        (0.0, 3.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    loaded = LoadedImage(
        source_sha256="0" * 64,
        original_dtype="uint8",
        original_channels=3,
        exif_orientation=1,
        normalized_bgr=np.zeros((4, 5, 3), dtype=np.uint8),
        forward_transform=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        inverse_transform=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        original_size=(10, 10),
        normalized_size=(5, 4),
        full_normalized_size=(10, 10),
        analysis_to_full_transform=analysis_to_full,
    )
    truth = ((0.0, 0.0), (9.0, 0.0), (9.0, 9.0), (0.0, 9.0))

    analysis = map_full_normalized_to_analysis(loaded, truth)

    np.testing.assert_allclose(loaded.map_analysis_to_full(analysis), truth, atol=1e-9)


def test_coordinate_decision_uses_durable_import_or_run_evidence_only():
    loaded = decode_bytes(_jpeg(6))
    annotation = _annotation("image-1", ((1, 1), (4, 1), (4, 6), (1, 6)))
    annotation["run_id"] = "run-1"
    annotation["provenance"] = {
        "source_format": "legacy_corners_info",
        "source_json_sha256": "a" * 64,
        "original_filename": "scan.jpg",
        "algorithm_result_available": True,
    }
    stored_rule = {
        "coordinate_space": "stored_raster_original",
        "evidence_document_hash": "sha256:" + "b" * 64,
        "batch_id": "batch-1",
    }
    # Legacy import rows inherit run_id from their immutable parent run.
    legacy_import = {"image_id": "image-1", "legacy_import": True}

    stored = derive_coordinate_decision(
        annotation,
        loaded,
        run_identity={"run_id": "run-1", "batch_id": "batch-1"},
        run_image=legacy_import,
        stored_coordinate_rule=stored_rule,
    )
    assert stored["coordinate_space"] == "stored_raster_original"

    current_run_image = {
        "image_id": "image-1",
        "run_id": "run-1",
        "legacy_info": {
            "source_original_size": list(loaded.original_size),
            "normalized_size": list(loaded.full_normalized_size),
            "detection_identity": {"orientation_transform": "exif_6"},
        },
    }
    full = derive_coordinate_decision(
        annotation,
        loaded,
        run_identity={"run_id": "run-1", "batch_id": "batch-2"},
        run_image=current_run_image,
    )
    assert full["coordinate_space"] == "full_exif_normalized"


def test_coordinate_decision_accepts_clean_legacy_gui_size_identity_and_rejects_ambiguity():
    loaded = decode_bytes(_jpeg(6))
    annotation = _annotation("image-1", ((1, 1), (4, 1), (4, 6), (1, 6)))
    annotation["run_id"] = "run-1"
    old_run = {
        "run_id": "run-1",
        "batch_id": "batch-old",
        "runtime": {
            "git": {"available": True, "commit": "abc", "dirty": False},
            "environment": {"opencv": "4.10.0"},
        },
    }
    old_image = {
        "image_id": "image-1",
        "run_id": "run-1",
        "legacy_info": {"original_size": list(loaded.full_normalized_size)},
    }

    decision = derive_coordinate_decision(
        annotation,
        loaded,
        run_identity=old_run,
        run_image=old_image,
    )
    assert decision["coordinate_space"] == "full_exif_normalized"

    dirty = dict(old_run, runtime={**old_run["runtime"], "git": {"available": True, "commit": "abc", "dirty": True}})
    with pytest.raises(ValueError, match="ambiguous"):
        derive_coordinate_decision(annotation, loaded, run_identity=dirty, run_image=old_image)
