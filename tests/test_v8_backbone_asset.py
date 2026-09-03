import hashlib
import json

import pytest

from photocut.algorithms.v8.backbone_asset import (
    BackboneAssetError,
    load_backbone_manifest,
    verify_backbone_asset,
)


def _manifest(weight: bytes, notice: bytes) -> dict:
    return {
        "schema_version": 1,
        "asset_id": "mobilenet-v3-large-imagenet1k-v2",
        "architecture": "mobilenet_v3_large",
        "weight_enum": "MobileNet_V3_Large_Weights.IMAGENET1K_V2",
        "source_url": "https://download.pytorch.org/models/example.pth",
        "source_repository": "https://github.com/pytorch/vision",
        "source_revision": "v0.25.0",
        "license": "BSD-3-Clause",
        "notice_path": "NOTICE.txt",
        "notice_sha256": "sha256:" + hashlib.sha256(notice).hexdigest(),
        "weight_filename": "backbone.pth",
        "weight_size": len(weight),
        "weight_sha256": "sha256:" + hashlib.sha256(weight).hexdigest(),
        "redistributable": True,
    }


def test_backbone_manifest_and_bytes_are_verified_together(tmp_path):
    weight = b"fake torch weights"
    notice = b"BSD license notice"
    (tmp_path / "backbone.pth").write_bytes(weight)
    (tmp_path / "NOTICE.txt").write_bytes(notice)
    manifest_path = tmp_path / "backbone.json"
    manifest_path.write_text(json.dumps(_manifest(weight, notice)), encoding="utf-8")

    manifest = load_backbone_manifest(manifest_path)
    verified = verify_backbone_asset(manifest, manifest_path.parent)

    assert verified == (tmp_path / "backbone.pth").resolve()
    assert manifest.weight_enum.endswith("IMAGENET1K_V2")

    (tmp_path / "backbone.pth").write_bytes(b"changed")
    with pytest.raises(BackboneAssetError, match="size|hash"):
        verify_backbone_asset(manifest, manifest_path.parent)


@pytest.mark.parametrize("missing", ["source_url", "source_repository", "source_revision", "license", "notice_path"])
def test_backbone_manifest_rejects_missing_provenance(tmp_path, missing):
    weight, notice = b"weights", b"notice"
    value = _manifest(weight, notice)
    value.pop(missing)
    path = tmp_path / "backbone.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(BackboneAssetError):
        load_backbone_manifest(path)
