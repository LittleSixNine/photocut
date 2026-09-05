import hashlib
import json
from argparse import Namespace
from pathlib import Path

from photocut import cli as photocut_cli
from photocut.algorithms.v7.model_manifest import load_model_manifest, verify_model_artifact
from photocut.algorithms.v8.code_seal import compute_validated_core_sha256
from photocut.algorithms.v8.policy import load_v8_policy
from photocut.data.local import default_dataset_root
from photocut.selector import SELECTOR_NAME, SELECTOR_RECORD_ID, SELECTOR_VERSION


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "photocut"


def test_selector_has_a_formal_versioned_identity():
    assert SELECTOR_NAME == "PhotoCut Selector v4"
    assert SELECTOR_VERSION == "4.0"
    assert SELECTOR_RECORD_ID == "auto-v4"


def test_v82_model_is_redistributable_and_matches_the_sealed_runtime():
    model_dir = PACKAGE_ROOT / photocut_cli.V8_DEFAULT_MODEL_RELATIVE_PATH
    manifest_path = model_dir / "model-manifest.json"
    manifest = load_model_manifest(manifest_path)
    policy = load_v8_policy(
        PACKAGE_ROOT / photocut_cli.V8_DEFAULT_POLICY_RELATIVE_PATH
    )

    assert manifest.license == "MIT"
    assert manifest.redistributable is True
    assert manifest.source_repository == "https://github.com/LittleSixNine/photocut"
    assert verify_model_artifact(manifest, model_dir / manifest.model_filename).is_file()
    assert policy.model_manifest_sha256 == (
        "sha256:" + hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    )
    assert policy.validated_core_sha256 == compute_validated_core_sha256(
        PACKAGE_ROOT
    )


def test_default_selector_preflight_uses_packaged_v8_without_degrading():
    resolved = photocut_cli._resolve_auto_v4_runtime(
        Namespace(
            detector="auto",
            auto_engine=None,
            scene_profile="scanner_white",
            v8_policy=None,
            v8_model_dir=None,
        )
    )

    assert resolved.effective_engine == "v8"
    assert resolved.degraded is False
    assert resolved.v8_runtime is not None


def test_local_config_can_point_to_a_private_sibling(tmp_path, monkeypatch):
    public = tmp_path / "photocut"
    public.mkdir()
    (public / ".photocut-local.json").write_text(
        json.dumps({"dataset_root": "../photocut-private/data"}),
        encoding="utf-8",
    )
    monkeypatch.delenv("PHOTOCUT_DATASET_ROOT", raising=False)

    assert default_dataset_root(public) == public / "../photocut-private/data"


def test_environment_override_has_priority(tmp_path, monkeypatch):
    target = tmp_path / "private-data"
    monkeypatch.setenv("PHOTOCUT_DATASET_ROOT", str(target))

    assert default_dataset_root(tmp_path / "project") == target


def test_package_version_and_v84_model_data_are_declared():
    import photocut

    package_config = (ROOT / "pyproject.toml").read_text()
    assert photocut.__version__ == "0.2.0"
    assert f'version = "{photocut.__version__}"' in package_config
    for suffix in ("json", "onnx", "txt"):
        assert f'"models/v8_4/*.{suffix}"' in package_config
