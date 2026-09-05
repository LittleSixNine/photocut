"""Keep the public deployment allowlist narrow, including small checkpoints."""
import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_public_tree.py"


def load_check():
    spec = importlib.util.spec_from_file_location("public_tree_check", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_only_explicitly_released_models_are_allowed(tmp_path, monkeypatch, capsys):
    check = load_check()
    paths = []
    for relative in sorted(check.ALLOWED_MODEL_FILES):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic deployment placeholder")
        paths.append(path)
    monkeypatch.setattr(check, "ROOT", tmp_path)
    monkeypatch.setattr(check, "tracked_files", lambda: paths)
    assert check.main() == 0
    assert "passed" in capsys.readouterr().out


def test_small_training_checkpoint_and_unapproved_onnx_are_rejected(
    tmp_path, monkeypatch, capsys
):
    check = load_check()
    paths = []
    for name in ("best.pt", "weights.pth", "other.onnx", "best.ckpt", "best.safetensors"):
        path = tmp_path / name
        path.write_bytes(b"small but private")
        paths.append(path)
    monkeypatch.setattr(check, "ROOT", tmp_path)
    monkeypatch.setattr(check, "tracked_files", lambda: paths)
    assert check.main() == 1
    output = capsys.readouterr().out
    for path in paths:
        assert f"unapproved model or training checkpoint: {path.name}" in output


def test_photos_and_source_paths_remain_rejected(tmp_path, monkeypatch, capsys):
    check = load_check()
    photo = tmp_path / "scan.jpg"
    photo.write_bytes(b"synthetic photo placeholder")
    text = tmp_path / "manifest.json"
    text.write_text('{"source": "' + "/" + 'Users/example/input.jpg"}')
    monkeypatch.setattr(check, "ROOT", tmp_path)
    monkeypatch.setattr(check, "tracked_files", lambda: [photo, text])
    assert check.main() == 1
    output = capsys.readouterr().out
    assert "image file is tracked" in output
    assert "personal macOS path" in output
