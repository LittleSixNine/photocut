import json
import subprocess
import sys

import pytest


def test_runtime_import_does_not_require_or_import_torch():
    code = """
import json, sys
from photocut.algorithms import v8
import photocut.algorithms.v8.mask_model
print(json.dumps({
    'torch_loaded': 'torch' in sys.modules,
    'v8_loaded': 'photocut.algorithms.v8' in sys.modules,
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == {"torch_loaded": False, "v8_loaded": True}


def test_lraspp_output_and_loss_contract():
    torch = pytest.importorskip("torch")
    from photocut.algorithms.v8.mask_model import build_model, export_logits, segmentation_loss

    torch.manual_seed(1)
    model = build_model()
    model.eval()
    inputs = torch.zeros((1, 3, 64, 64), dtype=torch.float32)
    with torch.no_grad():
        logits = export_logits(model)(inputs)
    assert tuple(logits.shape) == (1, 2, 64, 64)

    target = torch.zeros((1, 64, 64), dtype=torch.long)
    target[:, 16:48, 16:48] = 1
    perfect = torch.full((1, 2, 64, 64), -8.0)
    perfect[:, 0] = torch.where(target == 0, 8.0, -8.0)
    perfect[:, 1] = torch.where(target == 1, 8.0, -8.0)
    wrong = -perfect

    perfect_loss = segmentation_loss(perfect, target)
    wrong_loss = segmentation_loss(wrong, target)
    assert torch.isfinite(perfect_loss)
    assert float(perfect_loss) < float(wrong_loss)


def test_random_init_identity_is_explicit():
    pytest.importorskip("torch")
    from photocut.algorithms.v8.mask_model import backbone_identity, build_model

    model = build_model()
    assert backbone_identity(model) == {"kind": "random_init"}
