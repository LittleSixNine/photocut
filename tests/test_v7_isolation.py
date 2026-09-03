"""Characterization tests that freeze the v5.2 surface before v7 work."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def test_legacy_imports_do_not_load_v7_in_fresh_process():
    code = """
import json, sys
before = set(sys.modules)
from photocut import config
from photocut.algorithms.v5_2 import detector as corner_detector
after = set(sys.modules)
print(json.dumps({
    'algorithm_version': config.ALGORITHM_VERSION,
    'v7_before': 'photocut.algorithms.v7' in before,
    'v7_after': 'photocut.algorithms.v7' in after,
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )
    result = json.loads(completed.stdout)
    assert result == {"algorithm_version": "5.2", "v7_before": False, "v7_after": False}


def test_v52_detect_corners_routes_to_detailed(monkeypatch):
    from photocut.algorithms.v5_2 import detector as corner_detector

    expected = ([[1, 2], [3, 4], [5, 6], [7, 8]], [0.1] * 4, [{"legacy": True}] * 4)
    calls = []

    def detailed(*args, **kwargs):
        calls.append((args, kwargs))
        return expected

    monkeypatch.setattr(corner_detector, "detect_corners_detailed", detailed)
    result = corner_detector.detect_corners(np.zeros((8, 8, 3), dtype=np.uint8))

    assert result == expected[0]
    assert len(calls) == 1


def test_importing_v7_has_no_opencv_gui_or_legacy_config_side_effects():
    code = """
import json, sys
from photocut import config
before = dict(vars(config))
from photocut.algorithms import v7
after = dict(vars(config))
new_modules = sorted(name for name in sys.modules if name == 'cv2' or name.startswith(('tkinter', 'PyQt', 'PySide')))
print(json.dumps({'same_config': before == after, 'new_modules': new_modules, 'v7_file': v7.__file__}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    assert result["same_config"] is True
    assert result["new_modules"] == []
    assert Path(result["v7_file"]).name == "__init__.py"


def test_v7_regression_manifest_contains_only_approved_external_identities():
    manifest_path = Path(__file__).parent / "fixtures" / "v7_regression_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert set(manifest) == {"version", "samples"}
    assert manifest["version"] == 1
    assert len(manifest["samples"]) == 3

    expected = {
        "13d289eef09b…": "scanner_outer_frame",
        "a5bcfea9c6c2…": "white_margin",
        "d7688826a899…": "perspective_severe",
    }
    assert {sample["id"]: sample["failure_category"] for sample in manifest["samples"]} == expected
    for sample in manifest["samples"]:
        assert set(sample) == {"id", "failure_category"}
        assert "/" not in sample["id"]
        assert "path" not in sample
