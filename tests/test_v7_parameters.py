import math

import numpy as np
import pytest

from photocut.algorithms.v7.parameters import V7Parameters


def test_defaults_are_bounded_and_hash_is_insertion_order_independent():
    params = V7Parameters()
    assert params.work_edges == (800, 1600)
    assert params.fused_budget == 40
    assert params.refined_budget == 5
    assert params.mode == "safe"
    assert params.scene_profile == "scanner_white"
    a = params.replace(provider_work_limits={"b": 2, "a": 1})
    b = params.replace(provider_work_limits={"a": 1, "b": 2})
    assert a.sha256() == b.sha256()
    assert a.to_dict()["mode"] == "safe"
    assert a.to_dict()["scene_profile"] == "scanner_white"


def test_replace_rejects_unknown_fields_and_mode():
    with pytest.raises((TypeError, ValueError)):
        V7Parameters(unknown=1)
    with pytest.raises(ValueError):
        V7Parameters(mode="balanced")
    with pytest.raises(ValueError):
        V7Parameters(scene_profile="multiple_photos")


def test_scene_profile_is_explicit_and_changes_parameter_identity():
    scanner = V7Parameters(scene_profile="scanner_white")
    generic = V7Parameters(scene_profile="generic_single")

    assert generic.to_dict()["scene_profile"] == "generic_single"
    assert scanner.sha256() != generic.sha256()


@pytest.mark.parametrize("kwargs", [
    {"max_input_pixels": True}, {"fused_budget": -1}, {"refined_budget": 41},
    {"work_edges": (799, 1600)}, {"work_edges": (800, 1601)},
    {"edge_band": math.nan}, {"dedup_distance": math.inf},
    {"safe_threshold": 1.1}, {"aggressive_threshold": -0.1},
])
def test_parameters_reject_nonfinite_bool_or_out_of_range_values(kwargs):
    with pytest.raises((TypeError, ValueError)):
        V7Parameters(**kwargs)


def test_every_resource_budget_changes_hash():
    base = V7Parameters()
    for field, value in (("max_input_pixels", base.max_input_pixels + 1),
                         ("fused_budget", base.fused_budget - 1),
                         ("refined_budget", base.refined_budget - 1),
                         ("provider_work_limits", {"contour": 999}),
                         ("provider_timeout_ms", {"contour": 999})):
        assert base.replace(**{field: value}).sha256() != base.sha256()


def test_numpy_scalars_are_accepted_but_numpy_bool_is_not():
    params = V7Parameters(
        max_input_pixels=np.int64(1_000_000),
        edge_band=np.float32(0.01),
        safe_threshold=np.float64(0.8),
    )
    assert params.max_input_pixels == 1_000_000
    assert params.edge_band == pytest.approx(0.01)
    with pytest.raises((TypeError, ValueError)):
        V7Parameters(max_input_pixels=np.bool_(True))
