import math

from photocut.algorithms.v8.evaluation import (
    candidate_feasibility_gate,
    first_strict_rank,
    summarize_candidate_oracle,
)


def test_first_strict_rank_uses_max_normalized_corner_error():
    truth = ((0, 0), (100, 0), (100, 100), (0, 100))
    candidates = (
        ((0, 0), (101, 0), (100, 100), (0, 100)),
        ((0, 0), (100.5, 0), (100, 100), (0, 100)),
    )

    assert first_strict_rank(candidates, truth, image_size=(100, 100)) == 2
    assert first_strict_rank((), truth, image_size=(100, 100)) is None


def test_candidate_oracle_summary_separates_full_pool_from_budgets_and_subsets():
    rows = (
        {"training_subset": "fit", "v7_strict": True, "edge_first_strict_rank": None, "mask_strict": False},
        {"training_subset": "fit", "v7_strict": False, "edge_first_strict_rank": 2, "mask_strict": False},
        {"training_subset": "calibration", "v7_strict": False, "edge_first_strict_rank": 9, "mask_strict": False},
        {"training_subset": "calibration", "v7_strict": False, "edge_first_strict_rank": None, "mask_strict": True},
        {"training_subset": "calibration", "v7_strict": False, "edge_first_strict_rank": None, "mask_strict": False},
    )

    summary = summarize_candidate_oracle(rows, edge_budgets=(1, 8, 16))

    assert summary["sample_count"] == 5
    assert summary["v7_oracle_count"] == 1
    assert summary["edge_oracle_count"] == 2
    assert summary["mask_oracle_count"] == 1
    assert summary["union_oracle_count"] == 4
    assert summary["edge_unique_vs_v7_count"] == 2
    assert summary["mask_unique_vs_v7_edge_count"] == 1
    assert summary["edge_budget_union_counts"] == {"1": 2, "8": 3, "16": 4}
    assert summary["subsets"]["calibration"]["union_oracle_count"] == 2
    assert summary["subsets"]["fit"]["union_oracle_count"] == 2


def test_candidate_feasibility_gate_enforces_full_and_calibration_thresholds():
    rows = []
    for index in range(220):
        calibration = index < 44
        local_index = index if calibration else index - 44
        v7_limit = 30 if calibration else 131
        union_limit = 43 if calibration else 167
        v7 = local_index < v7_limit
        edge = v7_limit <= local_index < union_limit
        rows.append({
            "training_subset": "calibration" if calibration else "fit",
            "v7_strict": v7,
            "edge_first_strict_rank": 1 if edge else None,
            "mask_strict": False,
        })
    summary = summarize_candidate_oracle(rows)

    gate = candidate_feasibility_gate(
        summary,
        frozen_accessed=False,
        source_modified_count=0,
    )

    assert summary["union_oracle_count"] == 210
    assert gate["required_union_count"] == math.ceil(0.95 * 220)
    assert gate["passed"] is True

    failed = candidate_feasibility_gate(
        summary,
        frozen_accessed=True,
        source_modified_count=0,
    )
    assert failed["passed"] is False
    assert "frozen_accessed" in failed["failed_conditions"]
