import pytest

from photocut.algorithms.v8.seed_selection import EdgeSeedSelectionConfig, select_edge_seed_candidates


def _candidate(candidate_id, sources, *, pre=None, selected=None):
    stage_ranks = {}
    if pre is not None:
        stage_ranks["pre_score"] = pre
    if selected is not None:
        stage_ranks["selected"] = selected
    return {"candidate_id": candidate_id, "sources": sources, "stage_ranks": stage_ranks}


def test_seed_selector_keeps_background_sources_and_v7_rank_one_only():
    candidates = (
        _candidate("line-top", ["lines:lsd"], pre=1, selected=1),
        _candidate("line-low", ["lines:lsd"], pre=2, selected=3),
        _candidate("background-low", ["background:border_connected"], selected=7),
        _candidate("mixed", ["background:min_area_rect", "lines:lsd"], selected=4),
        _candidate("contour-low", ["contour:external_quad"], pre=4, selected=5),
    )

    selected = select_edge_seed_candidates(candidates)

    assert [item["candidate_id"] for item in selected] == [
        "line-top", "background-low", "mixed",
    ]


def test_seed_selector_is_stable_deduplicated_and_bounded():
    candidates = [
        _candidate("same", ["background:min_area_rect"], selected=1),
        _candidate("same", ["background:border_connected"], selected=6),
        *(
            _candidate(f"bg-{index}", ["background:border_connected"], selected=index + 2)
            for index in range(8)
        ),
    ]

    selected = select_edge_seed_candidates(
        candidates,
        EdgeSeedSelectionConfig(max_seeds=4),
    )

    assert [item["candidate_id"] for item in selected] == ["same", "bg-0", "bg-1", "bg-2"]


@pytest.mark.parametrize("max_seeds", (0, 9, True))
def test_seed_selector_rejects_unbounded_config(max_seeds):
    with pytest.raises((TypeError, ValueError)):
        EdgeSeedSelectionConfig(max_seeds=max_seeds)
