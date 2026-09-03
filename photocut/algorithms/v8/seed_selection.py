"""Truth-independent coarse-seed selection for scanner-white edge search."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class EdgeSeedSelectionConfig:
    max_seeds: int = 8

    def __post_init__(self) -> None:
        if type(self.max_seeds) is not int or not 1 <= self.max_seeds <= 8:
            raise ValueError("max_seeds must be an integer in [1, 8]")


def _eligible(candidate: Mapping[str, Any]) -> bool:
    sources = candidate.get("sources", ())
    if isinstance(sources, str):
        sources = (sources,)
    if not isinstance(sources, (list, tuple)) or any(not isinstance(source, str) for source in sources):
        raise ValueError("candidate sources must be strings")
    ranks = candidate.get("stage_ranks", {})
    if not isinstance(ranks, Mapping):
        raise ValueError("candidate stage_ranks must be a mapping")
    background = any(source.startswith("background:") for source in sources)
    v7_top1 = ranks.get("selected") == 1 or ranks.get("pre_score") == 1
    return background or v7_top1


def select_edge_seed_candidates(
    candidates: Iterable[Mapping[str, Any]],
    config: EdgeSeedSelectionConfig | None = None,
) -> tuple[Mapping[str, Any], ...]:
    """Keep V7 background candidates plus its rank-one safety seed.

    Input order is retained so the rule cannot silently introduce a new V7
    ranking. Duplicate candidate identities are evaluated once.
    """
    config = config or EdgeSeedSelectionConfig()
    if not isinstance(config, EdgeSeedSelectionConfig):
        raise TypeError("config must be EdgeSeedSelectionConfig")
    selected = []
    seen: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise TypeError("edge seed candidate must be a mapping")
        candidate_id = candidate.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError("edge seed candidate_id is required")
        if candidate_id in seen or not _eligible(candidate):
            continue
        seen.add(candidate_id)
        selected.append(candidate)
        if len(selected) == config.max_seeds:
            break
    return tuple(selected)


__all__ = [
    "EdgeSeedSelectionConfig",
    "select_edge_seed_candidates",
]
