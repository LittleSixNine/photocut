"""Pure confirmation view model for opt-in v7 results."""
from __future__ import annotations

from typing import Any, Mapping

from photocut.confirmation.model import ConfirmationState
from .types import DetectionStatus


_BLOCKED = {DetectionStatus.ERROR, DetectionStatus.NO_PRIMARY_PHOTO, DetectionStatus.CANCELLED}
_RISK_LABELS = {
    "weak_edge": "边缘置信度较低",
    "suspected_outer_frame": "可能检测到扫描器外框",
    "provider_timeout": "部分检测器超时",
    "provider_error": "部分检测器失败",
    "multiple_primary_ambiguity": "存在多个可能的主体照片",
    "no_photo_evidence": "未发现明确照片主体",
}


def _integer_corners(corners: Any) -> list[list[int]] | None:
    if corners is None:
        return None
    try:
        return [[int(round(float(point[0]))), int(round(float(point[1])))] for point in corners]
    except (TypeError, ValueError, IndexError):
        return None


class V7ConfirmationViewModel:
    @classmethod
    def from_entry(cls, entry: Mapping[str, Any], *, image_size: tuple[int, int]) -> "V7ConfirmationViewModel":
        """Build a pure model from one projected ``corners_info`` entry."""
        class _Result:
            pass
        result = _Result()
        result.status = entry.get("detection_status", DetectionStatus.ERROR)
        result.corners = entry.get("algorithm_boundary_corners") or entry.get("corners")
        result.alternate_corners = entry.get("alternate_corners")
        result.overall_confidence = entry.get("overall_confidence")
        result.edge_confidences = tuple(entry.get("edge_confidences", entry.get("confidences", ())))
        result.risks = tuple(entry.get("risks", ()))
        result.top1_sources = tuple(entry.get("candidate_sources", ()))
        result.alternate_sources = tuple(entry.get("alternate_sources", ()))
        return cls(result, image_size=image_size, v52_corners=entry.get("v52_corners"))

    def __init__(self, result: Any, *, image_size: tuple[int, int],
                 v52_corners: Any = None):
        self.result = result
        self.image_size = tuple(image_size)
        self.status = result.status if isinstance(result.status, DetectionStatus) else DetectionStatus(result.status)
        self._top1 = _integer_corners(result.corners)
        self._alternate = _integer_corners(result.alternate_corners)
        self._v52 = _integer_corners(v52_corners)
        self.selection = "top1"
        initial = self._top1 or self._alternate or self._v52
        self._algorithm = [point[:] for point in initial] if initial else []
        self.state = ConfirmationState(self._algorithm, [point[:] for point in self._algorithm], self.image_size) if len(self._algorithm) == 4 else None
        self._dragged = False
        self._skipped = False
        self._skip_reason: str | None = None

    @property
    def algorithm_corners(self) -> list[list[int]]:
        return [point[:] for point in self._algorithm]

    @property
    def work_corners(self) -> list[list[int]]:
        return [point[:] for point in (self.state.work_corners if self.state else [])]

    @property
    def selected_corners(self) -> list[list[int]]:
        return self.work_corners

    @property
    def skip_reason(self) -> str | None:
        return self._skip_reason

    @property
    def confidence(self) -> float | None:
        return self.result.overall_confidence

    @property
    def candidate_sources(self) -> tuple[str, ...]:
        if self.selection == "alternate":
            return tuple(self.result.alternate_sources)
        if self.selection == "v52":
            return ("v5.2",)
        return tuple(self.result.top1_sources)

    @property
    def operation(self) -> str:
        if self._skipped:
            return "skipped"
        if self._dragged:
            return "dragged"
        if self.selection == "alternate":
            return "accepted_alternate"
        if self.selection == "v52":
            return "fallback"
        return "direct_top1"

    @property
    def can_confirm(self) -> bool:
        return self.status not in _BLOCKED and not self._skipped and self.state is not None

    @property
    def status_label(self) -> str:
        return {
            DetectionStatus.V7_RECOMMENDED: "recommended",
            DetectionStatus.V7_LOW_CONFIDENCE: "low_confidence",
            DetectionStatus.V52_FALLBACK: "v5.2 fallback",
            DetectionStatus.NO_PRIMARY_PHOTO: "no primary photo",
            DetectionStatus.CANCELLED: "cancelled",
            DetectionStatus.ERROR: "error",
        }[self.status]

    @property
    def risk_labels(self) -> tuple[str, ...]:
        return tuple(_RISK_LABELS.get(str(risk), str(risk)) for risk in self.result.risks)

    @property
    def edge_colors(self) -> tuple[str, ...]:
        values = tuple(self.result.edge_confidences or ())
        threshold = .5
        return tuple("green" if value >= .75 else "yellow" if value >= threshold else "red" for value in values[:4])

    def _ensure_toggle_allowed(self) -> None:
        if self._dragged:
            raise ValueError("reset before switching candidates after dragging")

    def _switch(self, selection: str, corners: list[list[int]] | None) -> None:
        self._ensure_toggle_allowed()
        if corners is None:
            raise ValueError(f"{selection} candidate is unavailable")
        self.selection = selection
        self._algorithm = [point[:] for point in corners]
        self.state = ConfirmationState(self.algorithm_corners, self.algorithm_corners, self.image_size)

    def toggle_alternate(self) -> str:
        if self.selection == "alternate":
            self._switch("top1", self._top1)
        else:
            self._switch("alternate", self._alternate)
        return self.selection

    def toggle_v52(self) -> str:
        if self.selection == "v52":
            self._switch("top1", self._top1)
        else:
            self._switch("v52", self._v52)
        return self.selection

    def select_corner(self, index: int) -> None:
        if self.state is None:
            raise ValueError("no candidate corners available")
        self.state.selected = int(index)

    def move_selected(self, dx: int, dy: int) -> None:
        if self.state is None:
            raise ValueError("no candidate corners available")
        self.state.move_selected(dx, dy)
        self._dragged = True

    def reset(self) -> None:
        if self.state is None:
            return
        self.state.reset()
        self._dragged = False
        self._skipped = False
        self._skip_reason = None

    def confirm(self) -> tuple[str, list[list[int]]]:
        if not self.can_confirm:
            raise ValueError("this detection state cannot be confirmed")
        return self.operation, self.work_corners

    def skip(self, reason: str) -> str:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("skip reason is required")
        self._skipped = True
        self._skip_reason = reason
        return "skipped"


__all__ = ["V7ConfirmationViewModel"]
