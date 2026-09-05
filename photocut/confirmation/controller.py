"""Pure entry-level confirmation editing shared by native and web clients."""
from __future__ import annotations

import copy
import json
import math
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

import numpy as np

from photocut.config import ALGORITHM_VERSION, V7_ALGORITHM_VERSION
from photocut.confirmation.identity import FinalizedDetectionEvidence
from photocut.confirmation.model import ConfirmationState, VALID_ZOOM_LEVELS
from photocut.confirmation.version import WEB_GUI_VERSION
from photocut.algorithms.v7.gui_model import V7ConfirmationViewModel


class ConfirmationActionError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ConfirmationAction:
    action_id: str
    expected_revision: int
    kind: str
    payload: Mapping[str, object]


class ConfirmationSessionError(RuntimeError):
    def __init__(self, code: str, message: str, scope: str = "action"):
        super().__init__(message)
        if scope not in {"action", "image", "session"}:
            raise ValueError("invalid confirmation error scope")
        self.code = code
        self.scope = scope


@dataclass
class LoadedConfirmationItem:
    entry: dict
    image: np.ndarray
    image_token: str
    display_path: str
    editor: "ConfirmationEntryController"
    finalized_evidence: FinalizedDetectionEvidence | None
    previous_annotation: dict | None
    started_at: float
    is_revision: bool


class ConfirmationSessionBackend(Protocol):
    def count(self) -> int:
        raise NotImplementedError

    def load(self, index: int) -> LoadedConfirmationItem:
        raise NotImplementedError

    def load_preview(self, index: int) -> tuple[np.ndarray, str]:
        raise NotImplementedError

    def checkpoint(self, item: LoadedConfirmationItem) -> None:
        raise NotImplementedError

    def commit(self, item: LoadedConfirmationItem, duration_ms: int, gui_version: str) -> dict:
        raise NotImplementedError

    def skip(self, item: LoadedConfirmationItem, reason: str) -> None:
        raise NotImplementedError


def _valid_quad(value: Any, image_size: tuple[int, int]):
    from photocut.algorithms.v7.geometry import GeometryError, validate_quad

    try:
        return validate_quad(value, image_size=image_size)
    except (GeometryError, TypeError, ValueError):
        return None


class AutoV4ConfirmationViewModel:
    """Pure candidate-selection model for auto-v4 confirmation entries."""

    def __init__(self, entry, *, image_size):
        self.entry = entry
        self.image_size = tuple(image_size)
        saved_selection = entry.get("confirmation_selected_candidate_id")
        saved_draft = entry.get("boundary_corners", entry.get("corners"))
        self._candidates = _auto_v4_confirmation_candidates(entry, self.image_size)
        self._v52 = _auto_v4_candidate(
            "v52:gui", ALGORITHM_VERSION, entry.get("v52_corners"),
            self.image_size, ("v5.2",),
        )
        self._index = 0
        self._before_v52_index = 0
        self.selection = self._candidates[0]["candidate_id"] if self._candidates else "none"
        self._algorithm = []
        self.state = None
        self._dragged = False
        self._skipped = False
        self._skip_reason = None
        if self._candidates:
            self._switch_candidate(self._candidates[0], allow_dragged=True)
            saved_candidate = next(
                (
                    candidate
                    for candidate in self._candidates
                    if candidate["candidate_id"] == saved_selection
                ),
                self._v52 if self._v52 and self._v52["candidate_id"] == saved_selection else None,
            )
            if saved_candidate is not None:
                if saved_candidate is self._v52:
                    self._before_v52_index = self._index
                else:
                    self._index = self._candidates.index(saved_candidate)
                self._switch_candidate(saved_candidate, allow_dragged=True)
            draft = _valid_quad(saved_draft, self.image_size)
            if draft is not None:
                restored = [
                    [int(round(x)), int(round(y))]
                    for x, y in draft
                ]
                self.state = ConfirmationState(
                    self.algorithm_corners, restored, self.image_size,
                )
                self._dragged = bool(self.state.adjusted_corner_indices)

    @property
    def algorithm_corners(self):
        return [point[:] for point in self._algorithm]

    @property
    def work_corners(self):
        return [point[:] for point in (self.state.work_corners if self.state else [])]

    @property
    def selected_corners(self):
        return self.work_corners

    @property
    def skip_reason(self):
        return self._skip_reason

    @property
    def confidence(self):
        return self.entry.get("overall_confidence")

    @property
    def candidate_sources(self):
        current = self._current_candidate()
        return tuple(current.get("sources", ())) if current else ()

    @property
    def operation(self):
        if self._skipped:
            return "skipped"
        if self._dragged:
            return "dragged"
        if self.selection.startswith("v52:"):
            return "fallback"
        if self.selection == self.entry.get("confirmation_primary_candidate_id"):
            return "direct_primary"
        return "accepted_candidate"

    @property
    def can_confirm(self):
        return not self._skipped and self.state is not None

    @property
    def can_toggle_alternate(self):
        return len(self._candidates) > 1

    @property
    def can_toggle_v52(self):
        return self._v52 is not None

    @property
    def status_label(self):
        return f"{self.entry.get('auto_v4_status', 'manual_review')} / {self.selection}"

    @property
    def risk_labels(self):
        return tuple(str(value) for value in self.entry.get("risks", ()) if value != "none")

    @property
    def edge_colors(self):
        values = tuple(self.entry.get("edge_confidences", self.entry.get("confidences", ())))
        return tuple("green" if value >= .75 else "yellow" if value >= .5 else "red" for value in values[:4])

    def _current_candidate(self):
        if self.selection.startswith("v52:"):
            return self._v52
        if not self._candidates:
            return None
        return self._candidates[self._index]

    def _ensure_toggle_allowed(self):
        if self._dragged or (self.state and self.state.adjusted_corner_indices):
            raise ValueError("reset before switching candidates after dragging")

    def _switch_candidate(self, candidate, *, allow_dragged=False):
        if not allow_dragged:
            self._ensure_toggle_allowed()
        if candidate is None:
            raise ValueError("candidate is unavailable")
        self.selection = candidate["candidate_id"]
        self._algorithm = [point[:] for point in candidate["corners"]]
        self.state = ConfirmationState(
            self.algorithm_corners, self.algorithm_corners, self.image_size,
        )
        self.entry["confirmation_selected_candidate_id"] = self.selection
        self.entry["confirmation_selected_algorithm_version"] = candidate["algorithm_version"]

    def toggle_alternate(self):
        self._ensure_toggle_allowed()
        if not self._candidates:
            raise ValueError("candidate is unavailable")
        if self.selection.startswith("v52:"):
            self._index = (self._before_v52_index + 1) % len(self._candidates)
        else:
            self._index = (self._index + 1) % len(self._candidates)
        self._switch_candidate(self._candidates[self._index])
        return self.selection

    def toggle_v52(self):
        self._ensure_toggle_allowed()
        if self.selection.startswith("v52:"):
            if not self._candidates:
                raise ValueError("primary candidate is unavailable")
            self._index = self._before_v52_index
            self._switch_candidate(self._candidates[self._index])
        else:
            if self._v52 is None:
                raise ValueError("v5.2 candidate is unavailable")
            self._before_v52_index = self._index
            self._switch_candidate(self._v52)
        return self.selection

    def select_corner(self, index):
        if self.state is None:
            raise ValueError("no candidate corners available")
        self.state.selected = int(index)

    def move_selected(self, dx, dy):
        if self.state is None:
            raise ValueError("no candidate corners available")
        self.state.move_selected(dx, dy)
        self._dragged = True

    def reset(self):
        self._dragged = False
        self._skipped = False
        self._skip_reason = None
        if self._candidates:
            self._index = 0
            self._switch_candidate(self._candidates[0], allow_dragged=True)

    def confirm(self):
        if not self.can_confirm:
            raise ValueError("this detection state cannot be confirmed")
        return self.operation, self.work_corners

    def skip(self, reason):
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("skip reason is required")
        self._skipped = True
        self._skip_reason = reason
        return "skipped"


def _auto_v4_candidate(candidate_id, algorithm_version, corners, image_size, sources=()):
    if not isinstance(candidate_id, str) or not candidate_id.strip():
        return None
    quad = _valid_quad(corners, image_size)
    if quad is None:
        return None
    return {
        "candidate_id": candidate_id,
        "algorithm_version": str(algorithm_version),
        "corners": [[int(round(x)), int(round(y))] for x, y in quad],
        "sources": tuple(str(value) for value in sources),
    }


def _auto_v4_confirmation_candidates(entry, image_size):
    primary = _auto_v4_candidate(
        entry.get("confirmation_primary_candidate_id", "auto-v4:primary"),
        entry.get("confirmation_selected_algorithm_version")
        or entry.get("algorithm_version", V7_ALGORITHM_VERSION),
        entry.get("algorithm_boundary_corners") or entry.get("corners"),
        image_size, entry.get("candidate_sources", ()),
    )
    candidates = [primary] if primary is not None else []
    if entry.get("auto_v4_status") == "automatic":
        return candidates

    audits = entry.get("candidate_audit")
    if not isinstance(audits, list):
        audits = []
    ranked = []
    for audit in audits:
        if not isinstance(audit, dict):
            continue
        rank = audit.get("stage_ranks", {}).get("selected")
        rank = rank if type(rank) is int else 10**9
        ranked.append((rank, str(audit.get("candidate_id", "")), audit))
    for _, raw_id, audit in sorted(ranked)[:2]:
        if not raw_id:
            continue
        candidate_id = raw_id if raw_id.startswith("v7:") else f"v7:{raw_id}"
        candidate = _auto_v4_candidate(
            candidate_id, V7_ALGORITHM_VERSION,
            audit.get("adopted_refined_corners")
            or audit.get("pre_topk_corners")
            or audit.get("original_legal_corners"),
            image_size, audit.get("sources", ()),
        )
        if candidate is None:
            continue
        duplicate = False
        width, height = image_size
        for current in candidates:
            if candidate_id == current["candidate_id"]:
                duplicate = True
                break
            maximum = max(
                math.hypot(
                    (left[0] - right[0]) / max(1, width - 1),
                    (left[1] - right[1]) / max(1, height - 1),
                )
                for left, right in zip(candidate["corners"], current["corners"])
            )
            if maximum <= 1e-6:
                duplicate = True
                break
        if not duplicate:
            candidates.append(candidate)
    return candidates


def _v7_view_model_for_entry(entry, image_size):
    """Create the pure v7 confirmation model for integration/testing."""
    if entry.get("auto_cascade_version") == "auto-v4":
        return AutoV4ConfirmationViewModel(entry, image_size=image_size)
    if entry.get("detector_requested") == "v8.4":
        projected = dict(entry)
        projected["detection_status"] = (
            "v7_low_confidence" if entry.get("success") and
            entry.get("detection_status") == "candidate_requires_confirmation" else "error"
        )
        return V7ConfirmationViewModel.from_entry(projected, image_size=image_size)
    if entry.get("detector_used", entry.get("detector")) not in {"v7", "manual_review"}:
        return None
    try:
        from photocut.algorithms.v7.types import DetectionStatus
        status = entry.get("detection_status")
        if status is not None and status not in {item.value for item in DetectionStatus}:
            return None
    except Exception:
        return None
    return V7ConfirmationViewModel.from_entry(entry, image_size=image_size)


def build_initial_confirmation_state(entry, image_size, model):
    algorithm = entry.get(
        "algorithm_boundary_corners", entry.get("algorithm_corners", entry.get("corners", []))
    )
    draft = entry.get("boundary_corners", entry.get("corners", algorithm))
    render_algorithm = algorithm or [[0, 0]] * 4
    state = ConfirmationState(
        [list(map(int, point)) for point in render_algorithm],
        [list(map(int, point)) for point in (draft or render_algorithm)],
        image_size,
    )
    if isinstance(model, AutoV4ConfirmationViewModel) and model.state is not None:
        return model.state
    if model is not None and not isinstance(model, AutoV4ConfirmationViewModel):
        model.state = state
        model._algorithm = [point[:] for point in state.algorithm_corners]
        model._dragged = bool(state.adjusted_corner_indices)
    return state


def candidate_snapshot(model, entry):
    algorithm_version = entry.get("confirmation_selected_algorithm_version")
    if entry.get("detector_requested") == "v8.4":
        algorithm_version = "8.4"
    elif isinstance(model, V7ConfirmationViewModel):
        algorithm_version = (
            ALGORITHM_VERSION
            if model.selection == "v52"
            else V7_ALGORITHM_VERSION
        )
    return {
        "id": getattr(model, "selection", "primary"),
        "algorithm_version": algorithm_version,
        "status": getattr(model, "status_label", "legacy"),
        "sources": list(getattr(model, "candidate_sources", ())),
    }


def capability_snapshot(model, state, entry):
    dirty = bool(state.adjusted_corner_indices)
    dragged = bool(getattr(model, "_dragged", False))
    legal_draft = _valid_quad(state.work_corners, state.image_size) is not None
    can_confirm = legal_draft and bool(getattr(model, "can_confirm", True))
    can_toggle_alternate = bool(getattr(model, "can_toggle_alternate", False))
    can_toggle_v52 = bool(getattr(model, "can_toggle_v52", False))
    if isinstance(model, V7ConfirmationViewModel):
        can_toggle_alternate = _valid_quad(
            entry.get("alternate_corners"), state.image_size
        ) is not None
        can_toggle_v52 = _valid_quad(
            entry.get("v52_corners"), state.image_size
        ) is not None
    return {
        "confirm": can_confirm,
        "candidate": can_toggle_alternate and not dirty and not dragged,
        "v52": can_toggle_v52 and not dirty and not dragged,
        "reset": dirty or dragged or getattr(model, "selection", "primary") not in ("primary", "top1"),
        "skip": model is not None,
        "move": can_confirm,
    }


class ConfirmationEntryController:
    def __init__(self, entry: dict, image_size: tuple[int, int]):
        self.entry = entry
        self.image_size = tuple(image_size)
        self.model = _v7_view_model_for_entry(entry, image_size)
        self.state = build_initial_confirmation_state(entry, image_size, self.model)

    def select_corner(self, index: int) -> None:
        if index not in range(4):
            raise ConfirmationActionError("invalid_corner", "corner index must be 0..3")
        if self.model is not None:
            self.model.select_corner(index)
            self.state = self.model.state
        else:
            self.state.selected = index

    def move_selected(self, dx: int, dy: int) -> None:
        try:
            if self.model is not None:
                self.model.move_selected(int(dx), int(dy))
                self.state = self.model.state
            else:
                self.state.move_selected(int(dx), int(dy))
        except ValueError as exc:
            raise ConfirmationActionError("move_rejected", str(exc)) from exc

    def set_selected_corner(self, x: int, y: int) -> None:
        if self.state.selected not in range(4):
            raise ConfirmationActionError("corner_not_selected", "select a corner first")
        current_x, current_y = self.state.work_corners[self.state.selected]
        self.move_selected(int(x) - current_x, int(y) - current_y)

    def set_zoom(self, zoom: int) -> None:
        self.state.set_zoom(zoom)

    def _toggle(self, method_name: str) -> None:
        if self.model is None:
            raise ConfirmationActionError("candidate_unavailable", "candidate is unavailable")
        try:
            getattr(self.model, method_name)()
        except ValueError as exc:
            code = "reset_required" if (
                self.state.adjusted_corner_indices or getattr(self.model, "_dragged", False)
            ) else "candidate_unavailable"
            raise ConfirmationActionError(code, str(exc)) from exc
        self.state = self.model.state

    def toggle_candidate(self) -> None:
        self._toggle("toggle_alternate")

    def toggle_v52(self) -> None:
        self._toggle("toggle_v52")

    def reset(self) -> None:
        if self.model is not None:
            self.model.reset()
            if self.model.state is not None:
                self.state = self.model.state
        else:
            self.state.reset()
        self.state.selected = -1

    def skip(self, reason: str) -> None:
        if self.model is None:
            raise ConfirmationActionError("skip_unavailable", "skip is unavailable")
        self.model.skip(reason)

    def confirm(self):
        if _valid_quad(self.state.work_corners, self.state.image_size) is None:
            raise ConfirmationActionError(
                "confirm_unavailable", "this detection state cannot be confirmed"
            )
        if self.model is None:
            return None, [point[:] for point in self.state.work_corners]
        try:
            return self.model.confirm()
        except ValueError as exc:
            raise ConfirmationActionError("confirm_unavailable", str(exc)) from exc

    def snapshot(self) -> dict:
        return {
            "image_size": list(self.image_size),
            "corners": [point[:] for point in self.state.work_corners],
            "algorithm_corners": [point[:] for point in self.state.algorithm_corners],
            "selected_corner": self.state.selected,
            "zoom": self.state.zoom,
            "dirty": bool(self.state.adjusted_corner_indices),
            "adjusted_corner_indices": sorted(self.state.adjusted_corner_indices),
            "candidate": candidate_snapshot(self.model, self.entry),
            "risks": list(getattr(self.model, "risk_labels", ())),
            "capabilities": capability_snapshot(self.model, self.state, self.entry),
        }


class ConfirmationSessionController:
    """Serialize confirmation actions for one locally served confirmation session."""

    _ALLOWED_PAYLOADS = {
        "select_corner": {"index"},
        "move": {"dx", "dy"},
        "set_corner": {"index", "x", "y"},
        "set_zoom": {"zoom"},
        "candidate": set(),
        "v52": set(),
        "reset": set(),
        "skip": {"reason"},
        "previous": set(),
        "next": set(),
        "confirm": set(),
        "pause": set(),
        "quit": set(),
    }
    _RESULT_CACHE_SIZE = 256

    def __init__(self, backend: ConfirmationSessionBackend, gui_version: str):
        self.backend = backend
        self.gui_version = gui_version
        self.initial_zoom = 2 if gui_version == WEB_GUI_VERSION else 4
        self.revision = 0
        self.index = 0
        self.status = "active"
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.results: OrderedDict[str, tuple[str, dict]] = OrderedDict()
        self._current_item = backend.load(0)
        self._current_item.editor.set_zoom(self.initial_zoom)

    @property
    def current_item(self) -> LoadedConfirmationItem:
        return self._current_item

    def snapshot(self) -> dict:
        with self.lock:
            item = self.current_item
            editor = item.editor.snapshot()
            evidence = item.finalized_evidence
            return {
                "revision": self.revision,
                "session": {"status": self.status, "readonly": False},
                "progress": {
                    "index": self.index,
                    "number": self.index + 1,
                    "total": self.backend.count(),
                },
                "image": {
                    "token": item.image_token,
                    "filename": item.entry["filename"],
                    "width": item.image.shape[1],
                    "height": item.image.shape[0],
                },
                "editor": editor,
                "identity": {
                    "gui": self.gui_version,
                    "algorithm": getattr(evidence, "algorithm_version", None),
                    "requested": getattr(evidence, "detector_requested", None),
                    "used": getattr(evidence, "detector_used", None),
                    "detection_id": getattr(evidence, "detection_id", None),
                },
                "storage": {
                    "dirty": editor["dirty"],
                    "formal": bool(item.entry.get("confirmed")),
                },
                "error": None,
            }

    def current_image(self) -> tuple[np.ndarray, str]:
        with self.lock:
            return self.current_item.image, self.current_item.image_token

    def next_preview_source(self) -> tuple[np.ndarray, str]:
        with self.lock:
            if self.index + 1 >= self.backend.count():
                raise ConfirmationSessionError(
                    "no_next_image", "there is no next preview", "image"
                )
            source_index = self.index
            source_token = self.current_item.image_token
            target = source_index + 1
        image, image_token = self.backend.load_preview(target)
        with self.lock:
            if (
                self.index != source_index
                or self.current_item.image_token != source_token
            ):
                del image
                raise ConfirmationSessionError(
                    "stale_preview",
                    "confirmation image changed during next preview load",
                    "image",
                )
            return image, image_token

    def wait_until_stopped(self) -> None:
        self.stop_event.wait()

    def technical_details(self) -> dict:
        with self.lock:
            item = self.current_item
            previous_gui = None
            if item.previous_annotation is not None:
                previous_gui = item.previous_annotation.get("gui_version") or "legacy / unknown"
            evidence = item.finalized_evidence
            return {
                "path": item.display_path,
                "source_sha256": item.entry.get("source_sha256"),
                "detection_id": getattr(evidence, "detection_id", None),
                "previous_gui": previous_gui,
                "candidate_audit": copy.deepcopy(item.entry.get("candidate_audit", [])),
            }

    def _advance(self, delta: int) -> None:
        target = self.index + delta
        if not 0 <= target < self.backend.count():
            raise ConfirmationSessionError(
                "navigation_boundary", "navigation target is unavailable"
            )
        self.backend.checkpoint(self.current_item)
        next_item = self.backend.load(target)
        self._replace_current(next_item, target)

    def _replace_current(self, next_item: LoadedConfirmationItem, target: int) -> None:
        previous = self._current_item
        next_item.editor.set_zoom(self.initial_zoom)
        self._current_item = next_item
        self.index = target
        del previous

    def _preload_terminal_next(self) -> LoadedConfirmationItem | None:
        if self.index + 1 >= self.backend.count():
            return None
        return self.backend.load(self.index + 1)

    def _apply(self, action: ConfirmationAction) -> None:
        editor = self.current_item.editor
        payload = action.payload
        if action.kind == "select_corner":
            editor.select_corner(int(payload["index"]))
        elif action.kind == "move":
            editor.move_selected(int(payload["dx"]), int(payload["dy"]))
        elif action.kind == "set_corner":
            editor.select_corner(int(payload["index"]))
            editor.set_selected_corner(int(payload["x"]), int(payload["y"]))
        elif action.kind == "set_zoom":
            editor.set_zoom(int(payload["zoom"]))
        elif action.kind == "candidate":
            editor.toggle_candidate()
        elif action.kind == "v52":
            editor.toggle_v52()
        elif action.kind == "reset":
            editor.reset()
            self.backend.checkpoint(self.current_item)
        elif action.kind == "skip":
            if editor.model is None:
                raise ConfirmationActionError("skip_unavailable", "skip is unavailable")
            next_item = self._preload_terminal_next()
            try:
                self.backend.skip(self.current_item, payload["reason"])
            except Exception:
                del next_item
                raise
            editor.skip(payload["reason"])
            self._advance_after_terminal_action(next_item)
        elif action.kind == "previous":
            self._advance(-1)
        elif action.kind == "next":
            self._advance(1)
        elif action.kind == "confirm":
            next_item = self._preload_terminal_next()
            elapsed = round((time.perf_counter() - self.current_item.started_at) * 1000)
            try:
                self.backend.commit(self.current_item, elapsed, self.gui_version)
            except Exception:
                del next_item
                raise
            self._advance_after_terminal_action(next_item)
        elif action.kind in {"pause", "quit"}:
            self.backend.checkpoint(self.current_item)
            self.status = action.kind
            self.stop_event.set()
        else:
            raise ConfirmationSessionError("unknown_action", "unknown confirmation action")

    def _advance_after_terminal_action(self, next_item: LoadedConfirmationItem | None) -> None:
        if next_item is None:
            self.status = "completed"
            self.stop_event.set()
            return
        self._replace_current(next_item, self.index + 1)

    @classmethod
    def _fingerprint(cls, action: ConfirmationAction) -> str:
        return json.dumps(
            {
                "expected_revision": action.expected_revision,
                "kind": action.kind,
                "payload": action.payload,
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def _validate_action(cls, action: ConfirmationAction) -> None:
        if not isinstance(action, ConfirmationAction):
            raise ConfirmationSessionError("invalid_action", "invalid confirmation action")
        if not isinstance(action.action_id, str) or not 1 <= len(action.action_id) <= 128:
            raise ConfirmationSessionError("invalid_action_id", "invalid action ID")
        if type(action.expected_revision) is not int:
            raise ConfirmationSessionError("invalid_revision", "revision must be an integer")
        if not isinstance(action.kind, str):
            raise ConfirmationSessionError("invalid_action", "invalid action kind or payload")
        expected_keys = cls._ALLOWED_PAYLOADS.get(action.kind)
        if (
            expected_keys is None
            or not isinstance(action.payload, Mapping)
            or set(action.payload) != expected_keys
        ):
            raise ConfirmationSessionError("invalid_action", "invalid action kind or payload")
        payload = action.payload
        integer_fields = {
            "select_corner": ("index",),
            "move": ("dx", "dy"),
            "set_corner": ("index", "x", "y"),
            "set_zoom": ("zoom",),
        }.get(action.kind, ())
        if any(type(payload[field]) is not int for field in integer_fields):
            raise ConfirmationSessionError("invalid_action", "invalid action payload values")
        if action.kind in {"select_corner", "set_corner"} and not 0 <= payload["index"] < 4:
            raise ConfirmationSessionError("invalid_action", "invalid action payload values")
        if action.kind == "set_zoom" and payload["zoom"] not in VALID_ZOOM_LEVELS:
            raise ConfirmationSessionError("invalid_action", "invalid action payload values")
        if action.kind == "skip":
            reason = payload["reason"]
            if type(reason) is not str or not reason.strip() or len(reason) > 256:
                raise ConfirmationSessionError("invalid_action", "invalid action payload values")

    def dispatch(self, action: ConfirmationAction) -> dict:
        self._validate_action(action)
        fingerprint = self._fingerprint(action)
        with self.lock:
            previous = self.results.get(action.action_id)
            if previous is not None:
                if previous[0] != fingerprint:
                    raise ConfirmationSessionError("action_id_conflict", "action ID was reused")
                return copy.deepcopy(previous[1])
            if self.status != "active":
                raise ConfirmationSessionError(
                    "session_stopped", "confirmation session is no longer active", "session"
                )
            if action.expected_revision != self.revision:
                raise ConfirmationSessionError("stale_revision", "stale confirmation revision")
            try:
                self._apply(action)
            except ConfirmationActionError as exc:
                raise ConfirmationSessionError("action_rejected", str(exc)) from exc
            self.revision += 1
            result = self.snapshot()
            self.results[action.action_id] = (fingerprint, copy.deepcopy(result))
            while len(self.results) > self._RESULT_CACHE_SIZE:
                self.results.popitem(last=False)
            return result


__all__ = [
    "AutoV4ConfirmationViewModel", "ConfirmationAction", "ConfirmationActionError",
    "ConfirmationEntryController", "ConfirmationSessionBackend", "ConfirmationSessionController",
    "ConfirmationSessionError", "LoadedConfirmationItem", "_auto_v4_candidate",
    "_auto_v4_confirmation_candidates", "_v7_view_model_for_entry",
]
