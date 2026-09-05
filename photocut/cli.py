#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
photocut 命令行入口
"""

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import stat
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any
import numpy as np
import cv2
from pathlib import Path


def is_gui_available() -> bool:
    """检测 OpenCV GUI 是否可用"""
    # 方法1：检查 highgui 窗口后端是否支持
    # 方法2：尝试创建并立即销毁一个测试窗口
    try:
        test_window = "__gui_test__"
        cv2.namedWindow(test_window, cv2.WINDOW_NORMAL)
        cv2.destroyWindow(test_window)
        cv2.waitKey(1)  # 需要处理事件才能真正销毁
        return True
    except Exception:
        return False


def check_gui_and_prompt() -> bool:
    """检测 GUI 可用性，不可用则提示用户"""
    if is_gui_available():
        return True
    print("错误：无法创建 GUI 窗口")
    print("可能原因：")
    print("  1. 运行在无图形界面的环境（如纯终端、SSH）")
    print("  2. DISPLAY 环境变量未设置（Linux）")
    print("  3. OpenCV 编译时缺少 GUI 支持")
    print("\n解决方案：")
    print("  - 在有图形界面的终端中运行：photocut input -o output --confirm")
    print("  - 或使用 VNC/X11 转发")
    return False


from photocut.core import (
    detect_and_save_corners,
    detect_and_save_corners_auto,
    detect_and_save_corners_v7,
    detect_and_save_corners_v84,
    detect_and_save_corners_v8_dormant,
    crop_image,
    load_corners_info,
    save_corners_info,
    reconcile_confirmation_events,
    load_image,
)
from photocut.data.crop_event_store import CropEventStore
from photocut.data.dataset_store import BatchLock, DatasetStore, atomic_write_json, sha256_file
from photocut.geometry.crop import InsetGeometryError, inset_quadrilateral
from photocut.confirmation.model import (
    ConfirmationState,
    DisplayTransform,
    MagnifierDragCapture,
    build_crop_event,
    display_to_original,
    legacy_preview_corners,
    magnifier_drag_delta,
    original_to_display,
    render_magnifier_source,
)
from photocut.confirmation.version import (
    OPENCV_GUI_VERSION,
    WEB_GUI_VERSION,
    gui_identity_lines,
    set_native_gui_title,
)
from photocut.confirmation.pointer import create_global_pointer_reader
from photocut.config import (
    ALGORITHM_VERSION,
    DEFAULT_DETECTOR,
    DEFAULT_SCENE_PROFILE,
    BATCH_REFERENCE_FILE,
    CORNERS_INFO_FILE,
    DATASET_SCHEMA_VERSION,
    OUTPUT_DIR,
    CORNER_SHRINK_MIN,
    CORNER_SHRINK_MAX,
    PS_LEVELS_HIGHLIGHT,
    V7_ALGORITHM_VERSION,
)
from photocut.data.local import default_dataset_root
from photocut.selector import SELECTOR_NAME, SELECTOR_RECORD_ID, SELECTOR_VERSION
from photocut.detection_parameters import DEFAULT_DETECTION_PARAMETERS, DetectionParameters
from photocut.confirmation.controller import (
    AutoV4ConfirmationViewModel,
    ConfirmationAction,
    ConfirmationSessionError,
    _auto_v4_candidate,
    _auto_v4_confirmation_candidates,
    _v7_view_model_for_entry,
)
from photocut.confirmation.backend import (
    commit_confirmation,
    confirmation_persistence_error_message,
    create_confirmation_session,
    reset_confirmation_boundary,
    save_confirmation_draft,
    save_manual_boundary,
    select_confirmation_entries,
    _annotation_store_from_reference,
    _dataset_root_from_reference,
    _entry_uses_normalized_snapshot,
    _gui_identity_entry,
    _load_archived_v7_image,
    _load_entry_image,
    _load_finalized_detection_evidence,
)


GUI_FRAME_DELAY_MS = 16
COARSE_MOVE_PX = 50
ARROW_MOVE_PX = 5
V8_DEFAULT_POLICY_RELATIVE_PATH = Path(
    "models/v8_2/v8.2-policy.json"
)
V8_DEFAULT_MODEL_RELATIVE_PATH = Path("models/v8_2")
V8_BOUNDARY_QUALITY_RELATIVE_PATH = Path(
    "algorithms/v8/assets/v8.2-boundary-quality-v6.json"
)
AUTO_V4_POLICY_RELATIVE_PATH = Path("selector/assets/selector-v4.json")


@dataclass(frozen=True)
class AutoV4Policy:
    """Canonical policy for the CLI-only auto-v4 orchestration layer."""

    schema_version: int
    cascade_version: str
    scanner_white_default_engine: str
    max_v7_calls: int
    max_v8_calls: int
    max_v52_calls: int
    v52_worker_timeout_s: float
    v52_witness_auto_accept: bool
    witness_reason_allowlist: tuple[str, ...]
    semantic_terminal_statuses: tuple[str, ...]
    semantic_terminal_reasons: tuple[str, ...]
    hard_risks: tuple[str, ...]
    max_mean_normalized_corner_distance: float
    minimum_polygon_iou: float
    policy_sha256: str


def _load_auto_v4_policy(path: str | Path) -> AutoV4Policy:
    """Load a closed-schema policy without following filesystem aliases."""
    policy_path = Path(path)
    try:
        info = policy_path.lstat()
    except OSError as exc:
        raise ValueError("auto-v4 policy is unavailable") from exc
    if policy_path.is_symlink() or not stat.S_ISREG(info.st_mode):
        raise ValueError("auto-v4 policy must be a regular file")
    try:
        value = json.loads(policy_path.read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("auto-v4 policy must be valid JSON") from exc
    expected = {
        "schema_version", "cascade_version", "scanner_white_default_engine",
        "max_v7_calls", "max_v8_calls", "max_v52_calls",
        "v52_worker_timeout_s", "v52_witness_auto_accept",
        "witness_reason_allowlist", "semantic_terminal_statuses",
        "semantic_terminal_reasons", "hard_risks", "agreement",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("auto-v4 policy fields do not match schema")
    agreement = value.get("agreement")
    if not isinstance(agreement, dict) or set(agreement) != {
        "max_mean_normalized_corner_distance", "minimum_polygon_iou",
    }:
        raise ValueError("auto-v4 agreement fields do not match schema")

    def integer(name: str, expected_value: int) -> int:
        item = value.get(name)
        if type(item) is not int or item != expected_value:
            raise ValueError(f"auto-v4 {name} must equal {expected_value}")
        return item

    def names(name: str) -> tuple[str, ...]:
        raw = value.get(name)
        if (
            not isinstance(raw, list)
            or not raw
            or any(not isinstance(item, str) or not item for item in raw)
            or raw != sorted(set(raw))
        ):
            raise ValueError(f"auto-v4 {name} must be a sorted unique string list")
        return tuple(raw)

    if value.get("schema_version") != 1:
        raise ValueError("unsupported auto-v4 policy schema")
    if value.get("cascade_version") != SELECTOR_RECORD_ID:
        raise ValueError("unsupported auto-v4 cascade version")
    if value.get("scanner_white_default_engine") != "v8":
        raise ValueError("auto-v4 scanner-white engine must be v8")
    if type(value.get("v52_witness_auto_accept")) is not bool:
        raise ValueError("auto-v4 witness switch must be boolean")
    timeout = value.get("v52_worker_timeout_s")
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not np.isfinite(timeout)
        or not 0.0 < float(timeout) <= 3.0
    ):
        raise ValueError("auto-v4 v5.2 timeout must be in (0, 3]")
    distance = agreement.get("max_mean_normalized_corner_distance")
    minimum_iou = agreement.get("minimum_polygon_iou")
    if (
        isinstance(distance, bool)
        or not isinstance(distance, (int, float))
        or not np.isfinite(distance)
        or not 0.0 < float(distance) <= 0.005
    ):
        raise ValueError("auto-v4 agreement distance is too loose")
    if (
        isinstance(minimum_iou, bool)
        or not isinstance(minimum_iou, (int, float))
        or not np.isfinite(minimum_iou)
        or not 0.98 <= float(minimum_iou) <= 1.0
    ):
        raise ValueError("auto-v4 agreement IoU is too loose")
    allowlist = names("witness_reason_allowlist")
    terminal_statuses = names("semantic_terminal_statuses")
    terminal_reasons = names("semantic_terminal_reasons")
    hard_risks = names("hard_risks")
    if set(allowlist) & (set(terminal_reasons) | set(hard_risks)):
        raise ValueError("auto-v4 witness allowlist overlaps fail-closed reasons")
    canonical = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return AutoV4Policy(
        schema_version=1,
        cascade_version=SELECTOR_RECORD_ID,
        scanner_white_default_engine="v8",
        max_v7_calls=integer("max_v7_calls", 2),
        max_v8_calls=integer("max_v8_calls", 1),
        max_v52_calls=integer("max_v52_calls", 1),
        v52_worker_timeout_s=float(timeout),
        v52_witness_auto_accept=value["v52_witness_auto_accept"],
        witness_reason_allowlist=allowlist,
        semantic_terminal_statuses=terminal_statuses,
        semantic_terminal_reasons=terminal_reasons,
        hard_risks=hard_risks,
        max_mean_normalized_corner_distance=float(distance),
        minimum_polygon_iou=float(minimum_iou),
        policy_sha256="sha256:" + hashlib.sha256(canonical).hexdigest(),
    )


@dataclass(frozen=True)
class V8CliRuntime:
    """One verified V8 policy/model session reused for a complete batch."""

    policy: Any
    mask_provider: Any
    runtime_config: dict[str, Any]
    runtime_config_sha256: str
    model_manifest_sha256: str
    boundary_quality_artifact: Any = None


@dataclass(frozen=True)
class AutoV4Runtime:
    """Batch-fixed auto-v4 route plus its optional verified V8 session."""

    policy: AutoV4Policy
    request_mode: str
    requested_engine: str
    effective_engine: str
    degraded: bool
    degradation_reason: str | None
    v8_runtime: V8CliRuntime | Any | None


_AUTO_V4_V8_CALL_LOCK = threading.Lock()


@dataclass(frozen=True)
class ConfirmationPreview:
    scaled_image: np.ndarray
    transform: DisplayTransform
    scaled_width: int
    scaled_height: int


@dataclass
class ConfirmationDraftTracker:
    """Track unsaved corner changes without performing frame-loop I/O."""
    initial_corners: list[list[int]]

    def __post_init__(self) -> None:
        self._checkpoint = self._signature(self.initial_corners)
        self._current = self._checkpoint

    @staticmethod
    def _signature(corners) -> tuple[tuple[int, int], ...]:
        return tuple((int(point[0]), int(point[1])) for point in corners)

    def observe(self, corners) -> None:
        self._current = self._signature(corners)

    @property
    def dirty(self) -> bool:
        return self._current != self._checkpoint

    def checkpoint(self) -> None:
        self._checkpoint = self._current


def begin_magnifier_drag(mouse_state: dict, x: int, y: int) -> bool:
    """Start native capture when available; otherwise use local window drag."""
    reader = mouse_state.get("global_pointer_reader")
    capture = mouse_state.setdefault("global_magnifier_capture", MagnifierDragCapture())
    if reader is not None:
        try:
            sample = reader.sample()
            if not sample.left_down:
                # A working reader that reports a released button is a real
                # release, not an unavailable platform. Never start local
                # capture here, because the pointer may already be outside.
                cancel_magnifier_drag(mouse_state)
                return False
            capture.begin(sample)
            mouse_state["global_magnifier_capture_active"] = True
            mouse_state["magnifier_dragging"] = True
            return True
        except Exception:
            # Native support is optional before a drag starts; local dragging
            # remains available if the platform reader cannot be sampled.
            mouse_state["global_magnifier_capture_active"] = False
    capture.cancel()
    mouse_state["global_magnifier_capture_active"] = False
    mouse_state["magnifier_dragging"] = True
    mouse_state["magnifier_drag_anchor"] = (x, y)
    return False


def cancel_magnifier_drag(mouse_state: dict) -> None:
    capture = mouse_state.get("global_magnifier_capture")
    if capture is not None:
        capture.cancel()
    mouse_state["global_magnifier_capture_active"] = False
    mouse_state["magnifier_dragging"] = False


def poll_magnifier_drag(mouse_state: dict) -> bool:
    """Consume one global pointer sample, including release outside the window."""
    if not mouse_state.get("global_magnifier_capture_active"):
        return False
    reader = mouse_state.get("global_pointer_reader")
    capture = mouse_state.get("global_magnifier_capture")
    state = mouse_state.get("state")
    if reader is None or capture is None or state is None or state.selected < 0:
        cancel_magnifier_drag(mouse_state)
        return False
    try:
        delta = capture.update(reader.sample(), zoom=state.zoom)
    except Exception:
        # Once native capture has started, fail closed. The pointer may be
        # outside the window, so leaving the local drag flag set could stick.
        cancel_magnifier_drag(mouse_state)
        return False
    if not capture.active:
        mouse_state["global_magnifier_capture_active"] = False
        mouse_state["magnifier_dragging"] = False
        return True
    if delta != (0, 0):
        dispatch_move = mouse_state.get("dispatch_move")
        if dispatch_move is None:
            state.move_selected(*delta)
        else:
            dispatch_move(*delta)
    return True


def coarse_move_delta(key_ascii: int):
    return {
        ord("w"): (0, -COARSE_MOVE_PX),
        ord("a"): (-COARSE_MOVE_PX, 0),
        ord("s"): (0, COARSE_MOVE_PX),
        ord("d"): (COARSE_MOVE_PX, 0),
    }.get(key_ascii)


def v7_candidate_key_action(key_ascii: int):
    return {
        ord("C"): "alternate",
        ord("B"): "v5.2",
        ord("b"): "v5.2",
        ord("X"): "skip",
    }.get(key_ascii)


def apply_corner_delta(state, delta, mouse_state=None) -> bool:
    if state.selected < 0:
        return False
    if mouse_state is not None:
        cancel_magnifier_drag(mouse_state)
        mouse_state["dragging"] = False
    state.move_selected(*delta)
    return True


def apply_coarse_corner_key(state, key_ascii: int, mouse_state=None) -> bool:
    delta = coarse_move_delta(key_ascii)
    if delta is None:
        return False
    return apply_corner_delta(state, delta, mouse_state)


def build_confirmation_preview(
    image: np.ndarray, *, viewport_width: int, viewport_height: int
) -> ConfirmationPreview:
    """Build the expensive full-image resize exactly once per GUI entry."""
    value = np.asarray(image)
    if value.ndim != 3 or value.shape[0] <= 0 or value.shape[1] <= 0:
        raise ValueError("confirmation image must be a non-empty HxWxC array")
    if (type(viewport_width) is not int or type(viewport_height) is not int or
            viewport_width <= 0 or viewport_height <= 0):
        raise ValueError("confirmation viewport must be positive")
    original_height, original_width = value.shape[:2]
    scale = min(viewport_width / original_width, viewport_height / original_height)
    scaled_width = max(1, round(original_width * scale))
    scaled_height = max(1, round(original_height * scale))
    offset_x = (viewport_width - scaled_width) // 2
    transform = DisplayTransform(
        scale=scale,
        offset_x=offset_x,
        offset_y=0,
        original_width=original_width,
        original_height=original_height,
    )
    scaled = cv2.resize(value, (scaled_width, scaled_height))
    return ConfirmationPreview(scaled, transform, scaled_width, scaled_height)


def record_preview_render_error(entry: dict, exc: BaseException) -> None:
    """Record a non-confirming GUI failure without discarding the entry."""
    entry["confirmed"] = False
    entry["preview_render_error"] = {
        "code": "preview_render_error",
        "message": f"{type(exc).__name__}: {exc}",
    }


def observe_confirmation_frame(
    tracker: ConfirmationDraftTracker, state: ConfirmationState, v7_model
) -> None:
    """Update in-memory GUI state; this hot path deliberately performs no I/O."""
    tracker.observe(state.work_corners)
    if v7_model is not None:
        v7_model._dragged = bool(state.adjusted_corner_indices)


def add_detector_arguments(parser, *, default=argparse.SUPPRESS) -> None:
    """Add detector selection to both legacy and subcommand CLI forms."""
    parser.add_argument(
        "--detector", choices=("v8.4", "auto", "v5.2", "v7", "v8"), default=default,
        help="角点检测器（默认 V8.4，所有结果需人工确认；auto 为旧版回滚）",
    )
    parser.add_argument(
        "--v7-mode", choices=("safe", "aggressive"), default=argparse.SUPPRESS,
        help="v7 风险阈值模式（仅 --detector v7）",
    )
    parser.add_argument(
        "--scene-profile", choices=("scanner_white", "generic_single"),
        default=argparse.SUPPRESS,
        help="单照片场景：白色扫描底或通用背景（V8 仅支持白色扫描底）",
    )
    parser.add_argument(
        "--v8-policy",
        default=argparse.SUPPRESS,
        help="V8 已验证策略文件（通常无需指定）",
    )
    parser.add_argument(
        "--v8-model-dir",
        default=argparse.SUPPRESS,
        help="V8 已验证模型目录（通常无需指定）",
    )
    parser.add_argument(
        "--auto-engine",
        choices=("v8", "v7"),
        default=argparse.SUPPRESS,
        help=f"{SELECTOR_NAME} 主引擎；scanner_white 默认使用 V8.2，v7 为回滚",
    )


INSET_HELP = "裁剪阶段沿四条真实边界向内偏移的原图像素数（默认: 50）"


def add_inset_argument(parser, default=argparse.SUPPRESS) -> None:
    parser.add_argument("--inset", type=float, default=default, help=INSET_HELP)


def add_confirmation_arguments(parser, *, default=argparse.SUPPRESS) -> None:
    parser.add_argument(
        "--confirm-ui",
        choices=("opencv", "web"),
        default=default,
        help="确认界面：web（默认，本地 GUI 2.0）或 opencv（回退 GUI 1.0）",
    )


def update_confirmation_gui_identity(
    cv2_module,
    window_key,
    entry,
    previous_annotation=None,
):
    """Update the native title and return identity lines for sidebar fallback."""
    title = set_native_gui_title(
        cv2_module, window_key, entry.get("algorithm_version")
    )
    return title, gui_identity_lines(
        entry, previous_annotation, gui_version=OPENCV_GUI_VERSION
    )


def _regular_directory(path: Path, label: str) -> Path:
    raw_path = Path(path)
    if ".." in raw_path.parts:
        raise ValueError(f"{label} must not contain parent traversal")
    path = Path(os.path.abspath(raw_path))
    try:
        info = path.lstat()
    except OSError as exc:
        raise ValueError(f"{label} must be an existing directory") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ValueError(f"{label} must be a regular directory")
    return path


def _safe_cropped_directory(output_dir: Path) -> Path:
    output_dir = _regular_directory(output_dir, "crop output directory")
    cropped_dir = output_dir / OUTPUT_DIR
    if os.path.lexists(cropped_dir):
        _regular_directory(cropped_dir, "crop result directory")
    else:
        cropped_dir.mkdir()
        _regular_directory(cropped_dir, "crop result directory")
    return cropped_dir


def resolve_crop_event_store(output_dir: Path, project_root: Path) -> CropEventStore | None:
    """Resolve the batch-owned crop log, or retain legacy output-only mode."""
    reference_path = Path(output_dir) / BATCH_REFERENCE_FILE
    if not os.path.lexists(reference_path):
        return None
    if reference_path.is_symlink() or not reference_path.is_file():
        raise ValueError("batch reference must be a regular file")
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    dataset_root = _dataset_root_from_reference(reference["dataset_root"], project_root)
    store = DatasetStore(dataset_root)
    batch_id = reference["batch_id"]
    paths = store.batch_paths_from_candidate(store.batches_dir / batch_id)
    if paths is None:
        raise ValueError("batch reference contains an invalid batch_id")
    return CropEventStore(paths.crops, batch_dir=paths.batch_dir)


_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _input_root(input_dir: str) -> Path:
    raw = Path(input_dir)
    if ".." in raw.parts:
        raise ValueError("input directory must not contain parent traversal")
    try:
        info = raw.lstat()
    except OSError as exc:
        raise ValueError("input directory must be an existing regular directory") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ValueError("input directory must be an existing regular directory")
    return raw.resolve(strict=True)


def derive_default_output(input_dir: str) -> str:
    root = _input_root(input_dir)
    if not root.name:
        raise ValueError("cannot safely derive output directory from input")
    return str(root.parent / f"{root.name}裁剪")


def _reject_output_under_input(input_root: Path, output_dir: str | None) -> Path | None:
    if output_dir is None:
        return None
    output_root = Path(output_dir).resolve(strict=False)
    try:
        output_root.relative_to(input_root)
    except ValueError:
        return output_root
    raise ValueError("output directory must not be inside input directory")


def find_images(input_dir: str, output_dir: str | None = None) -> list[tuple[str, str]]:
    """Return supported regular files as (absolute path, input-relative POSIX path)."""
    root = _input_root(input_dir)
    output_root = _reject_output_under_input(root, output_dir)
    images = []
    for current, dirs, files in os.walk(root, followlinks=False):
        current_path = Path(current)
        safe_dirs = []
        for name in dirs:
            candidate = current_path / name
            if name.startswith(".") or candidate.is_symlink():
                continue
            if output_root is not None and candidate.resolve(strict=False) == output_root:
                continue
            safe_dirs.append(name)
        dirs[:] = safe_dirs
        for name in files:
            candidate = current_path / name
            if candidate.is_symlink() or candidate.suffix.lower() not in _IMAGE_SUFFIXES:
                continue
            try:
                info = candidate.lstat()
            except OSError:
                continue
            if not stat.S_ISREG(info.st_mode):
                continue
            images.append((str(candidate.resolve()), candidate.relative_to(root).as_posix()))
    return sorted(images, key=lambda item: item[1])


def normalize_entry_relative_paths(entries, discovered):
    """Resolve manifest filenames to scanned relative paths without guessing.

    Exact relative paths win.  A legacy basename is accepted only when it names
    exactly one scanned source.  Invalid, duplicate, and colliding entries are
    returned as rejected so callers can report them without touching a source.
    """
    relative_paths = {relative for _, relative in discovered}
    basename_matches = {}
    for relative in relative_paths:
        basename_matches.setdefault(Path(relative).name, []).append(relative)

    resolved = []
    rejected = {}
    for entry in entries:
        filename = entry.get("filename") if isinstance(entry, dict) else None
        if not isinstance(filename, str) or not filename:
            rejected[id(entry)] = "拒绝无效 filename"
            continue
        candidate = Path(filename)
        if candidate.is_absolute() or ".." in candidate.parts:
            rejected[id(entry)] = "拒绝不安全路径"
            continue
        if filename in relative_paths:
            relative_path = filename
        elif len(candidate.parts) == 1:
            matches = basename_matches.get(filename, [])
            if len(matches) != 1:
                rejected[id(entry)] = "拒绝不唯一的旧版 basename"
                continue
            relative_path = matches[0]
        else:
            rejected[id(entry)] = "拒绝不存在的相对路径"
            continue
        resolved.append((entry, relative_path))

    def reject_groups(key, message):
        grouped = {}
        for entry, relative_path in resolved:
            if id(entry) not in rejected:
                grouped.setdefault(key(relative_path), []).append(entry)
        for group in grouped.values():
            if len(group) > 1:
                for entry in group:
                    rejected[id(entry)] = message

    reject_groups(lambda relative: relative, "拒绝重复的输入路径记录")
    reject_groups(
        lambda relative: (Path(relative).parent / f"{Path(relative).stem}_裁切.jpg").as_posix(),
        "拒绝输出路径与其他输入文件冲突",
    )

    accepted = []
    for entry, relative_path in resolved:
        if id(entry) in rejected:
            continue
        entry["filename"] = relative_path
        accepted.append((entry, relative_path))
    return accepted, [(entry, rejected[id(entry)]) for entry in entries if id(entry) in rejected]


def _preserve_confirmed_detection(previous, info, *, detector: str = DEFAULT_DETECTOR) -> None:
    """Keep human-approved boundaries unchanged during a redetection."""
    if not previous or not previous.get("confirmed"):
        return
    used_detector = info.get("detector_used", detector)
    if used_detector in {"v7", "v8", "v8.4", "manual_review"}:
        # A normalized modern result is only allowed to inherit confirmation when it is for
        # the same immutable source identity and the previous record carries a
        # durable annotation head.  Otherwise fail closed and leave the new
        # detection unconfirmed rather than silently authorizing a crop.
        if (
            not previous.get("annotation_id")
            or not previous.get("image_id")
            or previous.get("image_id") != info.get("image_id")
        ):
            info["confirmed"] = False
            info["manually_adjusted"] = False
            info["manual_corners"] = None
            info["manual_preview_corners"] = None
            return
    if used_detector == "v8.4":
        origin = previous.get("confirmation_origin_batch_id") or previous.get("batch_id")
        if origin:
            info["confirmation_origin_batch_id"] = origin
        # Preserve the approved boundary, including an accepted unadjusted result.
        for field in ("boundary_corners", "corners", "preview_corners"):
            if field in previous:
                info[field] = copy.deepcopy(previous[field])
        info["success"] = bool(previous.get("success"))
    info["confirmed"] = True
    if previous.get("annotation_id") is not None:
        info["annotation_id"] = previous["annotation_id"]
    if "confirm_timestamp" in previous:
        info["confirm_timestamp"] = previous["confirm_timestamp"]
    for field in ("manually_adjusted", "manual_corners", "manual_preview_corners"):
        if field in previous:
            info[field] = previous[field]
    if info.get("manually_adjusted"):
        info["boundary_corners"] = previous.get(
            "boundary_corners", previous.get("corners", info["boundary_corners"])
        )
        info["corners"] = previous.get("corners", info["corners"])
        info["preview_corners"] = previous.get("preview_corners", info["preview_corners"])


def _source_records(sources):
    records = []
    for position, item in enumerate(sources):
        if isinstance(item, tuple):
            source, relative_path = item
        else:
            source = item
            relative_path = Path(source).name
        source_path = str(source.resolve())
        source_identity = json.dumps(
            {"position": position, "source_path": source_path},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        records.append(
            {
                "source_id": f"source:{hashlib.sha256(source_identity).hexdigest()}",
                "image_id": f"sha256:{sha256_file(source)}",
                "source_filename": relative_path,
                "source_path": source_path,
            }
        )
    return records


def effective_detection_parameters(threshold: int | None = None) -> DetectionParameters:
    params = DEFAULT_DETECTION_PARAMETERS
    if threshold is not None:
        params = params.replace(ps_highlight=threshold)
    return params


def _v52_code_identity() -> str:
    """Hash the exact legacy detector source files used by the runtime."""
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parent
    for relative in (
        "algorithms/v5_2/detector.py",
        "detection_parameters.py",
        "config.py",
    ):
        payload = (root / relative).read_bytes()
        name = relative.encode("utf-8")
        digest.update(name); digest.update(b"\0")
        digest.update(str(len(payload)).encode("ascii")); digest.update(b"\0")
        digest.update(payload)
    return digest.hexdigest()


def _v8_regular_bytes(path: Path, label: str) -> bytes:
    """Read one V8 identity file without following a symlink."""
    try:
        info = path.lstat()
    except OSError as exc:
        raise ValueError(f"{label} 不存在或无法读取") from exc
    if path.is_symlink() or not stat.S_ISREG(info.st_mode):
        raise ValueError(f"{label} 必须是普通文件")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ValueError(f"{label} 无法读取") from exc


def _v8_sha256(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _load_v8_cli_runtime(policy_path: str | Path, model_dir: str | Path) -> V8CliRuntime:
    """Load and verify the local V8 policy, model, and deterministic runtime."""
    from photocut.algorithms.v7.model_manifest import load_model_manifest, verify_model_artifact
    from photocut.algorithms.v8.code_seal import compute_validated_core_sha256
    from photocut.algorithms.v8.boundary_quality import load_boundary_quality_artifact
    from photocut.algorithms.v8.policy import load_v8_policy
    from photocut.algorithms.v8.provider import LazyOnnxRuntimeBackend, PhotoMaskProvider

    project_root = Path(__file__).resolve().parent
    policy_path = Path(policy_path)
    model_dir = Path(model_dir)
    _v8_regular_bytes(policy_path, "V8 策略文件")
    try:
        model_info = model_dir.lstat()
    except OSError as exc:
        raise ValueError("V8 模型目录不存在或无法读取") from exc
    if model_dir.is_symlink() or not stat.S_ISDIR(model_info.st_mode):
        raise ValueError("V8 模型目录必须是普通目录")

    policy = load_v8_policy(policy_path)
    if compute_validated_core_sha256(project_root) != policy.validated_core_sha256:
        raise ValueError("V8 程序文件与已验证策略不一致")

    manifest_path = model_dir / "model-manifest.json"
    runtime_path = model_dir / "runtime-config.json"
    manifest_raw = _v8_regular_bytes(manifest_path, "V8 模型清单")
    runtime_raw = _v8_regular_bytes(runtime_path, "V8 运行配置")
    manifest_sha256 = _v8_sha256(manifest_raw)
    runtime_sha256 = _v8_sha256(runtime_raw)
    if manifest_sha256 != policy.model_manifest_sha256:
        raise ValueError("V8 模型清单与已验证策略不一致")
    if runtime_sha256 != policy.runtime_config_sha256:
        raise ValueError("V8 运行配置与已验证策略不一致")
    try:
        runtime_config = json.loads(runtime_raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("V8 运行配置不是有效 JSON") from exc
    if not isinstance(runtime_config, dict):
        raise ValueError("V8 运行配置必须是对象")

    manifest = load_model_manifest(manifest_path)
    model_path = verify_model_artifact(manifest, model_dir / manifest.model_filename)
    if manifest.model_id != policy.model_id or manifest.model_sha256 != policy.model_sha256:
        raise ValueError("V8 模型与已验证策略不一致")
    provider = PhotoMaskProvider(
        LazyOnnxRuntimeBackend(model_path, manifest),
        manifest,
        threshold=policy.parameters.mask_threshold,
    )
    boundary_quality_artifact = None
    if policy.schema_version == 3:
        boundary_quality_artifact = load_boundary_quality_artifact(
            project_root / V8_BOUNDARY_QUALITY_RELATIVE_PATH,
            expected_sha256=policy.boundary_quality_artifact_sha256,
        )
    return V8CliRuntime(
        policy=policy,
        mask_provider=provider,
        runtime_config=runtime_config,
        runtime_config_sha256=runtime_sha256,
        model_manifest_sha256=manifest_sha256,
        boundary_quality_artifact=boundary_quality_artifact,
    )


def _v8_runtime_from_args(args) -> V8CliRuntime:
    project_root = Path(__file__).resolve().parent
    policy_path = getattr(args, "v8_policy", None) or (
        project_root / V8_DEFAULT_POLICY_RELATIVE_PATH
    )
    model_dir = getattr(args, "v8_model_dir", None) or (
        project_root / V8_DEFAULT_MODEL_RELATIVE_PATH
    )
    try:
        return _load_v8_cli_runtime(policy_path, model_dir)
    except Exception as exc:
        raise RuntimeError(f"V8 本机运行文件不可用：{exc}") from exc


def _resolve_auto_v4_runtime(
    args,
    *,
    existing_route: dict[str, Any] | None = None,
) -> AutoV4Runtime:
    """Freeze the auto route once for a new invocation or resumable batch."""
    project_root = Path(__file__).resolve().parent
    policy = _load_auto_v4_policy(project_root / AUTO_V4_POLICY_RELATIVE_PATH)
    scene_profile = getattr(args, "scene_profile", None) or DEFAULT_SCENE_PROFILE
    explicit_engine = getattr(args, "auto_engine", None)
    if explicit_engine not in {None, "v7", "v8"}:
        raise ValueError("auto engine must be v7 or v8")

    if existing_route is not None:
        required = {
            "auto_engine_request_mode", "auto_engine_requested",
            "auto_engine_effective", "auto_engine_degraded",
            "auto_engine_degradation_reason",
        }
        if not isinstance(existing_route, dict) or not required <= set(existing_route):
            raise RuntimeError("existing auto-v4 route is incomplete")
        request_mode = existing_route["auto_engine_request_mode"]
        requested_engine = existing_route["auto_engine_requested"]
        effective_engine = existing_route["auto_engine_effective"]
        degraded = existing_route["auto_engine_degraded"]
        degradation_reason = existing_route["auto_engine_degradation_reason"]
        if (
            request_mode not in {"implicit_default", "explicit"}
            or requested_engine not in {"v7", "v8"}
            or effective_engine not in {"v7", "v8"}
            or type(degraded) is not bool
            or degradation_reason is not None
            and not isinstance(degradation_reason, str)
        ):
            raise RuntimeError("existing auto-v4 route is invalid")
        if explicit_engine is not None and (
            request_mode != "explicit" or explicit_engine != requested_engine
        ):
            raise RuntimeError("auto engine request mode does not match resumable batch")
        if scene_profile == "generic_single" and effective_engine != "v7":
            raise RuntimeError("generic_single resumable batch cannot use V8")
        v8_runtime = None
        if effective_engine == "v8":
            try:
                v8_runtime = _v8_runtime_from_args(args)
            except RuntimeError as exc:
                raise RuntimeError(
                    "cannot resume effective V8 auto-v4 batch without its verified runtime"
                ) from exc
        return AutoV4Runtime(
            policy, request_mode, requested_engine, effective_engine,
            degraded, degradation_reason, v8_runtime,
        )

    request_mode = "explicit" if explicit_engine is not None else "implicit_default"
    if scene_profile == "generic_single":
        if explicit_engine == "v8":
            raise RuntimeError("generic_single does not support auto engine V8")
        return AutoV4Runtime(
            policy, request_mode, "v7", "v7", False, None, None
        )

    requested_engine = explicit_engine or policy.scanner_white_default_engine
    if requested_engine == "v7":
        return AutoV4Runtime(
            policy, request_mode, "v7", "v7", False, None, None
        )
    try:
        v8_runtime = _v8_runtime_from_args(args)
    except RuntimeError as exc:
        if request_mode == "explicit":
            raise RuntimeError("explicit auto engine V8 preflight failed") from exc
        return AutoV4Runtime(
            policy, request_mode, "v8", "v7", True,
            "v8_runtime_unavailable", None,
        )
    return AutoV4Runtime(
        policy, request_mode, "v8", "v8", False, None, v8_runtime
    )


def _auto_v4_valid_quad(value: Any, image_size: tuple[int, int]):
    from photocut.algorithms.v7.geometry import GeometryError, validate_quad

    try:
        return validate_quad(value, image_size=image_size)
    except (GeometryError, TypeError, ValueError):
        return None


def _call_v8_adapter_observed(*args, **kwargs):
    """Call the sealed adapter once while observing its internal V7 calls."""
    from photocut import core as photocut_module

    adapter = detect_and_save_corners_v8_dormant
    production_adapter = photocut_module.detect_and_save_corners_v8_dormant
    if adapter is not production_adapter:
        result = adapter(*args, **kwargs)
        envelope = getattr(result, "audit_envelope", {})
        observed = envelope.get("auto_v4_observed_v7_calls") if isinstance(
            envelope, dict
        ) else None
        if type(observed) is not int or observed not in {1, 2}:
            raise RuntimeError("injected V8 adapter must report observed V7 calls")
        return result, observed

    with _AUTO_V4_V8_CALL_LOCK:
        original_v7 = photocut_module.detect_and_save_corners_v7
        observed = 0

        def counted_v7(*inner_args, **inner_kwargs):
            nonlocal observed
            observed += 1
            return original_v7(*inner_args, **inner_kwargs)

        photocut_module.detect_and_save_corners_v7 = counted_v7
        try:
            result = adapter(*args, **kwargs)
        finally:
            photocut_module.detect_and_save_corners_v7 = original_v7
    if observed not in {1, 2}:
        raise RuntimeError(f"sealed V8 adapter made unexpected V7 call count: {observed}")
    return result, observed


def _run_auto_v4_legacy(
    loaded_input,
    *,
    timeout_s: float,
    shrink_min: int,
    shrink_max: int,
    params: DetectionParameters,
    cancellation_token: Any = None,
    relative_path: str | None = None,
) -> dict[str, Any]:
    """Run one legacy worker on the same immutable normalized snapshot."""
    from photocut.core import _map_analysis_entry_to_full
    from photocut.algorithms.v7.legacy_worker import SourceSnapshot, run_legacy_worker

    snapshot = SourceSnapshot(
        loaded_input.normalized_bgr,
        loaded_input.source_sha256,
        loaded_input.original_size,
        loaded_input.normalized_size,
        loaded_input.orientation_transform,
    )
    worker = run_legacy_worker(
        snapshot,
        timeout_s=timeout_s,
        cancellation_token=cancellation_token,
        shrink_min=shrink_min,
        shrink_max=shrink_max,
        params=params,
    )
    output = dict(worker) if isinstance(worker, dict) else {
        "status": "error", "error": "invalid_legacy_worker_result",
    }
    result = output.get("result")
    if output.get("status") == "ok" and isinstance(result, dict):
        projected = dict(result)
        projected["filename"] = relative_path or projected.get("filename", "")
        output["result"] = _map_analysis_entry_to_full(projected, loaded_input)
    output.update({
        "source_sha256": loaded_input.source_sha256,
        "source_original_size": list(loaded_input.original_size),
        "normalized_size": list(loaded_input.normalized_size),
        "normalized_orientation": loaded_input.orientation_transform,
    })
    return output


def _evaluate_auto_v4_witness(
    v8_info: dict[str, Any],
    legacy_worker: dict[str, Any],
    *,
    loaded_input,
    policy: AutoV4Policy,
    reason: str,
) -> dict[str, Any]:
    """Return a truth-free decision that can only accept the current V8 draft."""
    from photocut.algorithms.v7.cascade import _legacy_strong
    from photocut.algorithms.v7.geometry import polygon_iou

    result = legacy_worker.get("result")
    full_size = tuple(int(value) for value in loaded_input.full_normalized_size)
    rejection = None
    v8_quad = _auto_v4_valid_quad(
        v8_info.get("algorithm_boundary_corners") or v8_info.get("corners"),
        full_size,
    )
    legacy_quad = _auto_v4_valid_quad(
        result.get("corners") if isinstance(result, dict) else None,
        full_size,
    )
    if reason not in policy.witness_reason_allowlist:
        rejection = "reason_not_allowlisted"
    elif legacy_worker.get("status") != "ok" or not isinstance(result, dict):
        rejection = "legacy_worker_unavailable"
    elif (
        legacy_worker.get("source_sha256") != loaded_input.source_sha256
        or legacy_worker.get("normalized_orientation")
        != loaded_input.orientation_transform
        or tuple(legacy_worker.get("source_original_size") or ())
        != tuple(loaded_input.original_size)
        or tuple(legacy_worker.get("normalized_size") or ())
        != tuple(loaded_input.normalized_size)
    ):
        rejection = "source_snapshot_mismatch"
    elif v8_quad is None:
        rejection = "invalid_v8_quad"
    elif legacy_quad is None or not _legacy_strong(result, full_size):
        rejection = "legacy_not_strong"
    risks = {
        str(item) for item in (v8_info.get("risks") or ())
        if item not in (None, "none")
    }
    if rejection is None and (reason in policy.hard_risks or risks & set(policy.hard_risks)):
        rejection = "hard_risk"

    mean_distance = None
    iou = None
    if rejection is None:
        width, height = full_size
        left = tuple(
            (float(x) / (width - 1.0), float(y) / (height - 1.0))
            for x, y in v8_quad
        )
        right = tuple(
            (float(x) / (width - 1.0), float(y) / (height - 1.0))
            for x, y in legacy_quad
        )
        mean_distance = sum(
            math.hypot(ax - bx, ay - by)
            for (ax, ay), (bx, by) in zip(left, right)
        ) / 4.0
        iou = float(polygon_iou(left, right))
        if mean_distance > policy.max_mean_normalized_corner_distance:
            rejection = "geometry_distance_conflict"
        elif iou < policy.minimum_polygon_iou:
            rejection = "geometry_iou_conflict"
    if rejection is not None:
        action = "reject"
    elif policy.v52_witness_auto_accept:
        action = "accept_current_v8"
    else:
        action = "disabled_safe_proposal"
    return {
        "action": action,
        "reason": rejection,
        "mean_normalized_corner_distance": mean_distance,
        "polygon_iou": iou,
        "policy_sha256": policy.policy_sha256,
    }


def _detect_auto_v4_v8(
    *,
    img_path: Path,
    output_dir: str,
    auto_runtime: AutoV4Runtime,
    loaded_input,
    relative_path: str | None,
    request_id: str | None,
    image_id: str | None,
    shrink_min: int,
    shrink_max: int,
    params: DetectionParameters,
    cancellation_token: Any = None,
) -> dict[str, Any]:
    """Run the effective-V8 auto-v4 state table without editing sealed core."""
    from photocut.algorithms.v8.cascade import V8CascadeStatus

    if auto_runtime.effective_engine != "v8" or auto_runtime.v8_runtime is None:
        raise RuntimeError("auto-v4 V8 dispatch requires an effective V8 runtime")
    if loaded_input is None:
        raise RuntimeError("auto-v4 V8 dispatch requires one predecoded input")
    started = time.perf_counter()
    v8_runtime = auto_runtime.v8_runtime
    adapter_started = time.perf_counter()
    result, v7_calls = _call_v8_adapter_observed(
        str(img_path),
        output_dir,
        policy=v8_runtime.policy,
        mask_provider=v8_runtime.mask_provider,
        runtime_config=v8_runtime.runtime_config,
        runtime_config_sha256=v8_runtime.runtime_config_sha256,
        model_manifest_sha256=v8_runtime.model_manifest_sha256,
        boundary_quality_artifact=getattr(
            v8_runtime, "boundary_quality_artifact", None
        ),
        loaded_input=loaded_input,
        scene_profile="scanner_white",
        relative_path=relative_path,
        request_id=request_id,
        image_id=image_id,
        cancellation_token=cancellation_token,
    )
    adapter_ms = (time.perf_counter() - adapter_started) * 1000.0
    core = copy.deepcopy(dict(result.core_payload))
    envelope = copy.deepcopy(dict(result.audit_envelope))
    status = result.status
    status_value = status.value if hasattr(status, "value") else str(status)
    info = copy.deepcopy(core)
    info["cascade_v8_result"] = {
        "status": status_value,
        "core_payload": core,
        "audit_envelope": envelope,
    }
    reason = str(
        envelope.get("v8_selection_reason")
        or envelope.get("v8_fallback_reason")
        or "unknown_v8_reason"
    )
    provider_status = envelope.get("v8_provider_status")
    info.update({
        "detector_requested": "auto",
        "auto_cascade_version": auto_runtime.policy.cascade_version,
        "auto_cascade_policy_sha256": auto_runtime.policy.policy_sha256,
        "auto_engine_request_mode": auto_runtime.request_mode,
        "auto_engine_requested": auto_runtime.requested_engine,
        "auto_engine_effective": auto_runtime.effective_engine,
        "auto_engine_degraded": auto_runtime.degraded,
        "auto_engine_degradation_reason": auto_runtime.degradation_reason,
        "v8_policy_sha256": v8_runtime.policy.policy_sha256,
        "v8_provider_status": provider_status,
        "v8_fallback_reason": envelope.get("v8_fallback_reason"),
        "v8_decision_status": status_value,
        "v8_selection_reason": envelope.get("v8_selection_reason"),
        "cascade_calls": {"v7": v7_calls, "v8": 1, "v5.2": 0},
        "v7_retry_count": v7_calls - 1,
        "auto_v4_status": "manual_review",
        "auto_v4_witness": {
            "action": "not_attempted", "reason": "state_not_allowlisted",
            "policy_sha256": auto_runtime.policy.policy_sha256,
        },
    })
    full_size = tuple(int(value) for value in loaded_input.full_normalized_size)
    core_status = str(core.get("detection_status", "")).lower()
    risks = {str(item) for item in (core.get("risks") or ())}
    semantic_terminal = (
        core_status in auto_runtime.policy.semantic_terminal_statuses
        or reason in auto_runtime.policy.semantic_terminal_reasons
        or bool(risks & set(auto_runtime.policy.semantic_terminal_reasons))
    )
    legacy_worker = None
    if status is V8CascadeStatus.AUTOMATIC:
        info.update({
            "detector": "v8", "detector_used": "v8",
            "auto_v4_status": "automatic",
            "confirmation_primary_candidate_id": "v8:draft",
            "confirmation_selected_candidate_id": "v8:draft",
            "confirmation_selected_algorithm_version": str(
                info.get("algorithm_version", v8_runtime.policy.algorithm_version)
            ),
        })
    elif status is V8CascadeStatus.CANCELLED or semantic_terminal:
        info.update({
            "detector": "manual_review", "detector_used": "manual_review",
            "auto_v4_status": "cancelled" if status is V8CascadeStatus.CANCELLED else "manual_review",
            "confirmed": False,
        })
        info["auto_v4_witness"]["reason"] = (
            "cancelled" if status is V8CascadeStatus.CANCELLED else "semantic_terminal"
        )
    elif status is V8CascadeStatus.MANUAL_REVIEW:
        v8_quad = _auto_v4_valid_quad(
            core.get("algorithm_boundary_corners") or core.get("corners"), full_size
        )
        info.update({
            "detector": "manual_review", "detector_used": "manual_review",
            "detection_status": "manual_review", "confirmed": False,
        })
        if v8_quad is not None:
            info.update({
                "confirmation_primary_candidate_id": "v8:draft",
                "confirmation_selected_candidate_id": "v8:draft",
                "confirmation_selected_algorithm_version": str(
                    info.get("algorithm_version", v8_runtime.policy.algorithm_version)
                ),
            })
        if reason in auto_runtime.policy.witness_reason_allowlist:
            legacy_worker = _run_auto_v4_legacy(
                loaded_input,
                timeout_s=auto_runtime.policy.v52_worker_timeout_s,
                shrink_min=shrink_min,
                shrink_max=shrink_max,
                params=params,
                cancellation_token=cancellation_token,
                relative_path=relative_path,
            )
            info["cascade_calls"]["v5.2"] = 1
            legacy_worker.setdefault("source_sha256", loaded_input.source_sha256)
            legacy_worker.setdefault(
                "normalized_orientation", loaded_input.orientation_transform
            )
            legacy_worker.setdefault(
                "source_original_size", list(loaded_input.original_size)
            )
            legacy_worker.setdefault(
                "normalized_size", list(loaded_input.normalized_size)
            )
            witness = _evaluate_auto_v4_witness(
                core,
                legacy_worker,
                loaded_input=loaded_input,
                policy=auto_runtime.policy,
                reason=reason,
            )
            info["auto_v4_witness"] = witness
            if witness["action"] == "accept_current_v8":
                info.update({
                    "detector": "v8", "detector_used": "v8",
                    "detection_status": "v8_recommended",
                    "auto_v4_status": "automatic",
                })
    elif status is V8CascadeStatus.V7_FALLBACK:
        v7_quad = _auto_v4_valid_quad(
            core.get("algorithm_boundary_corners") or core.get("corners"), full_size
        )
        info.update({
            "detector": "manual_review", "detector_used": "manual_review",
            "detection_status": "manual_review", "confirmed": False,
            "auto_v4_witness": {
                "action": "not_applicable_fallback", "reason": reason,
                "policy_sha256": auto_runtime.policy.policy_sha256,
            },
        })
        if v7_quad is not None:
            info.update({
                "confirmation_primary_candidate_id": "v7:fallback",
                "confirmation_selected_candidate_id": "v7:fallback",
                "confirmation_selected_algorithm_version": str(
                    info.get("algorithm_version", V7_ALGORITHM_VERSION)
                ),
            })
        legacy_worker = _run_auto_v4_legacy(
            loaded_input,
            timeout_s=auto_runtime.policy.v52_worker_timeout_s,
            shrink_min=shrink_min,
            shrink_max=shrink_max,
            params=params,
            cancellation_token=cancellation_token,
            relative_path=relative_path,
        )
        info["cascade_calls"]["v5.2"] = 1
    else:
        info.update({
            "detector": "manual_review", "detector_used": "manual_review",
            "detection_status": "manual_review", "confirmed": False,
        })
        info["auto_v4_witness"]["reason"] = "unknown_v8_status"

    if legacy_worker is not None:
        info["cascade_v52_result"] = copy.deepcopy(legacy_worker)
        legacy_result = legacy_worker.get("result")
        legacy_quad = _auto_v4_valid_quad(
            legacy_result.get("corners") if isinstance(legacy_result, dict) else None,
            full_size,
        )
        if legacy_quad is not None:
            legacy_corners = [[float(x), float(y)] for x, y in legacy_quad]
            info["v52_corners"] = copy.deepcopy(legacy_corners)
            if (
                status is V8CascadeStatus.V7_FALLBACK
                and not info.get("confirmation_primary_candidate_id")
            ):
                for field in (
                    "algorithm_boundary_corners", "boundary_corners",
                    "algorithm_corners", "corners",
                ):
                    info[field] = copy.deepcopy(legacy_corners)
                info.update({
                    "algorithm_version": ALGORITHM_VERSION,
                    "success": True,
                    "confirmation_primary_candidate_id": "v52:fallback_gui_only",
                    "confirmation_selected_candidate_id": "v52:fallback_gui_only",
                    "confirmation_selected_algorithm_version": ALGORITHM_VERSION,
                })

    total_ms = (time.perf_counter() - started) * 1000.0
    info["auto_v4_timings_ms"] = {
        "adapter": adapter_ms,
        "total": total_ms,
        "wrapper": max(0.0, total_ms - adapter_ms),
    }
    return info


def _runtime_for_detection(
    args,
    source_records,
    params: DetectionParameters,
    *,
    v8_runtime: V8CliRuntime | Any | None = None,
    auto_runtime: AutoV4Runtime | None = None,
):
    # argparse always supplies the configured default (V8.4).  Direct Python
    # callers from the pre-v7 API may omit the field; preserve their legacy
    # v5.2 behavior instead of silently changing an in-process call.
    detector = getattr(args, "detector", None)
    if detector is None:
        detector = "v5.2"
    v7_mode = getattr(args, "v7_mode", None)
    scene_profile = getattr(args, "scene_profile", None) or DEFAULT_SCENE_PROFILE
    runtime_algorithm_version = ALGORITHM_VERSION
    if detector in {"v7", "auto"}:
        from photocut.algorithms.v7.parameters import V7Parameters
        v7_params = V7Parameters(
            mode="safe" if detector == "auto" else (v7_mode or "safe"),
            scene_profile=scene_profile,
        )
        from photocut.algorithms.v7.cascade import CASCADE_POLICY_SHA256, CASCADE_POLICY_VERSION
        detector_parameters = {
            "detector": detector,
            "v7_mode": v7_params.mode,
            "scene_profile": v7_params.scene_profile,
            "v7_algorithm_version": V7_ALGORITHM_VERSION,
            "v7_parameter_sha256": v7_params.sha256(),
            "v7_parameters": v7_params.to_dict(),
        }
        if detector == "auto":
            auto_runtime = auto_runtime or _resolve_auto_v4_runtime(args)
            runtime_algorithm_version = auto_runtime.policy.cascade_version
            detector_parameters.update({
                "detector_requested": "auto",
                "selector_name": SELECTOR_NAME,
                "selector_version": SELECTOR_VERSION,
                "auto_cascade_version": auto_runtime.policy.cascade_version,
                "auto_cascade_policy_sha256": auto_runtime.policy.policy_sha256,
                "auto_v4_wrapper_sha256": "sha256:" + hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "auto_engine_request_mode": auto_runtime.request_mode,
                "auto_engine_requested": auto_runtime.requested_engine,
                "auto_engine_effective": auto_runtime.effective_engine,
                "auto_engine_degraded": auto_runtime.degraded,
                "auto_engine_degradation_reason": auto_runtime.degradation_reason,
                "auto_v3_policy_version": CASCADE_POLICY_VERSION,
                "auto_v3_policy_sha256": CASCADE_POLICY_SHA256,
                "v7_safe_parameter_sha256": v7_params.sha256(),
                "v5_2_parameter_sha256": hashlib.sha256(json.dumps(
                    params.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                ).encode("utf-8")).hexdigest(),
                "v5_2_code_identity": _v52_code_identity(),
                "v5_2_worker_timeout_s": auto_runtime.policy.v52_worker_timeout_s,
                "v5_2_witness_auto_accept": (
                    auto_runtime.policy.v52_witness_auto_accept
                ),
            })
            if auto_runtime.effective_engine == "v8":
                if auto_runtime.v8_runtime is None:
                    raise RuntimeError("effective V8 auto-v4 route has no runtime")
                active = auto_runtime.v8_runtime
                policy = active.policy
                detector_parameters.update({
                    "v8_algorithm_version": policy.algorithm_version,
                    "v8_policy_sha256": policy.policy_sha256,
                    "v8_validated_core_sha256": policy.validated_core_sha256,
                    "v8_model_id": policy.model_id,
                    "v8_model_sha256": policy.model_sha256,
                    "v8_model_manifest_sha256": active.model_manifest_sha256,
                    "v8_runtime_config_sha256": active.runtime_config_sha256,
                    "v8_parameters": policy.parameters.to_dict(),
                })
                boundary_quality_sha256 = getattr(
                    policy, "boundary_quality_artifact_sha256", None
                )
                if boundary_quality_sha256 is not None:
                    detector_parameters[
                        "v8_boundary_quality_artifact_sha256"
                    ] = boundary_quality_sha256
    elif detector == "v8.4":
        if v8_runtime is None:
            raise RuntimeError("V8.4 runtime is required")
        runtime_algorithm_version = "8.4"
        detector_parameters = {
            "detector": "v8.4", "detector_requested": "v8.4",
            "scene_profile": "scanner_white", "model_sha256": v8_runtime.model_sha256,
            "requires_confirmation": True,
        }
    elif detector == "v8":
        if v8_runtime is None:
            raise RuntimeError("V8 检测需要先加载已验证的本机运行文件")
        policy = v8_runtime.policy
        runtime_algorithm_version = policy.algorithm_version
        detector_parameters = {
            "detector": "v8",
            "detector_requested": "v8",
            "scene_profile": "scanner_white",
            "v8_algorithm_version": policy.algorithm_version,
            "v8_policy_sha256": policy.policy_sha256,
            "v8_validated_core_sha256": policy.validated_core_sha256,
            "v8_model_id": policy.model_id,
            "v8_model_sha256": policy.model_sha256,
            "v8_model_manifest_sha256": v8_runtime.model_manifest_sha256,
            "v8_runtime_config_sha256": v8_runtime.runtime_config_sha256,
            "v8_parameters": policy.parameters.to_dict(),
        }
        boundary_quality_sha256 = getattr(
            policy, "boundary_quality_artifact_sha256", None
        )
        if boundary_quality_sha256 is not None:
            detector_parameters["v8_boundary_quality_artifact_sha256"] = (
                boundary_quality_sha256
            )
    else:
        # Preserve the byte-for-byte v5.2 runtime shape for omitted detector
        # invocations and existing resumable batches.
        detector_parameters = {}
    runtime_parameters = {
        "command": "detect",
        "output_identity": str(Path(args.output).resolve()),
        "shrink_min": args.shrink_min,
        "shrink_max": args.shrink_max,
        "effective_detector_parameters": params.to_dict(),
        "input_images": source_records,
    }
    runtime_parameters.update(detector_parameters)
    return {
        "algorithm_version": runtime_algorithm_version,
        "parameters": runtime_parameters,
    }


def _selected_detection(
    *, detector: str, v7_mode: str | None, img_path: Path, output_dir: str,
    shrink_min: int, shrink_max: int, params: DetectionParameters,
    img=None, relative_path: str | None = None, request_id: str | None = None,
    image_id: str | None = None, loaded_input=None,
    scene_profile: str | None = None,
    v8_runtime: V8CliRuntime | Any | None = None,
    auto_runtime: AutoV4Runtime | None = None,
):
    """Dispatch one image while keeping the v5.2 call path untouched."""
    if detector == "v8.4":
        if (scene_profile or DEFAULT_SCENE_PROFILE) != "scanner_white":
            raise ValueError("V8.4 supports scanner_white; use --detector auto --scene-profile generic_single")
        return detect_and_save_corners_v84(
            str(img_path), output_dir, runtime=v8_runtime, loaded_input=loaded_input,
            relative_path=relative_path, request_id=request_id, image_id=image_id,
        )
    if detector == "v7":
        return detect_and_save_corners_v7(
            str(img_path), output_dir, img=img, v7_mode=v7_mode or "safe",
            scene_profile=scene_profile or DEFAULT_SCENE_PROFILE,
            relative_path=relative_path, request_id=request_id, image_id=image_id,
            loaded_input=loaded_input,
        )
    if detector == "auto":
        if auto_runtime is not None and auto_runtime.effective_engine == "v8":
            return _detect_auto_v4_v8(
                img_path=img_path,
                output_dir=output_dir,
                auto_runtime=auto_runtime,
                loaded_input=loaded_input,
                relative_path=relative_path,
                request_id=request_id,
                image_id=image_id,
                shrink_min=shrink_min,
                shrink_max=shrink_max,
                params=params,
            )
        return detect_and_save_corners_auto(
            str(img_path), output_dir, img=img, relative_path=relative_path,
            request_id=request_id, image_id=image_id, shrink_min=shrink_min,
            shrink_max=shrink_max, params=params, loaded_input=loaded_input,
            scene_profile=scene_profile or DEFAULT_SCENE_PROFILE,
        )
    if detector == "v8":
        if v8_runtime is None:
            raise RuntimeError("V8 检测需要先加载已验证的本机运行文件")
        result = detect_and_save_corners_v8_dormant(
            str(img_path),
            output_dir,
            policy=v8_runtime.policy,
            mask_provider=v8_runtime.mask_provider,
            runtime_config=v8_runtime.runtime_config,
            runtime_config_sha256=v8_runtime.runtime_config_sha256,
            model_manifest_sha256=v8_runtime.model_manifest_sha256,
            boundary_quality_artifact=getattr(
                v8_runtime, "boundary_quality_artifact", None
            ),
            loaded_input=loaded_input,
            scene_profile=scene_profile or DEFAULT_SCENE_PROFILE,
            relative_path=relative_path,
            request_id=request_id,
            image_id=image_id,
        )
        info = dict(result.core_payload)
        envelope = dict(result.audit_envelope)
        info["detector_requested"] = "v8"
        for key in (
            "v8_provider_status",
            "v8_fallback_reason",
            "v8_policy_sha256",
            "v8_decision_status",
            "v8_selection_reason",
        ):
            if key in envelope:
                info[key] = envelope[key]
        return info
    return detect_and_save_corners(
        str(img_path), output_dir, shrink_min, shrink_max, img=img, params=params,
        relative_path=relative_path,
    )


def _prepare_detection_input(
    detector: str, source: Path, *, v7_mode: str | None,
    scene_profile: str | None = None,
):
    """Prepare one archived detection input without pre-decoding V7 at full size."""
    if detector in {"auto", "v7", "v8", "v8.4"}:
        from photocut.algorithms.v7.input import decode_bytes_for_analysis
        from photocut.algorithms.v7.parameters import V7Parameters

        mode = v7_mode or "safe"
        parameters = V7Parameters(
            mode=mode, scene_profile=scene_profile or DEFAULT_SCENE_PROFILE
        )
        loaded = decode_bytes_for_analysis(
            source.read_bytes(),
            max_pixels=parameters.max_input_pixels,
            analysis_max_edge=4096,
        )
        return None, loaded, tuple(loaded.full_normalized_size)
    image = load_image(str(source))
    if image is None:
        return None, None, None
    height, width = image.shape[:2]
    return image, None, (width, height)


def _stable_runtime(runtime):
    if not isinstance(runtime, dict):
        raise ValueError("runtime must be an object")
    algorithm_version = runtime.get("algorithm_version")
    parameters = runtime.get("parameters")
    if not isinstance(algorithm_version, str) or not isinstance(parameters, dict):
        raise ValueError("runtime is missing stable fields")
    return {"algorithm_version": algorithm_version, "parameters": parameters}


def _existing_auto_v4_route(
    store: DatasetStore,
    *,
    output_identity: str,
    source_records: list[dict[str, Any]],
    allow_final_repair: bool,
) -> dict[str, Any] | None:
    """Find one resumable auto-v4 route before attempting V8 preflight."""
    matches: list[dict[str, Any]] = []
    if not store.batches_dir.exists():
        return None
    route_fields = (
        "auto_engine_request_mode", "auto_engine_requested",
        "auto_engine_effective", "auto_engine_degraded",
        "auto_engine_degradation_reason",
    )
    for batch_dir in sorted(store.batches_dir.iterdir()):
        try:
            paths = store.batch_paths_from_candidate(batch_dir)
        except ValueError as exc:
            raise RuntimeError(
                f"cannot safely inspect batch {batch_dir.name}: invalid batch paths"
            ) from exc
        if paths is None:
            continue
        run = None
        if os.path.lexists(paths.run_in_progress):
            try:
                run = store.load_in_progress_run(paths)
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"cannot safely inspect auto-v4 batch {batch_dir.name}"
                ) from exc
        elif allow_final_repair and os.path.lexists(paths.production_run):
            try:
                with BatchLock(paths):
                    run = store.load_finalized_run(paths)
            except (OSError, ValueError, RuntimeError, TypeError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"cannot safely inspect finalized auto-v4 batch {batch_dir.name}"
                ) from exc
        if run is None:
            continue
        runtime = run.get("runtime")
        parameters = runtime.get("parameters") if isinstance(runtime, dict) else None
        if not isinstance(parameters, dict):
            raise RuntimeError(f"batch {batch_dir.name} has invalid runtime parameters")
        if (
            parameters.get("output_identity") != output_identity
            or parameters.get("input_images") != source_records
        ):
            continue
        if (
            parameters.get("detector_requested") != "auto"
            or parameters.get("auto_cascade_version") != "auto-v4"
        ):
            continue
        route = {field: parameters.get(field) for field in route_fields}
        route["batch_id"] = batch_dir.name
        matches.append(route)
    if len(matches) > 1:
        raise RuntimeError("multiple matching auto-v4 batches require manual recovery")
    return matches[0] if matches else None


def _resume_or_start_batch(store, runtime, allow_final_repair):
    inprogress_candidates = []
    finalized_candidates = []
    requested = _stable_runtime(runtime)
    requested_output = requested["parameters"]["output_identity"]
    if store.batches_dir.exists():
        for batch_dir in sorted(store.batches_dir.iterdir()):
            try:
                paths = store.batch_paths_from_candidate(batch_dir)
            except ValueError as exc:
                raise RuntimeError(
                    f"cannot safely inspect batch {batch_dir.name}: invalid batch paths"
                ) from exc
            if paths is None:
                continue
            if os.path.lexists(paths.run_in_progress):
                try:
                    run = store.load_in_progress_run(paths)
                    output_identity = run["runtime"]["parameters"].get("output_identity")
                    if output_identity != requested_output:
                        continue
                    recorded = _stable_runtime(run["runtime"])
                except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                    raise RuntimeError(
                        f"cannot safely resume batch {batch_dir.name}: invalid run metadata"
                    ) from exc
                if recorded != requested:
                    raise RuntimeError(
                        "in-progress batch runtime does not match this output, ordered source "
                        "collection, and effective detector parameters; refuse resume"
                    )
                inprogress_candidates.append((paths, run))
            elif allow_final_repair and os.path.lexists(paths.production_run):
                try:
                    with BatchLock(paths):
                        final = store.load_finalized_run(paths)
                    output_identity = final["runtime"]["parameters"].get("output_identity")
                    if output_identity != requested_output:
                        continue
                    recorded = _stable_runtime(final["runtime"])
                except (
                    OSError, ValueError, KeyError, TypeError, json.JSONDecodeError
                ) as exc:
                    raise RuntimeError(
                        f"cannot safely inspect finalized batch {batch_dir.name}"
                    ) from exc
                if recorded == requested:
                    finalized_candidates.append(paths)
    if len(inprogress_candidates) > 1:
        raise RuntimeError("multiple matching in-progress batches require manual recovery")
    if inprogress_candidates:
        return inprogress_candidates[0][0], None
    if len(finalized_candidates) > 1:
        raise RuntimeError("multiple matching finalized batches require manual recovery")
    if finalized_candidates:
        return finalized_candidates[0], True
    return store.start_batch(runtime=runtime), None


def _result_from_info(info):
    result = {
        "source_id": info["source_id"],
        "image_id": info["image_id"],
        "batch_id": info["batch_id"],
        "run_id": info["run_id"],
        "algorithm_version": info["algorithm_version"],
        "algorithm_boundary_corners": info["algorithm_boundary_corners"],
        "boundary_corners": info["boundary_corners"],
        "confidences": info.get("confidences", []),
        "detection_duration_ms": info.get("detection_duration_ms", 0.0),
        "success": info["success"],
        "error": info.get("error_message"),
        "legacy_info": info,
    }
    if (info.get("detector") in {"v7", "auto", "v8", "v8.4"} or
            info.get("detector_used") in {"v7", "auto", "v8", "v8.4"} or
            info.get("detector_requested") in {"v7", "auto", "v8", "v8.4"}):
        result.update({
            "detector": info.get("detector"),
            "detector_requested": info.get("detector_requested"),
            "detector_used": info.get("detector_used", info.get("detector")),
            "cascade_policy_version": info.get("cascade_policy_version"),
            "cascade_policy_sha256": info.get("cascade_policy_sha256"),
            "cascade_calls": info.get("cascade_calls"),
            "v5_2_parameter_sha256": info.get("v5_2_parameter_sha256"),
            "v5_2_code_identity": info.get("v5_2_code_identity"),
            "source_sha256": info.get("source_sha256"),
            "detection_id": info.get("detection_id"),
            "detection_identity": info.get("detection_identity"),
            "detection_status": info.get("detection_status"),
            "v7_mode": info.get("v7_mode"),
            "scene_profile": info.get("scene_profile"),
        })
        if info.get("detector_requested") == "v8.4":
            result.update({key: info.get(key) for key in (
                "model_sha256", "v84_status", "requires_confirmation",
            )})
        if info.get("detector_requested") == "v8":
            result.update({
                "v8_policy_sha256": info.get("v8_policy_sha256"),
                "v8_provider_status": info.get("v8_provider_status"),
                "v8_fallback_reason": info.get("v8_fallback_reason"),
                "v8_decision_status": info.get("v8_decision_status"),
                "v8_selection_reason": info.get("v8_selection_reason"),
            })
        if info.get("auto_cascade_version") == "auto-v4":
            for field in (
                "auto_cascade_version",
                "auto_cascade_policy_sha256",
                "auto_engine_request_mode",
                "auto_engine_requested",
                "auto_engine_effective",
                "auto_engine_degraded",
                "auto_engine_degradation_reason",
                "auto_v4_status",
                "auto_v4_witness",
                "auto_v4_timings_ms",
                "cascade_v8_result",
                "cascade_v52_result",
                "confirmation_primary_candidate_id",
                "confirmation_selected_candidate_id",
                "confirmation_selected_algorithm_version",
                "v52_corners",
                "v8_policy_sha256",
                "v8_provider_status",
                "v8_fallback_reason",
                "v8_decision_status",
                "v8_selection_reason",
            ):
                if field in info:
                    result[field] = copy.deepcopy(info[field])
    return result


def _decode_failure_info(source, source_id, image_id, batch_id, run_id, filename=None,
                         *, detector: str = DEFAULT_DETECTOR, v7_mode: str | None = None,
                         scene_profile: str | None = None,
                         params: DetectionParameters | None = None,
                         v8_runtime: V8CliRuntime | Any | None = None,
                         auto_runtime: AutoV4Runtime | None = None):
    info = {
        "filename": filename or source.name,
        "source_id": source_id,
        "image_id": image_id,
        "batch_id": batch_id,
        "run_id": run_id,
        "algorithm_version": ALGORITHM_VERSION,
        "algorithm_boundary_corners": [],
        "boundary_corners": [],
        "algorithm_corners": [],
        "algorithm_preview_corners": [],
        "corners": [],
        "preview_corners": [],
        "manual_corners": None,
        "manual_preview_corners": None,
        "original_size": [0, 0],
        "preview_size": [0, 0],
        "confidences": [],
        "detection_debug": [],
        "detection_duration_ms": 0.0,
        "success": False,
        "manually_adjusted": False,
        "algorithm_generated": True,
        "confirmed": False,
        "error_message": "cannot decode image",
        "adjust_count": 0,
        "adjust_timestamp": None,
    }
    if detector == "v8.4":
        digest = getattr(v8_runtime, "model_sha256", None)
        info.update({
            "algorithm_version": "8.4", "detector": "v8.4",
            "detector_requested": "v8.4", "detector_used": "v8.4",
            "model_sha256": digest, "detection_status": "error", "v84_status": "error",
            "requires_confirmation": True, "scene_profile": "scanner_white",
            "source_sha256": image_id.removeprefix("sha256:"),
            "detection_id": source_id, "normalized_orientation": "unknown",
            "detection_identity": {
                "request_id": source_id, "image_id": image_id,
                "algorithm_version": "8.4", "model_sha256": digest,
                "orientation_transform": "unknown", "mode": "requires_confirmation",
            },
        })
        return info
    if detector in {"v7", "auto", "v8", "v8.4"}:
        from photocut.algorithms.v7.parameters import V7Parameters
        mode = v7_mode or "safe"
        profile = scene_profile or DEFAULT_SCENE_PROFILE
        v7_params = V7Parameters(mode=mode, scene_profile=profile)
        requested = detector
        used = "manual_review" if detector in {"auto", "v8"} else "v7"
        algorithm_version = (
            v8_runtime.policy.algorithm_version
            if detector == "v8" and v8_runtime is not None
            else V7_ALGORITHM_VERSION
        )
        parameter_sha256 = (
            v8_runtime.policy.policy_sha256.removeprefix("sha256:")
            if detector == "v8" and v8_runtime is not None
            else v7_params.sha256()
        )
        detection_parameters = (
            v8_runtime.policy.parameters.to_dict()
            if detector == "v8" and v8_runtime is not None
            else v7_params.to_dict()
        )
        info.update({
            "algorithm_version": algorithm_version,
            "detector": used,
            "detector_requested": requested,
            "detector_used": used,
            "v7_mode": mode,
            "scene_profile": profile,
            "detection_status": "error",
            "detection_id": source_id,
            "detection_identity": {
                "request_id": source_id,
                "image_id": image_id,
                "orientation_transform": "unknown",
                "algorithm_version": algorithm_version,
                "parameter_sha256": parameter_sha256,
                "mode": mode,
            },
            "detection_parameters": detection_parameters,
        })
        if detector == "auto":
            from photocut.algorithms.v7.cascade import CASCADE_POLICY_SHA256, CASCADE_POLICY_VERSION
            v5_params = params or DEFAULT_DETECTION_PARAMETERS
            v5_parameter_sha256 = hashlib.sha256(json.dumps(
                v5_params.to_dict(), sort_keys=True, separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")).hexdigest()
            info.update({
                "cascade_policy_version": CASCADE_POLICY_VERSION,
                "cascade_policy_sha256": CASCADE_POLICY_SHA256,
                "cascade_calls": {"v7": 0, "v5.2": 0},
                "v5_2_parameter_sha256": v5_parameter_sha256,
                "v5_2_code_identity": _v52_code_identity(),
                "source_sha256": image_id.removeprefix("sha256:"),
                "normalized_orientation": "unknown",
                "source_original_size": [0, 0],
                "normalized_size": [0, 0],
            })
            if auto_runtime is not None:
                info.update({
                    "auto_cascade_version": auto_runtime.policy.cascade_version,
                    "auto_cascade_policy_sha256": auto_runtime.policy.policy_sha256,
                    "auto_engine_request_mode": auto_runtime.request_mode,
                    "auto_engine_requested": auto_runtime.requested_engine,
                    "auto_engine_effective": auto_runtime.effective_engine,
                    "auto_engine_degraded": auto_runtime.degraded,
                    "auto_engine_degradation_reason": auto_runtime.degradation_reason,
                    "auto_v4_status": "error",
                    "auto_v4_witness": {
                        "action": "not_attempted",
                        "reason": "input_decode_error",
                        "policy_sha256": auto_runtime.policy.policy_sha256,
                    },
                    "cascade_calls": {"v7": 0, "v8": 0, "v5.2": 0},
                })
                if auto_runtime.v8_runtime is not None:
                    info.update({
                        "v8_policy_sha256": (
                            auto_runtime.v8_runtime.policy.policy_sha256
                        ),
                        "v8_provider_status": "not_run",
                        "v8_fallback_reason": "input_decode_error",
                        "v8_decision_status": "manual_review",
                        "v8_selection_reason": "input_decode_error",
                    })
        elif detector == "v8" and v8_runtime is not None:
            info.update({
                "v8_policy_sha256": v8_runtime.policy.policy_sha256,
                "v8_provider_status": "not_run",
                "v8_fallback_reason": "input_decode_error",
                "v8_decision_status": "manual_review",
                "v8_selection_reason": "input_decode_error",
            })
    return info


def _restore_info(corners_dict, source_record, result):
    if result.get("source_id") != source_record["source_id"]:
        raise RuntimeError("production run source_id conflicts with ordered source collection")
    if result.get("image_id") != source_record["image_id"]:
        raise RuntimeError("production run image_id conflicts with ordered source collection")
    legacy_info = result.get("legacy_info")
    if not isinstance(legacy_info, dict):
        raise RuntimeError("production run result is missing recoverable legacy_info")
    if legacy_info.get("source_id") != source_record["source_id"]:
        raise RuntimeError("production run legacy_info conflicts with source_id")
    corners_dict[source_record["source_filename"]] = legacy_info


def _dataset_reference(project_root, dataset_root, batch):
    try:
        dataset_path = dataset_root.resolve().relative_to(project_root).as_posix()
    except ValueError:
        dataset_path = str(dataset_root.resolve())
    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "batch_id": batch.batch_id,
        "run_id": f"run_{batch.batch_id}",
        "dataset_root": dataset_path,
    }


def _repair_finalized_output(
    batch,
    final,
    source_records,
    corners_dict,
    json_path,
    reference_path,
    project_root,
    dataset_root,
):
    results = final.get("images")
    if not isinstance(results, list):
        raise RuntimeError("finalized production run has invalid results")
    by_source = {}
    for result in results:
        if not isinstance(result, dict) or not isinstance(result.get("source_id"), str):
            raise RuntimeError("finalized production run has invalid source result")
        source_id = result["source_id"]
        if source_id in by_source:
            raise RuntimeError("finalized production run has conflicting source results")
        by_source[source_id] = result
    if set(by_source) != {record["source_id"] for record in source_records}:
        raise RuntimeError("finalized production run does not match ordered sources")
    for source_record in source_records:
        _restore_info(corners_dict, source_record, by_source[source_record["source_id"]])
    save_corners_info(json_path, list(corners_dict.values()))
    atomic_write_json(
        reference_path,
        _dataset_reference(project_root, dataset_root, batch),
    )


def detect_command(args) -> None:
    """
    检测模式：扫描目录下的图片，检测四角
    """
    input_dir = args.input
    output_dir = args.output
    shrink_min = args.shrink_min
    shrink_max = args.shrink_max
    params = effective_detection_parameters(getattr(args, "threshold", None))
    # argparse supplies the configured default (V8.4).  Direct Python callers
    # from the legacy API may omit this field; keep those calls on v5.2.
    detector = getattr(args, "detector", None)
    if detector is None:
        detector = "v5.2"
    v7_mode = getattr(args, "v7_mode", None)
    scene_profile = getattr(args, "scene_profile", None) or DEFAULT_SCENE_PROFILE

    # 查找图片
    images = find_images(input_dir, output_dir)
    if not images:
        print(f"在 {input_dir} 中未找到图片")
        return

    print(f"找到 {len(images)} 张图片")
    v8_runtime = _v8_runtime_from_args(args) if detector == "v8" else None
    if detector == "v8.4":
        if scene_profile != "scanner_white":
            raise ValueError("V8.4 supports scanner_white; use --detector auto --scene-profile generic_single")
        from photocut.algorithms.v8_4.runtime import V84Runtime
        v8_runtime = V84Runtime()
    auto_runtime = None

    # 加载已存在的 corners_info（只保留当前input中存在的图片记录）
    json_path = os.path.join(output_dir, CORNERS_INFO_FILE)
    corners_info = load_corners_info(json_path)
    current_filenames = {relative_path for _, relative_path in images}
    # 过滤：只保留当前input中存在的图片记录，清除历史残留
    corners_dict = {e["filename"]: e for e in corners_info if e["filename"] in current_filenames}

    if getattr(args, "no_dataset_archive", False) is True:
        if detector == "auto":
            auto_runtime = _resolve_auto_v4_runtime(args)
            v8_runtime = auto_runtime.v8_runtime
        os.makedirs(output_dir, exist_ok=True)
        print("警告：数据归档已关闭，本批次不会进入长期数据集")
        for i, (img_path, filename) in enumerate(images, 1):
            print(f"[{i}/{len(images)}] 检测: {filename}", end=" ")
            if detector in {"v7", "auto", "v8", "v8.4"}:
                source = Path(img_path)
                try:
                    img, loaded_input, _ = _prepare_detection_input(
                        detector,
                        source,
                        v7_mode=v7_mode,
                        scene_profile=scene_profile,
                    )
                except Exception as exc:
                    img, loaded_input = None, None
                    input_error = exc
                else:
                    input_error = None
                if input_error is None:
                    info = _selected_detection(
                        detector=detector, v7_mode=v7_mode,
                        scene_profile=scene_profile, img_path=source,
                        output_dir=output_dir, shrink_min=shrink_min,
                        shrink_max=shrink_max, params=params, img=img,
                        relative_path=filename, loaded_input=loaded_input,
                        v8_runtime=v8_runtime, auto_runtime=auto_runtime,
                    )
                else:
                    source_record = _source_records([(source, filename)])[0]
                    info = _decode_failure_info(
                        source,
                        source_record["source_id"],
                        source_record["image_id"],
                        "no_dataset_archive",
                        "no_dataset_archive",
                        filename,
                        detector=detector,
                        v7_mode=v7_mode,
                        scene_profile=scene_profile,
                        params=params,
                        v8_runtime=v8_runtime,
                        auto_runtime=auto_runtime,
                    )
                    info["error_message"] = (
                        f"{type(input_error).__name__}: {input_error}"
                    )
            else:
                info = detect_and_save_corners(
                    img_path, output_dir, shrink_min, shrink_max, params=params, relative_path=filename
                )
            _preserve_confirmed_detection(corners_dict.get(filename), info, detector=detector)
            corners_dict[filename] = info
            save_corners_info(json_path, list(corners_dict.values()))
            print("✓" if info["success"] else f"✗ ({info.get('error_message', 'unknown error')})")
        print(f"\n检测完成，结果保存到: {json_path}")
        return

    project_root = Path(__file__).resolve().parents[1]
    dataset_root = Path(
        getattr(args, "dataset_root", default_dataset_root(project_root))
    )
    store = DatasetStore(dataset_root)
    source_paths = [(Path(path), relative_path) for path, relative_path in images]
    source_files = [source for source, _ in source_paths]
    store.preflight_sources(source_files)
    source_records = _source_records(source_paths)
    reference_path = Path(output_dir) / BATCH_REFERENCE_FILE
    if detector == "auto":
        existing_route = _existing_auto_v4_route(
            store,
            output_identity=str(Path(output_dir).resolve()),
            source_records=source_records,
            allow_final_repair=not reference_path.exists(),
        )
        auto_runtime = _resolve_auto_v4_runtime(
            args,
            existing_route=existing_route,
        )
        v8_runtime = auto_runtime.v8_runtime
    runtime = _runtime_for_detection(
        args,
        source_records,
        params,
        v8_runtime=v8_runtime,
        auto_runtime=auto_runtime,
    )
    batch, finalized = _resume_or_start_batch(
        store, runtime, allow_final_repair=not reference_path.exists()
    )
    os.makedirs(output_dir, exist_ok=True)
    if finalized is not None:
        try:
            with BatchLock(batch):
                final = store.load_finalized_run(batch)
        except (OSError, ValueError, RuntimeError) as exc:
            raise RuntimeError(
                f"cannot safely repair finalized batch {batch.batch_id}"
            ) from exc
        if _stable_runtime(final["runtime"]) != _stable_runtime(runtime):
            raise RuntimeError("finalized batch runtime changed during repair")
        _repair_finalized_output(
            batch,
            final,
            source_records,
            corners_dict,
            json_path,
            reference_path,
            project_root,
            dataset_root,
        )
        print(f"\n检测完成，结果保存到: {json_path}")
        return
    run_id = f"run_{batch.batch_id}"

    try:
        deferred_error = None
        with BatchLock(batch):
            try:
                run = json.loads(batch.run_in_progress.read_text(encoding="utf-8"))
                completed = {
                    entry["source_id"]: entry
                    for entry in run.get("images", [])
                    if isinstance(entry, dict) and isinstance(entry.get("source_id"), str)
                }
                for i, ((source, filename), source_record) in enumerate(zip(source_paths, source_records), 1):
                    print(f"[{i}/{len(images)}] 检测: {filename}", end=" ")
                    existing = completed.get(source_record["source_id"])
                    if existing is not None:
                        _restore_info(corners_dict, source_record, existing)
                        save_corners_info(json_path, list(corners_dict.values()))
                        print("✓ (恢复已有结果)")
                        continue

                    input_error = None
                    try:
                        img, loaded_input, image_size = _prepare_detection_input(
                            detector, source, v7_mode=v7_mode,
                            scene_profile=scene_profile,
                        )
                    except Exception as exc:
                        img, loaded_input, image_size = None, None, None
                        input_error = exc
                    archived = store.archive_image(source, image_size=image_size)
                    if archived.image_id != source_record["image_id"]:
                        raise RuntimeError("source content changed after preflight")
                    if loaded_input is not None and (
                        f"sha256:{loaded_input.source_sha256}" != archived.image_id
                    ):
                        raise RuntimeError("bounded source snapshot changed before archive")
                    store.register_archived_image(
                        batch, archived, source_id=source_record["source_id"]
                    )
                    if input_error is not None or (detector == "v5.2" and img is None):
                        info = _decode_failure_info(
                            source,
                            source_record["source_id"],
                            archived.image_id,
                            batch.batch_id,
                            run_id,
                            filename,
                            detector=detector,
                            v7_mode=v7_mode,
                            scene_profile=scene_profile,
                            params=params,
                            v8_runtime=v8_runtime,
                            auto_runtime=auto_runtime,
                        )
                        if input_error is not None:
                            info["error_message"] = f"{type(input_error).__name__}: {input_error}"
                        store.update_run(batch, _result_from_info(info))
                        corners_dict[filename] = info
                        save_corners_info(json_path, list(corners_dict.values()))
                        print("✗ (cannot decode image)")
                        continue

                    info = _selected_detection(
                        detector=detector, v7_mode=v7_mode, scene_profile=scene_profile,
                        img_path=source,
                        output_dir=output_dir, shrink_min=shrink_min, shrink_max=shrink_max,
                        params=params, img=img, relative_path=filename,
                        request_id=source_record["source_id"], image_id=archived.image_id,
                        loaded_input=loaded_input, v8_runtime=v8_runtime,
                        auto_runtime=auto_runtime,
                    )
                    info["source_id"] = source_record["source_id"]
                    info["image_id"] = archived.image_id
                    info["batch_id"] = batch.batch_id
                    info["run_id"] = run_id
                    _preserve_confirmed_detection(corners_dict.get(filename), info, detector=detector)
                    store.update_run(batch, _result_from_info(info))
                    corners_dict[filename] = info
                    save_corners_info(json_path, list(corners_dict.values()))
                    print("✓" if info["success"] else f"✗ ({info.get('error_message', 'unknown error')})")
                store.finalize_run(batch)
            except BaseException as exc:
                deferred_error = exc
    except RuntimeError as exc:
        if "batch already locked" in str(exc):
            raise RuntimeError(
                f"batch is locked; manual recovery required before resume: {batch.lock}"
            ) from exc
        raise

    if deferred_error is not None:
        raise deferred_error

    atomic_write_json(
        reference_path,
        _dataset_reference(project_root, dataset_root, batch),
    )
    print(f"\n检测完成，结果保存到: {json_path}")


def crop_command(args) -> None:
    """
    裁剪模式：根据 corners_info.json 裁剪图片
    """
    input_dir = args.input
    output_dir = args.output
    skip = set(args.skip.split(',')) if args.skip else set()

    # 加载 corners_info
    json_path = os.path.join(output_dir, CORNERS_INFO_FILE)
    corners_info = load_corners_info(json_path)

    if not corners_info:
        print(f"未找到 {json_path}，请先运行检测命令")
        return
    try:
        discovered = find_images(input_dir, output_dir)
    except ValueError as exc:
        print(f"无法安全扫描输入目录: {exc}")
        return
    accepted_entries, rejected_entries = normalize_entry_relative_paths(corners_info, discovered)
    for entry, reason in rejected_entries:
        print(f"裁剪跳过 {entry.get('filename', '')}: {reason}")
    sources_by_relative = {relative: Path(path) for path, relative in discovered}
    basename_counts = {
        basename: sum(Path(relative).name == basename for relative in sources_by_relative)
        for basename in {Path(relative).name for relative in sources_by_relative}
    }
    save_corners_info(json_path, corners_info)

    try:
        cropped_dir = _safe_cropped_directory(Path(output_dir))
    except (OSError, ValueError) as exc:
        print(f"无法创建安全的裁剪输出目录: {exc}")
        return
    try:
        crop_event_store = resolve_crop_event_store(
            Path(output_dir), Path(__file__).resolve().parents[1]
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"无法打开裁剪事件存储: {exc}")
        return
    if crop_event_store is None:
        print("警告：未找到批次引用，将按兼容模式仅保存裁剪结果")

    # 统计
    total = 0
    success = 0
    skipped = 0
    for entry, relative_path in accepted_entries:
        filename = entry["filename"]

        basename = Path(filename).name
        # Relative paths match exactly; bare basenames only match unambiguous sources.
        if filename in skip or (basename in skip and basename_counts[basename] == 1):
            skipped += 1
            continue

        # 跳过未成功的
        if not entry.get("success", False):
            skipped += 1
            continue

        # 跳过未确认的
        if not entry.get("confirmed", False):
            skipped += 1
            continue

        source = sources_by_relative.get(relative_path)
        if source is None:
            print(f"图片不存在或路径不安全: {filename}")
            skipped += 1
            continue

        total += 1
        boundary_corners = entry.get("boundary_corners", entry["corners"])
        try:
            crop_corners = inset_quadrilateral(
                boundary_corners,
                distance_px=args.inset,
                image_size=tuple(entry["original_size"]),
            )
        except InsetGeometryError as exc:
            print(f"裁剪跳过 {filename}: {exc}")
            skipped += 1
            continue

        # V7 and all auto-cascade entries use the verified EXIF-normalized
        # snapshot that produced their corners. Explicit legacy entries retain
        # the original loader.
        if _entry_uses_normalized_snapshot(entry):
            entry_img = _load_entry_image(entry, source, output_dir=Path(output_dir))
            if entry_img is None:
                skipped += 1
                continue
        else:
            # Keep the byte-for-byte legacy crop seam (and its lazy loading)
            # unchanged for v5.2/old entries.
            entry_img = None
        if entry_img is None:
            ok, output_path = crop_image(
                str(source), crop_corners, str(cropped_dir), relative_path
            )
        else:
            ok, output_path = crop_image(
                str(source), crop_corners, str(cropped_dir), relative_path, img=entry_img
            )

        entry["boundary_corners"] = boundary_corners
        entry["crop_corners"] = crop_corners
        entry["inset"] = {
            "method": "parallel_edge_offset",
            "distance_px": float(args.inset),
        }

        if ok:
            if crop_event_store is not None:
                try:
                    event = build_crop_event(entry, output_path, entry["inset"])
                    crop_event_store.append_idempotent(event)
                except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                    print(f"裁剪文件已生成但溯源记录失败 {filename}: {exc}")
                    continue
            save_corners_info(json_path, corners_info)
            success += 1
            print(f"[{success}] 裁剪完成: {filename}")
        else:
            if crop_event_store is not None:
                try:
                    event = build_crop_event(
                        entry, output_path, entry["inset"], error="crop_image returned failure"
                    )
                    crop_event_store.append_idempotent(event)
                except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                    print(f"裁剪失败且无法记录溯源 {filename}: {exc}")
            print(f"裁剪失败: {filename}")

    print(f"\n裁剪完成: {success}/{total} 张成功，{skipped} 张跳过")


def opencv_confirmation_action(key: int, key_ascii: int, *, navigate: int = 0):
    """Translate one native input into the shared session action vocabulary."""
    if navigate == -1:
        return "previous", {}
    if navigate == 1:
        return "next", {}
    if key_ascii in (32, 13):
        return "confirm", {}
    if key_ascii in (ord("r"), ord("R")):
        return "reset", {}
    candidate = v7_candidate_key_action(key_ascii)
    if candidate == "alternate":
        return "candidate", {}
    if candidate == "v5.2":
        return "v52", {}
    if candidate == "skip":
        return "skip", {"reason": "user_skip"}
    if key_ascii in (ord("p"), ord("P")):
        return "pause", {}
    if key_ascii == ord("q"):
        return "quit", {}
    arrow_delta = {
        65361: (-ARROW_MOVE_PX, 0),
        63234: (-ARROW_MOVE_PX, 0),
        65363: (ARROW_MOVE_PX, 0),
        63235: (ARROW_MOVE_PX, 0),
        65362: (0, -ARROW_MOVE_PX),
        63232: (0, -ARROW_MOVE_PX),
        65364: (0, ARROW_MOVE_PX),
        63233: (0, ARROW_MOVE_PX),
    }.get(key)
    delta = arrow_delta or coarse_move_delta(key_ascii)
    if delta is not None:
        return "move", {"dx": delta[0], "dy": delta[1]}
    return None


def dispatch_opencv_confirmation_action(session, kind: str, payload: dict):
    revision = session.snapshot()["revision"]
    return session.dispatch(
        ConfirmationAction(uuid.uuid4().hex, revision, kind, payload)
    )


def opencv_window_is_closed(cv2_module, window_name: str) -> bool:
    getter = getattr(cv2_module, "getWindowProperty", None)
    if getter is None:
        return False
    property_id = getattr(cv2_module, "WND_PROP_VISIBLE", 0)
    try:
        return float(getter(window_name, property_id)) < 1.0
    except Exception:
        return False


def checkpoint_closed_opencv_window(session, cv2_module, window_name: str) -> bool:
    """Checkpoint a native title-bar close exactly once through the session."""
    if not opencv_window_is_closed(cv2_module, window_name):
        return False
    if session.status == "active":
        dispatch_opencv_confirmation_action(session, "quit", {})
    return True


def confirm_command(args) -> None:
    frontend = getattr(args, "confirm_ui", "web")
    gui_version = WEB_GUI_VERSION if frontend == "web" else OPENCV_GUI_VERSION
    try:
        session = create_confirmation_session(args, gui_version)
    except ValueError as exc:
        print(f"无法安全建立确认会话: {exc}")
        return
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"无法打开确认事件存储: {exc}")
        return
    if session is None:
        return
    if frontend == "web":
        run_web_confirmation(session)
        return
    if not check_gui_and_prompt():
        return
    print(f"需要确认 {session.backend.count()} 张图片")
    run_opencv_confirmation(session)


def run_web_confirmation(session) -> None:
    """Serve the opt-in browser UI and always release its local listener."""
    import webbrowser

    from photocut.confirmation.web.server import LocalConfirmationServer

    with LocalConfirmationServer(session) as server:
        url = server.start()
        print(f"本地确认界面：{url}")
        try:
            opened = webbrowser.open(url, new=1, autoraise=True)
        except Exception:
            opened = False
        if not opened:
            print("浏览器未自动打开，请复制上面的本地地址。")
        try:
            server.wait_until_session_stops()
        except KeyboardInterrupt:
            if session.status == "active":
                dispatch_opencv_confirmation_action(session, "pause", {})
            print("已暂停，当前进度已保存")


def run_opencv_confirmation(session) -> None:
    """Render the OpenCV GUI while all writes flow through session actions."""
    import cv2

    current_idx = session.snapshot()["progress"]["index"]
    to_confirm = session.backend.entries

    # 角点颜色方案：角1=绿色，角2=黄色，角3=青色，角4=紫红色
    CORNER_COLORS = [
        (0, 255, 0),       # 绿色
        (0, 255, 255),     # 黄色
        (255, 255, 0),     # 青色
        (255, 0, 255),     # 紫红色
    ]

    def draw_corners_and_lines(img, corners, selected=-1):
        """
        在图像上绘制角点和四边形连线

        Args:
            img: 原始图像副本
            corners: 角点坐标列表 [[x,y], ...]
            selected: 选中的角点索引 (-1 表示无选中)

        Returns:
            绘制完成的图像
        """
        # The caller supplies a fresh frame canvas, so drawing in place avoids
        # another multi-megabyte allocation on every GUI refresh.
        result = img
        h, w = img.shape[:2]

        # 绘制四边形连线（闭合四边形）- 虚线样式
        # 连接顺序：0->1->2->3->0
        def draw_dashed_line(img, pt1, pt2, color, thickness=3, dash_length=15, gap_length=10):
            """绘制虚线"""
            x1, y1 = pt1
            x2, y2 = pt2

            # 计算线段长度和方向
            dx = x2 - x1
            dy = y2 - y1
            dist = np.sqrt(dx**2 + dy**2)

            if dist < 1e-6:
                return

            # 单位方向向量
            ux, uy = dx / dist, dy / dist

            # 绘制虚线段
            current_dist = 0
            while current_dist < dist:
                # 计算当前段的起始点
                seg_start_x = int(x1 + ux * current_dist)
                seg_start_y = int(y1 + uy * current_dist)

                # 计算当前段的结束点（不超出总长度）
                seg_end_dist = min(current_dist + dash_length, dist)
                seg_end_x = int(x1 + ux * seg_end_dist)
                seg_end_y = int(y1 + uy * seg_end_dist)

                # 绘制这一段
                cv2.line(img, (seg_start_x, seg_start_y), (seg_end_x, seg_end_y),
                        color, thickness, cv2.LINE_AA)

                # 移动到下一个虚线段（加上间隔）
                current_dist += dash_length + gap_length

        line_indices = [(0, 1), (1, 2), (2, 3), (3, 0)]
        for i, j in line_indices:
            x1, y1 = int(corners[i][0]), int(corners[i][1])
            x2, y2 = int(corners[j][0]), int(corners[j][1])
            # 绿色虚线，3像素粗
            draw_dashed_line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 绘制角点（十字准星）
        for i, (x, y) in enumerate(corners):
            x, y = int(x), int(y)
            color = CORNER_COLORS[i]

            if i == selected:
                # 选中的角点用大十字准星
                cross_size = 36
                cross_gap = 12
                # 垂直线（上下）
                cv2.line(result, (x, y - cross_size), (x, y - cross_gap), color, 2)
                cv2.line(result, (x, y + cross_gap), (x, y + cross_size), color, 2)
                # 水平线（左右）
                cv2.line(result, (x - cross_size, y), (x - cross_gap, y), color, 2)
                cv2.line(result, (x + cross_gap, y), (x + cross_size, y), color, 2)
                # 中心实心圆
                cv2.circle(result, (x, y), 4, color, -1)
            else:
                # 未选中的角点用小十字准星
                cross_size = 20
                cross_gap = 6
                # 垂直线（上下）
                cv2.line(result, (x, y - cross_size), (x, y - cross_gap), color, 1)
                cv2.line(result, (x, y + cross_gap), (x, y + cross_size), color, 1)
                # 水平线（左右）
                cv2.line(result, (x - cross_size, y), (x - cross_gap, y), color, 1)
                cv2.line(result, (x + cross_gap, y), (x + cross_size, y), color, 1)

            # 绘制角点编号
            cv2.putText(result, str(i + 1), (x + 12, y - 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return result

    def draw_button(sidebar, x, y, width, height, text, enabled=True, hovered=False, pressed=False):
        """绘制导航按钮"""
        if not enabled:
            bg_color = (40, 40, 40)
            border_color = (60, 60, 60)
            text_color = (120, 120, 120)
        elif pressed:
            # 按下状态：背景变暗，边框高亮（内凹效果）
            bg_color = (40, 40, 40)
            border_color = (180, 180, 180)
            text_color = (255, 255, 255)
        elif hovered:
            # 悬停状态：背景变亮，边框高亮
            bg_color = (80, 80, 80)
            border_color = (150, 150, 150)
            text_color = (255, 255, 255)
        else:
            # 正常状态
            bg_color = (60, 60, 60)
            border_color = (100, 100, 100)
            text_color = (255, 255, 255)

        # 背景
        cv2.rectangle(sidebar, (x, y), (x + width, y + height), bg_color, -1)
        # 边框
        cv2.rectangle(sidebar, (x, y), (x + width, y + height), border_color, 2)
        # 文字居中（按下时稍微偏移，模拟内凹效果）
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = x + (width - text_size[0]) // 2
        text_y = y + (height + text_size[1]) // 2
        if pressed:
            text_x += 1
            text_y += 1
        cv2.putText(sidebar, text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

    def is_in_magnifier(x, y):
        """判断鼠标坐标是否在放大镜区域内"""
        # 布局常量（与主循环中保持一致）
        TOTAL_W = 2560
        SIDEBAR_W = 500
        MAIN_W = TOTAL_W - SIDEBAR_W  # 2060
        MAGNIFIER_OFFSET_X = 50
        MAGNIFIER_OFFSET_Y = 50
        MAGNIFIER_SIZE = 400

        # 放大镜在主窗口中的绝对坐标
        mag_x1 = MAIN_W + MAGNIFIER_OFFSET_X
        mag_x2 = mag_x1 + MAGNIFIER_SIZE
        mag_y1 = MAGNIFIER_OFFSET_Y
        mag_y2 = mag_y1 + MAGNIFIER_SIZE

        return mag_x1 <= x < mag_x2 and mag_y1 <= y < mag_y2

    def dispatch_mouse_move(dx, dy):
        try:
            dispatch_opencv_confirmation_action(
                session, "move", {"dx": int(dx), "dy": int(dy)}
            )
        except ConfirmationSessionError as exc:
            print(f"角点移动被拒绝: {exc}")

    # 鼠标拖动状态（使用字典在回调和主循环间共享）
    mouse_state = {
        "state": None,
        "dragging": False,
        "transform": None,
        "magnifier_dragging": False,  # 新增：是否在放大镜拖动
        "magnifier_drag_anchor": (0, 0),
        "global_pointer_reader": create_global_pointer_reader(),
        "global_magnifier_capture": MagnifierDragCapture(),
        "global_magnifier_capture_active": False,
        "navigate": 0,  # 导航指令: -1=上一张, 0=无, 1=下一张
        "prev_button_hovered": False,  # 上一张按钮悬停状态
        "next_button_hovered": False,  # 下一张按钮悬停状态
        "prev_button_pressed": False,  # 上一张按钮按下状态
        "next_button_pressed": False,  # 下一张按钮按下状态
        "dispatch_move": dispatch_mouse_move,
    }

    def mouse_callback(event, x, y, flags, param):
        """鼠标回调：处理角点拖动和放大镜拖动"""
        ms = param["mouse_state"]
        state = ms["state"]

        if event == cv2.EVENT_LBUTTONDOWN:
            # 检测导航按钮点击
            # 按钮在边栏内，边栏从 x=MAIN_W 开始
            if x >= MAIN_W:
                sidebar_x = x - MAIN_W
                sidebar_y = y

                # 上一张按钮区域（只设置按下状态，不立即导航）
                if (PREV_BUTTON_X <= sidebar_x <= PREV_BUTTON_X + NAV_BUTTON_WIDTH and
                    NAV_BUTTON_Y <= sidebar_y <= NAV_BUTTON_Y + NAV_BUTTON_HEIGHT):
                    ms["prev_button_pressed"] = True
                    return

                # 下一张按钮区域（只设置按下状态，不立即导航）
                if (NEXT_BUTTON_X <= sidebar_x <= NEXT_BUTTON_X + NAV_BUTTON_WIDTH and
                    NAV_BUTTON_Y <= sidebar_y <= NAV_BUTTON_Y + NAV_BUTTON_HEIGHT):
                    ms["next_button_pressed"] = True
                    return

            # 判断是否在放大镜区域且已选中角点
            if state is not None and is_in_magnifier(x, y) and state.selected >= 0:
                # 开始在放大镜拖动
                begin_magnifier_drag(ms, x, y)
            else:
                transform = ms["transform"]
                if state is not None and transform is not None:
                    min_dist = float('inf')
                    nearest_idx = -1
                    for i, corner in enumerate(state.work_corners):
                        cx, cy = original_to_display(corner, transform)
                        dist = np.hypot(x - cx, y - cy)
                        if dist < 30 and dist < min_dist:
                            min_dist = dist
                            nearest_idx = i
                    if nearest_idx >= 0:
                        try:
                            dispatch_opencv_confirmation_action(
                                session, "select_corner", {"index": nearest_idx}
                            )
                            ms["dragging"] = True
                        except ConfirmationSessionError as exc:
                            print(f"角点选择被拒绝: {exc}")

        elif event == cv2.EVENT_MOUSEMOVE:
            # 检测按钮悬停状态
            if x >= MAIN_W:
                sidebar_x = x - MAIN_W
                sidebar_y = y

                # 检测上一张按钮悬停
                in_prev = (
                    PREV_BUTTON_X <= sidebar_x <= PREV_BUTTON_X + NAV_BUTTON_WIDTH and
                    NAV_BUTTON_Y <= sidebar_y <= NAV_BUTTON_Y + NAV_BUTTON_HEIGHT
                )
                ms["prev_button_hovered"] = in_prev

                # 检测下一张按钮悬停
                in_next = (
                    NEXT_BUTTON_X <= sidebar_x <= NEXT_BUTTON_X + NAV_BUTTON_WIDTH and
                    NAV_BUTTON_Y <= sidebar_y <= NAV_BUTTON_Y + NAV_BUTTON_HEIGHT
                )
                ms["next_button_hovered"] = in_next

                # 如果按钮处于按下状态但移出了按钮区域，取消按下状态
                if ms.get("prev_button_pressed") and not in_prev:
                    ms["prev_button_pressed"] = False
                if ms.get("next_button_pressed") and not in_next:
                    ms["next_button_pressed"] = False
            else:
                # 鼠标不在边栏，清除悬停和按下状态
                ms["prev_button_hovered"] = False
                ms["next_button_hovered"] = False
                ms["prev_button_pressed"] = False
                ms["next_button_pressed"] = False

            if (state is not None and ms["magnifier_dragging"] and
                    not ms.get("global_magnifier_capture_active") and state.selected >= 0):
                anchor_x, anchor_y = ms["magnifier_drag_anchor"]
                dx, dy = magnifier_drag_delta(x - anchor_x, y - anchor_y, state.zoom)
                if dx or dy:
                    dispatch_mouse_move(dx, dy)
                    # Keep sub-pixel screen motion accumulated until it crosses a source pixel.
                    ms["magnifier_drag_anchor"] = (x, y)

            elif state is not None and ms["dragging"] and state.selected >= 0:
                transform = ms["transform"]
                if transform is not None:
                    target_x, target_y = display_to_original([x, y], transform)
                    try:
                        dispatch_opencv_confirmation_action(
                            session,
                            "set_corner",
                            {"index": state.selected, "x": target_x, "y": target_y},
                        )
                    except ConfirmationSessionError as exc:
                        print(f"角点拖动被拒绝: {exc}")

        elif event == cv2.EVENT_LBUTTONUP:
            # 处理按钮释放（如果在按钮区域内释放才触发导航）
            if x >= MAIN_W and ms.get("prev_button_pressed"):
                sidebar_x = x - MAIN_W
                sidebar_y = y
                # 检查是否还在上一张按钮区域内
                if (PREV_BUTTON_X <= sidebar_x <= PREV_BUTTON_X + NAV_BUTTON_WIDTH and
                    NAV_BUTTON_Y <= sidebar_y <= NAV_BUTTON_Y + NAV_BUTTON_HEIGHT):
                    ms["navigate"] = -1
                ms["prev_button_pressed"] = False

            if x >= MAIN_W and ms.get("next_button_pressed"):
                sidebar_x = x - MAIN_W
                sidebar_y = y
                # 检查是否还在下一张按钮区域内
                if (NEXT_BUTTON_X <= sidebar_x <= NEXT_BUTTON_X + NAV_BUTTON_WIDTH and
                    NAV_BUTTON_Y <= sidebar_y <= NAV_BUTTON_Y + NAV_BUTTON_HEIGHT):
                    ms["navigate"] = 1
                ms["next_button_pressed"] = False

            ms["dragging"] = False
            cancel_magnifier_drag(ms)

    # 固定窗口名（窗口复用）
    WINDOW_NAME = "PhotoCut"

    # 在循环前创建窗口并注册鼠标回调（只创建一次）
    cv2.namedWindow(WINDOW_NAME)
    cv2.imshow(WINDOW_NAME, np.zeros((100, 100, 3), dtype=np.uint8))
    cv2.waitKey(100)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback, {"mouse_state": mouse_state})

    while session.status == "active":
        session_snapshot = session.snapshot()
        current_idx = session_snapshot["progress"]["index"]
        item = session.current_item
        entry = item.entry
        filename = entry["filename"]
        img = item.image
        finalized_evidence = item.finalized_evidence
        display_identity = _gui_identity_entry(entry, finalized_evidence)
        previous_annotation = item.previous_annotation
        state = item.editor.state
        v7_model = item.editor.model

        # 初始化鼠标状态
        mouse_state["state"] = state
        mouse_state["dragging"] = False
        cancel_magnifier_drag(mouse_state)

        # 布局常量：左右分栏布局
        TOTAL_W = 2560
        TOTAL_H = 1440
        SIDEBAR_W = 500
        MAIN_W = TOTAL_W - SIDEBAR_W  # 2060
        IMG_H = 1200
        HINT_H = TOTAL_H - IMG_H  # 240

        # 放大镜常量
        MAGNIFIER_SIZE = 400
        MAGNIFIER_OFFSET_X = 50
        MAGNIFIER_OFFSET_Y = 50
        MAGNIFIER_INFO_Y = 480  # 坐标详情起始 Y 位置

        # 导航按钮常量
        NAV_BUTTON_Y = 1370
        NAV_BUTTON_WIDTH = 100
        NAV_BUTTON_HEIGHT = 40
        PREV_BUTTON_X = 50
        NEXT_BUTTON_X = 170
        PROGRESS_Y = 1300

        cv2.resizeWindow(WINDOW_NAME, TOTAL_W, TOTAL_H)
        _, identity_lines = update_confirmation_gui_identity(
            cv2, WINDOW_NAME, display_identity, previous_annotation
        )

        try:
            preview_cache = build_confirmation_preview(
                img, viewport_width=MAIN_W, viewport_height=IMG_H
            )
        except (cv2.error, MemoryError, ValueError) as exc:
            print(f"预览生成失败，已保留待确认状态 {filename}: {type(exc).__name__}: {exc}")
            dispatch_opencv_confirmation_action(session, "pause", {})
            cv2.destroyAllWindows()
            return

        # 主循环
        while True:
            poll_magnifier_drag(mouse_state)
            # 从鼠标状态获取选中状态
            selected = state.selected
            work_corners = state.work_corners
            manually_adjusted = bool(state.adjusted_corner_indices)

            # 原图坐标是唯一工作状态；主预览在进入本图时只缩放一次。
            scaled_w = preview_cache.scaled_width
            scaled_h = preview_cache.scaled_height
            transform = preview_cache.transform
            img_offset_x = transform.offset_x
            mouse_state["transform"] = transform

            scaled_corners = [original_to_display(corner, transform) for corner in work_corners]

            # 创建左侧主画布：图片区域 + 提示区域
            main_canvas = np.zeros((IMG_H + HINT_H, MAIN_W, 3), dtype=np.uint8) + 30
            # 图片居中放置在图片区域
            main_canvas[:scaled_h, img_offset_x:img_offset_x + scaled_w] = preview_cache.scaled_image
            main_canvas[:IMG_H] = draw_corners_and_lines(
                main_canvas[:IMG_H], scaled_corners, selected
            )

            # 在提示区域显示操作提示
            hint_panel = main_canvas[IMG_H:, :]
            hints = [
                "SPACE/ENTER: Confirm  r/R: Reset  q: Save&Quit  p/P: Pause",
                "1-4: Select  WASD: 50px  Arrows: 5px  +/-: Zoom  Mouse: Drag"
            ]
            for i, hint in enumerate(hints):
                cv2.putText(hint_panel, hint, (15, 35 + i * 28),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

            # 创建右侧边栏
            sidebar = np.zeros((TOTAL_H, SIDEBAR_W, 3), dtype=np.uint8) + 30

            # ========== 放大镜窗口 ==========
            # 放大镜边框
            cv2.rectangle(sidebar,
                         (MAGNIFIER_OFFSET_X, MAGNIFIER_OFFSET_Y),
                         (MAGNIFIER_OFFSET_X + MAGNIFIER_SIZE, MAGNIFIER_OFFSET_Y + MAGNIFIER_SIZE),
                         (80, 80, 80), 2)

            # 放大镜区域（深灰色背景）
            magnifier_roi = sidebar[MAGNIFIER_OFFSET_Y:MAGNIFIER_OFFSET_Y + MAGNIFIER_SIZE,
                                  MAGNIFIER_OFFSET_X:MAGNIFIER_OFFSET_X + MAGNIFIER_SIZE]
            magnifier_roi[:] = (20, 20, 20)

            # 当有选中角点时，绘制放大后的图像区域
            if selected >= 0:
                source = render_magnifier_source(
                    img, work_corners[selected], state.zoom, MAGNIFIER_SIZE
                )
                magnifier_roi[:] = cv2.resize(
                    source,
                    (MAGNIFIER_SIZE, MAGNIFIER_SIZE),
                    interpolation=cv2.INTER_NEAREST,
                )

                # 放大镜中心十字准星
                center_x = MAGNIFIER_OFFSET_X + MAGNIFIER_SIZE // 2
                center_y = MAGNIFIER_OFFSET_Y + MAGNIFIER_SIZE // 2
                color = CORNER_COLORS[selected]
                cv2.line(sidebar, (center_x - 30, center_y), (center_x - 10, center_y), color, 2)
                cv2.line(sidebar, (center_x + 10, center_y), (center_x + 30, center_y), color, 2)
                cv2.line(sidebar, (center_x, center_y - 30), (center_x, center_y - 10), color, 2)
                cv2.line(sidebar, (center_x, center_y + 10), (center_x, center_y + 30), color, 2)
                cv2.circle(sidebar, (center_x, center_y), 3, color, -1)

                # 在放大镜中绘制四边形连线
                def draw_dashed_line_on_sidebar(sidebar_img, pt1, pt2, color, thickness=2, dash_length=10, gap_length=6):
                    """在边栏上绘制虚线"""
                    x1, y1 = pt1
                    x2, y2 = pt2

                    dx = x2 - x1
                    dy = y2 - y1
                    dist = np.sqrt(dx**2 + dy**2)

                    if dist < 1e-6:
                        return

                    ux, uy = dx / dist, dy / dist

                    current_dist = 0
                    while current_dist < dist:
                        seg_start_x = int(x1 + ux * current_dist)
                        seg_start_y = int(y1 + uy * current_dist)

                        seg_end_dist = min(current_dist + dash_length, dist)
                        seg_end_x = int(x1 + ux * seg_end_dist)
                        seg_end_y = int(y1 + uy * seg_end_dist)

                        cv2.line(sidebar_img, (seg_start_x, seg_start_y), (seg_end_x, seg_end_y),
                                color, thickness, cv2.LINE_AA)

                        current_dist += dash_length + gap_length

                # 计算四个角点在放大镜中的位置
                # 选中角点在放大镜中心 (center_x, center_y)
                # 其他角点根据相对位置计算
                mag_corner_positions = []
                selected_cx, selected_cy = work_corners[selected]

                for i, (corner_x, corner_y) in enumerate(work_corners):
                    # 以原图像素的相对偏移绘制在当前放大倍率的视图中。
                    rel_x = corner_x - selected_cx
                    rel_y = corner_y - selected_cy

                    mag_x = center_x + int(rel_x * state.zoom)
                    mag_y = center_y + int(rel_y * state.zoom)

                    mag_corner_positions.append((mag_x, mag_y))

                # 绘制四边形连线（虚线）
                line_color = (0, 255, 0)  # 绿色
                line_indices = [(0, 1), (1, 2), (2, 3), (3, 0)]
                for i, j in line_indices:
                    draw_dashed_line_on_sidebar(
                        sidebar,
                        mag_corner_positions[i],
                        mag_corner_positions[j],
                        line_color,
                        2
                    )

                # 添加遮罩：用放大镜边框覆盖超出区域，只保留内部
                # 创建遮罩层，四个方向遮挡超出部分
                mask_color = (30, 30, 30)  # 边栏背景色
                # 上方遮挡
                sidebar[0:MAGNIFIER_OFFSET_Y, :] = mask_color
                # 下方遮挡
                sidebar[MAGNIFIER_OFFSET_Y + MAGNIFIER_SIZE:TOTAL_H, :] = mask_color
                # 左侧遮挡（只在边栏范围内）
                sidebar[:, 0:MAGNIFIER_OFFSET_X] = mask_color
                # 右侧遮挡
                sidebar[:, MAGNIFIER_OFFSET_X + MAGNIFIER_SIZE:SIDEBAR_W] = mask_color

                # 重新绘制放大镜边框（覆盖遮罩边缘）
                cv2.rectangle(sidebar,
                             (MAGNIFIER_OFFSET_X, MAGNIFIER_OFFSET_Y),
                             (MAGNIFIER_OFFSET_X + MAGNIFIER_SIZE, MAGNIFIER_OFFSET_Y + MAGNIFIER_SIZE),
                             (80, 80, 80), 2)

            # 未选中角点时显示提示文字
            if selected < 0:
                hint_text = "Press 1-4 to select corner"
                cv2.putText(magnifier_roi, hint_text, (70, 200),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 120, 120), 1)

            # 边栏标题
            cv2.putText(sidebar, "Corner Details", (15, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            for identity_index, identity_line in enumerate(identity_lines):
                cv2.putText(
                    sidebar,
                    identity_line,
                    (15, 72 + identity_index * 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (210, 210, 210),
                    1,
                )
            if v7_model is not None:
                cv2.putText(
                    sidebar,
                    f"{SELECTOR_NAME if isinstance(v7_model, AutoV4ConfirmationViewModel) else 'Detector'}: {v7_model.status_label}",
                    (15, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 220, 255) if v7_model.can_confirm else (0, 120, 255), 1,
                )
                risk_text = " | ".join(v7_model.risk_labels[:2])
                if risk_text:
                    cv2.putText(sidebar, risk_text[:48], (15, 210),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 180, 255), 1)
                cv2.putText(sidebar, "C: alternate  B/b: v5.2  X: skip",
                            (15, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                            (180, 180, 180), 1)
            cv2.line(sidebar, (15, 45), (SIDEBAR_W - 15, 45), (100, 100, 100), 1)
            cv2.putText(sidebar, f"Zoom: {state.zoom}x", (300, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

            # 显示四角坐标详情
            for i, (x, y) in enumerate(work_corners):
                color = CORNER_COLORS[i]
                label = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"][i]
                text = f"C{i+1}: {label}"
                cv2.putText(sidebar, text, (15, MAGNIFIER_INFO_Y + i * 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
                cv2.putText(sidebar, f"X: {int(x):4d}  Y: {int(y):4d}", (25, MAGNIFIER_INFO_Y + 25 + i * 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # 显示当前选中角点
            if selected >= 0:
                cv2.putText(sidebar, f"Selected: C{selected + 1}",
                           (15, TOTAL_H - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                           CORNER_COLORS[selected], 1)

            # 绘制导航按钮
            # 上一张按钮（第一张时禁用，支持悬停和点击效果）
            prev_enabled = current_idx > 0
            prev_hovered = prev_enabled and mouse_state.get("prev_button_hovered", False)
            prev_pressed = prev_enabled and mouse_state.get("prev_button_pressed", False)
            draw_button(sidebar, PREV_BUTTON_X, NAV_BUTTON_Y,
                       NAV_BUTTON_WIDTH, NAV_BUTTON_HEIGHT, "< Prev", prev_enabled, prev_hovered, prev_pressed)

            # 下一张按钮（最后一张时禁用，支持悬停和点击效果）
            next_enabled = current_idx < len(to_confirm) - 1
            next_hovered = next_enabled and mouse_state.get("next_button_hovered", False)
            next_pressed = next_enabled and mouse_state.get("next_button_pressed", False)
            draw_button(sidebar, NEXT_BUTTON_X, NAV_BUTTON_Y,
                       NAV_BUTTON_WIDTH, NAV_BUTTON_HEIGHT, "Next >", next_enabled, next_hovered, next_pressed)

            # 在右侧边栏显示进度（大号字体，居中）
            progress_text = f"{current_idx + 1} / {len(to_confirm)}"
            text_size = cv2.getTextSize(progress_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            progress_x = (SIDEBAR_W - text_size[0]) // 2
            cv2.putText(sidebar, progress_text, (progress_x, PROGRESS_Y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            # 水平拼接：左侧主画布 + 右侧边栏
            combined = np.hstack([main_canvas, sidebar])

            # 复用窗口显示
            cv2.imshow(WINDOW_NAME, combined)

            # Give the native window manager time to process resize/fullscreen
            # and input events instead of saturating the thread with redraws.
            key = cv2.waitKeyEx(GUI_FRAME_DELAY_MS)
            if checkpoint_closed_opencv_window(session, cv2, WINDOW_NAME):
                cv2.destroyAllWindows()
                print("窗口已关闭，当前进度已保存")
                return
            # 普通字符键用 ASCII 码 (key & 0xFF)，方向键用扩展码
            key_ascii = key & 0xFF if 0 <= key <= 255 else 0
            # 方向键扩展码 (waitKeyEx 返回值)
            ARROW_LEFT = 65361   # 左箭头
            ARROW_RIGHT = 65363  # 右箭头
            ARROW_DOWN = 65364   # 下箭头
            ARROW_UP = 65362     # 上箭头

            action = opencv_confirmation_action(
                key, key_ascii, navigate=mouse_state.get("navigate", 0)
            )
            if action is None and key_ascii in map(ord, "1234"):
                action = "select_corner", {"index": key_ascii - ord("1")}
            elif action is None and key_ascii in (ord("+"), ord("=")):
                next_zoom = next(
                    (zoom for zoom in (4, 8, 16) if zoom > state.zoom), 16
                )
                action = "set_zoom", {"zoom": next_zoom}
            elif action is None and key_ascii in (ord("-"), ord("_")):
                next_zoom = next(
                    (zoom for zoom in (16, 8, 4) if zoom < state.zoom), 4
                )
                action = "set_zoom", {"zoom": next_zoom}

            if action is None:
                continue
            kind, payload = action
            if kind == "move" and state.selected < 0:
                continue
            cancel_magnifier_drag(mouse_state)
            mouse_state["navigate"] = 0
            try:
                dispatch_opencv_confirmation_action(session, kind, payload)
            except ConfirmationSessionError as exc:
                print(f"确认动作被拒绝 {filename}: {exc}")
                continue
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                print(confirmation_persistence_error_message(filename, exc))
                continue

            if kind in {"pause", "quit"}:
                cv2.destroyAllWindows()
                if kind == "pause":
                    print(
                        f"已暂停，进度已保存 ({current_idx}/{len(to_confirm)} 张已确认)"
                    )
                else:
                    print("已保存并退出")
                return
            if kind in {"confirm", "skip", "previous", "next"}:
                break
            state = session.current_item.editor.state
            v7_model = session.current_item.editor.model
            mouse_state["state"] = state

    cv2.destroyAllWindows()
    print("所有图片已确认")


def main():
    parser = argparse.ArgumentParser(
        description="photocut - 老照片批量裁剪工具",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("input", help="输入目录（包含源图片）")
    parser.add_argument("-o", "--output", default=None, help="输出目录（默认: 输入目录同级的 <名称>裁剪）")
    parser.add_argument("-t", "--threshold", type=int, default=240,
                       help="白边检测阈值（默认: 240）")
    parser.add_argument("-s", "--shrink-min", type=int, default=CORNER_SHRINK_MIN,
                       help=f"角点最小收缩量（默认: {CORNER_SHRINK_MIN}）")
    parser.add_argument("-S", "--shrink-max", type=int, default=CORNER_SHRINK_MAX,
                       help=f"角点最大收缩量（默认: {CORNER_SHRINK_MAX}）")
    parser.add_argument("--skip", default="", help="跳过的文件名（逗号分隔）")
    add_inset_argument(parser, default=50.0)
    add_detector_arguments(parser, default=DEFAULT_DETECTOR)
    add_confirmation_arguments(parser, default="web")
    parser.set_defaults(v7_mode=None)
    parser.set_defaults(scene_profile=None)
    parser.set_defaults(v8_policy=None)
    parser.set_defaults(v8_model_dir=None)
    parser.set_defaults(auto_engine=None)

    subparsers = parser.add_subparsers(dest="command", help="命令")

    # detect 命令
    detect_parser = subparsers.add_parser("detect", help="检测四角")
    detect_option_parser = subparsers.add_parser("--detect", help="检测四角")
    for command_parser in (detect_parser, detect_option_parser):
        command_parser.add_argument(
            "--no-dataset-archive",
            action="store_true",
            default=argparse.SUPPRESS,
            help="兼容模式：不归档原图或生成长期数据（该批次不能用于算法优化）",
        )
        add_detector_arguments(command_parser)

    # crop 命令
    crop_parser = subparsers.add_parser("crop", help="裁剪图片")
    add_inset_argument(crop_parser)

    # confirm 命令
    confirm_parser = subparsers.add_parser("confirm", help="确认四角")
    add_confirmation_arguments(confirm_parser)
    confirm_parser.add_argument(
        "--revise-confirmed",
        action="append",
        default=[],
        metavar="FILENAME",
        help="显式打开已确认图片进行交互修订；确认后追加 superseding 标注事件",
    )

    # --detect, --crop, --confirm 作为独立参数
    parser.add_argument("--detect", action="store_true", help="检测四角")
    parser.add_argument(
        "--no-dataset-archive",
        action="store_true",
        help="兼容模式：不归档原图或生成长期数据（该批次不能用于算法优化）",
    )
    parser.add_argument("--crop", action="store_true", help="裁剪图片")
    parser.add_argument(
        "--revise-confirmed",
        action="append",
        default=[],
        metavar="FILENAME",
        help="与 --confirm 一起使用，显式交互修订已确认图片",
    )
    parser.add_argument("--confirm", action="store_true", help="确认四角")

    args = parser.parse_args()
    if getattr(args, "v7_mode", None) is not None and getattr(args, "detector", DEFAULT_DETECTOR) != "v7":
        parser.error("--v7-mode requires --detector v7; auto uses its fixed safe policy")
    if (getattr(args, "auto_engine", None) is not None and
            getattr(args, "detector", DEFAULT_DETECTOR) != "auto"):
        parser.error("--auto-engine requires --detector auto")
    if (getattr(args, "scene_profile", None) is not None and
            getattr(args, "detector", DEFAULT_DETECTOR) not in {"auto", "v7", "v8", "v8.4"}):
        parser.error("--scene-profile requires --detector v8.4, auto, v7, or v8")
    if (getattr(args, "detector", DEFAULT_DETECTOR) == "v8.4" and
            getattr(args, "scene_profile", None) not in {None, "scanner_white"}):
        parser.error("V8.4 supports scanner_white; use --detector auto --scene-profile generic_single")
    if (getattr(args, "detector", DEFAULT_DETECTOR) == "v8" and
            getattr(args, "scene_profile", None) not in {None, "scanner_white"}):
        parser.error("--detector v8 only supports --scene-profile scanner_white")
    if (getattr(args, "detector", DEFAULT_DETECTOR) == "auto" and
            getattr(args, "auto_engine", None) == "v8" and
            getattr(args, "scene_profile", None) == "generic_single"):
        parser.error("--auto-engine v8 only supports --scene-profile scanner_white")
    if ((getattr(args, "v8_policy", None) is not None or
            getattr(args, "v8_model_dir", None) is not None) and
            getattr(args, "detector", DEFAULT_DETECTOR) != "v8"):
        parser.error("--v8-policy/--v8-model-dir require --detector v8")
    if args.inset < 0:
        parser.error("--inset must be non-negative")
    if args.output is None:
        raw_input = Path(args.input)
        if not raw_input.name:
            parser.error("cannot safely derive output directory from input")
        args.output = str(raw_input.parent / f"{raw_input.name}裁剪")

    # 确定命令
    if args.command in ["detect", "--detect"] or args.detect:
        detect_command(args)
    elif args.command == "crop" or args.crop:
        crop_command(args)
    elif args.command == "confirm" or args.confirm:
        confirm_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
