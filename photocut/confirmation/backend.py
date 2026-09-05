"""PhotoCut persistence and image-loading backend for confirmation sessions."""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import stat
import time
from pathlib import Path
from typing import Callable

import numpy as np

from photocut.data.annotation_store import AnnotationStore
from photocut.confirmation.controller import (
    ConfirmationEntryController,
    ConfirmationSessionController,
    LoadedConfirmationItem,
)
from photocut.confirmation.identity import FinalizedDetectionEvidence, match_finalized_detection
from photocut.confirmation.model import build_annotation_event, legacy_preview_corners
from photocut.config import BATCH_REFERENCE_FILE, CORNERS_INFO_FILE
from photocut.data.dataset_store import BatchLock, DatasetStore
from photocut.confirmation.version import OPENCV_GUI_VERSION
from photocut.core import (
    load_corners_info,
    load_image,
    reconcile_confirmation_events,
    save_corners_info,
)


def save_manual_boundary(entry, manual_corners, manual_preview_corners) -> None:
    entry["manual_preview_corners"] = [list(corner) for corner in manual_preview_corners]
    entry["manual_corners"] = [list(corner) for corner in manual_corners]
    entry["preview_corners"] = [list(corner) for corner in manual_preview_corners]
    entry["corners"] = [list(corner) for corner in manual_corners]
    entry["boundary_corners"] = [list(corner) for corner in manual_corners]
    entry["manually_adjusted"] = True


def reset_confirmation_boundary(entry) -> None:
    algorithm_boundary = entry.get(
        "algorithm_boundary_corners", entry.get("algorithm_corners", entry["corners"])
    )
    algorithm_preview = entry.get(
        "algorithm_preview_corners", entry.get("preview_corners", [])
    )
    entry["manual_corners"] = None
    entry["manual_preview_corners"] = None
    entry["boundary_corners"] = [list(corner) for corner in algorithm_boundary]
    entry["corners"] = [list(corner) for corner in algorithm_boundary]
    entry["preview_corners"] = [list(corner) for corner in algorithm_preview]
    entry["manually_adjusted"] = False


def save_confirmation_draft(entry, state) -> None:
    """Save an unconfirmed original-pixel edit without creating an annotation."""
    boundary = [point[:] for point in state.work_corners]
    entry["boundary_corners"] = boundary
    entry["corners"] = [point[:] for point in boundary]
    entry["manual_corners"] = [point[:] for point in boundary]
    entry["manually_adjusted"] = bool(state.adjusted_corner_indices)
    entry["draft_adjusted_corner_indices"] = sorted(state.adjusted_corner_indices)
    entry["confirmed"] = False


def commit_confirmation(
    entry,
    state,
    annotation_store,
    duration_ms,
    *,
    finalized_evidence: FinalizedDetectionEvidence | None = None,
    detection_evidence: FinalizedDetectionEvidence | None = None,
    gui_version: str = OPENCV_GUI_VERSION,
    previous_annotation: dict | None = None,
):
    """Durably append the formal annotation before changing mutable UI state."""
    if finalized_evidence is not None and detection_evidence is not None:
        raise ValueError("provide only one finalized detection evidence argument")
    if detection_evidence is not None:
        finalized_evidence = detection_evidence
    identity = {}
    candidate_identity = {}
    if entry.get("auto_cascade_version") == "auto-v4":
        for source_field, event_field in (
            ("confirmation_primary_candidate_id", "confirmation_primary_candidate_id"),
            ("confirmation_selected_candidate_id", "selected_candidate_id"),
            (
                "confirmation_selected_algorithm_version",
                "selected_candidate_algorithm_version",
            ),
        ):
            value = entry.get(source_field)
            if value is not None:
                candidate_identity[event_field] = value
        if set(candidate_identity) != {
            "confirmation_primary_candidate_id",
            "selected_candidate_id",
            "selected_candidate_algorithm_version",
        }:
            raise ValueError("auto-v4 confirmation candidate identity is incomplete")
    if finalized_evidence is not None:
        if candidate_identity:
            algorithm_corners = [point[:] for point in state.algorithm_corners]
        else:
            algorithm_corners = [
                list(point) for point in finalized_evidence.algorithm_boundary_corners
            ]
        adjusted_corner_indices = [
            index
            for index, (before, after) in enumerate(
                zip(algorithm_corners, state.work_corners)
            )
            if before != after
        ]
        identity = {
            "algorithm_version": candidate_identity.get(
                "selected_candidate_algorithm_version",
                finalized_evidence.algorithm_version,
            ),
            "detection_id": finalized_evidence.detection_id,
            "detector_requested": finalized_evidence.detector_requested,
            "detector_used": finalized_evidence.detector_used,
        }
    else:
        algorithm_corners = state.algorithm_corners
        if candidate_identity.get("selected_candidate_algorithm_version"):
            identity["algorithm_version"] = candidate_identity[
                "selected_candidate_algorithm_version"
            ]
        if previous_annotation is None and entry.get("annotation_id"):
            previous_annotation = next(
                (
                    event
                    for event in annotation_store.events()
                    if event["annotation_id"] == entry["annotation_id"]
                ),
                None,
            )
            if previous_annotation is None:
                raise ValueError("revision target has no formal annotation")
        if isinstance(previous_annotation, dict) and previous_annotation.get(
            "algorithm_boundary_corners"
        ):
            algorithm_corners = [
                list(point)
                for point in previous_annotation["algorithm_boundary_corners"]
            ]
        adjusted_corner_indices = [
            index
            for index, (before, after) in enumerate(
                zip(algorithm_corners, state.work_corners)
            )
            if before != after
        ]
    event = build_annotation_event(
        image_id=entry["image_id"],
        run_id=entry["run_id"],
        algorithm_boundary_corners=algorithm_corners,
        boundary_corners=state.work_corners,
        adjusted_corner_indices=adjusted_corner_indices,
        confirmation_duration_ms=duration_ms,
        supersedes_annotation_id=entry.get("annotation_id"),
        schema_version=2,
        gui_version=gui_version,
        **identity,
        **candidate_identity,
    )
    event = annotation_store.append_idempotent(event)
    boundary = [point[:] for point in event["boundary_corners"]]
    entry["boundary_corners"] = boundary
    entry["corners"] = [point[:] for point in boundary]
    entry["manual_corners"] = [point[:] for point in boundary]
    entry["confirmed"] = True
    entry["manually_adjusted"] = event["confirmation"] == "adjusted"
    entry["annotation_id"] = event["annotation_id"]
    entry["confirm_timestamp"] = event["confirmed_at"]
    entry.pop("draft_adjusted_corner_indices", None)
    return event


def _gui_identity_entry(entry, finalized_evidence):
    """Build display-only identity from verified evidence, never mutable version fields."""
    if finalized_evidence is None:
        return {
            "algorithm_version": None,
            "detector_requested": None,
            "detector_used": None,
        }
    return {
        "algorithm_version": finalized_evidence.algorithm_version,
        "detector_requested": finalized_evidence.detector_requested,
        "detector_used": finalized_evidence.detector_used,
    }


def _skip_stale_revision(filename, reason) -> None:
    print(
        f"跳过修订条目 {filename}: {reason}。未进入交互；"
        "请重新打开确认模式以加载最新正式标注。"
    )


def confirmation_persistence_error_message(filename, exc) -> str:
    """Explain a rejected confirmation without discarding the open UI state."""
    return (
        f"确认未提交 {filename}: 标注存储冲突或写入失败 ({exc})。"
        "当前调整尚未写入正式标注；请重新打开确认模式以加载最新 head 后重试。"
    )


def select_confirmation_entries(corners_info, revise_filenames=(), annotation_store=None):
    """Return normal work plus explicitly requested, formally confirmed revisions."""
    requested = set(revise_filenames)
    by_name = {entry["filename"]: entry for entry in corners_info}
    missing = requested - set(by_name)
    if missing:
        raise ValueError(f"unknown revision filenames: {', '.join(sorted(missing))}")
    for filename in requested:
        entry = by_name[filename]
        if not entry.get("confirmed") or not entry.get("annotation_id"):
            raise ValueError(
                f"{filename} has no formal annotation; import legacy truth first"
            )
        if not entry.get("image_id"):
            raise ValueError(f"{filename} has no source image identity")

    valid_revisions = requested
    if requested and annotation_store is not None:
        events_by_id = {
            event["annotation_id"]: event for event in annotation_store.events()
        }
        latest_by_image = annotation_store.latest_by_image()
        valid_revisions = set()
        for filename in requested:
            entry = by_name[filename]
            annotation_id = entry["annotation_id"]
            recorded = events_by_id.get(annotation_id)
            if recorded is None:
                _skip_stale_revision(
                    filename, f"找不到正式 annotation_id {annotation_id}"
                )
                continue
            if recorded["image_id"] != entry["image_id"]:
                _skip_stale_revision(filename, "annotation_id 属于另一张图片")
                continue
            current = latest_by_image.get(entry["image_id"])
            if current is None:
                _skip_stale_revision(filename, "该图片没有当前正式 head")
                continue
            if current["annotation_id"] != annotation_id:
                _skip_stale_revision(filename, "annotation_id 不是当前正式 head")
                continue
            valid_revisions.add(filename)
    return [
        entry
        for entry in corners_info
        if not entry.get("confirmed", False) or entry["filename"] in valid_revisions
    ]


class _OriginAnnotationStore:
    """Route inherited confirmations to their original append-only history."""

    def __init__(self, current, entries):
        self.current = current
        self.path = current.path
        self.routes = {}
        dataset = DatasetStore(current.path.parents[2])
        stores = {}
        for entry in entries:
            origin = entry.get("confirmation_origin_batch_id")
            if origin is None:
                continue
            if not isinstance(origin, str) or Path(origin).name != origin:
                raise ValueError("invalid confirmation origin batch")
            paths = dataset.batch_paths_from_candidate(dataset.batches_dir / origin)
            if paths is None:
                raise ValueError("confirmation origin batch is unavailable")
            if origin not in stores:
                target = AnnotationStore(paths.annotations)
                stores[origin] = (target, {event["annotation_id"]: event for event in target.events()})
            target, events = stores[origin]
            previous = events.get(entry.get("annotation_id"))
            image_id = entry.get("image_id")
            if previous is None or previous["image_id"] != image_id:
                raise ValueError("confirmation origin has no matching image annotation")
            if image_id in self.routes and self.routes[image_id].path != target.path:
                raise ValueError("conflicting confirmation origins for the same image")
            self.routes[image_id] = target

    def events(self):
        values = [event for event in self.current.events() if event["image_id"] not in self.routes]
        grouped = {}
        for image_id, target in self.routes.items():
            grouped.setdefault(target.path, (target, set()))[1].add(image_id)
        for target, image_ids in grouped.values():
            values.extend(event for event in target.events() if event["image_id"] in image_ids)
        return values

    def latest_by_image(self):
        values = {key: event for key, event in self.current.latest_by_image().items() if key not in self.routes}
        grouped = {}
        for image_id, target in self.routes.items():
            grouped.setdefault(target.path, (target, set()))[1].add(image_id)
        for target, image_ids in grouped.values():
            values.update({key: event for key, event in target.latest_by_image().items() if key in image_ids})
        return values

    def append_idempotent(self, event):
        return self.routes.get(event["image_id"], self.current).append_idempotent(event)


def _annotation_store_from_reference(output_dir, project_root=None):
    reference_path = Path(output_dir) / BATCH_REFERENCE_FILE
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    dataset_root = Path(reference["dataset_root"])
    if not dataset_root.is_absolute():
        dataset_root = (
            Path(project_root or output_dir).resolve() / dataset_root
        ).resolve()
    return AnnotationStore(
        dataset_root / "batches" / reference["batch_id"] / "annotations.jsonl"
    )


def _dataset_root_from_reference(dataset_root: str, project_root: Path) -> Path:
    raw_root = Path(dataset_root)
    if ".." in raw_root.parts:
        raise ValueError("batch reference dataset_root must not contain parent traversal")
    root = Path(project_root).resolve()
    candidate = raw_root if raw_root.is_absolute() else root / raw_root
    candidate = Path(os.path.abspath(candidate))
    current = Path(candidate.anchor)
    for part in candidate.parts[1:]:
        current /= part
        try:
            info = current.lstat()
        except OSError as exc:
            raise ValueError(
                "batch reference dataset_root must be an existing directory"
            ) from exc
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise ValueError("batch reference dataset_root must be a regular directory")
    resolved = candidate.resolve(strict=True)
    if not raw_root.is_absolute():
        try:
            resolved.relative_to(root)
        except ValueError:
            raise ValueError(
                "batch reference dataset_root escapes the project"
            ) from None
    return resolved


def _load_finalized_detection_evidence(
    entry, output_dir: Path, project_root: Path | None = None
) -> FinalizedDetectionEvidence | None:
    """Load immutable per-image identity for one GUI confirmation entry."""
    reference_path = Path(output_dir) / BATCH_REFERENCE_FILE
    if not os.path.lexists(reference_path):
        return None
    if reference_path.is_symlink() or not reference_path.is_file():
        raise ValueError("batch reference must be a regular file")
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    if not isinstance(reference, dict):
        raise ValueError("batch reference must be an object")
    dataset_root = _dataset_root_from_reference(
        reference["dataset_root"], Path(project_root or output_dir)
    )
    store = DatasetStore(dataset_root)
    batch_id = reference["batch_id"]
    if not isinstance(batch_id, str) or not batch_id:
        raise ValueError("batch reference contains an invalid batch_id")
    paths = store.batch_paths_from_candidate(store.batches_dir / batch_id)
    if paths is None:
        raise ValueError("batch reference contains an invalid batch_id")
    if not os.path.lexists(paths.production_run):
        return None
    with BatchLock(paths):
        finalized_run = store.load_finalized_run(paths)
    return match_finalized_detection(entry, reference, finalized_run)


def _load_archived_v7_image(entry, output_dir: Path) -> np.ndarray | None:
    """Reopen the batch-owned immutable object for a V7 coordinate entry."""
    image_id = entry.get("image_id")
    reference_path = Path(output_dir) / BATCH_REFERENCE_FILE
    if not isinstance(image_id, str) or not image_id.startswith("sha256:"):
        return None
    if not reference_path.is_file() or reference_path.is_symlink():
        return None
    try:
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        project_root = Path(__file__).resolve().parents[2]
        dataset_root = _dataset_root_from_reference(
            reference["dataset_root"], project_root
        )
        store = DatasetStore(dataset_root)
        paths = store.batch_paths_from_candidate(
            store.batches_dir / reference["batch_id"]
        )
        if paths is None:
            return None
        manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
        records = [
            item
            for item in manifest.get("images", [])
            if item.get("image_id") == image_id
        ]
        if len(records) != 1:
            return None
        relative_object = records[0].get("object_path")
        relative_path = Path(relative_object) if isinstance(relative_object, str) else None
        if (
            relative_path is None
            or relative_path.is_absolute()
            or ".." in relative_path.parts
        ):
            return None
        object_path = dataset_root / relative_path
        info = object_path.lstat()
        if object_path.is_symlink() or not stat.S_ISREG(info.st_mode):
            return None
        data = object_path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != image_id.removeprefix("sha256:"):
            return None
        expected = entry.get("source_sha256")
        if expected and digest != expected:
            return None
        from photocut.algorithms.v7.input import decode_full_bgr_bytes

        orientation_value = str(entry.get("normalized_orientation", "exif_1"))
        if not orientation_value.startswith("exif_"):
            return None
        orientation = int(orientation_value.removeprefix("exif_"))
        expected_size = tuple(
            entry.get("normalized_size") or entry.get("original_size") or ()
        )
        if len(expected_size) != 2:
            return None
        return decode_full_bgr_bytes(
            data,
            exif_orientation=orientation,
            expected_size=expected_size,
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _entry_uses_normalized_snapshot(entry) -> bool:
    used = entry.get("detector_used", entry.get("detector"))
    return (
        entry.get("detector_requested") in {"auto", "v8", "v8.4"}
        or used in {"v7", "v8", "v8.4"}
        or (used == "manual_review" and isinstance(entry.get("cascade_v7_result"), dict))
    )


def _load_entry_image(
    entry, source: Path, *, output_dir: Path | None = None
) -> np.ndarray | None:
    """Load a verified normalized V7 image or the unchanged legacy image."""
    if not _entry_uses_normalized_snapshot(entry):
        return load_image(str(source))
    try:
        archived = (
            _load_archived_v7_image(entry, output_dir)
            if output_dir is not None
            else None
        )
        if archived is not None:
            return archived
        data = source.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        expected = entry.get("source_sha256")
        if expected and digest != expected:
            raise ValueError("source snapshot changed since detection")
        from photocut.algorithms.v7.input import decode_full_bgr_bytes

        orientation_value = str(entry.get("normalized_orientation", "exif_1"))
        if not orientation_value.startswith("exif_"):
            raise ValueError("invalid stored EXIF orientation")
        orientation = int(orientation_value.removeprefix("exif_"))
        expected_size = tuple(
            entry.get("normalized_size") or entry.get("original_size") or ()
        )
        return decode_full_bgr_bytes(
            data,
            exif_orientation=orientation,
            expected_size=expected_size,
        )
    except Exception as exc:
        print(f"无法加载已验证的 V7 快照 {source}: {exc}")
        return None


class PhotoCutConfirmationBackend:
    """O(1)-image adapter from PhotoCut storage to a confirmation session."""

    def __init__(
        self,
        *,
        corners_info,
        entries,
        input_dir,
        output_dir,
        json_path,
        annotation_store,
        project_root,
        image_loader: Callable | None = None,
        evidence_loader: Callable | None = None,
        commit_entry: Callable | None = None,
        save_view: Callable | None = None,
        clock: Callable[[], float] | None = None,
    ):
        if save_view is None:
            save_view = save_corners_info
        self.corners_info = corners_info
        self.entries = entries
        self.input_dir = Path(input_dir).resolve()
        self.output_dir = Path(output_dir)
        self.json_path = Path(json_path)
        self.annotation_store = annotation_store
        self.project_root = Path(project_root)
        if image_loader is None:
            resolved_image_loader = _load_entry_image
            image_loader = lambda entry, source, output: resolved_image_loader(
                entry, source, output_dir=output
            )
        self.image_loader = image_loader
        self.evidence_loader = (
            _load_finalized_detection_evidence
            if evidence_loader is None
            else evidence_loader
        )
        self.commit_entry = commit_confirmation if commit_entry is None else commit_entry
        self.save_view = save_view
        self.clock = time.perf_counter if clock is None else clock
        self._token_secret = secrets.token_bytes(32)

    def count(self) -> int:
        return len(self.entries)

    def _entry(self, index: int) -> dict:
        if type(index) is not int or not 0 <= index < len(self.entries):
            raise IndexError("confirmation entry index is out of range")
        return self.entries[index]

    def _image_token(self, index: int) -> str:
        self._entry(index)
        return hmac.new(
            self._token_secret,
            f"confirmation-entry:{index}".encode("ascii"),
            hashlib.sha256,
        ).hexdigest()

    def _source(self, entry: dict) -> Path:
        filename = entry.get("filename")
        if not isinstance(filename, str) or not filename:
            raise ValueError("source image filename is invalid")
        relative = Path(filename)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("source image path is unsafe")
        source = self.input_dir / relative
        try:
            info = source.lstat()
        except OSError as exc:
            raise ValueError(f"source image is missing: {filename}") from exc
        if source.is_symlink() or not stat.S_ISREG(info.st_mode):
            raise ValueError(f"source image is not a regular file: {filename}")
        try:
            source.resolve(strict=True).relative_to(self.input_dir)
        except ValueError:
            raise ValueError("source image path escapes input directory") from None
        return source

    def _load_image(self, entry: dict) -> np.ndarray:
        source = self._source(entry)
        image = self.image_loader(entry, source, self.output_dir)
        if not isinstance(image, np.ndarray) or image.ndim < 2 or image.size == 0:
            raise ValueError(f"could not load image: {entry['filename']}")
        return image

    def _previous_annotation(self, entry: dict) -> dict | None:
        annotation_id = entry.get("annotation_id")
        if not annotation_id:
            return None
        previous = next(
            (
                event
                for event in self.annotation_store.events()
                if event.get("annotation_id") == annotation_id
            ),
            None,
        )
        if previous is None or previous.get("image_id") != entry.get("image_id"):
            raise ValueError("revision target has no matching formal annotation")
        return previous

    def load(self, index: int) -> LoadedConfirmationItem:
        entry = self._entry(index)
        image = self._load_image(entry)
        try:
            evidence = self.evidence_loader(
                entry, self.output_dir, self.project_root
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            print(
                f"无法核验算法身份 {entry['filename']}: {exc}；将显示 unknown"
            )
            evidence = None
        previous = self._previous_annotation(entry)
        editor = ConfirmationEntryController(
            entry, image_size=(image.shape[1], image.shape[0])
        )
        return LoadedConfirmationItem(
            entry=entry,
            image=image,
            image_token=self._image_token(index),
            display_path=str(self._source(entry)),
            editor=editor,
            finalized_evidence=evidence,
            previous_annotation=previous,
            started_at=self.clock(),
            is_revision=bool(entry.get("annotation_id")),
        )

    def load_preview(self, index: int) -> tuple[np.ndarray, str]:
        image = self._load_image(self._entry(index))
        return image, self._image_token(index)

    def checkpoint(self, item: LoadedConfirmationItem) -> None:
        if item.is_revision:
            return
        entry = item.entry
        state = item.editor.state
        model = item.editor.model
        selection = getattr(model, "selection", None)
        primary_selection = entry.get("confirmation_primary_candidate_id")
        is_primary = model is None or selection in {
            "top1",
            "primary",
            primary_selection,
        }
        if not state.adjusted_corner_indices and is_primary:
            reset_confirmation_boundary(entry)
            entry["draft_adjusted_corner_indices"] = []
            entry["confirmed"] = False
        else:
            preview_width, preview_height = entry.get("preview_size", [0, 0])
            if preview_width > 0 and preview_height > 0:
                preview_corners = legacy_preview_corners(
                    state.work_corners,
                    state.image_size,
                    (preview_width, preview_height),
                )
            else:
                preview_corners = [point[:] for point in state.work_corners]
            save_manual_boundary(entry, state.work_corners, preview_corners)
            entry["manually_adjusted"] = bool(state.adjusted_corner_indices)
            entry["draft_adjusted_corner_indices"] = sorted(
                state.adjusted_corner_indices
            )
            entry["confirmed"] = False
        self.save_view(self.json_path, self.corners_info)

    def commit(
        self, item: LoadedConfirmationItem, duration_ms: int, gui_version: str
    ) -> dict:
        operation, _ = item.editor.confirm()
        event = self.commit_entry(
            item.entry,
            item.editor.state,
            self.annotation_store,
            duration_ms,
            finalized_evidence=item.finalized_evidence,
            gui_version=gui_version,
            previous_annotation=item.previous_annotation,
        )
        if operation is not None:
            item.entry["v7_operation"] = operation
            item.entry["v7_confirmed_corners"] = [
                point[:] for point in item.editor.state.work_corners
            ]
        try:
            self.save_view(self.json_path, self.corners_info)
        except Exception as exc:
            print(
                f"警告：正式标注已提交，但可变视图刷新失败 "
                f"{item.entry['filename']}: {exc}；重新打开确认模式时将自动 reconcile。"
            )
        return event

    def skip(self, item: LoadedConfirmationItem, reason: str) -> None:
        if item.is_revision:
            return
        item.entry["confirmed"] = False
        item.entry["v7_operation"] = "skipped"
        item.entry["v7_skip_reason"] = reason
        self.save_view(self.json_path, self.corners_info)


def create_confirmation_session(args, gui_version):
    """Build one session using the existing safe discovery and recovery order."""
    # These two scanners intentionally remain in the CLI module per the Task 4
    # boundary. All migrated persistence/image helpers above are called locally.
    from photocut.cli import find_images, normalize_entry_relative_paths

    input_dir = args.input
    output_dir = args.output
    json_path = Path(output_dir) / CORNERS_INFO_FILE
    corners_info = load_corners_info(str(json_path))
    if not corners_info:
        print(f"未找到 {json_path}，请先运行检测命令")
        return None

    discovered = find_images(input_dir, output_dir)
    accepted_entries, rejected_entries = normalize_entry_relative_paths(
        corners_info, discovered
    )
    for entry, reason in rejected_entries:
        print(f"确认跳过 {entry.get('filename', '')}: {reason}")
    save_corners_info(str(json_path), corners_info)

    project_root = Path(__file__).resolve().parents[2]
    annotation_store = _annotation_store_from_reference(output_dir, project_root)
    accepted = [entry for entry, _ in accepted_entries]
    if any(entry.get("confirmation_origin_batch_id") for entry in accepted):
        annotation_store = _OriginAnnotationStore(annotation_store, accepted)
    reconcile_confirmation_events(accepted, annotation_store)
    save_corners_info(str(json_path), corners_info)
    entries = select_confirmation_entries(
        accepted,
        getattr(args, "revise_confirmed", ()),
        annotation_store=annotation_store,
    )
    if not entries:
        print("所有图片都已确认")
        return None

    backend = PhotoCutConfirmationBackend(
        corners_info=corners_info,
        entries=entries,
        input_dir=input_dir,
        output_dir=output_dir,
        json_path=json_path,
        annotation_store=annotation_store,
        project_root=project_root,
    )
    return ConfirmationSessionController(backend, gui_version=gui_version)


__all__ = [
    "PhotoCutConfirmationBackend",
    "commit_confirmation",
    "confirmation_persistence_error_message",
    "create_confirmation_session",
    "reset_confirmation_boundary",
    "save_confirmation_draft",
    "save_manual_boundary",
    "select_confirmation_entries",
    "_annotation_store_from_reference",
    "_dataset_root_from_reference",
    "_entry_uses_normalized_snapshot",
    "_gui_identity_entry",
    "_load_archived_v7_image",
    "_load_entry_image",
    "_load_finalized_detection_evidence",
]
