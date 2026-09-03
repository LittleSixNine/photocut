"""Independent identity and display helpers for the confirmation GUI."""
from __future__ import annotations

import re
from collections.abc import Mapping


OPENCV_GUI_VERSION = "1.0"
WEB_GUI_VERSION = "2.0"
GUI_VERSION = WEB_GUI_VERSION
_VERSION_RE = re.compile(r"[0-9]+(?:\.[0-9]+){1,2}")


def gui_version_for_frontend(frontend: str) -> str:
    versions = {"opencv": OPENCV_GUI_VERSION, "web": WEB_GUI_VERSION}
    try:
        return versions[frontend]
    except KeyError:
        raise ValueError("confirmation frontend must be opencv or web") from None


def validate_version(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= 32
        or value != value.strip()
        or _VERSION_RE.fullmatch(value) is None
    ):
        raise ValueError(f"{field} must be a 1-32 character dotted numeric version")
    return value


def _display_version(value: object) -> str:
    try:
        return validate_version(value, "algorithm_version")
    except ValueError:
        return "unknown"


def format_gui_title(
    algorithm_version: object,
    gui_version: str = GUI_VERSION,
) -> str:
    return f"PhotoCut GUI {gui_version} · Algorithm {_display_version(algorithm_version)}"


def gui_identity_lines(
    entry: Mapping,
    previous_annotation: Mapping | None = None,
    gui_version: str = GUI_VERSION,
) -> tuple[str, ...]:
    lines = [
        f"GUI: {gui_version}",
        f"Algorithm: {_display_version(entry.get('algorithm_version'))}",
    ]
    requested = entry.get("detector_requested")
    used = entry.get("detector_used", entry.get("detector"))
    if isinstance(requested, str) and requested:
        lines.append(f"Requested: {requested}")
    if isinstance(used, str) and used:
        lines.append(f"Used: {used}")
    if previous_annotation is not None:
        previous = previous_annotation.get("gui_version")
        lines.append(
            f"Previous GUI: {previous}"
            if isinstance(previous, str) and previous
            else "Previous GUI: legacy / unknown"
        )
    return tuple(lines)


def set_native_gui_title(
    cv2_module: object,
    window_key: str,
    algorithm_version: object,
) -> str:
    title = format_gui_title(algorithm_version, gui_version=OPENCV_GUI_VERSION)
    try:
        setter = getattr(cv2_module, "setWindowTitle")
        setter(window_key, title)
    except Exception:
        pass
    return title
