"""Public Python API for PhotoCut.

Core image dependencies are loaded lazily so algorithm-only imports stay light.
"""

from importlib import import_module

__version__ = "0.1.0"

_CORE_EXPORTS = {
    "apply_annotation_to_entry",
    "create_preview",
    "crop_image",
    "detect_and_save_corners",
    "detect_and_save_corners_auto",
    "detect_and_save_corners_v7",
    "detect_and_save_corners_v8_dormant",
    "load_corners_info",
    "load_image",
    "reconcile_confirmation_events",
    "save_corners_info",
    "update_corners_entry",
}

__all__ = ["__version__", *sorted(_CORE_EXPORTS)]


def __getattr__(name: str):
    if name not in _CORE_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(".core", __name__), name)
