#!/usr/bin/env python3
"""Fail when the public checkout contains common private or generated data."""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_MODEL_FILES = {
    "photocut/models/v8_2/model.onnx",
    "photocut/models/v8_4/model.onnx",
}
ALLOWED_LARGE_FILES = ALLOWED_MODEL_FILES
MODEL_SUFFIXES = {".onnx", ".pt", ".pth", ".ckpt", ".safetensors", ".h5", ".hdf5"}
ALLOWED_IMAGE_FILES = {
    "assets/readme/example-01-after.jpg",
    "assets/readme/example-01-before.jpg",
    "assets/readme/example-02-after.jpg",
    "assets/readme/example-02-before.jpg",
}
FORBIDDEN_PARTS = {
    ".photocut",
    ".local",
    ".superpowers",
    "experiments",
    "input",
    "output",
    "outputs",
    "sample_photos",
}
TEXT_SUFFIXES = {
    ".css", ".html", ".ini", ".js", ".json", ".md", ".py", ".txt", ".yaml", ".yml"
}
PATTERNS = {
    "personal macOS path": re.compile("/" + r"Users/[^/\s]+/"),
    "personal Linux path": re.compile("/" + r"home/[^/\s]+/"),
    "private model marker": re.compile(
        "private-" + "research|local:" + "photocut", re.I
    ),
    "GitHub token": re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}"),
    "generic secret assignment": re.compile(
        r"(?i)(api[_-]?key|access[_-]?token|client[_-]?secret|password)\s*[:=]\s*[\"'][^\"']{8,}[\"']"
    ),
}


def tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return sorted(
        ROOT / raw.decode("utf-8")
        for raw in result.stdout.split(b"\0")
        if raw
    )


def main() -> int:
    errors: list[str] = []
    for path in tracked_files():
        relative = path.relative_to(ROOT).as_posix()
        if FORBIDDEN_PARTS.intersection(path.relative_to(ROOT).parts):
            errors.append(f"private/generated path is tracked: {relative}")
            continue
        if path.suffix.lower() in MODEL_SUFFIXES and relative not in ALLOWED_MODEL_FILES:
            errors.append(f"unapproved model or training checkpoint: {relative}")
        size = path.stat().st_size
        if size > 5 * 1024 * 1024 and relative not in ALLOWED_LARGE_FILES:
            errors.append(f"unexpected file larger than 5 MiB: {relative}")
        if (
            path.suffix.lower()
            in {".jpg", ".jpeg", ".png", ".heic", ".tif", ".tiff", ".dng"}
            and relative not in ALLOWED_IMAGE_FILES
        ):
            errors.append(f"image file is tracked: {relative}")
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeError:
            errors.append(f"text file is not UTF-8: {relative}")
            continue
        for label, pattern in PATTERNS.items():
            if pattern.search(text):
                errors.append(f"{label} in {relative}")
    if errors:
        print("Public-tree check failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("Public-tree check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
