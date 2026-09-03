"""V8 scanner-white segmentation research components.

Runtime model imports stay out of this package initializer so V7.1 remains
usable on machines without the optional V8 training dependencies.
"""

LEGACY_V8_ALGORITHM_VERSION = "8.0"
PRACTICAL_V8_ALGORITHM_VERSION = "8.1"
V8_ALGORITHM_VERSION = "8.2"
PRODUCTION_PROMOTION = False

__all__ = [
    "LEGACY_V8_ALGORITHM_VERSION",
    "PRACTICAL_V8_ALGORITHM_VERSION",
    "PRODUCTION_PROMOTION",
    "V8_ALGORITHM_VERSION",
]
