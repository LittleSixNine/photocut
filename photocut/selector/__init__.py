"""Public identity for the default PhotoCut detector orchestrator."""

SELECTOR_NAME = "PhotoCut Selector v4"
SELECTOR_SHORT_NAME = "Selector v4"
SELECTOR_VERSION = "4.0"

# Existing result files use this stable machine-readable identity.  It remains
# unchanged so old batches can still be resumed and confirmed.
SELECTOR_RECORD_ID = "auto-v4"

__all__ = [
    "SELECTOR_NAME",
    "SELECTOR_SHORT_NAME",
    "SELECTOR_VERSION",
    "SELECTOR_RECORD_ID",
]
