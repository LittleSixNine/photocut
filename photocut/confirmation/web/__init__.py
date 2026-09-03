"""Loopback-only web transport for the PhotoCut confirmation session."""

from .server import LocalConfirmationServer, WriterLease

__all__ = ["LocalConfirmationServer", "WriterLease"]
