"""Pluggable VAD backends."""

from .base import (VADBackend, get_backend, register_backend)
from .silero import SileroVAD

__all__ = ["VADBackend", "get_backend", "register_backend", "SileroVAD"]
