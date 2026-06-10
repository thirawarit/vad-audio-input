"""VAD backend protocol and registry."""

import logging
from typing import (Callable, Dict, Final, List, Protocol, runtime_checkable)

from ..types import (SpeechSpan, Waveform)

_LOGGER: Final[logging.Logger] = logging.getLogger("vad_audio_input.vad")

# Sample rate every backend receives audio at.
VAD_SAMPLE_RATE: Final[int] = 16000


@runtime_checkable
class VADBackend(Protocol):
    """A voice-activity-detection engine.

    Implementations receive a mono float32 waveform sampled at ``VAD_SAMPLE_RATE``
    and return the detected speech intervals in milliseconds.
    """

    name: str

    def detect(self, samples: Waveform, sample_rate: int) -> List[SpeechSpan]:
        """Detect speech intervals in a waveform.

        Args:
            samples: Mono float32 waveform sampled at ``VAD_SAMPLE_RATE``.
            sample_rate: Sample rate of ``samples``, in Hz.

        Returns:
            The detected speech spans, in milliseconds.
        """
        ...


_REGISTRY: Final[Dict[str, "Callable[[], VADBackend]"]] = {}


def register_backend(name: str, factory: "Callable[[], VADBackend]") -> None:
    """Register a backend factory under ``name``.

    Args:
        name: Backend identifier used by ``--vad-backend``.
        factory: Zero-argument callable that returns a backend instance.
    """
    _REGISTRY[name] = factory


def get_backend(name: str) -> VADBackend:
    """Instantiate the registered backend ``name``.

    Args:
        name: A registered backend identifier.

    Returns:
        A new backend instance.

    Raises:
        ValueError: If ``name`` is not registered.
    """
    if name not in _REGISTRY:
        available: str = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise ValueError(f"unknown VAD backend '{name}'; available: {available}")
    _LOGGER.debug("instantiating VAD backend '%s'", name)
    return _REGISTRY[name]()


def available_backends() -> List[str]:
    """Return the sorted list of registered backend names.

    Returns:
        Registered backend identifiers, sorted alphabetically.
    """
    return sorted(_REGISTRY)
