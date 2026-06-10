"""Silero VAD backend (the default)."""

import logging
from typing import (Any, Final, List, Optional)

from ..types import (SpeechSpan, Waveform)
from .base import (VAD_SAMPLE_RATE, register_backend)

_LOGGER: Final[logging.Logger] = logging.getLogger("vad_audio_input.vad.silero")


class SileroVAD:
    """Neural VAD backed by the ``silero-vad`` package.

    The model is loaded lazily on first ``detect`` call and reused thereafter.
    """

    name: str = "silero"

    def __init__(self) -> None:
        """Create the backend without loading the model (loaded on first use)."""
        self._model: Optional[Any] = None
        self._get_speech_timestamps: Optional[Any] = None

    def _ensure_loaded(self) -> None:
        """Load the Silero model and helper on first call; a no-op thereafter."""
        if self._model is not None:
            return
        # Imported lazily so the package can be inspected without torch present.
        from silero_vad import (load_silero_vad, get_speech_timestamps)

        _LOGGER.debug("loading Silero VAD model")
        self._model = load_silero_vad()
        self._get_speech_timestamps = get_speech_timestamps

    def detect(self, samples: Waveform, sample_rate: int) -> List[SpeechSpan]:
        """Detect speech intervals using Silero VAD.

        Args:
            samples: Mono float32 waveform sampled at ``VAD_SAMPLE_RATE``.
            sample_rate: Sample rate of ``samples``; must equal ``VAD_SAMPLE_RATE``.

        Returns:
            The detected speech spans, in milliseconds.

        Raises:
            ValueError: If ``sample_rate`` is not ``VAD_SAMPLE_RATE``.
        """
        if sample_rate != VAD_SAMPLE_RATE:
            raise ValueError(
                f"Silero expects {VAD_SAMPLE_RATE} Hz, received {sample_rate} Hz"
            )
        self._ensure_loaded()
        assert self._model is not None and self._get_speech_timestamps is not None

        import torch

        tensor = torch.from_numpy(samples)
        # Return timestamps in seconds so we can convert to ms directly.
        raw: List[dict] = self._get_speech_timestamps(
            tensor,
            self._model,
            sampling_rate=sample_rate,
            return_seconds=True,
        )
        spans: List[SpeechSpan] = [
            SpeechSpan(
                start_ms=float(item["start"]) * 1000.0,
                end_ms=float(item["end"]) * 1000.0,
            )
            for item in raw
        ]
        return spans


register_backend("silero", SileroVAD)
