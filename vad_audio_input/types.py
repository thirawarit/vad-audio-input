"""Shared dataclasses and type aliases."""

from dataclasses import dataclass
from typing import (Final, Literal)

import numpy as np

# A mono float32 waveform in the range [-1.0, 1.0].
Waveform = np.ndarray

ConflictPolicy = Literal["shift", "truncate"]

SUPPORTED_INPUT_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {".wav", ".mp3", ".m4a", ".flac"}
)


@dataclass(frozen=True)
class SpeechSpan:
    """A detected speech interval, relative to the source audio.

    Attributes:
        start_ms: Interval start, in milliseconds from the start of the source.
        end_ms: Interval end, in milliseconds from the start of the source.
    """

    start_ms: float
    end_ms: float


@dataclass(frozen=True)
class Segment:
    """A planned output segment derived from segmentation, in source-audio time.

    Attributes:
        index: 1-based sequence number of the segment within its source file.
        start_ms: Segment start, in milliseconds from the start of the source.
        end_ms: Segment end, in milliseconds from the start of the source.
    """

    index: int
    start_ms: float
    end_ms: float

    @property
    def duration_ms(self) -> float:
        """Return the segment length in milliseconds (``end_ms - start_ms``)."""
        return self.end_ms - self.start_ms
