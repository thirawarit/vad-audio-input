"""Turn detected speech spans into output segments.

Algorithm overview:

1. From the speech spans, derive the *silence gaps* between consecutive spans.
2. Any silence gap whose width >= ``min_silence_ms`` is a candidate cut point; the cut
   is placed at the gap midpoint.
3. Speech is grouped into segments by walking the spans and closing a segment at a
   candidate cut whenever the running segment has reached (or is nearest to) the
   ``initial_duration_ms`` target length.
4. ``conflict`` resolves what happens when the ideal target boundary does not coincide
   with a candidate (``min_silence``) cut:
     - ``shift``    : move the boundary to the nearest candidate cut (default).
     - ``truncate`` : hard-cut at the candidate cut, dropping the overflow.

All times are milliseconds in source-audio time.
"""

import logging
from dataclasses import dataclass
from typing import (Final, List, Optional)

from .types import (ConflictPolicy, Segment, SpeechSpan)

_LOGGER: Final[logging.Logger] = logging.getLogger("vad_audio_input.segmentation")

# A segment must be > 0ms and <= 30 minutes (project definition of "segment audio").
_MAX_SEGMENT_MS: Final[float] = 30.0 * 60.0 * 1000.0


@dataclass(frozen=True)
class SegmentationConfig:
    """Tunable parameters controlling how spans become segments.

    Attributes:
        min_silence_ms: Minimum inter-span silence width (ms) that yields a cut.
        initial_duration_ms: Target/ideal segment length (ms) to steer cuts toward.
        conflict: Boundary policy when the target does not align with a cut
            (``"shift"`` or ``"truncate"``).
    """

    min_silence_ms: int
    initial_duration_ms: int
    conflict: ConflictPolicy


def _candidate_cuts(spans: List[SpeechSpan], min_silence_ms: int) -> List[float]:
    """Return midpoints of inter-span silence gaps at least ``min_silence_ms`` wide.

    Args:
        spans: Speech spans in ascending time order.
        min_silence_ms: Minimum gap width to qualify as a cut point.

    Returns:
        Candidate cut times (gap midpoints), in milliseconds.
    """
    cuts: List[float] = []
    for prev, nxt in zip(spans, spans[1:]):
        gap: float = nxt.start_ms - prev.end_ms
        if gap >= min_silence_ms:
            cuts.append((prev.end_ms + nxt.start_ms) / 2.0)
    return cuts


def _nearest(cuts: List[float], target_ms: float) -> Optional[float]:
    """Return the cut nearest to ``target_ms``, or ``None`` if there are no cuts.

    Args:
        cuts: Candidate cut times, in milliseconds.
        target_ms: The time to find the closest cut to.

    Returns:
        The nearest cut, or ``None`` when ``cuts`` is empty.
    """
    if not cuts:
        return None
    return min(cuts, key=lambda c: abs(c - target_ms))


def segment_spans(
    spans: List[SpeechSpan],
    config: SegmentationConfig,
) -> List[Segment]:
    """Group speech spans into output segments per ``config``.

    Args:
        spans: Detected speech spans in ascending time order.
        config: Segmentation parameters.

    Returns:
        Contiguous, 1-based segments covering the full speech range, or an empty
        list when ``spans`` is empty.
    """
    if not spans:
        return []

    speech_start: float = spans[0].start_ms
    speech_end: float = spans[-1].end_ms
    cuts: List[float] = _candidate_cuts(spans, config.min_silence_ms)
    _LOGGER.debug("candidate cuts: %d", len(cuts))

    boundaries: List[float] = _plan_boundaries(speech_start, speech_end, cuts, config)
    return _boundaries_to_segments(boundaries)


def _plan_boundaries(
    speech_start: float,
    speech_end: float,
    cuts: List[float],
    config: SegmentationConfig,
) -> List[float]:
    """Produce the ordered boundary list ``[start, ..., end]`` enclosing all segments.

    Walks candidate cuts, closing a segment near the ``initial_duration_ms`` target
    per the conflict policy, and enforces the 30-minute cap on every segment.

    Args:
        speech_start: First speech start time, in milliseconds.
        speech_end: Last speech end time, in milliseconds.
        cuts: Candidate cut times, in milliseconds.
        config: Segmentation parameters.

    Returns:
        Ascending boundary times bracketing every segment.
    """
    boundaries: List[float] = [speech_start]
    # Cuts strictly inside the current open segment, ascending.
    remaining: List[float] = sorted(c for c in cuts if speech_start < c < speech_end)

    seg_start: float = speech_start
    target: float = float(config.initial_duration_ms)

    while remaining:
        ideal: float = seg_start + target
        if ideal >= speech_end:
            break

        # The candidate cut we would align the boundary to.
        nearest: Optional[float] = _nearest(remaining, ideal)
        if nearest is None:
            break

        if config.conflict == "shift":
            boundary: float = nearest
        else:  # truncate: never extend past the target; cut at the candidate <= ideal
            at_or_before: List[float] = [c for c in remaining if c <= ideal]
            boundary = at_or_before[-1] if at_or_before else nearest

        # Enforce the 30-minute cap regardless of cut availability.
        if boundary - seg_start > _MAX_SEGMENT_MS:
            boundary = seg_start + _MAX_SEGMENT_MS

        boundaries.append(boundary)
        seg_start = boundary
        remaining = [c for c in remaining if c > boundary]

    # Close out any tail, splitting if it exceeds the 30-minute cap.
    while speech_end - seg_start > _MAX_SEGMENT_MS:
        seg_start += _MAX_SEGMENT_MS
        boundaries.append(seg_start)
    boundaries.append(speech_end)
    return boundaries


def _boundaries_to_segments(boundaries: List[float]) -> List[Segment]:
    """Convert an ordered boundary list into 1-based, non-empty segments.

    Args:
        boundaries: Ascending boundary times, in milliseconds.

    Returns:
        Segments between consecutive boundaries, skipping any zero-length pair.
    """
    segments: List[Segment] = []
    index: int = 1
    for start, end in zip(boundaries, boundaries[1:]):
        if end - start <= 0.0:
            continue
        segments.append(Segment(index=index, start_ms=start, end_ms=end))
        index += 1
    return segments
