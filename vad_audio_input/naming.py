"""Segment filename construction.

Pattern: ``{stem}_segment{NNNNN}_{start}_{end}`` where times are seconds with three
decimals, dynamic integer width, and ``'.'`` replaced by ``'p'``.

Example: segment #1 of ``audio-test`` spanning 0.000s-1.150s ->
``audio-test_segment00001_0p000_1p150``.
"""

from typing import Final

_SEQ_WIDTH: Final[int] = 5
_DECIMALS: Final[int] = 3


def format_time(seconds: float) -> str:
    """Format a time as ``<int>p<frac>`` with three decimals and dynamic int width.

    The decimal point is replaced by ``'p'`` and the integer part is not padded,
    e.g. ``1.15 -> "1p150"`` and ``125.4 -> "125p400"``.

    Args:
        seconds: Time in seconds.

    Returns:
        The formatted time string.
    """
    text: str = f"{seconds:.{_DECIMALS}f}"
    return text.replace(".", "p")


def segment_basename(stem: str, index: int, start_ms: float, end_ms: float) -> str:
    """Build the segment basename (without extension).

    The pattern is ``{stem}_segment{NNNNN}_{start}_{end}`` where ``NNNNN`` is the
    5-digit zero-padded sequence number and the times are formatted by
    :func:`format_time`.

    Args:
        stem: Input file stem (filename without extension).
        index: 1-based segment sequence number.
        start_ms: Segment start, in milliseconds.
        end_ms: Segment end, in milliseconds.

    Returns:
        The segment basename, e.g. ``"audio-test_segment00001_0p000_1p150"``.
    """
    seq: str = str(index).zfill(_SEQ_WIDTH)
    start: str = format_time(start_ms / 1000.0)
    end: str = format_time(end_ms / 1000.0)
    return f"{stem}_segment{seq}_{start}_{end}"
