"""Tests for the segmentation logic.

Covers candidate-cut derivation, target steering, the shift/truncate conflict
policies, structural invariants (contiguity, coverage, 1-based indexing), the
30-minute cap, and edge cases.
"""

from typing import List

import pytest

from vad_audio_input.segmentation import (
    SegmentationConfig,
    _candidate_cuts,
    _nearest,
    segment_spans,
)
from vad_audio_input.types import (Segment, SpeechSpan)

_MAX_SEGMENT_MS: float = 30.0 * 60.0 * 1000.0


def make_config(
    min_silence_ms: int = 500,
    initial_duration_ms: int = 10000,
    conflict: str = "shift",
) -> SegmentationConfig:
    """Build a SegmentationConfig with sensible defaults."""
    return SegmentationConfig(
        min_silence_ms=min_silence_ms,
        initial_duration_ms=initial_duration_ms,
        conflict=conflict,  # type: ignore[arg-type]
    )


def spans_from_pairs(pairs: List[tuple[float, float]]) -> List[SpeechSpan]:
    """Build SpeechSpan list from (start_ms, end_ms) tuples."""
    return [SpeechSpan(start_ms=s, end_ms=e) for s, e in pairs]


def assert_well_formed(segments: List[Segment], spans: List[SpeechSpan]) -> None:
    """Assert structural invariants common to every segmentation result."""
    assert segments, "expected at least one segment"
    # 1-based, contiguous indices.
    assert [seg.index for seg in segments] == list(range(1, len(segments) + 1))
    # Every segment is non-empty.
    assert all(seg.duration_ms > 0.0 for seg in segments)
    # Boundaries are contiguous: each end == next start.
    for cur, nxt in zip(segments, segments[1:]):
        assert cur.end_ms == nxt.start_ms
    # The union spans exactly [first speech start, last speech end].
    assert segments[0].start_ms == spans[0].start_ms
    assert segments[-1].end_ms == spans[-1].end_ms
    # No segment exceeds the 30-minute cap.
    assert all(seg.duration_ms <= _MAX_SEGMENT_MS for seg in segments)


# --------------------------------------------------------------------------- #
# _candidate_cuts
# --------------------------------------------------------------------------- #


def test_candidate_cuts_includes_only_wide_gaps() -> None:
    spans = spans_from_pairs([(0, 1000), (1200, 2000), (3000, 4000)])
    # gap 1: 200ms (< 500, excluded); gap 2: 1000ms (>= 500, included).
    cuts = _candidate_cuts(spans, min_silence_ms=500)
    assert cuts == [pytest.approx((2000 + 3000) / 2.0)]


def test_candidate_cuts_uses_gap_midpoint() -> None:
    spans = spans_from_pairs([(0, 1000), (2000, 3000)])
    cuts = _candidate_cuts(spans, min_silence_ms=500)
    assert cuts == [pytest.approx(1500.0)]


def test_candidate_cuts_boundary_is_inclusive() -> None:
    # A gap exactly equal to min_silence_ms must qualify.
    spans = spans_from_pairs([(0, 1000), (1500, 2000)])
    cuts = _candidate_cuts(spans, min_silence_ms=500)
    assert cuts == [pytest.approx(1250.0)]


def test_candidate_cuts_empty_when_no_wide_gaps() -> None:
    spans = spans_from_pairs([(0, 1000), (1100, 2000), (2100, 3000)])
    assert _candidate_cuts(spans, min_silence_ms=500) == []


# --------------------------------------------------------------------------- #
# _nearest
# --------------------------------------------------------------------------- #


def test_nearest_returns_closest_cut() -> None:
    assert _nearest([1000.0, 5000.0, 9000.0], 4000.0) == 5000.0


def test_nearest_none_when_empty() -> None:
    assert _nearest([], 1000.0) is None


# --------------------------------------------------------------------------- #
# segment_spans: edge cases
# --------------------------------------------------------------------------- #


def test_empty_spans_yield_no_segments() -> None:
    assert segment_spans([], make_config()) == []


def test_single_span_is_one_segment() -> None:
    spans = spans_from_pairs([(500, 4500)])
    segments = segment_spans(spans, make_config())
    assert len(segments) == 1
    assert segments[0].start_ms == 500
    assert segments[0].end_ms == 4500
    assert_well_formed(segments, spans)


def test_no_qualifying_cuts_yields_single_segment() -> None:
    # All gaps narrower than min_silence -> no cut points -> one segment.
    spans = spans_from_pairs([(0, 1000), (1100, 2000), (2100, 3000)])
    segments = segment_spans(spans, make_config(min_silence_ms=500))
    assert len(segments) == 1
    assert_well_formed(segments, spans)


# --------------------------------------------------------------------------- #
# segment_spans: target steering
# --------------------------------------------------------------------------- #


def test_cut_near_target_duration() -> None:
    # Speech 0..40s with silence gaps every ~10s; target 20s should cut near 20s.
    spans = spans_from_pairs(
        [(0, 9000), (10000, 19000), (20000, 29000), (30000, 40000)]
    )
    segments = segment_spans(spans, make_config(initial_duration_ms=20000))
    assert_well_formed(segments, spans)
    # First boundary should be the cut nearest 20000 (the 19000/20000 gap midpoint).
    assert segments[0].end_ms == pytest.approx(19500.0)


def test_large_target_produces_single_segment() -> None:
    # Target far beyond total speech -> never reaches a cut -> one segment.
    spans = spans_from_pairs([(0, 9000), (10000, 19000), (20000, 29000)])
    segments = segment_spans(spans, make_config(initial_duration_ms=600000))
    assert len(segments) == 1
    assert_well_formed(segments, spans)


# --------------------------------------------------------------------------- #
# segment_spans: conflict policies
# --------------------------------------------------------------------------- #


def _conflict_spans() -> List[SpeechSpan]:
    # Cuts available at midpoints: 8500 (8000/9000) and 13500 (13000/14000).
    return spans_from_pairs(
        [(0, 8000), (9000, 13000), (14000, 22000)]
    )


def test_shift_picks_nearest_cut_even_if_past_target() -> None:
    # target 10000 -> ideal 10000; nearest cut is 8500 (dist 1500) vs 13500 (dist 3500).
    segments = segment_spans(
        _conflict_spans(), make_config(initial_duration_ms=10000, conflict="shift")
    )
    assert segments[0].end_ms == pytest.approx(8500.0)


def test_truncate_never_exceeds_target() -> None:
    # target 10000 -> ideal 10000; truncate must pick the cut <= 10000, i.e. 8500.
    segments = segment_spans(
        _conflict_spans(), make_config(initial_duration_ms=10000, conflict="truncate")
    )
    assert segments[0].end_ms == pytest.approx(8500.0)
    assert segments[0].duration_ms <= 10000


def test_truncate_falls_back_to_nearest_when_no_cut_before_target() -> None:
    # Only cut is at 9500 (9000/10000 midpoint), past a tiny target of 2000.
    # truncate has no cut <= ideal, so it must fall back to the nearest cut.
    spans = spans_from_pairs([(0, 9000), (10000, 16000)])
    segments = segment_spans(
        spans, make_config(initial_duration_ms=2000, conflict="truncate")
    )
    assert segments[0].end_ms == pytest.approx(9500.0)
    assert_well_formed(segments, spans)


def test_shift_vs_truncate_differ_when_nearest_is_past_target() -> None:
    # target 12000 -> ideal 12000. nearest is 13500 (dist 1500) < 8500 (dist 3500).
    # shift -> 13500 (overshoots target); truncate -> 8500 (the cut <= 12000).
    shift = segment_spans(
        _conflict_spans(), make_config(initial_duration_ms=12000, conflict="shift")
    )
    truncate = segment_spans(
        _conflict_spans(), make_config(initial_duration_ms=12000, conflict="truncate")
    )
    assert shift[0].end_ms == pytest.approx(13500.0)
    assert truncate[0].end_ms == pytest.approx(8500.0)
    assert shift[0].end_ms != truncate[0].end_ms


# --------------------------------------------------------------------------- #
# segment_spans: 30-minute cap
# --------------------------------------------------------------------------- #


def test_tail_longer_than_cap_is_split() -> None:
    # One 70-minute span with no internal cuts must split into >= 3 capped segments.
    seventy_min = 70 * 60 * 1000
    spans = spans_from_pairs([(0, seventy_min)])
    segments = segment_spans(spans, make_config(initial_duration_ms=600000))
    assert len(segments) >= 3
    assert_well_formed(segments, spans)
    # All but the last are exactly the cap length.
    for seg in segments[:-1]:
        assert seg.duration_ms == pytest.approx(_MAX_SEGMENT_MS)


# --------------------------------------------------------------------------- #
# Property: union coverage is exact and gapless for many configs
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("target_ms", [3000, 7000, 15000, 30000])
@pytest.mark.parametrize("conflict", ["shift", "truncate"])
def test_coverage_invariant_across_configs(target_ms: int, conflict: str) -> None:
    spans = spans_from_pairs(
        [
            (0, 5000),
            (6000, 11000),
            (12000, 18000),
            (19000, 25000),
            (26000, 33000),
        ]
    )
    config = make_config(initial_duration_ms=target_ms, conflict=conflict)
    segments = segment_spans(spans, config)
    assert_well_formed(segments, spans)
