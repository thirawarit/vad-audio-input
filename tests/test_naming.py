"""Tests for the segment filename pattern."""

from vad_audio_input.naming import (format_time, segment_basename)


def test_format_time_replaces_dot_and_keeps_three_decimals() -> None:
    assert format_time(1.15) == "1p150"
    assert format_time(0.0) == "0p000"


def test_format_time_dynamic_integer_width() -> None:
    # Integer part is not left-padded; width grows with the value.
    assert format_time(125.4) == "125p400"


def test_format_time_rounds_to_three_decimals() -> None:
    assert format_time(1.23456) == "1p235"


def test_segment_basename_matches_spec_example() -> None:
    # IDEA.md example: segment #1 of 'audio-test' spanning 0.000s-1.150s.
    name = segment_basename("audio-test", 1, start_ms=0.0, end_ms=1150.0)
    assert name == "audio-test_seg00001_0p000_1p150"


def test_segment_basename_zero_pads_sequence_to_five_digits() -> None:
    name = segment_basename("clip", 42, start_ms=0.0, end_ms=1000.0)
    assert name.startswith("clip_seg00042_")


def test_segment_basename_large_index_not_truncated() -> None:
    name = segment_basename("clip", 123456, start_ms=0.0, end_ms=1000.0)
    assert "seg123456_" in name
