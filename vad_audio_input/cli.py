"""Command-line interface."""

import argparse
import logging
from datetime import (datetime, timedelta, timezone)
from pathlib import Path
from typing import (Final, List, Optional, Sequence, get_args)

from .i18n import (Language, SUPPORTED_LANGUAGES)
from .logging_setup import setup_logging
from .pipeline import (OutputConfig, RunResult, run)
from .segmentation import SegmentationConfig
from .types import ConflictPolicy
from .vad.base import available_backends

_OUTPUT_FORMATS: Final[tuple[str, ...]] = ("wav", "flac")
_CONFLICT_CHOICES: Final[tuple[str, ...]] = get_args(ConflictPolicy)


def _positive_int(text: str) -> int:
    """Parse a strictly positive integer for argparse.

    Args:
        text: The raw command-line value.

    Returns:
        The parsed integer.

    Raises:
        argparse.ArgumentTypeError: If the value is not ``> 0``.
    """
    value: int = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be > 0, got {value}")
    return value


def _non_negative_int(text: str) -> int:
    """Parse a non-negative integer for argparse.

    Args:
        text: The raw command-line value.

    Returns:
        The parsed integer.

    Raises:
        argparse.ArgumentTypeError: If the value is ``< 0``.
    """
    value: int = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {value}")
    return value


def get_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser.

    Returns:
        The configured :class:`argparse.ArgumentParser`.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        prog="vad-audio-input",
        description="Segment long-form speech audio at silence boundaries using VAD.",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        metavar="INPUT",
        help="Input audio file(s) or directory(ies) (WAV/MP3/M4A/FLAC).",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        type=Path,
        help="Root output directory.",
    )
    parser.add_argument(
        "--vad-backend",
        default="silero",
        choices=available_backends(),
        help="VAD engine (default: silero).",
    )
    parser.add_argument(
        "--min-silence",
        required=True,
        type=_positive_int,
        metavar="MS",
        help="Minimum silence duration (ms) that triggers a cut.",
    )
    parser.add_argument(
        "--initial-duration",
        required=True,
        type=_positive_int,
        metavar="MS",
        help="Target/ideal segment length (ms).",
    )
    parser.add_argument(
        "--conflict",
        default="shift",
        choices=_CONFLICT_CHOICES,
        help="Conflict policy when target and silence cut disagree (default: shift).",
    )
    parser.add_argument(
        "--pre-buffer",
        default=0,
        type=_non_negative_int,
        metavar="MS",
        help="Digital silence (ms) prepended to each segment (default: 0).",
    )
    parser.add_argument(
        "--post-buffer",
        default=0,
        type=_non_negative_int,
        metavar="MS",
        help="Digital silence (ms) appended to each segment (default: 0).",
    )
    parser.add_argument(
        "--format",
        dest="out_format",
        default="wav",
        choices=_OUTPUT_FORMATS,
        help="Output segment format (default: wav).",
    )
    parser.add_argument(
        "--sample-rate",
        default=16000,
        type=_positive_int,
        metavar="HZ",
        help="Output sample rate (default: 16000).",
    )
    parser.add_argument(
        "--channels",
        default=1,
        type=_positive_int,
        help="Output channel count (default: 1).",
    )
    parser.add_argument(
        "--lang",
        default="en",
        choices=list(SUPPORTED_LANGUAGES),
        help="Language for log/UI messages (default: en).",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Log file path (default: logs/run_<timestamp>.log).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser


def _default_log_file() -> Path:
    """Build a timestamped default log file path under ``logs/``.

    Returns:
        A path like ``logs/run_<YYYYMMDD_HHMMSS>.log``, timestamped in Bangkok time.
    """
    bangkok: timezone = timezone(timedelta(hours=7))
    stamp: str = datetime.now(tz=bangkok).strftime("%Y%m%d_%H%M%S")
    return Path("logs") / f"run_{stamp}.log"


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the command-line interface.

    Parses arguments, configures logging, and processes the inputs.

    Args:
        argv: Argument vector to parse; defaults to ``sys.argv`` when ``None``.

    Returns:
        The process exit code (``0`` on full success, ``1`` if any input failed).
    """
    parser: argparse.ArgumentParser = get_parser()
    args: argparse.Namespace = parser.parse_args(argv)

    log_file: Path = args.log_file if args.log_file is not None else _default_log_file()
    level: int = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=level, log_file=log_file)

    lang: Language = args.lang
    seg_config: SegmentationConfig = SegmentationConfig(
        min_silence_ms=args.min_silence,
        initial_duration_ms=args.initial_duration,
        conflict=args.conflict,
    )
    out_config: OutputConfig = OutputConfig(
        out_format=args.out_format,
        sample_rate=args.sample_rate,
        channels=args.channels,
        pre_buffer_ms=args.pre_buffer,
        post_buffer_ms=args.post_buffer,
    )

    inputs: List[Path] = list(args.inputs)
    result: RunResult = run(
        inputs=inputs,
        output_root=args.output_dir,
        vad_backend=args.vad_backend,
        seg_config=seg_config,
        out_config=out_config,
        lang=lang,
    )
    return result.exit_code
