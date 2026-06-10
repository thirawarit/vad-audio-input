"""End-to-end processing: per input file, VAD-segment and write outputs + manifest."""

import json
import logging
from dataclasses import (dataclass, field)
from pathlib import Path
from typing import (Final, List)

from . import audio_io
from .i18n import (Language, translate)
from .naming import segment_basename
from .segmentation import (SegmentationConfig, segment_spans)
from .types import (Segment, SpeechSpan, SUPPORTED_INPUT_EXTENSIONS)
from .vad import (VADBackend, get_backend)
from .vad.base import VAD_SAMPLE_RATE

_LOGGER: Final[logging.Logger] = logging.getLogger("vad_audio_input.pipeline")


@dataclass(frozen=True)
class OutputConfig:
    """Output encoding and buffer settings.

    Attributes:
        out_format: Output container/codec (e.g. ``"wav"``, ``"flac"``).
        sample_rate: Output sample rate, in Hz.
        channels: Output channel count.
        pre_buffer_ms: Milliseconds of leading digital silence per segment.
        post_buffer_ms: Milliseconds of trailing digital silence per segment.
    """

    out_format: str
    sample_rate: int
    channels: int
    pre_buffer_ms: int
    post_buffer_ms: int


@dataclass
class RunResult:
    """Aggregate outcome of a run across all inputs.

    Attributes:
        succeeded: Input paths processed without error.
        failed: Input paths skipped due to per-file errors.
    """

    succeeded: List[Path] = field(default_factory=list)
    failed: List[Path] = field(default_factory=list)

    @property
    def exit_code(self) -> int:
        """Return ``1`` if any input failed, otherwise ``0``."""
        return 1 if self.failed else 0


def discover_inputs(paths: List[Path]) -> List[Path]:
    """Expand files and directories into supported audio files.

    Directories are scanned recursively. Only files with a supported extension
    are included.

    Args:
        paths: Input file and/or directory paths.

    Returns:
        Matching audio file paths, sorted within each scanned directory.
    """
    found: List[Path] = []
    for path in paths:
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file() and child.suffix.lower() in SUPPORTED_INPUT_EXTENSIONS:
                    found.append(child)
        elif path.is_file() and path.suffix.lower() in SUPPORTED_INPUT_EXTENSIONS:
            found.append(path)
    return found


def _build_manifest(
    source: Path,
    source_duration_ms: float,
    seg_config: SegmentationConfig,
    out_config: OutputConfig,
    vad_backend: str,
    segments: List[Segment],
    filenames: List[str],
) -> dict:
    """Assemble the manifest dictionary for one input file.

    Args:
        source: The input file the manifest describes.
        source_duration_ms: Total source duration, in milliseconds.
        seg_config: Segmentation parameters used.
        out_config: Output/buffer settings used.
        vad_backend: Name of the VAD backend used.
        segments: The planned segments, in order.
        filenames: Output filenames aligned 1:1 with ``segments``.

    Returns:
        A JSON-serializable manifest dictionary.
    """
    return {
        "source": source.name,
        "source_duration_ms": source_duration_ms,
        "config": {
            "vad_backend": vad_backend,
            "min_silence_ms": seg_config.min_silence_ms,
            "initial_duration_ms": seg_config.initial_duration_ms,
            "conflict": seg_config.conflict,
            "pre_buffer_ms": out_config.pre_buffer_ms,
            "post_buffer_ms": out_config.post_buffer_ms,
            "format": out_config.out_format,
            "sample_rate": out_config.sample_rate,
            "channels": out_config.channels,
        },
        "segments": [
            {
                "index": seg.index,
                "filename": name,
                "start_ms": seg.start_ms,
                "end_ms": seg.end_ms,
                "duration_ms": seg.duration_ms,
                "pre_buffer_ms": out_config.pre_buffer_ms,
                "post_buffer_ms": out_config.post_buffer_ms,
            }
            for seg, name in zip(segments, filenames)
        ],
    }


def process_file(
    source: Path,
    output_root: Path,
    backend: VADBackend,
    seg_config: SegmentationConfig,
    out_config: OutputConfig,
    lang: Language,
) -> None:
    """Process a single input file: detect speech, segment, and write outputs.

    Loads the audio, runs VAD on a 16 kHz mono copy, segments the speech, then
    writes each padded segment plus a ``manifest.json`` into a per-stem subfolder
    of ``output_root``.

    Args:
        source: Input audio file.
        output_root: Root directory; outputs go to ``output_root/<stem>/``.
        backend: VAD backend used for speech detection.
        seg_config: Segmentation parameters.
        out_config: Output encoding and buffer settings.
        lang: Language for log messages.

    Raises:
        RuntimeError: If the input cannot be decoded.
    """
    _LOGGER.info(translate("processing_file", lang, path=str(source)))

    loaded: audio_io.LoadedAudio = audio_io.load_audio(source)
    _LOGGER.info(
        translate(
            "loaded_audio",
            lang,
            path=str(source),
            duration_ms=loaded.duration_ms,
            sample_rate=loaded.sample_rate,
            channels=loaded.channels,
        )
    )

    # VAD runs on a 16 kHz mono copy regardless of the source/output rate.
    vad_samples = audio_io.resample(loaded.samples, loaded.sample_rate, VAD_SAMPLE_RATE)
    spans: List[SpeechSpan] = backend.detect(vad_samples, VAD_SAMPLE_RATE)
    _LOGGER.info(translate("detected_spans", lang, count=len(spans)))

    segments: List[Segment] = segment_spans(spans, seg_config)

    stem: str = source.stem
    out_dir: Path = output_root / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    ext: str = out_config.out_format.lower()

    filenames: List[str] = []
    for seg in segments:
        basename: str = segment_basename(stem, seg.index, seg.start_ms, seg.end_ms)
        filename: str = f"{basename}.{ext}"
        filenames.append(filename)

        raw: audio_io.Waveform = audio_io.slice_ms(
            loaded.samples, loaded.sample_rate, seg.start_ms, seg.end_ms
        )
        padded: audio_io.Waveform = audio_io.pad_with_silence(
            raw, loaded.sample_rate, out_config.pre_buffer_ms, out_config.post_buffer_ms
        )
        audio_io.write_segment(
            padded,
            loaded.sample_rate,
            out_dir / filename,
            out_config.out_format,
            out_config.sample_rate,
            out_config.channels,
        )

    manifest: dict = _build_manifest(
        source,
        loaded.duration_ms,
        seg_config,
        out_config,
        backend.name,
        segments,
        filenames,
    )
    manifest_path: Path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _LOGGER.info(translate("wrote_segments", lang, count=len(segments), path=str(out_dir)))


def run(
    inputs: List[Path],
    output_root: Path,
    vad_backend: str,
    seg_config: SegmentationConfig,
    out_config: OutputConfig,
    lang: Language,
) -> RunResult:
    """Process all discovered inputs, skipping and logging per-file failures.

    Args:
        inputs: Input file and/or directory paths.
        output_root: Root output directory.
        vad_backend: Name of the VAD backend to use.
        seg_config: Segmentation parameters.
        out_config: Output encoding and buffer settings.
        lang: Language for log messages.

    Returns:
        A :class:`RunResult` summarizing which inputs succeeded or failed.
    """
    files: List[Path] = discover_inputs(inputs)
    _LOGGER.info(translate("scanning_inputs", lang, count=len(inputs)))
    if not files:
        _LOGGER.warning(translate("no_inputs", lang))
        return RunResult()

    backend: VADBackend = get_backend(vad_backend)
    result: RunResult = RunResult()

    for source in files:
        try:
            process_file(source, output_root, backend, seg_config, out_config, lang)
            result.succeeded.append(source)
        except Exception as exc:  # skip & log, continue
            _LOGGER.warning(
                translate("skip_file_error", lang, path=str(source), error=str(exc))
            )
            result.failed.append(source)

    _LOGGER.info(
        translate("done", lang, ok=len(result.succeeded), failed=len(result.failed))
    )
    return result
