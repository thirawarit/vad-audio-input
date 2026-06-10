"""Audio loading, resampling and segment writing.

Uses ``soundfile`` for WAV/FLAC and ``librosa`` (ffmpeg under the hood) for compressed
formats such as MP3/M4A. All in-memory waveforms are mono float32.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import librosa
import numpy as np
import soundfile as sf

from .types import Waveform

_LOGGER: Final[logging.Logger] = logging.getLogger("vad_audio_input.audio_io")

# Formats soundfile can read natively; others go through librosa/ffmpeg.
_SOUNDFILE_EXTENSIONS: Final[frozenset[str]] = frozenset({".wav", ".flac"})


@dataclass(frozen=True)
class LoadedAudio:
    """A loaded waveform plus its sample rate and original channel count.

    Attributes:
        samples: Mono float32 waveform in the range ``[-1.0, 1.0]``.
        sample_rate: Native sample rate of ``samples``, in Hz.
        channels: Channel count of the original file before down-mixing to mono.
    """

    samples: Waveform  # mono float32
    sample_rate: int
    channels: int

    @property
    def duration_ms(self) -> float:
        """Return the waveform duration in milliseconds (0.0 if the rate is invalid)."""
        if self.sample_rate <= 0:
            return 0.0
        return (self.samples.shape[0] / self.sample_rate) * 1000.0


def load_audio(path: Path) -> LoadedAudio:
    """Load an audio file as a mono float32 waveform at its native sample rate.

    WAV/FLAC are read via ``soundfile``; other formats (MP3/M4A) are decoded by
    ``librosa`` using its audioread/ffmpeg backend. Multi-channel audio is
    down-mixed to mono by averaging channels.

    Args:
        path: Path to the input audio file.

    Returns:
        The decoded audio as a :class:`LoadedAudio`.

    Raises:
        RuntimeError: If the file cannot be decoded.
    """
    suffix: str = path.suffix.lower()
    try:
        if suffix in _SOUNDFILE_EXTENSIONS:
            data: np.ndarray
            sample_rate: int
            data, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
            channels: int = int(data.shape[1])
            mono: Waveform = data.mean(axis=1).astype(np.float32)
        else:
            # librosa decodes MP3/M4A via its audioread/ffmpeg backend.
            loaded: np.ndarray
            loaded, native_rate = librosa.load(str(path), sr=None, mono=False)
            sample_rate = int(native_rate)
            if loaded.ndim == 1:
                channels = 1
                mono = loaded.astype(np.float32)
            else:
                channels = int(loaded.shape[0])
                mono = loaded.mean(axis=0).astype(np.float32)
    except Exception as exc:  # decode/IO failure
        raise RuntimeError(f"failed to decode audio: {exc}") from exc

    return LoadedAudio(samples=mono, sample_rate=int(sample_rate), channels=channels)


def resample(samples: Waveform, source_rate: int, target_rate: int) -> Waveform:
    """Resample a mono waveform to ``target_rate``.

    Args:
        samples: Mono float32 waveform.
        source_rate: Current sample rate of ``samples``, in Hz.
        target_rate: Desired sample rate, in Hz.

    Returns:
        The resampled waveform, or ``samples`` unchanged when the rates match.
    """
    if source_rate == target_rate:
        return samples
    resampled: np.ndarray = librosa.resample(
        samples, orig_sr=source_rate, target_sr=target_rate
    )
    return resampled.astype(np.float32)


def slice_ms(
    samples: Waveform,
    sample_rate: int,
    start_ms: float,
    end_ms: float,
) -> Waveform:
    """Return the sample range covering ``[start_ms, end_ms)``, clamped to bounds.

    Args:
        samples: Mono float32 waveform.
        sample_rate: Sample rate of ``samples``, in Hz.
        start_ms: Range start, in milliseconds.
        end_ms: Range end, in milliseconds.

    Returns:
        The sliced waveform, or an empty array if the range is empty after clamping.
    """
    total: int = samples.shape[0]
    start_idx: int = max(0, int(round(start_ms / 1000.0 * sample_rate)))
    end_idx: int = min(total, int(round(end_ms / 1000.0 * sample_rate)))
    if end_idx <= start_idx:
        return np.zeros(0, dtype=np.float32)
    return samples[start_idx:end_idx]


def pad_with_silence(
    samples: Waveform,
    sample_rate: int,
    pre_buffer_ms: int,
    post_buffer_ms: int,
) -> Waveform:
    """Prepend/append digital (synthetic) silence to a waveform.

    The padding is always generated silence and never neighboring source audio.

    Args:
        samples: Mono float32 waveform.
        sample_rate: Sample rate of ``samples``, in Hz.
        pre_buffer_ms: Milliseconds of silence to prepend.
        post_buffer_ms: Milliseconds of silence to append.

    Returns:
        The padded waveform, or ``samples`` unchanged when both buffers are zero.
    """
    pre: int = max(0, int(round(pre_buffer_ms / 1000.0 * sample_rate)))
    post: int = max(0, int(round(post_buffer_ms / 1000.0 * sample_rate)))
    if pre == 0 and post == 0:
        return samples
    return np.concatenate(
        [
            np.zeros(pre, dtype=np.float32),
            samples,
            np.zeros(post, dtype=np.float32),
        ]
    )


def _to_output_channels(samples: Waveform, channels: int) -> np.ndarray:
    """Expand a mono waveform to ``channels`` by duplicating it across columns.

    Args:
        samples: Mono float32 waveform.
        channels: Desired output channel count; ``<= 1`` returns ``samples`` as-is.

    Returns:
        A 1-D array for mono, or a 2-D ``(frames, channels)`` array otherwise.
    """
    if channels <= 1:
        return samples
    return np.tile(samples.reshape(-1, 1), (1, channels))


def write_segment(
    samples: Waveform,
    source_rate: int,
    path: Path,
    out_format: str,
    out_sample_rate: int,
    out_channels: int,
) -> None:
    """Resample/expand a mono segment and write it to disk.

    Creates the parent directory if needed.

    Args:
        samples: Mono float32 segment waveform.
        source_rate: Sample rate of ``samples``, in Hz.
        path: Destination file path.
        out_format: Output container/codec (e.g. ``"wav"``, ``"flac"``).
        out_sample_rate: Output sample rate, in Hz.
        out_channels: Output channel count (mono is duplicated to fill channels).
    """
    resampled: Waveform = resample(samples, source_rate, out_sample_rate)
    framed: np.ndarray = _to_output_channels(resampled, out_channels)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), framed, out_sample_rate, format=out_format.upper())
    _LOGGER.debug("wrote segment '%s'", path)
