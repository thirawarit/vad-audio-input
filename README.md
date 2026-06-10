# vad-audio-input

Segment long-form speech audio at silence boundaries using Voice Activity Detection
(VAD). Each input file is split into speech segments at natural silences, written to
disk with a structured filename and a per-input JSON manifest.

- **Input**: long-form speech audio — `WAV` (default), `MP3`, `M4A`, `FLAC`.
- **Output**: per-segment audio files plus a `manifest.json`, in a per-input subfolder.
- **VAD**: pluggable backend architecture; **Silero VAD** is the default.

> All time-related options are in **milliseconds**.

## Requirements

- Python 3.10
- [ffmpeg](https://ffmpeg.org/) on `PATH` (used to decode MP3/M4A)

## Setup

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python -m vad_audio_input INPUT [INPUT ...] -o OUTPUT_DIR \
  --min-silence MS --initial-duration MS [options]
```

`INPUT` may be one or more files **or directories** (directories are scanned
recursively for supported audio).

### Example

```bash
python -m vad_audio_input audio-test.wav -o output \
  --min-silence 500 --initial-duration 15000 \
  --pre-buffer 200 --post-buffer 200
```

### Run script

`run.sh` activates `.venv` (if present), runs the tool, and tees all output to a
timestamped log file under `logs/`:

```bash
./run.sh audio-test.wav -o output --min-silence 500 --initial-duration 15000
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `INPUT` (positional) | — | One or more audio files or directories. |
| `-o`, `--output-dir` | *required* | Root output directory. |
| `--min-silence MS` | *required* | Minimum silence (ms) that triggers a cut. |
| `--initial-duration MS` | *required* | Target/ideal segment length (ms). |
| `--vad-backend` | `silero` | VAD engine. |
| `--conflict` | `shift` | `shift` or `truncate` (see below). |
| `--pre-buffer MS` | `0` | Leading digital silence per segment. |
| `--post-buffer MS` | `0` | Trailing digital silence per segment. |
| `--format` | `wav` | Output format: `wav` or `flac`. |
| `--sample-rate HZ` | `16000` | Output sample rate. |
| `--channels` | `1` | Output channel count. |
| `--lang` | `en` | Log/UI language: `en`, `th`, `zh`. |
| `--log-file` | `logs/run_<timestamp>.log` | Log file path. |
| `-v`, `--verbose` | off | Enable debug logging. |

## How segmentation works

1. The VAD backend yields **speech spans**; the silences between them are gaps.
2. A silence gap at least `--min-silence` wide is a **candidate cut** (placed at the
   gap midpoint).
3. Segments are steered toward `--initial-duration` as a **target length**, cutting at
   the candidate nearest that target.
4. When the ideal target boundary does not align with a candidate cut, `--conflict`
   decides:
   - **`shift`** (default): move the boundary to the *nearest* candidate cut.
   - **`truncate`**: cut at the candidate at or before the target (dropping overflow).
5. Every segment is capped at 30 minutes; longer runs are split.

**Buffers** (`--pre-buffer` / `--post-buffer`) add synthetic digital silence to each
segment — never neighboring audio. The `start_ms`/`end_ms` recorded in filenames and
the manifest always refer to the **source-audio speech boundaries**, independent of the
added buffers.

## Output layout

```
OUTPUT_DIR/
  <input-stem>/
    <input-stem>_segment00001_<start>_<end>.<ext>
    <input-stem>_segment00002_<start>_<end>.<ext>
    ...
    manifest.json
```

### Filename pattern

`{stem}_segment{NNNNN}_{start}_{end}`

- `NNNNN` — 1-based sequence, 5-digit zero-padded.
- `start` / `end` — seconds with 3 decimals, `.` replaced by `p`, dynamic integer
  width.

Example: the first segment of `audio-test` spanning 0.000s–1.150s →
`audio-test_segment00001_0p000_1p150.wav`.

### Manifest

Each input gets a `manifest.json` with the source, its duration, the run config, and
the per-segment metadata:

```json
{
  "source": "audio-test.wav",
  "source_duration_ms": 2100000,
  "config": {
    "vad_backend": "silero",
    "min_silence_ms": 500,
    "initial_duration_ms": 15000,
    "conflict": "shift",
    "pre_buffer_ms": 200,
    "post_buffer_ms": 200,
    "format": "wav",
    "sample_rate": 16000,
    "channels": 1
  },
  "segments": [
    {
      "index": 1,
      "filename": "audio-test_segment00001_0p000_1p150.wav",
      "start_ms": 0.0,
      "end_ms": 1150.0,
      "duration_ms": 1150.0,
      "pre_buffer_ms": 200,
      "post_buffer_ms": 200
    }
  ]
}
```

## Error handling

- Invalid CLI/config arguments **fail fast** before any processing.
- A file that cannot be read or decoded is **skipped and logged**, and processing
  continues with the remaining inputs.
- The process exits **non-zero if any input failed**, zero otherwise.

## Development

Run the test suite (segmentation and naming logic):

```bash
python -m pytest
```

Type-check the package:

```bash
python -m mypy vad_audio_input --ignore-missing-imports
```

## Project layout

```
vad_audio_input/
  cli.py            # argparse parser + entry point
  pipeline.py       # discover inputs → process → write segments + manifest
  segmentation.py   # silence cuts, target steering, conflict policies, 30-min cap
  audio_io.py       # load / resample / slice / silence-pad / write
  naming.py         # segment filename pattern
  i18n.py           # en / th / zh message catalog
  logging_setup.py  # formatter + Bangkok-timezone timestamps
  types.py          # shared dataclasses and type aliases
  vad/              # pluggable VAD backends (Silero default)
tests/              # pytest suite
run.sh              # convenience runner with timestamped logging
```
