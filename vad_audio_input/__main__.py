"""Module entry point: ``python -m vad_audio_input``."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
