#!/usr/bin/env bash
#
# Run the VAD segmenter, teeing stdout+stderr to a timestamped log file.
#
# Usage: ./run.sh INPUT [INPUT ...] -o OUTPUT_DIR --min-silence MS --initial-duration MS [options]
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate the virtualenv if present.
if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TIMESTAMP="$(date +'%Y%m%d_%H%M%S')"
LOG_FILE="${LOG_DIR}/run_${TIMESTAMP}.log"

echo "Logging to ${LOG_FILE}"

# Pass the same log file to the program and also capture all shell-level output.
python -m vad_audio_input --log-file "$LOG_FILE" "$@" 2>&1 | tee -a "$LOG_FILE"

# Propagate the python exit code (not tee's).
exit "${PIPESTATUS[0]}"
