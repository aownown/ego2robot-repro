#!/usr/bin/env bash
set -euo pipefail
STEP_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ $# -lt 1 ]]; then
    echo 'Usage: bash run_wilor_tracks.sh outputs/wilor_export/run_... [--hand-masks outputs/sam3_hand_masks/run_...]' >&2
    exit 2
fi
SOURCE_DIR=$(realpath "$1")
shift
mkdir -p "$STEP_ROOT/outputs/wilor_tracks" "$STEP_ROOT/logs"
RUN_DIR=$(mktemp -d "$STEP_ROOT/outputs/wilor_tracks/run_$(date +%Y%m%d_%H%M%S)_XXXXXX")
LOG="$STEP_ROOT/logs/wilor_tracks_$(basename "$RUN_DIR").log"
echo "Log: $LOG"
"$STEP_ROOT/envs/wilor/bin/python" -u "$STEP_ROOT/src/track_wilor.py" \
    --input "$SOURCE_DIR" --output "$RUN_DIR" "$@" 2>&1 | tee "$LOG"
