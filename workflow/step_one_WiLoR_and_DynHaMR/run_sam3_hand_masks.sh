#!/usr/bin/env bash
set -euo pipefail
STEP_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ $# -lt 1 ]]; then
    echo 'Usage: bash run_sam3_hand_masks.sh outputs/wilor_export/run_... [--limit 3]' >&2
    exit 2
fi
SOURCE_DIR=$(realpath "$1")
shift
SAM_ENV="$STEP_ROOT/../step_three_SAM3/envs/sam3"
export PATH="$SAM_ENV/bin:$PATH" MPLBACKEND=Agg PYTHONUNBUFFERED=1
mkdir -p "$STEP_ROOT/outputs/sam3_hand_masks" "$STEP_ROOT/logs"
RUN_DIR=$(mktemp -d "$STEP_ROOT/outputs/sam3_hand_masks/run_$(date +%Y%m%d_%H%M%S)_XXXXXX")
LOG="$STEP_ROOT/logs/sam3_hand_masks_$(basename "$RUN_DIR").log"
echo "Output: $RUN_DIR"
echo "Log: $LOG"
"$SAM_ENV/bin/python" "$STEP_ROOT/src/sam3_hand_masks.py" --input "$SOURCE_DIR" --output "$RUN_DIR" "$@" 2>&1 | tee "$LOG"
