#!/usr/bin/env bash
set -euo pipefail
STEP_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ $# -lt 1 ]]; then
    echo 'Usage: bash run_dynhamr.sh outputs/wilor_tracks/run_... [--limit 90] [--start 0]' >&2
    exit 2
fi
SOURCE_DIR=$(realpath "$1")
shift
export PATH="$STEP_ROOT/envs/dynhamr/bin:$PATH" PYOPENGL_PLATFORM=egl MPLBACKEND=Agg
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
mkdir -p "$STEP_ROOT/outputs/dynhamr" "$STEP_ROOT/logs"
RUN_DIR=$(mktemp -d "$STEP_ROOT/outputs/dynhamr/run_$(date +%Y%m%d_%H%M%S)_XXXXXX")
LOG="$STEP_ROOT/logs/dynhamr_$(basename "$RUN_DIR").log"
echo "Output: $RUN_DIR"
echo "Log: $LOG"
"$STEP_ROOT/envs/dynhamr/bin/python" "$STEP_ROOT/src/refine_dynhamr.py" --input "$SOURCE_DIR" --output "$RUN_DIR" "$@" 2>&1 | tee "$LOG"
