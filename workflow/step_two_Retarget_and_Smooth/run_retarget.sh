#!/usr/bin/env bash
set -euo pipefail
STEP_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SOURCE_DIR="$STEP_ROOT/../step_one_WiLoR_and_DynHaMR/outputs/dynhamr/run_20260911_111714_RC8V45"
if [[ $# -gt 0 && "$1" != --* ]]; then
    SOURCE_DIR=$(realpath "$1")
    shift
fi
if [[ ! -x "$STEP_ROOT/envs/retarget/bin/python" ]]; then
    echo 'Missing environment: run bash prepare.sh all first.' >&2
    exit 1
fi
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
mkdir -p "$STEP_ROOT/outputs/retarget" "$STEP_ROOT/logs"
RUN_DIR=$(mktemp -d "$STEP_ROOT/outputs/retarget/run_$(date +%Y%m%d_%H%M%S)_XXXXXX")
LOG="$STEP_ROOT/logs/retarget_$(basename "$RUN_DIR").log"
echo "Input: $SOURCE_DIR"
echo "Output: $RUN_DIR"
echo "Log: $LOG"
"$STEP_ROOT/envs/retarget/bin/python" "$STEP_ROOT/src/run_retarget.py" \
    --input "$SOURCE_DIR" --output "$RUN_DIR" "$@" 2>&1 | tee "$LOG"
