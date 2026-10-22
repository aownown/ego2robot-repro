#!/usr/bin/env bash
set -euo pipefail
STEP_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
mkdir -p "$STEP_ROOT/logs"
LOG_PATH=$(mktemp "$STEP_ROOT/logs/run_$(date +%Y%m%d_%H%M%S)_XXXXXX.log")
"$STEP_ROOT/envs/da3/bin/python" "$STEP_ROOT/src/run_pipeline.py" "$@" 2>&1 | tee "$LOG_PATH"
