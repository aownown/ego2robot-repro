#!/usr/bin/env bash
set -euo pipefail
STEP_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
mkdir -p "$STEP_ROOT/logs"
LOG_PATH=$(mktemp "$STEP_ROOT/logs/run_$(date +%Y%m%d_%H%M%S)_XXXXXX.log")
"$STEP_ROOT/envs/mujoco_mink/bin/python" "$STEP_ROOT/src/run_production.py" "$@" 2>&1 | tee "$LOG_PATH"
