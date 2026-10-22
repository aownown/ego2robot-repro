#!/usr/bin/env bash
set -euo pipefail
STEP_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PY="$STEP_ROOT/envs/wilor/bin/python"
export PATH="$STEP_ROOT/envs/wilor/bin:$PATH"
export PYOPENGL_PLATFORM=egl YOLO_AUTOINSTALL=false MPLBACKEND=Agg
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    exec "$PY" "$STEP_ROOT/src/export_wilor.py" --help
fi
mkdir -p "$STEP_ROOT/outputs/wilor_export" "$STEP_ROOT/logs"
RUN_DIR=$(mktemp -d "$STEP_ROOT/outputs/wilor_export/run_$(date +%Y%m%d_%H%M%S)_XXXXXX")
LOG="$STEP_ROOT/logs/wilor_export_$(basename "$RUN_DIR").log"
echo "Output: $RUN_DIR"
echo "Log: $LOG"
"$PY" -u "$STEP_ROOT/src/export_wilor.py" --output "$RUN_DIR" "$@" 2>&1 | tee "$LOG"
