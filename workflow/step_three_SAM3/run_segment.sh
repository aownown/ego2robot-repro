#!/usr/bin/env bash
set -euo pipefail
STEP_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
mkdir -p "$STEP_ROOT/outputs/person_masks" "$STEP_ROOT/logs"
exec "$STEP_ROOT/envs/sam3/bin/python" "$STEP_ROOT/src/segment_video.py" "$@"
