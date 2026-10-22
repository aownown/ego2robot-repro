#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$ROOT/workflow/step_two_Retarget_and_Smooth/src:$ROOT/workflow/step_one_WiLoR_and_DynHaMR/src:${PYTHONPATH:-}"
python3 "$ROOT/workflow/step_two_Retarget_and_Smooth/src/check.py"
python3 "$ROOT/tools/check_metadata.py"
