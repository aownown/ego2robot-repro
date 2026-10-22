#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
VENV=${VENV:-"$ROOT/.venv"}
if [[ ! -x "$VENV/bin/python" ]]; then
  python3 -m venv "$VENV"
fi
"$VENV/bin/python" -m pip install --quiet -r "$ROOT/tests/requirements.txt"
"$VENV/bin/python" "$ROOT/tools/repo_check.py"
"$VENV/bin/python" -m compileall -q "$ROOT/workflow" "$ROOT/test_data/_test1019/scripts" "$ROOT/tools"
PYTHONPATH="$ROOT/workflow/step_two_Retarget_and_Smooth/src:$ROOT/workflow/step_one_WiLoR_and_DynHaMR/src"   "$VENV/bin/python" "$ROOT/workflow/step_two_Retarget_and_Smooth/src/check.py"
"$VENV/bin/python" "$ROOT/tools/check_metadata.py"
