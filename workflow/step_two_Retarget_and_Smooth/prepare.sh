#!/usr/bin/env bash
set -euo pipefail
STEP_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ENV_NAME=retarget
PY_VERSION=3.11
source "$STEP_ROOT/../common/prepare_lib.sh"
sources() { echo 'Stage two uses local geometry code under src/; no upstream model repository.'; }
env() { create_env; "$PY" -m pip install -r "$STEP_ROOT/requirements.txt"; finish_env; }
weights() { echo 'Stage two requires no model weights.'; }
check() { "$PY" "$STEP_ROOT/src/check.py"; }
model_check() { check; }
dispatch "$@"
