#!/usr/bin/env bash
set -euo pipefail
STEP_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ENV_NAME=da3
PY_VERSION=3.11
source "$STEP_ROOT/../common/prepare_lib.sh"
sources() {
    fetch_repo https://github.com/ByteDance-Seed/Depth-Anything-3.git Depth-Anything-3 3d835ec1a5802d64a8b8b15f817a1ab54809bfe4
}
env() {
    create_env; install_torch 2.10.0 0.25.0
    "$PY" -m pip install 'xformers==0.0.34' --index-url https://download.pytorch.org/whl/cu128
    "$PY" -m pip install -c "$STEP_ROOT/requirements.txt" -r "$STEP_ROOT/requirements.txt" -e "$STEP_ROOT/repos/Depth-Anything-3"
    finish_env
}
weights() {
    hf_download depth-anything/DA3METRIC-LARGE 4010e39f3634a45bc60553321fb49fb760bd594e "$STEP_ROOT/assets/DA3METRIC-LARGE" config.json model.safetensors
}
check() {
    gpu_check
    "$PY" -c 'from depth_anything_3.api import DepthAnything3; print("DA3 API import passed")'
    "$PY" "$STEP_ROOT/src/check.py"
}
model_check() { "$PY" "$STEP_ROOT/src/check_model.py"; }
dispatch "$@"
