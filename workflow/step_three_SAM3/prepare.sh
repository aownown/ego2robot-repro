#!/usr/bin/env bash
set -euo pipefail
STEP_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ENV_NAME=sam3
PY_VERSION=3.12
source "$STEP_ROOT/../common/prepare_lib.sh"
sources() {
    fetch_repo https://github.com/facebookresearch/sam3.git sam3 660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7
}
env() {
    create_env; install_torch 2.10.0 0.25.0
    "$PY" -m pip install -c "$STEP_ROOT/requirements.txt" -r "$STEP_ROOT/requirements.txt" -e "$STEP_ROOT/repos/sam3"
    finish_env
}
weights() {
    local target="$STEP_ROOT/assets/sam3/sam3.pt"
    local existing=${SAM3_WEIGHT_PATH:-}
    mkdir -p "$STEP_ROOT/assets/sam3"
    if [[ -n "$existing" && ! -e "$target" && -s "$existing" ]]; then
        ln -s "$existing" "$target"
        echo 'Reusing an existing SAM 3 checkpoint via symlink.'
        printf '%s\n' "$existing" > "$STEP_ROOT/logs/sam3-weight-source.txt"
    fi
    if [[ -s "$target" ]]; then
        sha256sum "$target" > "$STEP_ROOT/logs/sam3-weight.sha256"
    else
        hf_download facebook/sam3 3c879f39826c281e95690f02c7821c4de09afae7 "$STEP_ROOT/assets/sam3" sam3.pt config.json
    fi
}
check() {
    gpu_check
    "$PY" -c 'from sam3.model_builder import build_sam3_video_predictor; import cv2; print("SAM 3 video API imports passed (cv2 loader)")'
}
model_check() { "$PY" "$STEP_ROOT/src/check_model.py"; }
dispatch "$@"
