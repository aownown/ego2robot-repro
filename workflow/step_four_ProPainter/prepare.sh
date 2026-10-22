#!/usr/bin/env bash
set -euo pipefail
STEP_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ENV_NAME=propainter
PY_VERSION=3.10
source "$STEP_ROOT/../common/prepare_lib.sh"
sources() {
    fetch_repo https://github.com/sczhou/ProPainter.git ProPainter e870e79321c31b733e2031af5aa2fb1fe3ac7eec
}
env() {
    create_env; install_torch 2.7.0 0.22.0
    "$PY" -m pip install -c "$STEP_ROOT/requirements.txt" -r "$STEP_ROOT/requirements.txt" -r "$STEP_ROOT/repos/ProPainter/requirements.txt"
    finish_env
}
weights() {
    sources
    local name
    for name in ProPainter.pth recurrent_flow_completion.pth raft-things.pth; do
        download "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/$name" "$STEP_ROOT/repos/ProPainter/weights/$name"
    done
}
check() {
    gpu_check
    (cd "$STEP_ROOT/repos/ProPainter"; "$PY" inference_propainter.py --help)
    "$PY" - <<'PY'
import torch
from torchvision.ops import deform_conv2d
x = torch.randn(1, 2, 8, 8, device='cuda')
y = deform_conv2d(x, torch.zeros(1, 18, 6, 6, device='cuda'), torch.randn(2, 2, 3, 3, device='cuda'))
assert y.shape == (1, 2, 6, 6) and torch.isfinite(y).all()
print('CUDA deform_conv2d passed')
PY
}
model_check() { (cd "$STEP_ROOT/repos/ProPainter"; "$PY" "$STEP_ROOT/src/check_model.py"); }
dispatch "$@"
