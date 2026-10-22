#!/usr/bin/env bash
# Run one preparation stage at a time. No stage starts inference automatically.
set -euo pipefail

STEP_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CONDA_BIN=${CONDA_BIN:-$HOME/miniforge3/bin/conda}
WILOR_ENV="$STEP_ROOT/envs/wilor"
DYN_ENV="$STEP_ROOT/envs/dynhamr"
WILOR_REPO="$STEP_ROOT/repos/WiLoR"
DYN_REPO="$STEP_ROOT/repos/Dyn-HaMR"
BMC_REPO="$STEP_ROOT/repos/Hand-BMC-pytorch"
WILOR_REV=fcb911312a38fa8badd30d9656a167485d61b8f9
DYN_REV=fa9cd7412c205fd15ee4139c8caacf79bf6167e6

# This host's global Git config rewrites GitHub to an unavailable cnpmjs mirror.
# These variables affect this script and its children; no Git config is modified.
export GIT_CONFIG_GLOBAL=/dev/null
export GIT_CONFIG_NOSYSTEM=1
export PYOPENGL_PLATFORM=egl
# Resolve dependencies during explicit environment setup, never during inference.
export YOLO_AUTOINSTALL=false

usage() {
    cat <<'EOF'
Usage: bash prepare.sh STAGE [ARGUMENT]
  sources          Fetch pinned WiLoR, Dyn-HaMR, and Hand-BMC sources
  wilor-env        Create the Python 3.10 / PyTorch 2.7.0 CUDA 12.8 environment
  wilor-weights    Download detector.pt and wilor_final.ckpt (about 2.62 GB)
  mano PATH        Copy a separately licensed MANO_RIGHT.pkl into both projects
  wilor-check      Check CUDA, resources, and WiLoR model loading
  dynhamr-base     Create the camera-relative optimization environment
  dynhamr-assets   Populate MANO mean parameters and the shared detector
  bmc              Download joint data and create eight BMC statistics
  dynhamr-bundles  Download optional upstream HaMeR, DROID, and HMP bundles
  no standard invocation loads models implicitly
EOF
}

require_repo() {
    if [[ ! -d "$1/.git" ]]; then
        echo "Missing repository: $1. Run: bash prepare.sh sources" >&2
        exit 1
    fi
}

require_env() {
    if [[ ! -x "$1/bin/python" ]]; then
        echo "Missing environment: $1" >&2
        exit 1
    fi
}

copy_if_same_or_missing() {
    local source=$1 target=$2
    mkdir -p -- "$(dirname -- "$target")"
    if [[ -e "$target" ]]; then
        cmp -s -- "$source" "$target" || {
            echo "Existing file differs; left unchanged: $target" >&2
            exit 1
        }
    else
        cp -- "$source" "$target"
    fi
}

fetch_repo() {
    local url=$1 target=$2 revision=${3:-}
    if [[ ! -e "$target" ]]; then
        git clone "$url" "$target"
    fi
    require_repo "$target"
    if [[ -n "$revision" && "$(git -C "$target" rev-parse HEAD)" != "$revision" ]]; then
        if [[ -n "$(git -C "$target" status --porcelain)" ]]; then
            echo "Repository has local changes; left unchanged: $target" >&2
            exit 1
        fi
        git -C "$target" cat-file -e "$revision^{commit}" 2>/dev/null || git -C "$target" fetch origin "$revision"
        git -C "$target" checkout --detach "$revision"
    fi
    git -C "$target" submodule update --init --recursive
    git -C "$target" rev-parse HEAD > "$STEP_ROOT/logs/$(basename "$target")-commit.txt"
}

create_env() {
    local env_dir=$1 requirements=$2
    if [[ ! -x "$env_dir/bin/python" ]]; then
        "$CONDA_BIN" create --prefix "$env_dir" --override-channels -c conda-forge python=3.10 pip -y
    fi
    export PATH="$env_dir/bin:$PATH"
    "$env_dir/bin/python" -m pip install 'pip==24.3.1' 'setuptools==69.5.1' 'wheel==0.45.1'
    "$env_dir/bin/python" -m pip install 'torch==2.7.0' 'torchvision==0.22.0' --index-url https://download.pytorch.org/whl/cu128
    "$env_dir/bin/python" -m pip install -r "$requirements"
    "$env_dir/bin/python" -m pip install --no-build-isolation 'https://codeload.github.com/mattloper/chumpy/zip/580566eafc9ac68b2614b64d6f7aaa84eebb70da#sha256=b668f0bd571bc8ccc0b09ebc696b52689949b3be3e893b3fa5cd27a8956cb10e'
    "$env_dir/bin/python" -m pip check
    "$env_dir/bin/python" -m pip freeze > "$STEP_ROOT/logs/$(basename "$env_dir")-pip-freeze.txt"
    "$env_dir/bin/python" - <<'PY'
import torch
print('PyTorch:', torch.__version__, 'CUDA runtime:', torch.version.cuda)
assert torch.cuda.is_available(), 'PyTorch cannot access CUDA'
print('GPU:', torch.cuda.get_device_name(), 'capability:', torch.cuda.get_device_capability())
x = torch.randn(128, 128, device='cuda')
assert torch.isfinite(x @ x.T).all().item()
torch.cuda.synchronize()
print('GPU computation passed. This is not a model/pipeline test.')
PY
}

download() {
    local url=$1 target=$2 expected_bytes=${3:-0}
    mkdir -p -- "$(dirname -- "$target")"
    if [[ -s "$target" ]]; then
        if (( expected_bytes > 0 )) && [[ "$(stat -c %s -- "$target")" != "$expected_bytes" ]]; then
            echo "Unexpected existing file size; left unchanged: $target" >&2
            exit 1
        fi
        return
    fi
    curl --fail --location --retry 5 --retry-delay 3 --connect-timeout 30 \
        --continue-at - --output "$target.part" "$url"
    if (( expected_bytes > 0 )); then
        [[ "$(stat -c %s -- "$target.part")" == "$expected_bytes" ]] || {
            echo "Unexpected download size: $target.part" >&2
            exit 1
        }
    fi
    mv -- "$target.part" "$target"
    sha256sum "$target" >> "$STEP_ROOT/logs/downloads.sha256"
}

stage=${1:-help}
case "$stage" in
    help|-h|--help) usage; exit 0 ;;
    sources|wilor-env|wilor-weights|mano|wilor-check|dynhamr-base|dynhamr-assets|bmc|dynhamr-bundles) ;;
    *) usage >&2; exit 2 ;;
esac
mkdir -p "$STEP_ROOT/logs" "$STEP_ROOT/downloads" "$STEP_ROOT/assets/mano" "$STEP_ROOT/repos" "$STEP_ROOT/envs"

case "$stage" in
    sources)
        fetch_repo https://github.com/rolpotamias/WiLoR.git "$WILOR_REPO" "$WILOR_REV"
        fetch_repo https://github.com/ZhengdiYu/Dyn-HaMR.git "$DYN_REPO" "$DYN_REV"
        fetch_repo https://github.com/MengHao666/Hand-BMC-pytorch.git "$BMC_REPO"
        ;;
    wilor-env)
        create_env "$WILOR_ENV" "$STEP_ROOT/requirements-wilor.txt"
        ;;
    wilor-weights)
        require_repo "$WILOR_REPO"
        download https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt \
            "$WILOR_REPO/pretrained_models/detector.pt" 53582271
        download https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/wilor_final.ckpt \
            "$WILOR_REPO/pretrained_models/wilor_final.ckpt" 2564989533
        ;;
    mano)
        require_repo "$WILOR_REPO"
        require_repo "$DYN_REPO"
        if [[ $# != 2 || ! -s "$2" ]]; then
            echo 'Usage: bash prepare.sh mano /absolute/path/to/MANO_RIGHT.pkl' >&2
            exit 2
        fi
        copy_if_same_or_missing "$2" "$STEP_ROOT/assets/mano/MANO_RIGHT.pkl"
        copy_if_same_or_missing "$2" "$WILOR_REPO/mano_data/MANO_RIGHT.pkl"
        copy_if_same_or_missing "$2" "$DYN_REPO/_DATA/data/mano/MANO_RIGHT.pkl"
        ;;
    wilor-check)
        require_repo "$WILOR_REPO"
        require_env "$WILOR_ENV"
        cd "$WILOR_REPO"
        # Trusted upstream checkpoints use the legacy serialized-object format.
        PATH="$WILOR_ENV/bin:$PATH" TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 "$WILOR_ENV/bin/python" - <<'PY'
from pathlib import Path
import importlib.util
import sys
if importlib.util.find_spec('dill') is None:
    raise SystemExit(f'Missing checkpoint dependency. Run: {sys.executable} -m pip install dill==0.3.8')
import torch
from ultralytics import YOLO
from wilor.models import load_wilor
for name in ['pretrained_models/detector.pt', 'pretrained_models/wilor_final.ckpt',
             'pretrained_models/model_config.yaml', 'mano_data/MANO_RIGHT.pkl',
             'mano_data/mano_mean_params.npz']:
    assert Path(name).is_file(), f'Missing: {name}'
assert torch.cuda.is_available()
detector = YOLO('pretrained_models/detector.pt').to('cuda')
model, cfg = load_wilor('pretrained_models/wilor_final.ckpt', 'pretrained_models/model_config.yaml')
model.eval().to('cuda')
torch.cuda.synchronize()
print('WiLoR detector / model / MANO loading passed.')
print('No inference or image-rendering test has been performed.')
PY
        ;;
    dynhamr-base)
        require_repo "$DYN_REPO"
        create_env "$DYN_ENV" "$STEP_ROOT/requirements-dynhamr-base.txt"
        "$DYN_ENV/bin/python" -m pip install --no-build-isolation --no-deps 'https://codeload.github.com/otaheri/MANO/zip/5869ab059c1bf31cc724f57eaf93e041135e8960#sha256=339525663f6a574997cd4e51ccb83c01052c30a4f16261323237a2c1403bfcc8'
        "$DYN_ENV/bin/python" -m pip install --no-build-isolation --no-deps -e "$DYN_REPO"
        "$DYN_ENV/bin/python" -m pip check
        "$DYN_ENV/bin/python" -m pip freeze > "$STEP_ROOT/logs/dynhamr-pip-freeze.txt"
        echo 'Base environment installed. Full Dyn-HaMR/HMP/SLAM runtime remains unvalidated; see SETUP.md.'
        ;;
    dynhamr-assets)
        require_repo "$WILOR_REPO"
        require_repo "$DYN_REPO"
        copy_if_same_or_missing "$WILOR_REPO/mano_data/mano_mean_params.npz" "$DYN_REPO/_DATA/data/mano_mean_params.npz"
        copy_if_same_or_missing "$WILOR_REPO/pretrained_models/detector.pt" "$DYN_REPO/third-party/hamer/pretrained_models/detector.pt"
        ;;
    bmc)
        require_repo "$BMC_REPO"
        require_repo "$DYN_REPO"
        require_env "$DYN_ENV"
        if [[ ! -s "$STEP_ROOT/downloads/bmc_joints.zip" ]]; then
            "$DYN_ENV/bin/python" -m gdown 1_wV8QjmsVCMBEBhm56gFA2XTyU8VEHzk -O "$STEP_ROOT/downloads/bmc_joints.zip.part"
            "$DYN_ENV/bin/python" -m zipfile -t "$STEP_ROOT/downloads/bmc_joints.zip.part"
            mv "$STEP_ROOT/downloads/bmc_joints.zip.part" "$STEP_ROOT/downloads/bmc_joints.zip"
        fi
        "$DYN_ENV/bin/python" -m zipfile -e "$STEP_ROOT/downloads/bmc_joints.zip" "$BMC_REPO"
        cd "$BMC_REPO"
        [[ -s joints/rhd_train.npy && -s joints/rhd_test.npy && -s joints/gan_train.npy ]] || {
            echo 'Unexpected BMC archive layout.' >&2
            exit 1
        }
        MPLBACKEND=Agg "$DYN_ENV/bin/python" calculate_bmc.py
        # Preserve upstream thresholds; disable its default interactive visualization.
        MPLBACKEND=Agg "$DYN_ENV/bin/python" - <<'PY'
from pathlib import Path
import sys
path = Path('calculate_convex_hull.py')
code = path.read_text()
assert code.count('default=True') == 1, 'Upstream changed; review visualization flag'
code = code.replace('default=True', 'default=False')
sys.argv = [str(path)]
exec(compile(code, str(path), 'exec'), {'__name__': '__main__', '__file__': str(path)})
PY
        for name in bone_len_max bone_len_min CONVEX_HULLS curvatures_max curvatures_min joint_angles PHI_max PHI_min; do
            copy_if_same_or_missing "$BMC_REPO/BMC/$name.npy" "$DYN_REPO/_DATA/BMC/$name.npy"
        done
        ;;
    dynhamr-bundles)
        require_repo "$DYN_REPO"
        require_env "$DYN_ENV"
        # Optional upstream full-pipeline resource set, not required for WiLoR inference.
        download https://www.cs.utexas.edu/~pavlakos/hamer/data/hamer_demo_data.tar.gz \
            "$STEP_ROOT/downloads/hamer_demo_data.tar.gz" 6037554929
        tar -tzf "$STEP_ROOT/downloads/hamer_demo_data.tar.gz" > "$STEP_ROOT/logs/hamer-archive-files.txt"
        tar --no-same-owner --no-same-permissions --skip-old-files -xzf "$STEP_ROOT/downloads/hamer_demo_data.tar.gz" -C "$DYN_REPO"
        mkdir -p "$DYN_REPO/_DATA"
        if [[ ! -s "$DYN_REPO/_DATA/droid.pth" ]]; then
            "$DYN_ENV/bin/python" -m gdown 1VD1vGhl_NPzy8mza4Fx6vvqFpnlzZ86L -O "$DYN_REPO/_DATA/droid.pth.part"
            mv "$DYN_REPO/_DATA/droid.pth.part" "$DYN_REPO/_DATA/droid.pth"
        fi
        if [[ ! -s "$STEP_ROOT/downloads/hmp_model.zip" ]]; then
            "$DYN_ENV/bin/python" -m gdown 1_rp_SMxZA6Gjm6Z49M6hXam9kjMD6-GQ -O "$STEP_ROOT/downloads/hmp_model.zip.part"
            "$DYN_ENV/bin/python" -m zipfile -t "$STEP_ROOT/downloads/hmp_model.zip.part"
            mv "$STEP_ROOT/downloads/hmp_model.zip.part" "$STEP_ROOT/downloads/hmp_model.zip"
        fi
        "$DYN_ENV/bin/python" -m zipfile -e "$STEP_ROOT/downloads/hmp_model.zip" "$DYN_REPO/_DATA"
        echo 'Upstream resource bundles downloaded; this does not install HaMeR/SLAM/HMP runtimes.'
        ;;
esac
