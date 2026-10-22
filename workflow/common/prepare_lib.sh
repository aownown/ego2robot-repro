#!/usr/bin/env bash
# Shared only by steps 2–6. STEP_ROOT / ENV_NAME / PY_VERSION are set by caller.
set -euo pipefail
CONDA_BIN=${CONDA_BIN:-$HOME/miniforge3/bin/conda}
ENV_DIR="$STEP_ROOT/envs/$ENV_NAME"
PY="$ENV_DIR/bin/python"
export GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1
export PYOPENGL_PLATFORM=egl MUJOCO_GL=egl MPLBACKEND=Agg
export PIP_DISABLE_PIP_VERSION_CHECK=1
mkdir -p "$STEP_ROOT"/{repos,envs,assets,downloads,logs,configs,src,outputs}

require_env() {
    [[ -x "$PY" ]] || { echo "Run bash prepare.sh env first: $ENV_DIR" >&2; exit 1; }
    export PATH="$ENV_DIR/bin:$PATH"
}

fetch_repo() {
    local url=$1 name=$2 revision=$3 sparse=${4:-}
    local target="$STEP_ROOT/repos/$name"
    if [[ ! -e "$target" ]]; then
        if [[ -n "$sparse" ]]; then
            git clone --depth 1 --filter=blob:none --sparse "$url" "$target"
        else
            git clone --depth 1 "$url" "$target"
        fi
    fi
    [[ -d "$target/.git" ]] || { echo "Not a Git checkout: $target" >&2; exit 1; }
    if [[ "$(git -C "$target" rev-parse HEAD)" != "$revision" ]]; then
        [[ -z "$(git -C "$target" status --porcelain)" ]] || {
            echo "Repository has local changes; inspect before resetting: $target" >&2; exit 1;
        }
        git -C "$target" cat-file -e "$revision^{commit}" 2>/dev/null || git -C "$target" fetch --depth 1 origin "$revision"
        git -C "$target" checkout --detach "$revision"
    fi
    [[ -z "$sparse" ]] || git -C "$target" sparse-checkout set "$sparse"
    git -C "$target" rev-parse HEAD > "$STEP_ROOT/logs/$name-commit.txt"
}

create_env() {
    if [[ ! -x "$PY" ]]; then
        "$CONDA_BIN" create --prefix "$ENV_DIR" --override-channels -c conda-forge "python=$PY_VERSION" pip -y
    fi
    require_env
    # SAM3 imports pkg_resources; setuptools 82 removed that module.
    "$PY" -m pip install 'pip==25.0.1' 'setuptools==80.9.0' 'wheel==0.45.1'
}

install_torch() {
    "$PY" -m pip install "torch==$1+cu128" "torchvision==$2+cu128" --index-url https://download.pytorch.org/whl/cu128
}

finish_env() {
    "$PY" -m pip check
    "$PY" -m pip freeze > "$STEP_ROOT/logs/$ENV_NAME-pip-freeze.txt"
    "$CONDA_BIN" list --prefix "$ENV_DIR" --explicit > "$STEP_ROOT/logs/$ENV_NAME-conda-explicit.txt"
}

gpu_check() {
    "$PY" "$STEP_ROOT/../common/check_gpu.py"
}

download() {
    local url=$1 target=$2
    mkdir -p "$(dirname -- "$target")"
    if [[ ! -s "$target" ]]; then
        curl --fail --location --retry 5 --connect-timeout 30 --continue-at - --output "$target.part" "$url"
        [[ -s "$target.part" ]] || { echo "Empty download: $target.part" >&2; exit 1; }
        mv -- "$target.part" "$target"
    fi
    sha256sum "$target" >> "$STEP_ROOT/logs/downloads.sha256"
}

hf_download() {
    require_env
    "$PY" "$STEP_ROOT/../common/download_hf.py" "$@"
}

usage() {
    echo "Usage: bash prepare.sh {sources|env|weights|check|model-check|all}"
    echo "all runs sources, env, weights, and check. model-check loads models separately."
}

dispatch() {
    local action=${1:-help}
    case "$action" in
        sources) sources ;;
        env) sources; env ;;
        weights) weights ;;
        check) require_env; check 2>&1 | tee "$STEP_ROOT/logs/check.log" ;;
        model-check) require_env; model_check 2>&1 | tee "$STEP_ROOT/logs/model-check.log" ;;
        all) sources; env; weights; check 2>&1 | tee "$STEP_ROOT/logs/check.log" ;;
        help|-h|--help) usage ;;
        *) usage >&2; exit 2 ;;
    esac
}
