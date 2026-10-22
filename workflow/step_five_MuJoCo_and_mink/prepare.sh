#!/usr/bin/env bash
set -euo pipefail
STEP_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ENV_NAME=mujoco_mink
PY_VERSION=3.11
source "$STEP_ROOT/../common/prepare_lib.sh"
sources() {
    fetch_repo https://github.com/kevinzakka/mink.git mink ab45779fea46933832dee1c240f94103633347a1
    fetch_repo https://github.com/google-deepmind/mujoco_menagerie.git mujoco_menagerie 8161bba264d7fa7c99ca301e91e7fb44737676ad franka_emika_panda
}
env() {
    create_env
    "$PY" -m pip install -r "$STEP_ROOT/requirements.txt"
    finish_env
}
weights() { echo 'No neural weights are required; the Panda MJCF and meshes come from sources.'; }
check() { "$PY" "$STEP_ROOT/src/check.py"; }
model_check() { check; }
dispatch "$@"
