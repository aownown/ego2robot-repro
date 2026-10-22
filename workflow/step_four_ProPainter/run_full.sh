#!/usr/bin/env bash
set -euo pipefail
STEP_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ $# -gt 2 ]]; then
    echo 'Usage: bash run_full.sh [NEW_OUTPUT_DIRECTORY [PREPARED_INPUT_DIRECTORY]]' >&2
    exit 2
fi
input=$(realpath -- "${2:-$STEP_ROOT/outputs/prepared_inputs/arm_20260911_143312_full450}")
[[ -f "$input/manifest.json" ]] || { echo "Missing prepared input: $input" >&2; exit 1; }
"$STEP_ROOT/envs/propainter/bin/python" - "$input" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
m = json.loads((p / 'manifest.json').read_text())
assert m['frames'] == 450 and float(m['fps']) == 30, 'full runner requires 450 frames at 30 fps'
assert m['resolution_wh'] == [1280, 720], 'full preset expects 1280x720 source images'
assert Path(m['rgb_directory']).resolve() == (p / 'rgb').resolve()
assert Path(m['mask_directory']).resolve() == (p / 'masks').resolve()
assert len(list((p / 'rgb').glob('*.jpg'))) == len(list((p / 'masks').glob('*.png'))) == 450
PY
if [[ $# -ge 1 ]]; then
    output=$(realpath -m -- "$1")
    mkdir -- "$output"  # Refuse to mix a new run with an existing directory.
else
    output=$(mktemp -d "$STEP_ROOT/outputs/inpaint_full_450_640x360_$(date +%Y%m%d_%H%M%S)_XXXXXX")
fi
cp -- "$input/manifest.json" "$output/input_manifest.json"
cp -- "$input/frame_mapping.csv" "$output/frame_mapping.csv"
sha256sum "$STEP_ROOT/run.sh" "$STEP_ROOT/run_full.sh" \
    "$STEP_ROOT/repos/ProPainter/inference_propainter.py" > "$output/code_sha256.txt"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
command=(bash "$STEP_ROOT/run.sh" "$input/rgb" "$input/masks" "$output"
    --width 640 --height 360 --neighbor_length 10 --ref_stride 10
    --subvideo_length 80 --mask_dilation 4 --raft_iter 20 --save_fps 30 --fp16 --save_frames)
{
    printf 'PYTORCH_CUDA_ALLOC_CONF=%q PYTHONUNBUFFERED=1 ' "$PYTORCH_CUDA_ALLOC_CONF"
    printf '%q ' "${command[@]}"
    printf '\n'
} > "$output/command.txt"
printf '%s\n' '450 frames / 15 seconds; processing and output: 640x360.' "Output: $output"
"${command[@]}" 2>&1 | tee "$output/run.log"
printf '\nVideo: %s/rgb/inpaint_out.mp4\n' "$output"
