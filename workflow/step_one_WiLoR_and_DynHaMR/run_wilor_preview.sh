#!/usr/bin/env bash
# First inference exercise: 15 RGB frames, upstream WiLoR demo, no fast/pruning.
set -euo pipefail
STEP_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WILOR_REPO="$STEP_ROOT/repos/WiLoR"
PY="$STEP_ROOT/envs/wilor/bin/python"
INPUT_DIR="$STEP_ROOT/inputs/wilor_preview_15"
SOURCE_DIR="$STEP_ROOT/../../test_data/rgb/frames"
action=${1:-run}
case "$action" in
    prepare|run) ;;
    -h|--help|help)
        echo 'Usage: bash run_wilor_preview.sh [prepare|run]'
        echo 'prepare: validate and link 15 samples; run (default): prepare + upstream demo'
        exit 0 ;;
    *) echo 'Expected prepare or run' >&2; exit 2 ;;
esac
[[ -x "$PY" ]] || { echo 'WiLoR environment missing; see SETUP.md' >&2; exit 1; }

"$PY" - "$SOURCE_DIR" "$INPUT_DIR" <<'PY'
import json
import sys
from pathlib import Path
from PIL import Image

source, target = (Path(p).resolve() for p in sys.argv[1:])
frames = sorted(source.glob('frame_*.jpg'))
assert len(frames) == 450, f'Expected the prepared 450-frame clip, got {len(frames)}'
assert [p.name for p in frames] == [f'frame_{i:06d}.jpg' for i in range(450)]
sample = frames[::30]
target.mkdir(parents=True, exist_ok=True)
allowed = {p.name for p in sample} | {'samples.json'}
unexpected = {p.name for p in target.iterdir()} - allowed
assert not unexpected, f'Unexpected files in sample folder; left unchanged: {unexpected}'
records = []
for p in sample:
    with Image.open(p) as image:
        assert image.size == (1280, 720), (p, image.size)
        image.verify()
    link = target / p.name
    if link.is_symlink():
        assert link.resolve() == p, f'Existing link differs: {link}'
    elif link.exists():
        assert link.read_bytes() == p.read_bytes(), f'Existing image differs: {link}'
    else:
        link.symlink_to(p)
    index = int(p.stem.split('_')[-1])
    records.append(dict(file=p.name, source=str(p), clip_frame_index=index,
                        clip_time_seconds=index / 30., original_video_time_seconds=6 + index / 30.))
(target / 'samples.json').write_text(json.dumps(records, indent=2) + '\n')
print(f'Prepared {len(sample)} RGB samples: {target}')
PY

[[ "$action" != prepare ]] || exit 0
for resource in pretrained_models/wilor_final.ckpt pretrained_models/detector.pt \
    pretrained_models/model_config.yaml mano_data/MANO_RIGHT.pkl; do
    [[ -s "$WILOR_REPO/$resource" ]] || { echo "Missing $resource; see SETUP.md" >&2; exit 1; }
done
export PATH="$STEP_ROOT/envs/wilor/bin:$PATH"
export PYOPENGL_PLATFORM=egl
export YOLO_AUTOINSTALL=false
# Scoped to this script: load the existing official legacy model checkpoints.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export MPLBACKEND=Agg

"$PY" - <<'PY'
import torch
assert torch.cuda.is_available(), 'CUDA is unavailable'
free, total = torch.cuda.mem_get_info()
print(torch.cuda.get_device_name(), f'free GPU memory: {free / 2**30:.1f} GiB')
PY

mkdir -p "$STEP_ROOT/outputs/wilor_preview" "$STEP_ROOT/logs"
RUN_DIR=$(mktemp -d "$STEP_ROOT/outputs/wilor_preview/run_$(date +%Y%m%d_%H%M%S)_XXXXXX")
LOG="$STEP_ROOT/logs/wilor_preview_$(basename "$RUN_DIR").log"
echo "Output: $RUN_DIR"
echo "Log: $LOG"
cp "$INPUT_DIR/samples.json" "$RUN_DIR/samples.json"
cd "$WILOR_REPO"
"$PY" -u demo.py --img_folder "$INPUT_DIR" --out_folder "$RUN_DIR" --save_mesh 2>&1 | tee "$LOG"

"$PY" - "$INPUT_DIR" "$RUN_DIR" <<'PY' | tee -a "$LOG"
import json
import sys
from pathlib import Path
sample, output = map(Path, sys.argv[1:])
expected = sorted(p.name for p in sample.glob('*.jpg'))
rendered = sorted(p.name for p in output.glob('*.jpg'))
missing = sorted(set(expected) - set(rendered))
summary = dict(input_frames=len(expected), rendered_frames=len(rendered),
               mesh_files=len(list(output.glob('*.obj'))), frames_without_overlay=missing,
               note='Upstream demo skips frames with no detections. Mesh indices are not tracked hand IDs.')
(output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
print('Preview only: calibrated 21-keypoint / MANO parameter export and Dyn-HaMR are not run.')
assert rendered, 'No hand overlays produced; inspect RGB samples and the detection log.'
PY
echo "Preview finished: $RUN_DIR"
