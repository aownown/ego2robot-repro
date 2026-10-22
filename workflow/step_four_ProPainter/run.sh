#!/usr/bin/env bash
set -euo pipefail
STEP_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ $# -lt 3 ]]; then
    echo 'Usage: bash run.sh RGB_FRAME_DIR MASK_PNG_DIR OUTPUT_DIR [extra upstream options]' >&2
    exit 2
fi
rgb=$(realpath "$1"); masks=$(realpath "$2"); output=$(realpath -m "$3")
shift 3
py="$STEP_ROOT/envs/propainter/bin/python"
# Validate one-to-one frames before upstream code (which can broadcast one mask).
"$py" - "$rgb" "$masks" <<'PY'
import sys
from pathlib import Path
from PIL import Image
rgb = sorted(p for p in Path(sys.argv[1]).iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png'))
mask = sorted(Path(sys.argv[2]).glob('*.png'))
assert len(rgb) == len(mask) > 0, (len(rgb), len(mask))
for a, b in zip(rgb, mask):
    assert a.stem == b.stem, (a, b)
    with Image.open(a) as i, Image.open(b) as m:
        assert i.size == m.size, (a, b)
print('RGB/mask count, names and dimensions matched:', len(rgb))
PY
export MPLBACKEND=Agg
cd "$STEP_ROOT/repos/ProPainter"
exec "$py" inference_propainter.py --video "$rgb" --mask "$masks" --output "$output" \
    --fp16 --neighbor_length 10 --ref_stride 10 --subvideo_length 80 \
    --mask_dilation 4 --raft_iter 20 --save_fps 30 --save_frames "$@"
