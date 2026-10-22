"""SAM 3 hand masks for Ego2Robot A.1 candidate gating (not A.4 person masks)."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
from PIL import Image
import torch
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

STEP = Path(__file__).resolve().parents[1]
SAM = STEP.parent / 'step_three_SAM3'


def gate_keypoints(mask, points):
    """Off-image/nonfinite points are outside. Reject strictly >80% outside."""
    height, width = mask.shape
    finite = np.isfinite(points).all(axis=-1)
    xy = np.rint(np.where(finite[..., None], points, 0)).astype(np.int64)
    inside_image = finite & (xy[..., 0] >= 0) & (xy[..., 0] < width) & (xy[..., 1] >= 0) & (xy[..., 1] < height)
    inside = np.zeros(finite.shape, dtype=bool)
    inside[inside_image] = mask[xy[..., 1][inside_image], xy[..., 0][inside_image]]
    counts = inside.sum(axis=-1)
    return counts, (points.shape[-2] - counts) <= .8 * points.shape[-2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True, help='Full WiLoR export')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--prompt', default='hand')
    p.add_argument('--confidence', type=float, default=.5)
    a = p.parse_args()
    source, output = a.input.resolve(), a.output.resolve()
    run = json.loads((source / 'run.json').read_text())
    if run['status'] != 'succeeded' or a.limit < 0 or not 0 < a.confidence < 1:
        p.error('Expected successful export and valid limit/confidence')
    if output.exists() and any(output.iterdir()):
        p.error('Output must be empty; existing results are preserved')
    clip = Path(run['input'])
    with (source / 'frame_alignment.csv').open() as f:
        rows = list(csv.DictReader(f))
    selected = rows[:a.limit] if a.limit else rows
    for subdir in ('masks', 'frames', 'preview_samples'):
        (output / subdir).mkdir(parents=True, exist_ok=True)
    checkpoint = SAM / 'assets/sam3/sam3.pt'
    info = dict(schema='pkdata.sam3.hand_masks.v1', status='running', source=str(source),
                purpose='A.1 hand candidate gating; not A.4 person/arm removal masks',
                prompt=a.prompt, confidence_threshold=a.confidence, outside_fraction_threshold=.8,
                method='Independent per-frame SAM3 image text segmentation; union of all hand masks',
                precision='bfloat16 autocast', checkpoint=str(checkpoint.resolve()),
                sam3_revision=subprocess.check_output(['git', '-C', str(SAM/'repos/sam3'), 'rev-parse', 'HEAD'], text=True).strip(),
                code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                requested_frames=len(selected), available_frames=len(rows))
    (output / 'run.json').write_text(json.dumps(info, indent=2) + '\n')
    model = build_sam3_image_model(checkpoint_path=str(checkpoint), load_from_HF=False, compile=False)
    processor = Sam3Processor(model, confidence_threshold=a.confidence)
    rejected, empty, candidates = [], [], 0
    start = time.monotonic()
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for i, row in enumerate(selected):
            assert int(row['rgb_frame_index']) == i
            path = clip / row['rgb_file']
            with Image.open(path) as image:
                rgb = image.convert('RGB')
                state = processor.set_image(rgb)
                state = processor.set_text_prompt(a.prompt, state)
                masks = state['masks'].squeeze(1).cpu().numpy()
                scores = state['scores'].float().cpu().numpy()
                mask = masks.any(axis=0) if len(masks) else np.zeros((rgb.height, rgb.width), dtype=bool)
                with np.load(source / 'frames' / f'frame_{i:06d}.npz', allow_pickle=False) as f:
                    points = f['keypoints_2d_model_px']
                    ids = f['detection_id']
                counts, keep = gate_keypoints(mask, points)
                candidates += len(keep)
                rejected.extend(dict(frame_index=i, detection_id=int(ids[j]), inside_count=int(counts[j])) for j in np.flatnonzero(~keep))
                if not mask.any():
                    empty.append(i)
                assert cv2.imwrite(str(output / 'masks' / f'frame_{i:06d}.png'), mask.astype(np.uint8) * 255)
                np.savez_compressed(output / 'frames' / f'frame_{i:06d}.npz', frame_index=i,
                                    detection_id=ids, inside_count=counts, mask_keep=keep, instance_scores=scores)
                if i % 30 == 0 or i in (14, 45, 58, 59, 100, 144, 155, 221, 232, 325, len(selected)-1):
                    preview = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
                    preview[mask] = (.65 * preview[mask] + .35 * np.array([30, 210, 30])).astype(np.uint8)
                    for j, hand in enumerate(points):
                        color = (255, 255, 0) if keep[j] else (0, 0, 255)
                        for x, y in hand:
                            if np.isfinite([x, y]).all():
                                cv2.circle(preview, (int(round(x)), int(round(y))), 3, color, -1)
                        cv2.putText(preview, f'det{int(ids[j])}: {counts[j]}/21 keep={keep[j]}', (15, 65+25*j), cv2.FONT_HERSHEY_SIMPLEX, .65, color, 2)
                    cv2.putText(preview, f'SAM3 hand gate | frame {i}', (15, 30), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)
                    cv2.imwrite(str(output / 'preview_samples' / f'frame_{i:06d}.jpg'), preview)
                del state
            if i % 25 == 0 or i == len(selected)-1:
                print(f'{i+1}/{len(selected)} frames | rejected candidates {len(rejected)} | {time.monotonic()-start:.1f}s', flush=True)
    info.update(status='succeeded', frames_processed=len(selected), complete_clip=len(selected)==len(rows),
                total_candidates=candidates, rejected_candidates=len(rejected), rejected=rejected,
                empty_mask_frames=empty, elapsed_seconds=time.monotonic()-start)
    for name in ('run.json', 'summary.json'):
        (output / name).write_text(json.dumps(info, indent=2)+'\n')
    print('Output:', output, flush=True)


if __name__ == '__main__':
    main()
