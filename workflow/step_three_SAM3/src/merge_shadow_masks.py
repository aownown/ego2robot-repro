"""Merge nearby SAM3 shadow candidates with existing arm masks; CPU only.

This is a local heuristic for a reviewable ProPainter input, not shadow ground truth.
No mask is invented from darkness, and the original arm mask is always preserved.
"""
import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import tempfile
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from segment_video import STEP, VideoWriter, sha256, write_json, write_mask


def select_additions(arm, shadow, max_distance=96, contact_distance=16, min_area=64):
    if arm.shape != shadow.shape or arm.ndim != 2 or arm.dtype != bool or shadow.dtype != bool:
        raise ValueError('expected matching 2D boolean masks')
    if not 0 <= contact_distance <= max_distance or min_area < 1:
        raise ValueError('invalid distance/area settings')
    if not arm.any():
        raise ValueError('arm mask is empty; cannot associate a shadow with an arm')
    distance = cv2.distanceTransform((~arm).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    candidate = shadow & ~arm & (distance <= max_distance)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate.astype(np.uint8), connectivity=8)
    touching = np.unique(labels[candidate & (distance <= contact_distance)])
    keep = np.zeros(count, dtype=bool)
    keep[touching] = stats[touching, cv2.CC_STAT_AREA] >= min_area
    keep[0] = False
    return keep[labels]


def read_rows(path):
    with path.open(newline='') as f:
        reader = csv.DictReader(f)
        fields, rows = reader.fieldnames, list(reader)
    if not rows:
        raise ValueError(f'empty alignment: {path}')
    ids = [int(row['rgb_frame_index']) for row in rows]
    if ids != list(range(ids[0], ids[0] + len(ids))):
        raise ValueError(f'duplicate/non-contiguous frame indices: {path}')
    return fields, rows


def read_mask(path, shape):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape != shape or not np.isin(mask, [0, 255]).all():
        raise ValueError(f'invalid binary mask: {path}')
    return mask != 0


def load_sources(arm_run, shadow_run):
    configs = [json.loads((p / 'config.json').read_text()) for p in (arm_run, shadow_run)]
    summaries = [json.loads((p / 'summary.json').read_text()) for p in (arm_run, shadow_run)]
    if any(s['status'] != 'succeeded' for s in summaries):
        raise ValueError('both segmentation runs must be complete')
    if configs[0]['config']['prompt'] != 'arm':
        raise ValueError('arm source must use prompt=arm')
    if 'shadow' not in configs[1]['config']['prompt'].lower():
        raise ValueError('shadow source must use a shadow prompt')
    for key in ('clip', 'fps', 'resolution_wh', 'checkpoint_sha256', 'source_sha256'):
        if configs[0][key] != configs[1][key]:
            raise ValueError(f'input mismatch between arm and shadow runs: {key}')
    _, arms = read_rows(arm_run / 'selected_frames.csv')
    fields, shadows = read_rows(shadow_run / 'selected_frames.csv')
    arm_by_frame = {int(row['rgb_frame_index']): row for row in arms}
    for row in shadows:
        if arm_by_frame.get(int(row['rgb_frame_index'])) != row:
            raise ValueError('shadow frames/timestamps do not match the arm source')
    clip = Path(configs[0]['clip']).resolve()
    _, clip_rows = read_rows(clip / 'frame_alignment.csv')
    clip_by_frame = {int(row['rgb_frame_index']): row for row in clip_rows}
    if any(clip_by_frame.get(int(row['rgb_frame_index'])) != row for row in shadows):
        raise ValueError('current clip alignment differs from segmentation inputs')
    for name, digest in configs[0]['source_sha256'].items():
        if sha256(clip / name) != digest:
            raise ValueError(f'clip metadata has changed: {name}')
    return configs, clip, fields, shadows, len(clip_rows)


def decorate(rgb, arm, addition, label, frame, time_s):
    overlay = rgb.copy()
    for mask, color in ((arm, (30, 220, 30)), (addition, (0, 165, 255))):
        overlay[mask] = (.55 * overlay[mask] + .45 * np.array(color)).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 2)
    panel = cv2.copyMakeBorder(overlay, 64, 0, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
    cv2.putText(panel, label, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(panel, f'frame {frame:03d} | {time_s:.2f}s | green: arm | orange: added candidate shadow',
                (16, 53), cv2.FONT_HERSHEY_SIMPLEX, .5, (220, 220, 220), 1, cv2.LINE_AA)
    return panel


def merge(args, output):
    arm_run, shadow_run = args.arm_run.resolve(), args.shadow_run.resolve()
    configs, clip, fields, rows, available = load_sources(arm_run, shadow_run)
    width, height = configs[0]['resolution_wh']
    fps = float(configs[0]['fps'])
    shape = (height, width)
    n = len(rows)
    if not np.isfinite(fps) or fps <= 0 or width % 2 or height % 2:
        raise ValueError('invalid video geometry/FPS')
    for folder in ('masks', 'shadow_added', 'rgb', 'preview_samples'):
        (output / folder).mkdir()
    metadata = dict(schema='pkdata.arm_shadow_masks.config.v1', clip=str(clip),
        source_arm_run=str(arm_run), source_shadow_run=str(shadow_run),
        source_config_sha256={str(p): sha256(p / 'config.json') for p in (arm_run, shadow_run)},
        max_distance_px=args.max_distance, contact_distance_px=args.contact_distance,
        min_component_area_px=args.min_area, resolution_wh=[width, height], fps=fps,
        selection='shadow outside arm, within maximum distance; keep components near arm and above minimum area',
        no_shadow_temporal_fill=True, original_arm_preserved=True, visual_quality_review='pending',
        code_sha256=sha256(Path(__file__)))
    write_json(output / 'config.json', metadata)
    with (output / 'selected_frames.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    with (output / 'frame_mapping.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['output_index', 'output_filename'] + fields)
        writer.writeheader()
        writer.writerows(dict(output_index=i, output_filename=f'{i:04d}.png', **row) for i, row in enumerate(rows))
    for name in ('frame_alignment.csv', 'camera_parameters.json'):
        shutil.copy2(clip / name, output / name)
    stats, digests = [], {name: hashlib.sha256() for name in ('arm', 'shadow', 'merged')}
    samples = set(np.linspace(0, n - 1, min(n, 8), dtype=int))
    with ExitStack() as stack:
        preview = VideoWriter(output / 'arm_vs_arm_shadow.mp4', width * 2, height + 64, fps, stack)
        overlay = VideoWriter(output / 'mask_overlay.mp4', width, height + 64, fps, stack)
        for t, row in enumerate(rows):
            frame = int(row['rgb_frame_index'])
            filename = f'frame_{frame:06d}.png'
            rgb_path = clip / row['rgb_file']
            if rgb_path.stem != Path(filename).stem:
                raise ValueError('RGB/mask filename mismatch')
            rgb = cv2.imread(str(rgb_path))
            if rgb is None or rgb.shape != (height, width, 3):
                raise ValueError(f'invalid RGB frame: {rgb_path}')
            arm = read_mask(arm_run / 'masks' / filename, shape)
            # Deliberately use raw shadow masks: no borrowed masks across empty frames.
            shadow = read_mask(shadow_run / 'masks_raw' / filename, shape)
            addition = select_additions(arm, shadow, args.max_distance, args.contact_distance, args.min_area)
            merged = arm | addition
            assert not np.any(arm & ~merged) and not np.any(addition & ~shadow)
            write_mask(output / 'masks' / filename, merged)
            write_mask(output / 'shadow_added' / filename, addition)
            if not np.array_equal(read_mask(output / 'masks' / filename, shape), merged):
                raise ValueError('merged mask PNG round-trip mismatch')
            (output / 'rgb' / rgb_path.name).symlink_to(rgb_path)
            for name, mask in (('arm', arm), ('shadow', shadow), ('merged', merged)):
                digests[name].update(filename.encode()); digests[name].update(mask.tobytes())
            record = dict(frame=frame, raw_shadow_px=int(shadow.sum()), arm_px=int(arm.sum()),
                          added_shadow_px=int(addition.sum()), merged_px=int(merged.sum()),
                          rejected_shadow_outside_arm_px=int((shadow & ~arm & ~addition).sum()))
            record['review_large_addition'] = record['added_shadow_px'] > .75 * record['arm_px']
            stats.append(record)
            empty = np.zeros(shape, dtype=bool)
            left = decorate(rgb, arm, empty, 'Existing ARM', frame, float(row['clip_video_time_s']))
            right = decorate(rgb, arm, addition, 'ARM + nearby SHADOW candidates', frame, float(row['clip_video_time_s']))
            canvas = np.concatenate((left, right), axis=1)
            preview.write(canvas); overlay.write(right)
            if t in samples:
                if not cv2.imwrite(str(output / 'preview_samples' / filename), canvas):
                    raise RuntimeError('failed to save preview sample')
            if (t + 1) % 30 == 0 or t + 1 == n:
                print(f'Merged and rendered {t+1}/{n} frames', flush=True)
    videos = {}
    for filename, expected_width in (('arm_vs_arm_shadow.mp4', width * 2), ('mask_overlay.mp4', width)):
        probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,avg_frame_rate,nb_read_frames', '-of', 'json', str(output / filename)]))['streams'][0]
        numerator, denominator = map(int, probe['avg_frame_rate'].split('/'))
        if (int(probe['nb_read_frames']) != n or int(probe['width']) != expected_width
                or int(probe['height']) != height + 64 or abs(numerator / denominator - fps) > 1e-6):
            raise ValueError(f'video validation failed: {filename}')
        videos[filename] = probe
    write_json(output / 'frame_stats.json', stats)
    no_addition = [s['frame'] for s in stats if not s['added_shadow_px']]
    summary = dict(schema='pkdata.arm_shadow_masks.v1', status='succeeded', frames_processed=n,
        available_frames=available, complete_clip=n == available and int(rows[0]['rgb_frame_index']) == 0,
        resolution_wh=[width, height], fps=fps, original_arm_preserved=True,
        frames_without_added_shadow=no_addition,
        frames_with_large_addition=[s['frame'] for s in stats if s['review_large_addition']],
        mask_sha256={name: digest.hexdigest() for name, digest in digests.items()},
        digest_definition='ordered filename bytes + boolean mask bytes',
        visual_quality_review='pending; semantic shadow recognition and heuristic association are unverified',
        videos=videos)
    write_json(output / 'summary.json', summary)
    write_json(output / 'manifest.json', dict(status='prepared_not_inferred', source_mask_run=str(output),
        source_arm_run=str(arm_run), source_shadow_run=str(shadow_run), clip=str(clip),
        prompt='arm + spatially filtered shadow', frames=n, fps=fps, resolution_wh=[width, height],
        rgb_directory=str(output / 'rgb'), mask_directory=str(output / 'masks'),
        start_frame=int(rows[0]['rgb_frame_index']), end_frame_exclusive=int(rows[-1]['rgb_frame_index']) + 1,
        visual_quality_review='pending'))
    print(f'Candidate masks saved. Frames without additions: {len(no_addition)}/{n}', flush=True)
    print(f'Preview: {output / "arm_vs_arm_shadow.mp4"}', flush=True)
    if len(no_addition) == n:
        print('No new shadow pixels were found: these masks equal the existing arm masks; inspect raw predictions.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm-run', type=Path, default=STEP / 'outputs/person_masks/run_20260911_143312_xgz87fo5')
    parser.add_argument('--shadow-run', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--max-distance', type=float, default=96, help='max shadow distance from arm, in source-image pixels')
    parser.add_argument('--contact-distance', type=float, default=16, help='component must contain pixels this close to arm')
    parser.add_argument('--min-area', type=int, default=64, help='minimum shadow component area in source-image pixels')
    args = parser.parse_args()
    if (not np.isfinite([args.max_distance, args.contact_distance]).all()
            or not 0 <= args.contact_distance <= args.max_distance or args.min_area < 1):
        parser.error('invalid distance or area settings')
    parent = STEP / 'outputs/arm_shadow_masks'
    parent.mkdir(parents=True, exist_ok=True)
    output = args.output.resolve() if args.output else Path(tempfile.mkdtemp(prefix=f'run_{datetime.now():%Y%m%d_%H%M%S}_', dir=parent))
    if output.exists() and any(output.iterdir()):
        parser.error('output directory must be new or empty')
    output.mkdir(parents=True, exist_ok=True)
    progress = dict(status='running', output=str(output), visual_quality_review='pending')
    write_json(output / 'run.json', progress)
    try:
        merge(args, output)
        progress['status'] = 'succeeded'
    except BaseException as error:
        progress.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error))
        raise
    finally:
        write_json(output / 'run.json', progress)


if __name__ == '__main__':
    main()
