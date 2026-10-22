"""Export raw per-frame WiLoR detections. Does not associate tracks or run Dyn-HaMR."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from fractions import Fraction

import cv2
import numpy as np
import torch
from camera_init import crop_to_pixels, fit_translation, project

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / 'repos/WiLoR'
sys.path.insert(0, str(REPO))

FIELDS = {
    'detection_id': ((), np.int32), 'is_right': ((), np.int8),
    'score': ((), np.float32), 'bbox_xyxy': ((4,), np.float32),
    'box_center': ((2,), np.float32), 'box_size': ((), np.float32),
    'pred_cam_raw': ((3,), np.float32), 'pred_cam_t_crop_raw': ((3,), np.float32),
    'mano_global_orient_raw': ((1, 3, 3), np.float32),
    'mano_hand_pose_raw': ((15, 3, 3), np.float32), 'mano_betas_raw': ((10,), np.float32),
    'keypoints_model_m': ((21, 3), np.float32), 'vertices_model_m': ((778, 3), np.float32),
    'keypoints_crop_normalized': ((21, 2), np.float32), 'keypoints_2d_model_px': ((21, 2), np.float32),
    'keypoints_local_m': ((21, 3), np.float32),
    'camera_translation_demo_m': ((3,), np.float32),
    'camera_translation_pinhole_m': ((3,), np.float32),
    'keypoints_cam_pinhole_m': ((21, 3), np.float32),
    'keypoints_2d_pinhole_px': ((21, 2), np.float32),
    'translation_fit_valid': ((), bool), 'translation_fit_rmse_px': ((), np.float32),
}


def pack(records):
    """Keep zero-detection frames with explicit [0,...] shapes."""
    return {key: np.asarray([r[key] for r in records], dtype=dtype).reshape((len(records), *shape))
            for key, (shape, dtype) in FIELDS.items()}


def annotate(image, records, index):
    im = image.copy()
    h, w = im.shape[:2]
    cv2.putText(im, f'frame {index:06d}  detections={len(records)}  RAW / NOT TRACKED',
                (15, 28), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2, cv2.LINE_AA)
    for record in records:
        color = (0, 190, 255) if record['is_right'] else (255, 210, 0)
        pts = np.asarray(record['keypoints_2d_model_px'])
        def point(i):
            if not np.isfinite(pts[i]).all() or np.max(np.abs(pts[i])) > 1e6:
                return None
            return tuple(np.rint(pts[i]).astype(int))
        for start in (1, 5, 9, 13, 17):
            for i, j in zip([0, start, start+1, start+2], [start, start+1, start+2, start+3]):
                a, b = point(i), point(j)
                if a is not None and b is not None:
                    cv2.line(im, a, b, color, 2, cv2.LINE_AA)
        for i in range(21):
            p = point(i)
            if p is not None and 0 <= p[0] < w and 0 <= p[1] < h:
                cv2.circle(im, p, 3, color, -1, cv2.LINE_AA)
        x1, y1, x2, y2 = np.rint(record['bbox_xyxy']).astype(int)
        cv2.rectangle(im, (x1, y1), (x2, y2), color, 1)
        side = 'R' if record['is_right'] else 'L'
        label = f"{side} det{record['detection_id']} {record['score']:.2f} fit={record['translation_fit_rmse_px']:.1f}px"
        cv2.putText(im, label, (max(0, x1), max(55, y1-8)), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1, cv2.LINE_AA)
    return im


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clip', type=Path, default=ROOT.parent.parent/'test_data')
    parser.add_argument('--output', type=Path, required=True, help='must be an empty directory')
    parser.add_argument('--limit', type=int, default=0, help='0=all frames; positive=smoke test')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--confidence', type=float, default=.3)
    args = parser.parse_args()
    if args.limit < 0 or args.batch_size < 1 or not 0 < args.confidence < 1:
        parser.error('invalid limit, batch size or confidence')
    clip, output = args.clip.resolve(), args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error('output directory must be empty; existing results are preserved')
    metadata = json.loads((clip/'manifest.json').read_text())
    camera = json.loads((clip/'camera_parameters.json').read_text())
    rgb = camera['rgb']; intr = rgb['intrinsics']
    width, height = rgb['width'], rgb['height']
    K = np.array([[intr['fx'], 0, intr['cx']], [0, intr['fy'], intr['cy']], [0, 0, 1]], dtype=np.float64)
    assert np.isfinite(K).all() and K[0, 0] > 0 and K[1, 1] > 0
    with (clip/'frame_alignment.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == metadata['clip']['frames']
    assert [int(r['rgb_frame_index']) for r in rows] == list(range(len(rows)))
    for row in rows:
        assert (clip/row['rgb_file']).is_file(), row['rgb_file']
    selected = rows[:args.limit] if args.limit else rows
    fps = float(Fraction(metadata['rgb_video']['avg_frame_rate']))
    assert fps > 0 and width % 2 == 0 and height % 2 == 0
    assert shutil.which('ffmpeg'), 'ffmpeg is required for the preview video'
    assert torch.cuda.is_available(), 'CUDA is unavailable'
    output.mkdir(parents=True, exist_ok=True)
    (output/'frames').mkdir()
    (output/'preview_samples').mkdir()
    for name in ('camera_parameters.json', 'frame_alignment.csv'):
        shutil.copy2(clip/name, output/name)
    shutil.copy2(clip/'manifest.json', output/'source_manifest.json')
    run = dict(schema='pkdata.wilor.detections.v1', status='running', input=str(clip),
               requested_frames=len(selected), available_frames=len(rows), fps=fps,
               K=K.tolist(), camera_frame='OpenCV RGB camera: x right, y down, z forward',
               translation='translation-only least squares fit to model-predicted 2D joints with calibrated K',
               metric_status='MANO-scale monocular initialization; not depth-validated',
               distortion_applied=False, tracking_applied=False, dynhamr_applied=False,
               handedness='0 left / 1 right; detection_id is local to each frame',
               mano_convention='raw right-MANO canonical parameters; mirror output X for is_right=0',
               keypoint_order='OpenPose21: wrist=0, thumb_tip=4, index_tip=8, middle_tip=12',
               args={'batch_size':args.batch_size,'confidence':args.confidence,'rescale_factor':2.,'fast':False},
               code_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in (Path(__file__), Path(__file__).with_name('camera_init.py'))},
               wilor_revision=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip())
    (output/'run.json').write_text(json.dumps(run, indent=2)+'\n')
    from wilor.models import load_wilor
    from wilor.datasets.vitdet_dataset import ViTDetDataset
    from wilor.utils import recursive_to
    from wilor.utils.renderer import cam_crop_to_full
    from ultralytics import YOLO
    from tqdm import tqdm

    os.chdir(REPO)  # upstream MANO paths are relative to the repository
    model, cfg = load_wilor('./pretrained_models/wilor_final.ckpt', './pretrained_models/model_config.yaml')
    model = model.eval().cuda()
    detector = YOLO('./pretrained_models/detector.pt').to('cuda')
    demo_focal = float(cfg.EXTRA.FOCAL_LENGTH / cfg.MODEL.IMAGE_SIZE * max(width, height))
    run['demo_focal_px'] = demo_focal
    all_stats = []
    ffmpeg_log = (output/'ffmpeg.log').open('w')
    video = subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-nostdin',
                              '-f','rawvideo','-pix_fmt','bgr24','-s',f'{width}x{height}',
                              '-r',str(fps),'-i','pipe:0','-an','-c:v','libx264','-preset','fast',
                              '-crf','20','-pix_fmt','yuv420p',str(output/'keypoints_preview.mp4')],
                             stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=ffmpeg_log)
    try:
        for row in tqdm(selected, desc='WiLoR export', unit='frame'):
            index = int(row['rgb_frame_index'])
            im = cv2.imread(str(clip/row['rgb_file']))
            assert im is not None and im.shape == (height, width, 3), row['rgb_file']
            detections = detector(im, conf=args.confidence, verbose=False)[0].boxes
            boxes = detections.xyxy.detach().cpu().numpy()
            scores = detections.conf.detach().cpu().numpy()
            right = detections.cls.detach().cpu().numpy().astype(np.int8)
            assert np.isin(right, [0, 1]).all(), 'unexpected detector classes'
            records = []
            if len(boxes):
                dataset = ViTDetDataset(cfg, im, boxes, right, rescale_factor=2., fp16=False)
                loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
                for batch in loader:
                    batch = recursive_to(batch, 'cuda')
                    with torch.inference_mode():
                        pred = model(batch)
                    demo_cam = pred['pred_cam'].clone()
                    demo_cam[:, 1] *= 2 * batch['right'] - 1
                    demo_trans = cam_crop_to_full(demo_cam, batch['box_center'].float(), batch['box_size'].float(),
                                                  batch['img_size'].float(), demo_focal).cpu().numpy()
                    raw = {key:value.detach().cpu().numpy() for key,value in pred.items() if torch.is_tensor(value)}
                    mano = {key:value.detach().cpu().numpy() for key,value in pred['pred_mano_params'].items()}
                    for n in range(len(batch['img'])):
                        det_id = int(batch['personid'][n].item())
                        center = batch['box_center'][n].cpu().numpy()
                        size = float(batch['box_size'][n].item())
                        joints = raw['pred_keypoints_3d'][n].copy()
                        joints[:, 0] *= 1 if right[det_id] else -1
                        uv = crop_to_pixels(raw['pred_keypoints_2d'][n], center, size, right[det_id])
                        t, rmse, valid = fit_translation(joints, uv, K)
                        cam_joints = joints + t
                        projected = project(cam_joints, K) if valid else np.full((21,2),np.nan)
                        record = dict(detection_id=det_id, is_right=right[det_id], score=scores[det_id],
                            bbox_xyxy=boxes[det_id], box_center=center, box_size=size,
                            pred_cam_raw=raw['pred_cam'][n], pred_cam_t_crop_raw=raw['pred_cam_t'][n],
                            mano_global_orient_raw=mano['global_orient'][n].reshape(1,3,3),
                            mano_hand_pose_raw=mano['hand_pose'][n].reshape(15,3,3), mano_betas_raw=mano['betas'][n],
                            keypoints_model_m=raw['pred_keypoints_3d'][n], vertices_model_m=raw['pred_vertices'][n],
                            keypoints_crop_normalized=raw['pred_keypoints_2d'][n], keypoints_2d_model_px=uv,
                            keypoints_local_m=joints, camera_translation_demo_m=demo_trans[n],
                            camera_translation_pinhole_m=t, keypoints_cam_pinhole_m=cam_joints,
                            keypoints_2d_pinhole_px=projected, translation_fit_valid=valid, translation_fit_rmse_px=rmse)
                        records.append(record)
            arrays = pack(records)
            np.savez_compressed(output/'frames'/f'frame_{index:06d}.npz', **arrays, K=K,
                frame_index=np.int64(index), rgb_source_frame_index=np.int64(row['rgb_source_frame_index']),
                rgb_header_timestamp_ns=np.int64(row['rgb_header_timestamp_ns']),
                clip_video_time_s=np.float64(row['clip_video_time_s']))
            display = annotate(im, records, index)
            video.stdin.write(display.tobytes())
            if index % 30 == 0 or index == int(selected[-1]['rgb_frame_index']):
                assert cv2.imwrite(str(output/'preview_samples'/f'frame_{index:06d}.jpg'), display)
            all_stats.append(dict(frame_index=index, detections=len(records),
                left=int(np.sum(right==0)), right=int(np.sum(right==1)),
                valid_translation_fits=int(arrays['translation_fit_valid'].sum())))
    finally:
        video.stdin.close()
        returncode = video.wait()
        ffmpeg_log.close()
    assert returncode == 0, f'ffmpeg failed: see {output / "ffmpeg.log"}'
    total_detections = sum(r['detections'] for r in all_stats)
    summary = dict(frames_processed=len(all_stats), available_frames=len(rows),
        complete_clip=len(all_stats)==len(rows), total_detections=total_detections,
        zero_detection_frames=[r['frame_index'] for r in all_stats if not r['detections']],
        missing_left_frames=[r['frame_index'] for r in all_stats if not r['left']],
        missing_right_frames=[r['frame_index'] for r in all_stats if not r['right']],
        duplicate_handedness_frames=[r['frame_index'] for r in all_stats if r['left']>1 or r['right']>1],
        invalid_translation_fits=total_detections-sum(r['valid_translation_fits'] for r in all_stats),
        tracking_applied=False, dynhamr_applied=False)
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (output/'frame_stats.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['frame_index','detections','left','right','valid_translation_fits'])
        writer.writeheader(); writer.writerows(all_stats)
    run['status']='succeeded'; run['processed_frames']=len(all_stats)
    (output/'run.json').write_text(json.dumps(run,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
    print('Output:', output)


if __name__ == '__main__':
    main()
