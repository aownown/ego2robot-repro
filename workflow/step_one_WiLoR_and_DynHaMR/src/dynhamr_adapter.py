"""WiLoR rotations/translation -> Dyn-HaMR, in a camera-relative proxy frame.

No world camera trajectory is fabricated. Identity extrinsics express the
explicit choice to refine camera-relative tracks, without SLAM/world recovery.
"""
from pathlib import Path
import json
import sys

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import torch

STEP = Path(__file__).resolve().parents[1]
REPO = STEP / 'repos/Dyn-HaMR'
sys.path.insert(0, str(REPO / 'dyn-hamr'))

from body_model import MANO, run_mano


def invalid_runs(valid):
    padded = np.r_[True, valid, True]
    starts = np.flatnonzero(padded[:-1] & ~padded[1:])
    stops = np.flatnonzero(~padded[:-1] & padded[1:])
    return [(int(s), int(e)) for s, e in zip(starts, stops)]


def interpolate_initialization(trans, rotations, valid, max_gap):
    """Fill bounded short gaps for finite optimizer initialization only."""
    trans, rotations = trans.copy(), rotations.copy()
    runs = invalid_runs(valid)
    for start, stop in runs:
        if start == 0 or stop == len(valid) or stop-start > max_gap:
            raise ValueError(f'Unbounded or long gap [{start}, {stop}); choose a segment with valid boundaries. Robot home filling is a later stage.')
        ids = np.arange(start, stop)
        w = ((ids-start+1)/(stop-start+1))[:, None]
        trans[ids] = (1-w)*trans[start-1] + w*trans[stop]
        for j in range(rotations.shape[1]):
            rotations[ids, j] = Slerp([start-1, stop], Rotation.from_matrix(rotations[[start-1, stop], j]))(ids).as_matrix()
    return trans, rotations, runs


def build_input(source, start=0, limit=0, max_gap=10):
    source = Path(source).resolve()
    summary = json.loads((source / 'summary.json').read_text())
    if summary['schema'] != 'pkdata.wilor.association.v1' or not summary['sam3_mask_gate_applied']:
        raise ValueError('Expected associated tracks with SAM3 hand masks applied')
    raw = []
    for side in ('left', 'right'):
        with np.load(source / f'{side}_track.npz', allow_pickle=False) as f:
            raw.append({k:f[k] for k in f.files})
    total = len(raw[0]['frame_index'])
    end = min(total, start+limit) if limit else total
    if start < 0 or end-start < 3:
        raise ValueError('Need >=3 consecutive frames')
    sl = slice(start, end)
    n = end-start
    assert np.array_equal(raw[0]['frame_index'], raw[1]['frame_index'])
    assert np.array_equal(raw[0]['K'], raw[1]['K'])
    K = raw[0]['K'].astype(np.float32)
    data, gaps = {}, {}
    rotations_all, translations, raw_rotations, raw_betas, validity = [], [], [], [], []
    for name, f in zip(('left', 'right'), raw):
        valid = f['valid'][sl].astype(bool)
        if valid.sum() < 3:
            raise ValueError(f'Too few valid observations for {name}')
        rot = np.concatenate([f['mano_global_orient_raw'][sl], f['mano_hand_pose_raw'][sl]], axis=1)
        trans = f['camera_translation_pinhole_m'][sl].copy()
        trans, rot_init, gap = interpolate_initialization(trans, rot, valid, max_gap)
        rotations_all.append(rot_init)
        raw_rotations.append(rot)
        translations.append(trans)
        raw_betas.append(f['mano_betas_raw'][sl])
        validity.append(valid)
        gaps[name] = [[s+start, e+start] for s,e in gap]
    valid = np.stack(validity)
    raw_betas = np.stack(raw_betas)
    shared_betas = np.stack([b[v].mean(0) for b,v in zip(raw_betas,valid)])
    rotations = np.stack(rotations_all)
    aa = Rotation.from_matrix(rotations.reshape(-1,3,3)).as_rotvec().reshape(2,n,16,3).astype(np.float32)
    trans = np.stack(translations).astype(np.float32)
    # Upstream run_mano mirrors the final X, INCLUDING translation. WiLoR's
    # saved pinhole translation is added AFTER mirroring local geometry.
    trans[0,:,0] *= -1
    data.update(init_trans=trans, init_root_orient=aa[:,:,0], init_body_pose=aa[:,:,1:],
                init_body_shape=np.broadcast_to(shared_betas[:,None],(2,n,10)).copy(),
                is_right=np.broadcast_to(np.array([[0],[1]],np.float32),(2,n)).copy(),
                vis_mask=np.where(valid,1,-1).astype(np.float32), observed_valid=valid,
                raw_mano_rotations=np.stack(raw_rotations).astype(np.float32),
                shared_betas_target=shared_betas.astype(np.float32),
                joints3d=np.stack([f['keypoints_cam_pinhole_m'][sl] for f in raw]).astype(np.float32))
    uv = np.stack([f['keypoints_2d_model_px'][sl] for f in raw]).astype(np.float32)
    scores = np.stack([f['score'][sl] for f in raw])
    conf = np.broadcast_to(np.where(valid,scores,0)[...,None,None],(2,n,21,1))
    data['joints2d'] = np.concatenate([uv,conf],axis=-1).astype(np.float32)
    # Invalid observations carry finite placeholders and zero confidence; only
    # observed_valid may decide whether they participate in data losses.
    for key in ('joints3d','joints2d','raw_mano_rotations'):
        data[key] = np.nan_to_num(data[key],nan=0.,posinf=0.,neginf=0.)
    assert all(np.isfinite(v).all() for v in data.values())
    cam = dict(cam_R=np.broadcast_to(np.eye(3,dtype=np.float32),(n,3,3)).copy(),
               cam_t=np.zeros((n,3),np.float32),
               intrins=np.broadcast_to(np.array([K[0,0],K[1,1],K[0,2],K[1,2]],np.float32),(n,4)).copy())
    meta = dict(source=str(source), total_available_frames=total, start_frame=start, end_frame_exclusive=end,
                frames=n, complete_clip=(start==0 and end==total), gaps=gaps,
                K=K.tolist(), coordinate_frame='OpenCV RGB camera-relative proxy, not a recovered world frame',
                camera_motion_estimated=False, sam3_mask_gate_applied=True,
                translation_convention='DynHaMR pre-mirror translation: negate left X relative to WiLoR pinhole translation')
    return data, cam, meta, raw


def make_model(n, device='cuda'):
    model = MANO(model_path=str(REPO/'_DATA/data/mano'), batch_size=2*n, pose2rot=True,
                 use_pca=False, flat_hand_mean=True).to(device)
    return model


def tensor_dict(data, device='cuda'):
    return {k:torch.as_tensor(v,device=device) for k,v in data.items()}


def validate_mano_mapping(model, raw, start, end):
    """Compare upstream MANO and mirrored translations against all source joints."""
    report = {}
    for side,f in enumerate(raw):
        present = f['association_present'][start:end]
        ids = np.flatnonzero(present)+start
        if not len(ids):
            raise ValueError('No source MANO observations')
        rot = np.concatenate([f['mano_global_orient_raw'][ids],f['mano_hand_pose_raw'][ids]],1)
        aa = Rotation.from_matrix(rot.reshape(-1,3,3)).as_rotvec().reshape(len(ids),48)
        args = dict(global_orient=aa[:,:3],hand_pose=aa[:,3:],betas=f['mano_betas_raw'][ids])
        with torch.no_grad():
            out = model(**{k:torch.as_tensor(v,dtype=torch.float32,device='cuda') for k,v in args.items()})
            local = out.joints.cpu().numpy()
        local[:,:,0] *= 2*side-1
        camera = local+f['camera_translation_pinhole_m'][ids,None]
        diff = float(np.abs(camera-f['keypoints_cam_pinhole_m'][ids]).max())
        if diff > 2e-5:
            raise ValueError(f'MANO conversion mismatch for side {side}: {diff} m')
        report[['left','right'][side]] = dict(compared_frames=len(ids),max_abs_joint_error_m=diff)
    # Also test the actual upstream run_mano pre-mirror translation path, with
    # a shared shape and >=2 frames (upstream helper rejects T=1).
    n=model.batch_size//2
    common=np.zeros((2,10),np.float32)
    zeros=np.zeros((2,n,3),np.float32)
    trans=zeros.copy();trans[:,:,0]=.12;trans[:,:,2]=.35
    sides=np.broadcast_to(np.array([[0],[1]],np.float32),(2,n)).copy()
    with torch.no_grad():
        a=run_mano(model,**tensor_dict(dict(trans=trans,root_orient=zeros,body_pose=np.zeros((2,n,45),np.float32),is_right=sides,betas=common)))
        b=run_mano(model,**tensor_dict(dict(trans=zeros,root_orient=zeros,body_pose=np.zeros((2,n,45),np.float32),is_right=sides,betas=common)))
        expected=trans.copy();expected[0,:,0]*=-1
        err=(a['joints']-b['joints']-torch.as_tensor(expected,device='cuda')[:,:,None]).abs().max().item()
    assert err < 2e-6, err
    report['upstream_translation_max_error_m']=err
    return report
