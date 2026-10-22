"""Ego2Robot A.1 association core; mask gating and Dyn-HaMR are separate stages."""
import numpy as np


def associate(frames):
    """Seed using best L+R score; propagate by nearest previous 2D wrist.

    Frames contain is_right, score and keypoints_2d_model_px. Return row indices
    [T,2] in left/right order. -1 means missing; no gap filling is performed.
    Each candidate votes for its nearest track; conflicts retain highest score.
    Across a missing frame, the last available wrist is held for association.
    """
    if not frames:
        raise ValueError('empty sequence')
    eligible = [np.isfinite(f['keypoints_2d_model_px'][:,0]).all(axis=1) & np.isfinite(f['score'])
                & f.get('sam3_mask_keep', np.ones(len(f['score']), dtype=bool)) for f in frames]
    seed, seed_score, seed_rows = None, -np.inf, None
    for t, (f, good) in enumerate(zip(frames, eligible)):
        rows=[]
        for side in (0,1):
            candidates=np.flatnonzero(good & (f['is_right']==side))
            if not len(candidates):
                break
            rows.append(int(candidates[np.argmax(f['score'][candidates])]))
        if len(rows)==2:
            score=float(f['score'][rows].sum())
            if score>seed_score:
                seed,seed_score,seed_rows=t,score,rows
    if seed is None:
        raise ValueError('no frame has two finite L/R seed candidates')
    selected=np.full((len(frames),2),-1,dtype=np.int32)
    selected[seed]=seed_rows
    for direction in (1,-1):
        previous=frames[seed]['keypoints_2d_model_px'][seed_rows,0].copy()
        for t in range(seed+direction,len(frames) if direction==1 else -1,direction):
            f=frames[t]
            ids=np.flatnonzero(eligible[t])
            if not len(ids):
                continue
            wrist=f['keypoints_2d_model_px'][ids,0]
            closest=np.argmin(np.linalg.norm(wrist[:,None,:]-previous[None,:,:],axis=-1),axis=1)
            for side in (0,1):
                candidates=ids[closest==side]
                if len(candidates):
                    best=int(candidates[np.argmax(f['score'][candidates])])
                    selected[t,side]=best
                    previous[side]=f['keypoints_2d_model_px'][best,0]
    return selected,int(seed),seed_score


def jump_mask(wrist, valid):
    """Paper's per-frame displacement rule; flag the later endpoint of a jump.

    Uses a single pass, original adjacent samples only. Returns mask, threshold,
    displacement per frame (first/invalid pairs are NaN). Does not interpolate.
    """
    wrist=np.asarray(wrist,dtype=np.float64)
    valid=np.asarray(valid,dtype=bool) & np.isfinite(wrist).all(axis=1)
    velocity=np.full(len(wrist),np.nan)
    pairs=valid[1:] & valid[:-1]
    indices=np.flatnonzero(pairs)+1
    velocity[indices]=np.linalg.norm(wrist[indices]-wrist[indices-1],axis=1)
    finite=velocity[np.isfinite(velocity)]
    threshold=max(4.*float(np.median(finite)),.003) if len(finite) else .003
    return np.isfinite(velocity)&(velocity>threshold),threshold,velocity
