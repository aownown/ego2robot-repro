"""Ego2Robot Eq. 3 / Appendix A.3 core, for one complete, finite hand track.

Input MANO/OpenPose-21 order: wrist=0, thumb/index/middle tips=4/8/12.
Coordinates must already be in the RGB camera frame and in meters.
Optional observed_valid marks short gaps to interpolate in target space.
Long gaps/home blending and robot TCP offsets require a chosen embodiment.
"""
import argparse
from pathlib import Path
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation, Slerp


def _slerp(q0, q1, alpha):
    return Slerp([0., 1.], Rotation.from_quat([q0, q1]))([alpha]).as_quat()[0]


def hemisphere_correct(quat):
    quat = np.array(quat, dtype=np.float64, copy=True)
    for t in range(1, len(quat)):
        if quat[t] @ quat[t - 1] < 0:
            quat[t] *= -1
    return quat


def interpolate_gaps(pos, quat, width, observed_valid, max_gap_frames=10):
    """Fill bounded short gaps; never invent a robot home pose or extrapolate."""
    valid = np.asarray(observed_valid, dtype=bool)
    if valid.shape != (len(pos),):
        raise ValueError("observed_valid must have shape [T]")
    if not valid[0] or not valid[-1]:
        raise ValueError("boundary observation gaps require an explicit policy; cannot extrapolate")
    missing = np.flatnonzero(~valid)
    if not len(missing):
        return pos.copy(), quat.copy(), width.copy()
    runs = np.split(missing, np.flatnonzero(np.diff(missing) != 1) + 1)
    if any(len(run) > max_gap_frames for run in runs):
        raise ValueError("observation gap exceeds 10 frames; robot home blending is not configured")
    good = np.flatnonzero(valid)
    pos, quat, width = pos.copy(), quat.copy(), width.copy()
    for axis in range(3):
        pos[missing, axis] = np.interp(missing, good, pos[good, axis])
    width[missing] = np.interp(missing, good, width[good])
    quat[missing] = Slerp(good, Rotation.from_quat(quat[good]))(missing).as_quat()
    return pos, hemisphere_correct(quat), width


def retarget(keypoints_cam_m, handedness, fps=30., observed_valid=None):
    k = np.asarray(keypoints_cam_m, dtype=np.float64)
    if k.ndim != 3 or k.shape[1:] != (21, 3) or len(k) == 0:
        raise ValueError("expected nonempty [T,21,3] keypoints")
    if not np.isfinite(k).all() or not np.isfinite(fps) or fps <= 0:
        raise ValueError("finite keypoints and positive fps required; resolve missing tracks first")
    if handedness not in ("left", "right"):
        raise ValueError("handedness must be left or right")
    valid = np.ones(len(k), dtype=bool) if observed_valid is None else np.asarray(observed_valid, dtype=bool)
    if valid.shape != (len(k),) or not valid[0] or not valid[-1]:
        raise ValueError("observed_valid must have shape [T] with observed endpoints")
    wrist, thumb = k[:, 0], k[:, 4]
    virtual = .7 * k[:, 8] + .3 * k[:, 12]
    pos = .5 * (thumb + virtual)
    width = np.linalg.norm(thumb - virtual, axis=-1)
    matrices, last = [], None
    held = np.zeros(len(k), dtype=bool)
    for t in range(len(k)):
        z = (1 if handedness == "right" else -1) * (thumb[t] - virtual[t]) / max(width[t], 1e-12)
        y = np.cross(z, virtual[t] - wrist[t])
        if width[t] < .01 or np.linalg.norm(y) < 1e-6:
            if last is None:
                raise ValueError("first frame has degenerate orientation; start at a valid frame")
            matrix = last
            held[t] = True
        else:
            y /= np.linalg.norm(y)
            matrix = np.column_stack((np.cross(y, z), y, z))
            if valid[t]:
                last = matrix
        matrices.append(matrix.copy())
    quat = hemisphere_correct(Rotation.from_matrix(matrices).as_quat())  # xyzw
    raw = dict(tcp_raw_cam_m=pos.copy(), rotation_raw_cam_gripper=np.asarray(matrices),
               quaternion_raw_xyzw=quat.copy(), gripper_width_raw_m=width.copy())
    # Exclude inferred observations from the filter context, then enforce linear
    # position/width + SLERP at these same short gaps again after smoothing.
    pos, quat, width = interpolate_gaps(pos, quat, width, valid)
    n = len(k)
    replaced = np.zeros(n, dtype=bool)
    for _ in range(2):
        if n < 3:
            break
        dp = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        dr = (Rotation.from_quat(quat[:-1]).inv() * Rotation.from_quat(quat[1:])).magnitude()
        bad = np.r_[False, (dp > max(5 * np.median(dp), .9 / fps)) |
                    (dr > max(5 * np.median(dr), 10. / fps))]
        good = np.flatnonzero(~bad)
        if len(good) < 2:
            raise ValueError("insufficient valid frames after velocity filtering")
        replaced |= bad
        for axis in range(3):
            pos[bad, axis] = np.interp(np.flatnonzero(bad), good, pos[good, axis])
        width[bad] = np.interp(np.flatnonzero(bad), good, width[good])
        # At a boundary, hold the nearest valid orientation (explicit extension).
        query = np.clip(np.flatnonzero(bad), good[0], good[-1])
        if len(query):
            quat[bad] = Slerp(good, Rotation.from_quat(quat[good]))(query).as_quat()
        quat = hemisphere_correct(quat)
    window = min(21, n if n % 2 else n - 1)
    pos = savgol_filter(pos, window, min(3, window - 1), axis=0)
    width = np.maximum(0., savgol_filter(width, window, min(3, window - 1)))
    smoothed = np.empty_like(quat)
    for t in range(n):
        ids = np.arange(max(0, t - 10), min(n, t + 11))
        weights = np.exp(-.5 * ((ids - t) / 10.) ** 2)
        q, total = quat[ids[0]].copy(), weights[0]
        # Deterministic incremental weighted SLERP; the paper does not specify
        # the reduction order for the Gaussian-weighted quaternion window.
        for idx, weight in zip(ids[1:], weights[1:]):
            q = _slerp(q, quat[idx], weight / (total + weight))
            total += weight
        smoothed[t] = q
    pos, smoothed, width = interpolate_gaps(pos, hemisphere_correct(smoothed), width, valid)
    return dict(tcp_cam_m=pos, rotation_cam_gripper=Rotation.from_quat(smoothed).as_matrix(),
                quaternion_xyzw=smoothed, gripper_width_m=width, replaced=replaced,
                orientation_held=held, gap_interpolated=~valid, **raw)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path, help="NPZ key: keypoints_cam_m [T,21,3]")
    p.add_argument("output", type=Path)
    p.add_argument("--hand", required=True, choices=["left", "right"])
    p.add_argument("--fps", type=float, default=30.)
    a = p.parse_args()
    with np.load(a.input, allow_pickle=False) as data:
        result = retarget(data["keypoints_cam_m"], a.hand, a.fps,
                          data["observed_valid"] if "observed_valid" in data else None)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("xb") as f:
        np.savez_compressed(f, **result, fps=a.fps, handedness=a.hand)


if __name__ == "__main__":
    main()
