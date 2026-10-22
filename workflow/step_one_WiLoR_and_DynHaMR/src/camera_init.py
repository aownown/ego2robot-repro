"""Translation-only initialization with calibrated K and predicted image points.

Pinhole approximation: lens distortion is NOT applied. These are model-derived
observations, not independently measured keypoints or metric ground truth.
"""
import numpy as np
from scipy.optimize import least_squares


def crop_to_pixels(keypoints_crop, center, size, is_right):
    xy = np.asarray(keypoints_crop, dtype=np.float64).copy()
    xy[:, 0] *= 1 if is_right else -1
    return xy * float(size) + np.asarray(center)


def project(points_cam, K):
    p = np.asarray(points_cam) @ np.asarray(K).T
    return p[:, :2] / p[:, 2:3]


def fit_translation(points_local, pixels, K):
    """Fit t with fixed MANO pose/shape; returns t, pixel RMSE, numerical status."""
    xyz = np.asarray(points_local, dtype=np.float64)
    uv = np.asarray(pixels, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    if xyz.shape != (21, 3) or uv.shape != (21, 2) or K.shape != (3, 3):
        raise ValueError('expected 21 XYZ/UV points and 3x3 K')
    if not all(np.isfinite(x).all() for x in (xyz, uv, K)):
        return np.full(3, np.nan), np.nan, False
    rays = np.column_stack((uv, np.ones(len(uv)))) @ np.linalg.inv(K).T
    rays = rays[:, :2] / rays[:, 2:3]
    A = np.zeros((42, 3))
    A[0::2, 0] = 1; A[1::2, 1] = 1
    A[0::2, 2] = -rays[:, 0]; A[1::2, 2] = -rays[:, 1]
    rhs = (rays * xyz[:, 2:3] - xyz[:, :2]).reshape(-1)
    initial, _, rank, _ = np.linalg.lstsq(A, rhs, rcond=None)
    if rank < 3:
        return np.full(3, np.nan), np.nan, False
    zmin = max(1e-5, -float(xyz[:, 2].min()) + 1e-5)
    initial[2] = max(initial[2], zmin + .01)
    result = least_squares(lambda t: (project(xyz + t, K) - uv).ravel(),
                           initial, bounds=([-np.inf, -np.inf, zmin], np.inf), max_nfev=100)
    valid = bool(result.success and np.isfinite(result.x).all() and
                 np.all((xyz + result.x)[:, 2] > 0))
    rmse = float(np.sqrt(np.mean(np.sum((project(xyz + result.x, K) - uv) ** 2, axis=1))))
    if not valid:
        return np.full(3, np.nan), rmse, False
    return result.x, rmse, valid
