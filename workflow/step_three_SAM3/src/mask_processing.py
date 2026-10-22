"""Ego2Robot A.4 mask postprocessing; no model or GPU dependency.

The paper does not specify the mask interpolation representation. This adapter
linearly interpolates signed distance fields and thresholds at zero.
"""
import cv2
import numpy as np


def chunk_ranges(n, size=400, overlap=50):
    if n < 1 or size < 2 or not 0 <= overlap < size:
        raise ValueError("require n>=1, chunk size>=2 and 0<=overlap<size")
    chunks, start = [], 0
    while True:
        end = min(start + size, n)
        chunks.append((start, end))
        if end == n:
            return chunks
        start = end - overlap


def signed_distance(mask):
    mask = np.asarray(mask, dtype=np.uint8)
    # Zero padding supplies an outside boundary even for a full foreground mask.
    inside = cv2.distanceTransform(np.pad(mask, 1), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
    return (inside
            - cv2.distanceTransform(1 - mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE))


def postprocess_masks(raw, max_gap=3, median_window=11, area_ratio=.5, close_size=5):
    raw = np.asarray(raw)
    if raw.ndim != 3 or raw.shape[0] < 1 or raw.dtype != np.bool_:
        raise ValueError("expected nonempty boolean [T,H,W] masks")
    if max_gap < 0 or median_window < 1 or median_window % 2 != 1 or not 0 < area_ratio < 1:
        raise ValueError("invalid gap/median/area parameters")
    if close_size < 1 or close_size % 2 != 1:
        raise ValueError("closing kernel must have positive odd size")
    n = len(raw)
    masks = raw.copy()
    raw_area = raw.sum(axis=(1, 2), dtype=np.int64)
    gap_filled = np.zeros(n, dtype=bool)
    missing = np.flatnonzero(raw_area == 0)
    runs = np.split(missing, np.flatnonzero(np.diff(missing) != 1) + 1) if len(missing) else []
    for run in runs:
        a, b = int(run[0]) - 1, int(run[-1]) + 1
        if len(run) > max_gap or a < 0 or b >= n:
            continue
        # Endpoints are nonempty by construction.
        da, db = signed_distance(raw[a]), signed_distance(raw[b])
        for t in run:
            alpha = (int(t) - a) / (b - a)
            masks[t] = ((1 - alpha) * da + alpha * db) > 0
            gap_filled[t] = True
    gap_area = masks.sum(axis=(1, 2), dtype=np.int64)
    radius = median_window // 2
    local_median = np.array([np.median(gap_area[max(0, t-radius):min(n, t+radius+1)]) for t in range(n)])
    small = gap_area < area_ratio * local_median
    good = np.flatnonzero((gap_area > 0) & ~small)
    replacement = np.full(n, -1, dtype=np.int64)
    for t in np.flatnonzero(small):
        if not len(good):
            break
        # Non-cascading replacement; equal temporal distances choose earlier.
        nearest = good[np.argmin(np.abs(good - t))]
        masks[t] = masks[nearest]
        replacement[t] = nearest
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    close_changed = np.zeros(n, dtype=bool)
    for t in range(n):
        closed = cv2.morphologyEx(masks[t].astype(np.uint8), cv2.MORPH_CLOSE, kernel) != 0
        close_changed[t] = np.any(closed != masks[t])
        masks[t] = closed
    return masks, dict(raw_area_px=raw_area, area_after_gap_px=gap_area,
                       final_area_px=masks.sum(axis=(1, 2), dtype=np.int64),
                       gap_interpolated=gap_filled, local_area_median_px=local_median,
                       small_area_replaced=replacement >= 0, replacement_source_row=replacement,
                       close_changed=close_changed)
