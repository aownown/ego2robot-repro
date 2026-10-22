"""CPU-only checks of chunking, postprocessing, frame alignment and video writing.

Synthetic masks are deliberately used here. This does not validate SAM3 inference
or claim to measure segmentation accuracy on the user's clip.
"""
import json
import tempfile
from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np

from mask_processing import chunk_ranges, postprocess_masks
from segment_video import build_previews, prepare_input, union_instances, STEP


def main():
    # The 450-frame clip must retain a 50-frame overlap and all original indices.
    assert chunk_ranges(450) == [(0,400), (350,450)]
    for n in (1, 90, 400, 401, 700, 750, 751, 1101):
        chunks = chunk_ranges(n)
        coverage = np.zeros(n, dtype=int)
        for a, b in chunks:
            assert 0 <= a < b <= n and b-a <= 400
            coverage[a:b] += 1
        assert np.all(coverage > 0)
        assert max(coverage) <= 2
        for (_, b), (a, _) in zip(chunks, chunks[1:]):
            assert b-a == 50

    # Union keeps disconnected arms, and genuine empty predictions stay empty.
    parts = np.zeros((2, 64, 96), dtype=bool)
    parts[0, 10:30, 5:20] = True
    parts[1, 20:45, 65:80] = True
    merged, _ = union_instances(dict(out_binary_masks=parts, out_obj_ids=np.array([2,7]), out_probs=np.array([.8,.9])), (64,96))
    assert merged.sum() == parts.sum()
    empty, _ = union_instances(dict(out_binary_masks=parts[:0], out_obj_ids=np.array([], dtype=int), out_probs=np.array([])), (64,96))
    assert not empty.any()

    # A moving rectangle should interpolate near its temporal midpoint.
    moving = np.zeros((3,64,96), dtype=bool)
    moving[0,15:35,20:40] = True
    moving[2,15:35,24:44] = True
    fixed, stats = postprocess_masks(moving, close_size=1)
    assert stats["gap_interpolated"].tolist() == [False,True,False]
    np.testing.assert_allclose(np.nonzero(fixed[1])[1].mean(), 31.5, atol=.6)
    assert fixed.dtype == bool
    np.testing.assert_array_equal(fixed[[0,2]], moving[[0,2]])

    # Small-area replacement reads a fixed set of good neighbors; tie -> earlier.
    base = np.repeat(parts.any(axis=0)[None], 15, axis=0)
    damaged = base.copy()
    damaged[7] = False
    damaged[7,20,10] = True
    repaired, stats = postprocess_masks(damaged, close_size=1)
    assert stats["replacement_source_row"][7] == 6
    np.testing.assert_array_equal(repaired[7], base[7])
    damaged = base.copy()
    damaged[5:8] = False
    repaired, stats = postprocess_masks(damaged, close_size=1)
    assert stats["gap_interpolated"].sum() == 3
    np.testing.assert_array_equal(repaired, base)

    # No hallucinated masks when the whole clip is empty; long gaps stay marked.
    none, stats = postprocess_masks(np.zeros_like(base))
    assert not none.any() and not stats["gap_interpolated"].any()
    long_gap = base.copy()
    long_gap[2:13] = False
    _, stats = postprocess_masks(long_gap, close_size=1)
    assert not stats["gap_interpolated"].any()
    assert stats["final_area_px"][7] == 0
    punctured = base.copy()
    punctured[:,19,10] = False
    closed, _ = postprocess_masks(punctured)
    assert closed[:,19,10].all()

    # Verify the real RGB clip contract without GPU inference.
    clip, rows, _, count, shape, fps = prepare_input(Namespace(clip=STEP.parent.parent / "test_data", start=0, limit=3))
    assert count == 450 and shape == (720,1280) and fps == 30
    assert [int(r["rgb_source_frame_index"]) for r in rows] == [180,181,182]

    # Exercise both video encoders with synthetic images/masks in a temporary
    # directory. These files are removed and never labeled as model predictions.
    with tempfile.TemporaryDirectory(prefix="sam3_pipeline_cpu_check_") as root:
        root = Path(root)
        (root / "preview_samples").mkdir()
        rows = []
        for t in range(3):
            name = f"test_{t}.jpg"
            image = np.full((360,640,3), 70 + t*10, dtype=np.uint8)
            assert cv2.imwrite(str(root / name), image)
            rows.append(dict(rgb_frame_index=str(t), rgb_file=name))
        masks = np.zeros((3,360,640), dtype=bool)
        masks[:,100:280,70:170] = True
        masks[:,120:320,450:550] = True
        flags = dict(gap_interpolated=np.zeros(3,bool), small_area_replaced=np.zeros(3,bool))
        probes = build_previews(root, rows, root, masks, flags, 30)
        assert all(int(probe["nb_read_frames"]) == 3 for probe in probes.values())
    print("CPU pipeline checks passed: chunks, union, gaps, area repair, closing, real RGB alignment, both video encoders")
    print("SAM3 inference and real segmentation quality have NOT been tested by this check.")


if __name__ == "__main__":
    main()
