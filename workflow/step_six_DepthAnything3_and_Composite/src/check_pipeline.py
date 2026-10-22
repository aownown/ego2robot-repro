"""Independent pixel/data checks; --output verifies every saved result frame."""
import argparse
from pathlib import Path
import cv2
import numpy as np
from composite import composite
from metric_depth import convert_depth
from pipeline_io import (load_inputs, load_frame, read_json, rows, check_identity,
                         mask, rgb, probe, digest, write_json)


def regression_checks():
    # Distinct arm, near-scene gripper, far-scene gripper and protected gripper.
    wh = (32, 24)
    bg = np.full((24, 32, 3), [20, 30, 40], np.uint8)
    robot = np.full_like(bg, [120, 130, 140])
    rm = np.zeros((24, 32), bool)
    for y, x in ((2, 2), (2, 20), (20, 2), (20, 20)):
        rm[y, x] = True
    gm = rm.copy(); gm[2, 2] = False
    scene = np.full((24, 32), .2, np.float32)
    scene[20, 2] = .5
    sim = np.full_like(scene, np.inf); sim[rm] = .3
    human = np.zeros_like(rm); human[20, 18] = True  # 5x5 dilation reaches x=20
    out, visible = composite(bg, robot, scene, sim, rm, gm, human)
    expected = np.zeros_like(rm); expected[2, 2] = expected[20, 2] = expected[20, 20] = True
    assert np.array_equal(visible, expected)
    assert np.array_equal(out[~expected], bg[~expected])
    # An invalid scene pixel matters only where a comparison is required.
    scene[0, 0] = np.nan
    composite(bg, robot, scene, sim, rm, gm, human)
    scene[2, 20] = np.nan
    try:
        composite(bg, robot, scene, sim, rm, gm, human)
    except ValueError:
        pass
    else:
        raise AssertionError("Missing gripper depth was accepted")
    # Known physical conversion: raw=2, processed focal=450 => Z=3m, not 6m.
    K_net = np.array([[400., 0, 3], [0, 500., 2], [0, 0, 1]])
    depth, focal = convert_depth(np.full((4, 6), 2.), K_net, wh)
    assert depth.shape == (24, 32) and np.allclose(depth, 3.) and focal == 450.
    print("Occlusion/protected edge, background preservation, invalid depth and metric conversion checks passed.")


def check_output(output):
    output = Path(output).resolve()
    manifest = read_json(output / "manifest.json")
    if manifest["status"] != "succeeded":
        raise ValueError("Cannot accept an incomplete run")
    inputs = load_inputs(manifest["robot_dir"], manifest["clip"])
    if digest(inputs["robot"] / "manifest.json") != manifest["source_manifest_sha256"]:
        raise ValueError("Source stage-5 manifest changed")
    selected = manifest["frame_indices"]
    mapping = rows(output / "frame_mapping.csv")
    quality = rows(output / "frame_quality.csv")
    hashes = read_json(output / "source_hashes.json")
    if not len(selected) == len(mapping) == len(quality) == len(hashes) == manifest["frames_processed"]:
        raise ValueError("Frame count mismatch")
    wh = inputs["output_wh"]
    hidden_counts = []
    for i, row, record, source in zip(selected, mapping, quality, hashes):
        frame = inputs["frames"][i]
        check_identity(row, frame["row"])
        assert int(record["frame_index"]) == source["frame_index"] == i
        for key, file in frame["files"].items():
            if digest(file) != source["sha256"][key]:
                raise ValueError(f"Source changed: {file}")
        _, bg, robot, sim, rm, gm, human, parts = load_frame(inputs, frame)
        stem = f"{i:04d}"
        result = rgb(output / "frames" / f"{stem}.png", wh)
        visible = mask(output / "visible_robot_mask" / f"{stem}.png", wh)
        hidden = mask(output / "occlusion_mask" / f"{stem}.png", wh)
        with np.load(output / "scene_depth" / f"{stem}.npz", allow_pickle=False) as npz:
            scene = npz["depth_m"]
            assert scene.shape == wh[::-1] and scene.dtype == np.float32
            assert np.isfinite(scene).all() and (scene > 0).all()
            assert int(npz["frame_index"]) == i
            assert int(npz["rgb_header_timestamp_ns"]) == int(row["rgb_header_timestamp_ns"])
            assert np.allclose(npz["K"], inputs["K_output"])
            wnet, hnet = npz["network_wh"]
            kn = npz["K_network"]
            expected_k = inputs["K_original"].copy()
            expected_k[0] *= wnet / inputs["original_wh"][0]
            expected_k[1] *= hnet / inputs["original_wh"][1]
            assert np.allclose(kn, expected_k, atol=1e-4)
            assert abs(float(npz["focal_network_px"]) - (kn[0, 0] + kn[1, 1]) / 2) < 1e-4
        protected = cv2.dilate(human.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1) > 0
        expected_hidden = gm & ~protected & (scene < sim)
        expected_visible = rm & ~expected_hidden
        assert np.array_equal(hidden, expected_hidden)
        assert np.array_equal(visible, expected_visible)
        assert np.array_equal(result[visible], robot[visible])
        assert np.array_equal(result[~visible], bg[~visible])
        assert not np.any(hidden & (rm & ~gm))
        assert visible.mean() <= float(frame["stage5_quality"]["robot_area_ratio"]) + 1e-10
        assert int(record["occluded_gripper_pixels"]) == hidden.sum()
        for side, label in (("left", 2), ("right", 4)):
            assert int(record[side + "_visible_gripper_pixels"]) == ((parts == label) & visible).sum()
        for key in ("geometry_valid", "paper_style_valid", "strict_valid"):
            assert record["stage5_" + key] == frame["stage5_quality"][key]
        hidden_counts.append(int(hidden.sum()))
    for folder, suffix in (("frames", ".png"), ("scene_depth", ".npz"),
                            ("visible_robot_mask", ".png"), ("occlusion_mask", ".png")):
        assert {p.name for p in (output / folder).glob("*" + suffix)} == {f"{i:04d}{suffix}" for i in selected}
    shapes = {"composite": wh, "comparison": (wh[0] * 2, wh[1] + 32),
              "scene_depth_preview": (wh[0], wh[1] + 32), "occlusion_overlay": (wh[0], wh[1] + 32)}
    videos = {k: probe(output / f"{k}.mp4", len(selected), manifest["video_fps"], size) for k, size in shapes.items()}
    summary = read_json(output / "summary.json")
    assert summary["frames_with_gripper_occlusion"] == sum(c > 0 for c in hidden_counts)
    assert summary["complete_clip"] == (selected == list(range(len(inputs["frames"]))))
    result = {"status": "passed", "frames_verified": len(selected),
              "source_hashes_verified": True, "identities_and_timestamps_verified": True,
              "occlusion_and_background_pixels_verified": True, "video_streams": videos,
              "physical_scale_validated": False, "visual_quality_accepted": False}
    write_json(output / "validation.json", result)
    print(f"Verified all {len(selected)} frames, source hashes, camera grids, pixels, flags and 4 video streams: {output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    regression_checks()
    if args.output:
        check_output(args.output)
