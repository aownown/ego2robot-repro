"""Independent geometry/projection checks plus optional saved-run verification."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import argparse
import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import yaml
from PIL import Image
from scipy.spatial.transform import Rotation

from robot_scene import ROOT, CameraRenderer, add_camera, build_scene, contacts, geom_labels
from run_pipeline import ArmIK


def check_geometry():
    cfg = yaml.safe_load((ROOT / "configs/panda_debug.yaml").read_text())
    ik = ArmIK(cfg)
    q_known = ik.home.copy()
    q_known[:7] += [.1, -.1, .08, .06, -.05, .08, -.1]
    ik.configuration.update(q_known)
    data = ik.configuration.data
    target = np.eye(4)
    target[:3, 3] = data.site_xpos[ik.site_id]
    target[:3, :3] = data.site_xmat[ik.site_id].reshape(3, 3)
    hand = ik.model.body("left_hand").id
    R = data.xmat[hand].reshape(3, 3)
    # Independent physical conventions: approach points along the hand's +Z,
    # jaw axis along +Y, TCP at the symmetric fingertip pad center height.
    np.testing.assert_allclose(target[:3, :3][:, 0], R[:, 2], atol=1e-8)
    np.testing.assert_allclose(target[:3, :3][:, 2], R[:, 1], atol=1e-8)
    np.testing.assert_allclose(target[:3, 3], data.xpos[hand] + R @ [0, 0, .0584 + .0445], atol=1e-8)
    result = ik.solve(target, .06, ik.home)
    assert result[4], result[2:]
    np.testing.assert_allclose(result[1], target, atol=2e-5)
    np.testing.assert_allclose(result[0][-2:], [.03, .03], atol=1e-8)
    for joint in range(ik.model.njnt):
        value = result[0][ik.model.jnt_qposadr[joint]]
        lo, hi = ik.model.jnt_range[joint]
        assert lo - 1e-8 <= value <= hi + 1e-8
    clipped = ik.solve(target, .11, result[0])
    np.testing.assert_allclose(clipped[0][-2:], [.04, .04], atol=1e-8)
    # Check camera-from-base direction and left/right assembly against known FK.
    base = np.eye(4)
    base[:3, :3] = Rotation.from_euler("xyz", [20, 10, -30], degrees=True).as_matrix()
    base[:3, 3] = [.3, -.2, .5]
    model = mujoco.MjModel.from_xml_string(build_scene(cfg, {"left": base, "right": np.eye(4)}))
    d = mujoco.MjData(model)
    d.qpos[:] = np.tile(q_known, 2)
    mujoco.mj_forward(model, d)
    np.testing.assert_allclose(d.site("left_geometric_tcp").xpos, (base @ target)[:3, 3], atol=1e-8)
    np.testing.assert_allclose(d.site("right_geometric_tcp").xpos, target[:3, 3], atol=1e-8)
    labels = geom_labels(model)
    assert set(labels) == {1, 2, 3, 4}
    # Deliberately overlapping robots must be detected as cross-arm collisions.
    overlap = mujoco.MjModel.from_xml_string(build_scene(cfg, {s: np.eye(4) for s in ("left", "right")}))
    od = mujoco.MjData(overlap)
    od.qpos[:] = np.tile(ik.home, 2)
    assert contacts(overlap, od, geom_labels(overlap))["cross"] > 0
    separated = np.eye(4)
    separated[0, 3] = 3
    far = mujoco.MjModel.from_xml_string(build_scene(cfg, {"left": np.eye(4), "right": separated}))
    fd = mujoco.MjData(far)
    fd.qpos[:] = np.tile(ik.home, 2)
    assert contacts(far, fd, geom_labels(far))["cross"] == 0
    print("TCP axes, jaw limit, FK/IK round trip, base transform, dual labels and collision checks passed")


def check_camera():
    # Unequal focal lengths and off-center principal point expose axis/sign bugs.
    K = np.array([[400., 0, 335.], [0, 380., 171.], [0, 0, 1.]])
    root = ET.fromstring('<mujoco><visual><global offwidth="640" offheight="360"/><quality offsamples="0"/></visual><worldbody/></mujoco>')
    world = root.find("worldbody")
    add_camera(world, K, 640, 360)
    for p in [(0., 0., .5), (.1, .08, .5), (-.15, -.08, .8)]:
        ET.SubElement(world, "geom", type="sphere", size=".004", pos=" ".join(map(str, p)))
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with mujoco.Renderer(model, height=360, width=640) as renderer:
        renderer.update_scene(data, camera="rgb")
        renderer.enable_segmentation_rendering()
        seg = renderer.render()
        renderer.disable_segmentation_rendering()
        renderer.enable_depth_rendering()
        depth = renderer.render()
        for i, p in enumerate(model.geom_pos):
            y, x = np.where(seg[:, :, 0] == i)
            expected = (K @ p)[:2] / p[2]
            assert np.max(np.abs(expected - [x.mean(), y.mean()])) < 1
            assert abs(float(depth[y, x].min()) - (p[2] - .004)) < .001
    print("OpenCV pixel-center projection (<1px) and optical-Z depth (<1mm) checks passed")


def check_output(path):
    manifest = json.loads((path / "manifest.json").read_text())
    summary = json.loads((path / "summary.json").read_text())
    assert manifest["status"] == summary["status"] == "succeeded"
    n = summary["frames_processed"]
    w, h = manifest["width"], manifest["height"]
    with (path / "frame_mapping.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == n
    for side in ("left", "right"):
        with np.load(path / f"{side}_robot_trajectory.npz") as data:
            assert data["qpos"].shape == (n, 9) and np.isfinite(data["qpos"]).all()
            np.testing.assert_array_equal(data["rgb_header_timestamp_ns"], [int(r["rgb_header_timestamp_ns"]) for r in rows])
            np.testing.assert_allclose(data["applied_gripper_width_m"], data["qpos"][:, -2:].sum(1), atol=1e-8)
    for folder, ext in (("robot_rgb", "png"), ("robot_depth", "npz"), ("robot_mask", "png"),
                        ("gripper_mask", "png"), ("robot_parts", "png")):
        assert len(list((path / folder).glob("*." + ext))) == n
    for i in range(n):
        stem = f"{i:04d}"
        rgb = np.array(Image.open(path / "robot_rgb" / f"{stem}.png"))
        mask = np.array(Image.open(path / "robot_mask" / f"{stem}.png")) > 0
        gripper = np.array(Image.open(path / "gripper_mask" / f"{stem}.png")) > 0
        parts = np.array(Image.open(path / "robot_parts" / f"{stem}.png"))
        with np.load(path / "robot_depth" / f"{stem}.npz") as data:
            depth = data["depth_m"]
        assert rgb.shape == (h, w, 3) and depth.shape == mask.shape == (h, w)
        assert depth.dtype == np.float32
        np.testing.assert_array_equal(mask, parts > 0)
        np.testing.assert_array_equal(gripper, (parts == 2) | (parts == 4))
        assert np.all(np.isfinite(depth[mask]) & (depth[mask] > 0)) and np.isposinf(depth[~mask]).all()
    for name, width in (("robot_overlay.mp4", w), ("comparison.mp4", w * 2)):
        reader = imageio.get_reader(path / name)
        try:
            assert reader.count_frames() == n
            meta = reader.get_meta_data()
            assert meta["size"] == (width, h) and abs(meta["fps"] - manifest["fps"]) < .001
        finally:
            reader.close()
    print(f"Saved products: {n} frames, matching RGB/depth/masks/timestamps and exact video dimensions")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    check_geometry()
    check_camera()
    if args.output:
        check_output(args.output)
