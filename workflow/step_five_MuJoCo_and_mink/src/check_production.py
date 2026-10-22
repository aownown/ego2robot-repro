"""Regression tests for collision/visibility rejection and accepted final products."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import argparse
import csv
import json
from pathlib import Path
from unittest.mock import patch
import mink
import mujoco
import numpy as np
from PIL import Image
from collision_distance import DistanceOracle
from constrained_ik import CompactCollision, ConstrainedIK, FrameDisplacementLimit
from production_validation import visibility
from robot_scene import ROOT, contacts, geom_labels
from check_pipeline import check_geometry, check_camera, check_output


def check_collision_math():
    # An unreachable target across a wall must not cause the QP to cross the wall.
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="moving"><joint name="x" type="slide" axis="1 0 0" range="-1 1"/>
      <geom name="moving_geom" type="sphere" size=".04" mass="1"/><site name="tip"/></body>
      <body name="wall" mocap="true"><geom name="wall_geom" type="sphere" size=".05"/></body>
      </worldbody></mujoco>''')
    configuration = mink.Configuration(model)
    configuration.update(np.array([.2]))
    task = mink.FrameTask("tip", "site", position_cost=1., orientation_cost=0.)
    target = np.eye(4)
    task.set_target(mink.SE3.from_matrix(target))
    limit = CompactCollision(model, [(["moving_geom"], ["wall_geom"])],
                             minimum_distance_from_collisions=.003, collision_detection_distance=.3, gain=.7,
                             include_all_pairs=True)
    for _ in range(30):
        v = mink.solve_ik(configuration, [task], .02, solver="quadprog", limits=[mink.ConfigurationLimit(model), limit])
        configuration.integrate_inplace(v, .02)
        assert configuration.q[0] >= .093 - 1e-9, configuration.q
    np.testing.assert_allclose(configuration.q[0], .093, atol=1e-7)
    # Total inter-frame displacement stays bounded across many IK iterations.
    frame_limit = FrameDisplacementLimit(model, [0], .015)
    frame_limit.anchor = np.array([.2])
    configuration.update(np.array([.2]))
    target[0, 3] = .8
    task.set_target(mink.SE3.from_matrix(target))
    for _ in range(10):
        v = mink.solve_ik(configuration, [task], .02, solver="quadprog", limits=[frame_limit])
        configuration.integrate_inplace(v, .02)
    np.testing.assert_allclose(configuration.q[0], .215, atol=1e-8)
    print("QP clearance displacement units and cumulative frame-motion bounds passed")


def check_certificate():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body><geom name="a" type="box" size=".1 .1 .1"/></body>
      <body pos=".5 0 0"><geom name="b" type="box" size=".1 .1 .1"/></body>
      </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    oracle = DistanceOracle(model)
    def bad_zero(m, d, a, b, cap, points):
        points[:] = [.1, 0, 0, .4, 0, 0]
        return 0.
    # Inject the documented engine failure; the known box gap is exactly 0.3m.
    with patch.object(mujoco, "mj_geomDistance", bad_zero):
        distance, _, _, certificate = oracle.query(data, 0, 1, 1.)
        assert certificate
        np.testing.assert_allclose(distance, .3, atol=1e-8)
        # Keep the false witnesses but make the actual boxes overlap. The oracle
        # must refuse the witness-length shortcut and leave the pair rejected.
        model.body_pos[2] = [.1, 0, 0]
        mujoco.mj_forward(model, data)
        distance, _, _, certificate = oracle.query(data, 0, 1, 1.)
        assert not certificate and distance == 0.
    print("Separating-plane fallback certifies separation and refuses overlapping geometry")


def check_visibility_gate():
    cfg = {"single_arm_max_area_ratio": .25, "maximum_robot_area_ratio": .45,
           "maximum_workspace_arm_area_ratio": .10, "minimum_gripper_pixels": 60}
    labels = np.zeros((360, 640), np.uint8)
    labels[20:50, 20:50] = 2
    labels[20:50, 570:600] = 4
    box = [220, 90, 440, 220]
    assert visibility(labels, box, ("left", "right"), cfg)["accepted"]
    labels[50:350, :600] = 1
    assert not visibility(labels, box, ("left", "right"), cfg)["accepted"]
    old = ROOT / "outputs/panda_full_20260911_174352/robot_parts/0076.png"
    if old.exists():
        old_labels = np.array(Image.open(old))
        failure = visibility(old_labels, box, ("left", "right"), cfg)
        assert not failure["accepted"] and "robot_area" in failure["reasons"]
    print("Visibility admission rejects the original frame-76 screen obstruction")


def check_accepted_run(path):
    check_output(path)
    manifest = json.loads((path / "manifest.json").read_text())
    cfg = manifest["config"]
    assert manifest["robot_geometry_accepted"] and manifest["full_paper_candidate_grid"]
    cache = Path(manifest["search_cache"])
    for side in ("left", "right"):
        with (cache / f"{side}_candidates.jsonl").open() as f:
            candidate_ids = [json.loads(line)["candidate"] for line in f]
        assert len(candidate_ids) == len(set(candidate_ids))
    verification = json.loads((path / "validation/pair_verification.json").read_text())
    assert len(verification) == 25, "Expected all top-5 x top-5 pairs"
    with (path / "frame_quality.csv").open() as f:
        quality = list(csv.DictReader(f))
    trajectories = [np.load(path / f"{side}_robot_trajectory.npz") for side in ("left", "right")]
    q_full = np.concatenate([d["qpos"] for d in trajectories], axis=1)
    ik = ConstrainedIK(cfg, ("left", "right"))
    bases = {s: np.array(manifest["selected_camera_from_base"][s]) for s in ik.sides}
    ik.set_bases(bases)
    actual_model = mujoco.MjModel.from_xml_path(str(path / "scene.xml"))
    data = mujoco.MjData(actual_model)
    labels = geom_labels(actual_model)
    for i, q in enumerate(q_full):
        ik.configuration.update(q)
        clear = ik.clearance()
        assert min(clear) >= cfg["collision_clearance_m"] - 1e-6, (i, clear)
        data.qpos[:] = q
        assert not any(contacts(actual_model, data, labels).values()), i
        if i:
            assert np.max(np.abs(q[ik.joint_ids] - q_full[i - 1, ik.joint_ids])) <= cfg["maximum_joint_step_rad"] + 1e-8
            for f in (.25, .5, .75):
                ik.configuration.update((1-f)*q_full[i-1]+f*q)
                assert min(ik.clearance()) >= 0, (i, f)
        parts = np.array(Image.open(path / "robot_parts" / f"{i:04d}.png"))
        v = visibility(parts, manifest["workspace_box"], ik.sides, cfg)
        assert v["accepted"], (i, v)
        np.testing.assert_allclose(v["robot_area_ratio"], float(quality[i]["robot_area_ratio"]), atol=1e-8)
    for d in trajectories:
        d.close()
    print(f"Accepted production run: all {len(q_full)} frames independently pass contacts, clearance, motion and image gates")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    check_geometry()
    check_camera()
    check_collision_math()
    check_certificate()
    check_visibility_gate()
    if args.output:
        check_accepted_run(args.output.resolve())
