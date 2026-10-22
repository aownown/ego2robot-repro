"""Local stage-5 runner: candidate search, two-arm IK, aligned render products."""
import argparse
import csv
import hashlib
import itertools
import json
import shutil
import tempfile
import time
from pathlib import Path

import imageio.v2 as imageio
import mink
import mujoco
import numpy as np
import yaml
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

from robot_scene import ROOT, CameraRenderer, build_scene, contacts, geom_labels

TARGET_DEFAULT = ROOT.parent / "step_two_Retarget_and_Smooth/outputs/retarget/run_20260911_114939_41iJgn"
BACKGROUND_DEFAULT = ROOT.parent / "step_four_ProPainter/outputs/inpaint_full_arm_shadow_20260911_170624"
SIDES = ("left", "right")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_inputs(target_dir, background_dir):
    targets = {}
    for side in SIDES:
        with np.load(target_dir / f"{side}_targets.npz", allow_pickle=False) as data:
            targets[side] = {k: data[k].copy() for k in data.files}
    left, right = (targets[s] for s in SIDES)
    count = len(left["frame_index"])
    for side in SIDES:
        d = targets[side]
        assert str(d["length_unit"]) == "meter", "Expected stage-2 meter-labeled proxy"
        assert d["tcp_cam_m"].shape == (count, 3)
        assert d["rotation_cam_gripper"].shape == (count, 3, 3)
        assert d["gripper_width_m"].shape == (count,)
        assert np.isfinite(d["tcp_cam_m"]).all() and np.isfinite(d["gripper_width_m"]).all()
        rotation = d["rotation_cam_gripper"]
        assert np.isfinite(rotation).all()
        np.testing.assert_allclose(rotation @ rotation.transpose(0, 2, 1),
                                   np.broadcast_to(np.eye(3), rotation.shape), atol=1e-6)
        np.testing.assert_allclose(np.linalg.det(rotation), 1, atol=1e-6)
        for k in ("frame_index", "rgb_header_timestamp_ns", "rgb_source_frame_index", "K", "fps"):
            np.testing.assert_array_equal(d[k], left[k])
    with (background_dir / "frame_mapping.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == count, "Stage-2/background frame count mismatch"
    frames = []
    size = None
    for i, row in enumerate(rows):
        for field, key in (("rgb_frame_index", "frame_index"),
                           ("rgb_source_frame_index", "rgb_source_frame_index"),
                           ("rgb_header_timestamp_ns", "rgb_header_timestamp_ns")):
            assert int(row[field]) == int(left[key][i]), (i, field, "alignment mismatch")
        assert int(row["output_index"]) == i
        path = background_dir / "rgb/frames" / row["output_filename"]
        with Image.open(path) as image:
            if size is None:
                size = image.size
            assert image.size == size, "Background dimensions vary"
        frames.append(path)
    camera = json.loads((target_dir / "camera_parameters.json").read_text())
    K = left["K"].copy()
    K[0] *= size[0] / camera["rgb"]["width"]
    K[1] *= size[1] / camera["rgb"]["height"]
    return targets, frames, rows, K, size, float(left["fps"]), camera


class ArmIK:
    def __init__(self, config):
        self.model = mujoco.MjModel.from_xml_string(build_scene(config, {"left": np.eye(4)}))
        self.configuration = mink.Configuration(self.model)
        self.task = mink.FrameTask("left_geometric_tcp", "site", position_cost=1.,
                                   orientation_cost=1., lm_damping=1e-3)
        self.limits = [mink.ConfigurationLimit(self.model)]
        self.site_id = self.model.site("left_geometric_tcp").id
        self.home = np.array(config["home_qpos"], dtype=float)
        self.labels = geom_labels(self.model)
        self.tolerance = float(config["ik"]["tolerance"])
        self.iterations = int(config["ik"]["iterations"])
        self.max_width = float(config["maximum_gripper_width_m"])

    def solve(self, target, width, seed):
        q = np.array(seed, dtype=float, copy=True)
        q[-2:] = np.clip(width, 0, self.max_width) / 2
        self.configuration.update(q)
        self.task.set_target(mink.SE3.from_matrix(target))
        qp_failed = False
        iterations = 0
        for iterations in range(self.iterations):
            error = self.task.compute_error(self.configuration)
            if np.linalg.norm(error[:3]) < self.tolerance and np.linalg.norm(error[3:]) < self.tolerance:
                break
            try:
                velocity = mink.solve_ik(self.configuration, [self.task], .02,
                                         solver="quadprog", damping=1e-3,
                                         limits=self.limits, safety_break=True)
            except mink.NoSolutionFound:
                qp_failed = True
                break
            self.configuration.integrate_inplace(velocity, .02)
        q = self.configuration.q.copy()
        # Finger targets are prescribed explicitly, not solved from the TCP task.
        q[-2:] = np.clip(width, 0, self.max_width) / 2
        self.configuration.update(q)
        data = self.configuration.data
        actual = np.eye(4)
        actual[:3, 3] = data.site_xpos[self.site_id]
        actual[:3, :3] = data.site_xmat[self.site_id].reshape(3, 3)
        position_error = float(np.linalg.norm(actual[:3, 3] - target[:3, 3]))
        angle_error = float(Rotation.from_matrix(actual[:3, :3].T @ target[:3, :3]).magnitude())
        converged = position_error < self.tolerance and angle_error < self.tolerance and not qp_failed
        return q, actual, position_error, angle_error, converged, iterations + 1, qp_failed


def target_pose(data, index):
    T = np.eye(4)
    T[:3, 3] = data["tcp_cam_m"][index]
    T[:3, :3] = data["rotation_cam_gripper"][index]
    return T


def candidates(data, side, config):
    radius = config["search_radius_m"]
    center = np.mean(data["tcp_cam_m"], axis=0)
    keys = ("lateral_reach_multipliers", "forward_reach_multipliers", "vertical_reach_multipliers",
            "pitch_degrees", "yaw_degrees", "roll_degrees")
    for lat, forward, vertical, pitch, yaw, roll in itertools.product(*(config[k] for k in keys)):
        base = np.eye(4)
        base[:3, 3] = center + radius * np.array([lat * (-1 if side == "left" else 1), vertical, forward])
        distance = np.linalg.norm(data["tcp_cam_m"] - base[:3, 3], axis=1)
        if (np.linalg.norm(base[:3, 3]) < config["minimum_camera_distance_m"] or
                np.max(distance) > radius * config["maximum_trajectory_reach_ratio"] or
                np.min(distance) < config["minimum_trajectory_base_distance_m"]):
            continue
        # Camera +Y points down. Base +Z initially points up, +X inward.
        x = center - base[:3, 3]
        x[1] = 0
        x /= np.linalg.norm(x)
        z = np.array([0., -1., 0.])
        neutral = np.column_stack([x, np.cross(z, x), z])
        base[:3, :3] = neutral @ Rotation.from_euler("xyz", [roll, pitch, yaw], degrees=True).as_matrix()
        yield base, [lat, forward, vertical, pitch, yaw, roll], distance


def search_arm(ik, data, side, config, keyframes):
    options = list(candidates(data, side, config))
    if not options:
        raise ValueError(f"No surviving base candidates for {side}")
    records = []
    print(f"[{side}] search {len(options)} candidates x {len(keyframes)} keyframes", flush=True)
    for index, (base, grid, distance) in enumerate(options):
        inv = np.linalg.inv(base)
        q = ik.home.copy()
        qs, success, self_contacts = [], [], []
        for frame in keyframes:
            q, _, pe, ae, ok, _, _ = ik.solve(inv @ target_pose(data, frame), data["gripper_width_m"][frame], q)
            collision = contacts(ik.model, ik.configuration.data, ik.labels,
                                 config["collision_penetration_epsilon_m"])["left_self"]
            qs.append(q.copy())
            success.append(ok)
            self_contacts.append(collision)
        fraction = float(np.mean(success))
        reach_ratio = float(np.mean(distance[keyframes]) / config["search_radius_m"])
        record = {"candidate": index, "grid": grid, "camera_from_base": base.tolist(),
                  "feasibility_rate": fraction, "mean_reach_ratio": reach_ratio,
                  "score": fraction - 5 * abs(reach_ratio - .65),
                  "self_collision_keyframes": int(np.count_nonzero(self_contacts)),
                  "q_keyframes": np.array(qs), "success_keyframes": np.array(success)}
        records.append(record)
        if (index + 1) % 12 == 0 or index + 1 == len(options):
            best = max(records, key=lambda r: r["score"])
            print(f"[{side}] {index+1}/{len(options)}, best FR={best['feasibility_rate']:.3f}, score={best['score']:.3f}", flush=True)
    records.sort(key=lambda r: (-r["score"], r["self_collision_keyframes"], r["candidate"]))
    return records


def choose_pair(records, config, K, size, keyframes):
    # Reuse one compiled dual model; fixed base body transforms change per pair.
    model = mujoco.MjModel.from_xml_string(build_scene(config, {s: np.eye(4) for s in SIDES}, K, *size))
    data = mujoco.MjData(model)
    labels = geom_labels(model)
    pairs = []
    top = config["top_candidates_per_arm"]
    from robot_scene import quat_wxyz
    for li, ri in itertools.product(range(min(top, len(records["left"]))), range(min(top, len(records["right"])))):
        chosen = {"left": records["left"][li], "right": records["right"][ri]}
        for side, record in chosen.items():
            T = np.array(record["camera_from_base"])
            body = model.body(side + "_link0")
            model.body_pos[body.id] = T[:3, 3]
            model.body_quat[body.id] = quat_wxyz(T[:3, :3])
        mujoco.mj_setConst(model, data)
        valid, penetration = [], []
        for k in range(len(keyframes)):
            data.qpos[:] = np.concatenate([chosen[s]["q_keyframes"][k] for s in SIDES])
            c = contacts(model, data, labels, config["collision_penetration_epsilon_m"])
            ok = all(chosen[s]["success_keyframes"][k] for s in SIDES)
            valid.append(ok and c["left_self"] == 0 and c["right_self"] == 0 and c["cross"] <= 1)
            penetration.append(sum(c.values()))
        pairs.append({"left_rank": li, "right_rank": ri, "valid_keyframes": int(sum(valid)),
                      "penetrating_contacts": int(sum(penetration)),
                      "individual_score_sum": sum(chosen[s]["score"] for s in SIDES)})
    pairs.sort(key=lambda p: (-p["valid_keyframes"], p["penetrating_contacts"], -p["individual_score_sum"]))
    best = pairs[0]
    selected = {s: records[s][best[s + "_rank"]] for s in SIDES}
    print(f"[pair] best {best['valid_keyframes']}/{len(keyframes)} IK/collision-valid keyframes; {len(pairs)} pairs checked", flush=True)
    return selected, pairs


def caption(frame, text):
    # In-video label explains that this is a stage-5 overlay without scene occlusion.
    result = Image.fromarray(frame)
    draw = ImageDraw.Draw(result)
    draw.rectangle((0, 0, frame.shape[1], 22), fill=(15, 15, 15))
    draw.text((8, 5), text, fill=(255, 255, 255))
    return np.asarray(result)


def run(args):
    started = time.monotonic()
    config = yaml.safe_load(args.config.read_text())
    targets, frames, rows, K, size, fps, source_camera = load_inputs(args.targets, args.background)
    count = len(frames) if args.max_frames is None else min(args.max_frames, len(frames))
    if count <= 0:
        raise ValueError("max-frames must be positive")
    if args.output:
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=False)
    else:
        (ROOT / "outputs").mkdir(exist_ok=True)
        output = Path(tempfile.mkdtemp(prefix=time.strftime("panda_debug_%Y%m%d_%H%M%S_"), dir=ROOT / "outputs"))
    print(f"OUTPUT={output}", flush=True)
    for folder in ("robot_rgb", "robot_depth", "robot_mask", "gripper_mask", "robot_parts"):
        (output / folder).mkdir()
    manifest = {"schema": "pkdata.stage5.panda_debug.v1", "status": "running", "frames_requested": count,
                "target_dir": str(args.targets.resolve()), "background_dir": str(args.background.resolve()),
                "target_sha256": {s: digest(args.targets / f"{s}_targets.npz") for s in SIDES},
                "background_mapping_sha256": digest(args.background / "frame_mapping.csv"),
                "background_manifest": json.loads((args.background / "input_manifest.json").read_text()),
                "code_sha256": {p.name: digest(p) for p in (Path(__file__), ROOT / "src/robot_scene.py")},
                "config": config, "coordinate_frame": str(targets["left"]["coordinate_frame"]),
                "metric_depth_validated": False, "world_trajectory_recovered": False,
                "exact_paper_reproduction": False, "scene_object_collision_checked": False,
                "depth_occlusion_applied": False, "camera_distortion_applied": False,
                "width": size[0], "height": size[1], "fps": fps, "K": K.tolist(),
                "source_camera": source_camera,
                "depth_convention": "float32 optical-axis Z, model meters; +inf outside robot",
                "part_labels": {"0": "background", "1": "left_arm", "2": "left_gripper", "3": "right_arm", "4": "right_gripper"}}
    write_json(output / "manifest.json", manifest)
    shutil.copy2(args.config, output / "config.yaml")
    with (output / "frame_mapping.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows[:count])
    ik = ArmIK(config)
    # Search always covers the FULL input trajectory, even for a short preview.
    keyframes = np.unique(np.linspace(0, len(frames) - 1, min(config["keyframes_max"], len(frames)), dtype=int))
    records = {s: search_arm(ik, targets[s], s, config, keyframes) for s in SIDES}
    selected, pairs = choose_pair(records, config, K, size, keyframes)
    serial_records = {s: [{k: v for k, v in record.items() if k not in ("q_keyframes", "success_keyframes")}
                          for record in records[s]] for s in SIDES}
    bases = {s: np.array(selected[s]["camera_from_base"]) for s in SIDES}
    write_json(output / "base_search.json", {"keyframes": keyframes.tolist(), "candidates": serial_records,
               "pairs": pairs, "selected_camera_from_base": {s: bases[s].tolist() for s in SIDES},
               "note": "Reduced local grid; all frames used for distance filtering. Pair check uses full pose convergence and penetrating robot contacts."})
    scene = build_scene(config, bases, K, *size)
    (output / "scene.xml").write_text(scene)
    model = mujoco.MjModel.from_xml_string(scene)
    data = mujoco.MjData(model)
    labels = geom_labels(model)
    assert model.nq == 18 and all(model.joint(s + "_joint1").qposadr[0] == 9 * i for i, s in enumerate(SIDES))
    renderer = CameraRenderer(model, size[1], size[0])
    inverse = {s: np.linalg.inv(bases[s]) for s in SIDES}
    previous = {s: selected[s]["q_keyframes"][0].copy() for s in SIDES}
    arrays = {s: {"qpos": [], "actual_tcp_cam": [], "position_error_m": [], "rotation_error_rad": [],
                   "converged": [], "iterations": [], "qp_failed": [], "width_clipped": []} for s in SIDES}
    diagnostics = []
    video = imageio.get_writer(output / "robot_overlay.mp4", fps=fps, codec="libx264", quality=8,
                               macro_block_size=1, ffmpeg_log_level="error")
    comparison = imageio.get_writer(output / "comparison.mp4", fps=fps, codec="libx264", quality=8,
                                    macro_block_size=1, ffmpeg_log_level="error")
    try:
        for i in range(count):
            jump = False
            for side in SIDES:
                target = inverse[side] @ target_pose(targets[side], i)
                width = targets[side]["gripper_width_m"][i]
                result = ik.solve(target, width, previous[side])
                if not result[4]:
                    # Deterministic alternative seed only when warm-start fails.
                    retry = ik.solve(target, width, ik.home)
                    if (not result[4] and retry[4]) or (retry[2] + .1 * retry[3] < result[2] + .1 * result[3]):
                        result = retry
                q, actual, pe, ae, ok, iterations, qp_failed = result
                if not np.isfinite(q).all():
                    raise RuntimeError(f"Non-finite IK at {side} frame {i}")
                jump |= bool(np.max(np.abs(q[:7] - previous[side][:7])) > .25 and i > 0)
                previous[side] = q.copy()
                values = [q, bases[side] @ actual, pe, ae, ok, iterations, qp_failed,
                          bool(width < 0 or width > config["maximum_gripper_width_m"])]
                for key, value in zip(arrays[side], values):
                    arrays[side][key].append(value)
            data.qpos[:] = np.concatenate([previous[s] for s in SIDES])
            collision = contacts(model, data, labels, config["collision_penetration_epsilon_m"])
            robot, depth, parts = renderer.render(data)
            mask = parts > 0
            gripper = (parts == 2) | (parts == 4)
            assert np.all(np.isfinite(depth[mask]) & (depth[mask] > 0))
            visible = {s: int(np.count_nonzero((parts == (1 + 2 * j)) | (parts == (2 + 2 * j)))) for j, s in enumerate(SIDES)}
            ratio = float(mask.mean())
            observed = all(bool(targets[s]["observed_valid"][i]) for s in SIDES)
            track_ok = all(arrays[s]["position_error_m"][-1] < config["tracking_position_limit_m"] for s in SIDES)
            collision_ok = collision["left_self"] == 0 and collision["right_self"] == 0 and collision["cross"] <= 1
            basic_valid = bool(observed and track_ok and collision_ok and min(visible.values()) > 0 and ratio <= .7)
            strict_valid = basic_valid and all(arrays[s]["rotation_error_rad"][-1] < np.deg2rad(config["tracking_rotation_limit_deg"])
                                               and not arrays[s]["width_clipped"][-1] for s in SIDES)
            diagnostics.append({"frame_index": int(targets["left"]["frame_index"][i]), **collision,
                                "left_visible_pixels": visible["left"], "right_visible_pixels": visible["right"],
                                "robot_area_ratio": ratio, "paper_style_valid": basic_valid,
                                "strict_valid": bool(strict_valid), "joint_jump_over_0_25rad": bool(jump)})
            stem = f"{i:04d}"
            Image.fromarray(robot).save(output / "robot_rgb" / f"{stem}.png")
            np.savez_compressed(output / "robot_depth" / f"{stem}.npz", depth_m=depth)
            Image.fromarray(mask.astype(np.uint8) * 255).save(output / "robot_mask" / f"{stem}.png")
            Image.fromarray(gripper.astype(np.uint8) * 255).save(output / "gripper_mask" / f"{stem}.png")
            Image.fromarray(parts).save(output / "robot_parts" / f"{stem}.png")
            background = np.array(Image.open(frames[i]).convert("RGB"))
            overlay = background.copy()
            overlay[mask] = robot[mask]
            shown = caption(overlay, f"Stage 5 | frame {i:03d} | no scene depth occlusion | valid={int(basic_valid)}")
            video.append_data(shown)
            comparison.append_data(np.concatenate([caption(background, "Step 4 background"), shown], axis=1))
            if i == 0:
                Image.fromarray(overlay).save(output / "first_frame.png")
            if (i + 1) % 30 == 0 or i + 1 == count:
                print(f"[render] {i+1}/{count}; L error={arrays['left']['position_error_m'][-1]*1000:.2f}mm; R error={arrays['right']['position_error_m'][-1]*1000:.2f}mm", flush=True)
    finally:
        video.close()
        comparison.close()
        renderer.close()
    for side in SIDES:
        result = {k: np.asarray(v) for k, v in arrays[side].items()}
        result.update({k: targets[side][k][:count] for k in ("frame_index", "rgb_header_timestamp_ns", "rgb_source_frame_index",
                        "clip_video_time_s", "observed_valid", "optimization_inferred", "gripper_width_m")})
        result.update(camera_from_base=bases[side], K=K, fps=fps,
                      applied_gripper_width_m=np.clip(targets[side]["gripper_width_m"][:count], 0, config["maximum_gripper_width_m"]),
                      joint_names=np.array([side + "_joint" + str(j) for j in range(1, 8)] + [side + "_finger_joint1", side + "_finger_joint2"]))
        np.savez_compressed(output / f"{side}_robot_trajectory.npz", **result)
    with (output / "frame_quality.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=diagnostics[0].keys())
        writer.writeheader()
        writer.writerows(diagnostics)
    summary = {"status": "succeeded", "frames_processed": count, "available_frames": len(frames),
               "complete_clip": count == len(frames), "elapsed_seconds": time.monotonic() - started,
               "paper_style_valid_frames": sum(d["paper_style_valid"] for d in diagnostics),
               "strict_valid_frames": sum(d["strict_valid"] for d in diagnostics),
               "joint_jump_frames": [d["frame_index"] for d in diagnostics if d["joint_jump_over_0_25rad"]],
               "self_collision_frames": sum(d["left_self"] > 0 or d["right_self"] > 0 for d in diagnostics),
               "cross_collision_frames": sum(d["cross"] > 0 for d in diagnostics),
               "hands": {s: {"converged_frames": int(sum(arrays[s]["converged"])),
                              "position_error_max_mm": float(np.max(arrays[s]["position_error_m"]) * 1000),
                              "rotation_error_max_deg": float(np.rad2deg(np.max(arrays[s]["rotation_error_rad"]))),
                              "width_clipped_frames": int(sum(arrays[s]["width_clipped"]))} for s in SIDES},
               "quality_note": "Successful execution is not quality acceptance. Frames are retained even when quality flags fail; no stability erosion/L2/L3 filtering. See manifest for proxy scale and camera assumptions."}
    write_json(output / "summary.json", summary)
    manifest.update(status="succeeded", frames_processed=count)
    write_json(output / "manifest.json", manifest)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"PREVIEW={output / 'comparison.mp4'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, default=TARGET_DEFAULT)
    parser.add_argument("--background", type=Path, default=BACKGROUND_DEFAULT)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/panda_debug.yaml")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-frames", type=int)
    run(parser.parse_args())
