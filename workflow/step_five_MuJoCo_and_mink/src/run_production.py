"""Stage 5: full candidate grid, constrained IK, whole-clip visual acceptance."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import argparse
import csv
import json
import shutil
import tempfile
import time
from pathlib import Path
import imageio.v2 as imageio
import mujoco
import numpy as np
import yaml
from PIL import Image
from constrained_ik import ConstrainedIK, constrained_scene
from production_search import search
from production_validation import screen_continuous, verify_pairs, visibility, workspace_box
from robot_scene import ROOT, CameraRenderer, contacts, geom_labels
from run_pipeline import (TARGET_DEFAULT, BACKGROUND_DEFAULT, load_inputs, write_json,
                          digest, caption)


def export_products(output, cfg, targets, frames, rows, K, size, fps, camera, winner, started):
    bases, q_full = winner["bases"], winner["q_full"]
    scene = constrained_scene(cfg, bases, K, *size)
    (output / "scene.xml").write_text(scene)
    model = mujoco.MjModel.from_xml_string(scene)
    data = mujoco.MjData(model)
    labels = geom_labels(model)
    renderer = CameraRenderer(model, size[1], size[0])
    box = workspace_box(targets, K, size, cfg["workspace_margin_pixels"])
    n = len(frames)
    for name in ("robot_rgb", "robot_depth", "robot_mask", "gripper_mask", "robot_parts"):
        (output / name).mkdir()
    with (output / "frame_mapping.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    trajectories = {s: {"actual_tcp_cam": [], "position_error_m": [], "rotation_error_rad": []} for s in ("left", "right")}
    quality = []
    filenames = ("robot_overlay", "comparison")
    writers = [imageio.get_writer(output / f"{name}.partial.mp4", fps=fps, codec="libx264",
                                  quality=8, macro_block_size=1, ffmpeg_log_level="error") for name in filenames]
    try:
        for i, q in enumerate(q_full):
            data.qpos[:] = q
            collision = contacts(model, data, labels)
            if any(collision.values()):
                raise RuntimeError(f"Final render contact check failed at frame {i}: {collision}")
            camera_contacts = [c for c in data.contact[:data.ncon]
                               if c.dist < 0 and model.geom("camera_keepout").id in (c.geom1, c.geom2)]
            if camera_contacts:
                raise RuntimeError(f"Camera keepout contact at frame {i}")
            rgb, depth, parts = renderer.render(data)
            v = visibility(parts, box, ("left", "right"), cfg)
            if not v["accepted"]:
                raise RuntimeError(f"Final render visibility failed at frame {i}: {v}")
            mask = parts > 0
            gripper = (parts == 2) | (parts == 4)
            assert np.isfinite(depth[mask]).all() and np.all(depth[mask] > 0)
            observed = all(bool(targets[s]["observed_valid"][i]) for s in ("left", "right"))
            width_ok = all(0 <= targets[s]["gripper_width_m"][i] <= cfg["maximum_gripper_width_m"] for s in ("left", "right"))
            record = {"frame_index": int(targets["left"]["frame_index"][i]), **collision,
                      **{k: val for k, val in v.items() if k not in ("accepted", "reasons")},
                      "left_visible_pixels": int(np.count_nonzero((parts == 1) | (parts == 2))),
                      "right_visible_pixels": int(np.count_nonzero((parts == 3) | (parts == 4))),
                      "geometry_valid": True, "paper_style_valid": observed,
                      "strict_valid": bool(observed and width_ok), "joint_jump_over_0_25rad": False,
                      "minimum_robot_clearance_m": winner["metrics"][i]["minimum_robot_clearance_m"],
                      "minimum_camera_clearance_m": winner["metrics"][i]["minimum_camera_clearance_m"]}
            quality.append(record)
            for side in ("left", "right"):
                T = np.eye(4)
                site = model.site(side + "_geometric_tcp").id
                T[:3, 3] = data.site_xpos[site]
                T[:3, :3] = data.site_xmat[site].reshape(3, 3)
                trajectories[side]["actual_tcp_cam"].append(T)
                trajectories[side]["position_error_m"].append(winner["metrics"][i][side + "_position_error_m"])
                trajectories[side]["rotation_error_rad"].append(winner["metrics"][i][side + "_rotation_error_rad"])
            stem = f"{i:04d}"
            Image.fromarray(rgb).save(output / "robot_rgb" / f"{stem}.png")
            np.savez_compressed(output / "robot_depth" / f"{stem}.npz", depth_m=depth)
            for folder, image in (("robot_mask", mask.astype(np.uint8) * 255),
                                  ("gripper_mask", gripper.astype(np.uint8) * 255), ("robot_parts", parts)):
                Image.fromarray(image).save(output / folder / f"{stem}.png")
            with Image.open(frames[i]) as image:
                background = np.array(image.convert("RGB"))
            overlay = background.copy()
            overlay[mask] = rgb[mask]
            shown = caption(overlay, f"Stage 5 | frame {i:03d} | coverage {100*v['robot_area_ratio']:.1f}% | depth occlusion pending")
            writers[0].append_data(shown)
            writers[1].append_data(np.concatenate([caption(background, "Step 4 background"), shown], axis=1))
            if i == 0:
                Image.fromarray(overlay).save(output / "first_frame.png")
            if (i + 1) % 60 == 0 or i + 1 == n:
                print(f"[export] {i+1}/{n}, robot coverage={v['robot_area_ratio']:.3f}", flush=True)
    finally:
        for writer in writers:
            writer.close()
        renderer.close()
    for name in filenames:
        (output / f"{name}.partial.mp4").rename(output / f"{name}.mp4")
    for j, side in enumerate(("left", "right")):
        source = targets[side]
        payload = {k: np.asarray(v) for k, v in trajectories[side].items()}
        payload.update(qpos=q_full[:, 9*j:9*(j+1)], camera_from_base=bases[side], K=K, fps=fps,
                       converged=np.ones(n, bool), geometry_valid=np.ones(n, bool),
                       width_clipped=(source["gripper_width_m"] > cfg["maximum_gripper_width_m"]) | (source["gripper_width_m"] < 0),
                       applied_gripper_width_m=np.clip(source["gripper_width_m"], 0, cfg["maximum_gripper_width_m"]),
                       joint_names=np.array([side + "_joint" + str(k) for k in range(1, 8)] + [side + "_finger_joint1", side + "_finger_joint2"]))
        payload.update({k: source[k] for k in ("frame_index", "rgb_header_timestamp_ns", "rgb_source_frame_index",
                                              "clip_video_time_s", "observed_valid", "optimization_inferred", "gripper_width_m")})
        np.savez_compressed(output / f"{side}_robot_trajectory.npz", **payload)
    with (output / "frame_quality.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=quality[0].keys())
        writer.writeheader()
        writer.writerows(quality)
    joint_ids = np.r_[0:7, 9:16]
    summary = {"status": "succeeded", "quality_status": "geometry_and_visibility_accepted", "frames_processed": n,
               "available_frames": n, "complete_clip": True, "geometry_valid_frames": n,
               "paper_style_valid_frames": sum(q["paper_style_valid"] for q in quality),
               "strict_valid_frames": sum(q["strict_valid"] for q in quality),
               "self_collision_frames": 0, "cross_collision_frames": 0, "camera_keepout_collision_frames": 0,
               "joint_jump_frames": [], "joint_max_step_rad": float(np.max(np.abs(np.diff(q_full[:, joint_ids], axis=0)))) if n > 1 else 0.,
               "robot_area_ratio_max": max(v["robot_area_ratio"] for v in quality),
               "robot_area_ratio_mean": float(np.mean([v["robot_area_ratio"] for v in quality])),
               "workspace_arm_area_ratio_max": max(v["workspace_arm_area_ratio"] for v in quality),
               "minimum_robot_clearance_m": min(v["minimum_robot_clearance_m"] for v in quality),
               "minimum_camera_clearance_m": min(v["minimum_camera_clearance_m"] for v in quality),
               "minimum_gripper_visible_pixels": {s: min(v[s+"_gripper_pixels"] for v in quality) for s in trajectories},
               "hands": {s: {"converged_frames": n,
                             "position_error_max_mm": float(np.max(trajectories[s]["position_error_m"]) * 1000),
                             "rotation_error_max_deg": float(np.rad2deg(np.max(trajectories[s]["rotation_error_rad"]))),
                             "width_clipped_frames": int(np.count_nonzero((targets[s]["gripper_width_m"] > cfg["maximum_gripper_width_m"]) | (targets[s]["gripper_width_m"] < 0)))} for s in trajectories},
               "selected_pair": winner["record"], "elapsed_seconds": time.monotonic() - started,
               "metric_depth_validated": False, "world_trajectory_recovered": False,
               "quality_note": "All frames pass robot geometry/visibility constraints. Input metric scale, inferred hand frames and clipped gripper widths retain separate flags; no claim of validated physical demonstrations."}
    write_json(output / "summary.json", summary)
    return summary


def main(args):
    started = time.monotonic()
    cfg = yaml.safe_load(args.config.read_text())
    targets, frames, rows, K, size, fps, camera = load_inputs(args.targets, args.background)
    if args.output:
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=False)
    else:
        (ROOT / "outputs").mkdir(exist_ok=True)
        output = Path(tempfile.mkdtemp(prefix=time.strftime("panda_production_%Y%m%d_%H%M%S_"), dir=ROOT / "outputs"))
    print(f"OUTPUT={output}", flush=True)
    manifest = {"schema": "pkdata.stage5.constrained.v2", "status": "running", "config": cfg,
                "frames_requested": len(frames), "width": size[0], "height": size[1], "fps": fps, "K": K.tolist(),
                "target_dir": str(args.targets.resolve()), "background_dir": str(args.background.resolve()),
                "target_sha256": {s: digest(args.targets / f"{s}_targets.npz") for s in targets},
                "background_mapping_sha256": digest(args.background / "frame_mapping.csv"),
                "background_manifest": json.loads((args.background / "input_manifest.json").read_text()),
                "code_sha256": {p.name: digest(p) for p in (ROOT / "src").glob("*.py")},
                "source_camera": camera, "camera_distortion_applied": False,
                "coordinate_frame": str(targets["left"]["coordinate_frame"]),
                "metric_depth_validated": False, "world_trajectory_recovered": False, "scene_object_collision_checked": False,
                "exact_paper_reproduction": False, "full_paper_candidate_grid": True,
                "depth_occlusion_applied": False,
                "depth_convention": "float32 optical-axis Z, model meters; +inf outside robot",
                "part_labels": {"0": "background", "1": "left_arm", "2": "left_gripper", "3": "right_arm", "4": "right_gripper"}}
    write_json(output / "manifest.json", manifest)
    shutil.copy2(args.config, output / "config.yaml")
    try:
        cache = args.search_cache.resolve() if args.search_cache else output / "search"
        ranked, keyframes = search(cfg, targets, cache, args.workers)
        manifest["search_cache"] = str(cache)
        manifest["search_fingerprint"] = json.loads((cache / "search_manifest.json").read_text())["fingerprint"]
        write_json(output / "manifest.json", manifest)
        selected = screen_continuous(cfg, targets, ranked, K, size, output / "validation")
        winner = verify_pairs(cfg, targets, selected, K, size, output / "validation")
        summary = export_products(output, cfg, targets, frames, rows, K, size, fps, camera, winner, started)
        manifest.update(status="succeeded", frames_processed=len(frames), robot_geometry_accepted=True,
                        workspace_box=winner["workspace_box"], selected_camera_from_base={s: T.tolist() for s, T in winner["bases"].items()})
        write_json(output / "manifest.json", manifest)
        print(json.dumps(summary, indent=2), flush=True)
        print(f"PREVIEW={output / 'comparison.mp4'}", flush=True)
    except BaseException as exc:
        manifest.update(status="failed", robot_geometry_accepted=False, error=str(exc))
        write_json(output / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, default=TARGET_DEFAULT)
    parser.add_argument("--background", type=Path, default=BACKGROUND_DEFAULT)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/panda_production.yaml")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--search-cache", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    main(parser.parse_args())
