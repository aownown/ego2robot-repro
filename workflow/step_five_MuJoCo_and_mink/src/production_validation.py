"""Continuous trajectory and rendered-visibility admission gates."""
import csv
import json
from pathlib import Path
import mujoco
import numpy as np
from constrained_ik import ConstrainedIK
from robot_scene import CameraRenderer
from run_pipeline import target_pose, write_json


def workspace_box(targets, K, size, margin):
    xyz = np.concatenate([targets[s]["tcp_cam_m"] for s in ("left", "right")])
    uv = (xyz @ K.T)[:, :2] / xyz[:, 2:3]
    lo = np.floor(uv.min(0) - margin).astype(int)
    hi = np.ceil(uv.max(0) + margin).astype(int)
    lo = np.maximum(lo, [0, 0])
    hi = np.minimum(hi, size)
    if np.any(hi <= lo):
        raise ValueError("No visible workspace from the supplied camera-relative targets")
    return [int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1])]


def visibility(parts, box, sides, cfg, pixel_scale=1.):
    x0, y0, x1, y1 = box
    arm = (parts == 1) | (parts == 3)
    roi_arm = float(arm[y0:y1, x0:x1].mean())
    area = float((parts > 0).mean())
    grippers = {s: int(np.count_nonzero(parts == (2 if s == "left" else 4))) for s in sides}
    area_limit = cfg["single_arm_max_area_ratio"] if len(sides) == 1 else cfg["maximum_robot_area_ratio"]
    reasons = []
    if area > area_limit:
        reasons.append("robot_area")
    if roi_arm > cfg["maximum_workspace_arm_area_ratio"]:
        reasons.append("workspace_arm_occlusion")
    if any(v < cfg["minimum_gripper_pixels"] * pixel_scale ** 2 for v in grippers.values()):
        reasons.append("gripper_not_visible")
    return {"accepted": not reasons, "reasons": reasons, "robot_area_ratio": area,
            "workspace_arm_area_ratio": roi_arm, **{s + "_gripper_pixels": n for s, n in grippers.items()}}


def screen_continuous(cfg, targets, ranked, K, size, output):
    """Take top 5 only AFTER every frame passes constrained IK and visibility."""
    output.mkdir(exist_ok=True, parents=True)
    scale = cfg["search_render_scale"]
    small_size = tuple(round(d * scale) for d in size)
    small_K = K.copy()
    small_K[:2] *= scale
    box = workspace_box(targets, small_K, small_size, cfg["workspace_margin_pixels"] * scale)
    selected = {}
    evaluations = []
    n = len(targets["left"]["frame_index"])
    for side in ("left", "right"):
        ik = ConstrainedIK(cfg, (side,), small_K, small_size)
        renderer = CameraRenderer(ik.model, small_size[1], small_size[0])
        selected[side] = []
        try:
            for rank, record in enumerate(ranked[side]):
                base = np.array(record["camera_from_base"])
                ik.set_bases({side: base})
                q = np.array(record["q_keyframes"][0])
                reference = np.array(record["reference_q"])
                full, area, roi, gp = [], [], [], []
                reason = None
                # Reject a visually bad sparse posture before expensive full replay.
                for frame_q in record["q_keyframes"]:
                    ik.configuration.update(np.array(frame_q))
                    mujoco.mj_forward(ik.model, ik.configuration.data)
                    v = visibility(renderer.render_parts(ik.configuration.data), box, (side,), cfg, scale)
                    if not v["accepted"]:
                        reason = "keyframe_render:" + ",".join(v["reasons"])
                        break
                if reason is None:
                    for frame in range(n):
                        step = ik.solve({side: target_pose(targets[side], frame)},
                                        {side: targets[side]["gripper_width_m"][frame]},
                                        q, reference=reference, continuous=frame > 0)
                        if not step["accepted"]:
                            reason = f"continuous_ik_or_clearance:{frame}"
                            break
                        q = step["q"].copy()
                        mujoco.mj_forward(ik.model, ik.configuration.data)
                        v = visibility(renderer.render_parts(ik.configuration.data), box, (side,), cfg, scale)
                        if not v["accepted"]:
                            reason = f"continuous_render:{frame}:" + ",".join(v["reasons"])
                            break
                        full.append(q)
                        area.append(v["robot_area_ratio"])
                        roi.append(v["workspace_arm_area_ratio"])
                        gp.append(v[side + "_gripper_pixels"])
                evaluation = {"side": side, "rank": rank, "candidate": record["candidate"],
                              "passed_frames": len(full), "accepted": reason is None, "reject_reason": reason}
                evaluations.append(evaluation)
                if reason is None:
                    entry = dict(record)
                    entry.update(q_full=np.array(full), full_max_area=max(area), full_max_workspace_arm=max(roi),
                                 full_min_gripper_pixels=min(gp), original_rank=rank)
                    selected[side].append(entry)
                    np.savez_compressed(output / f"{side}_candidate_{record['candidate']}.npz",
                                        qpos=np.array(full), camera_from_base=base,
                                        reference_q=reference, robot_area_ratio=area, workspace_arm_area_ratio=roi)
                    print(f"[{side} continuous] candidate={record['candidate']} accepted {n}/{n}, max area={max(area):.3f}; admitted={len(selected[side])}/5", flush=True)
                elif rank % 25 == 0:
                    print(f"[{side} continuous] rank={rank}, rejected={reason}", flush=True)
                if len(selected[side]) == cfg["top_candidates_per_arm"]:
                    break
        finally:
            renderer.close()
        write_json(output / "continuous_screening.json", evaluations)
        print(f"[{side}] full-trajectory/visibility qualified: {len(selected[side])}", flush=True)
    if any(not selected[s] for s in selected):
        raise RuntimeError("No admissible continuous visible trajectory for one or both arms; no success output produced")
    return selected


def verify_pairs(cfg, targets, selected, K, size, output):
    """Recompute every selected pair at native output resolution, for all frames."""
    output.mkdir(exist_ok=True, parents=True)
    box = workspace_box(targets, K, size, cfg["workspace_margin_pixels"])
    ik = ConstrainedIK(cfg, ("left", "right"), K, size)
    renderer = CameraRenderer(ik.model, size[1], size[0])
    n = len(targets["left"]["frame_index"])
    results, winner = [], None
    try:
        for li, left in enumerate(selected["left"]):
            for ri, right in enumerate(selected["right"]):
                bases = {"left": np.array(left["camera_from_base"]), "right": np.array(right["camera_from_base"])}
                ik.set_bases(bases)
                q_full = np.concatenate([left["q_full"], right["q_full"]], axis=1)
                metrics, reason = [], None
                for frame, q in enumerate(q_full):
                    ik.configuration.update(q)
                    clear = ik.clearance()
                    if min(clear) < cfg["collision_clearance_m"] - 1e-6:
                        reason = f"clearance:{frame}"
                        break
                    if frame:
                        if np.max(np.abs(q[ik.joint_ids] - q_full[frame - 1, ik.joint_ids])) > cfg["maximum_joint_step_rad"] + 1e-8:
                            reason = f"joint_step:{frame}"
                            break
                        # Additional midpoint checks catch between-frame crossings.
                        for fraction in (.25, .5, .75):
                            ik.configuration.update((1 - fraction) * q_full[frame - 1] + fraction * q)
                            if min(ik.clearance()) < 0:
                                reason = f"between_frame_collision:{frame}"
                                break
                        ik.configuration.update(q)
                        if reason:
                            break
                    mujoco.mj_forward(ik.model, ik.configuration.data)
                    # Independently recompute full-pose residuals on the combined model.
                    pe, ae = [], []
                    from scipy.spatial.transform import Rotation
                    for side, site in zip(ik.sides, ik.site_ids):
                        T = target_pose(targets[side], frame)
                        pe.append(np.linalg.norm(ik.configuration.data.site_xpos[site] - T[:3, 3]))
                        ae.append(Rotation.from_matrix(ik.configuration.data.site_xmat[site].reshape(3, 3).T @ T[:3, :3]).magnitude())
                    if max(pe + ae) >= cfg["ik"]["tolerance"]:
                        reason = f"dual_fk_error:{frame}"
                        break
                    v = visibility(renderer.render_parts(ik.configuration.data), box, ik.sides, cfg)
                    if not v["accepted"]:
                        reason = f"render:{frame}:" + ",".join(v["reasons"])
                        break
                    metrics.append({"frame_index": frame, **v, "minimum_robot_clearance_m": clear[0],
                                    "minimum_camera_clearance_m": clear[1],
                                    "left_position_error_m": float(pe[0]), "right_position_error_m": float(pe[1]),
                                    "left_rotation_error_rad": float(ae[0]), "right_rotation_error_rad": float(ae[1])})
                record = {"left_rank": li, "right_rank": ri, "left_candidate": left["candidate"],
                          "right_candidate": right["candidate"], "accepted": reason is None,
                          "passed_frames": len(metrics), "reject_reason": reason,
                          "paper_score_sum": left["score"] + right["score"]}
                if reason is None:
                    record.update(max_robot_area_ratio=max(v["robot_area_ratio"] for v in metrics),
                                  mean_robot_area_ratio=float(np.mean([v["robot_area_ratio"] for v in metrics])),
                                  max_workspace_arm_area_ratio=max(v["workspace_arm_area_ratio"] for v in metrics))
                    # All hard constraints already hold; visual footprint breaks ties
                    # within the independently ranked paper top-five candidates.
                    quality = (record["max_robot_area_ratio"], record["mean_robot_area_ratio"], -record["paper_score_sum"])
                    if winner is None or quality < winner["quality"]:
                        winner = {"quality": quality, "record": record, "bases": bases, "q_full": q_full,
                                  "metrics": metrics, "workspace_box": box}
                results.append(record)
                write_json(output / "pair_verification.json", results)
                print(f"[pair {li},{ri}] {len(metrics)}/{n} accepted={reason is None} reason={reason}", flush=True)
    finally:
        renderer.close()
    if winner is None:
        raise RuntimeError("All dual-arm combinations failed continuous collision/visibility acceptance; no success output produced")
    np.savez_compressed(output / "accepted_trajectory.npz", qpos=winner["q_full"],
                        left_camera_from_base=winner["bases"]["left"], right_camera_from_base=winner["bases"]["right"])
    write_json(output / "accepted_pair.json", winner["record"])
    return winner
