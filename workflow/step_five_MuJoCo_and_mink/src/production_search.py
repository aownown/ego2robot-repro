"""Full-grid constrained search with resumable, fingerprinted candidate records."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing
import time
from pathlib import Path
import numpy as np
import yaml
from constrained_ik import ConstrainedIK
from robot_scene import ROOT
from run_pipeline import load_inputs, candidates, target_pose, TARGET_DEFAULT, BACKGROUND_DEFAULT, write_json

WORKER = None


def initialize_worker(config, side, data, keyframes):
    global WORKER
    WORKER = (ConstrainedIK(config, (side,)), config, side, data, keyframes)


def evaluate_candidate(item):
    ik, cfg, side, data, keyframes = WORKER
    index, base, grid, ratio = item
    ik.set_bases({side: base})
    result = {"candidate": index, "camera_from_base": base.tolist(), "grid": grid,
              "mean_reach_ratio": ratio, "score": 1 - 5 * abs(ratio - .65),
              "accepted": False, "keyframes_passed": 0, "reject_reason": "no_safe_initial_solution"}
    for offset in cfg["seed_joint1_offsets_rad"]:
        q = ik.home.copy()
        q[0] += offset
        ik.configuration.update(ik.set_widths(q, {side: data["gripper_width_m"][0]}))
        if min(ik.clearance()) < cfg["collision_clearance_m"] - 1e-6:
            continue
        first = ik.solve({side: target_pose(data, 0)}, {side: data["gripper_width_m"][0]}, q, reference=q)
        if not first["accepted"]:
            continue
        reference = first["q"].copy()
        q = reference.copy()
        track = []
        failure = None
        for frame in keyframes:
            step = ik.solve({side: target_pose(data, frame)}, {side: data["gripper_width_m"][frame]}, q, reference=reference)
            if not step["accepted"]:
                failure = "keyframe_ik_or_clearance"
                break
            q = step["q"].copy()
            track.append(q.tolist())
        result["keyframes_passed"] = max(result["keyframes_passed"], len(track))
        if failure is None:
            result.update(accepted=True, reject_reason=None, seed_offset=offset,
                          q_keyframes=track, reference_q=reference.tolist(), feasibility_rate=1.0)
            return result
        result["reject_reason"] = failure
    return result


def fingerprint(cfg, targets):
    h = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode())
    for side in ("left", "right"):
        for k in ("tcp_cam_m", "rotation_cam_gripper", "gripper_width_m", "rgb_header_timestamp_ns"):
            h.update(np.ascontiguousarray(targets[side][k]).tobytes())
    for name in ("constrained_ik.py", "collision_distance.py", "production_search.py", "robot_scene.py", "run_pipeline.py"):
        h.update((ROOT / "src" / name).read_bytes())
    return h.hexdigest()


def search(cfg, targets, output, workers=4):
    output.mkdir(parents=True, exist_ok=True)
    keyframes = np.unique(np.linspace(0, len(targets["left"]["frame_index"]) - 1,
                                      min(cfg["keyframes_max"], len(targets["left"]["frame_index"])), dtype=int))
    signature = fingerprint(cfg, targets)
    metadata = output / "search_manifest.json"
    if metadata.exists():
        if json.loads(metadata.read_text())["fingerprint"] != signature:
            raise ValueError("Search cache configuration/input/code mismatch; use a new output directory")
    else:
        write_json(metadata, {"fingerprint": signature, "config": cfg, "keyframes": keyframes.tolist(),
                              "status": "running", "full_grid_candidates_per_arm": 11025,
                              "early_rejection": "Any failed hard IK/clearance gate rejects this seed; scores rank fully passing candidates only."})
    accepted = {}
    for side in ("left", "right"):
        options = list(candidates(targets[side], side, cfg))
        items = [(i, T, grid, float(np.mean(distance[keyframes]) / cfg["search_radius_m"]))
                 for i, (T, grid, distance) in enumerate(options)]
        cache = output / f"{side}_candidates.jsonl"
        done = {}
        if cache.exists():
            # A crash may leave a partial final JSONL record. Keep complete lines.
            complete = []
            for line in cache.read_text().splitlines():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    break
                done[r["candidate"]] = r
                complete.append(line)
            cache.write_text("\n".join(complete) + ("\n" if complete else ""))
        pending = [item for item in items if item[0] not in done]
        print(f"[{side}] grid=11025, distance-survivors={len(items)}, cached={len(done)}, pending={len(pending)}", flush=True)
        started = time.monotonic()
        with cache.open("a", buffering=1) as f:
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers,
                     mp_context=multiprocessing.get_context("spawn"), initializer=initialize_worker,
                     initargs=(cfg, side, targets[side], keyframes)) as pool:
                for r in pool.map(evaluate_candidate, pending, chunksize=4):
                    done[r["candidate"]] = r
                    f.write(json.dumps(r, allow_nan=False) + "\n")
                    if len(done) % 100 == 0 or len(done) == len(items):
                        passed = sum(x["accepted"] for x in done.values())
                        print(f"[{side}] checked {len(done)}/{len(items)}, all-keyframe-safe={passed}, elapsed={time.monotonic()-started:.1f}s", flush=True)
        accepted[side] = sorted([r for r in done.values() if r["accepted"]],
                                key=lambda r: (-r["score"], r["candidate"]))
        write_json(output / f"{side}_ranked.json", accepted[side])
    m = json.loads(metadata.read_text())
    m.update(status="succeeded", accepted_counts={s: len(accepted[s]) for s in accepted})
    write_json(metadata, m)
    return accepted, keyframes


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "configs/panda_production.yaml")
    p.add_argument("--targets", type=Path, default=TARGET_DEFAULT)
    p.add_argument("--background", type=Path, default=BACKGROUND_DEFAULT)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    targets, *_ = load_inputs(args.targets, args.background)
    search(yaml.safe_load(args.config.read_text()), targets, args.output, args.workers)
