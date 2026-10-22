"""Convert both stage-one tracks to camera-relative gripper targets and a preview."""
import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import scipy
from scipy.spatial.transform import Rotation, Slerp

from retarget import retarget


STEP_ROOT = Path(__file__).resolve().parents[1]
TIME_KEYS = ("frame_index", "clip_video_time_s", "rgb_header_timestamp_ns", "rgb_source_frame_index")
COLORS = {"left": (255, 230, 0), "right": (0, 220, 255)}  # BGR


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_track(path, hand, start, limit):
    with np.load(path, allow_pickle=False) as data:
        required = {"keypoints_cam_m", "observed_valid", "optimization_inferred", "finite",
                    "handedness", "K", "source_row", *TIME_KEYS}
        if not required.issubset(data.files):
            raise ValueError(f"{path}: missing stage-one fields {sorted(required - set(data.files))}")
        if str(data["handedness"].item()) != hand:
            raise ValueError(f"{path}: wrong hand identity")
        count = len(data["keypoints_cam_m"])
        if start < 0 or start >= count or (limit is not None and limit < 1):
            raise ValueError("start/limit outside source track")
        end = count if limit is None else min(count, start + limit)
        out = {key: data[key][start:end].copy() for key in required - {"handedness", "K"}}
        out["K"] = data["K"].copy()
    n = end - start
    if out["keypoints_cam_m"].shape != (n, 21, 3) or not np.isfinite(out["keypoints_cam_m"]).all():
        raise ValueError(f"{hand}: expected finite [T,21,3] camera-relative keypoints")
    for key in required - {"handedness", "K", "keypoints_cam_m"}:
        if out[key].shape != (n,) or not np.isfinite(out[key]).all():
            raise ValueError(f"{hand}: invalid {key}")
    for key in ("observed_valid", "optimization_inferred", "finite"):
        if out[key].dtype != np.bool_:
            raise ValueError(f"{hand}: {key} must be boolean")
    if not out["finite"].all() or not np.array_equal(out["optimization_inferred"], ~out["observed_valid"]):
        raise ValueError(f"{hand}: inconsistent quality flags")
    if out["K"].shape != (3, 3) or not np.isfinite(out["K"]).all():
        raise ValueError(f"{hand}: invalid camera K")
    if n < 3:
        raise ValueError("runner requires at least 3 frames")
    if not np.all(np.diff(out["frame_index"]) == 1) or not np.all(np.diff(out["rgb_header_timestamp_ns"]) > 0):
        raise ValueError(f"{hand}: non-contiguous or non-monotonic time axis")
    return out, count


def motion_metrics(pos, quat, width):
    step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    angular = (Rotation.from_quat(quat[:-1]).inv() * Rotation.from_quat(quat[1:])).magnitude()
    return {
        "tcp_max_step_mm": float(step.max() * 1000),
        "tcp_acceleration_rms_mm_per_frame2": float(np.sqrt(np.mean(np.sum(np.diff(pos, n=2, axis=0) ** 2, axis=1))) * 1000),
        "rotation_max_step_deg": float(np.rad2deg(angular.max())),
        "rotation_step_rms_deg": float(np.rad2deg(np.sqrt(np.mean(angular ** 2)))),
        "width_acceleration_rms_mm_per_frame2": float(np.sqrt(np.mean(np.diff(width, n=2) ** 2)) * 1000),
        "width_min_max_mm": [float(width.min() * 1000), float(width.max() * 1000)],
    }


def validate_targets(result, source):
    r = result["rotation_cam_gripper"]
    q = result["quaternion_xyzw"]
    orth_error = float(np.max(np.abs(r.transpose(0, 2, 1) @ r - np.eye(3))))
    det_error = float(np.max(np.abs(np.linalg.det(r) - 1)))
    quat_error = float(np.max(np.abs(Rotation.from_quat(q).as_matrix() - r)))
    if not all(np.isfinite(value).all() for value in result.values()):
        raise ValueError("non-finite target output")
    if max(orth_error, det_error, quat_error) > 1e-8 or np.any(result["gripper_width_m"] < 0):
        raise ValueError("invalid target rotation or width")
    if np.any(np.sum(q[1:] * q[:-1], axis=1) < -1e-12):
        raise ValueError("quaternion hemisphere discontinuity")
    good = np.flatnonzero(source["observed_valid"])
    gaps = np.flatnonzero(~source["observed_valid"])
    gap_pos_error = gap_rot_error = gap_width_error = 0.
    if len(gaps):
        expected = np.column_stack([np.interp(gaps, good, result["tcp_cam_m"][good, axis]) for axis in range(3)])
        gap_pos_error = float(np.max(np.abs(expected - result["tcp_cam_m"][gaps])))
        expected_rot = Slerp(good, Rotation.from_quat(q[good]))(gaps)
        gap_rot_error = float(np.max((expected_rot.inv() * Rotation.from_quat(q[gaps])).magnitude()))
        expected_width = np.interp(gaps, good, result["gripper_width_m"][good])
        gap_width_error = float(np.max(np.abs(expected_width - result["gripper_width_m"][gaps])))
        if max(gap_pos_error, gap_rot_error, gap_width_error) > 1e-8:
            raise ValueError("final short-gap interpolation did not hold")
    return dict(all_finite=True, rotation_orthogonality_max_error=orth_error,
                rotation_determinant_max_error=det_error, quaternion_matrix_max_error=quat_error,
                gap_position_max_error_m=gap_pos_error, gap_rotation_max_error_rad=gap_rot_error,
                gap_width_max_error_m=gap_width_error, passed=True)


def project(points, K):
    points = np.asarray(points)
    if not np.isfinite(points).all() or np.any(points[:, 2] <= .001):
        return None
    pixels = points @ K.T
    return np.rint(np.clip(pixels[:, :2] / pixels[:, 2:], -100000, 100000)).astype(np.int32)


def line3d(image, a, b, K, color, thickness=2):
    points = project([a, b], K)
    if points is None:
        return
    clipped, p0, p1 = cv2.clipLine((0, 0, image.shape[1], image.shape[0]), tuple(points[0]), tuple(points[1]))
    if clipped:
        cv2.line(image, p0, p1, color, thickness, cv2.LINE_AA)


def draw_target(image, source, target, hand, t, raw):
    K, k = source["K"], source["keypoints_cam_m"][t]
    for base in (1, 5, 9, 13, 17):
        for a, b in zip((0, base, base + 1, base + 2), (base, base + 1, base + 2, base + 3)):
            line3d(image, k[a], k[b], K, (150, 150, 150), 1)
    p = target["tcp_raw_cam_m" if raw else "tcp_cam_m"][t]
    r = target["rotation_raw_cam_gripper" if raw else "rotation_cam_gripper"][t]
    w = target["gripper_width_raw_m" if raw else "gripper_width_m"][t]
    # Schematic U-shaped gripper: fingertips at TCP +/- z*w/2, body behind x.
    ends = [p - r[:, 2] * w / 2, p + r[:, 2] * w / 2]
    backs = [e - r[:, 0] * .045 for e in ends]
    for a, b in ((ends[0], backs[0]), (backs[0], backs[1]), (backs[1], ends[1])):
        line3d(image, a, b, K, (0, 0, 0), 7)
        line3d(image, a, b, K, COLORS[hand], 4)
    for axis, color in enumerate(((30, 30, 255), (30, 255, 30), (255, 100, 30))):
        line3d(image, p, p + r[:, axis] * .025, K, color, 2)
    xy = project([p], K)
    if xy is not None and 0 <= xy[0, 0] < image.shape[1] and 0 <= xy[0, 1] < image.shape[0]:
        cv2.drawMarker(image, tuple(xy[0]), (255, 255, 255), cv2.MARKER_CROSS, 12, 2)
    flags = []
    if target["gap_interpolated"][t]:
        flags.append("inferred" if raw else "gap interpolated")
    if target["orientation_held"][t]:
        flags.append("orientation held")
    if target["replaced"][t] and not raw:
        flags.append("velocity replaced")
    label = f'{hand[0].upper()} width {w * 1000:.1f} mm' + (" | " + ", ".join(flags) if flags else "")
    cv2.putText(image, label, (16, 62 if hand == "left" else 88), cv2.FONT_HERSHEY_SIMPLEX, .58, COLORS[hand], 1, cv2.LINE_AA)


def render_preview(output, clip, rows, sources, results, fps):
    ids = sources["left"]["frame_index"]
    first = cv2.imread(str(clip / rows[int(ids[0])]["rgb_file"]))
    if first is None:
        raise ValueError("cannot read source RGB")
    h, w = first.shape[:2]
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-n", "-f", "rawvideo",
               "-pixel_format", "bgr24", "-video_size", f"{2*w}x{h}", "-framerate", str(fps),
               "-i", "pipe:0", "-an", "-c:v", "libx264", "-threads", "4", "-preset", "fast",
               "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output / "comparison.mp4")]
    sample_ids = set(np.linspace(0, len(ids) - 1, min(6, len(ids)), dtype=int))
    sample_ids.update(np.flatnonzero(results["left"]["gap_interpolated"] | results["right"]["gap_interpolated"]).tolist())
    (output / "preview_samples").mkdir()
    with (output / "ffmpeg.log").open("w") as error_log:
        proc = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=error_log)
        try:
            for t, frame_index in enumerate(ids):
                rgb = cv2.imread(str(clip / rows[int(frame_index)]["rgb_file"]))
                if rgb is None or rgb.shape != first.shape:
                    raise ValueError(f"invalid RGB frame {frame_index}")
                panels = []
                for raw in (True, False):
                    panel = rgb.copy()
                    cv2.rectangle(panel, (0, 0), (w, 100), (20, 20, 20), -1)
                    title = "Geometric targets from stage 1" if raw else "Smoothed gripper targets"
                    cv2.putText(panel, f'{title} | frame {frame_index}', (16, 30), cv2.FONT_HERSHEY_SIMPLEX, .73, (255, 255, 255), 2, cv2.LINE_AA)
                    for hand in ("left", "right"):
                        draw_target(panel, sources[hand], results[hand], hand, t, raw)
                    cv2.rectangle(panel, (0, h - 30), (w, h), (20, 20, 20), -1)
                    cv2.putText(panel, "Camera-relative | schematic grippers | x: red  y: green  z: blue", (16, h - 9), cv2.FONT_HERSHEY_SIMPLEX, .55, (230, 230, 230), 1, cv2.LINE_AA)
                    panels.append(panel)
                canvas = np.concatenate(panels, axis=1)
                proc.stdin.write(canvas.tobytes())
                if t in sample_ids and not cv2.imwrite(str(output / "preview_samples" / f"frame_{frame_index:06d}.jpg"), canvas):
                    raise RuntimeError("failed to write preview sample")
                if (t + 1) % 90 == 0 or t + 1 == len(ids):
                    print(f"Preview {t + 1}/{len(ids)}", flush=True)
        finally:
            proc.stdin.close()
            code = proc.wait(timeout=60)
        if code:
            raise RuntimeError(f"ffmpeg failed: see {output / 'ffmpeg.log'}")
    probe = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
        "stream=width,height,nb_read_frames,avg_frame_rate,duration", "-of", "json", str(output / "comparison.mp4")
    ]))["streams"][0]
    if int(probe["nb_read_frames"]) != len(ids):
        raise ValueError("preview frame count mismatch")
    return probe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="stage-one DynHaMR output directory")
    parser.add_argument("--output", type=Path, required=True, help="new/empty output directory")
    parser.add_argument("--clip", type=Path, default=STEP_ROOT.parent.parent / "test_data")
    parser.add_argument("--start", type=int, default=0, help="row offset in the stage-one tracks")
    parser.add_argument("--limit", type=int, help="optional preview length; default full clip")
    parser.add_argument("--no-video", action="store_true", help="only export targets and numerical validation")
    args = parser.parse_args()
    source, output, clip = args.input.resolve(), args.output.resolve(), args.clip.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be empty; previous runs are never overwritten")
    output.mkdir(parents=True, exist_ok=True)
    meta = json.loads((source / "summary.json").read_text())
    if meta.get("status") != "succeeded" or meta.get("schema") != "pkdata.dynhamr.camera_relative.v1":
        raise ValueError("expected a successful stage-one camera-relative adaptation output")
    sources, counts = {}, {}
    for hand in ("left", "right"):
        sources[hand], counts[hand] = load_track(source / f"{hand}_track.npz", hand, args.start, args.limit)
    left, right = sources["left"], sources["right"]
    for key in (*TIME_KEYS, "K"):
        if not np.array_equal(left[key], right[key]):
            raise ValueError(f"left/right {key} mismatch")
    if counts["left"] != counts["right"]:
        raise ValueError("left/right source frame counts differ")
    camera = json.loads((clip / "camera_parameters.json").read_text())["rgb"]
    fps = float(camera["fps"])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("invalid source FPS")
    intr = camera["intrinsics"]
    expected_k = np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1.]])
    if not np.allclose(expected_k, left["K"], atol=1e-8, rtol=0):
        raise ValueError("RGB clip intrinsics do not match track K")
    if not np.allclose(np.diff(left["clip_video_time_s"]), 1 / fps, atol=1e-7, rtol=0):
        raise ValueError("non-uniform video frame times; resampling is required")
    with (clip / "frame_alignment.csv").open() as f:
        rows = list(csv.DictReader(f))
    for t, index in enumerate(left["frame_index"]):
        if index < 0 or index >= len(rows):
            raise ValueError("source frame index outside RGB clip")
        row = rows[int(index)]
        for key in ("rgb_source_frame_index", "rgb_header_timestamp_ns"):
            if int(row[key]) != int(left[key][t]):
                raise ValueError(f"RGB/track alignment mismatch at frame {index}: {key}")
    n = len(left["frame_index"])
    config = {
        "source": str(source), "clip": str(clip), "start_row": args.start, "frames": n, "fps": fps,
        "source_sha256": {name: sha256(source / name) for name in ("left_track.npz", "right_track.npz", "summary.json")},
        "code_sha256": {name: sha256(STEP_ROOT / "src" / name) for name in ("retarget.py", "run_retarget.py")},
        "parameter_record": (STEP_ROOT / "configs/paper.yaml").read_text(),
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__, "opencv": cv2.__version__},
        "gap_policy": "bounded <=10 frames: linear TCP/width + SLERP before filtering and again after smoothing",
        "orientation_reduction": "incremental Gaussian-weighted SLERP in chronological order",
        "width_policy": "nonnegative after Savitzky-Golay; no hardware range clamp",
        "preview_projection": "same pinhole K as stage one; no new lens-distortion correction",
    }
    write_json(output / "config.json", config)
    summary = {
        "schema": "pkdata.retarget.camera_relative.v1", "source": str(source), "frames": n, "fps": fps,
        "start_frame": int(left["frame_index"][0]), "end_frame_exclusive": int(left["frame_index"][-1]) + 1,
        "complete_clip": bool(n == len(rows) and left["frame_index"][0] == 0),
        "coordinate_frame": meta["coordinate_frame"], "metric_depth_validated": meta.get("metric_depth_validated", False),
        "world_trajectory_recovered": False, "robot_embodiment_applied": False, "ik_applied": False,
        "exact_paper_reproduction": False, "hands": {},
        "metric_note": "Roughness and target changes measure temporal behavior, not accuracy against ground truth.",
    }
    results, validation = {}, {}
    for hand, track in sources.items():
        print(f"Retarget {hand}: {n} frames", flush=True)
        result = retarget(track["keypoints_cam_m"], hand, fps, track["observed_valid"])
        validation[hand] = validate_targets(result, track)
        results[hand] = result
        common = {key: track[key] for key in (*TIME_KEYS, "K", "source_row", "observed_valid", "optimization_inferred")}
        common.update(handedness=hand, fps=fps, coordinate_frame=meta["coordinate_frame"],
                      finite=np.ones(n, dtype=bool), schema=summary["schema"], length_unit="meter")
        with (output / f"{hand}_targets.npz").open("xb") as f:
            np.savez_compressed(f, **result, **common)
        summary["hands"][hand] = {
            "observed_frames": int(track["observed_valid"].sum()),
            "gap_interpolated_frames": track["frame_index"][result["gap_interpolated"]].tolist(),
            "orientation_held_frames": track["frame_index"][result["orientation_held"]].tolist(),
            "velocity_replaced_frames": track["frame_index"][result["replaced"]].tolist(),
            "raw": motion_metrics(result["tcp_raw_cam_m"], result["quaternion_raw_xyzw"], result["gripper_width_raw_m"]),
            "smoothed": motion_metrics(result["tcp_cam_m"], result["quaternion_xyzw"], result["gripper_width_m"]),
            "tcp_change_rms_mm": float(np.sqrt(np.mean(np.sum((result["tcp_cam_m"] - result["tcp_raw_cam_m"]) ** 2, axis=1))) * 1000),
            "rotation_change_rms_deg": float(np.rad2deg(np.sqrt(np.mean((Rotation.from_quat(result["quaternion_raw_xyzw"]).inv() * Rotation.from_quat(result["quaternion_xyzw"])).magnitude() ** 2)))),
        }
    for name in ("camera_parameters.json", "frame_alignment.csv"):
        shutil.copy2(clip / name, output / name)
    if not args.no_video:
        validation["video"] = render_preview(output, clip, rows, sources, results, fps)
    validation["passed"] = True
    summary["status"] = "succeeded"
    summary["preview_generated"] = not args.no_video
    write_json(output / "validation.json", validation)
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
