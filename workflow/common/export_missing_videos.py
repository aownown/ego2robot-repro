"""Export missing viewing copies for the selected completed steps 1--5.

Uses saved videos/PNG/NPZ only. Does not run models, IK, or change source results.
Existing videos are checked and retained. Quantitative masks/depth stay PNG/NPZ.
Run with step_two_Retarget_and_Smooth/envs/retarget/bin/python.
"""
import argparse
import csv
import hashlib
import json
import math
import subprocess
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

WORKFLOW = Path(__file__).resolve().parents[1]
CLIP = WORKFLOW.parent / "test_data"
S1 = WORKFLOW / "step_one_WiLoR_and_DynHaMR/outputs/dynhamr/run_20260911_111714_RC8V45"
S2 = WORKFLOW / "step_two_Retarget_and_Smooth/outputs/retarget/run_20260911_114939_41iJgn"
ARM = WORKFLOW / "step_three_SAM3/outputs/person_masks/run_20260911_143312_xgz87fo5"
SHADOW = WORKFLOW / "step_three_SAM3/outputs/shadow_masks/full_20260911_164715"
COMBINED = SHADOW / "arm_shadow"
S4 = WORKFLOW / "step_four_ProPainter/outputs/inpaint_full_arm_shadow_20260911_170624"
S5 = WORKFLOW / "step_five_MuJoCo_and_mink/outputs/panda_production_20260911_v5"
FRAMES, FPS = 450, 30
RECORDS = []


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for data in iter(lambda: f.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def probe(path, size=None):
    stream = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
        "stream=width,height,nb_read_frames,avg_frame_rate,duration,codec_name,pix_fmt", "-of", "json", str(path)
    ]))["streams"][0]
    assert int(stream["nb_read_frames"]) == FRAMES, (path, stream)
    assert Fraction(stream["avg_frame_rate"]) == FPS, (path, stream)
    assert abs(float(stream["duration"]) - FRAMES / FPS) < 1e-3
    if size:
        assert (stream["width"], stream["height"]) == size, (path, stream)
    return stream


def sequence(directory, pattern, size, binary=False):
    paths = [directory / (pattern % i) for i in range(FRAMES)]
    h = hashlib.sha256()
    for path in paths:
        frame = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        assert frame is not None and (frame.shape[1], frame.shape[0]) == size, path
        if binary:
            assert frame.ndim == 2 and np.all((frame == 0) | (frame == 255)), path
        h.update(path.name.encode())
        h.update(bytes.fromhex(sha256(path)))
    return {"directory": str(directory), "pattern": pattern, "frames": FRAMES,
            "sequence_sha256": h.hexdigest(), "binary_source": binary}


def encode_args(crf):
    return ["-an", "-c:v", "libx264", "-threads", "4", "-preset", "fast", "-crf", str(crf),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart"]


def save_video(path, size, label, inputs, arguments=None, raw_frames=None, crf=18, note=""):
    record = {"label": label, "path": str(path), "inputs": inputs, "note": note,
              "visualization_only": True, "script_sha256": sha256(Path(__file__))}
    if path.exists():
        record.update(action="existing_verified", stream=probe(path, size))
        RECORDS.append(record)
        print(f"EXISTS {path}", flush=True)
        return
    partial = path.with_name(path.stem + ".partial.mp4")
    if partial.exists():
        raise FileExistsError(f"Inspect previous partial export first: {partial}")
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-n"]
    if raw_frames is not None:
        command += ["-f", "rawvideo", "-pixel_format", "bgr24", "-video_size", f"{size[0]}x{size[1]}",
                    "-framerate", str(FPS), "-i", "pipe:0"]
    else:
        command += arguments
    command += encode_args(crf) + [str(partial)]
    log = path.with_suffix(".export.log")
    with log.open("w") as error_log:
        proc = subprocess.Popen(command, stdin=subprocess.PIPE if raw_frames is not None else subprocess.DEVNULL,
                                stderr=error_log)
        try:
            if raw_frames is not None:
                count = 0
                for frame in raw_frames():
                    assert frame.shape == (size[1], size[0], 3) and frame.dtype == np.uint8
                    proc.stdin.write(np.ascontiguousarray(frame).tobytes())
                    count += 1
                proc.stdin.close()
                assert count == FRAMES
            if proc.wait(timeout=60):
                raise RuntimeError(f"Encoding failed: {log}")
        except BaseException:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=15)
            raise
    record.update(action="generated", stream=probe(partial, size), output_sha256=sha256(partial))
    partial.rename(path)
    record["command"] = command
    write_json(path.with_suffix(".video.json"), record)
    RECORDS.append(record)
    print(f"CREATED {path}", flush=True)


def png_video(directory, pattern, path, label, size, binary=False, note=""):
    source = sequence(directory, pattern, size, binary)
    save_video(path, size, label, [source],
               arguments=["-framerate", str(FPS), "-start_number", "0", "-i", str(directory / pattern)],
               crf=0 if binary else 18, note=note)


def depth_preview():
    files = [S5 / "robot_depth" / f"{i:04d}.npz" for i in range(FRAMES)]
    minimum, maximum = np.inf, -np.inf
    source_hash = hashlib.sha256()
    for p in files:
        with np.load(p, allow_pickle=False) as d:
            z = d["depth_m"]
        assert z.shape == (360, 640) and z.dtype == np.float32
        finite = np.isfinite(z)
        assert finite.any() and np.all(z[finite] > 0) and np.isposinf(z[~finite]).all()
        minimum = min(minimum, float(z[finite].min()))
        maximum = max(maximum, float(z[finite].max()))
        source_hash.update(p.name.encode())
        source_hash.update(bytes.fromhex(sha256(p)))
    near = math.floor(minimum * 20) / 20
    far = math.ceil(maximum * 20) / 20
    assert far > near
    def frames():
        for i, p in enumerate(files):
            with np.load(p, allow_pickle=False) as data:
                z = data["depth_m"]
            valid = np.isfinite(z)
            normalized = np.zeros(z.shape, np.uint8)
            normalized[valid] = np.rint(255 * (1 - np.clip((z[valid] - near) / (far - near), 0, 1))).astype(np.uint8)
            color = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
            color[~valid] = 0
            canvas = np.zeros((360, 752, 3), np.uint8)
            canvas[:, :640] = color
            cv2.putText(canvas, 'Model Z', (648, 22), cv2.FONT_HERSHEY_SIMPLEX, .46, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, 'meters', (652, 42), cv2.FONT_HERSHEY_SIMPLEX, .42, (230, 230, 230), 1, cv2.LINE_AA)
            ramp = np.linspace(255, 0, 260).astype(np.uint8)[:, None]
            canvas[58:318, 651:671] = cv2.applyColorMap(np.repeat(ramp, 20, axis=1), cv2.COLORMAP_TURBO)
            for fraction in np.linspace(0, 1, 5):
                y = int(58 + fraction * 259)
                cv2.putText(canvas, f'{near + fraction*(far-near):.2f}', (677, y + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, .40, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, f'f {i:03d}', (654, 345), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1, cv2.LINE_AA)
            yield canvas
    save_video(S5 / "robot_depth_preview.mp4", (752, 360), "Stage 5 robot depth preview",
               [{"directory": str(S5 / "robot_depth"), "key": "depth_m", "sequence_sha256": source_hash.hexdigest(),
                 "global_color_range_m": [near, far], "data_min_max_m": [minimum, maximum]}], raw_frames=frames,
               note="Fixed full-clip color range: near red, far blue; black means no robot. The right 112 pixels are legend. Values are model metric Z; physical scale is not validated. Use original NPZ files for quantitative work.")


def main():
    global CLIP, S1, S2, ARM, SHADOW, COMBINED, S4, S5
    parser = argparse.ArgumentParser(description=__doc__)
    for option, default in (("clip", CLIP), ("step-one", S1), ("step-two", S2),
                            ("arm-run", ARM), ("shadow-run", SHADOW),
                            ("prepared-run", COMBINED), ("background", S4), ("robot", S5)):
        parser.add_argument("--" + option, type=Path, default=default)
    parser.add_argument("--report", type=Path, default=WORKFLOW / "video_exports_20260912.json")
    args = parser.parse_args()
    CLIP, S1, S2 = args.clip.resolve(), args.step_one.resolve(), args.step_two.resolve()
    ARM, SHADOW = args.arm_run.resolve(), args.shadow_run.resolve()
    COMBINED, S4, S5 = args.prepared_run.resolve(), args.background.resolve(), args.robot.resolve()
    RECORDS.clear()
    with (CLIP / "frame_alignment.csv").open() as f:
        source_rows = list(csv.DictReader(f))
    assert len(source_rows) == FRAMES
    for directory, name in ((S1, "frame_alignment.csv"), (S2, "frame_alignment.csv"),
                             (ARM, "selected_frames.csv"), (SHADOW, "selected_frames.csv"),
                             (COMBINED, "selected_frames.csv"), (S4, "frame_mapping.csv"), (S5, "frame_mapping.csv")):
        with (directory / name).open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == FRAMES
        assert [r["rgb_header_timestamp_ns"] for r in rows] == [r["rgb_header_timestamp_ns"] for r in source_rows]
    for directory, filename, label in ((S1, "hand_pose_overlay.mp4", "Stage 1 refined hand pose"),
                                       (S2, "retarget_overlay.mp4", "Stage 2 smoothed gripper targets")):
        source = directory / "comparison.mp4"
        probe(source, (2560, 720))
        save_video(directory / filename, (1280, 720), label,
                   [{"path": str(source), "sha256": sha256(source), "panel": "right"}],
                   arguments=["-i", str(source), "-vf", "crop=1280:720:1280:0"],
                   note="Extracted the result side from an existing comparison video; labels, frame order, and camera projection are retained.")
    for directory, label in ((ARM, "Stage 3 ARM binary mask"), (SHADOW, "Stage 3 SHADOW candidate binary mask"),
                              (COMBINED, "Stage 3 merged ARM+SHADOW binary mask")):
        png_video(directory / "masks", "frame_%06d.png", directory / "mask_binary.mp4", label,
                   (1280, 720), binary=True, note="White is the mask; black is background. Original PNG masks are retained. Shadow results remain experimental.")
    raw_source = sequence(CLIP / "rgb/frames", "frame_%06d.jpg", (1280, 720))
    inpaint_source = sequence(S4 / "rgb/frames", "%04d.png", (640, 360))
    save_video(S4 / "comparison.mp4", (1280, 360), "Stage 4 source vs. inpainted comparison", [raw_source, inpaint_source],
               arguments=["-framerate", str(FPS), "-i", str(CLIP / "rgb/frames/frame_%06d.jpg"),
                          "-framerate", str(FPS), "-i", str(S4 / "rgb/frames/%04d.png"),
                          "-filter_complex", "[0:v]scale=640:360:flags=area,setsar=1[a];[1:v]setsar=1[b];[a][b]hstack=inputs=2[v]",
                          "-map", "[v]"], note="Left: downsampled source RGB. Right: ProPainter output for the arm+shadow mask.")
    png_video(S5 / "robot_rgb", "%04d.png", S5 / "robot_rgb.mp4", "Stage 5 robot-only render", (640, 360))
    for folder, label in (("robot_mask", "Stage 5 robot binary mask"), ("gripper_mask", "Stage 5 gripper binary mask")):
        png_video(S5 / folder, "%04d.png", S5 / f"{folder}.mp4", label, (640, 360), binary=True)
    depth_preview()
    report = {"scope": "Viewing copies for completed steps 1--5", "frames": FRAMES, "fps": FPS,
              "frame_alignment_verified": True, "generated_count": sum(r["action"] == "generated" for r in RECORDS),
              "videos": RECORDS, "quantitative_sources": "PNG masks and NPZ depth remain the computational inputs"}
    write_json(args.report.resolve(), report)
    print(f"DONE: {report['generated_count']} generated, {len(RECORDS)} verified", flush=True)


if __name__ == "__main__":
    main()
