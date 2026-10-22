"""Frame identity checks and lossless source I/O for stage 6."""
import csv
import hashlib
import json
import subprocess
from fractions import Fraction
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT.parent
CLIP = WORKFLOW.parent / "test_data"
ROBOT = WORKFLOW / "step_five_MuJoCo_and_mink/outputs/panda_production_20260911_v5"
IDENTITY = ("rgb_frame_index", "rgb_source_frame_index", "rgb_header_timestamp_ns")


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, obj):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rows(path):
    with Path(path).open(newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path, records):
    with Path(path).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)


def indexed(records):
    result = {int(r["rgb_frame_index"]): r for r in records}
    if len(result) != len(records):
        raise ValueError("Duplicate frame identities")
    return result


def check_identity(a, b):
    if any(a[k] != b[k] for k in IDENTITY):
        raise ValueError(f"Frame identity mismatch: {a} versus {b}")
    if abs(float(a["clip_video_time_s"]) - float(b["clip_video_time_s"])) > 1e-8:
        raise ValueError("Clip time mismatch")


def rgb(path, expected_wh=None):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None or (expected_wh and bgr.shape[:2] != tuple(expected_wh)[::-1]):
        raise ValueError(f"Missing or incorrectly sized RGB: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def mask(path, expected_wh):
    array = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if array is None or array.shape != tuple(expected_wh)[::-1]:
        raise ValueError(f"Missing or incorrectly sized mask: {path}")
    if np.any((array != 0) & (array != 255)):
        raise ValueError(f"Non-binary mask: {path}")
    return array != 0


def save_rgb(path, array):
    if not cv2.imwrite(str(path), cv2.cvtColor(array, cv2.COLOR_RGB2BGR)):
        raise IOError(path)


def save_mask(path, array):
    if not cv2.imwrite(str(path), array.astype(np.uint8) * 255):
        raise IOError(path)


def load_inputs(robot=ROBOT, clip=CLIP):
    robot, clip = Path(robot).resolve(), Path(clip).resolve()
    manifest = read_json(robot / "manifest.json")
    if manifest["status"] != "succeeded" or not manifest.get("robot_geometry_accepted"):
        raise ValueError("Use an accepted production stage-5 run")
    background = Path(manifest["background_dir"])
    upstream = read_json(background / "input_manifest.json")
    arm = Path(upstream["source_arm_run"])
    if read_json(arm / "summary.json")["prompt"] != "arm":
        raise ValueError("The human exclusion source must be the selected pure arm mask")
    sources = {"clip": indexed(rows(clip / "frame_alignment.csv")),
               "background": indexed(rows(background / "frame_mapping.csv")),
               "arm": indexed(rows(arm / "frame_alignment.csv"))}
    mapping = rows(robot / "frame_mapping.csv")
    if len(mapping) != manifest["frames_processed"]:
        raise ValueError("Stage-5 manifest and frame count disagree")
    if not manifest.get("complete_clip", True):
        raise ValueError("Stage-5 run is incomplete")
    if len(mapping) != len(sources["clip"]):
        raise ValueError("This runner expects the complete stage-5 clip")
    width, height = manifest["width"], manifest["height"]
    raw = manifest["source_camera"]["rgb"]
    K = np.array([[raw["intrinsics"]["fx"], 0, raw["intrinsics"]["cx"]],
                  [0, raw["intrinsics"]["fy"], raw["intrinsics"]["cy"]], [0, 0, 1.]])
    K_scaled = K.copy()
    K_scaled[0] *= width / raw["width"]
    K_scaled[1] *= height / raw["height"]
    if not np.allclose(K_scaled, manifest["K"], atol=1e-6):
        raise ValueError("Renderer camera does not match the original RGB camera")
    quality = {int(r["frame_index"]): r for r in rows(robot / "frame_quality.csv")}
    frames = []
    for row in mapping:
        i = int(row["rgb_frame_index"])
        for source in sources.values():
            check_identity(row, source[i])
        stem = Path(row["output_filename"]).stem
        # Stage 5 uses its output index for PNGs/NPZs; source frame IDs may differ.
        if int(stem) != int(row["output_index"]):
            raise ValueError("Unexpected stage-5 output filename mapping")
        files = {"original": clip / row["rgb_file"],
                 "background": background / "rgb/frames" / sources["background"][i]["output_filename"],
                 "human": arm / "masks" / f"frame_{i:06d}.png",
                 **{folder: robot / folder / (stem + (".npz" if folder == "robot_depth" else ".png"))
                    for folder in ("robot_rgb", "robot_depth", "robot_mask", "gripper_mask", "robot_parts")}}
        for file in files.values():
            if not file.is_file():
                raise FileNotFoundError(file)
        frames.append({"index": i, "row": row, "files": files, "stage5_quality": quality[i]})
    if [f["index"] for f in frames] != list(range(len(frames))):
        raise ValueError("Unexpected non-contiguous source clip indices")
    return {"robot": robot, "clip": clip, "background": background, "arm": arm,
            "frames": frames, "K_original": K, "K_output": K_scaled,
            "original_wh": (raw["width"], raw["height"]), "output_wh": (width, height),
            "fps": float(manifest["fps"]), "manifest": manifest}


def load_frame(inputs, frame):
    files = frame["files"]
    wh = inputs["output_wh"]
    original = rgb(files["original"], inputs["original_wh"])
    background = rgb(files["background"], wh)
    robot = rgb(files["robot_rgb"], wh)
    rm, gm = mask(files["robot_mask"], wh), mask(files["gripper_mask"], wh)
    human = mask(files["human"], inputs["original_wh"])
    human = cv2.resize(human.astype(np.uint8), wh, interpolation=cv2.INTER_NEAREST).astype(bool)
    with np.load(files["robot_depth"], allow_pickle=False) as npz:
        depth = npz["depth_m"].copy()
    if depth.shape != wh[::-1] or depth.dtype != np.float32:
        raise ValueError("Robot depth grid/type mismatch")
    if not np.all(np.isfinite(depth[rm]) & (depth[rm] > 0)) or not np.isposinf(depth[~rm]).all():
        raise ValueError("Invalid rendered Z depth")
    parts = cv2.imread(str(files["robot_parts"]), cv2.IMREAD_UNCHANGED)
    if not np.array_equal(rm, parts > 0) or not np.array_equal(gm, (parts == 2) | (parts == 4)):
        raise ValueError("Robot geom labels and masks disagree")
    return original, background, robot, depth, rm, gm, human, parts


def probe(path, count, fps, wh):
    data = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=width,height,nb_read_frames,avg_frame_rate,duration", "-of", "json", str(path)
    ]))["streams"][0]
    if (int(data["nb_read_frames"]) != count or (data["width"], data["height"]) != tuple(wh)
            or abs(float(Fraction(data["avg_frame_rate"])) - fps) > 1e-6
            or abs(float(data["duration"]) - count / fps) > 1e-3):
        raise ValueError(f"Incorrect video stream: {path}: {data}")
    return data
