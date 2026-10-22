"""Synchronously compare two SAM3 prompt runs using original RGB and final PNGs."""
import argparse
import csv
import hashlib
import json
import subprocess
import tempfile
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from segment_video import STEP, VideoWriter, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    sources = [args.left.resolve(), args.right.resolve()]
    configs = [json.loads((p / "config.json").read_text()) for p in sources]
    summaries = [json.loads((p / "summary.json").read_text()) for p in sources]
    if any(s["status"] != "succeeded" for s in summaries):
        raise ValueError("both runs must have completed")
    for key in ("clip", "requested_frames", "selected_start_frame", "fps", "resolution_wh",
                "checkpoint_sha256", "sam3_revision", "code_sha256", "source_sha256"):
        if configs[0][key] != configs[1][key]:
            raise ValueError(f"runs differ beyond the text prompt: {key}")
    parameters = [dict(c["config"]) for c in configs]
    prompts = [p.pop("prompt") for p in parameters]
    if parameters[0] != parameters[1]:
        raise ValueError("model/postprocessing parameters differ beyond the prompt")
    alignment = [(p / "selected_frames.csv").read_bytes() for p in sources]
    if alignment[0] != alignment[1]:
        raise ValueError("RGB frames/timestamps are not aligned")
    with (sources[0] / "selected_frames.csv").open() as f:
        rows = list(csv.DictReader(f))
    n, fps = len(rows), float(configs[0]["fps"])
    if n != configs[0]["requested_frames"] or any(s["frames_processed"] != n for s in summaries):
        raise ValueError("run/frame count mismatch")
    width, height = configs[0]["resolution_wh"]
    parent = STEP / "outputs/prompt_comparisons"
    parent.mkdir(parents=True, exist_ok=True)
    output = args.output.resolve() if args.output else Path(tempfile.mkdtemp(
        prefix=f"person_vs_arm_{datetime.now():%Y%m%d_%H%M%S}_", dir=parent))
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    (output / "preview_samples").mkdir()
    print(f"Output: {output}", flush=True)
    digests = [hashlib.sha256(), hashlib.sha256()]
    clip = Path(configs[0]["clip"])
    samples = {0, n//2, n-1}
    with ExitStack() as stack:
        regular = VideoWriter(output / "person_vs_arm.mp4", 2*width, height+64, fps, stack)
        slow = VideoWriter(output / "person_vs_arm_slow3x.mp4", 2*width, height+64, fps/3, stack)
        for t, row in enumerate(rows):
            index = int(row["rgb_frame_index"])
            rgb = cv2.imread(str(clip / row["rgb_file"]))
            if rgb is None or rgb.shape != (height,width,3):
                raise ValueError(f"invalid RGB frame {index}")
            panels = []
            for side, (source, prompt) in enumerate(zip(sources, prompts)):
                filename = f"frame_{index:06d}.png"
                mask_path = source / "masks" / filename
                digests[side].update(filename.encode())
                digests[side].update(mask_path.read_bytes())
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is None or mask.shape != (height,width) or not np.isin(mask, [0,255]).all():
                    raise ValueError(f"invalid final mask {mask_path}")
                mask = mask != 0
                overlay = rgb.copy()
                overlay[mask] = (.6 * overlay[mask] + .4*np.array([30,210,30])).astype(np.uint8)
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, (30,255,30), 2)
                # Add labels above the image, preserving all original edge pixels.
                panel = np.full((height+64,width,3), 20, dtype=np.uint8)
                panel[64:] = overlay
                label = f'{"LEFT" if side == 0 else "RIGHT"} | prompt: {prompt}'
                cv2.putText(panel, label, (18,29), cv2.FONT_HERSHEY_SIMPLEX, .85, (255,255,255), 2, cv2.LINE_AA)
                detail = f'frame {index:03d} | clip time {float(row["clip_video_time_s"]):.2f}s | green = final mask'
                cv2.putText(panel, detail, (18,53), cv2.FONT_HERSHEY_SIMPLEX, .53, (210,210,210), 1, cv2.LINE_AA)
                panels.append(panel)
            canvas = np.concatenate(panels, axis=1)
            regular.write(canvas)
            slow.write(canvas)
            if t in samples and not cv2.imwrite(str(output / "preview_samples" / f"frame_{index:06d}.jpg"), canvas):
                raise RuntimeError("failed to save comparison sample")
            if (t+1) % 30 == 0 or t+1 == n:
                print(f"Rendered {t+1}/{n}", flush=True)
    probes = {}
    for filename, rate in (("person_vs_arm.mp4", fps), ("person_vs_arm_slow3x.mp4", fps/3)):
        probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,avg_frame_rate,duration,nb_read_frames", "-of", "json", str(output / filename)]))["streams"][0]
        num, den = map(int, probe["avg_frame_rate"].split("/"))
        if (int(probe["nb_read_frames"]) != n or int(probe["width"]) != 2*width or int(probe["height"]) != height+64
                or abs(num/den-rate) > 1e-6 or abs(float(probe["duration"])-n/rate) > 1e-5):
            raise ValueError(f"video validation failed: {filename}")
        probes[filename] = probe
    write_json(output / "comparison.json", dict(status="succeeded", frames=n,
        left=dict(source=str(sources[0]), prompt=prompts[0], masks_sha256=digests[0].hexdigest()),
        right=dict(source=str(sources[1]), prompt=prompts[1], masks_sha256=digests[1].hexdigest()),
        same_inputs_weights_code_and_parameters_except_prompt=True,
        selected_frames_sha256=hashlib.sha256(alignment[0]).hexdigest(),
        comparison_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        annotation="Same green overlay and contour style; original RGB plus final PNG masks; no model inference rerun",
        videos=probes))
    print(f"Saved and verified: {output}", flush=True)


if __name__ == "__main__":
    main()
