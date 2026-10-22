"""SAM 3 person video masks, overlap union, A.4 postprocessing and visual previews."""
import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

from mask_processing import chunk_ranges, postprocess_masks

STEP = Path(__file__).resolve().parents[1]


class Tee:
    def __init__(self, console, log):
        self.console, self.log = console, log

    def write(self, value):
        self.console.write(value)
        self.log.write(value)
        return len(value)

    def flush(self):
        self.console.flush()
        self.log.flush()

    def isatty(self):
        return False

    def __getattr__(self, name):
        return getattr(self.console, name)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_mask(path, mask):
    if not cv2.imwrite(str(path), mask.astype(np.uint8) * 255):
        raise RuntimeError(f"could not write {path}")


def union_instances(outputs, shape):
    masks = outputs["out_binary_masks"]
    if hasattr(masks, "detach"):
        masks = masks.detach().cpu().numpy()
    masks = np.asarray(masks)
    if masks.ndim != 3 or masks.shape[1:] != shape or masks.dtype != np.bool_:
        raise ValueError(f"SAM3 binary output mismatch: {masks.shape} {masks.dtype}, expected [N,{shape}]")
    ids = np.asarray(outputs["out_obj_ids"])
    scores = np.asarray(outputs["out_probs"])
    if len(ids) != len(masks) or len(scores) != len(masks) or not np.isfinite(scores).all():
        raise ValueError("SAM3 mask/ID/score mismatch")
    return masks.any(axis=0), dict(object_ids=ids.tolist(), scores=scores.tolist())


def prepare_input(args):
    clip = args.clip.resolve()
    with (clip / "frame_alignment.csv").open() as f:
        reader = csv.DictReader(f)
        fields, all_rows = reader.fieldnames, list(reader)
    if not all_rows or args.start < 0 or args.start >= len(all_rows) or args.limit < 0:
        raise ValueError("invalid clip/start/limit")
    rows = all_rows[args.start:args.start+args.limit] if args.limit else all_rows[args.start:]
    camera = json.loads((clip / "camera_parameters.json").read_text())["rgb"]
    shape, fps = (int(camera["height"]), int(camera["width"])), float(camera["fps"])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("invalid FPS")
    for i, row in enumerate(rows):
        if int(row["rgb_frame_index"]) != args.start+i:
            raise ValueError("non-contiguous RGB frame indices")
        path = clip / row["rgb_file"]
        if path.stem != f'frame_{int(row["rgb_frame_index"]):06d}' or not path.is_file():
            raise ValueError(f"unexpected/missing RGB file: {path}")
        image = cv2.imread(str(path))
        if image is None or image.shape[:2] != shape:
            raise ValueError(f"RGB resolution mismatch: {path}")
    times = np.array([float(row["clip_video_time_s"]) for row in rows])
    timestamps = np.array([int(row["rgb_header_timestamp_ns"]) for row in rows], dtype=np.int64)
    if not np.allclose(np.diff(times), 1 / fps, atol=1e-7, rtol=0) or np.any(np.diff(timestamps) <= 0):
        raise ValueError("non-uniform video time / non-monotonic camera timestamp")
    if shape[0] % 2 or shape[1] % 2:
        raise ValueError("preview yuv420p encoder requires even image dimensions")
    return clip, rows, fields, len(all_rows), shape, fps


def infer_chunks(clip, rows, output, shape, config, progress):
    import torch
    from sam3.model_builder import build_sam3_video_predictor

    if not torch.cuda.is_available():
        raise RuntimeError("SAM3 requires CUDA; check this environment with bash prepare.sh check")
    torch.set_num_threads(4)
    torch.manual_seed(0)
    np.random.seed(0)
    checkpoint = Path(config["checkpoint"])
    print("Loading SAM 3 video predictor (local checkpoint, no compilation)...", flush=True)
    predictor = build_sam3_video_predictor(checkpoint_path=str(checkpoint), gpus_to_use=[0],
                                           compile=False, video_loader_type="cv2", async_loading_frames=False)
    coverage = np.zeros(len(rows), dtype=np.int32)
    chunks = chunk_ranges(len(rows), config["chunk_frames"], config["overlap_frames"])
    records = []
    try:
        for chunk_index, (start, end) in enumerate(chunks):
            chunk_start = time.monotonic()
            anchor = (end-start) // 2
            chunk_dir = output / "chunks" / f"chunk_{chunk_index:03d}"
            chunk_dir.mkdir(parents=True)
            (chunk_dir / "masks").mkdir()
            seen, outputs_meta = set(), {}
            record = dict(chunk_index=chunk_index, start_row=start, end_row_exclusive=end,
                          start_frame=int(rows[start]["rgb_frame_index"]),
                          end_frame_exclusive=int(rows[end-1]["rgb_frame_index"])+1,
                          anchor_frame=int(rows[start+anchor]["rgb_frame_index"]), anchor_local=anchor)
            print(f"Chunk {chunk_index+1}/{len(chunks)}: frames {record['start_frame']}–{record['end_frame_exclusive']-1}, anchor {record['anchor_frame']}", flush=True)
            with tempfile.TemporaryDirectory(prefix="sam3_rgb_", dir=output) as folder:
                # SAM3 gets local integer filenames; original RGB remains untouched.
                for local, row in enumerate(rows[start:end]):
                    (Path(folder) / f"{local:06d}.jpg").symlink_to(clip / row["rgb_file"])
                session_id = None
                try:
                    session_id = predictor.handle_request(dict(type="start_session", resource_path=folder,
                        offload_video_to_cpu=True, offload_state_to_cpu=True))["session_id"]

                    def receive(response):
                        local = int(response["frame_index"])
                        if not 0 <= local < end-start:
                            raise ValueError("SAM3 returned out-of-chunk frame")
                        mask, info = union_instances(response["outputs"], shape)
                        # The anchor may be returned more than once. Keep its last
                        # propagated result; OR is reserved for separate chunks.
                        write_mask(chunk_dir / "masks" / f"{local:06d}.png", mask)
                        outputs_meta[local] = info
                        newly_seen = local not in seen
                        seen.add(local)
                        if newly_seen and (len(seen) % 25 == 0 or len(seen) == end-start):
                            print(f"  chunk {chunk_index+1}: {len(seen)}/{end-start} unique frames | {time.monotonic()-chunk_start:.1f}s", flush=True)

                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                        receive(predictor.handle_request(dict(type="add_prompt", session_id=session_id,
                            frame_index=anchor, text=config["prompt"])))
                        for response in predictor.handle_stream_request(dict(type="propagate_in_video", session_id=session_id,
                                propagation_direction="both", start_frame_index=anchor)):
                            receive(response)
                finally:
                    if session_id is not None:
                        predictor.handle_request(dict(type="close_session", session_id=session_id))
                if len(seen) != end-start:
                    raise RuntimeError(f"chunk {chunk_index} incomplete: {len(seen)}/{end-start} frames")
            for local, row in enumerate(rows[start:end]):
                t = start+local
                stored = cv2.imread(str(chunk_dir / "masks" / f"{local:06d}.png"), cv2.IMREAD_GRAYSCALE)
                if stored is None or stored.shape != shape or not np.isin(stored, [0,255]).all():
                    raise ValueError("missing/corrupt chunk mask")
                mask = stored != 0
                merged_path = output / "masks_raw" / f'frame_{int(row["rgb_frame_index"]):06d}.png'
                if coverage[t]:
                    previous = cv2.imread(str(merged_path), cv2.IMREAD_GRAYSCALE)
                    if previous is None or previous.shape != shape:
                        raise ValueError("missing/corrupt overlap mask")
                    mask |= previous != 0
                write_mask(merged_path, mask)
                coverage[t] += 1
            record.update(status="succeeded", frames=len(seen), elapsed_seconds=time.monotonic()-chunk_start)
            write_json(chunk_dir / "summary.json", record)
            write_json(chunk_dir / "instances.json", outputs_meta)
            records.append(record)
            progress.update(completed_chunks=len(records), chunks=records)
            write_json(output / "run.json", progress)
    finally:
        predictor.shutdown()
    expected = np.zeros_like(coverage)
    for start, end in chunks:
        expected[start:end] += 1
    if not np.array_equal(coverage, expected) or np.any(coverage == 0):
        raise ValueError("chunk coverage mismatch")
    return coverage, records, dict(torch=torch.__version__, cuda=torch.version.cuda,
                                   gpu=torch.cuda.get_device_name(0), peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3)


class VideoWriter:
    def __init__(self, path, width, height, fps, stack):
        self.path = path
        log = stack.enter_context(path.with_suffix(".ffmpeg.log").open("w"))
        self.proc = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-n", "-f", "rawvideo",
            "-pixel_format", "bgr24", "-video_size", f"{width}x{height}", "-framerate", str(fps),
            "-i", "pipe:0", "-an", "-c:v", "libx264", "-threads", "2", "-preset", "fast", "-crf", "20",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE, stderr=log)
        stack.callback(self.close)

    def write(self, frame):
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self):
        self.proc.stdin.close()
        try:
            code = self.proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
            raise RuntimeError(f"video encoder stalled: {self.path}")
        if code:
            raise RuntimeError(f"video encoding failed; see {self.path.with_suffix('.ffmpeg.log')}")


def build_previews(clip, rows, output, masks, stats, fps, prompt="person"):
    n, h, w = masks.shape
    samples = set(np.linspace(0, n-1, min(n, 8), dtype=int))
    corrected = np.flatnonzero(stats["gap_interpolated"] | stats["small_area_replaced"])
    samples.update(corrected[:20].tolist())
    with ExitStack() as stack:
        overlay_video = VideoWriter(output / "mask_overlay.mp4", w, h, fps, stack)
        comparison = VideoWriter(output / "mask_comparison.mp4", 3*w, h, fps, stack)
        for t, row in enumerate(rows):
            frame = int(row["rgb_frame_index"])
            rgb = cv2.imread(str(clip / row["rgb_file"]))
            mask = masks[t]
            overlay = rgb.copy()
            overlay[mask] = (.6 * overlay[mask] + .4 * np.array([30, 210, 30])).astype(np.uint8)
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (30, 255, 30), 2)
            # Clean full-resolution overlay first; labels belong to the comparison.
            overlay_video.write(overlay)
            binary = np.repeat((mask.astype(np.uint8)*255)[..., None], 3, axis=2)
            panels = [rgb.copy(), overlay.copy(), binary]
            labels = ["Original RGB", f"SAM3 {prompt} | green = removal region", "Final mask | white = removal region"]
            flags = []
            if stats["gap_interpolated"][t]:
                flags.append("gap interpolated")
            if stats["small_area_replaced"][t]:
                flags.append("small area replaced")
            if not mask.any():
                flags.append("EMPTY: review needed")
            for panel, label in zip(panels, labels):
                cv2.rectangle(panel, (0, 0), (w, 67), (20, 20, 20), -1)
                cv2.putText(panel, label, (14, 26), cv2.FONT_HERSHEY_SIMPLEX, .68, (255,255,255), 2, cv2.LINE_AA)
                detail = f"frame {frame} | area {mask.mean()*100:.1f}%" + (" | " + ", ".join(flags) if flags else "")
                cv2.putText(panel, detail, (14, 52), cv2.FONT_HERSHEY_SIMPLEX, .53, (230,230,230), 1, cv2.LINE_AA)
            canvas = np.concatenate(panels, axis=1)
            comparison.write(canvas)
            if t in samples and not cv2.imwrite(str(output / "preview_samples" / f"frame_{frame:06d}.jpg"), canvas):
                raise RuntimeError("could not save preview image")
            if (t+1) % 50 == 0 or t+1 == n:
                print(f"Preview {t+1}/{n}", flush=True)
    videos = {}
    for filename in ("mask_overlay.mp4", "mask_comparison.mp4"):
        probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
            "-show_entries", "stream=width,height,avg_frame_rate,duration,nb_read_frames", "-of", "json", str(output / filename)]))["streams"][0]
        numerator, denominator = map(int, probe["avg_frame_rate"].split("/"))
        expected_width = w if filename == "mask_overlay.mp4" else 3*w
        if int(probe["nb_read_frames"]) != n or int(probe["width"]) != expected_width or int(probe["height"]) != h or abs(numerator/denominator - fps) > 1e-6:
            raise ValueError(f"preview stream mismatch: {filename}")
        videos[filename] = probe
    return videos


def run(args, output, progress):
    clip, rows, fields, available, shape, fps = prepare_input(args)
    config = yaml.safe_load(args.config.read_text())
    if config["model"] != "sam3" or config["anchor"] != "middle_frame" or config["propagation_direction"] != "both" or config["overlap_merge"] != "bitwise_or":
        raise ValueError("unsupported model/anchor/propagation/merge configuration")
    config["checkpoint"] = str((STEP / config["checkpoint"]).resolve())
    if args.prompt is not None:
        config["prompt"] = args.prompt
    if not config["prompt"].strip():
        raise ValueError("empty text prompt")
    if "output_prob_thresh" in config:
        raise ValueError("this SAM3 model ignores output_prob_thresh; remove that unsupported setting")
    if not Path(config["checkpoint"]).is_file():
        raise FileNotFoundError("SAM3 checkpoint missing; run bash prepare.sh weights")
    chunk_ranges(len(rows), config["chunk_frames"], config["overlap_frames"])
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError("system ffmpeg and ffprobe are required")
    cfg_post = config["postprocess"]
    if cfg_post["small_area_fallback"] != "nearest_valid" or cfg_post["close_shape"] != "ellipse" or cfg_post["close_kernel"][0] != cfg_post["close_kernel"][1]:
        raise ValueError("unsupported mask postprocessing configuration")
    for folder in ("masks_raw", "masks", "preview_samples"):
        (output / folder).mkdir()
    metadata = dict(config=config, clip=str(clip), selected_start_frame=int(rows[0]["rgb_frame_index"]),
        requested_frames=len(rows), available_frames=available, fps=fps, resolution_wh=list(reversed(shape)),
        precision="bfloat16 autocast; upstream frame storage float16", offload_video_to_cpu=True, offload_state_to_cpu=True,
        upstream_defaults="unchanged SAM3 temporal disambiguation; detection threshold 0.5, new detection threshold 0.7, hotstart delay 15 frames",
        mask_interpolation="linear signed-distance field interpolation, threshold >0; a local choice not specified by the paper",
        area_replacement="after gap interpolation; one pass, nearest nonempty non-outlier mask, earlier on ties; clipped median window at endpoints",
        checkpoint_sha256=sha256(Path(config["checkpoint"])),
        source_sha256={name: sha256(clip / name) for name in ("camera_parameters.json", "frame_alignment.csv")},
        code_sha256={name: sha256(STEP / "src" / name) for name in ("segment_video.py", "mask_processing.py")},
        sam3_revision=subprocess.check_output(["git", "-C", str(STEP / "repos/sam3"), "rev-parse", "HEAD"], text=True).strip(),
        exact_paper_reproduction=False)
    write_json(output / "config.json", metadata)
    progress.update(clip=str(clip), requested_frames=len(rows), complete_clip=False, prompt=config["prompt"])
    write_json(output / "run.json", progress)
    shutil.copy2(clip / "camera_parameters.json", output / "camera_parameters.json")
    shutil.copy2(clip / "frame_alignment.csv", output / "frame_alignment.csv")
    with (output / "selected_frames.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    coverage, chunks, device = infer_chunks(clip, rows, output, shape, config, progress)
    print("Applying mask gap/area/closing postprocessing...", flush=True)
    raw = np.empty((len(rows), *shape), dtype=bool)
    for t, row in enumerate(rows):
        mask = cv2.imread(str(output / "masks_raw" / f'frame_{int(row["rgb_frame_index"]):06d}.png'), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != shape or not np.isin(mask, [0,255]).all():
            raise ValueError("raw mask invalid")
        raw[t] = mask != 0
    masks, stats = postprocess_masks(raw, max_gap=cfg_post["interpolate_gaps_max_frames"],
        median_window=cfg_post["area_median_window"], area_ratio=cfg_post["minimum_area_ratio"], close_size=cfg_post["close_kernel"][0])
    del raw
    frame_index = np.array([int(row["rgb_frame_index"]) for row in rows], dtype=np.int64)
    replacement_frames = np.full(len(rows), -1, dtype=np.int64)
    replaced = stats["replacement_source_row"] >= 0
    replacement_frames[replaced] = frame_index[stats["replacement_source_row"][replaced]]
    np.savez_compressed(output / "mask_stats.npz", **stats, frame_index=frame_index, chunk_coverage=coverage,
        replacement_source_frame= replacement_frames,
        clip_video_time_s=np.array([float(row["clip_video_time_s"]) for row in rows]),
        rgb_header_timestamp_ns=np.array([int(row["rgb_header_timestamp_ns"]) for row in rows], dtype=np.int64),
        rgb_source_frame_index=np.array([int(row["rgb_source_frame_index"]) for row in rows], dtype=np.int64))
    for t, index in enumerate(frame_index):
        path = output / "masks" / f"frame_{index:06d}.png"
        write_mask(path, masks[t])
        loaded = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if loaded is None or loaded.shape != shape or not np.array_equal(loaded, masks[t].astype(np.uint8)*255):
            raise ValueError(f"saved mask round-trip mismatch: {path}")
    videos = build_previews(clip, rows, output, masks, stats, fps, prompt=config["prompt"])
    empty = frame_index[stats["final_area_px"] == 0].tolist()
    summary = dict(schema="pkdata.sam3.person_masks.v1", status="succeeded", frames_processed=len(rows),
        available_frames=available, complete_clip=len(rows)==available and args.start==0,
        start_frame=int(frame_index[0]), end_frame_exclusive=int(frame_index[-1])+1,
        fps=fps, resolution_wh=list(reversed(shape)), prompt=config["prompt"], chunk_count=len(chunks), chunks=chunks,
        overlap_frames=frame_index[coverage>1].tolist(), raw_empty_mask_frames=frame_index[stats["raw_area_px"]==0].tolist(),
        gap_interpolated_frames=frame_index[stats["gap_interpolated"]].tolist(),
        small_area_replaced_frames=frame_index[stats["small_area_replaced"]].tolist(),
        empty_mask_frames=empty, area_fraction_min_max=[float(masks.mean(axis=(1,2)).min()), float(masks.mean(axis=(1,2)).max())],
        final_mask_count=len(list((output / "masks").glob("*.png"))), device=device,
        visual_quality_review="pending; numerical completion does not verify arm coverage or object preservation",
        purpose="A.4 person/arm removal masks for ProPainter; distinct from A.1 hand detection gate")
    if summary["final_mask_count"] != len(rows):
        raise ValueError("final mask count mismatch")
    write_json(output / "validation.json", dict(structural_checks_passed=True, binary_png_roundtrip_passed=True,
        chunk_coverage_passed=True, frame_alignment_passed=True, videos=videos, visual_quality_review="pending"))
    write_json(output / "summary.json", summary)
    progress.update(status="succeeded", frames_processed=len(rows), complete_clip=summary["complete_clip"], empty_mask_frames=empty)
    print(f"Done: {len(rows)} masks | empty frames: {empty}", flush=True)
    print(f"Overlay: {output / 'mask_overlay.mp4'}")
    print(f"Comparison: {output / 'mask_comparison.mp4'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", type=Path, default=STEP.parent.parent / "test_data")
    parser.add_argument("--config", type=Path, default=STEP / "configs/arm.yaml",
                        help="default: configs/arm.yaml (current workflow choice)")
    parser.add_argument("--output", type=Path, help="new/empty output directory; otherwise timestamped automatically")
    parser.add_argument("--start", type=int, default=0, help="starting clip frame index")
    parser.add_argument("--limit", type=int, default=0, help="0=complete clip; use 90 for a short preview")
    parser.add_argument("--prompt", help="override the selected config's prompt; recorded in the run")
    args = parser.parse_args()
    (STEP / "outputs/person_masks").mkdir(parents=True, exist_ok=True)
    output = args.output.resolve() if args.output else Path(tempfile.mkdtemp(prefix=f"run_{datetime.now():%Y%m%d_%H%M%S}_", dir=STEP / "outputs/person_masks"))
    if output.exists() and any(output.iterdir()):
        parser.error(f"output directory is not empty: {output}; choose a new directory")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    progress = dict(schema="pkdata.sam3.person_masks.run.v1", status="running", output=str(output))
    (STEP / "logs").mkdir(exist_ok=True)
    log_path = STEP / "logs" / f"sam3_{datetime.now():%Y%m%d_%H%M%S}_{output.name}.log"
    old_stdout, old_stderr = sys.stdout, sys.stderr
    with log_path.open("x") as log:
        sys.stdout, sys.stderr = Tee(old_stdout, log), Tee(old_stderr, log)
        try:
            print(f"Output: {output}\nLog: {log_path}", flush=True)
            write_json(output / "run.json", progress)
            run(args, output, progress)
        except BaseException as error:
            progress.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                            error_type=type(error).__name__, error=str(error))
            traceback.print_exc()
            raise
        finally:
            progress["elapsed_seconds"] = time.monotonic()-started
            write_json(output / "run.json", progress)
            sys.stdout, sys.stderr = old_stdout, old_stderr


if __name__ == "__main__":
    main()
