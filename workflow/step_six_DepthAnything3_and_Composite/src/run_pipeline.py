"""Stage 6: sequential DA3 depth estimation and Appendix A.4 compositing."""
import argparse
import tempfile
import time
from pathlib import Path
import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from scipy.ndimage import binary_dilation
from composite import composite
from metric_depth import MetricDepth
from pipeline_io import (ROOT, ROBOT, CLIP, load_inputs, load_frame, write_json,
                         write_csv, digest, save_rgb, save_mask, probe)


def title(rgb, text):
    """Place captions above the image, preserving the complete image grid."""
    result = np.zeros((rgb.shape[0] + 32, rgb.shape[1], 3), np.uint8)
    result[32:] = rgb
    cv2.putText(result, text, (9, 22), cv2.FONT_HERSHEY_SIMPLEX, .48,
                (255, 255, 255), 1, cv2.LINE_AA)
    return result


def color_depth(depth, near, far):
    normal = np.rint(255 * (1 - np.clip((depth - near) / (far - near), 0, 1))).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(normal, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)


def summarize(records, inputs, elapsed, peak):
    counts = np.array([r["occluded_gripper_pixels"] for r in records])
    return {"status": "succeeded", "frames_processed": len(records),
            "available_frames": len(inputs["frames"]),
            "complete_clip": len(records) == len(inputs["frames"]),
            "frames_with_gripper_occlusion": int((counts > 0).sum()),
            "occluded_gripper_pixels_min_max": [int(counts.min()), int(counts.max())],
            "occluded_fraction_of_gripper_max": max(r["occluded_fraction_of_gripper"] for r in records),
            "visible_robot_area_ratio_max": max(r["visible_robot_area_ratio"] for r in records),
            "scene_depth_min_max_m": [min(r["scene_min_m"] for r in records), max(r["scene_max_m"] for r in records)],
            "geometry_valid_frames_from_stage5": sum(r["stage5_geometry_valid"] for r in records),
            "strict_valid_frames_from_stage5": sum(r["stage5_strict_valid"] for r in records),
            "metric_conversion_verified": True, "physical_scale_validated": False,
            "temporal_depth_optimization_applied": False,
            "visual_quality_accepted": False, "peak_cuda_allocated_gib": peak,
            "elapsed_seconds": elapsed,
            "note": "Numerical completion only. Model-predicted meters and camera-relative robot placement require physical scale validation. Upstream inpainting and hand/width quality flags remain."}


def generate_videos(output, selected, inputs, records, sparse):
    wh = inputs["output_wh"]
    fps = 1.0 if sparse else inputs["fps"]
    # One global scale over the run; no per-frame normalization hiding depth drift.
    near = max(0., np.floor(min(r["scene_p05_m"] for r in records) * 20) / 20)
    far = np.ceil(max(r["scene_p95_m"] for r in records) * 20) / 20
    far = max(far, near + .05)
    shapes = {"composite": wh, "comparison": (wh[0] * 2, wh[1] + 32),
              "scene_depth_preview": (wh[0], wh[1] + 32),
              "occlusion_overlay": (wh[0], wh[1] + 32)}
    writers = {name: imageio.get_writer(output / f"{name}.partial.mp4", fps=fps,
                 codec="libx264", macro_block_size=1, ffmpeg_log_level="error",
                 ffmpeg_params=["-crf", "18", "-preset", "fast", "-threads", "2", "-movflags", "+faststart"])
               for name in shapes}
    samples = set(np.linspace(0, len(selected) - 1, min(6, len(selected))).round().astype(int))
    panels = []
    try:
        for pos, (frame, record) in enumerate(zip(selected, records)):
            i, stem = frame["index"], f"{frame['index']:04d}"
            original, bg, robot, _, rm, _, _, _ = load_frame(inputs, frame)
            result = cv2.cvtColor(cv2.imread(str(output / "frames" / f"{stem}.png")), cv2.COLOR_BGR2RGB)
            with np.load(output / "scene_depth" / f"{stem}.npz", allow_pickle=False) as npz:
                scene = npz["depth_m"]
            hidden = cv2.imread(str(output / "occlusion_mask" / f"{stem}.png"), 0) > 0
            before = bg.copy()
            before[rm] = robot[rm]
            overlay = before.copy()
            overlay[hidden] = (.35 * overlay[hidden] + .65 * np.array([255, 0, 255])).astype(np.uint8)
            comparison = np.concatenate([title(before, f"Step 5 | f{i:03d} | overlay"),
                           title(result, f"Step 6 | hidden gripper {record['occluded_gripper_pixels']} px")], axis=1)
            depth_view = title(color_depth(scene, near, far), f"DA3 Z | near red {near:.2f}m - far blue {far:.2f}m | f{i:03d}")
            writers["composite"].append_data(result)
            writers["comparison"].append_data(comparison)
            writers["scene_depth_preview"].append_data(depth_view)
            writers["occlusion_overlay"].append_data(title(overlay, f"Magenta: hidden gripper pixels | f{i:03d}"))
            if pos in samples:
                save_rgb(output / "quality_review" / f"comparison_{stem}.jpg", comparison)
                panels.append(np.concatenate([title(cv2.resize(original, wh, interpolation=cv2.INTER_AREA), f"Original | f{i:03d}"),
                                               depth_view, title(result, "Composite")], axis=1))
    finally:
        for writer in writers.values():
            writer.close()
    streams = {}
    for name, size in shapes.items():
        partial = output / f"{name}.partial.mp4"
        streams[name] = probe(partial, len(selected), fps, size)
        partial.rename(output / f"{name}.mp4")
    save_rgb(output / "quality_review/sampled_frames.jpg", np.concatenate(panels, axis=0))
    return {"streams": streams, "depth_color_range_m": [near, far],
            "sparse_keyframe_slideshow": sparse, "video_fps": fps}


def run(args):
    started = time.monotonic()
    inputs = load_inputs(args.robot, args.clip)
    if args.frame_indices:
        indices = [int(i) for i in args.frame_indices.split(",")]
        if indices != sorted(set(indices)):
            raise ValueError("Sample indices must be unique and sorted")
        sparse = True
    else:
        end = len(inputs["frames"]) if args.max_frames is None else args.start_frame + args.max_frames
        indices = list(range(args.start_frame, end))
        sparse = False
    if not indices or min(indices) < 0 or max(indices) >= len(inputs["frames"]):
        raise ValueError("Requested frame interval is outside the available clip")
    selected = [inputs["frames"][i] for i in indices]
    print(f"[input] {len(selected)} frames; {inputs['output_wh']}; {inputs['fps']} fps; pure ARM mask", flush=True)
    for frame in selected:
        load_frame(inputs, frame)  # Check all input arrays before GPU work.
    if args.check_inputs:
        print("Input identities, timestamps, camera intrinsics, all RGB/mask/depth grids passed.", flush=True)
        return
    if args.output:
        output = Path(args.output).resolve()
        output.mkdir(parents=True, exist_ok=False)
    else:
        (ROOT / "outputs").mkdir(exist_ok=True)
        output = Path(tempfile.mkdtemp(prefix=time.strftime("composite_%Y%m%d_%H%M%S_"), dir=ROOT / "outputs"))
    print(f"OUTPUT={output}", flush=True)
    manifest = {"schema": "pkdata.stage6.depth_composite.v1", "status": "running",
                "robot_dir": str(inputs["robot"]), "background_dir": str(inputs["background"]),
                "human_mask_dir": str(inputs["arm"]), "clip": str(inputs["clip"]),
                "source_manifest_sha256": digest(inputs["robot"] / "manifest.json"),
                "source_alignment_sha256": digest(inputs["clip"] / "frame_alignment.csv"),
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "model_config_sha256": digest(Path(args.checkpoint) / "config.json"),
                "checkpoint_sha256": digest(Path(args.checkpoint) / "model.safetensors"),
                "code_sha256": {p.name: digest(p) for p in sorted((ROOT / "src").glob("*.py"))},
                "frame_indices": indices, "fps": inputs["fps"], "output_wh": inputs["output_wh"],
                "K_output": inputs["K_output"].tolist(), "K_original": inputs["K_original"].tolist(),
                "depth_source": "original_rgb", "process_res": args.process_res,
                "process_res_method": "upper_bound_resize", "depth_resize": "bilinear",
                "depth_convention": "optical-axis Z, predicted meters", "scene_depth_scale_multiplier": 1.0,
                "metric_conversion": "raw_depth * mean(processed_fx, processed_fy) / 300",
                "physical_scale_validated": False, "camera_distortion_applied": False,
                "temporal_depth_optimization_applied": False,
                "human_mask_resize": "nearest", "human_mask_dilation": {"kernel": [5, 5], "iterations": 1, "grid_wh": inputs["output_wh"]},
                "rule": "arm always visible; hide gripper when scene Z < robot Z outside dilated pure ARM mask",
                "references": ["https://arxiv.org/html/2608.02580v1#A4",
                               "https://github.com/ByteDance-Seed/Depth-Anything-3#-faq"]}
    write_json(output / "manifest.json", manifest)
    records, source_hashes = [], []
    try:
        for name in ("frames", "scene_depth", "visible_robot_mask", "occlusion_mask", "quality_review"):
            (output / name).mkdir()
        mapping = []
        for pos, frame in enumerate(selected):
            row = frame["row"].copy()
            row.update(output_index=str(pos), output_filename=f"{frame['index']:04d}.png",
                       stage5_output_filename=frame["row"]["output_filename"])
            mapping.append(row)
        write_csv(output / "frame_mapping.csv", mapping)
        model = MetricDepth(args.checkpoint, args.process_res)
        torch.cuda.reset_peak_memory_stats()
        for pos, frame in enumerate(selected):
            original, bg, robot, sim, rm, gm, human, parts = load_frame(inputs, frame)
            scene, meta = model.infer(original, inputs["K_original"], inputs["output_wh"])
            result, visible = composite(bg, robot, scene, sim, rm, gm, human)
            hidden = rm & ~visible
            protected = binary_dilation(human, structure=np.ones((5, 5), bool))
            # Independent invariants, not just a successful function return.
            if np.any(hidden & (~gm | protected)) or np.any(visible & ~rm):
                raise RuntimeError("Compositing changed a protected or non-gripper pixel")
            if not np.array_equal(result[~visible], bg[~visible]):
                raise RuntimeError("Background preservation failed")
            i, stem = frame["index"], f"{frame['index']:04d}"
            np.savez_compressed(output / "scene_depth" / f"{stem}.npz", depth_m=scene,
                    frame_index=i, rgb_header_timestamp_ns=int(frame["row"]["rgb_header_timestamp_ns"]),
                    K=inputs["K_output"], K_network=np.array(meta["K_network"]), network_wh=meta["network_wh"],
                    focal_network_px=meta["focal_network_px"])
            save_rgb(output / "frames" / f"{stem}.png", result)
            save_mask(output / "visible_robot_mask" / f"{stem}.png", visible)
            save_mask(output / "occlusion_mask" / f"{stem}.png", hidden)
            record = {"frame_index": i, "scene_min_m": float(scene.min()),
                      "scene_p05_m": float(np.percentile(scene, 5)),
                      "scene_median_m": float(np.median(scene)),
                      "scene_p95_m": float(np.percentile(scene, 95)), "scene_max_m": float(scene.max()),
                      "robot_min_m": float(sim[rm].min()), "robot_max_m": float(sim[rm].max()),
                      "gripper_pixels": int(gm.sum()), "protected_gripper_pixels": int((gm & protected).sum()),
                      "occluded_gripper_pixels": int(hidden.sum()),
                      "occluded_fraction_of_gripper": float(hidden.sum() / max(1, gm.sum())),
                      "visible_robot_area_ratio": float(visible.mean()),
                      "stage5_geometry_valid": frame["stage5_quality"]["geometry_valid"] == "True",
                      "stage5_paper_style_valid": frame["stage5_quality"]["paper_style_valid"] == "True",
                      "stage5_strict_valid": frame["stage5_quality"]["strict_valid"] == "True"}
            for side, label in (("left", 2), ("right", 4)):
                side_mask = parts == label
                record[side + "_visible_gripper_pixels"] = int((side_mask & visible).sum())
                eligible = side_mask & ~protected
                record[side + "_eligible_gripper_pixels"] = int(eligible.sum())
                record[side + "_scene_minus_robot_median_m"] = float(np.median(scene[eligible] - sim[eligible])) if eligible.any() else None
            records.append(record)
            source_hashes.append({"frame_index": i, "sha256": {k: digest(p) for k, p in frame["files"].items()},
                                  "network": meta})
            if pos == 0 or (pos + 1) % 30 == 0 or pos + 1 == len(selected):
                print(f"[depth+composite] {pos+1}/{len(selected)} frame={i} hidden={hidden.sum()} px "
                      f"scene median={np.median(scene):.3f}m elapsed={time.monotonic()-started:.1f}s", flush=True)
        peak = torch.cuda.max_memory_allocated() / 2**30
        del model
        torch.cuda.empty_cache()
        write_csv(output / "frame_quality.csv", records)
        write_json(output / "source_hashes.json", source_hashes)
        videos = generate_videos(output, selected, inputs, records, sparse)
        write_json(output / "video_validation.json", videos)
        summary = summarize(records, inputs, time.monotonic() - started, peak)
        manifest.update(status="succeeded", frames_processed=len(records),
                        complete_clip=summary["complete_clip"], video_fps=videos["video_fps"],
                        sparse_keyframe_slideshow=sparse)
        write_json(output / "summary.json", summary)
        write_json(output / "manifest.json", manifest)
        print(f"DONE: {output}/comparison.mp4", flush=True)
        print(f"Occlusion in {summary['frames_with_gripper_occlusion']}/{len(records)} frames. "
              "Physical scale and visual quality remain unvalidated.", flush=True)
    except BaseException as error:
        manifest.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                        error=f"{type(error).__name__}: {error}", frames_processed=len(records))
        write_json(output / "manifest.json", manifest)
        if records:
            write_csv(output / "frame_quality.partial.csv", records)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--robot", type=Path, default=ROBOT)
    p.add_argument("--clip", type=Path, default=CLIP)
    p.add_argument("--checkpoint", type=Path, default=ROOT / "assets/DA3METRIC-LARGE")
    p.add_argument("--output", type=Path, help="New output directory; never overwrite a run")
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--max-frames", type=int, help="Default: all frames to clip end")
    p.add_argument("--frame-indices", help="Comma-separated sparse QA frames; videos are 1-fps slideshows")
    p.add_argument("--process-res", type=int, default=1008, help="DA3 longest-side target before patch rounding")
    p.add_argument("--check-inputs", action="store_true", help="Check input arrays only; no inference or output")
    args = p.parse_args()
    if args.process_res < 112:
        p.error("--process-res must be >= 112")
    if args.frame_indices and (args.start_frame or args.max_frames is not None):
        p.error("Use either sparse indices or a continuous frame interval")
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    run(args)


if __name__ == "__main__":
    main()
