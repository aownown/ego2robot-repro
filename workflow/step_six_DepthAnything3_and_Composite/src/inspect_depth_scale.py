"""Diagnostic DA3 / hand joint Z comparison; never fits or applies a scale."""
import argparse
from pathlib import Path
import numpy as np
from pipeline_io import read_json, write_json, load_inputs, load_frame, digest


def inspect(output):
    output = Path(output).resolve()
    run = read_json(output / "manifest.json")
    inputs = load_inputs(run["robot_dir"], run["clip"])
    config = read_json(Path(inputs["manifest"]["target_dir"]) / "config.json")
    source = Path(config["source"])
    records = []
    for side in ("left", "right"):
        path = source / f"{side}_track.npz"
        if digest(path) != config["source_sha256"][path.name]:
            raise ValueError(f"Upstream hand track changed: {path}")
        with np.load(path, allow_pickle=False) as data:
            lookup = {int(index): n for n, index in enumerate(data["frame_index"])}
            for i in run["frame_indices"]:
                n = lookup[i]
                if not data["observed_valid"][n]:
                    continue
                row = inputs["frames"][i]["row"]
                if int(data["rgb_header_timestamp_ns"][n]) != int(row["rgb_header_timestamp_ns"]):
                    raise ValueError("Hand and scene timestamp mismatch")
                _, _, _, _, _, _, human, _ = load_frame(inputs, inputs["frames"][i])
                with np.load(output / "scene_depth" / f"{i:04d}.npz", allow_pickle=False) as npz:
                    depth = npz["depth_m"]
                points = data["keypoints_cam_m"][n, [0, 5, 9, 13, 17]]
                p = points @ inputs["K_output"].T
                uv = np.rint(p[:, :2] / p[:, 2:]).astype(int)
                wh = inputs["output_wh"]
                valid = (uv[:, 0] >= 0) & (uv[:, 0] < wh[0]) & (uv[:, 1] >= 0) & (uv[:, 1] < wh[1]) & (points[:, 2] > 0)
                ratios = [float(depth[y, x] / points[j, 2]) for j, (x, y) in enumerate(uv)
                          if valid[j] and human[y, x]]
                records.append({"frame_index": i, "side": side, "samples": len(ratios),
                                "median_da3_to_hand_joint_z_ratio": float(np.median(ratios)) if ratios else None})
    values = [r["median_da3_to_hand_joint_z_ratio"] for r in records if r["samples"]]
    report = {"diagnostic_only": True, "physical_scale_validated": False,
              "source_tracks": str(source), "scale_correction_applied": False,
              "note": "DA3 samples at projected observed wrist/MCP joints inside ARM mask. Joint centers are not skin surfaces. This is a consistency diagnostic, not physical calibration or a scale-fitting objective.",
              "median_ratio": float(np.median(values)) if values else None,
              "ratio_min_max": [min(values), max(values)] if values else None, "records": records}
    write_json(output / "quality_review/hand_depth_sanity.json", report)
    print(f"DA3 / projected observed hand joint Z ratios: median={report['median_ratio']}, range={report['ratio_min_max']}; no scale correction applied")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    inspect(p.parse_args().output)
