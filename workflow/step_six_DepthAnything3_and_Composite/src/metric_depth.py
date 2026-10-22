"""Pinned DA3METRIC-LARGE inference with the processed camera focal length.

The metric checkpoint has no temporal attention. Process one frame at a time;
do not claim temporal optimization or sensor calibration from this conversion.
"""
from pathlib import Path
import cv2
import numpy as np
import torch


def convert_depth(raw, K_network, output_wh):
    raw = np.asarray(raw, dtype=np.float32)
    K = np.asarray(K_network, dtype=np.float64)
    if raw.ndim != 2 or not np.all(np.isfinite(raw) & (raw > 0)):
        raise ValueError("DA3 depth must be a finite positive 2-D array")
    focal = float((K[0, 0] + K[1, 1]) / 2)
    if not np.isfinite(focal) or focal <= 0:
        raise ValueError("Invalid processed focal length")
    metric = raw * np.float32(focal / 300.0)
    # The runner permits resize only, so restoring the full image is invertible
    # in pixel coordinates. A cropped prediction cannot be stretched this way.
    aligned = cv2.resize(metric, tuple(output_wh), interpolation=cv2.INTER_LINEAR)
    return aligned.astype(np.float32), focal


class MetricDepth:
    def __init__(self, checkpoint, process_res=1008):
        from depth_anything_3.api import DepthAnything3
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required by this run configuration")
        self.model = DepthAnything3.from_pretrained(str(Path(checkpoint).resolve()),
                                                   local_files_only=True).to("cuda").eval()
        if self.model.model_name != "da3metric-large":
            raise ValueError("This conversion applies only to da3metric-large")
        self.process_res = process_res

    @torch.inference_mode()
    def infer(self, rgb, K, output_wh):
        original_K = np.asarray(K, dtype=np.float32)
        images, _, processed_K = self.model.input_processor(
            image=[rgb], intrinsics=original_K[None].copy(),
            process_res=self.process_res, process_res_method="upper_bound_resize",
            sequential=True, num_workers=1, print_progress=False)
        h, w = images.shape[-2:]
        kn = processed_K[0].numpy()
        expected = original_K.copy()
        expected[0] *= w / rgb.shape[1]
        expected[1] *= h / rgb.shape[0]
        if not np.allclose(kn, expected, rtol=1e-6, atol=1e-4):
            raise ValueError("Unexpected crop or camera transform in DA3 preprocessing")
        output = self.model.forward(images[None].to("cuda"),
                                    intrinsics=processed_K[None].to("cuda"),
                                    export_feat_layers=[])
        raw = output["depth"].float().cpu().numpy()[0, 0]
        if raw.ndim == 3 and raw.shape[-1] == 1:
            raw = raw[..., 0]
        if raw.shape != (h, w):
            raise ValueError(f"Unexpected DA3 output shape: {raw.shape}, input {(h,w)}")
        aligned, focal = convert_depth(raw, kn, output_wh)
        return aligned, {"network_wh": [w, h], "K_network": kn.tolist(),
                         "focal_network_px": focal, "raw_depth_min_max":
                         [float(raw.min()), float(raw.max())]}
