"""Ego2Robot Appendix A.4 compositing, on a shared RGB pixel grid.

Both depths must be optical-axis Z in METERS. The caller must align/calibrate
scene depth first. Native Orbbec uint16 depth is not accepted as meter depth.
"""
import numpy as np
from scipy.ndimage import binary_dilation


def composite(background_rgb, robot_rgb, scene_depth_m, robot_depth_m,
              robot_mask, gripper_mask, human_mask):
    bg, robot = np.asarray(background_rgb), np.asarray(robot_rgb)
    scene, sim = np.asarray(scene_depth_m), np.asarray(robot_depth_m)
    if bg.ndim != 3 or bg.shape[-1] != 3 or robot.shape != bg.shape:
        raise ValueError("RGB arrays must have identical [H,W,3] shapes")
    shape = bg.shape[:2]
    if any(np.shape(a) != shape for a in (scene, sim, robot_mask, gripper_mask, human_mask)):
        raise ValueError("all depth/mask arrays must share the RGB pixel grid")
    if scene.dtype.kind != "f" or sim.dtype.kind != "f":
        raise ValueError("depth must be floating point meters, not raw sensor units")
    rm, gm, hm = [np.asarray(a, dtype=bool) for a in (robot_mask, gripper_mask, human_mask)]
    if np.any(gm & ~rm):
        raise ValueError("gripper_mask must be a subset of robot_mask")
    if np.any(rm & (~np.isfinite(sim) | (sim <= 0))):
        raise ValueError("robot pixels need finite positive rendered depth")
    # Unknown scene depth cannot decide occlusion: fail explicitly at gripper.
    dilated = binary_dilation(hm, structure=np.ones((5, 5), dtype=bool), iterations=1)
    if np.any(gm & ~dilated & (~np.isfinite(scene) | (scene <= 0))):
        raise ValueError("valid scene depth is required outside the human mask at gripper pixels")
    occluded = gm & ~dilated & (scene < sim)
    visible = rm & ~occluded  # arm body always overlays, per Appendix A.4
    result = bg.copy()
    result[visible] = robot[visible]
    return result, visible
