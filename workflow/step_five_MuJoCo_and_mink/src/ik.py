"""Single end-effector IK core. Base search and dual-arm scene assembly come later."""
import numpy as np
import mink


def solve_pose(configuration, target_cam_ee, site_name, dt=.02, max_iters=100, tolerance=1e-5):
    """World must already equal the desired camera frame; target is SE(3), meters.

    Updates configuration in place; returns (feasible, pose_error). Does not check
    self/cross-arm collision. A generic geometric gripper frame needs a calibrated
    fixed transform to the embodiment's MuJoCo attachment site before calling.
    """
    task = mink.FrameTask(frame_name=site_name, frame_type="site",
                          position_cost=1., orientation_cost=1., lm_damping=1e-3)
    task.set_target(mink.SE3.from_matrix(np.asarray(target_cam_ee)))
    for _ in range(max_iters):
        err = task.compute_error(configuration)
        if np.linalg.norm(err[:3]) < tolerance and np.linalg.norm(err[3:]) < tolerance:
            return True, err
        vel = mink.solve_ik(configuration, [task], dt, solver="quadprog", damping=1e-3)
        configuration.integrate_inplace(vel, dt)
    err = task.compute_error(configuration)
    return bool(np.linalg.norm(err[:3]) < tolerance and np.linalg.norm(err[3:]) < tolerance), err
