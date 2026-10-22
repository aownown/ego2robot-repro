import numpy as np
from scipy.spatial.transform import Rotation
from retarget import retarget, interpolate_gaps

k = np.zeros((30, 21, 3))
k[:, 0] = [0, -.1, .3]
k[:, 4] = [.04, 0, .3]
k[:, 8] = [-.04, 0, .3]
k[:, 12] = [-.04, 0, .3]
for hand in ("left", "right"):
    out = retarget(k, hand)
    r = out["rotation_cam_gripper"]
    np.testing.assert_allclose(out["tcp_cam_m"], np.tile([0, 0, .3], (30, 1)), atol=1e-10)
    np.testing.assert_allclose(out["gripper_width_m"], .08, atol=1e-10)
    np.testing.assert_allclose(r @ r.transpose(0, 2, 1), np.tile(np.eye(3), (30, 1, 1)), atol=1e-10)
    np.testing.assert_allclose(np.linalg.det(r), 1, atol=1e-10)
    for n in (1, 2, 4):
        assert len(retarget(k[:n], hand)["tcp_cam_m"]) == n
broken = k.copy(); broken[5] = np.nan
try:
    retarget(broken, "right")
except ValueError:
    pass
else:
    raise AssertionError("missing tracks must not silently become robot commands")

# A camera rigid transform must rotate/translate the TCP and axes consistently.
transform = Rotation.from_euler("xyz", [20, -15, 35], degrees=True).as_matrix()
translation = np.array([.1, .2, .4])
for hand in ("left", "right"):
    before = retarget(k, hand)
    after = retarget(k @ transform.T + translation, hand)
    np.testing.assert_allclose(after["tcp_cam_m"], before["tcp_cam_m"] @ transform.T + translation, atol=1e-10)
    np.testing.assert_allclose(after["rotation_cam_gripper"], transform @ before["rotation_cam_gripper"], atol=1e-10)
    np.testing.assert_allclose(after["gripper_width_m"], before["gripper_width_m"], atol=1e-10)

# Inferred observations cannot steer the output, even if their geometry is corrupted.
observed = np.ones(30, dtype=bool)
observed[10:12] = False
corrupted = k.copy()
corrupted[10:12] += [10, -8, 5]
baseline = retarget(k, "right", observed_valid=observed)
repaired = retarget(corrupted, "right", observed_valid=observed)
for key in ("tcp_cam_m", "rotation_cam_gripper", "gripper_width_m"):
    np.testing.assert_allclose(baseline[key], repaired[key], atol=1e-10)
assert repaired["gap_interpolated"].sum() == 2

# A short rotational gap across +/-180 degrees follows the short arc, not zero.
pos = np.array([[0, 0, .3], [100, 100, 100], [2, 0, .3]], dtype=float)
quat = Rotation.from_euler("z", [[179], [0], [-179]], degrees=True).as_quat()
p, q, w = interpolate_gaps(pos, quat, np.array([.02, 20, .04]), [True, False, True])
np.testing.assert_allclose(p[1], [1, 0, .3])
np.testing.assert_allclose(w[1], .03)
np.testing.assert_allclose(Rotation.from_quat(q[1]).apply([1, 0, 0]), [-1, 0, 0], atol=1e-10)

closed = k.copy()
closed[5, 4] = closed[5, 8]
held = retarget(closed, "right")
assert held["orientation_held"][5]
np.testing.assert_allclose(held["rotation_raw_cam_gripper"][5], held["rotation_raw_cam_gripper"][4])

spike = k.copy()
spike[15] += [.2, 0, 0]
filtered = retarget(spike, "right")
assert filtered["replaced"][15]
np.testing.assert_allclose(filtered["tcp_cam_m"], baseline["tcp_cam_m"], atol=1e-10)

for mask in (np.r_[False, np.ones(29, bool)], np.r_[np.ones(29, bool), False],
             np.r_[np.ones(5, bool), np.zeros(11, bool), np.ones(14, bool)]):
    try:
        retarget(k, "right", observed_valid=mask)
    except ValueError:
        pass
    else:
        raise AssertionError("unsupported gaps must not become unmarked targets")
print("Passed: geometry, rigid transforms, rotations, gap exclusion/interpolation, wraparound, degenerate pose and jump filtering")
