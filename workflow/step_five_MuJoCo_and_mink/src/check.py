import os
os.environ.setdefault("MUJOCO_GL", "egl")
from pathlib import Path
import numpy as np
import mujoco
import mink
import qpsolvers
from ik import solve_pose

assert "quadprog" in qpsolvers.available_solvers
# Three translational DoFs give an independently known reachable target.
model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body>
<joint name="x" type="slide" axis="1 0 0"/><joint name="y" type="slide" axis="0 1 0"/>
<joint name="z" type="slide" axis="0 0 1"/><geom type="sphere" size=".02" mass="1"/>
<site name="tcp"/></body></worldbody></mujoco>''')
c = mink.Configuration(model)
target = np.eye(4); target[:3, 3] = [.03, -.02, .04]
success, error = solve_pose(c, target, "tcp")
assert success, error
np.testing.assert_allclose(c.q, target[:3, 3], atol=1e-5)
root = Path(__file__).resolve().parents[1]
panda = mujoco.MjModel.from_xml_path(str(root / "repos/mujoco_menagerie/franka_emika_panda/scene.xml"))
data = mujoco.MjData(panda)
mujoco.mj_forward(panda, data)
with mujoco.Renderer(panda, height=120, width=160) as renderer:
    renderer.update_scene(data)
    rgb = renderer.render().copy()
    renderer.enable_depth_rendering()
    depth = renderer.render().copy()
    renderer.disable_depth_rendering()
    renderer.enable_segmentation_rendering()
    segmentation = renderer.render().copy()
assert rgb.shape == (120, 160, 3) and depth.shape == (120, 160)
assert segmentation.shape == (120, 160, 2) and np.isfinite(depth).all()
print("quadprog IK, Panda MJCF, EGL RGB/depth/segmentation checks passed")
