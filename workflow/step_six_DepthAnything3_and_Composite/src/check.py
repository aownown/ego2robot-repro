import numpy as np
from composite import composite

shape = (16, 16)
bg = np.zeros((*shape, 3), np.uint8)
robot = np.full_like(bg, 200)
scene = np.full(shape, .2)
sim = np.full(shape, .3)
rm = np.zeros(shape, bool); rm[2, 2] = rm[10, 10] = True
gm = np.zeros(shape, bool); gm[10, 10] = True
hm = np.zeros(shape, bool)
out, visible = composite(bg, robot, scene, sim, rm, gm, hm)
assert visible[2, 2] and not visible[10, 10]  # arm overlays, gripper occludes
hm[10, 9] = True
assert composite(bg, robot, scene, sim, rm, gm, hm)[1][10, 10]
scene[:] = .4; hm[:] = False
assert composite(bg, robot, scene, sim, rm, gm, hm)[1][10, 10]
try:
    composite(bg, robot, scene.astype('uint16'), sim, rm, gm, hm)
except ValueError:
    pass
else:
    raise AssertionError("raw integer depth should be rejected")
print("Arm/gripper occlusion, human-mask dilation and depth type checks passed")
