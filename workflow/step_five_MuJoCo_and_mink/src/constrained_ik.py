"""Collision-constrained Panda IK. Final feasibility is checked independently."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import xml.etree.ElementTree as ET
import mink
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from robot_scene import build_scene, geom_labels, quat_wxyz
from collision_distance import DistanceOracle


def constrained_scene(config, bases, K=None, width=640, height=360):
    xml = ET.fromstring(build_scene(config, bases, K, width, height))
    xml.set("model", "pkdata_constrained_dual_panda")
    # A mocap body is independent of the fixed robot base in collision filtering.
    body = ET.SubElement(xml.find("worldbody"), "body", name="camera_keepout_body", mocap="true")
    ET.SubElement(body, "geom", name="camera_keepout", type="sphere",
                  size=str(config["camera_keepout_radius_m"]), group="5",
                  contype="1", conaffinity="1", rgba="0 0 0 0")
    return ET.tostring(xml, encoding="unicode")


class CompactCollision(mink.CollisionAvoidanceLimit):
    def __init__(self, *args, include_all_pairs=False, **kwargs):
        super().__init__(*args, **kwargs)
        if include_all_pairs:
            # A virtual world obstacle is not an adjacent robot link. Mink's
            # default parent/weld filter otherwise drops the base and joint-1
            # camera pairs. Explicit obstacle pairs must include fixed bodies.
            groups = kwargs.get("geom_pairs", args[1] if len(args) > 1 else [])
            pairs = set()
            for a_group, b_group in self._collision_pairs_to_geom_id_pairs(groups):
                for a in a_group:
                    for b in b_group:
                        if a != b:
                            pairs.add((min(a, b), max(a, b)))
            self.geom_id_pairs = sorted(pairs)
            self.max_num_contacts = len(pairs)
            self._init_broadphase(self.model)
        self.oracle = DistanceOracle(self.model)

    def compute_qp_inequalities(self, configuration, dt):
        rows, upper = [], []
        data = configuration.data
        for idx in self._broadphase_survivors(data):
            a, b = self.geom_id_pairs[idx]
            distance, points, normal, certified = self.oracle.query(data, a, b, self.collision_detection_distance)
            if distance >= self.collision_detection_distance - 1e-12:
                continue
            mujoco.mj_jac(self.model, data, self._jac1, None, points[:3], self.model.geom_bodyid[a])
            mujoco.mj_jac(self.model, data, self._jac2, None, points[3:], self.model.geom_bodyid[b])
            row = normal @ (self._jac2 - self._jac1)
            rows.append((-1 if distance >= 0 else 1) * row)
            # mink solves for delta_q, then divides by dt to return velocity.
            # Therefore the clearance budget here is a displacement, not m/s.
            upper.append(self.gain * max(distance - self.minimum_distance_from_collisions, 0) + self.bound_relaxation * dt)
        return mink.Constraint(G=np.array(rows), h=np.array(upper)) if rows else mink.Constraint()


class FrameDisplacementLimit(mink.Limit):
    def __init__(self, model, joint_ids, maximum):
        self.ids = np.array(joint_ids)
        self.matrix = np.eye(model.nv)[self.ids]
        self.maximum = maximum
        self.anchor = None

    def compute_qp_inequalities(self, configuration, dt):
        if self.anchor is None:
            return mink.Constraint()
        delta = configuration.q[self.ids] - self.anchor[self.ids]
        return mink.Constraint(G=np.vstack((self.matrix, -self.matrix)),
                               h=np.r_[self.maximum - delta, self.maximum + delta])


class ConstrainedIK:
    def __init__(self, config, sides=("left",), K=None, size=(640, 360)):
        self.config = config
        self.sides = tuple(sides)
        self.xml = constrained_scene(config, {s: np.eye(4) for s in sides}, K, *size)
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.configuration = mink.Configuration(self.model)
        self.labels = geom_labels(self.model)
        self.camera_id = self.model.geom("camera_keepout").id
        self.home = np.tile(config["home_qpos"], len(sides)).astype(float)
        self.joint_ids = np.array([9 * k + j for k in range(len(sides)) for j in range(7)])
        geoms = np.flatnonzero((self.model.geom_contype != 0) & (self.labels != 0)).tolist()
        self.robot_limit = CompactCollision(self.model, [(geoms, geoms)],
                                            minimum_distance_from_collisions=config["collision_clearance_m"],
                                            collision_detection_distance=config["collision_detection_distance_m"], gain=.7)
        self.camera_limit = CompactCollision(self.model, [(geoms, [self.camera_id])],
                                             minimum_distance_from_collisions=config["collision_clearance_m"],
                                             collision_detection_distance=config["collision_detection_distance_m"], gain=.7,
                                             include_all_pairs=True)
        self.tasks = {s: mink.FrameTask(s + "_geometric_tcp", "site", position_cost=1.,
                                       orientation_cost=1., lm_damping=1e-3) for s in sides}
        self.site_ids = [self.model.site(s + "_geometric_tcp").id for s in sides]
        self.posture = mink.PostureTask(self.model, cost=config["posture_cost"])
        self.frame_limit = FrameDisplacementLimit(self.model, self.joint_ids, config["maximum_joint_step_rad"])
        velocity_limits = {s + "_joint" + str(j): 6.0 for s in sides for j in range(1, 8)}
        self.limits = [mink.ConfigurationLimit(self.model), mink.VelocityLimit(self.model, velocity_limits),
                       self.robot_limit, self.camera_limit, self.frame_limit]
        self.base_ids = {s: self.model.body(s + "_link0").id for s in sides}
        self.reference = self.home.copy()

    def set_bases(self, bases):
        for side, base in bases.items():
            self.model.body_pos[self.base_ids[side]] = base[:3, 3]
            self.model.body_quat[self.base_ids[side]] = quat_wxyz(base[:3, :3])
        self.configuration.update(self.home)

    def set_widths(self, q, widths):
        q = np.array(q, copy=True)
        for k, side in enumerate(self.sides):
            q[k * 9 + 7:k * 9 + 9] = np.clip(widths[side], 0, self.config["maximum_gripper_width_m"]) / 2
        return q

    def clearance(self):
        """Distance lower bounds capped at detection distance; includes camera."""
        values = []
        for limit in (self.robot_limit, self.camera_limit):
            minimum = self.config["collision_detection_distance_m"]
            # Broadphase only rejects pairs provably farther than detection range.
            for idx in limit._broadphase_survivors(self.configuration.data):
                a, b = limit.geom_id_pairs[idx]
                distance, _, _, _ = limit.oracle.query(self.configuration.data, a, b,
                                                      self.config["collision_detection_distance_m"])
                minimum = min(minimum, distance)
            values.append(float(minimum))
        return values

    def solve(self, poses, widths, seed, reference=None, continuous=False):
        q = self.set_widths(seed, widths)
        self.configuration.update(q)
        self.frame_limit.anchor = np.array(seed, copy=True) if continuous else None
        self.posture.set_target(q if reference is None else self.set_widths(reference, widths))
        for s in self.sides:
            self.tasks[s].set_target(mink.SE3.from_matrix(poses[s]))
        tol = self.config["ik"]["tolerance"]
        failed = False
        it = 0
        for it in range(self.config["ik"]["iterations"]):
            errors = [task.compute_error(self.configuration) for task in self.tasks.values()]
            if all(np.linalg.norm(e[:3]) < tol and np.linalg.norm(e[3:]) < tol for e in errors):
                break
            try:
                vel = mink.solve_ik(self.configuration, [*self.tasks.values(), self.posture], .02,
                                     solver="quadprog", damping=1e-5, limits=self.limits, safety_break=True)
            except mink.NoSolutionFound:
                failed = True
                break
            self.configuration.integrate_inplace(vel, .02)
        q = self.set_widths(self.configuration.q, widths)
        self.configuration.update(q)
        actual, pe, ae = [], [], []
        for side, site in zip(self.sides, self.site_ids):
            T = np.eye(4)
            T[:3, 3] = self.configuration.data.site_xpos[site]
            T[:3, :3] = self.configuration.data.site_xmat[site].reshape(3, 3)
            actual.append(T)
            pe.append(float(np.linalg.norm(T[:3, 3] - poses[side][:3, 3])))
            ae.append(float(Rotation.from_matrix(T[:3, :3].T @ poses[side][:3, :3]).magnitude()))
        clearance = self.clearance()
        converged = max(pe) < tol and max(ae) < tol and not failed
        safe = min(clearance) >= self.config["collision_clearance_m"] - 1e-6
        return {"q": q, "actual": np.array(actual), "position_error": pe, "rotation_error": ae,
                "converged": converged, "safe": safe, "clearance": clearance, "iterations": it + 1,
                "qp_failed": failed, "accepted": bool(converged and safe)}
