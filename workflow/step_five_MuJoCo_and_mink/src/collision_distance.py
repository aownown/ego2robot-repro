"""Conservative separating-plane fallback for MuJoCo native CCD zero distances.

Upstream issue: https://github.com/google-deepmind/mujoco/issues/3383
Never replace a zero by witness length: that is NOT a certified lower bound.
Only a positive support-plane gap can certify separation of the actual geoms.
"""
import mujoco
import numpy as np


class DistanceOracle:
    def __init__(self, model):
        self.model = model
        self.vertices = {}
        self.fallback_count = 0
        for gid in range(model.ngeom):
            if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
                mid = model.geom_dataid[gid]
                start, count = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
                self.vertices[gid] = model.mesh_vert[start:start + count].astype(float)

    def support(self, data, gid, direction):
        R = data.geom_xmat[gid].reshape(3, 3)
        local = R.T @ direction
        size = self.model.geom_size[gid]
        kind = self.model.geom_type[gid]
        if gid in self.vertices:
            vertices = self.vertices[gid]
            point = vertices[np.argmax(vertices @ local)]
        elif kind == mujoco.mjtGeom.mjGEOM_BOX:
            point = np.where(local >= 0, size, -size)
        elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
            point = size[0] * local / np.linalg.norm(local)
        elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
            point = size[0] * local / np.linalg.norm(local)
            point[2] += size[1] * (1 if local[2] >= 0 else -1)
        else:
            raise ValueError(f"Unsupported support geometry {int(kind)}")
        return R @ point + data.geom_xpos[gid]

    def query(self, data, a, b, cap):
        points = np.zeros(6)
        distance = float(mujoco.mj_geomDistance(self.model, data, a, b, cap, points))
        delta = points[3:] - points[:3]
        length = np.linalg.norm(delta)
        normal = delta / length if length > 1e-12 else np.zeros(3)
        certified = False
        if distance == 0.0 and length > 1e-5:
            # A positive projected gap is a mathematical lower bound on convex
            # distance. Points are actual support vertices, not arbitrary witnesses.
            pa = self.support(data, a, normal)
            pb = self.support(data, b, -normal)
            gap = float(normal @ (pb - pa))
            if gap > 1e-8:
                distance = gap
                points[:] = np.r_[pa, pb]
                certified = True
                self.fallback_count += 1
        return distance, points, normal, certified
