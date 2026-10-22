"""Panda assembly and pinhole camera, with explicit geometric TCP convention."""
import copy
import os
import xml.etree.ElementTree as ET
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]


def numbers(values):
    return " ".join(f"{x:.12g}" for x in np.asarray(values).ravel())


def quat_wxyz(matrix):
    return np.roll(Rotation.from_matrix(matrix).as_quat(), 1)


def add_camera(world, K, width, height):
    # MuJoCo principalpixel is opposite to the OpenCV offset. The half-pixel
    # term makes integer OpenCV coordinates refer to pixel centers.
    ET.SubElement(world, "camera", name="rgb", pos="0 0 0", quat="0 1 0 0",
                  ipd="0", resolution=f"{width} {height}",
                  sensorsize=f"{width} {height}", focalpixel=numbers([K[0, 0], K[1, 1]]),
                  principalpixel=numbers([width / 2 - K[0, 2] - .5,
                                         height / 2 - K[1, 2] - .5]))


def build_scene(config, bases, K=None, width=640, height=360):
    source = (ROOT / config["model_xml"]).resolve()
    xml = ET.parse(source).getroot()
    meshdir = source.parent / xml.find("compiler").get("meshdir", ".")
    # Name implicit mesh assets before making their filenames absolute.
    for mesh in xml.findall("asset/mesh"):
        mesh.set("name", mesh.get("name", Path(mesh.get("file")).stem))
        mesh.set("file", str(meshdir / mesh.get("file")))
    hand = xml.find(".//body[@name='hand']")
    ET.SubElement(hand, "site", name="geometric_tcp",
                  pos=numbers(config["tcp_in_hand_m"]),
                  quat=numbers(quat_wxyz(config["rotation_hand_from_gripper"])),
                  size="0.002", group="5")
    for i, geom in enumerate(xml.findall(".//worldbody//geom")):
        geom.set("name", f"geom_{i}")
    result = ET.Element("mujoco", model="pkdata_dual_panda_debug")
    ET.SubElement(result, "compiler", angle="radian", autolimits="true")
    ET.SubElement(result, "option", gravity="0 0 0", integrator="implicitfast")
    ET.SubElement(result, "statistic", extent="2", center="0 0 0.5")
    visual = ET.SubElement(result, "visual")
    ET.SubElement(visual, "global", offwidth=str(width), offheight=str(height))
    ET.SubElement(visual, "quality", offsamples="0", shadowsize="1024")
    ET.SubElement(visual, "map", znear="0.005", zfar="5")
    ET.SubElement(visual, "headlight", ambient="0.25 0.25 0.25", diffuse="0.35 0.35 0.35", specular="0.1 0.1 0.1")
    world = ET.SubElement(result, "worldbody")
    ET.SubElement(world, "light", pos="0 -1 0", dir="0 0 1", directional="true", castshadow="false",
                  diffuse="0.4 0.4 0.4", specular="0.1 0.1 0.1")
    if K is not None:
        add_camera(world, K, width, height)
    # Reference attributes used by this pinned Panda XML. All names/classes
    # are prefixed; assets may safely reference the same absolute mesh file.
    references = {"name", "class", "childclass", "mesh", "material", "joint",
                  "joint1", "joint2", "body1", "body2", "tendon"}
    for side, base in bases.items():
        robot = copy.deepcopy(xml)
        for element in robot.iter():
            for attribute in references & element.attrib.keys():
                element.set(attribute, side + "_" + element.get(attribute))
        body = robot.find("worldbody/body")
        body.set("pos", numbers(base[:3, 3]))
        body.set("quat", numbers(quat_wxyz(base[:3, :3])))
        world.append(body)
        for section in ("default", "asset", "tendon", "equality", "actuator", "contact"):
            if robot.find(section) is None:
                continue
            dst = result.find(section)
            if dst is None:
                dst = ET.SubElement(result, section)
            dst.extend(list(robot.find(section)))
    return ET.tostring(result, encoding="unicode")


def geom_labels(model):
    # 0 background, 1 left arm, 2 left gripper, 3 right arm, 4 right gripper.
    labels = np.zeros(model.ngeom, dtype=np.uint8)
    for gid in range(model.ngeom):
        bid = int(model.geom_bodyid[gid])
        name = model.body(bid).name
        side = next((s for s in ("left", "right") if name.startswith(s + "_")), None)
        if side is None:
            continue
        gripper = False
        while bid:
            if model.body(bid).name == side + "_hand":
                gripper = True
            bid = int(model.body_parentid[bid])
        labels[gid] = (1 if side == "left" else 3) + int(gripper)
    return labels


def contacts(model, data, labels, epsilon=1e-5):
    mujoco.mj_forward(model, data)
    counts = {"left_self": 0, "right_self": 0, "cross": 0}
    for contact in data.contact[:data.ncon]:
        if contact.dist >= -epsilon:
            continue
        a, b = labels[contact.geom1], labels[contact.geom2]
        if not a or not b:
            continue
        sa, sb = (a - 1) // 2, (b - 1) // 2
        counts["cross" if sa != sb else ("left_self" if sa == 0 else "right_self")] += 1
    return counts


class CameraRenderer:
    def __init__(self, model, height, width):
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.labels = geom_labels(model)
        self.option = mujoco.MjvOption()
        self.option.geomgroup[:] = 0
        self.option.geomgroup[2] = 1  # Visual meshes, never collision meshes.
        self.option.sitegroup[:] = 0

    def render(self, data):
        r = self.renderer
        r.update_scene(data, camera="rgb", scene_option=self.option)
        rgb = r.render().copy()
        r.enable_depth_rendering()
        depth = r.render().copy()
        r.disable_depth_rendering()
        r.enable_segmentation_rendering()
        seg = r.render().copy()
        r.disable_segmentation_rendering()
        valid = (seg[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & (seg[:, :, 0] >= 0)
        label = np.zeros(depth.shape, np.uint8)
        label[valid] = self.labels[seg[:, :, 0][valid]]
        depth[label == 0] = np.inf
        return rgb, depth, label

    def render_parts(self, data):
        r = self.renderer
        r.update_scene(data, camera="rgb", scene_option=self.option)
        r.enable_segmentation_rendering()
        seg = r.render().copy()
        r.disable_segmentation_rendering()
        valid = (seg[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & (seg[:, :, 0] >= 0)
        label = np.zeros(seg.shape[:2], np.uint8)
        label[valid] = self.labels[seg[:, :, 0][valid]]
        return label

    def close(self):
        self.renderer.close()
