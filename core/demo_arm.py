"""
demo_arm.py — Builds the arduino_robot_arm digital twin from the REAL STL files.

The STL meshes are the actual files from:
  https://github.com/smart-methods/arduino_robot_arm

Joint definitions come directly from the robot's URDF:
  robot_arm_pkg/urdf/arduino_robot_arm.urdf
"""
import logging
import numpy as np
import open3d as o3d
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional

log = logging.getLogger("twinforge.demo_arm")

ASSETS_DIR = Path(__file__).parent.parent / "assets" / "arduino_robot_arm"

# ──────────────────────────────────────────────────────────────
# Real link definitions — directly from the URDF
# STL scale factor in URDF is 0.00098 (millimetre → metre, ~×0.001)
# ──────────────────────────────────────────────────────────────
# Each entry:
#   (link_id, stl_file, visual_rpy, scale, color_rgb)
# visual_rpy: from <visual><origin rpy="..."/> in URDF
# scale:      from <mesh scale="..."/> in URDF (0.00098 for most)
#
_LINK_DEFS = [
    ("base",    "base.stl",    [1.5708, 0,       0      ], 0.00098, (80,  80,  90 )),
    ("waist",   "waist.stl",   [1.5708, 0,       0      ], 0.00098, (45, 105, 195 )),
    ("arm1",    "arm1.stl",    [0,      0,       0      ], 0.00098, (40, 135, 210 )),
    ("arm2",    "arm2.stl",    [-1.4889,-1.5721, 1.5707 ], 0.00098, (45, 165, 175 )),
    ("gripper", "gripper.stl", [0.6012,-1.5685, -1.5701 ], 0.00391, (210, 120, 40 )),
]

# Real joint definitions — directly from the URDF
# (joint_id, parent, child, type, axis, xyz, rpy, lower, upper)
_JOINT_DEFS = [
    ("base_joint", "base", "waist",   "revolute", [0,0,1],
     [-0.00345, -0.000010, 0.04449], [0,       0,      0     ], -1.571, 1.571),
    ("shoulder",   "waist", "arm1",   "revolute", [0,0,1],
     [ 0.00396,  0.01369,  0.03521], [1.5708, 0.03778, 1.5708], -1.571, 1.571),
    ("elbow",      "arm1",  "arm2",   "revolute", [0,0,1],
     [-0.00007,  0.11689, -0.00792], [-0.0013,-3.14159,0.03778], -1.570, 1.570),
    ("wrist",      "arm2",  "gripper","revolute", [0,0,1],
     [-0.00922,  0.12639, -0.00514], [3.14056, 0,      0     ], -1.570, 1.570),
]


@dataclass
class LinkMeta:
    id:     str
    label:  str
    mesh:   o3d.geometry.TriangleMesh   # mesh in world frame (FK at zero pose)
    pcd:    o3d.geometry.PointCloud
    origin: np.ndarray
    extent: np.ndarray
    local_mesh: o3d.geometry.TriangleMesh = None  # mesh in the link's own frame (for URDF)


def _rpy_matrix(rpy):
    r, p, y = rpy
    Rx = np.array([[1,0,0],[0,np.cos(r),-np.sin(r)],[0,np.sin(r),np.cos(r)]])
    Ry = np.array([[np.cos(p),0,np.sin(p)],[0,1,0],[-np.sin(p),0,np.cos(p)]])
    Rz = np.array([[np.cos(y),-np.sin(y),0],[np.sin(y),np.cos(y),0],[0,0,1]])
    return Rz @ Ry @ Rx


def _link_world_transforms():
    """Forward kinematics at zero joint angles, from the URDF joint table."""
    T = {"base": np.eye(4)}
    for jid, parent, child, jt, axis, xyz, rpy, lo, hi in _JOINT_DEFS:
        J = np.eye(4); J[:3,:3] = _rpy_matrix(rpy); J[:3,3] = xyz
        T[child] = T[parent] @ J
    return T


def _load_and_transform(stl_path: Path, rpy, scale: float,
                        color_rgb) -> Optional[o3d.geometry.TriangleMesh]:
    """Load STL, apply URDF visual transform, scale, colour."""
    if not stl_path.exists():
        log.warning(f"STL not found: {stl_path}")
        return None

    mesh = o3d.io.read_triangle_mesh(str(stl_path))
    if len(np.asarray(mesh.vertices)) == 0:
        log.warning(f"Empty STL: {stl_path}")
        return None

    # Apply scale
    mesh.scale(scale, center=[0, 0, 0])

    # Apply visual RPY rotation (from URDF <visual><origin rpy="..."/>)
    r, p, y = rpy
    Rx = np.array([[1,0,0],[0,np.cos(r),-np.sin(r)],[0,np.sin(r),np.cos(r)]])
    Ry = np.array([[np.cos(p),0,np.sin(p)],[0,1,0],[-np.sin(p),0,np.cos(p)]])
    Rz = np.array([[np.cos(y),-np.sin(y),0],[np.sin(y),np.cos(y),0],[0,0,1]])
    R  = Rz @ Ry @ Rx
    mesh.rotate(R, center=[0, 0, 0])

    # Paint colour
    mesh.paint_uniform_color([c / 255.0 for c in color_rgb])
    mesh.compute_vertex_normals()
    return mesh


def build_demo_scan(pts_per_link: int = 1500) -> Tuple[o3d.geometry.PointCloud, List[LinkMeta]]:
    """
    Load the real arduino_robot_arm STL meshes and assemble them
    into a combined point cloud that simulates a 3D scan.

    Returns:
        full_pcd   – merged point cloud (what SfM would produce)
        link_metas – per-link mesh + pcd for segmentation
    """
    link_metas: List[LinkMeta] = []
    all_points = []
    all_colors = []

    link_T = _link_world_transforms()

    for lid, stl_name, visual_rpy, scale, color in _LINK_DEFS:
        stl_path = ASSETS_DIR / stl_name
        local = _load_and_transform(stl_path, visual_rpy, scale, color)

        if local is None:
            log.error(f"Skipping link {lid} — STL missing or empty")
            continue

        # World-frame copy = link FK pose applied to the link-local mesh
        import copy
        mesh = copy.deepcopy(local)
        mesh.transform(link_T[lid])

        # Sample point cloud from mesh surface
        pcd = mesh.sample_points_poisson_disk(pts_per_link)

        # Add realistic reconstruction noise
        pts = np.asarray(pcd.points)
        noise = np.random.normal(0, 0.0006, pts.shape)   # 0.6 mm noise
        pcd.points = o3d.utility.Vector3dVector(pts + noise)

        bb  = mesh.get_axis_aligned_bounding_box()
        ext = np.asarray(bb.max_bound) - np.asarray(bb.min_bound)
        ori = np.asarray(bb.min_bound).copy()

        lm = LinkMeta(id=lid, label=_label(lid), mesh=mesh, pcd=pcd,
                      origin=ori, extent=ext, local_mesh=local)
        link_metas.append(lm)

        all_points.append(np.asarray(pcd.points))
        all_colors.append(np.asarray(pcd.colors))

    if not all_points:
        raise RuntimeError("No STL meshes loaded — check assets/arduino_robot_arm/")

    full_pcd = o3d.geometry.PointCloud()
    full_pcd.points = o3d.utility.Vector3dVector(np.vstack(all_points))
    full_pcd.colors = o3d.utility.Vector3dVector(np.vstack(all_colors))

    log.info(f"Loaded real arduino_robot_arm: {len(link_metas)} links, "
             f"{len(full_pcd.points):,} points")
    return full_pcd, link_metas


def _label(lid: str) -> str:
    return {"base":"base","waist":"shoulder_yaw",
            "arm1":"upper_arm","arm2":"forearm","gripper":"end_effector"}.get(lid, lid)
