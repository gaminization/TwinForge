"""
joint_detector.py — Detect joints between adjacent robot links.

Demo:   uses the exact arduino_robot_arm joint definitions.
Real:   analyses geometry at link boundaries to infer axes + limits.
"""
import logging
import numpy as np
from dataclasses import dataclass
from typing import List, Optional

log = logging.getLogger("twinforge.joints")


@dataclass
class Joint:
    id:          str
    parent_id:   str
    child_id:    str
    type:        str            # "revolute" | "prismatic" | "fixed"
    axis:        np.ndarray    # unit vector in parent frame
    origin_xyz:  np.ndarray    # joint origin in parent frame
    origin_rpy:  np.ndarray    # roll-pitch-yaw
    limit_lower: float
    limit_upper: float
    confidence:  float
    world_frame: bool = False   # True: origin_xyz is in world coords (real scan)


# ──────────────────────────────────────────────────────────────
# arduino_robot_arm exact joint definitions — from real URDF
# robot_arm_pkg/urdf/arduino_robot_arm.urdf
# ──────────────────────────────────────────────────────────────
_DEMO_JOINTS = [
    # (parent, child, type, axis, xyz, rpy, lower_rad, upper_rad)
    ("base",  "waist",   "revolute", [0,0,1],
     [-0.00345, -0.00001, 0.04449], [0,       0,       0      ], -1.571, 1.571),
    ("waist", "arm1",    "revolute", [0,0,1],
     [ 0.00396,  0.01369, 0.03521], [1.5708,  0.03778, 1.5708 ], -1.571, 1.571),
    ("arm1",  "arm2",    "revolute", [0,0,1],
     [-0.00007,  0.11689,-0.00792], [-0.0013,-3.14159, 0.03778], -1.570, 1.570),
    ("arm2",  "gripper", "revolute", [0,0,1],
     [-0.00922,  0.12639,-0.00514], [3.14056,  0,       0      ], -1.570, 1.570),
]


def detect(comps, scan_data=None, overrides_path=None) -> List[Joint]:
    if getattr(scan_data, '_demo_link_meta', None) is not None:
        return _demo_joints(comps)
    joints = _geometry_joints(comps)
    if overrides_path:
        joints = apply_overrides(joints, overrides_path)
    return joints


def apply_overrides(joints, path) -> List[Joint]:
    """
    Replace detected joints with user-confirmed ones.
    JSON: [{"parent":"link_0","child":"link_1","type":"revolute",
            "axis":[0,0,1],"xyz":[x,y,z],"lower":-1.57,"upper":1.57}, ...]
    "xyz" is in WORLD coordinates of the scan; omitted -> keep detected position.
    """
    import json
    spec = _read_spec(path)
    by_pair = {(j.parent_id, j.child_id): j for j in joints}
    out = []
    for i, e in enumerate(spec):
        e.setdefault("parent", f"link_{i}")          # chain mode: joints listed base -> tip
        e.setdefault("child", f"link_{i + 1}")
        base = by_pair.get((e["parent"], e["child"]))
        xyz = np.array(e.get("xyz", base.origin_xyz if base else [0, 0, 0]), float)
        ax = np.array(e.get("axis", base.axis if base else [0, 0, 1]), float)
        ax = ax / (np.linalg.norm(ax) + 1e-12)
        out.append(Joint(
            id=f"joint_{e['parent']}_{e['child']}", parent_id=e["parent"], child_id=e["child"],
            type=e.get("type", "revolute"), axis=ax, origin_xyz=xyz, origin_rpy=np.zeros(3),
            limit_lower=float(e.get("lower", -1.57)), limit_upper=float(e.get("upper", 1.57)),
            confidence=1.0, world_frame=True))
    log.info(f"Applied {len(out)} user-specified joints from {path}")
    return out


def _read_spec(path):
    import json
    spec = json.load(open(path))
    return spec["joints"] if isinstance(spec, dict) else spec


def chain_positions(path):
    """World positions of the joints in a chain-mode file (entries without parent/child), else None."""
    spec = _read_spec(path)
    if all("parent" not in e and "child" not in e for e in spec) and all("xyz" in e for e in spec):
        return np.array([e["xyz"] for e in spec], float)
    return None


def write_template(joints, path):
    """Dump detected joints as an editable JSON the user can correct and pass via --joints."""
    import json
    json.dump([{"parent": j.parent_id, "child": j.child_id, "type": j.type,
                "axis": [round(float(v), 4) for v in j.axis],
                "xyz": [round(float(v), 5) for v in j.origin_xyz],
                "lower": round(float(j.limit_lower), 3), "upper": round(float(j.limit_upper), 3)}
               for j in joints], open(path, "w"), indent=2)


# ──────────────────────────────────────────────────────────────
# Demo: exact joints
# ──────────────────────────────────────────────────────────────

def _demo_joints(comps) -> List[Joint]:
    id_map = {c.id: c for c in comps}
    joints = []
    for parent_id, child_id, jtype, axis, xyz, rpy, lo, hi in _DEMO_JOINTS:
        if parent_id not in id_map or child_id not in id_map:
            continue
        joints.append(Joint(
            id=f"joint_{parent_id}_{child_id}",
            parent_id=parent_id,
            child_id=child_id,
            type=jtype,
            axis=np.array(axis, dtype=float),
            origin_xyz=np.array(xyz, dtype=float),
            origin_rpy=np.array(rpy, dtype=float),
            limit_lower=lo,
            limit_upper=hi,
            confidence=0.99,
        ))
    log.info(f"Demo joints ({len(joints)}):")
    for j in joints:
        log.info(f"  {j.id}: {j.type} axis={j.axis} limits=[{j.limit_lower:.3f},{j.limit_upper:.3f}]")
    return joints


# ──────────────────────────────────────────────────────────────
# Real scan: geometry-based joint detection
# ──────────────────────────────────────────────────────────────

def _geometry_joints(comps) -> List[Joint]:
    """
    For each adjacent pair of components (sorted bottom→top),
    estimate the joint axis from the contact-zone geometry.
    """
    # Sort by height (vertical chain)
    sorted_comps = sorted(comps, key=lambda c: c.origin[2])

    joints = []
    for i in range(len(sorted_comps) - 1):
        parent = sorted_comps[i]
        child  = sorted_comps[i + 1]

        axis, conf = _estimate_axis(parent, child, i)
        lo, hi = _estimate_limits(parent, child, i)
        origin = _contact_point(parent, child)
        jtype = "fixed" if conf < 0.3 else "revolute"

        joints.append(Joint(
            id=f"joint_{parent.id}_{child.id}",
            parent_id=parent.id,
            child_id=child.id,
            type=jtype,
            axis=axis,
            origin_xyz=origin,
            origin_rpy=np.zeros(3),
            limit_lower=lo,
            limit_upper=hi,
            confidence=conf,
            world_frame=True,
        ))

    log.info(f"Geometry joints: {len(joints)} detected")
    return joints


def _estimate_axis(parent, child, idx: int) -> tuple:
    """
    Estimate rotation axis from link geometry at the contact region.
    Uses PCA of the contact-zone points.
    """
    import open3d as o3d

    p_pts = np.asarray(parent.pcd.points)
    c_pts = np.asarray(child.pcd.points)
    if len(p_pts) == 0 or len(c_pts) == 0:
        return np.array([0, 0, 1]), 0.5

    # Contact zone = top 20% of parent + bottom 20% of child
    p_z = p_pts[:, 2]
    c_z = c_pts[:, 2]
    p_zone = p_pts[p_z > np.percentile(p_z, 80)]
    c_zone = c_pts[c_z < np.percentile(c_z, 20)]

    zone = np.vstack([p_zone, c_zone]) if len(p_zone) and len(c_zone) else p_pts
    if len(zone) < 3:
        return np.array([0, 0, 1]), 0.5

    # PCA: smallest eigenvector = normal to flat contact surface = rotation axis
    cov = np.cov(zone.T)
    evals, evecs = np.linalg.eigh(cov)
    # Sort ascending by eigenvalue
    order = np.argsort(evals)
    smallest_axis = evecs[:, order[0]]    # normal to plane
    largest_axis  = evecs[:, order[2]]    # principal elongation

    # For a cylindrical link, rotation axis = elongation axis
    candidate = largest_axis / (np.linalg.norm(largest_axis) + 1e-9)

    # Snap to nearest cardinal axis
    candidates = [np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1]),
                  np.array([-1,0,0]), np.array([0,-1,0]), np.array([0,0,-1])]
    dots = [abs(float(np.dot(candidate, c))) for c in candidates]
    best = candidates[int(np.argmax(dots))]
    confidence = max(dots)

    return best.astype(float), float(confidence)


def _estimate_limits(parent, child, idx: int) -> tuple:
    """Estimate joint limits based on link geometry ratios."""
    # Base link: full rotation
    if idx == 0:
        return -3.14159, 3.14159
    # Other joints: estimate from link height ratio
    h_parent = parent.extent[2]
    h_child  = child.extent[2]
    ratio = h_child / (h_parent + 1e-6)
    # Longer child → wider range
    span = min(3.14159, max(0.5236, ratio * 2.0))
    return -span / 2, span / 2


def _contact_point(parent, child) -> np.ndarray:
    """Estimate joint origin as midpoint between top of parent and bottom of child."""
    p_bb = parent.mesh.get_axis_aligned_bounding_box() if parent.mesh else None
    c_bb = child.mesh.get_axis_aligned_bounding_box() if child.mesh else None

    p_top = float(p_bb.max_bound[2]) if p_bb else parent.origin[2] + parent.extent[2]
    c_bot = float(c_bb.min_bound[2]) if c_bb else child.origin[2]
    mid_z = (p_top + c_bot) / 2

    cx = (parent.origin[0] + child.origin[0]) / 2
    cy = (parent.origin[1] + child.origin[1]) / 2
    return np.array([cx, cy, mid_z])
