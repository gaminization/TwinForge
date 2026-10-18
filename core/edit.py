"""
edit.py - post-scan correction of a generated twin (no re-scan needed).

Operates on a finished job directory  (<job>/<name>.urdf, <job>/meta.json, <job>/meshes/*.stl):

    edit_joint(job_dir, joint_id, type=, axis=, flip=, limit_lower=, limit_upper=, pivot_shift=)

* type        revolute | continuous | prismatic | fixed   ("fixed" = "this is not really a joint")
* axis        new axis in the joint frame (normalised); flip=True negates the current axis
* limits      radians (metres for prismatic)
* pivot_shift [dx,dy,dz] in the joint frame.  Moves the rotation centre WITHOUT moving any geometry:
              the child link frame moves by d, so child mesh vertices and the origins of the child's own
              joints are shifted by -d.
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

TYPES = ("revolute", "continuous", "prismatic", "fixed")


def _rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def _vec(s):
    return np.array([float(v) for v in s.split()])


def _fmt(v):
    return " ".join(f"{float(x):.5f}" for x in v)


def edit_joint(job_dir, joint_id: str, type=None, axis=None, flip=False,
               limit_lower=None, limit_upper=None, pivot_shift=None) -> dict:
    job_dir = Path(job_dir)
    urdf_path = next(job_dir.glob("*.urdf"))
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    joints = {j.get("name"): j for j in root.findall("joint")}
    if joint_id not in joints:
        raise KeyError(f"unknown joint {joint_id!r}; have {sorted(joints)}")
    j = joints[joint_id]
    meta_path = job_dir / "meta.json"
    meta = json.loads(meta_path.read_text())
    mj = next(m for m in meta["joints"] if m["id"] == joint_id)

    # ---- type / axis / limits ------------------------------------------------
    if type is not None:
        if type not in TYPES:
            raise ValueError(f"type must be one of {TYPES}")
        j.set("type", type); mj["type"] = type
    ax_el = j.find("axis")
    ax = _vec(ax_el.get("xyz")) if ax_el is not None else np.array([0, 0, 1.0])
    if axis is not None:
        a = np.asarray(axis, float)
        if np.linalg.norm(a) < 1e-9:
            raise ValueError("axis must be non-zero")
        ax = a / np.linalg.norm(a)
    if flip:
        ax = -ax
    if ax_el is None:
        ax_el = ET.SubElement(j, "axis")
    ax_el.set("xyz", " ".join(f"{x:.3f}" for x in ax)); mj["axis"] = [float(x) for x in ax]

    jt = j.get("type")
    lim = j.find("limit")
    if jt == "fixed":
        if lim is not None:
            j.remove(lim)
    elif jt == "continuous":
        if lim is not None:
            j.remove(lim)
        ET.SubElement(j, "limit", effort="10.0", velocity="1.0")
    else:
        if lim is None:
            lim = ET.SubElement(j, "limit", lower="-1.5708", upper="1.5708", effort="10.0", velocity="1.0")
        if limit_lower is not None:
            lim.set("lower", f"{float(limit_lower):.5f}")
        if limit_upper is not None:
            lim.set("upper", f"{float(limit_upper):.5f}")
        if float(lim.get("lower")) > float(lim.get("upper")):
            raise ValueError("lower limit above upper limit")
        mj["limit_lower"] = float(lim.get("lower")); mj["limit_upper"] = float(lim.get("upper"))

    # ---- pivot move (geometry stays put) -----------------------------------------
    if pivot_shift is not None and np.linalg.norm(pivot_shift) > 0:
        d = np.asarray(pivot_shift, float)
        org = j.find("origin")
        xyz = _vec(org.get("xyz")); rpy = _vec(org.get("rpy"))
        xyz = xyz + _rpy_to_R(*rpy) @ d
        org.set("xyz", _fmt(xyz)); mj["origin_xyz"] = [float(x) for x in xyz]
        child = j.find("child").get("link")
        import trimesh
        stl = job_dir / "meshes" / f"{child}.stl"
        if stl.exists():
            m = trimesh.load_mesh(stl)
            m.vertices = m.vertices - d
            m.export(stl)
        for cj in root.findall("joint"):                 # grand-children keep their world position
            if cj.find("parent").get("link") == child:
                o = cj.find("origin"); o.set("xyz", _fmt(_vec(o.get("xyz")) - d))
                for mm in meta["joints"]:
                    if mm["id"] == cj.get("name"):
                        mm["origin_xyz"] = [float(x) for x in _vec(o.get("xyz"))]

    ET.indent(tree)
    tree.write(urdf_path, xml_declaration=True, encoding="utf-8")
    meta["edited"] = True
    meta_path.write_text(json.dumps(meta, indent=2))
    return mj
