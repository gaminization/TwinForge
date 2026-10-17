"""
urdf_builder.py — Generate a URDF file from segmented components + joints.

Each link gets:
  • An STL mesh file (exported from the actual reconstructed mesh)
  • Visual and collision elements referencing that STL
  • Inertial properties estimated from mesh volume

Joints use the detected axes and limits.
"""
import logging
import numpy as np
import open3d as o3d
from pathlib import Path
from typing import List
from xml.etree import ElementTree as ET
from xml.dom import minidom

log = logging.getLogger("twinforge.urdf")


def build(comps, joints, robot_name: str, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    mesh_dir = out_dir / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    urdf_path = out_dir / f"{robot_name}.urdf"

    # ── Export per-link STL meshes ────────────────────────────
    stl_paths = {}
    local_meshes, rel_joints = _localize(comps, joints)
    for comp in comps:
        mesh = local_meshes[comp.id]
        stl_name = f"{comp.id}.stl"
        stl_path = mesh_dir / stl_name
        o3d.io.write_triangle_mesh(str(stl_path), mesh)
        n_tris = len(np.asarray(mesh.triangles))
        log.info(f"  STL: {stl_name}  ({n_tris} triangles)")
        stl_paths[comp.id] = stl_path

    # ── Build URDF XML ────────────────────────────────────────
    robot = ET.Element("robot", name=robot_name)

    for comp in comps:
        stl_rel = f"meshes/{comp.id}.stl"
        _add_link(robot, comp, stl_rel, local_meshes[comp.id])

    for j in rel_joints:
        _add_joint(robot, j)
    build.relative_joints = rel_joints      # link-relative joints (for the web viewer)

    # ── Write formatted XML ───────────────────────────────────
    raw = ET.tostring(robot, encoding="unicode")
    pretty = minidom.parseString(raw).toprettyxml(indent="  ")
    # Remove the XML declaration line minidom adds
    lines = [l for l in pretty.splitlines() if l.strip() and not l.startswith("<?xml")]
    xml_str = '<?xml version="1.0"?>\n' + "\n".join(lines)
    urdf_path.write_text(xml_str)

    log.info(f"URDF written: {urdf_path}")
    return urdf_path


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _get_mesh(comp) -> o3d.geometry.TriangleMesh:
    """Use pre-built mesh if available (demo), else run Poisson."""
    if getattr(comp, 'mesh', None) is not None and len(np.asarray(comp.mesh.triangles)) > 0:
        return comp.mesh
    # Fallback: Poisson from component point cloud
    log.info(f"  Poisson reconstruction for {comp.id}…")
    pcd = comp.pcd
    if not pcd.has_normals():
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(k=10)
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
    # Trim low density
    d = np.asarray(dens)
    mask = d < np.percentile(d, 10)
    mesh.remove_vertices_by_mask(mask)
    mesh.compute_vertex_normals()
    return mesh


def _inertia_from_mesh(mesh: o3d.geometry.TriangleMesh, density: float = 1200.0):
    """Rough inertia approximation: treat as cylinder of same bounding box."""
    verts = np.asarray(mesh.vertices)
    bb = mesh.get_axis_aligned_bounding_box()
    dx, dy, dz = (np.asarray(bb.max_bound) - np.asarray(bb.min_bound))
    r = max(dx, dy) / 2
    h = dz
    vol = np.pi * r ** 2 * h
    mass = max(0.01, density * vol)
    ixx = mass * (3 * r ** 2 + h ** 2) / 12
    iyy = ixx
    izz = mass * r ** 2 / 2
    return mass, ixx, iyy, izz


def _localize(comps, joints):
    """
    URDF requires each link mesh in the LINK's own frame and each joint origin
    relative to its PARENT link. Convert whatever the pipeline produced into that.
      - comp.local_mesh present  -> already link-local, joints already relative (demo)
      - otherwise                -> mesh + joint origins are world-frame (real scan);
                                    link frame = translation to its parent-joint point.
    """
    import copy
    local, rel = {}, []
    frame = {}                              # link id -> world origin of its frame
    children = {j.child_id for j in joints}
    for c in comps:
        if c.id not in children:
            frame[c.id] = np.zeros(3)
    for j in joints:
        if getattr(j, "world_frame", False):
            frame[j.child_id] = np.asarray(j.origin_xyz, float)
    for c in comps:
        m = getattr(c, "local_mesh", None)
        if m is not None:
            local[c.id] = m
        else:
            m = copy.deepcopy(_get_mesh(c))
            m.translate(-frame.get(c.id, np.zeros(3)))
            local[c.id] = m
    for j in joints:
        j2 = copy.copy(j)
        if getattr(j, "world_frame", False):
            j2.origin_xyz = np.asarray(j.origin_xyz, float) - frame.get(j.parent_id, np.zeros(3))
        rel.append(j2)
    return local, rel


def _add_link(robot: ET.Element, comp, stl_rel: str, mesh=None):
    link = ET.SubElement(robot, "link", name=comp.id)

    # Inertial
    if mesh is None:
        mesh = _get_mesh(comp)
    mass, ixx, iyy, izz = _inertia_from_mesh(mesh)
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "mass", value=f"{mass:.4f}")
    ET.SubElement(inertial, "inertia",
        ixx=f"{ixx:.6f}", ixy="0", ixz="0",
        iyy=f"{iyy:.6f}", iyz="0", izz=f"{izz:.6f}")

    # Visual
    visual = ET.SubElement(link, "visual")
    ET.SubElement(visual, "origin", xyz="0 0 0", rpy="0 0 0")
    vgeo = ET.SubElement(visual, "geometry")
    ET.SubElement(vgeo, "mesh", filename=f"package://{stl_rel}")
    mat = ET.SubElement(visual, "material", name=f"mat_{comp.id}")
    color = _link_color(comp)
    ET.SubElement(mat, "color", rgba=f"{color[0]:.3f} {color[1]:.3f} {color[2]:.3f} 1.0")

    # Collision (same mesh — for accurate collision checking)
    collision = ET.SubElement(link, "collision")
    ET.SubElement(collision, "origin", xyz="0 0 0", rpy="0 0 0")
    cgeo = ET.SubElement(collision, "geometry")
    ET.SubElement(cgeo, "mesh", filename=f"package://{stl_rel}")


def _add_joint(robot: ET.Element, j):
    joint_el = ET.SubElement(robot, "joint", name=j.id, type=j.type)
    ET.SubElement(joint_el, "parent", link=j.parent_id)
    ET.SubElement(joint_el, "child",  link=j.child_id)

    xyz = j.origin_xyz
    rpy = j.origin_rpy
    ET.SubElement(joint_el, "origin",
        xyz=f"{xyz[0]:.5f} {xyz[1]:.5f} {xyz[2]:.5f}",
        rpy=f"{rpy[0]:.5f} {rpy[1]:.5f} {rpy[2]:.5f}")

    ax = j.axis
    ET.SubElement(joint_el, "axis",
        xyz=f"{ax[0]:.3f} {ax[1]:.3f} {ax[2]:.3f}")

    if j.type != "fixed":
        ET.SubElement(joint_el, "limit",
            lower=f"{j.limit_lower:.5f}",
            upper=f"{j.limit_upper:.5f}",
            effort="10.0",
            velocity="1.0")
        ET.SubElement(joint_el, "dynamics", damping="0.1", friction="0.05")


def _link_color(comp) -> tuple:
    """Return RGB colour from mesh vertex colours, or fallback."""
    mesh = getattr(comp, 'mesh', None)
    if mesh is not None and mesh.has_vertex_colors():
        vc = np.asarray(mesh.vertex_colors)
        if len(vc) > 0:
            return tuple(vc.mean(axis=0).tolist())
    # Fallback palette
    palette = [
        (0.31, 0.31, 0.35),
        (0.18, 0.43, 0.76),
        (0.16, 0.53, 0.82),
        (0.15, 0.63, 0.67),
        (0.22, 0.71, 0.47),
        (0.84, 0.47, 0.16),
    ]
    return palette[getattr(comp, 'depth', 0) % len(palette)]
