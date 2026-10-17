"""
segmenter.py — Split point cloud (or mesh) into per-link components.

Demo path: uses the pre-built link meshes from demo_arm.py directly.
Real scan:  Poisson reconstruction → colour+height segmentation.
"""
import logging
import numpy as np
import open3d as o3d
from dataclasses import dataclass, field
from typing import List, Optional

log = logging.getLogger("twinforge.segmenter")


@dataclass
class Component:
    id:      str
    label:   str
    pcd:     o3d.geometry.PointCloud
    mesh:    Optional[o3d.geometry.TriangleMesh]
    origin:  np.ndarray          # bottom-centre in world coords
    extent:  np.ndarray          # bounding box [dx, dy, dz]
    movable: bool = True
    depth:   int  = 0            # in kinematic chain (0 = base)
    local_mesh: Optional[o3d.geometry.TriangleMesh] = None  # link-frame mesh (URDF export)


def segment(scan, n_links: int = 6, joints_path=None) -> List[Component]:
    if getattr(scan, '_demo_link_meta', None) is not None:
        return _from_demo_meta(scan._demo_link_meta)
    chain = None
    if joints_path:
        from core.joint_detector import chain_positions
        chain = chain_positions(joints_path)
    return _real_segment(scan.pcd, n_links, mesh=getattr(scan, 'mesh', None), chain=chain)


# ──────────────────────────────────────────────────────────────
# Demo fast-path: use pre-built accurate meshes
# ──────────────────────────────────────────────────────────────

def _from_demo_meta(link_metas) -> List[Component]:
    comps = []
    for i, lm in enumerate(link_metas):
        bb = lm.mesh.get_axis_aligned_bounding_box()
        extent = np.asarray(bb.max_bound) - np.asarray(bb.min_bound)
        origin = np.asarray(bb.min_bound)

        comps.append(Component(
            id=lm.id,
            label=lm.label,
            pcd=lm.pcd,
            mesh=lm.mesh,
            local_mesh=lm.local_mesh,
            origin=lm.origin,
            extent=extent,
            movable=(i > 0),    # base doesn't move
            depth=i,
        ))
    log.info(f"Demo segments: {[c.id for c in comps]}")
    return comps


# ──────────────────────────────────────────────────────────────
# Real scan: Poisson → colour/height segmentation
# ──────────────────────────────────────────────────────────────

def _seg_dist(P, A, B):
    """Distance from points P (N,3) to segment AB."""
    AB = B - A
    t = np.clip(((P - A) @ AB) / (AB @ AB + 1e-12), 0.0, 1.0)
    return np.linalg.norm(P - (A + t[:, None] * AB), axis=1)


def _split_by_chain(mesh, J):
    """Assign every triangle to the link whose skeleton segment is nearest.
    Skeleton = base footprint -> joint 1 -> ... -> joint n -> tip (links 0..n)."""
    V, F = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    foot = np.array([J[0, 0], J[0, 1], V[:, 2].min()])
    # tip = extremity of the mesh *beyond* the last joint, along the chain direction
    prev = J[-2] if len(J) > 1 else foot
    d = J[-1] - prev
    d = d / (np.linalg.norm(d) + 1e-12)
    tip = V[np.argmax((V - J[-1]) @ d)]
    P = np.vstack([foot, J, tip])
    C = V[F].mean(1)
    d = np.stack([_seg_dist(C, P[k], P[k + 1]) for k in range(len(P) - 1)], 1)
    lab = d.argmin(1)
    parts = []
    for k in range(len(P) - 1):
        m = o3d.geometry.TriangleMesh(mesh)
        m.remove_triangles_by_mask(lab != k)
        m.remove_unreferenced_vertices()
        parts.append(m)
    return parts


def _real_segment(pcd: o3d.geometry.PointCloud, n_links: int = 6,
                  mesh=None, chain=None) -> List[Component]:
    from core.mesh_reconstruction import reconstruct_mesh, segment_mesh_by_color

    if mesh is not None and len(mesh.triangles) > 0:
        log.info(f"Using the scan's own mesh ({len(mesh.triangles):,} triangles) - no re-meshing")
        full_mesh = mesh
    else:
        log.info("Real scan segmentation: Poisson reconstruction…")
        if not pcd.has_normals():
            pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
            pcd.orient_normals_consistent_tangent_plane(k=15)
        full_mesh = reconstruct_mesh(pcd, depth=9, voxel_size=0.003)

    if chain is not None:
        log.info(f"Splitting into {len(chain) + 1} links along the {len(chain)} user-confirmed joints")
        submeshes = _split_by_chain(full_mesh, chain)
    else:
        submeshes = segment_mesh_by_color(full_mesh, n_clusters=n_links)

    if not submeshes:
        log.warning("Segmentation returned 0 parts — using full mesh as one component")
        submeshes = [full_mesh]

    comps = []
    for i, sm in enumerate(submeshes):
        verts = np.asarray(sm.vertices)
        if len(verts) < 10:
            log.warning(f"link_{i} is (almost) empty after segmentation - skipped")
            continue
        bb_min = verts.min(axis=0)
        bb_max = verts.max(axis=0)
        extent = bb_max - bb_min
        origin = bb_min.copy()
        origin[0] = (bb_min[0] + bb_max[0]) / 2   # centre x
        origin[1] = (bb_min[1] + bb_max[1]) / 2   # centre y
        sm.compute_vertex_normals()
        sp = sm.sample_points_uniformly(1500)

        label = _guess_label(i, len(submeshes), extent)
        comps.append(Component(
            id=f"link_{i}",
            label=label,
            pcd=sp,
            mesh=sm,
            origin=origin,
            extent=extent,
            movable=(i > 0),
            depth=i,
        ))

    log.info(f"Segmented into {len(comps)} component(s)")
    return comps


def _guess_label(i: int, total: int, extent: np.ndarray) -> str:
    if i == 0:
        return "base"
    if i == total - 1:
        return "end_effector"
    h = extent[2]
    if h > 0.12:
        return "upper_arm"
    if h > 0.08:
        return "forearm"
    if h > 0.04:
        return "shoulder"
    return "wrist"
