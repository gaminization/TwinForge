"""
articulated.py — rest scan + one scan per joint  ->  link meshes + automatically detected joints.

See core/jointscan.py for the capture protocol and the algorithm.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from core import jointscan
from core.joint_detector import Joint
from core.segmenter import Component

log = logging.getLogger("twinforge.articulated")


def load_mesh(src, work_dir: Path, size_m=None) -> o3d.geometry.TriangleMesh:
    """A .ply mesh, or a folder of photos (reconstructed with the SfM + dense stereo pipeline)."""
    src = Path(src)
    if src.is_dir():
        from core.photo_recon import reconstruct_from_photos
        r = reconstruct_from_photos(str(src), str(work_dir / src.name), size_m=size_m)
        return r.mesh
    m = o3d.io.read_triangle_mesh(str(src))
    if len(m.triangles) == 0:
        raise ValueError(f"{src} has no triangles - pass a mesh (.ply) or a photo folder")
    if size_m:
        ext = np.ptp(np.asarray(m.vertices), axis=0).max()
        m.scale(size_m / ext, center=(0, 0, 0))
    return m


def _floor(mesh):
    """Gravity-aligned input assumed (z up); put the lowest point on z = 0."""
    mesh.translate((0, 0, -np.asarray(mesh.vertices)[:, 2].min()))
    return mesh


def _label_mesh(mesh, ref_pts, labels, n_links):
    V = np.asarray(mesh.vertices); T = np.asarray(mesh.triangles)
    vlab = labels[cKDTree(ref_pts).query(V)[1]]
    votes = np.zeros((len(T), n_links))
    for k in range(3):
        votes[np.arange(len(T)), vlab[T[:, k]]] += 1
    return votes.argmax(1)


def build_from_scans(rest_mesh, scan_meshes: List, names=None):
    """Returns (components, joints)."""
    rest = _floor(rest_mesh)
    scans = [_floor(m) for m in scan_meshes]
    pts = lambda m: np.asarray(m.sample_points_uniformly(60000).points)
    res = jointscan.estimate(pts(rest), [pts(m) for m in scans])
    # re-use the exact point set the labels refer to
    n_links = len(res.link_ids)
    tri_lab = _label_mesh(rest, res.ref, res.labels, n_links)
    T = np.asarray(rest.triangles)
    comps = []
    for li, lid in enumerate(res.link_ids):
        sel = np.where(tri_lab == li)[0]
        if len(sel) < 30:
            log.warning("%s has almost no surface; skipped", lid)
            continue
        sm = o3d.geometry.TriangleMesh(rest.vertices, o3d.utility.Vector3iVector(T[sel]))
        sm.remove_unreferenced_vertices(); sm.compute_vertex_normals()
        V = np.asarray(sm.vertices)
        lo, hi = V.min(0), V.max(0)
        origin = lo.copy(); origin[:2] = (lo[:2] + hi[:2]) / 2
        comps.append(Component(id=lid, label="base" if li == 0 else f"link_{li}",
                               pcd=sm.sample_points_uniformly(1500), mesh=sm, origin=origin,
                               extent=hi - lo, movable=li > 0, depth=li))
    ids = {c.id for c in comps}
    joints = []
    for j in res.joints:
        if j.child not in ids or j.parent not in ids:
            log.warning("joint %s dropped (link missing)", j.name)
            continue
        conf = {"high": 0.9, "medium": 0.6, "low": 0.3}[j.confidence]
        joints.append(Joint(id=j.name, parent_id=j.parent, child_id=j.child, type=j.type,
                            axis=np.asarray(j.axis, float), origin_xyz=np.asarray(j.xyz, float),
                            origin_rpy=np.zeros(3), limit_lower=j.lower, limit_upper=j.upper,
                            confidence=conf, world_frame=True))
    return comps, joints, res
