"""
Segmenter — pure geometry, no ML model required

Algorithm
---------
1. Voxel-downsample + remove statistical outliers
2. DBSCAN clustering → candidate rigid links
3. For each cluster: compute centroid, bounding box, PCA axes, shape ratios
4. Sort by vertical position (base = lowest centroid)
5. Build parent-child pairs by nearest centroid
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class Component:
    id: str
    label: str
    type: str                            # base | arm_link | wheel | end_effector | other
    pcd:       object                    # o3d.geometry.PointCloud
    centroid:  np.ndarray
    bbox:      object                    # o3d.geometry.AxisAlignedBoundingBox
    pca_axes:  np.ndarray                # 3×3, columns = eigenvectors (smallest→largest λ)
    pca_vals:  np.ndarray                # 3  eigenvalues
    extent:    np.ndarray                # bounding-box side lengths [x,y,z]
    n_points:  int = 0
    movable:   bool = True
    parent_id: Optional[str] = None
    mesh:      Optional[object] = None   # filled later by urdf_builder


def segment(scan_data, voxel=0.005, eps=0.025, min_pts=80) -> list[Component]:
    """
    Returns list of Component, ordered base → links → end-effector.
    Falls back to demo pre-labelled clusters if scan_data has _demo_link_pcds.
    """
    import open3d as o3d

    # ── Demo shortcut: use pre-labelled synthetic sub-clouds ─────────────
    if hasattr(scan_data, "_demo_link_pcds") and scan_data._demo_link_pcds:
        return _from_demo(scan_data._demo_link_pcds)

    # ── Real pipeline ─────────────────────────────────────────────────────
    pcd = _clean(scan_data.pcd, voxel)
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_pts,
                                         print_progress=False))
    n_clusters = labels.max() + 1
    log.info(f"DBSCAN → {n_clusters} raw clusters  (eps={eps}, min_pts={min_pts})")

    if n_clusters == 0:
        # Relax parameters and retry
        labels = np.array(pcd.cluster_dbscan(eps=eps*2, min_points=min_pts//2,
                                              print_progress=False))
        n_clusters = labels.max() + 1
        log.warning(f"Retry with relaxed params → {n_clusters} clusters")

    # Build per-cluster data
    raw = []
    for lbl in range(n_clusters):
        idx = np.where(labels == lbl)[0]
        if len(idx) < 40:
            continue
        sub = pcd.select_by_index(idx.tolist())
        raw.append(_analyse(sub))
    raw.sort(key=lambda c: c["centroid"][2])     # lowest Z first → base

    if not raw:
        raise RuntimeError("No components found — check scan quality or try --demo")

    # Label components by position in sorted list
    components = []
    n = len(raw)
    for i, r in enumerate(raw):
        if i == 0:
            label, ctype = "base", "base"
        elif i == n - 1:
            label, ctype = "end_effector", "end_effector"
        else:
            label, ctype = f"link_{i}", "arm_link"

        comp = Component(
            id=f"link_{i}", label=label, type=ctype,
            pcd=r["pcd"], centroid=r["centroid"],
            bbox=r["bbox"], pca_axes=r["pca_axes"],
            pca_vals=r["pca_vals"], extent=r["extent"],
            n_points=r["n_points"],
            movable=(i > 0),
            parent_id=f"link_{i-1}" if i > 0 else None,
        )
        components.append(comp)

    log.info(f"Segmented into {len(components)} components: "
             f"{[c.label for c in components]}")
    return components


# ── Demo shortcut ─────────────────────────────────────────────────────────────

def _from_demo(link_pcds: dict) -> list[Component]:
    """Build components directly from the known synthetic sub-clouds."""
    import open3d as o3d

    order = ["base", "joint_disc_1", "link_1",
             "joint_disc_2", "link_2", "link_3", "end_effector"]
    # Keep only keys that exist
    order = [k for k in order if k in link_pcds]

    # Merge joint discs into adjacent links for cleaner segmentation
    merged = {}
    skip   = set()
    for i, key in enumerate(order):
        if key in skip:
            continue
        if "joint_disc" in key and i + 1 < len(order):
            # Merge disc into next link
            next_key = order[i + 1]
            merged_pcd = link_pcds[key] + link_pcds[next_key]
            merged[next_key] = merged_pcd
            skip.add(next_key)
        elif key not in merged:
            merged[key] = link_pcds[key]

    keys_ordered = [k for k in order if k in merged]
    components   = []
    for i, key in enumerate(keys_ordered):
        sub  = merged[key]
        r    = _analyse(sub)
        ctype = ("base" if i == 0
                 else "end_effector" if i == len(keys_ordered)-1
                 else "arm_link")
        components.append(Component(
            id=f"link_{i}", label=key, type=ctype,
            pcd=sub, centroid=r["centroid"],
            bbox=r["bbox"], pca_axes=r["pca_axes"],
            pca_vals=r["pca_vals"], extent=r["extent"],
            n_points=r["n_points"],
            movable=(i > 0),
            parent_id=f"link_{i-1}" if i > 0 else None,
        ))
    log.info(f"Demo segments: {[c.label for c in components]}")
    return components


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clean(pcd, voxel: float):
    import open3d as o3d
    pcd = pcd.voxel_down_sample(voxel)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
    return pcd


def _analyse(sub) -> dict:
    """Per-cluster analysis: centroid, bbox, PCA."""
    import open3d as o3d
    pts      = np.asarray(sub.points)
    centroid = pts.mean(axis=0)
    cov      = np.cov(pts.T)
    vals, vecs = np.linalg.eigh(cov)   # ascending eigenvalues
    idx      = np.argsort(vals)
    vals     = vals[idx]; vecs = vecs[:, idx]
    bbox     = sub.get_axis_aligned_bounding_box()
    extent   = np.asarray(bbox.get_extent())
    return dict(pcd=sub, centroid=centroid, bbox=bbox,
                pca_axes=vecs, pca_vals=vals,
                extent=extent, n_points=len(pts))
