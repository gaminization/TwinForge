"""
mesh_reconstruction.py — Dense mesh reconstruction from a point cloud.

Takes the raw SfM point cloud from IncrementalReconstructor and produces
clean triangle meshes per segment, ready for URDF export.

Pipeline:
  1. Outlier removal + voxel downsample
  2. Normal estimation
  3. Poisson surface reconstruction  (watertight mesh)
  4. Trim low-density regions (crop to point cloud hull)
  5. Per-segment mesh extraction
"""
import numpy as np
import open3d as o3d
import logging
from typing import List, Optional, Tuple

log = logging.getLogger("twinforge.mesh")


def reconstruct_mesh(
    pcd: o3d.geometry.PointCloud,
    depth: int = 9,
    voxel_size: float = 0.003,
) -> o3d.geometry.TriangleMesh:
    """
    Full Poisson reconstruction from a coloured point cloud.

    Args:
        pcd:        Input point cloud (with normals, ideally)
        depth:      Poisson octree depth (8=fast, 9=good, 10=high-res)
        voxel_size: Pre-processing voxel size in metres

    Returns:
        Watertight triangle mesh with vertex colours.
    """
    log.info(f"Mesh reconstruction: {len(pcd.points):,} input points")

    # ── 1. Clean & downsample ─────────────────────────────────
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd = pcd.voxel_down_sample(voxel_size)
    log.info(f"  After cleaning: {len(pcd.points):,} points")

    # ── 2. Normals ────────────────────────────────────────────
    if not pcd.has_normals():
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 4, max_nn=30)
        )
    pcd.orient_normals_consistent_tangent_plane(k=15)

    # ── 3. Poisson reconstruction ─────────────────────────────
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, linear_fit=False
    )
    log.info(f"  Poisson: {len(mesh.triangles):,} triangles")

    # ── 4. Trim low-density vertices (remove floaters/noise) ──
    dens = np.asarray(densities)
    threshold = np.percentile(dens, 15)          # keep top 85%
    verts_to_remove = dens < threshold
    mesh.remove_vertices_by_mask(verts_to_remove)
    log.info(f"  After density trim: {len(mesh.triangles):,} triangles")

    # ── 5. Colour from nearest point cloud point ──────────────
    if pcd.has_colors():
        mesh = _transfer_colors(mesh, pcd)

    mesh.compute_vertex_normals()
    return mesh


def _transfer_colors(
    mesh: o3d.geometry.TriangleMesh,
    pcd: o3d.geometry.PointCloud,
) -> o3d.geometry.TriangleMesh:
    """Assign each mesh vertex the colour of its nearest point cloud point."""
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    verts  = np.asarray(mesh.vertices)
    colors = np.asarray(pcd.colors)
    vc = np.zeros((len(verts), 3))
    for i, v in enumerate(verts):
        _, idx, _ = kdtree.search_knn_vector_3d(v, 1)
        vc[i] = colors[idx[0]]
    mesh.vertex_colors = o3d.utility.Vector3dVector(vc)
    return mesh


def segment_mesh_by_color(
    mesh: o3d.geometry.TriangleMesh,
    n_clusters: int = 6,
) -> List[o3d.geometry.TriangleMesh]:
    """
    Split a coloured mesh into per-link submeshes using colour-based
    k-means clustering of triangles.  Falls back to vertical slicing
    if colour variance is too low.
    """
    if not mesh.has_vertex_colors():
        return _slice_by_height(mesh, n_clusters)

    from sklearn.cluster import KMeans

    verts  = np.asarray(mesh.vertices)
    colors = np.asarray(mesh.vertex_colors)
    tris   = np.asarray(mesh.triangles)

    if len(tris) == 0:
        return [mesh]

    # Compute per-triangle centroid and mean colour
    tri_verts = verts[tris]                       # Tx3x3
    centroids = tri_verts.mean(axis=1)            # Tx3
    tri_colors = colors[tris].mean(axis=1)        # Tx3

    # Feature: [centroid_z, r, g, b] — z weighted heavily to keep links separate
    z_scale = 3.0
    features = np.hstack([centroids[:, 2:3] * z_scale, tri_colors])

    color_var = np.var(tri_colors, axis=0).sum()
    if color_var < 0.005:
        log.info("Low colour variance — using height-based segmentation")
        return _slice_by_height(mesh, n_clusters)

    km = KMeans(n_clusters=min(n_clusters, len(tris)), n_init=10, random_state=42)
    labels = km.fit_predict(features)

    submeshes = []
    for k in range(km.n_clusters):
        tri_idx = np.where(labels == k)[0]
        if len(tri_idx) < 10:
            continue
        sub = mesh.select_by_index(list(set(tris[tri_idx].flatten())))
        sub.triangles = o3d.utility.Vector3iVector(
            _reindex_triangles(tris[tri_idx], sub)
        )
        sub.compute_vertex_normals()
        submeshes.append(sub)

    # Sort bottom → top
    submeshes.sort(key=lambda m: np.asarray(m.vertices)[:, 2].mean())
    log.info(f"Segmented into {len(submeshes)} submeshes by colour+height")
    return submeshes


def _slice_by_height(
    mesh: o3d.geometry.TriangleMesh,
    n: int,
) -> List[o3d.geometry.TriangleMesh]:
    """Fallback: divide mesh into n equal-height bands."""
    verts = np.asarray(mesh.vertices)
    tris  = np.asarray(mesh.triangles)
    zmin, zmax = verts[:, 2].min(), verts[:, 2].max()
    bands = np.linspace(zmin, zmax, n + 1)

    submeshes = []
    for i in range(n):
        z_lo, z_hi = bands[i], bands[i + 1]
        tri_verts_z = verts[tris][:, :, 2]   # Tx3
        mask = (tri_verts_z.min(axis=1) >= z_lo) & (tri_verts_z.max(axis=1) < z_hi)
        tri_idx = np.where(mask)[0]
        if len(tri_idx) < 5:
            continue
        sub = mesh.select_by_index(list(set(tris[tri_idx].flatten())))
        sub.compute_vertex_normals()
        submeshes.append(sub)

    log.info(f"Sliced into {len(submeshes)} height bands")
    return submeshes


def _reindex_triangles(
    tri_subset: np.ndarray,
    submesh: o3d.geometry.TriangleMesh,
) -> np.ndarray:
    """Re-index a triangle array to match submesh vertex indices."""
    old_ids = sorted(set(tri_subset.flatten()))
    mapping = {old: new for new, old in enumerate(old_ids)}
    return np.vectorize(mapping.get)(tri_subset)
