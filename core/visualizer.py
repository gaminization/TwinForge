"""
visualizer.py — Render animation frames of the reconstructed robot.

Uses matplotlib with the actual STL meshes projected as 3D polygons,
producing frames that show the real robot shape moving through a
±30° sweep of each joint.
"""
import logging
import numpy as np
from pathlib import Path
from typing import List, Optional

log = logging.getLogger("twinforge.visualizer")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def save_frames(
    urdf_path,
    frames_dir: Path,
    components=None,
    n_frames: int = 36,
) -> List[Path]:
    """
    Render n_frames PNG images showing the robot rotating.
    Uses actual mesh geometry from components.
    """
    frames_dir = Path(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    if components is None:
        log.warning("No components supplied to visualizer")
        return []

    saved = []
    for i in range(n_frames):
        angle = (i / n_frames) * 2 * np.pi   # full rotation for flyaround

        fig = plt.figure(figsize=(8, 8), facecolor="#08080f")
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor("#08080f")
        ax.grid(False)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        # Draw each link mesh
        for comp in components:
            _draw_mesh(ax, comp, animate_idx=i, n_frames=n_frames)

        # Ground grid
        gx = np.linspace(-0.1, 0.1, 5)
        gy = np.linspace(-0.1, 0.1, 5)
        for x in gx:
            ax.plot([x, x], [gy[0], gy[-1]], [0, 0], color="#1e1e30", lw=0.5, alpha=0.5)
        for y in gy:
            ax.plot([gx[0], gx[-1]], [y, y], [0, 0], color="#1e1e30", lw=0.5, alpha=0.5)

        # Orbit camera around
        elev = 20 + 10 * np.sin(angle * 0.5)
        azim = np.degrees(angle)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlim(-0.12, 0.12)
        ax.set_ylim(-0.12, 0.12)
        ax.set_zlim(0, 0.45)

        out_path = frames_dir / f"frame_{i:04d}.png"
        fig.savefig(str(out_path), dpi=100, bbox_inches="tight",
                    facecolor=fig.get_facecolor(), pad_inches=0.05)
        plt.close(fig)
        saved.append(out_path)

    log.info(f"Saved {len(saved)} frames to {frames_dir}")
    return saved


def _draw_mesh(ax, comp, animate_idx: int, n_frames: int):
    """Draw one component's mesh as a Poly3DCollection."""
    import open3d as o3d

    mesh = getattr(comp, 'mesh', None)
    if mesh is None:
        return

    import numpy as np
    verts = np.asarray(mesh.vertices)
    tris  = np.asarray(mesh.triangles)
    if len(verts) == 0 or len(tris) == 0:
        return

    # Animate: rotate around joint axis for links > 0
    if getattr(comp, 'depth', 0) > 0:
        t = (animate_idx / n_frames) * 2 * np.pi
        amp = 0.35 if comp.depth == 1 else 0.25
        theta = np.sin(t + comp.depth * 0.7) * amp

        # Rotate around bottom of this link
        origin = comp.origin.copy()
        R = _rot_y(theta)   # pitch rotation
        verts = ((verts - origin) @ R.T) + origin

    polys = verts[tris]    # Mx3x3

    # Colour from mesh or fallback
    if mesh.has_vertex_colors():
        vc = np.asarray(mesh.vertex_colors)
        face_colors = vc[tris].mean(axis=1)   # Mx3
    else:
        palette = [
            [0.31, 0.31, 0.35],
            [0.18, 0.43, 0.76],
            [0.16, 0.53, 0.82],
            [0.15, 0.63, 0.67],
            [0.22, 0.71, 0.47],
            [0.84, 0.47, 0.16],
        ]
        c = palette[comp.depth % len(palette)]
        face_colors = np.tile(c, (len(tris), 1))

    # Subsample large meshes for speed
    max_tris = 500
    if len(polys) > max_tris:
        idx = np.random.choice(len(polys), max_tris, replace=False)
        polys = polys[idx]
        face_colors = face_colors[idx]

    coll = Poly3DCollection(polys, alpha=0.92, linewidth=0)
    coll.set_facecolor(face_colors)
    ax.add_collection3d(coll)


def _rot_y(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def show_interactive(comps, joints):
    """Launch Open3D interactive viewer (desktop only)."""
    import open3d as o3d
    geometries = []
    for comp in comps:
        if getattr(comp, 'mesh', None) is not None:
            geometries.append(comp.mesh)
        else:
            geometries.append(comp.pcd)

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    geometries.append(frame)
    o3d.visualization.draw_geometries(geometries, window_name="TwinForge 3D")
