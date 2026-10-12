"""
Robot Visualizer — FK animation with two backends

  run_interactive(urdf_path)   → Open3D window (requires display + OpenGL)
  save_frames(urdf_path, out)  → matplotlib 3D renders (pure-CPU, always works)

Forward kinematics (FK):
  q_i(t) = mid_i + amp_i * sin(2π t/T + φ_i)
  T_world_link = chain of joint offset * joint motion transforms
  mesh.vertices ← T @ base_vertices    (per frame)
"""

from __future__ import annotations
import logging, time, math
from pathlib import Path
from xml.etree import ElementTree as ET
import numpy as np

log = logging.getLogger(__name__)


# ── FK math ───────────────────────────────────────────────────────────────────

def _rot(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues — axis must be unit vector."""
    c, s = math.cos(angle), math.sin(angle)
    x, y, z = axis
    t = 1.0 - c
    return np.array([
        [t*x*x+c,    t*x*y-s*z, t*x*z+s*y],
        [t*x*y+s*z,  t*y*y+c,   t*y*z-s*x],
        [t*x*z-s*y,  t*y*z+s*x, t*z*z+c  ],
    ])


def _rpy_mat(rpy) -> np.ndarray:
    r, p, y = rpy
    return (_rot(np.array([0.,0.,1.]), y)
            @ _rot(np.array([0.,1.,0.]), p)
            @ _rot(np.array([1.,0.,0.]), r))


def _joint_T(jtype: str, axis: np.ndarray, q: float) -> np.ndarray:
    T = np.eye(4)
    a = axis / (np.linalg.norm(axis) + 1e-12)
    if jtype in ("revolute", "continuous"):
        T[:3, :3] = _rot(a, q)
    elif jtype == "prismatic":
        T[:3, 3]  = a * q
    return T


def forward_kinematics(links: list, joints_map: dict,
                        joint_values: dict) -> dict[str, np.ndarray]:
    """
    Returns {link_id: 4×4 world transform}.
    joints_map: {child_link_id: joint_dict}
    """
    transforms: dict[str, np.ndarray] = {}

    def _fk(link_id: str, T_parent: np.ndarray):
        transforms[link_id] = T_parent
        for child_id, j in joints_map.items():
            if j["parent"] == link_id:
                T_off      = np.eye(4)
                T_off[:3, :3] = _rpy_mat(j["rpy"])
                T_off[:3,  3] = j["xyz"]
                q          = float(joint_values.get(j["id"], 0.0))
                T_mot      = _joint_T(j["type"], j["axis"], q)
                _fk(child_id, T_parent @ T_off @ T_mot)

    all_children = set(joints_map.keys())
    roots        = [l["id"] for l in links if l["id"] not in all_children]
    root_id      = roots[0] if roots else links[0]["id"]
    _fk(root_id, np.eye(4))
    return transforms


# ── URDF parser ───────────────────────────────────────────────────────────────

def parse_urdf(urdf_path: Path):
    """Returns (links, joints_map, root_id)."""
    root = ET.parse(str(urdf_path)).getroot()
    links = [{"id": el.get("name")} for el in root.findall("link")]

    joints_map: dict[str, dict] = {}
    for j in root.findall("joint"):
        child_id = j.find("child").get("link")
        orig     = j.find("origin")
        xyz = ([float(v) for v in orig.get("xyz", "0 0 0").split()]
               if orig is not None else [0., 0., 0.])
        rpy = ([float(v) for v in orig.get("rpy", "0 0 0").split()]
               if orig is not None else [0., 0., 0.])
        ax_el = j.find("axis")
        axis  = (np.array([float(v) for v in ax_el.get("xyz", "0 0 1").split()])
                 if ax_el is not None else np.array([0., 0., 1.]))
        lim   = j.find("limit")
        lo    = float(lim.get("lower", "-1.5708")) if lim is not None else -1.5708
        hi    = float(lim.get("upper",  "1.5708")) if lim is not None else  1.5708
        joints_map[child_id] = {
            "id":     j.get("name"),
            "type":   j.get("type", "fixed"),
            "parent": j.find("parent").get("link"),
            "xyz":    np.array(xyz, dtype=float),
            "rpy":    np.array(rpy, dtype=float),
            "axis":   axis.astype(float),
            "lo": lo, "hi": hi,
        }

    all_children = set(joints_map.keys())
    all_ids      = {l["id"] for l in links}
    roots        = list(all_ids - all_children)
    root_id      = roots[0] if roots else links[0]["id"]
    return links, joints_map, root_id


# ── Geometry loading ──────────────────────────────────────────────────────────

def _load_geoms(links, joints_map, urdf_path: Path, components):
    """Returns {link_id: (o3d.TriangleMesh, base_verts_copy)}"""
    import open3d as o3d

    mesh_dir = urdf_path.parent / "meshes"
    comp_map = {c.id: c for c in components} if components else {}

    LINK_COLORS = {
        "base":         [0.35, 0.35, 0.35],
        "arm_link":     [0.20, 0.55, 0.80],
        "end_effector": [0.90, 0.50, 0.10],
        "wheel":        [0.10, 0.10, 0.10],
    }

    geoms = {}
    for link in links:
        lid  = link["id"]
        comp = comp_map.get(lid)
        stl  = mesh_dir / f"{lid}.stl"

        if stl.exists():
            mesh = o3d.io.read_triangle_mesh(str(stl))
        elif comp and comp.mesh is not None:
            mesh = comp.mesh
        else:
            # Box from component extent
            sx, sy, sz = ([max(float(v), 0.01) for v in comp.extent]
                           if comp else [0.05, 0.05, 0.05])
            mesh = o3d.geometry.TriangleMesh.create_box(sx, sy, sz)
            mesh.translate(-mesh.get_center())

        mesh.compute_vertex_normals()
        ctype = comp.type if comp else "other"
        mesh.paint_uniform_color(LINK_COLORS.get(ctype, [0.65, 0.65, 0.65]))

        # Place mesh at its component centroid so FK offsets work
        if comp is not None:
            mesh.translate(comp.centroid - mesh.get_center())

        geoms[lid] = (mesh, np.asarray(mesh.vertices).copy())

    return geoms


# ── Animation helpers ─────────────────────────────────────────────────────────

def _joint_values_at(t: float, movable: dict, period: float) -> dict:
    jvals = {}
    for i, (_, j) in enumerate(movable.items()):
        lo, hi  = j["lo"], j["hi"]
        amp     = (hi - lo) / 2.0
        mid     = (hi + lo) / 2.0
        phase   = i * (2 * math.pi / max(len(movable), 1))
        jvals[j["id"]] = mid + amp * math.sin(2 * math.pi * t / period + phase)
    return jvals


# ── Interactive viewer (Open3D) ───────────────────────────────────────────────

def run_interactive(urdf_path: Path, components=None,
                    n_frames: int = 300, period: float = 4.0):
    """
    Open3D interactive window.  Requires a display and OpenGL.
    Press Q or close the window to exit.
    """
    import open3d as o3d

    urdf_path = Path(urdf_path)
    links, jmap, root_id = parse_urdf(urdf_path)
    geoms   = _load_geoms(links, jmap, urdf_path, components)
    movable = {cid: j for cid, j in jmap.items()
               if j["type"] in ("revolute", "prismatic", "continuous")}

    vis = o3d.visualization.Visualizer()
    vis.create_window("TwinForge 3D — Robot Viewer", width=1200, height=800)
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(0.1))
    for mesh, _ in geoms.values():
        vis.add_geometry(mesh)

    ro = vis.get_render_option()
    ro.light_on = True
    ro.mesh_show_back_face = False

    print(f"\n{'='*60}")
    print(f"  TwinForge 3D Viewer — {urdf_path.stem}")
    print(f"  {len(movable)} animated joint(s): "
          f"{[j['id'] for j in movable.values()]}")
    print(f"  Close window to exit")
    print(f"{'='*60}\n")

    dt = period / n_frames
    t  = 0.0
    while True:
        jvals  = _joint_values_at(t, movable, period)
        T_world = forward_kinematics(links, jmap, jvals)

        for lid, (mesh, base_verts) in geoms.items():
            T = T_world.get(lid, np.eye(4))
            R, tv = T[:3, :3], T[:3, 3]
            mesh.vertices = o3d.utility.Vector3dVector(
                (R @ base_verts.T).T + tv)
            mesh.compute_vertex_normals()
            vis.update_geometry(mesh)

        if not vis.poll_events():
            break
        vis.update_renderer()
        time.sleep(dt)
        t += dt

    vis.destroy_window()


# ── Headless renderer (matplotlib — no GPU required) ─────────────────────────

def save_frames(urdf_path: Path, output_dir: Path, components=None,
                n_frames: int = 60, period: float = 4.0):
    """
    Pure-CPU render using matplotlib 3D.  Works on any machine.
    Draws the kinematic skeleton: coloured link meshes as wire-spheres
    + lines between joint origins.
    """
    import matplotlib
    matplotlib.use("Agg")          # no display needed
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    urdf_path  = Path(urdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    links, jmap, root_id = parse_urdf(urdf_path)
    movable = {cid: j for cid, j in jmap.items()
               if j["type"] in ("revolute", "prismatic", "continuous")}
    comp_map = {c.id: c for c in components} if components else {}

    LINK_COLORS = {
        "base":         "#595959",
        "arm_link":     "#3389cc",
        "end_effector": "#e5801a",
        "wheel":        "#1a1a1a",
        "other":        "#aaaaaa",
    }

    # Pre-compute link centroid offsets (local frame)
    local_centroids = {}
    for link in links:
        lid  = link["id"]
        comp = comp_map.get(lid)
        local_centroids[lid] = comp.centroid if comp is not None else np.zeros(3)

    saved = []
    for fi in range(n_frames):
        t      = fi * period / n_frames
        jvals  = _joint_values_at(t, movable, period)
        T_world = forward_kinematics(links, jmap, jvals)

        fig = plt.figure(figsize=(9, 7), facecolor="#1a1a2e")
        ax  = fig.add_subplot(111, projection="3d",
                               facecolor="#1a1a2e")

        # Collect joint origins in world space
        joint_origins = {}
        for child_id, j in jmap.items():
            T_parent = T_world.get(j["parent"], np.eye(4))
            T_off    = np.eye(4)
            T_off[:3, :3] = _rpy_mat(j["rpy"])
            T_off[:3,  3] = j["xyz"]
            p_world  = (T_parent @ T_off)[:3, 3]
            joint_origins[child_id] = p_world

        # Draw links as thick lines + spheres at joint origins
        for link in links:
            lid  = link["id"]
            T    = T_world.get(lid, np.eye(4))
            loc  = local_centroids[lid]
            ctr  = (T[:3, :3] @ loc) + T[:3, 3]   # world centroid
            comp = comp_map.get(lid)
            ctype = comp.type if comp else "other"
            col   = LINK_COLORS.get(ctype, "#aaaaaa")

            # Draw link body (line from parent joint to child joint)
            if lid in jmap:
                j_in  = jmap[lid]
                p_in  = joint_origins.get(lid, ctr)
            else:
                p_in  = ctr

            # Children joints of this link
            child_joints = [j for cid, j in jmap.items() if j["parent"] == lid]
            if child_joints:
                for cj in child_joints:
                    p_out = joint_origins.get(
                        [cid for cid, j in jmap.items() if j is cj][0], ctr)
                    ax.plot([p_in[0], p_out[0]],
                            [p_in[1], p_out[1]],
                            [p_in[2], p_out[2]],
                            color=col, linewidth=8, solid_capstyle="round",
                            alpha=0.9)
            else:
                # End effector — draw as sphere only
                ax.scatter(*ctr, s=400, c=col, depthshade=True, alpha=0.9,
                           edgecolors="white", linewidths=0.5)

            # Joint sphere
            if lid in joint_origins:
                jp = joint_origins[lid]
                ax.scatter(*jp, s=200, c="#ff66aa",
                           depthshade=True, alpha=1.0,
                           edgecolors="white", linewidths=0.8)

        # Joint angle readout
        angle_txt = "\n".join(
            f"{j['id']}: {math.degrees(jvals.get(j['id'], 0)):+.1f}°"
            for j in movable.values()
        )
        ax.text2D(0.02, 0.98, angle_txt,
                  transform=ax.transAxes, fontsize=7,
                  color="white", va="top", fontfamily="monospace",
                  bbox=dict(facecolor="#00000088", pad=3, edgecolor="none"))

        # Axis styling
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.fill = False
            pane.set_edgecolor("#333355")
        ax.tick_params(colors="white", labelsize=6)
        ax.set_xlabel("X", color="#6688aa", fontsize=8)
        ax.set_ylabel("Y", color="#6688aa", fontsize=8)
        ax.set_zlabel("Z", color="#6688aa", fontsize=8)
        ax.set_xlim(-0.6, 0.6); ax.set_ylim(-0.6, 0.6); ax.set_zlim(-0.1, 0.8)
        ax.set_title(f"TwinForge 3D — {urdf_path.stem}  (t={t:.2f}s)",
                     color="white", fontsize=10)

        # Orbit camera slowly
        ax.view_init(elev=25, azim=45 + fi * 360 / n_frames)

        outpath = output_dir / f"frame_{fi:04d}.png"
        fig.savefig(str(outpath), dpi=110, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        saved.append(outpath)
        if fi % 10 == 0:
            log.info(f"  Frame {fi+1:3d}/{n_frames}")

    log.info(f"Saved {len(saved)} frames → {output_dir}")

    # Stitch to MP4 with cv2 (optional)
    try:
        import cv2
        sample = cv2.imread(str(saved[0]))
        h, w   = sample.shape[:2]
        mp4    = output_dir / "robot_animation.mp4"
        writer = cv2.VideoWriter(str(mp4),
                                  cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
        for p in saved:
            writer.write(cv2.imread(str(p)))
        writer.release()
        log.info(f"Animation video → {mp4}")
    except Exception as e:
        log.debug(f"Video stitching skipped: {e}")

    return saved
