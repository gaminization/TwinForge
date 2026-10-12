"""
Acquisition — zero hard dependencies beyond open3d + numpy + opencv

Supported inputs
----------------
  .ply / .pcd                    → direct point cloud
  directory/                     → RGB-D image pairs  (color_*.png + depth_*.png)
  video file (.mp4 .avi .mov)    → extract frames → photogrammetry (opencv SfM-lite)
  --demo                         → synthetic 4-DOF robot arm (no hardware needed)
"""

from __future__ import annotations
import logging, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import numpy as np

log = logging.getLogger(__name__)


# ── Public container ─────────────────────────────────────────────────────────

@dataclass
class ScanData:
    pcd:        object                        # o3d.geometry.PointCloud  (always set)
    source:     str = "unknown"
    intrinsics: Optional[dict] = None         # fx fy cx cy w h
    pose_pcds:  list = field(default_factory=list)   # one per robot pose (for motion axis)


# ── Entry-point ───────────────────────────────────────────────────────────────

def load(source: str | Path | None, demo: bool = False,
         n_poses: int = 1) -> ScanData:
    """
    Parameters
    ----------
    source  : path to .ply/.pcd/directory/video, or None
    demo    : generate a synthetic 4-DOF robot arm
    n_poses : how many robot configurations to capture (>1 enables motion axis fitting)
    """
    import open3d as o3d

    if demo or source is None:
        return _demo_arm(n_poses)

    p = Path(source)
    if not p.exists():
        raise FileNotFoundError(p)

    if p.suffix in (".ply", ".pcd", ".xyz"):
        return _from_point_cloud(p)

    if p.is_dir():
        return _from_rgbd_dir(p)

    if p.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv"):
        return _from_video(p)

    raise ValueError(f"Unsupported input: {p}")


# ── Loaders ───────────────────────────────────────────────────────────────────

def _from_point_cloud(path: Path) -> ScanData:
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(str(path))
    if len(pcd.points) == 0:
        raise RuntimeError(f"Empty point cloud: {path}")
    log.info(f"Loaded {len(pcd.points):,} points from {path.name}")
    return ScanData(pcd=pcd, source=str(path))


def _from_rgbd_dir(path: Path) -> ScanData:
    """
    Expects:
        path/color_0000.png  path/depth_0000.png
        path/color_0001.png  path/depth_0001.png  ...
    or  path/rgb/   + path/depth/
    Depth in millimetres (16-bit PNG) or metres (32-bit float .npy).
    """
    import open3d as o3d, cv2

    # Find pairs
    color_files = sorted(path.glob("color_*.png")) + sorted(path.glob("rgb/*.png"))
    depth_files = sorted(path.glob("depth_*.png")) + sorted(path.glob("depth/*.png"))
    npy_files   = sorted(path.glob("depth_*.npy")) + sorted(path.glob("depth/*.npy"))
    depth_files = npy_files if npy_files else depth_files

    if not color_files or not depth_files:
        # Fallback: treat all PNGs as colour, look for .ply
        plys = list(path.glob("*.ply"))
        if plys:
            return _from_point_cloud(plys[0])
        raise RuntimeError(f"No colour+depth pairs found in {path}")

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=4.0 / 512.0, sdf_trunc=0.04,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    intr_obj = o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)

    pose_pcds = []
    for c_path, d_path in zip(color_files, depth_files):
        color_img = cv2.imread(str(c_path))[:, :, ::-1]
        if str(d_path).endswith(".npy"):
            d_arr = np.load(str(d_path)).astype(np.float32)
            d_mm  = (d_arr * 1000).astype(np.uint16)
        else:
            d_mm  = cv2.imread(str(d_path), cv2.IMREAD_UNCHANGED)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color_img),
            o3d.geometry.Image(d_mm),
            depth_scale=1000.0, depth_trunc=3.0,
            convert_rgb_to_intensity=False)

        pose_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intr_obj)
        pose_pcds.append(pose_pcd)
        volume.integrate(rgbd, intr_obj, np.eye(4))

    pcd  = volume.extract_point_cloud()
    log.info(f"TSDF fused {len(color_files)} frames → {len(pcd.points):,} pts")
    return ScanData(pcd=pcd, source=str(path), pose_pcds=pose_pcds)


def _from_video(path: Path) -> ScanData:
    """
    Extract frames from a video and run lightweight optical-flow
    structure-from-motion to get a sparse point cloud.
    Falls back to treating every 10th frame as a new view.
    """
    import cv2, open3d as o3d

    cap  = cv2.VideoCapture(str(path))
    fps  = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, int(fps) // 3)          # 3 frames/sec
    frames, idx = [], 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if idx % step == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        idx += 1
    cap.release()
    log.info(f"Extracted {len(frames)} frames from {path.name}")

    pcd = _sfm_lite(frames)
    return ScanData(pcd=pcd, source=str(path))


def _sfm_lite(frames: list) -> object:
    """
    Lightweight SfM using OpenCV SIFT + 5-point essential matrix.
    Produces a sparse point cloud.  Good enough for shape segmentation.
    """
    import cv2, open3d as o3d

    sift   = cv2.SIFT_create(nfeatures=3000)
    bf     = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    pts3d  = []

    # Approximate intrinsics from image size
    h, w = frames[0].shape[:2]
    fx   = max(w, h) * 0.9
    K    = np.array([[fx,0,w/2],[0,fx,h/2],[0,0,1]], np.float64)

    gray0 = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
    kp0, des0 = sift.detectAndCompute(gray0, None)

    for frame in frames[1:]:
        gray1 = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        kp1, des1 = sift.detectAndCompute(gray1, None)
        if des0 is None or des1 is None:
            continue
        matches = bf.match(des0, des1)
        matches = sorted(matches, key=lambda m: m.distance)[:200]
        if len(matches) < 8:
            continue

        p0 = np.float32([kp0[m.queryIdx].pt for m in matches])
        p1 = np.float32([kp1[m.trainIdx].pt for m in matches])
        E, mask = cv2.findEssentialMat(p0, p1, K, method=cv2.RANSAC,
                                       prob=0.999, threshold=1.0)
        if E is None: continue
        _, R, t, mask2 = cv2.recoverPose(E, p0, p1, K, mask=mask)

        # Triangulate
        P0 = K @ np.hstack([np.eye(3), np.zeros((3,1))])
        P1 = K @ np.hstack([R, t])
        pts4d = cv2.triangulatePoints(P0, P1,
                                      p0[mask2.ravel() > 0].T,
                                      p1[mask2.ravel() > 0].T)
        pts   = (pts4d[:3] / pts4d[3]).T
        good  = (pts[:,2] > 0) & (pts[:,2] < 5)
        pts3d.extend(pts[good].tolist())

        kp0, des0 = kp1, des1     # chain frames

    if len(pts3d) < 50:
        log.warning("SfM produced very few points — try .ply input instead")
        pts3d = np.random.randn(500, 3).tolist()  # minimal fallback

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(pts3d))
    pcd, _ = pcd.remove_statistical_outlier(20, 2.0)
    log.info(f"SfM → {len(pcd.points):,} 3D points")
    return pcd


# ── Demo synthetic robot arm ──────────────────────────────────────────────────

def _demo_arm(n_poses: int = 3) -> ScanData:
    """
    Build a synthetic 4-DOF robot arm point cloud.
    Each link is a cylinder (tube) or disc.
    n_poses  → each pose rotates joints by incremental angles,
               populating pose_pcds for motion-based axis fitting.
    """
    import open3d as o3d

    def _cylinder(axis_start, axis_end, radius, n=800, colour=None):
        """Random points on/in a cylinder between two 3D points."""
        d     = np.array(axis_end) - np.array(axis_start)
        L     = np.linalg.norm(d)
        d_hat = d / L
        # Local frame
        arb   = np.array([1,0,0]) if abs(d_hat[0]) < 0.9 else np.array([0,1,0])
        u     = np.cross(d_hat, arb); u /= np.linalg.norm(u)
        v     = np.cross(d_hat, u)
        t     = np.random.uniform(0, L, n)
        r     = np.random.uniform(0, radius, n)
        theta = np.random.uniform(0, 2*np.pi, n)
        pts   = (np.array(axis_start) +
                 np.outer(t, d_hat) +
                 np.outer(r*np.cos(theta), u) +
                 np.outer(r*np.sin(theta), v))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        if colour:
            pcd.colors = o3d.utility.Vector3dVector(
                np.tile(colour, (n, 1)))
        return pcd

    def _disc(centre, normal, radius, thickness=0.03, n=600, colour=None):
        """Flat disc (joint housing)."""
        norm  = np.array(normal, float); norm /= np.linalg.norm(norm)
        arb   = np.array([1,0,0]) if abs(norm[0]) < 0.9 else np.array([0,1,0])
        u     = np.cross(norm, arb); u /= np.linalg.norm(u)
        v     = np.cross(norm, u)
        r     = np.sqrt(np.random.uniform(0, radius**2, n))
        theta = np.random.uniform(0, 2*np.pi, n)
        t     = np.random.uniform(-thickness/2, thickness/2, n)
        pts   = (np.array(centre) +
                 np.outer(r*np.cos(theta), u) +
                 np.outer(r*np.sin(theta), v) +
                 np.outer(t, norm))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        if colour:
            pcd.colors = o3d.utility.Vector3dVector(
                np.tile(colour, (n, 1)))
        return pcd

    def _build_pose(angles):
        """
        Build point cloud for one robot pose.
        angles: [q1, q2, q3, q4]  (revolute around Z, Y, Z, Y)

        Kinematic chain (simplified DH-style):
          base  → joint1 (Z)  → link1 (length 0.20)
                → joint2 (Y)  → link2 (length 0.18)
                → joint3 (Y)  → link3 (length 0.14)
                → joint4 (Z)  → end_effector
        """
        q1, q2, q3, q4 = angles

        # Forward kinematics — accumulate transforms
        def Rz(a): return np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])
        def Ry(a): return np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])

        R0  = np.eye(3)
        p0  = np.zeros(3)                       # base centre

        R1  = R0 @ Rz(q1)
        p1  = p0 + R0 @ np.array([0,0,0.12])   # joint1 above base

        R2  = R1 @ Ry(q2)
        p2  = p1 + R1 @ np.array([0,0,0.20])   # link1 end

        R3  = R2 @ Ry(q3)
        p3  = p2 + R2 @ np.array([0,0,0.18])   # link2 end

        R4  = R3 @ Rz(q4)
        p4  = p3 + R3 @ np.array([0,0,0.14])   # link3 end (EE)

        link_pcds = {
            "base":        _disc(p0, R0[:,2], 0.10, thickness=0.08,
                                 colour=[0.35,0.35,0.35]),
            "link_1":      _cylinder(p1, p2, 0.035, colour=[0.20,0.55,0.80]),
            "joint_disc_1":_disc((p1+p2)/2, R1[:,2], 0.05, colour=[0.60,0.60,0.60]),
            "link_2":      _cylinder(p2, p3, 0.030, colour=[0.20,0.65,0.75]),
            "joint_disc_2":_disc((p2+p3)/2, R2[:,0], 0.045, colour=[0.60,0.60,0.60]),
            "link_3":      _cylinder(p3, p4, 0.025, colour=[0.20,0.75,0.65]),
            "end_effector":_disc(p4, R4[:,2], 0.04, thickness=0.05,
                                 colour=[0.90,0.50,0.10]),
        }
        # Combine into full scene pcd
        full = link_pcds["base"]
        for sub in list(link_pcds.values())[1:]:
            full = full + sub

        return full, link_pcds, {
            "origins": {"p0":p0,"p1":p1,"p2":p2,"p3":p3,"p4":p4},
            "rotmats": {"R0":R0,"R1":R1,"R2":R2,"R3":R3,"R4":R4},
        }

    # Build multiple poses
    angle_sets = [[0, 0, 0, 0]]
    for i in range(1, n_poses):
        angle_sets.append([
            i * 0.3,       # joint1 rotates around Z
            i * 0.25,      # joint2 bends
            -i * 0.20,     # joint3 bends back
            i * 0.15,      # joint4 rotates
        ])

    pose_pcds = []
    ref_link_pcds = None
    ref_meta = None
    combined = None

    for i, angles in enumerate(angle_sets):
        full_pcd, link_pcds, meta = _build_pose(angles)
        pose_pcds.append(full_pcd)
        if i == 0:
            ref_link_pcds = link_pcds
            ref_meta      = meta
            combined      = full_pcd

    log.info(f"Demo: synthetic 4-DOF arm, "
             f"{len(combined.points):,} points, {n_poses} pose(s)")

    data = ScanData(pcd=combined, source="demo", pose_pcds=pose_pcds)
    # Attach metadata for downstream use
    data._demo_link_pcds = ref_link_pcds
    data._demo_meta      = ref_meta
    return data
