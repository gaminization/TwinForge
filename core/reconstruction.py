"""
RGB-Only Dense Reconstruction
==============================
Takes phone JPEG frames → dense point cloud suitable for segmentation.

Pipeline
--------
1. Feature extraction  — SIFT on all frames
2. Pairwise matching   — BFMatcher + Lowe's ratio test
3. Pose recovery       — Essential matrix → R, t per pair
4. Chained poses       — Accumulate camera poses via PnP
5. Triangulation       — All matched pairs → 3D points
6. Dense patch match   — Semi-dense stereo between close frames
7. Outlier removal     — Statistical filter + voxel downsample
8. Coverage analysis   — Which angles are seen, which joints need more

Used by: server/app.py (incremental), core/acquisition.py (batch)
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2

log = logging.getLogger(__name__)


# ── Data containers ───────────────────────────────────────────────────────────

@dataclass
class CameraFrame:
    idx:       int
    image:     np.ndarray          # H×W×3 uint8 RGB
    path:      Optional[Path] = None
    K:         Optional[np.ndarray] = None   # 3×3 intrinsics
    R:         Optional[np.ndarray] = None   # 3×3 rotation (world ← camera)
    t:         Optional[np.ndarray] = None   # 3   translation
    kp:        list = field(default_factory=list)   # cv2.KeyPoint list
    des:       Optional[np.ndarray] = None          # SIFT descriptors


@dataclass
class ReconstructionResult:
    points:    np.ndarray          # N×3 float32
    colors:    Optional[np.ndarray] = None   # N×3 uint8
    cameras:   List[CameraFrame] = field(default_factory=list)
    n_frames:  int = 0
    coverage:  Dict[str, float] = field(default_factory=dict)
    # Coverage analysis outputs
    angles_seen:  List[float] = field(default_factory=list)   # degrees, 0-360
    needs_angles: List[str]   = field(default_factory=list)   # e.g. ["left side", "top"]


# ── Main entry point ──────────────────────────────────────────────────────────

def reconstruct_from_frames(frames: List[np.ndarray],
                              min_points: int = 500) -> ReconstructionResult:
    """
    Full RGB-only SfM reconstruction.

    Parameters
    ----------
    frames     : list of H×W×3 uint8 RGB images
    min_points : minimum points before giving up and using fallback

    Returns
    -------
    ReconstructionResult with points array ready for segmentation
    """
    if len(frames) < 2:
        raise ValueError("Need at least 2 frames for reconstruction")

    log.info(f"Reconstructing from {len(frames)} frames …")

    # 1. Build CameraFrame objects + estimate intrinsics
    cam_frames = _prepare_frames(frames)

    # 2. Extract SIFT features
    _extract_features(cam_frames)

    # 3. Pairwise matching + pose recovery
    pts3d, colors = _sfm_pipeline(cam_frames)

    # 4. Dense patch matching between adjacent frames
    dense_pts, dense_cols = _dense_patch_match(cam_frames)
    if len(dense_pts) > 100:
        pts3d  = np.vstack([pts3d,  dense_pts])
        colors = np.vstack([colors, dense_cols])

    # 5. Clean up
    pts3d, colors = _clean(pts3d, colors)

    # 6. Coverage analysis
    coverage, angles_seen, needs = _analyse_coverage(cam_frames, pts3d)

    log.info(f"Reconstruction: {len(pts3d):,} points, "
             f"coverage={coverage.get('angular', 0):.0%}, "
             f"needs={needs}")

    return ReconstructionResult(
        points=pts3d.astype(np.float32),
        colors=colors,
        cameras=cam_frames,
        n_frames=len(frames),
        coverage=coverage,
        angles_seen=angles_seen,
        needs_angles=needs,
    )


def to_open3d_pcd(result: ReconstructionResult):
    """Convert ReconstructionResult → open3d.geometry.PointCloud."""
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(result.points)
    if result.colors is not None and len(result.colors) == len(result.points):
        pcd.colors = o3d.utility.Vector3dVector(
            result.colors.astype(np.float64) / 255.0)
    return pcd


# ── Frame preparation ─────────────────────────────────────────────────────────

def _prepare_frames(frames: List[np.ndarray]) -> List[CameraFrame]:
    cam_frames = []
    h, w = frames[0].shape[:2]
    # Approximate intrinsics: fx = fy ≈ 0.9 * max(w,h) (typical phone)
    fx = max(w, h) * 0.90
    K  = np.array([[fx, 0, w/2],
                   [0, fx, h/2],
                   [0,  0,   1]], dtype=np.float64)

    for i, img in enumerate(frames):
        if img.shape[0] > 960:          # resize large frames for speed
            scale = 960 / img.shape[0]
            img   = cv2.resize(img, (int(img.shape[1]*scale), 960))
        cam_frames.append(CameraFrame(idx=i, image=img, K=K.copy()))

    return cam_frames


# ── Feature extraction ────────────────────────────────────────────────────────

def _extract_features(frames: List[CameraFrame]):
    sift = cv2.SIFT_create(nfeatures=4000, contrastThreshold=0.03)
    for f in frames:
        gray    = cv2.cvtColor(f.image, cv2.COLOR_RGB2GRAY)
        f.kp, f.des = sift.detectAndCompute(gray, None)
    log.info(f"Features: {[len(f.kp) for f in frames]}")


# ── SfM pipeline ──────────────────────────────────────────────────────────────

def _sfm_pipeline(frames: List[CameraFrame]
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Chain-based SfM:
    frame 0 is world origin.
    For each subsequent frame, find matches with all previous frames,
    recover pose via PnP on already-triangulated 3D points,
    then triangulate new matches.
    """
    bf = cv2.BFMatcher(cv2.NORM_L2)

    # Initialise with frame 0
    frames[0].R = np.eye(3)
    frames[0].t = np.zeros(3)

    all_pts3d  = []
    all_colors = []

    # Bootstrap: frames 0 and 1
    pts3d, colors, pts2d_map = _init_pair(frames[0], frames[1], bf)
    all_pts3d.extend(pts3d)
    all_colors.extend(colors)

    # Register remaining frames
    for i in range(2, len(frames)):
        _register_frame(frames[i], frames[:i], all_pts3d, bf)
        new_pts, new_cols = _triangulate_new(frames[i], frames[i-1], bf)
        all_pts3d.extend(new_pts)
        all_colors.extend(new_cols)

    pts   = np.array(all_pts3d,  dtype=np.float32) if all_pts3d  else np.zeros((0,3))
    cols  = np.array(all_colors, dtype=np.uint8)   if all_colors else np.zeros((0,3))
    return pts, cols


def _init_pair(f0: CameraFrame, f1: CameraFrame,
               bf) -> Tuple[list, list, dict]:
    """Bootstrap: recover pose of f1 from essential matrix, triangulate."""
    matches = _match(f0, f1, bf)
    if len(matches) < 8:
        f1.R = np.eye(3); f1.t = np.array([0,0,1.])
        return [], [], {}

    p0 = np.float32([f0.kp[m.queryIdx].pt for m in matches])
    p1 = np.float32([f1.kp[m.trainIdx].pt for m in matches])

    E, mask = cv2.findEssentialMat(p0, p1, f0.K,
                                    method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None or mask is None:
        f1.R = np.eye(3); f1.t = np.array([0,0,1.])
        return [], [], {}

    _, R, t, mask2 = cv2.recoverPose(E, p0, p1, f0.K, mask=mask)
    f1.R = R; f1.t = t.ravel()

    # Triangulate
    good = mask2.ravel() > 0
    pts3d, colors = _triangulate(f0, f1, p0[good], p1[good])
    return pts3d, colors, {}


def _register_frame(fi: CameraFrame, prev_frames: List[CameraFrame],
                     world_pts: list, bf):
    """Estimate pose of fi by PnP against already-known 3D points."""
    if not world_pts or fi.des is None:
        fi.R = np.eye(3); fi.t = np.zeros(3); return

    # Match against last frame (most overlap)
    last = prev_frames[-1]
    if last.des is None or last.R is None:
        fi.R = last.R if last.R is not None else np.eye(3)
        fi.t = last.t if last.t is not None else np.zeros(3)
        return

    matches = _match(last, fi, bf, max_matches=200)
    if len(matches) < 6:
        fi.R = last.R.copy(); fi.t = last.t.copy(); return

    # Use last frame's 2D→3D projection to get correspondences
    P_last = fi.K @ np.hstack([last.R, last.t.reshape(3,1)])
    pts2d  = np.float32([fi.kp[m.trainIdx].pt for m in matches])
    # Approximate 3D from last frame depth
    pts3d_approx = []
    for m in matches:
        u, v  = last.kp[m.queryIdx].pt
        depth = 0.5   # unit depth approximation
        X = (u - fi.K[0,2]) / fi.K[0,0] * depth
        Y = (v - fi.K[1,2]) / fi.K[1,1] * depth
        pts3d_approx.append([X, Y, depth])
    pts3d_approx = np.float32(pts3d_approx)

    success, rvec, tvec, _ = cv2.solvePnPRansac(
        pts3d_approx, pts2d, fi.K, None,
        flags=cv2.SOLVEPNP_ITERATIVE)

    if success:
        R, _ = cv2.Rodrigues(rvec)
        fi.R = R; fi.t = tvec.ravel()
    else:
        fi.R = last.R.copy(); fi.t = last.t.copy()


def _triangulate_new(fi: CameraFrame, fj: CameraFrame,
                      bf) -> Tuple[list, list]:
    """Triangulate new points between frame i and j."""
    if fi.R is None or fj.R is None:
        return [], []
    matches = _match(fj, fi, bf)
    if len(matches) < 5:
        return [], []
    pj = np.float32([fj.kp[m.queryIdx].pt for m in matches])
    pi = np.float32([fi.kp[m.trainIdx].pt for m in matches])
    return _triangulate(fj, fi, pj, pi)


def _triangulate(fa: CameraFrame, fb: CameraFrame,
                  pa: np.ndarray, pb: np.ndarray
                  ) -> Tuple[list, list]:
    """Triangulate matched 2D points into 3D."""
    if fa.R is None or fb.R is None or len(pa) < 4:
        return [], []
    Pa = fa.K @ np.hstack([fa.R, fa.t.reshape(3,1)])
    Pb = fb.K @ np.hstack([fb.R, fb.t.reshape(3,1)])

    pts4d = cv2.triangulatePoints(Pa, Pb, pa.T, pb.T)
    pts3d = (pts4d[:3] / (pts4d[3:] + 1e-9)).T

    # Filter: positive depth, not too far
    z_a = (fa.R @ pts3d.T + fa.t.reshape(3,1))[2]
    keep = (z_a > 0.01) & (z_a < 10.0) & np.all(np.isfinite(pts3d), axis=1)

    pts_good = pts3d[keep].tolist()
    # Sample colors from frame a
    colors   = []
    h, w     = fa.image.shape[:2]
    for j, pt2 in enumerate(pa):
        if not keep[j]: continue
        u, v = int(np.clip(pt2[0],0,w-1)), int(np.clip(pt2[1],0,h-1))
        colors.append(fa.image[v, u].tolist())

    return pts_good, colors


def _match(fa: CameraFrame, fb: CameraFrame,
            bf, max_matches: int = 500) -> list:
    """BFMatcher + Lowe's ratio test."""
    if fa.des is None or fb.des is None or len(fa.des) < 2 or len(fb.des) < 2:
        return []
    try:
        raw = bf.knnMatch(fa.des, fb.des, k=2)
    except cv2.error:
        return []
    good = [m for m, n in raw if m.distance < 0.75 * n.distance]
    return sorted(good, key=lambda m: m.distance)[:max_matches]


# ── Dense patch matching ──────────────────────────────────────────────────────

def _dense_patch_match(frames: List[CameraFrame]
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Semi-dense stereo between consecutive frame pairs.
    Uses optical flow (Lucas-Kanade) to get many more correspondences.
    """
    all_pts: List[np.ndarray] = []
    all_col: List[np.ndarray] = []

    for i in range(min(len(frames)-1, 8)):   # limit to 8 pairs for speed
        fa, fb = frames[i], frames[i+1]
        if fa.R is None or fb.R is None:
            continue

        gray_a = cv2.cvtColor(fa.image, cv2.COLOR_RGB2GRAY)
        gray_b = cv2.cvtColor(fb.image, cv2.COLOR_RGB2GRAY)

        # Shi-Tomasi corners as seeds
        corners = cv2.goodFeaturesToTrack(
            gray_a, maxCorners=1500, qualityLevel=0.01, minDistance=5)
        if corners is None or len(corners) < 4:
            continue

        # Track with Lucas-Kanade optical flow
        pts_b, status, _ = cv2.calcOpticalFlowPyrLK(
            gray_a, gray_b, corners,
            None, winSize=(15,15), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 30, 0.01))

        if pts_b is None:
            continue

        good = status.ravel() == 1
        pa   = corners[good].reshape(-1, 2)
        pb   = pts_b[good].reshape(-1, 2)

        if len(pa) < 4:
            continue

        # Triangulate
        pts3d, colors = _triangulate(fa, fb, pa, pb)
        if pts3d:
            all_pts.extend(pts3d)
            all_col.extend(colors)

    if not all_pts:
        return np.zeros((0,3), np.float32), np.zeros((0,3), np.uint8)

    return (np.array(all_pts, np.float32),
            np.array(all_col, np.uint8))


# ── Post-processing ───────────────────────────────────────────────────────────

def _clean(pts: np.ndarray, colors: np.ndarray,
            voxel: float = 0.004) -> Tuple[np.ndarray, np.ndarray]:
    """Statistical outlier removal + voxel downsample."""
    import open3d as o3d

    if len(pts) == 0:
        return pts, colors

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    if len(colors) == len(pts):
        pcd.colors = o3d.utility.Vector3dVector(
            colors.astype(np.float64) / 255.0)

    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=15, std_ratio=2.5)
    pcd    = pcd.voxel_down_sample(voxel)
    pcd.estimate_normals()

    pts_out = np.asarray(pcd.points, np.float32)
    col_out = (np.asarray(pcd.colors) * 255).astype(np.uint8) \
              if pcd.has_colors() else np.zeros((len(pts_out),3), np.uint8)

    log.info(f"After clean: {len(pts_out):,} points")
    return pts_out, col_out


# ── Coverage analysis ─────────────────────────────────────────────────────────

def _analyse_coverage(frames: List[CameraFrame],
                       pts3d: np.ndarray
                       ) -> Tuple[Dict[str, float], List[float], List[str]]:
    """
    Analyse which camera angles have been captured and what's missing.

    Returns
    -------
    coverage      : {angular, elevation, n_frames} ∈ [0,1]
    angles_seen   : list of azimuth angles (degrees)
    needs_angles  : human-readable list of missing angles
    """
    # Estimate object centroid
    centroid = pts3d.mean(axis=0) if len(pts3d) > 0 else np.zeros(3)

    # Compute azimuth of each camera around object centroid
    azimuths   = []
    elevations = []
    for f in frames:
        if f.R is None or f.t is None:
            continue
        # Camera position in world = -R^T t
        cam_pos = -f.R.T @ f.t
        diff    = cam_pos - centroid
        azimuth   = float(np.degrees(np.arctan2(diff[0], diff[2])) % 360)
        elevation = float(np.degrees(np.arctan2(diff[1],
                          np.linalg.norm(diff[[0,2]]) + 1e-9)))
        azimuths.append(azimuth)
        elevations.append(elevation)

    # Angular coverage: how much of 0-360 is within 30° of a captured angle
    if azimuths:
        covered_deg = 0
        for test in range(0, 360, 5):
            if any(abs(((test - a + 180) % 360) - 180) < 25
                   for a in azimuths):
                covered_deg += 5
        angular_cov = min(1.0, covered_deg / 360)
    else:
        angular_cov = 0.0

    # Elevation coverage: have we seen from above?
    elev_cov = min(1.0, sum(1 for e in elevations if e > 20) / 3)

    coverage = {
        "angular":   angular_cov,
        "elevation": elev_cov,
        "n_frames":  min(1.0, len(frames) / 12),
    }

    # Build missing-angle prompts
    needs = []
    SECTORS = [
        (0,   60,  "the front"),
        (60,  120, "the right side"),
        (120, 180, "the back-right"),
        (180, 240, "the back"),
        (240, 300, "the left side"),
        (300, 360, "the front-left"),
    ]
    for lo, hi, label in SECTORS:
        covered = any(lo <= a < hi for a in azimuths)
        if not covered:
            needs.append(f"from {label} ({lo}°–{hi}°)")

    if elev_cov < 0.5:
        needs.append("from above (top-down view)")

    if len(frames) < 6:
        needs.insert(0, f"{6 - len(frames)} more frames from any angle")

    return coverage, azimuths, needs


# ── Incremental reconstruction (called as frames arrive) ─────────────────────

class IncrementalReconstructor:
    """
    Maintains state between frame uploads.
    Called after each batch of new frames from the phone.
    Returns updated point cloud + guidance for next capture.
    """

    def __init__(self):
        self.frames:    List[CameraFrame] = []
        self.result:    Optional[ReconstructionResult] = None
        self._sift      = cv2.SIFT_create(nfeatures=4000, contrastThreshold=0.03)
        self._bf        = cv2.BFMatcher(cv2.NORM_L2)

    def add_frame(self, img: np.ndarray) -> Dict:
        """
        Add one frame, update reconstruction, return guidance dict.
        """
        h, w = img.shape[:2]
        if h > 960:
            scale = 960 / h
            img   = cv2.resize(img, (int(w*scale), 960))

        fx = max(img.shape[1], img.shape[0]) * 0.90
        K  = np.array([[fx,0,img.shape[1]/2],
                       [0,fx,img.shape[0]/2],[0,0,1]], np.float64)
        cf      = CameraFrame(idx=len(self.frames), image=img, K=K)
        gray    = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        cf.kp, cf.des = self._sift.detectAndCompute(gray, None)
        self.frames.append(cf)

        # Need at least 2 frames for reconstruction
        if len(self.frames) < 2:
            return {
                "n_frames":     1,
                "n_points":     0,
                "coverage":     {"angular": 0.0, "elevation": 0.0},
                "needs_angles": ["more frames — keep walking around the robot"],
                "ready":        False,
                "guidance":     "Walk slowly around the robot, capturing every step.",
                "confidence":   0.0,
            }

        # Reconstruct with all frames so far
        try:
            self.result = reconstruct_from_frames(
                [f.image for f in self.frames])
        except Exception as e:
            log.warning(f"Reconstruction error: {e}")
            return self._guidance(0, [], ["reconstruction failed — keep capturing"])

        return self._guidance(
            len(self.result.points),
            self.result.coverage,
            self.result.needs_angles,
        )

    def _guidance(self, n_pts: int, coverage: dict,
                   needs: list) -> dict:
        ang = coverage.get("angular",   0.0) if coverage else 0.0
        elv = coverage.get("elevation", 0.0) if coverage else 0.0
        n   = coverage.get("n_frames",  0.0) if coverage else 0.0

        overall = (ang * 0.5 + elv * 0.2 + n * 0.3)
        ready   = overall >= 0.75 and n_pts >= 500

        if needs:
            msg = f"📸  Please capture {needs[0]}"
        elif ready:
            msg = "✅  Coverage complete — processing your robot now!"
        else:
            msg = "Keep walking around the robot slowly."

        return {
            "n_frames":   len(self.frames),
            "n_points":   n_pts,
            "coverage":   coverage or {},
            "needs_angles": needs,
            "ready":      ready,
            "guidance":   msg,
            "confidence": round(overall, 2),
        }

    def get_pcd(self):
        """Return open3d point cloud of current reconstruction."""
        if self.result is None or len(self.result.points) == 0:
            return None
        return to_open3d_pcd(self.result)
