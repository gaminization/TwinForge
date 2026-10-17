"""
reconstruction.py — Incremental 3D reconstruction from phone photos.

Backend priority:
  1. VGGT  (CVPR 2025 Best Paper — facebook/VGGT-1B)
     Dense per-pixel point maps, accurate poses, no COLMAP needed.
  2. DUSt3R (CVPR 2024 — naver/dust3r)
     Dense stereo reconstruction, global alignment.
  3. Enhanced SIFT SfM (always available, no GPU needed)
     SIFT + RANSAC Essential matrix + triangulation + dense SGM.

Install neural backends (run once):
  pip install torch torchvision huggingface_hub
  # VGGT:
  pip install git+https://github.com/facebookresearch/vggt.git
  # DUSt3R:
  git clone --recursive https://github.com/naver/dust3r && pip install -e dust3r/
"""
import cv2
import numpy as np
import open3d as o3d
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

log = logging.getLogger("twinforge.reconstruction")


@dataclass
class FrameResult:
    frame_idx:       int
    n_points:        int
    total_points:    int
    coverage:        float
    guidance:        str
    covered_sectors: List[int]
    backend:         str = "sift"


# ─────────────────────────────────────────────────────────────
# Backend detection
# ─────────────────────────────────────────────────────────────

def _detect_backend() -> str:
    try:
        import torch
        try:
            import vggt          # noqa
            return "vggt"
        except ImportError:
            pass
        try:
            import dust3r        # noqa
            return "dust3r"
        except ImportError:
            pass
        return "sift"
    except ImportError:
        return "sift"


# ─────────────────────────────────────────────────────────────
# IncrementalReconstructor — thin wrapper that dispatches
# ─────────────────────────────────────────────────────────────

class IncrementalReconstructor:
    N_SECTORS   = 12
    MIN_COVERAGE = 0.75

    def __init__(self, job_dir: Path):
        self.job_dir  = Path(job_dir)
        self._backend = _detect_backend()
        log.info(f"Reconstruction backend: {self._backend.upper()}")
        self._impl = _make_impl(self._backend, self.job_dir)

    def add_frame(self, img_bytes: bytes) -> FrameResult:
        return self._impl.add_frame(img_bytes)

    def get_pcd(self, voxel_size: float = 0.002) -> o3d.geometry.PointCloud:
        return self._impl.get_pcd(voxel_size)

    @property
    def coverage(self) -> float:
        return self._impl.coverage

    @property
    def is_complete(self) -> bool:
        return self.coverage >= self.MIN_COVERAGE


def _make_impl(backend: str, job_dir: Path):
    if backend == "vggt":
        try:
            return _VGGTReconstructor(job_dir)
        except Exception as e:
            log.warning(f"VGGT init failed ({e}), trying DUSt3R…")
            backend = "dust3r"
    if backend == "dust3r":
        try:
            return _DUSt3RReconstructor(job_dir)
        except Exception as e:
            log.warning(f"DUSt3R init failed ({e}), using SIFT fallback")
    return _SIFTReconstructor(job_dir)


# ─────────────────────────────────────────────────────────────
# VGGT backend  (CVPR 2025 Best Paper)
# ─────────────────────────────────────────────────────────────

class _VGGTReconstructor:
    """
    Uses facebook/VGGT-1B to get dense per-pixel point maps from images.
    Requires: pip install torch torchvision huggingface_hub vggt
    """

    def __init__(self, job_dir: Path):
        import torch
        from vggt.models.vggt import VGGT
        from vggt.utils.load_fn import load_and_preprocess_images

        self._torch = torch
        self._load_images = load_and_preprocess_images
        self.job_dir = Path(job_dir)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device
        log.info(f"VGGT: loading model on {device}…")
        self._model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()
        log.info("VGGT: model ready")

        self._frame_paths: List[Path] = []
        self._all_pts:  List[np.ndarray] = []
        self._all_clr:  List[np.ndarray] = []
        self._sector_hits = np.zeros(12, dtype=int)

    MAX_FRAMES = 24          # cap per forward pass (VRAM); evenly subsampled
    CONF_KEEP  = 0.5         # keep the most confident 50% of pixels

    def add_frame(self, img_bytes: bytes) -> FrameResult:
        import torch
        from vggt.utils.geometry import unproject_depth_map_to_point_map
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        idx  = len(self._frame_paths)
        path = self.job_dir / f"vggt_{idx:04d}.jpg"
        path.write_bytes(img_bytes)
        self._frame_paths.append(path)

        if len(self._frame_paths) < 2:
            return FrameResult(idx, 0, 0, 0.0,
                               "Capture a second view to start reconstruction",
                               [], backend="vggt")

        # Frames used this pass (evenly subsampled if there are many)
        paths = self._frame_paths
        if len(paths) > self.MAX_FRAMES:
            sel = np.linspace(0, len(paths) - 1, self.MAX_FRAMES).round().astype(int)
            paths = [paths[i] for i in sel]
        images = self._load_images([str(p) for p in paths], mode="crop")   # (S,3,H,W) in [0,1]

        use_cuda = torch.cuda.is_available() and self._device.startswith("cuda")
        dtype = (torch.bfloat16 if use_cuda and torch.cuda.get_device_capability()[0] >= 8
                 else torch.float16 if use_cuda else torch.bfloat16)

        with torch.no_grad():
            with torch.autocast(device_type="cuda" if use_cuda else "cpu", dtype=dtype):
                preds = self._model(images.to(self._device))

            # VGGT returns a dict of batched tensors: pose_enc (B,S,9), depth (B,S,H,W,1),
            # depth_conf (B,S,H,W).  Decode OpenCV world->camera extrinsics + intrinsics.
            extri, intri = pose_encoding_to_extri_intri(
                preds["pose_enc"].float(), images.shape[-2:])
            extri  = extri[0].float().cpu().numpy()              # (S,3,4)
            intri  = intri[0].float().cpu().numpy()              # (S,3,3)
            depth  = preds["depth"][0].float().cpu().numpy()     # (S,H,W,1)
            conf   = preds["depth_conf"][0].float().cpu().numpy()  # (S,H,W)

        world = unproject_depth_map_to_point_map(depth, extri, intri)   # (S,H,W,3)
        rgb   = images.float().permute(0, 2, 3, 1).cpu().numpy()                # (S,H,W,3)

        thr  = np.quantile(conf, 1.0 - self.CONF_KEEP)
        mask = (conf >= thr) & (depth[..., 0] > 0) & np.isfinite(world).all(-1)
        pts_all = world[mask]
        clr_all = rgb[mask]

        # Each pass is expressed in the first frame's coordinates, so REPLACE (never append).
        self._all_pts = [pts_all]
        self._all_clr = [clr_all]

        # Coverage: azimuth of each camera around the scene centre.
        R, t = extri[:, :, :3], extri[:, :, 3]
        centres = -np.einsum("sij,sj->si", R.transpose(0, 2, 1), t)    # camera centres in world
        mid = np.median(pts_all, axis=0) if len(pts_all) else centres.mean(0)
        self._sector_hits[:] = 0
        for c in centres:
            az = np.degrees(np.arctan2(c[1] - mid[1], c[0] - mid[0]))
            self._sector_hits[int((az % 360) / 30) % 12] += 1

        cov = float(np.sum(self._sector_hits > 0)) / 12
        covered = [i for i, h in enumerate(self._sector_hits) if h > 0]
        return FrameResult(idx, len(pts_all), len(pts_all), cov,
                           _guidance(cov, self._sector_hits), covered, backend="vggt")

    def get_pcd(self, voxel_size=0.002) -> o3d.geometry.PointCloud:
        if not self._all_pts:
            return o3d.geometry.PointCloud()
        pts = np.vstack(self._all_pts)
        clr = np.vstack(self._all_clr)
        return _build_pcd(pts, clr, voxel_size)

    @property
    def coverage(self) -> float:
        return float(np.sum(self._sector_hits > 0)) / 12


# ─────────────────────────────────────────────────────────────
# DUSt3R backend  (CVPR 2024)
# ─────────────────────────────────────────────────────────────

class _DUSt3RReconstructor:
    """
    Uses naver/DUSt3R for pairwise dense 3D reconstruction + global alignment.
    Requires: git clone --recursive https://github.com/naver/dust3r; pip install -e dust3r/
    Model: DUSt3R_ViTLarge_BaseDecoder_512_dpt (auto-downloaded from HuggingFace)
    """

    MODEL_NAME = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"

    def __init__(self, job_dir: Path):
        import torch
        from dust3r.model import AsymmetricCroCo3DStereo
        from dust3r.utils.image import load_images
        from dust3r.inference import inference
        from dust3r.post_process import estimate_focal_knowing_depth
        from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

        self._torch   = torch
        self._load    = load_images
        self._infer   = inference
        self._ga      = global_aligner
        self._GAMode  = GlobalAlignerMode
        self.job_dir  = Path(job_dir)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device  = device
        log.info(f"DUSt3R: loading model on {device}…")
        self._model   = AsymmetricCroCo3DStereo.from_pretrained(
            self.MODEL_NAME).to(device).eval()
        log.info("DUSt3R: model ready")

        self._frame_paths: List[Path] = []
        self._all_pts: List[np.ndarray] = []
        self._all_clr: List[np.ndarray] = []
        self._sector_hits = np.zeros(12, dtype=int)

    def add_frame(self, img_bytes: bytes) -> FrameResult:
        idx  = len(self._frame_paths)
        path = self.job_dir / f"dust3r_{idx:04d}.jpg"
        path.write_bytes(img_bytes)
        self._frame_paths.append(path)

        if len(self._frame_paths) < 2:
            return FrameResult(idx, 0, 0, 0.0,
                               "📸 Capture a second view to start reconstruction",
                               [], backend="dust3r")

        # Reconstruct using latest pair + all previous
        paths = [str(p) for p in self._frame_paths[-10:]]
        try:
            images = self._load(paths, size=512)
            pairs  = _make_pairs(images)
            output = self._infer(pairs, self._model, self._device,
                                 batch_size=1, verbose=False)
            scene  = self._ga(output, self._GAMode.GLOBAL_POINTS_SMOOTHED)
            scene.compute_global_alignment(init="mst", niter=50, schedule="linear")

            pts3d = scene.get_pts3d()
            masks = scene.get_masks()
            imgs  = scene.imgs

            pts_new, clr_new = [], []
            for pts, msk, img in zip(pts3d, masks, imgs):
                pts_np = pts.detach().cpu().numpy()
                msk_np = msk.detach().cpu().numpy().astype(bool)
                img_np = img  # H×W×3 float [0,1]
                pts_new.append(pts_np[msk_np])
                clr_new.append(img_np[msk_np])

                # Yaw from scene camera pose
                cam = scene.get_im_poses()
                if len(cam) > 0:
                    R = cam[-1].detach().cpu().numpy()[:3,:3]
                    yaw = float(np.degrees(np.arctan2(R[1,0], R[0,0])))
                    sector = int((yaw % 360) / 30) % 12
                    self._sector_hits[sector] += 1

            self._all_pts.extend(pts_new)
            self._all_clr.extend(clr_new)

        except Exception as e:
            log.warning(f"DUSt3R reconstruction error: {e}")

        total  = sum(len(p) for p in self._all_pts)
        cov    = float(np.sum(self._sector_hits > 0)) / 12
        covered = [i for i,h in enumerate(self._sector_hits) if h > 0]
        return FrameResult(idx, 0, total, cov,
                           _guidance(cov, self._sector_hits),
                           covered, backend="dust3r")

    def get_pcd(self, voxel_size=0.002) -> o3d.geometry.PointCloud:
        if not self._all_pts:
            return o3d.geometry.PointCloud()
        return _build_pcd(np.vstack(self._all_pts),
                          np.vstack(self._all_clr), voxel_size)

    @property
    def coverage(self) -> float:
        return float(np.sum(self._sector_hits > 0)) / 12


def _make_pairs(images):
    """All consecutive + skip-1 pairs."""
    pairs = []
    n = len(images)
    for i in range(n):
        for j in range(i+1, min(i+3, n)):
            pairs.append((images[i], images[j]))
    return pairs


# ─────────────────────────────────────────────────────────────
# Enhanced SIFT SfM fallback (no GPU, no large model)
# ─────────────────────────────────────────────────────────────

class _SIFTReconstructor:
    """
    Classical SfM: SIFT → BF matching → Essential matrix → triangulation.
    Always available (uses only OpenCV + numpy).
    Quality: good for 10–50 frames, adequate for robot arms.
    """
    MIN_INLIERS = 20

    def __init__(self, job_dir: Path):
        self.job_dir = Path(job_dir)
        self._sift   = cv2.SIFT_create(nfeatures=4000, contrastThreshold=0.03)
        self._bf     = cv2.BFMatcher(cv2.NORM_L2)
        self._frames : List[np.ndarray] = []
        self._kps    : List        = []
        self._descs  : List[np.ndarray] = []
        self._poses  : List[np.ndarray] = []
        self._pts    : List[np.ndarray] = []
        self._clrs   : List[np.ndarray] = []
        self._sector_hits = np.zeros(12, dtype=int)
        self._K: Optional[np.ndarray] = None

    def add_frame(self, img_bytes: bytes) -> FrameResult:
        nparr = np.frombuffer(img_bytes, np.uint8)
        bgr   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Could not decode image")

        h, w = bgr.shape[:2]
        if self._K is None:
            f = max(w, h) * 1.2
            self._K = np.array([[f,0,w/2],[0,f,h/2],[0,0,1]], dtype=float)

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        kp, desc = self._sift.detectAndCompute(gray, None)
        idx = len(self._frames)
        self._frames.append(bgr)
        self._kps.append(kp)
        self._descs.append(desc)

        new_pts = 0
        yaw = None

        if idx == 0:
            self._poses.append(np.eye(4))
        elif desc is not None and self._descs[-2] is not None:
            pose, pts3d, clr, yaw = self._process_pair(idx-1, idx, bgr)
            if pose is not None:
                self._poses.append(pose)
                if pts3d is not None and len(pts3d) > 0:
                    self._pts.append(pts3d)
                    self._clrs.append(clr)
                    new_pts = len(pts3d)
                    if yaw is not None:
                        sector = int((yaw % 360) / 30) % 12
                        self._sector_hits[sector] += 1
            else:
                self._poses.append(self._poses[-1].copy())

        total   = sum(len(p) for p in self._pts)
        cov     = float(np.sum(self._sector_hits > 0)) / 12
        covered = [i for i,h in enumerate(self._sector_hits) if h > 0]
        return FrameResult(idx, new_pts, total, cov,
                           _guidance(cov, self._sector_hits),
                           covered, backend="sift")

    def _process_pair(self, i, j, bgr_j):
        kp_i, desc_i = self._kps[i], self._descs[i]
        kp_j, desc_j = self._kps[j], self._descs[j]
        if desc_i is None or desc_j is None or len(desc_i)<8 or len(desc_j)<8:
            return None,None,None,None

        raw  = self._bf.knnMatch(desc_i, desc_j, k=2)
        good = [m for m,n in raw if m.distance < 0.75*n.distance]
        if len(good) < self.MIN_INLIERS:
            return None,None,None,None

        pts_i = np.float32([kp_i[m.queryIdx].pt for m in good])
        pts_j = np.float32([kp_j[m.trainIdx].pt for m in good])

        E, mask = cv2.findEssentialMat(pts_i, pts_j, self._K,
                                        method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None or mask is None: return None,None,None,None

        inl = mask.ravel().astype(bool)
        if inl.sum() < self.MIN_INLIERS: return None,None,None,None

        n_in, R, t, _ = cv2.recoverPose(E, pts_i[inl], pts_j[inl], self._K)
        if n_in < self.MIN_INLIERS: return None,None,None,None

        pose_i = self._poses[i]
        R_wi, t_wi = pose_i[:3,:3], pose_i[:3,3:]
        R_wj = R_wi @ R.T
        t_wj = t_wi - R_wj @ t
        pose_j = np.eye(4)
        pose_j[:3,:3] = R_wj
        pose_j[:3,3:] = t_wj

        P_i = self._K @ pose_i[:3]
        P_j = self._K @ pose_j[:3]
        pts4d = cv2.triangulatePoints(P_i, P_j, pts_i[inl].T, pts_j[inl].T)
        pts3d = (pts4d[:3] / pts4d[3]).T

        d_i = (R_wi @ pts3d.T + t_wi)[2]
        d_j = (R_wj @ pts3d.T + t_wj)[2]
        valid = (d_i>0) & (d_j>0) & (np.linalg.norm(pts3d,axis=1)<5.0)
        pts3d = pts3d[valid]
        if len(pts3d) == 0: return pose_j,None,None,None

        h,w  = bgr_j.shape[:2]
        kp_v = pts_j[inl][valid].astype(int)
        kp_v[:,0] = np.clip(kp_v[:,0], 0, w-1)
        kp_v[:,1] = np.clip(kp_v[:,1], 0, h-1)
        clr = bgr_j[kp_v[:,1], kp_v[:,0]][:,::-1].astype(float)/255.0

        yaw = float(np.degrees(np.arctan2(R_wj[1,0], R_wj[0,0])))
        return pose_j, pts3d, clr, yaw

    def get_pcd(self, voxel_size=0.002) -> o3d.geometry.PointCloud:
        if not self._pts:
            return o3d.geometry.PointCloud()
        return _build_pcd(np.vstack(self._pts), np.vstack(self._clrs), voxel_size)

    @property
    def coverage(self) -> float:
        return float(np.sum(self._sector_hits > 0)) / 12


# ─────────────────────────────────────────────────────────────
# Shared utilities
# ─────────────────────────────────────────────────────────────

def _build_pcd(pts: np.ndarray, clr: np.ndarray,
               voxel_size: float) -> o3d.geometry.PointCloud:
    """Clean + downsample + normals. Scale-adaptive: neural / SfM output has arbitrary units."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(pts, dtype=np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.clip(np.asarray(clr, dtype=np.float64), 0, 1))
    if len(pcd.points) == 0:
        return pcd
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    if len(pcd.points) == 0:
        return pcd
    # Keep only the dense core (drops far-flung background / sky points), then size voxels
    # relative to the object instead of assuming metres.
    ext  = np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound())
    diag = float(np.linalg.norm(ext))
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(diag / 400.0)
    ext  = np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound())
    diag = float(np.linalg.norm(ext))
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=diag / 60.0, max_nn=30))
    if len(pcd.points) <= 50000:
        try:
            pcd.orient_normals_consistent_tangent_plane(k=15)
        except Exception:
            pcd.orient_normals_towards_camera_location(pcd.get_center() + np.array([0, 0, 10 * diag]))
    else:   # too slow at this size: orient away from the centroid (object-centric scans)
        c = pcd.get_center(); n = np.asarray(pcd.normals); p = np.asarray(pcd.points)
        flip = np.einsum("ij,ij->i", n, p - c) < 0
        n[flip] *= -1
        pcd.normals = o3d.utility.Vector3dVector(n)
    log.info(f"Final PCD: {len(pcd.points):,} points")
    return pcd


def _guidance(coverage: float, hits: np.ndarray) -> str:
    missing = [i for i in range(12) if hits[i] == 0]
    if coverage >= 1.0:
        return "✅ Full coverage! Processing…"
    if coverage >= 0.75:
        return f"✅ {coverage:.0%} covered — tap 'Build Model' or keep going."
    if not missing:
        return f"📸 Keep capturing — {coverage:.0%} covered"

    # Longest contiguous missing run
    best_start, best_len, run_s, run_l = missing[0], 1, missing[0], 1
    for k in range(1, len(missing)):
        if missing[k] == missing[k-1]+1:
            run_l += 1
            if run_l > best_len:
                best_len, best_start = run_l, run_s
        else:
            run_s, run_l = missing[k], 1
    gap_center = (best_start + best_len//2) * 30
    dirs = ["front","front-right","right","right","right-back","back-right",
            "back","back-left","left","left","front-left","front-left"]
    direction = dirs[(gap_center//30) % 12]
    return f"📸 Move to the {direction} side (~{gap_center}°) — {coverage:.0%} covered"
