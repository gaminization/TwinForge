"""
live.py - real-time incremental reconstruction for the phone capture flow.

Every photo the phone uploads is queued and processed by ONE background worker thread, so the HTTP
server never blocks (the old code ran model download + inference inside the request handler, which
froze the server and left the phone at 0%).  After every processed batch the worker:

    1. updates the 3D model        (the engine adds the new view(s))
    2. recomputes real progress    (registered views, orbit coverage, point count)
    3. writes a display-ready preview point cloud  -> served at /jobs/<id>/preview
    4. reports everything through `on_update(state)` -> WebSocket -> phone + PC live viewer

Engines (same interface):
    SfMEngine   - CPU-only incremental Structure-from-Motion (SIFT + PnP + triangulation).  Each new
                  photo is localised against the existing map and adds new 3D points.
    VGGTEngine  - facebook/VGGT-1B.  Re-estimates all poses + dense depth jointly from the most recent
                  views each time a photo arrives (the whole cloud refines with every picture).
                  Weights (~5 GB) download in a background thread at server start (`ModelHost`).
"""
from __future__ import annotations

import io
import json
import logging
import os
import queue
import threading
import time
import traceback
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

log = logging.getLogger("twinforge.live")

N_SECTORS = 12
READY_COVERAGE = 0.75      # fraction of the 12 orbit sectors that must have a camera
READY_VIEWS = 12           # registered views needed before the model is called "ready"
TARGET_VIEWS = 24          # progress bar reaches its view-count share at this many registered views
MAX_PREVIEW_POINTS = 80_000


# ───────────────────────────── geometry helpers ─────────────────────────────

def _rot_a_to_b(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
    v = np.cross(a, b); c = float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def display_frame(pts: np.ndarray, centres: np.ndarray, cam_y_axes: np.ndarray):
    """
    Similarity transform into a viewer frame: +Y up, object centred at the origin, ~1 unit across.
    `cam_y_axes` are the cameras' image-down axes in world coordinates; a hand-held phone is roughly
    upright so  up = -mean(down).  Returns (R, centre, scale): p_display = R @ (p - centre) * scale.
    """
    down = cam_y_axes.mean(0)
    up = -down / (np.linalg.norm(down) + 1e-12)
    R = _rot_a_to_b(up, np.array([0.0, 1.0, 0.0]))
    centre = np.median(pts, axis=0) if len(pts) else centres.mean(0)
    q = (pts - centre) @ R.T
    lo, hi = np.percentile(q, [2, 98], axis=0) if len(q) > 20 else (q.min(0), q.max(0))
    scale = 1.0 / max(float((hi - lo).max()), 1e-9)
    return R, centre, scale


def orbit_coverage(centres_d: np.ndarray) -> Dict:
    """
    centres_d: camera centres in the display frame (y up, object at origin).
    Sector 0 is centred on the FIRST camera, so guidance is relative to where the user started.
    """
    az = np.degrees(np.arctan2(centres_d[:, 2], centres_d[:, 0]))
    az = (az - az[0] + 15.0) % 360.0                       # first camera in the middle of sector 0
    sectors = (az // (360.0 / N_SECTORS)).astype(int) % N_SECTORS
    hits = np.bincount(sectors, minlength=N_SECTORS)
    r_h = np.linalg.norm(centres_d[:, [0, 2]], axis=1) + 1e-9
    elev = np.degrees(np.arctan2(centres_d[:, 1], r_h))
    return {"hits": hits, "sectors": sectors, "azimuth": az,
            "n_covered": int((hits > 0).sum()), "high_view": bool((elev > 25).any()),
            "elevation": elev}


def guidance_text(cov: Dict, n_reg: int, n_recv: int, lost: bool, waiting_baseline: bool) -> str:
    if waiting_baseline:
        return "Move the phone sideways ~10 cm (don't just rotate) so the first two views can be matched"
    if lost:
        return "Tracking lost - go back to where you last captured, then take shots closer together"
    c = cov["n_covered"]
    if c >= N_SECTORS and cov["high_view"]:
        return "Full coverage - building the model"
    if c >= N_SECTORS * READY_COVERAGE and n_reg >= READY_VIEWS:
        if not cov["high_view"]:
            return "Good. Raise the phone and capture a few views from above, or tap Finish"
        return "Coverage good - tap Finish (or keep going for more detail)"
    if c == N_SECTORS:
        return "Raise the phone and capture a few views from above"
    return f"Keep circling the robot in the same direction ({c}/{N_SECTORS} directions covered)"


def progress_pct(n_cov: int, n_reg: int) -> int:
    return int(round(100 * (0.6 * n_cov / N_SECTORS + 0.4 * min(1.0, n_reg / TARGET_VIEWS))))


# ─────────────────────────────── SfM engine ────────────────────────────────

class _Frame:
    __slots__ = ("kp", "desc", "pose", "mid", "rgb", "path")

    def __init__(self, kp, desc, rgb, path):
        self.kp = kp                        # (N,2) float32 pixel coordinates
        self.desc = desc                    # (N,128) float32
        self.pose: Optional[np.ndarray] = None   # (3,4) world->camera
        self.mid = np.full(len(kp), -1, dtype=np.int64)   # keypoint -> map point id
        self.rgb = rgb                      # (H,W,3) uint8 (downscaled)
        self.path = path


class SfMEngine:
    """Incremental SfM.  CPU only.  Arbitrary (but consistent) scale; no bundle adjustment."""
    name = "sfm"
    MAX_SIDE = 960
    MIN_TRACK_INLIERS = 12

    def __init__(self, work_dir: Path):
        self.work_dir = Path(work_dir)
        self._sift = cv2.SIFT_create(nfeatures=6000, contrastThreshold=0.015)
        self._bf = cv2.BFMatcher(cv2.NORM_L2)
        self.frames: List[_Frame] = []
        self.reg: List[int] = []                   # indices of registered frames
        self.keyframes: List[int] = []
        self.map_xyz: List[np.ndarray] = []
        self.map_rgb: List[np.ndarray] = []
        self.map_obs: List[int] = []
        self.K: Optional[np.ndarray] = None
        self.ref_idx: Optional[int] = None         # first frame, waiting for a baseline
        self.lost = False
        self.waiting_baseline = False
        self.model_state = {"state": "ready", "msg": "SfM (CPU) - no model download needed"}

    # ---- public -----------------------------------------------------------
    def add_paths(self, paths: List[Path]) -> None:
        for p in paths:
            self._add(p)

    def snapshot(self):
        """(points (N,3), colours (N,3) uint8, cams) - cams: list of dict(R, t, K, wh) for registered views"""
        if not self.map_xyz:
            return None
        xyz = np.asarray(self.map_xyz); rgb = np.asarray(self.map_rgb, dtype=np.uint8)
        keep = np.asarray(self.map_obs) >= 2
        cams = []
        for i in self.reg:
            f = self.frames[i]
            h, w = f.rgb.shape[:2]
            cams.append({"R": f.pose[:, :3], "t": f.pose[:, 3], "K": self.K, "wh": (w, h)})
        return xyz[keep], rgb[keep], cams

    @property
    def n_registered(self) -> int:
        return len(self.reg)

    # ---- internals --------------------------------------------------------
    def _load(self, path: Path) -> Optional[_Frame]:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            log.warning("could not decode %s", path)
            return None
        h, w = bgr.shape[:2]
        s = self.MAX_SIDE / max(h, w)
        if s < 1:
            bgr = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        h, w = bgr.shape[:2]
        if self.K is None:
            f = 1.2 * max(w, h)               # typical phone main camera (~65-70 deg HFOV)
            self.K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        kp, desc = self._sift.detectAndCompute(gray, None)
        if desc is None or len(kp) < 30:
            return None
        pts = np.float32([k.pt for k in kp])
        return _Frame(pts, desc.astype(np.float32), bgr[:, :, ::-1].copy(), str(path))

    def _match(self, d_a, d_b):
        if len(d_a) < 8 or len(d_b) < 8:
            return np.empty((0, 2), int)
        knn = self._bf.knnMatch(d_a, d_b, k=2)
        out = [(m.queryIdx, m.trainIdx) for m, n in (x for x in knn if len(x) == 2)
               if m.distance < 0.82 * n.distance]
        # mutual-best check keeps the match set clean
        if not out:
            return np.empty((0, 2), int)
        out = np.array(out)
        back = self._bf.match(d_b[out[:, 1]], d_a)
        ok = np.array([b.trainIdx == q for b, q in zip(back, out[:, 0])])
        return out[ok]

    def _add(self, path: Path) -> None:
        fr = self._load(path)
        if fr is None:
            self.lost = True
            return
        self.frames.append(fr)
        idx = len(self.frames) - 1
        if self.ref_idx is None and not self.reg:
            self.ref_idx = idx
            self.waiting_baseline = True
            return
        if not self.reg:
            self._try_init(idx)
            if not self.reg and idx - self.ref_idx >= 6:   # reference too old: restart from this frame
                self.ref_idx = idx
        else:
            self._track(idx)

    # initialisation -------------------------------------------------------
    def _try_init(self, idx: int) -> None:
        a, b = self.frames[self.ref_idx], self.frames[idx]
        m = self._match(a.desc, b.desc)
        if len(m) < 25:
            return
        pa, pb = a.kp[m[:, 0]], b.kp[m[:, 1]]
        disp = np.median(np.linalg.norm(pa - pb, axis=1))
        if disp < 0.02 * self.K[0, 2] * 2:           # < 2% of image width: no usable baseline yet
            return
        E, mask = cv2.findEssentialMat(pa, pb, self.K, cv2.RANSAC, 0.999, 1.0)
        if E is None or mask is None or mask.sum() < 20:
            return
        inl = mask.ravel().astype(bool)
        _, R, t, mk = cv2.recoverPose(E, pa[inl], pb[inl], self.K)
        P0 = np.hstack([np.eye(3), np.zeros((3, 1))]); P1 = np.hstack([R, t])
        a.pose, b.pose = P0, P1
        n = self._triangulate(self.ref_idx, idx, m[inl])
        if n < 40:
            a.pose = b.pose = None
            a.mid[:] = -1; b.mid[:] = -1
            self.map_xyz.clear(); self.map_rgb.clear(); self.map_obs.clear()
            return
        # fix the gauge: median depth of the initial map = 1
        X = np.asarray(self.map_xyz); depth = (X @ P0[:, :3].T + P0[:, 3])[:, 2]
        s = 1.0 / max(np.median(depth), 1e-9)
        self.map_xyz = [x * s for x in self.map_xyz]
        b.pose = np.hstack([R, t * s])
        self.reg = [self.ref_idx, idx]; self.keyframes = [self.ref_idx, idx]
        self.waiting_baseline = False
        self.lost = False

    # tracking -------------------------------------------------------------
    def _track(self, idx: int) -> None:
        cur = self.frames[idx]
        refs = list(dict.fromkeys(self.reg[-3:] + self.keyframes[-6:]))
        obj, img, src = [], [], {}
        for ri in refs:
            rf = self.frames[ri]
            m = self._match(cur.desc, rf.desc)
            for q, t in m:
                mid = rf.mid[t]
                if mid >= 0 and q not in src:
                    src[q] = mid
        if len(src) < self.MIN_TRACK_INLIERS:
            self.lost = True
            return
        q_idx = np.array(list(src.keys())); m_idx = np.array([src[q] for q in q_idx])
        X = np.asarray(self.map_xyz)[m_idx]; x = cur.kp[q_idx].astype(np.float64)
        ok, rvec, tvec, inl = cv2.solvePnPRansac(
            X, x, self.K, None, iterationsCount=300, reprojectionError=4.0,
            confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
        if not ok or inl is None or len(inl) < self.MIN_TRACK_INLIERS:
            self.lost = True
            return
        inl = inl.ravel()
        rvec, tvec = cv2.solvePnPRefineLM(X[inl], x[inl], self.K, None, rvec, tvec)
        R, _ = cv2.Rodrigues(rvec)
        cur.pose = np.hstack([R, tvec.reshape(3, 1)])
        cur.mid[q_idx[inl]] = m_idx[inl]
        for mid in m_idx[inl]:
            self.map_obs[mid] += 1
        self.reg.append(idx)
        self.lost = False
        # grow the map: triangulate unmapped matches against the most recent registered neighbours
        for ri in self.reg[-3:-1][::-1]:
            rf = self.frames[ri]
            m = self._match(cur.desc, rf.desc)
            m = np.array([(q, t) for q, t in m if cur.mid[q] < 0 and rf.mid[t] < 0], dtype=int).reshape(-1, 2)
            if len(m):
                self._triangulate(ri, idx, np.stack([m[:, 1], m[:, 0]], 1))
        # keyframe policy: every 4th registered view is kept as a long-term reference
        if len(self.reg) % 4 == 0:
            self.keyframes.append(idx)

    def _triangulate(self, ia: int, ib: int, m: np.ndarray) -> int:
        """m rows: (kp index in frame ia, kp index in frame ib)."""
        A, B = self.frames[ia], self.frames[ib]
        if len(m) == 0:
            return 0
        pa, pb = A.kp[m[:, 0]].astype(np.float64), B.kp[m[:, 1]].astype(np.float64)
        PA, PB = self.K @ A.pose, self.K @ B.pose
        X = cv2.triangulatePoints(PA, PB, pa.T, pb.T)
        X = (X[:3] / X[3]).T
        ok = np.isfinite(X).all(1)
        ca, cb = A.pose[:, :3] @ X.T + A.pose[:, 3:], B.pose[:, :3] @ X.T + B.pose[:, 3:]
        ok &= (ca[2] > 0) & (cb[2] > 0)
        for P, p in ((PA, pa), (PB, pb)):
            pr = (P[:, :3] @ X.T + P[:, 3:]); pr = (pr[:2] / pr[2]).T
            ok &= np.linalg.norm(pr - p, axis=1) < 2.0
        # parallax angle between the two viewing rays
        Ca = -A.pose[:, :3].T @ A.pose[:, 3]; Cb = -B.pose[:, :3].T @ B.pose[:, 3]
        ra, rb = X - Ca, X - Cb
        cosang = (ra * rb).sum(1) / (np.linalg.norm(ra, axis=1) * np.linalg.norm(rb, axis=1) + 1e-12)
        ok &= np.degrees(np.arccos(np.clip(cosang, -1, 1))) > 1.2
        if self.map_xyz:
            med = np.median(np.linalg.norm(np.asarray(self.map_xyz) - Ca, axis=1))
            ok &= np.linalg.norm(ra, axis=1) < 6 * med
        n0 = len(self.map_xyz)
        for k in np.flatnonzero(ok):
            mid = len(self.map_xyz)
            self.map_xyz.append(X[k])
            u, v = pb[k].astype(int)
            h, w = B.rgb.shape[:2]
            self.map_rgb.append(B.rgb[min(max(v, 0), h - 1), min(max(u, 0), w - 1)])
            self.map_obs.append(2)
            A.mid[m[k, 0]] = mid; B.mid[m[k, 1]] = mid
        return len(self.map_xyz) - n0


# ─────────────────────────────── VGGT engine ───────────────────────────────

class ModelHost:
    """Loads VGGT once, in a background thread, and exposes a progress-reporting state dict."""
    REPO = "facebook/VGGT-1B"
    SIZE_BYTES = 5.03e9

    def __init__(self):
        self.state = {"state": "idle", "msg": "", "pct": 0}
        self.model = None
        self.device = "cpu"
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

    @staticmethod
    def available() -> bool:
        try:
            import torch  # noqa
            import vggt   # noqa
            return True
        except Exception:
            return False

    def start(self) -> None:
        with self._lock:
            if self._thread or self.model is not None:
                return
            self._thread = threading.Thread(target=self._load, name="vggt-loader", daemon=True)
            self._thread.start()

    def release(self) -> None:
        """Free the model's GPU/CPU memory (the final build needs the RAM); it reloads lazily from the cache."""
        with self._lock:
            self.model = None
            self._thread = None
            self.state = {"state": "idle", "msg": "VGGT unloaded", "pct": 0}
        try:
            import gc, torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _hf_pct(self) -> int:
        root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
        d = root / "models--facebook--VGGT-1B"
        try:
            sizes = [f.stat().st_size for f in d.rglob("*") if f.is_file() and not f.is_symlink()]
            return min(99, int(100 * sum(sizes) / self.SIZE_BYTES))
        except Exception:
            return 0

    def _load(self) -> None:
        try:
            import torch
            from vggt.models.vggt import VGGT
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.state = {"state": "loading", "pct": 0,
                          "msg": "Downloading VGGT weights (~5 GB, first run only)..."}
            stop = threading.Event()

            def watch():
                while not stop.wait(1.0):
                    p = self._hf_pct()
                    self.state = {"state": "loading", "pct": p,
                                  "msg": f"Downloading VGGT weights (~5 GB, first run only): {p}%"}
            threading.Thread(target=watch, daemon=True).start()
            try:
                model = VGGT.from_pretrained(self.REPO)
            finally:
                stop.set()
            self.state = {"state": "loading", "pct": 99, "msg": f"Moving VGGT to {self.device}..."}
            self.model = model.to(self.device).eval()
            self.state = {"state": "ready", "pct": 100, "msg": f"VGGT ready on {self.device}"}
            log.info("VGGT ready on %s", self.device)
        except Exception as e:
            log.error("VGGT load failed: %s\n%s", e, traceback.format_exc())
            self.state = {"state": "error", "pct": 0, "msg": f"VGGT failed to load: {e}"}


class VGGTEngine:
    name = "vggt"
    MAX_VIEWS = 16        # views per forward pass (VRAM / latency); newest always included
    CONF_KEEP = 0.5

    def __init__(self, work_dir: Path, host: ModelHost):
        self.work_dir = Path(work_dir)
        self.host = host
        self.paths: List[Path] = []
        self._snap = None
        self._n_reg = 0
        self.lost = False
        self.waiting_baseline = False

    @property
    def model_state(self):
        return self.host.state

    @property
    def n_registered(self) -> int:
        return self._n_reg

    def add_paths(self, paths: List[Path]) -> None:
        self.paths.extend(paths)
        if len(self.paths) < 2:
            self.waiting_baseline = True
            return
        self.waiting_baseline = False
        self._run()

    def _select(self) -> List[Path]:
        p = self.paths
        if len(p) <= self.MAX_VIEWS:
            return list(p)
        sel = set(np.linspace(0, len(p) - 1, self.MAX_VIEWS - 2).round().astype(int).tolist())
        sel |= {len(p) - 1, len(p) - 2}
        return [p[i] for i in sorted(sel)]

    def _run(self) -> None:
        import torch
        from vggt.utils.geometry import unproject_depth_map_to_point_map
        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        model, device = self.host.model, self.host.device
        paths = self._select()
        images = load_and_preprocess_images([str(p) for p in paths], mode="crop")
        use_cuda = device.startswith("cuda")
        dtype = (torch.bfloat16 if use_cuda and torch.cuda.get_device_capability()[0] >= 8
                 else torch.float16 if use_cuda else torch.bfloat16)
        with torch.no_grad():
            with torch.autocast(device_type="cuda" if use_cuda else "cpu", dtype=dtype):
                preds = model(images.to(device))
            extri, intri = pose_encoding_to_extri_intri(preds["pose_enc"].float(), images.shape[-2:])
            extri = extri[0].float().cpu().numpy(); intri = intri[0].float().cpu().numpy()
            depth = preds["depth"][0].float().cpu().numpy(); conf = preds["depth_conf"][0].float().cpu().numpy()
        world = unproject_depth_map_to_point_map(depth, extri, intri)
        rgb = images.float().permute(0, 2, 3, 1).cpu().numpy()
        thr = np.quantile(conf, 1.0 - self.CONF_KEEP)
        mask = (conf >= thr) & (depth[..., 0] > 0) & np.isfinite(world).all(-1)
        H, W = images.shape[-2:]
        cams = [{"R": extri[i, :, :3], "t": extri[i, :, 3], "K": intri[i], "wh": (int(W), int(H))}
                for i in range(len(extri))]
        self._snap = (world[mask], (np.clip(rgb[mask], 0, 1) * 255).astype(np.uint8), cams)
        self._n_reg = len(self.paths)

    def snapshot(self):
        return self._snap


# ────────────────────────────── LiveScan worker ─────────────────────────────

class LiveScan:
    """One live capture (one pose of the robot).  Thread-safe `submit`; one worker thread."""

    def __init__(self, scan_dir: Path, engine, on_update: Callable[[dict], None], label: str = "scan"):
        self.dir = Path(scan_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "frames").mkdir(exist_ok=True)
        self.engine = engine
        self.on_update = on_update
        self.label = label
        self.q: "queue.Queue[Optional[Path]]" = queue.Queue()
        self.n_received = 0
        self.n_processed = 0
        self.error: Optional[str] = None
        self.version = 0
        self.state: dict = {}
        self._cams: list = []
        self._lock = threading.Lock()
        self._stop = False
        self._t0 = time.time()
        self._publish()
        self._thread = threading.Thread(target=self._loop, name=f"live-{label}", daemon=True)
        self._thread.start()

    # ---- API ---------------------------------------------------------------
    def submit(self, data: bytes) -> int:
        with self._lock:
            i = self.n_received
            self.n_received += 1
        p = self.dir / "frames" / f"frame_{i:04d}.jpg"
        p.write_bytes(data)
        self.q.put(p)
        self._publish()                       # immediate feedback: counter + queue length
        return i

    def close(self):
        self._stop = True
        self.q.put(None)

    def preview_bytes(self, background: bool = False) -> bytes:
        f = self.dir / ("preview_bg.bin" if background else "preview.bin")
        return f.read_bytes() if f.exists() else b""

    # ---- worker ------------------------------------------------------------
    def _loop(self):
        while not self._stop:
            first = self.q.get()
            if first is None:
                return
            batch = [first]
            while True:                       # coalesce a backlog into one engine call
                try:
                    nxt = self.q.get_nowait()
                    if nxt is None:
                        self._stop = True
                        break
                    batch.append(nxt)
                except queue.Empty:
                    break
            try:
                # VGGT: wait (without losing frames) until the weights are loaded
                while getattr(self.engine, "model_state", {}).get("state") in ("idle", "loading"):
                    self._publish()
                    time.sleep(0.5)
                    if self._stop:
                        return
                ms = getattr(self.engine, "model_state", {})
                if ms.get("state") == "error":
                    raise RuntimeError(ms.get("msg", "model failed to load"))
                t = time.time()
                self.engine.add_paths(batch)
                self.n_processed += len(batch)
                self.error = None
                self._export_preview()
                self._ms = int(1000 * (time.time() - t))
                log.info("%s: +%d photo(s) -> %d registered, %d object points, %d background points dropped (%d ms)",
                         self.label, len(batch), self.engine.n_registered,
                         getattr(self, "_n_points", 0), getattr(self, "_n_bg", 0), self._ms)
            except Exception as e:                           # never swallow: surface to the UI
                self.error = f"{type(e).__name__}: {e}"
                log.error("live reconstruction failed: %s\n%s", e, traceback.format_exc())
                self.n_processed += len(batch)
            self._publish()

    def _export_preview(self):
        snap = self.engine.snapshot()
        if snap is None:
            return
        pts, rgb, cams = snap
        if len(pts) == 0 or len(cams) == 0:
            return
        from core.foreground import select_object, _centres_axes
        if len(pts) > 400_000:                           # dense engines: keep the scoring cheap
            sel = np.random.default_rng(0).choice(len(pts), 400_000, replace=False)
            pts, rgb = pts[sel], rgb[sel]
        fg, info = select_object(pts, cams, return_info=True)
        self.fg_info = info
        C, _, Y = _centres_axes(cams)
        R, centre, scale = display_frame(pts[fg], C, Y)  # frame fitted to the ROBOT, not the room
        to_d = lambda X: ((X - centre) @ R.T) * scale
        P, Pc = to_d(pts[fg]), rgb[fg]
        Pb, Pbc = to_d(pts[~fg]), rgb[~fg]
        Cd = to_d(C)
        rng = np.random.default_rng(0)
        if len(P) > MAX_PREVIEW_POINTS:
            sel = rng.choice(len(P), MAX_PREVIEW_POINTS, replace=False); P, Pc = P[sel], Pc[sel]
        keepb = np.abs(Pb).max(1) < 4.0 if len(Pb) else np.zeros(0, bool)
        Pb, Pbc = Pb[keepb], Pbc[keepb]
        if len(Pb) > 20_000:
            sel = rng.choice(len(Pb), 20_000, replace=False); Pb, Pbc = Pb[sel], Pbc[sel]
        cov = orbit_coverage(Cd)
        seen = getattr(self, "_seen", np.zeros(N_SECTORS, bool))
        seen = seen | (cov["hits"] > 0)           # a covered direction never becomes uncovered again
        self._seen = seen
        cov["hits"] = np.maximum(cov["hits"], seen.astype(int)); cov["n_covered"] = int(seen.sum())
        cov["high_view"] = cov["high_view"] or getattr(self, "_high", False); self._high = cov["high_view"]
        self._cov = cov
        self._cams = Cd
        self._n_points = len(P)
        self._n_bg = int((~fg).sum())

        def pack(X, c):
            return np.uint32(len(X)).tobytes() + X.astype(np.float32).tobytes() + c.astype(np.uint8).tobytes()
        for name, buf in (("preview", pack(P, Pc)), ("preview_bg", pack(Pb, Pbc))):
            tmp = self.dir / f"{name}.tmp"
            tmp.write_bytes(buf); tmp.replace(self.dir / f"{name}.bin")
        self.version += 1
        (self.dir / "display_frame.json").write_text(json.dumps(
            {"R": R.tolist(), "centre": centre.tolist(), "scale": scale, "foreground": info}, default=float))

    def _publish(self):
        eng = self.engine
        n_reg = eng.n_registered
        cov = getattr(self, "_cov", None)
        n_cov = cov["n_covered"] if cov else 0
        lost = bool(eng.lost)
        guide = None
        ms = getattr(eng, "model_state", {})
        if ms.get("state") in ("idle", "loading"):
            guide = ms.get("msg") or "Loading reconstruction model..."
        elif self.error:
            guide = "Reconstruction error: " + self.error
        elif cov is None:
            guide = guidance_text({"n_covered": 0, "hits": np.zeros(N_SECTORS), "sectors": [0],
                                   "high_view": False}, n_reg, self.n_received, lost,
                                  eng.waiting_baseline or n_reg == 0)
            if self.n_received == 0:
                guide = "Point the camera at the robot and tap Capture"
        else:
            guide = guidance_text(cov, n_reg, self.n_received, lost, False)
        pct = max(getattr(self, '_pct', 0), progress_pct(n_cov, n_reg)); self._pct = pct
        ready = bool(cov) and n_cov >= READY_COVERAGE * N_SECTORS and n_reg >= READY_VIEWS
        self.state = {
            "type": "frame_update",
            "label": self.label,
            "engine": eng.name,
            "n_frames": self.n_received,
            "n_processed": self.n_processed,
            "n_registered": n_reg,
            "queue": self.q.qsize(),
            "n_points": getattr(self, "_n_points", 0),
            "n_background_removed": getattr(self, "_n_bg", 0),
            "ms_per_batch": getattr(self, "_ms", 0),
            "coverage": n_cov / N_SECTORS,
            "covered_sectors": [int(i) for i in np.flatnonzero(cov["hits"])] if cov else [],
            "high_view": bool(cov["high_view"]) if cov else False,
            "progress": pct,
            "guidance": guide,
            "ready": ready,
            "lost": lost,
            "error": self.error,
            "model": ms,
            "preview_version": self.version,
            "cameras": (np.round(self._cams, 4).tolist() if len(getattr(self, "_cams", [])) else []),
        }
        try:
            self.on_update(self.state)
        except Exception:
            log.exception("on_update failed")
