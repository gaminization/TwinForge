"""
photo_recon.py — Photos -> metric, upright, coloured 3D model (no GPU / no model weights).

    COLMAP SfM (pycolmap)  ->  camera poses + sparse points
    sparse points          ->  seed GrabCut object masks in every photo
    masks + poses          ->  visual-hull voxel carving  ->  marching cubes mesh
    camera "up" vectors    ->  gravity-aligned (z up), base at z=0
    --size-m               ->  metric scale

This is the dependable classical path. VGGT / DUSt3R (core/reconstruction.py) give denser,
more detailed geometry when their weights are available. A visual hull cannot represent
concavities (e.g. the gap between gripper fingers seen only edge-on), so fine detail is
smoothed; overall shape and link proportions are what it recovers.

Needs: pycolmap opencv-python scikit-image scipy open3d numpy
"""
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np
import open3d as o3d

log = logging.getLogger("twinforge.photo")

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class PhotoRecon:
    pcd:     o3d.geometry.PointCloud          # surface samples (colour + normals), metric, z-up
    mesh:    o3d.geometry.TriangleMesh
    report:  Dict = field(default_factory=dict)
    cam_centres: Optional[np.ndarray] = None  # (N,3) in the output frame
    sparse:  Optional[np.ndarray] = None      # (M,3) sparse SfM points in the output frame
    masks:   Optional[List[np.ndarray]] = None
    to_world: Optional[np.ndarray] = None     # 4x4: SfM frame -> output frame


def available() -> bool:
    try:
        import pycolmap  # noqa: F401
        import skimage   # noqa: F401
        return True
    except Exception:
        return False


# ───────────────────────────── SfM ─────────────────────────────

def _run_sfm(image_dir: Path, work: Path):
    import pycolmap
    db = work / "colmap.db"
    sparse_dir = work / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    if db.exists():
        db.unlink()
    ro = pycolmap.ImageReaderOptions()
    ro.camera_model = "SIMPLE_RADIAL"
    pycolmap.extract_features(db, image_dir, camera_mode=pycolmap.CameraMode.SINGLE,
                              reader_options=ro, device=pycolmap.Device.cpu)
    pycolmap.match_exhaustive(db, device=pycolmap.Device.cpu)
    maps = pycolmap.incremental_mapping(db, image_dir, sparse_dir)
    if not maps:
        raise RuntimeError("SfM failed: no model could be reconstructed. Use well-lit, textured, "
                           "overlapping photos (20-60) taken all around the object.")
    return max(maps.values(), key=lambda r: r.num_reg_images())


def _project(cam, R, t, X: np.ndarray, scale: float):
    """Project world points X (N,3) with a SIMPLE_RADIAL / PINHOLE camera. Returns (uv, z)."""
    Xc = X @ R.T + t
    z = Xc[:, 2]
    zz = np.where(np.abs(z) < 1e-9, 1e-9, z)
    x, y = Xc[:, 0] / zz, Xc[:, 1] / zz
    p = cam.params
    name = str(cam.model).split(".")[-1]
    if name in ("SIMPLE_RADIAL", "RADIAL"):
        f, cx, cy, k = p[0], p[1], p[2], p[3]
        d = 1.0 + k * (x * x + y * y)
        u, v = f * x * d + cx, f * y * d + cy
    elif name == "SIMPLE_PINHOLE":
        u, v = p[0] * x + p[1], p[0] * y + p[2]
    elif name == "PINHOLE":
        u, v = p[0] * x + p[2], p[1] * y + p[3]
    else:
        raise ValueError(f"unsupported camera model {name}")
    return np.stack([u, v], 1) * scale, z


# ───────────────────────────── masks ───────────────────────────

def _grabcut_mask(img_bgr: np.ndarray, pts_uv: np.ndarray) -> np.ndarray:
    """Foreground mask seeded by projected sparse points (which all lie on the object)."""
    h, w = img_bgr.shape[:2]
    m = np.full((h, w), cv2.GC_BGD, np.uint8)
    if len(pts_uv) < 8:
        return np.zeros((h, w), np.uint8)
    hull = cv2.convexHull(pts_uv.astype(np.float32)).astype(np.int32)
    prior = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(prior, hull, 255)
    margin = max(5, int(0.05 * max(h, w)))
    wide = cv2.dilate(prior, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1,) * 2))
    m[wide > 0] = cv2.GC_PR_BGD
    m[prior > 0] = cv2.GC_PR_FGD
    for u, v in pts_uv:
        if 0 <= int(u) < w and 0 <= int(v) < h:
            cv2.circle(m, (int(u), int(v)), 2, int(cv2.GC_FGD), -1)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(img_bgr, m, None, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return (prior > 0).astype(np.uint8)
    fg = ((m == cv2.GC_FGD) | (m == cv2.GC_PR_FGD)).astype(np.uint8)
    # drop speckles, keep the biggest blob(s) close to the object
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if n > 1:
        keep = np.zeros_like(fg)
        big = stats[1:, cv2.CC_STAT_AREA].max()
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= 0.05 * big:
                keep[lab == i] = 1
        fg = keep
    return fg



# ─────────────────── dense stereo (hull-bounded plane sweep) ───────────────────

def _undistort(img_bgr, cam):
    p = cam.params
    name = str(cam.model).split(".")[-1]
    if name in ("SIMPLE_RADIAL", "RADIAL"):
        K = np.array([[p[0], 0, p[1]], [0, p[0], p[2]], [0, 0, 1]], np.float64)
        return cv2.undistort(img_bgr, K, np.array([p[3], 0, 0, 0], np.float64)), K
    if name == "SIMPLE_PINHOLE":
        return img_bgr, np.array([[p[0], 0, p[1]], [0, p[0], p[2]], [0, 0, 1]], np.float64)
    if name == "PINHOLE":
        return img_bgr, np.array([[p[0], 0, p[2]], [0, p[1], p[3]], [0, 0, 1]], np.float64)
    raise ValueError(f"unsupported camera model {name}")


def _hull_depth_bounds(scene, K, R, t, w, h):
    """Per-pixel near/far depth (z) of the visual hull; inf where the ray misses it."""
    ext = np.eye(4)
    ext[:3, :3], ext[:3, 3] = R, t
    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        o3d.core.Tensor(K, o3d.core.Dtype.Float64), o3d.core.Tensor(ext, o3d.core.Dtype.Float64), w, h)
    hit = scene.cast_rays(rays)["t_hit"].numpy()
    r = rays.numpy()
    org, dr = r[..., :3], r[..., 3:]
    dr = dr / (np.linalg.norm(dr, axis=-1, keepdims=True) + 1e-12)
    zc = dr @ R[2]                       # ray direction . camera z axis (world)
    near = hit * zc
    org2 = org + (np.where(np.isfinite(hit), hit, 0)[..., None] + 1e-4) * dr
    r2 = np.concatenate([org2, dr], -1).astype(np.float32)
    hit2 = scene.cast_rays(o3d.core.Tensor(r2))["t_hit"].numpy()
    far = np.where(np.isfinite(hit2), (hit + 1e-4 + hit2) * zc, np.inf)
    return near.astype(np.float32), far.astype(np.float32)


def _plane_sweep(ref, nbrs, near, far, span_max, K_levels=40, win=7, top=2):
    """
    ref/nbrs: dicts with torch 'gray' (1,1,H,W), 'K' (3,3), 'R' (3,3), 't' (3,).
    Search each pixel's depth in [near, min(far, near+span_max)]. Returns depth (H,W), ncc (H,W).
    """
    import torch
    import torch.nn.functional as F
    H, W = near.shape
    valid = np.isfinite(near)
    n = torch.from_numpy(np.where(valid, near, 1.0)).float()
    span = torch.from_numpy(np.where(valid, np.minimum(np.where(np.isfinite(far), far, near + span_max) - near,
                                                       span_max), 0.0)).clamp(min=1e-6).float()
    levels = torch.linspace(0, 1, K_levels).view(K_levels, 1, 1)
    D = n[None] + levels * span[None]                                   # (K,H,W)

    ys, xs = torch.meshgrid(torch.arange(H).float(), torch.arange(W).float(), indexing="ij")
    Kr = torch.from_numpy(ref["K"]).float()
    ray = torch.stack([(xs - Kr[0, 2]) / Kr[0, 0], (ys - Kr[1, 2]) / Kr[1, 1], torch.ones_like(xs)], 0)  # (3,H,W)
    Xc = D[:, None] * ray[None]                                          # (K,3,H,W) in ref camera
    Rr, tr = torch.from_numpy(ref["R"]).float(), torch.from_numpy(ref["t"]).float()
    Xw = torch.einsum("ji,kjhw->kihw", Rr, Xc - tr.view(1, 3, 1, 1))     # R^T (Xc - t)

    gr = ref["gray"]
    pad = win // 2
    mr = F.avg_pool2d(gr, win, 1, pad)
    vr = (F.avg_pool2d(gr * gr, win, 1, pad) - mr * mr).clamp(min=1e-6)
    ncc_all = []
    for nb in nbrs:
        Rn, tn = torch.from_numpy(nb["R"]).float(), torch.from_numpy(nb["t"]).float()
        Xn = torch.einsum("ij,kjhw->kihw", Rn, Xw) + tn.view(1, 3, 1, 1)
        Kn = torch.from_numpy(nb["K"]).float()
        z = Xn[:, 2].clamp(min=1e-6)
        u = Kn[0, 0] * Xn[:, 0] / z + Kn[0, 2]
        v = Kn[1, 1] * Xn[:, 1] / z + Kn[1, 2]
        grid = torch.stack([2 * u / (W - 1) - 1, 2 * v / (H - 1) - 1], -1)      # (K,H,W,2)
        wimg = F.grid_sample(nb["gray"].expand(K_levels, -1, -1, -1), grid, align_corners=True,
                             padding_mode="zeros")
        mw = F.avg_pool2d(wimg, win, 1, pad)
        vw = (F.avg_pool2d(wimg * wimg, win, 1, pad) - mw * mw).clamp(min=1e-6)
        cv = F.avg_pool2d(wimg * gr, win, 1, pad) - mw * mr
        ncc = cv / torch.sqrt(vw * vr)
        inside = ((u > 0) & (u < W - 1) & (v > 0) & (v < H - 1) & (Xn[:, 2] > 0)).float()
        ncc_all.append((ncc[:, 0] * inside) + (inside - 1.0))                   # outside -> -1
    ncc_all = torch.stack(ncc_all, 0)                                           # (M,K,H,W)
    kk = min(top, ncc_all.shape[0])
    score = ncc_all.topk(kk, dim=0).values.mean(0)                              # (K,H,W)
    best, idx = score.max(0)
    idx = idx.clamp(1, K_levels - 2)
    # parabolic sub-level refinement
    a = score.gather(0, (idx - 1)[None])[0]
    b = score.gather(0, idx[None])[0]
    c = score.gather(0, (idx + 1)[None])[0]
    den = (a - 2 * b + c)
    off = torch.where(den.abs() > 1e-6, 0.5 * (a - c) / den, torch.zeros_like(den)).clamp(-0.5, 0.5)
    lev = (idx.float() + off) / (K_levels - 1)
    depth = n + lev * span
    return depth.numpy(), best.numpy()


def _dense_mesh(cams, ims, hull_mesh, masks_full, scale=0.5, K_levels=40, n_nbrs=6,
                min_ncc=0.5, rel_eps=0.012, min_consistent=2, tsdf_res=320, progress=None):
    """
    Hull-bounded multi-view stereo -> multi-view consistency filter -> TSDF fusion -> mesh.
    Works in the SfM frame (arbitrary units).
    NOTE: uses UniformTSDFVolume - ScalableTSDFVolume returned empty meshes in Open3D 0.20.
    """
    import torch
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(hull_mesh))
    bb = hull_mesh.get_axis_aligned_bounding_box()
    diag = float(np.linalg.norm(bb.get_extent()))
    span_max = 0.14 * diag

    views = []
    for c, img in zip(cams, ims):
        und, K0 = _undistort(img, c["cam"])
        small = cv2.resize(und, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        Ks = K0.copy()
        Ks[:2] *= scale
        h, w = small.shape[:2]
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        near, far = _hull_depth_bounds(scene, Ks, c["R"], c["t"], w, h)
        views.append(dict(K=Ks, R=c["R"], t=c["t"], bgr=small, w=w, h=h, near=near, far=far,
                          gray=torch.from_numpy(gray)[None, None], centre=-c["R"].T @ c["t"]))
    centres = np.array([v["centre"] for v in views])

    # 1) per-view depth by plane sweep, neighbours chosen by proximity
    depth, ncc, ok = [], [], []
    for i, v in enumerate(views):
        order = np.argsort(np.linalg.norm(centres - v["centre"], axis=1))
        nb = [j for j in order if j != i][:n_nbrs]
        d, q = _plane_sweep(v, [views[j] for j in nb], v["near"], v["far"], span_max, K_levels=K_levels)
        m = cv2.resize(masks_full[i], (v["w"], v["h"]), interpolation=cv2.INTER_NEAREST) > 0
        depth.append(d)
        ncc.append(q)
        ok.append(m & np.isfinite(v["near"]) & (q > min_ncc))
        if progress:
            progress(0.7 * (i + 1) / len(views), f"Dense stereo view {i + 1}/{len(views)}")

    # 2) keep depths that agree with >= min_consistent neighbouring views
    def backproject(i, dep):
        v = views[i]
        h, w = dep.shape
        ys, xs = np.mgrid[0:h, 0:w]
        K = v["K"]
        X = np.stack([(xs - K[0, 2]) / K[0, 0] * dep, (ys - K[1, 2]) / K[1, 1] * dep, dep], -1)
        return (X - v["t"]) @ v["R"]

    vox_len = max(bb.get_extent()) * 1.15 / tsdf_res
    L = float(bb.get_extent().max()) * 1.15
    tsdf = o3d.pipelines.integration.UniformTSDFVolume(
        length=L, resolution=tsdf_res, sdf_trunc=5 * vox_len,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        origin=bb.get_center() - L / 2)
    kept = 0
    for i, v in enumerate(views):
        dep = np.where(ok[i], depth[i], 0).astype(np.float32)
        if min_consistent > 0:
            Xw = backproject(i, dep)
            cons = np.zeros(dep.shape, int)
            order = np.argsort(np.linalg.norm(centres - v["centre"], axis=1))
            for j in [j for j in order if j != i][:8]:
                vj = views[j]
                Xc = Xw @ vj["R"].T + vj["t"]
                z = Xc[..., 2]
                zz = np.maximum(z, 1e-9)
                u = np.round(vj["K"][0, 0] * Xc[..., 0] / zz + vj["K"][0, 2]).astype(int)
                vv = np.round(vj["K"][1, 1] * Xc[..., 1] / zz + vj["K"][1, 2]).astype(int)
                inb = (u >= 0) & (u < vj["w"]) & (vv >= 0) & (vv < vj["h"]) & (z > 0)
                uc, vc = u.clip(0, vj["w"] - 1), vv.clip(0, vj["h"] - 1)
                agree = inb & ok[j][vc, uc] & (np.abs(depth[j][vc, uc] - z) < rel_eps * z)
                cons += agree.astype(int)
            dep = np.where(cons >= min_consistent, dep, 0).astype(np.float32)
        kept += int((dep > 0).sum())
        K = v["K"]
        intr = o3d.camera.PinholeCameraIntrinsic(v["w"], v["h"], K[0, 0], K[1, 1], K[0, 2], K[1, 2])
        ext = np.eye(4)
        ext[:3, :3], ext[:3, 3] = v["R"], v["t"]
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(v["bgr"][:, :, ::-1])),
            o3d.geometry.Image(np.ascontiguousarray(dep)),
            depth_scale=1.0, depth_trunc=1e9, convert_rgb_to_intensity=False)
        tsdf.integrate(rgbd, intr, ext)
        if progress:
            progress(0.7 + 0.3 * (i + 1) / len(views), f"Fusing view {i + 1}/{len(views)}")
    mesh = tsdf.extract_triangle_mesh()
    return mesh, dict(dense_valid_px=kept, tsdf_voxel=float(vox_len))


def _gravity_scale_transform(cams, V: np.ndarray, size_m: Optional[float]):
    """SfM frame -> upright (z up), base at z=0 / footprint centred, metric if size_m given."""
    # images are taken upright, so the mean camera "up" is world up
    ups = np.array([-(c["R"].T @ np.array([0.0, 1.0, 0.0])) for c in cams])
    up = ups.mean(0)
    up /= np.linalg.norm(up)
    zax = up
    xax = np.cross([0.0, 1.0, 0.0], zax)
    if np.linalg.norm(xax) < 1e-6:
        xax = np.cross([1.0, 0.0, 0.0], zax)
    xax /= np.linalg.norm(xax)
    yax = np.cross(zax, xax)
    Rg = np.stack([xax, yax, zax])               # rows: new axes expressed in the SfM frame
    Vg = V @ Rg.T
    zmin = Vg[:, 2].min()
    foot = Vg[Vg[:, 2] < zmin + 0.03 * np.ptp(Vg[:, 2])]
    shift = np.array([foot[:, 0].mean(), foot[:, 1].mean(), zmin])
    longest = float(np.ptp(Vg, axis=0).max())
    k = (size_m / longest) if size_m else 1.0
    T = np.eye(4)
    T[:3, :3] = k * Rg
    T[:3, 3] = -k * shift
    return T, k


# ─────────────────────────── main entry ────────────────────────

def reconstruct_from_photos(image_dir, work_dir, size_m: Optional[float] = None,
                            grid: int = 240, mask_scale: float = 0.5,
                            miss_tolerance: int = 2, dense: bool = True,
                            progress: Optional[Callable[[float, str], None]] = None) -> PhotoRecon:
    from scipy import ndimage
    from skimage import measure

    def prog(p, msg):
        log.info(msg)
        if progress:
            progress(p, msg)

    image_dir, work = Path(image_dir), Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    rep: Dict = {}
    t0 = time.time()

    prog(0.02, "Structure-from-Motion (COLMAP) …")
    rec = _run_sfm(image_dir, work)
    n_total = len([p for p in image_dir.iterdir() if p.suffix.lower() in IMG_EXT])
    rep.update(n_images=n_total, n_registered=rec.num_reg_images(), n_sparse=rec.num_points3D(),
               reproj_px=float(rec.compute_mean_reprojection_error()),
               t_sfm=round(time.time() - t0, 1))
    if rec.num_reg_images() < max(4, int(0.5 * n_total)):
        raise RuntimeError(f"SfM only registered {rec.num_reg_images()}/{n_total} photos - not enough "
                           "overlap. Take more photos with ~70% overlap between neighbours.")

    sparse_world = np.array([p.xyz for p in rec.points3D.values()])
    # drop SfM outliers (far from the dense core)
    med = np.median(sparse_world, 0)
    dist = np.linalg.norm(sparse_world - med, axis=1)
    core = sparse_world[dist < np.percentile(dist, 97)]

    # per-image camera data
    cams = []
    for im in rec.images.values():
        if not im.has_pose:
            continue
        cam = rec.cameras[im.camera_id]
        M = im.cam_from_world().matrix()
        cams.append(dict(name=im.name, cam=cam, R=M[:, :3], t=M[:, 3]))

    prog(0.30, f"Segmenting the object in {len(cams)} photos …")
    masks, ims = [], []
    for c in cams:
        img = cv2.imread(str(image_dir / c["name"]))
        s = mask_scale
        small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        uv, z = _project(c["cam"], c["R"], c["t"], core, s)
        ok = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < small.shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < small.shape[0])
        masks.append(_grabcut_mask(small, uv[ok]))
        ims.append(img)
        c["mask"] = masks[-1]
        c["scale"] = s
    rep["mask_fraction"] = float(np.mean([m.mean() for m in masks]))

    prog(0.55, "Carving visual hull …")
    lo, hi = np.percentile(core, 1, axis=0), np.percentile(core, 99, axis=0)
    ext = hi - lo
    lo, hi = lo - 0.35 * ext, hi + 0.35 * ext
    step = float((hi - lo).max() / grid)
    dims = np.ceil((hi - lo) / step).astype(int) + 1
    gx, gy, gz = [lo[i] + step * np.arange(dims[i]) for i in range(3)]
    G = np.stack(np.meshgrid(gx, gy, gz, indexing="ij"), -1).reshape(-1, 3)
    misses = np.zeros(len(G), np.int16)
    seen = np.zeros(len(G), np.int16)
    for c in cams:
        uv, z = _project(c["cam"], c["R"], c["t"], G, c["scale"])
        m = c["mask"]
        h, w = m.shape
        u = np.round(uv[:, 0]).astype(int)
        v = np.round(uv[:, 1]).astype(int)
        inside = (z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        fg = np.zeros(len(G), bool)
        fg[inside] = m[v[inside], u[inside]] > 0
        seen += inside
        misses += (inside & ~fg)
    occ = ((misses <= miss_tolerance) & (seen >= max(4, int(0.5 * len(cams))))).reshape(dims)
    lab, n = ndimage.label(occ)
    if n == 0:
        raise RuntimeError("Visual hull is empty - object masks failed. Is the object centred and "
                           "against a reasonably plain background?")
    sizes = ndimage.sum(occ, lab, range(1, n + 1))
    occ = lab == (1 + int(np.argmax(sizes)))
    rep["hull_voxels"] = int(occ.sum())

    prog(0.75, "Meshing …")
    vol = ndimage.gaussian_filter(occ.astype(np.float32), 1.0)
    verts, faces, _, _ = measure.marching_cubes(vol, 0.5)
    verts = lo + verts * step
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(faces))
    mesh = mesh.filter_smooth_taubin(number_of_iterations=10)
    mesh.remove_degenerate_triangles()
    mesh.compute_vertex_normals()
    rep["mode"] = "visual_hull"

    if dense:
        try:
            prog(0.80, "Dense multi-view stereo + TSDF fusion …")
            td = time.time()
            dmesh, drep = _dense_mesh(cams, ims, mesh, masks, scale=0.5,
                                      progress=lambda f, m: prog(0.80 + 0.12 * f, m))
            tri_h = len(mesh.triangles)
            dmesh.remove_degenerate_triangles()
            dmesh.remove_unreferenced_vertices()
            cl, nt, _ = dmesh.cluster_connected_triangles()
            cl, nt = np.asarray(cl), np.asarray(nt)
            if len(nt):
                dmesh.remove_triangles_by_mask(cl != int(np.argmax(nt)))
                dmesh.remove_unreferenced_vertices()
            dmesh.compute_vertex_normals()
            rep.update(drep, t_dense=round(time.time() - td, 1))
            if len(dmesh.triangles) > 0.2 * tri_h:
                mesh = dmesh
                rep["mode"] = "dense_mvs_tsdf"
            else:
                log.warning("Dense stereo produced too little geometry; keeping the visual hull")
        except Exception as e:                                  # never lose the hull result
            log.warning(f"Dense stereo failed ({e}); keeping the visual hull")
            rep["dense_error"] = str(e)

    # colours: from the camera that sees each vertex most head-on
    V, N = np.asarray(mesh.vertices), np.asarray(mesh.vertex_normals)
    has_col = rep['mode'] == 'dense_mvs_tsdf' and mesh.has_vertex_colors()
    best = np.full(len(V), -2.0)
    col = np.full((len(V), 3), 0.6)
    for c, img in ([] if has_col else zip(cams, ims)):
        centre = -c["R"].T @ c["t"]
        d = centre - V
        d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-12
        facing = np.einsum("ij,ij->i", N, d)
        uv, z = _project(c["cam"], c["R"], c["t"], V, 1.0)
        h, w = img.shape[:2]
        u = np.clip(np.round(uv[:, 0]).astype(int), 0, w - 1)
        v = np.clip(np.round(uv[:, 1]).astype(int), 0, h - 1)
        sel = (facing > best) & (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        best[sel] = facing[sel]
        col[sel] = img[v[sel], u[sel]][:, ::-1] / 255.0
    if not has_col:
        mesh.vertex_colors = o3d.utility.Vector3dVector(col)

    T, k = _gravity_scale_transform(cams, np.asarray(mesh.vertices), size_m)
    mesh.transform(T)
    mesh.compute_vertex_normals()
    rep["scale_applied"] = k
    rep["scale_is_metric"] = bool(size_m)
    rep["extent_m"] = np.ptp(np.asarray(mesh.vertices), axis=0).round(4).tolist()

    pcd = mesh.sample_points_uniformly(80000)
    pcd.normals = pcd.normals          # keep
    rep.update(mesh_tris=len(mesh.triangles), t_total=round(time.time() - t0, 1))
    prog(1.0, f"Done: {len(mesh.triangles):,} triangles")

    def tf(P):
        return (np.asarray(P) @ T[:3, :3].T) + T[:3, 3]

    return PhotoRecon(pcd=pcd, mesh=mesh, report=rep,
                      cam_centres=tf(np.array([-c["R"].T @ c["t"] for c in cams])),
                      sparse=tf(sparse_world), masks=masks, to_world=T)
