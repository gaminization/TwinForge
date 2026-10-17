"""
acquisition.py — Load scan data from a file or the demo arm.
"""
import logging
import numpy as np
import open3d as o3d
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

log = logging.getLogger("twinforge.acquisition")


@dataclass
class ScanData:
    pcd:             o3d.geometry.PointCloud
    source:          str = "demo"
    _demo_link_meta: Optional[List] = field(default=None, repr=False)
    mesh:            Optional[o3d.geometry.TriangleMesh] = None   # already-meshed scan (skips Poisson)


def load(source=None, demo: bool = False, n_poses: int = 3,
         backend: str = "auto", size_m=None) -> ScanData:
    if demo or source is None:
        return _demo_arm()
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Scan file not found: {path}")
    if path.is_dir():
        return _from_images(path, backend=backend, size_m=size_m)
    ext = path.suffix.lower()
    if ext == ".ply":
        m = o3d.io.read_triangle_mesh(str(path))
        if len(m.triangles) > 0:                      # a mesh, not a bare point cloud
            m.compute_vertex_normals()
            pcd = m.sample_points_uniformly(80000)
            log.info(f"Loaded mesh with {len(m.triangles):,} triangles from {path}")
            return ScanData(pcd=pcd, source=str(path), mesh=m)
        pcd = o3d.io.read_point_cloud(str(path))
    elif ext == ".pcd":
        pcd = o3d.io.read_point_cloud(str(path))
    else:
        raise ValueError(f"Unsupported format: {ext} (use .ply or .pcd)")
    log.info(f"Loaded {len(pcd.points):,} points from {path}")
    return ScanData(pcd=pcd, source=str(path))


def _demo_arm() -> ScanData:
    from core.demo_arm import build_demo_scan
    pcd, link_metas = build_demo_scan(pts_per_link=2000)
    log.info(
        f"Demo: arduino_robot_arm replica — {len(pcd.points):,} points, "
        f"{len(link_metas)} links"
    )
    return ScanData(pcd=pcd, source="demo", _demo_link_meta=link_metas)


def _from_images(folder: Path, backend: str = "auto", size_m=None, max_side: int = 1024) -> ScanData:
    """
    Reconstruct a point cloud from a folder of photos.
      backend = "neural" : VGGT > DUSt3R (needs torch + weights)        - best, arbitrary scale/frame
      backend = "sfm"    : COLMAP SfM + dense stereo (pycolmap)         - metric, upright, no weights
      backend = "sift"   : sparse OpenCV SfM                            - last resort, very sparse
      backend = "auto"   : neural if available, else sfm, else sift
    """
    import cv2, tempfile
    from core.reconstruction import IncrementalReconstructor, _detect_backend
    from core import photo_recon
    if backend == "auto":
        nb = _detect_backend()
        backend = "neural" if nb in ("vggt", "dust3r") else ("sfm" if photo_recon.available() else "sift")
    log.info(f"Photo backend: {backend}")
    if backend == "sfm":
        work = Path(tempfile.mkdtemp(prefix="twinforge_sfm_"))
        r = photo_recon.reconstruct_from_photos(folder, work, size_m=size_m)
        log.info(f"Photo reconstruction: {r.report}")
        sd = ScanData(pcd=r.pcd, source=str(folder), mesh=r.mesh)
        sd.report = r.report
        return sd
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = sorted(f for f in folder.iterdir() if f.suffix.lower() in exts)
    if len(files) < 2:
        raise ValueError(f"Need at least 2 images in {folder}, found {len(files)}")
    rec = IncrementalReconstructor(Path(tempfile.mkdtemp(prefix="twinforge_job_")))
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            log.warning(f"Skipping unreadable image {f.name}")
            continue
        h, w = img.shape[:2]
        sc = max_side / max(h, w)
        if sc < 1:
            img = cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        r = rec.add_frame(buf.tobytes())
        log.info(f"  {f.name}: backend={r.backend} points={r.n_points:,}")
    pcd = rec.get_pcd()
    log.info(f"Reconstructed {len(pcd.points):,} points from {len(files)} images")
    return ScanData(pcd=pcd, source=str(folder))
