"""
foreground.py - keep the ROBOT, drop the room.

SfM / VGGT reconstruct everything the camera sees (wall, monitor, desk, cables).  Nothing in the photos says
"this is the robot", so we use the one thing every object-centric capture guarantees: the user keeps the
robot in the middle of the frame while walking around it.  Four checks, all geometric (no learned model,
no colour assumptions):

 1. FOCUS POINT   least-squares intersection of all camera optical axes = what everybody was looking at.
 2. CENTRALITY    a robot point projects near the image centre in (almost) every view; background points
                  only do so in the few views that happen to look straight at them.
 3. RADIUS        keep points within ~0.7 x (median camera distance) of the focus point: walls, monitors
                  and furniture further away than the robot are cut even on a partial orbit.
 4. SUPPORT PLANE the table/floor is a thin, very dense horizontal slab in the height histogram: removed,
                  as is everything below it.
 Finally only the biggest connected blob around the focus point survives (no floating specks).

`cams` is a list of dicts: R (3x3, world->camera), t (3,), K (3x3), wh (width, height).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy import ndimage


def _centres_axes(cams: List[dict]):
    C = np.array([-c["R"].T @ c["t"] for c in cams])
    Z = np.array([c["R"][2] for c in cams])            # optical axis in world
    Yd = np.array([c["R"][1] for c in cams])           # image-down axis in world
    return C, Z, Yd


def focus_point(C: np.ndarray, Z: np.ndarray) -> np.ndarray:
    A = np.zeros((3, 3)); b = np.zeros(3)
    for c, z in zip(C, Z):
        P = np.eye(3) - np.outer(z, z)
        A += P; b += P @ c
    return np.linalg.lstsq(A, b, rcond=None)[0]


def up_vector(Yd: np.ndarray) -> np.ndarray:
    u = -Yd.mean(0)
    return u / (np.linalg.norm(u) + 1e-12)


def central_fraction(P: np.ndarray, cams: List[dict], box=(0.9, 1.1)) -> np.ndarray:
    """Fraction of cameras in which each point is in front AND inside the central box (normalised coords)."""
    n = np.zeros(len(P))
    for c in cams:
        Xc = P @ c["R"].T + c["t"]
        z = Xc[:, 2]
        w, h = c["wh"]
        K = c["K"]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = K[0, 0] * Xc[:, 0] / z + K[0, 2]
            v = K[1, 1] * Xc[:, 1] / z + K[1, 2]
        nx = (u - w / 2) / (w / 2); ny = (v - h / 2) / (h / 2)
        n += (z > 0) & (np.abs(nx) < box[0]) & (np.abs(ny) < box[1])
    return n / max(len(cams), 1)


def support_plane_height(h: np.ndarray, unit: float):
    """Height (along up) of the dominant thin horizontal slab in the lower part of the cloud, or None."""
    if len(h) < 200:
        return None
    lo, hi = np.percentile(h, [1, 99])
    nb = max(8, int((hi - lo) / unit))
    hist, edges = np.histogram(h, bins=nb, range=(lo, hi))
    sm = ndimage.uniform_filter1d(hist.astype(float), 3)
    pk = int(np.argmax(sm))
    # a table is a spike: much denser than a typical bin, and in the lower half of the object
    if sm[pk] < 4.0 * max(np.median(hist[hist > 0]), 1.0) or pk > 0.5 * nb:
        return None
    return float(edges[pk + 1] + unit)


def select_object(P: np.ndarray, cams: List[dict], min_central: float = 0.4,
                  radius_factor: float = 0.7, return_info: bool = False):
    """Boolean mask of points that belong to the central object.  Falls back to 'keep all' if it can't tell."""
    keep_all = np.ones(len(P), bool)
    info: Dict = {"method": "none"}
    if len(cams) < 3 or len(P) < 50:
        return (keep_all, info) if return_info else keep_all
    C, Z, Yd = _centres_axes(cams)
    F = focus_point(C, Z)
    up = up_vector(Yd)
    dist = float(np.median(np.linalg.norm(C - F, axis=1)))
    if not np.isfinite(dist) or dist < 1e-9:
        return (keep_all, info) if return_info else keep_all
    R = radius_factor * dist

    m = np.linalg.norm(P - F, axis=1) < R                       # 3. radius
    if m.sum() < 30:
        return (keep_all, info) if return_info else keep_all
    m &= central_fraction(P, cams) >= min_central               # 2. centrality
    if m.sum() < 30:
        return (keep_all, info) if return_info else keep_all

    h = (P - F) @ up                                            # 4. support plane
    h0 = support_plane_height(h[m], R / 60.0)
    if h0 is not None:
        m &= h > h0

    # biggest connected blob (voxelised) containing/near the densest part
    if m.sum() > 60:
        vox = R / 40.0
        idx = np.floor((P[m] - P[m].min(0)) / vox).astype(int)
        shape = idx.max(0) + 3
        occ = np.zeros(shape, bool); occ[tuple((idx + 1).T)] = True
        occ = ndimage.binary_dilation(occ, iterations=1)       # bridge 1-voxel gaps
        lab, n = ndimage.label(occ)
        if n > 1:
            cnt = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
            big = 1 + int(np.argmax(cnt))
            ok = lab[tuple((idx + 1).T)] == big
            sel = np.flatnonzero(m); m[sel[~ok]] = False

    info = {"method": "focus+central+radius+plane", "focus": F.tolist(), "radius": R,
            "plane": h0, "kept": int(m.sum()), "dropped": int(len(P) - m.sum())}
    if m.sum() < 30:                                            # over-aggressive: don't destroy the model
        return (keep_all, {"method": "fallback-keep-all"}) if return_info else keep_all
    return (m, info) if return_info else m
