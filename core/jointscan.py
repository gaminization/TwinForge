"""
Capture protocol (this is what makes the problem well posed):
    scan 0 : robot at rest ("zero pose")
    scan i : the same robot with ONLY joint i moved by >= ~20 degrees, everything else as at rest

Then, for each scan i:
  1. align it to the rest scan on the static base,
  2. everything that did not move matches the rest scan; what is left is exactly the set of
     points DISTAL to joint i (one rigid body), so a single rigid motion explains it,
  3. that motion's screw axis IS the joint axis (direction + a point on it + signed angle),
  4. the distal sets are nested (or disjoint), which gives the kinematic tree and the links
     (link i = distal set of joint i minus the distal sets of its children).

Everything operates on metric-or-arbitrary-scale, z-up (gravity aligned) point clouds; scale of
the extra scans is recovered from the base.  Pose 0 defines joint zero.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

log = logging.getLogger("twinforge.jointscan")
reg = o3d.pipelines.registration


# ───────────────────────────── helpers ─────────────────────────────
def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def _pcd(P, normals=None):
    p = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(P, float)))
    if normals is not None:
        p.normals = o3d.utility.Vector3dVector(normals)
    return p


def _prep(P, voxel):
    p = _pcd(P).voxel_down_sample(voxel)
    p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 3, max_nn=40))
    f = reg.compute_fpfh_feature(p, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 7, max_nn=100))
    return p, f


def _T(R=None, t=None, s=1.0):
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R * s
    if t is not None:
        T[:3, 3] = t
    return T


def _icp_refine(src, dst, T, dist, iters=40):
    r = reg.registration_icp(src, dst, dist, T, reg.TransformationEstimationPointToPlane(),
                             reg.ICPConvergenceCriteria(max_iteration=iters))
    return r.transformation


def _inliers(P, T, tree, tol, nP=None, nD=None):
    Q = P @ T[:3, :3].T + T[:3, 3]
    d, ix = tree.query(Q)
    ok = d < tol
    if nP is not None:
        ok &= np.abs(np.einsum("ij,ij->i", nP @ T[:3, :3].T, nD[ix])) > 0.85
    return ok


# ─────────────── step 1: align every pose to the reference ───────────────
def _base_center(X, frac=0.03):
    """xy centre of the largest blob on the floor (the base), independent of scale."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    z = X[:, 2]
    b = X[z < z.min() + frac * np.ptp(z), :2]
    if len(b) < 20:
        return X[z < z.min() + 0.1 * np.ptp(z), :2].mean(0)
    pr = cKDTree(b).query_pairs(0.03 * np.linalg.norm(np.ptp(X, axis=0)), output_type="ndarray")
    g = coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(len(b), len(b)))
    _, comp = connected_components(g, directed=False)
    b = b[comp == np.argmax(np.bincount(comp))]
    return 0.5 * (b.min(0) + b.max(0))


def align_to_reference(ref: np.ndarray, P: np.ndarray, tol: float, slab_frac: float = 0.12,
                       n_top: int = 10, return_all: bool = False):
    """Similarity transform (yaw about z, uniform scale, translation) mapping P onto ref so that
    everything that did NOT move overlaps.  Both clouds are gravity aligned (z up, floor z=0), the
    base centres are matched, which leaves only (scale, yaw): searched exhaustively with a
    bidirectional overlap score, refined by similarity-ICP on a static region that grows upward from
    the floor while it keeps matching."""
    rng = np.random.default_rng(0)
    zr0, hr = ref[:, 2].min(), np.ptp(ref[:, 2])
    c_ref, c_P = _base_center(ref), _base_center(P)
    z0 = P[:, 2].min()
    tree_ref = cKDTree(ref)

    def sub(X, n):
        return X[rng.choice(len(X), min(n, len(X)), replace=False)]

    def place(s, deg, X):
        return ((X - np.r_[c_P, z0]) * s) @ _rz(np.radians(deg)).T + np.r_[c_ref, zr0]

    def overlap(Q_sub, ref_sub, ref_tree_sub, thr):
        d1 = tree_ref.query(Q_sub)[0]
        d2 = cKDTree(Q_sub).query(ref_sub)[0]
        return min((d1 < thr).mean(), (d2 < thr).mean())

    def keys(X, vox):
        k = np.floor(X / vox).astype(np.int64) + 20000
        return np.unique((k[:, 0] * 40000 + k[:, 1]) * 40000 + k[:, 2])

    def jacc(Q, ka, vox):
        kb = keys(Q, vox)
        inter = np.isin(kb, ka, assume_unique=True).sum()
        return inter / (len(ka) + len(kb) - inter)

    P_s = sub(P, 30000)
    h_ratio = hr / max(np.ptp(P[:, 2]), 1e-9)
    vox_c = tol * 2.5
    ka_c = keys(ref, vox_c)
    scored = []
    for s in h_ratio * np.exp(np.linspace(np.log(0.3), np.log(3.4), 90)):
        for deg in range(0, 360, 5):
            scored.append((jacc(place(s, deg, P_s), ka_c, vox_c), s, deg))
    # best coarse candidate per scale bucket (the right scale is NOT necessarily the global
    # overlap maximum when most of the robot has moved), most promising buckets first
    by_scale = {}
    for sc, s_, deg in scored:
        b = int(round(np.log(s_) / 0.08))
        if b not in by_scale or sc > by_scale[b][0]:
            by_scale[b] = (sc, s_, deg)
    picked = sorted(by_scale.values(), reverse=True)[:n_top]
    # local pattern search (scale, yaw, tx, ty) around each coarse candidate
    ka_f = keys(ref, tol * 1.3)
    diag_r = np.linalg.norm(np.ptp(ref, axis=0))

    def score(p):
        Q = place(p[0], p[1], P_s) + np.r_[p[2], p[3], 0]
        return jacc(Q, ka_f, tol * 1.3)

    fine = []
    for sc, s0, d0 in picked:
        p = np.array([s0, d0, 0.0, 0.0]); cur = score(p)
        step = np.array([0.025 * s0, 3.0, 3 * tol, 3 * tol])
        for rnd in range(3):
            improved = True
            while improved:
                improved = False
                for k in range(4):
                    for sg in (1, -1):
                        q = p.copy(); q[k] += sg * step[k]
                        v = score(q)
                        if v > cur + 1e-9:
                            p, cur, improved = q, v, True
            step = step / 2
        fine.append((cur, p[0], p[1], p[2], p[3]))
    picked = sorted(fine, reverse=True)[:10]
    best = (-1, None, None)
    cands_all = []
    for sc0, s, deg, tx, ty in picked:
        R = _rz(np.radians(deg))
        T = np.eye(4)
        T[:3, :3] = R * s
        T[:3, 3] = np.r_[c_ref, zr0] - R @ np.r_[c_P * s, 0] - R @ np.array([0, 0, s * z0]) + np.r_[tx, ty, 0]
        T_coarse = T.copy()
        h = slab_frac * hr
        for it in range(26):
            Q = P @ T[:3, :3].T + T[:3, 3]
            qsel = np.where(Q[:, 2] < zr0 + h + 3 * tol)[0]        # only the matching height range
            if len(qsel) < 50:
                break
            qtree = cKDTree(Q[qsel])
            region = np.where(ref[:, 2] < zr0 + h)[0]
            d, idx = qtree.query(ref[region])
            idx = qsel[idx]
            thr = tol * (4.0 if it < 5 else (2.5 if it < 10 else 1.5))
            keep = d < thr
            if keep.sum() >= 200:
                corr = o3d.utility.Vector2iVector(np.c_[idx[keep], region[keep]])
                dT = reg.TransformationEstimationPointToPoint(with_scaling=True).compute_transformation(
                    _pcd(Q), _pcd(ref), corr)
                T = dT @ T
                sc_now = np.cbrt(abs(np.linalg.det(T[:3, :3])))
                if not (0.9 * s <= sc_now <= 1.1 * s):          # keep scale near the searched value
                    T[:3, :3] *= np.clip(sc_now, 0.9 * s, 1.1 * s) / sc_now
            if it >= 4 and h < hr:
                Q = P @ T[:3, :3].T + T[:3, 3]
                band = np.where((ref[:, 2] >= zr0 + h) & (ref[:, 2] < zr0 + h + 0.05 * hr))[0]
                if len(band) > 30 and (cKDTree(Q).query(ref[band])[0] < tol * 1.5).mean() > 0.75:
                    h += 0.05 * hr
        for Tc in (T, T_coarse):
            Q = P @ Tc[:3, :3].T + Tc[:3, 3]
            d1 = tree_ref.query(sub(Q, 6000))[0]
            d2 = cKDTree(Q).query(sub(ref, 6000))[0]
            static = float(min((d1 < tol).mean(), (d2 < tol).mean()))
            cands_all.append((static, Tc, (round(float(s), 3), round(float(deg), 1))))
            if static > best[0]:
                best = (static, Tc, (round(float(s), 3), round(float(deg), 1)))
    if return_all:
        out = []
        for st, Tc, info in sorted(cands_all, key=lambda x: -x[0]):
            sc_ = np.cbrt(abs(np.linalg.det(Tc[:3, :3])))
            if all(abs(np.log(sc_ / np.cbrt(abs(np.linalg.det(T2[:3, :3]))))) > 0.035 or np.linalg.norm(Tc[:3, :3] / sc_ - T2[:3, :3] / np.cbrt(abs(np.linalg.det(T2[:3, :3])))) > 0.25 for _, T2 in out):
                out.append((st, Tc))
        return out
    T = best[1]
    log.info("  align: scale %.3f (cand %s), static overlap %.0f%%", np.cbrt(abs(np.linalg.det(T[:3, :3]))),
             best[2], 100 * best[0])
    return T


def _kabsch_batch(A, B):
    """A,B: (n,3,3) three points each. Returns R (n,3,3), t (n,3) with B ≈ R A + t."""
    ca, cb = A.mean(1, keepdims=True), B.mean(1, keepdims=True)
    H = np.einsum("nki,nkj->nij", A - ca, B - cb)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(np.einsum("nji,nkj->nik", Vt, U)))
    D = np.ones((len(A), 3)); D[:, 2] = d
    R = np.einsum("nji,nj,nkj->nik", Vt, D, U)
    t = cb[:, 0] - np.einsum("nij,nj->ni", R, ca[:, 0])
    return R, t


def _ransac_rigid(sp, sf, dp, df, tol, n_iter=30000, local=0.08, knn_feat=3, seed=0, topk=40, n_keep=4, prior=None, sn=None, dn=None):
    """Locality-sampled RANSAC. sp/dp: (n,3) points, sf/df: (n,33) FPFH.  Hypotheses are built
    from 3 FPFH matches whose source points lie within `local` metres of each other (so they very
    likely belong to the same rigid link), then scored by the geometric inlier count.
    Returns the best 4x4 and its inlier fraction of sp."""
    rng = np.random.default_rng(seed)
    n = len(sp)
    dtree = cKDTree(dp)
    _, cand = cKDTree(df).query(sf, k=knn_feat)                 # (n,k) candidate dst for each src point
    cand = cand.reshape(n, -1)
    stree = cKDTree(sp)
    nbrs = stree.query(sp, k=min(400, n))[1]
    sub = rng.choice(n, min(1500, n), replace=False)
    ssub = sp[sub]
    nsub = sn[sub] if sn is not None else None
    pool = []
    done = 0
    B = 6000
    while done < n_iter:
        i = rng.integers(0, n, B)
        j = nbrs[i, rng.integers(1, nbrs.shape[1], B)]
        k = nbrs[i, rng.integers(1, nbrs.shape[1], B)]
        sel = (j != k)
        i, j, k = i[sel], j[sel], k[sel]
        ci = cand[i, rng.integers(0, cand.shape[1], len(i))]
        cj = cand[j, rng.integers(0, cand.shape[1], len(i))]
        ck = cand[k, rng.integers(0, cand.shape[1], len(i))]
        A = np.stack([sp[i], sp[j], sp[k]], 1)
        Bq = np.stack([dp[ci], dp[cj], dp[ck]], 1)
        # edge-length consistency + non-degenerate triangle
        ok = np.ones(len(i), bool)
        for a, b in ((0, 1), (0, 2), (1, 2)):
            la = np.linalg.norm(A[:, a] - A[:, b], axis=1); lb = np.linalg.norm(Bq[:, a] - Bq[:, b], axis=1)
            ok &= (np.abs(la - lb) < tol * 1.5) & (la > 5 * tol)
        A, Bq = A[ok], Bq[ok]
        done += B
        if len(A) == 0:
            continue
        R, t = _kabsch_batch(A, Bq)
        if prior is not None:
            okp = prior(R, t)
            R, t = R[okp], t[okp]
            if len(R) == 0:
                continue
        # score on subsample, in chunks
        scores = np.zeros(len(R))
        for c0 in range(0, len(R), 400):
            Rc, tc = R[c0:c0 + 400], t[c0:c0 + 400]
            Q = np.einsum("nij,mj->nmi", Rc, ssub) + tc[:, None, :]
            d, ix = dtree.query(Q.reshape(-1, 3), distance_upper_bound=tol * 1.5)
            good = d < tol * 1.5
            if nsub is not None:
                ix2 = np.where(good, ix, 0)
                Nq = np.einsum("nij,mj->nmi", Rc, nsub).reshape(-1, 3)
                good &= np.abs(np.einsum("ij,ij->i", Nq, dn[ix2])) > 0.85
            scores[c0:c0 + 400] = good.reshape(len(Rc), -1).mean(1)
        for q in np.argsort(-scores)[:6]:
            T = np.eye(4); T[:3, :3] = R[q]; T[:3, 3] = t[q]
            pool.append((float(scores[q]), T))
        if len(pool) > 400:
            pool = sorted(pool, key=lambda x: -x[0])[:200]
    pool.sort(key=lambda x: -x[0])
    if not pool:
        return [(-1.0, np.eye(4))]
    # distinct candidates (flips of symmetric parts are legitimately different hypotheses)
    out = []
    for sc, T in pool:
        if sc < 0.8 * pool[0][0]:
            break
        if all(_differs(T, T2) for _, T2 in out):
            out.append((sc, T))
        if len(out) >= n_keep:
            break
    return out


def _joint_prior(anchors, max_axis_dist=0.035, max_angle=np.pi):
    """Hypothesis filter: relative to some already-placed parent (Tp, contact centroid cc),
    the motion must be a rotation about an axis passing near cc (or a tiny motion)."""
    def f(R, t):
        ok = np.zeros(len(R), bool)
        for Tp, cc in anchors:
            Rp, tp = Tp[:3, :3], Tp[:3, 3]
            Rr = np.einsum("ji,njk->nik", Rp, R)                      # Rp^T R
            tr = np.einsum("ji,nj->ni", Rp, t - tp)                    # Rp^T (t - tp)
            u = np.einsum("nij,j->ni", Rr, cc) + tr - cc
            ang = np.arccos(np.clip((np.einsum("nii->n", Rr) - 1) / 2, -1, 1))
            un = np.linalg.norm(u, axis=1)
            dperp = un / (2 * np.sin(ang / 2) + 1e-9)
            ok |= np.where(ang < 0.09, un < 0.012, (dperp < max_axis_dist) & (ang < max_angle))
        return ok
    return f


def _differs(T1, T2, ang=np.radians(15), dist=0.02):
    dR = T1[:3, :3].T @ T2[:3, :3]
    a = np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))
    return a > ang or np.linalg.norm(T1[:3, 3] - T2[:3, 3]) > dist


def _contact(ref, a_idx, b_idx, tol, min_pts=25):
    """Centroid of the points of a that touch b (None if they do not touch)."""
    if len(a_idx) == 0 or len(b_idx) == 0:
        return None
    d = cKDTree(ref[b_idx]).query(ref[a_idx])[0]
    sel = d < tol * 3.0
    if sel.sum() < min_pts:
        return None
    return ref[a_idx][sel].mean(0)


# ───────────────────────── data ─────────────────────────
@dataclass
class JointGuess:
    name: str
    parent: str
    child: str
    type: str
    xyz: np.ndarray          # point on the axis (ref frame)
    axis: np.ndarray         # unit vector, sign chosen so the observed motion is positive
    observed: float          # observed displacement (rad or m)
    lower: float
    upper: float
    rms_mm: float            # residual of the moving part after the fitted motion
    inlier: float            # fraction of the moving part explained
    confidence: str


@dataclass
class Result:
    ref: np.ndarray                      # reference cloud (N,3)
    labels: np.ndarray                   # link index per ref point (0 = base)
    link_ids: List[str]
    joints: List[JointGuess]
    aligned: List[np.ndarray] = field(default_factory=list)
    moving: List[np.ndarray] = field(default_factory=list)   # bool masks per scan


def _screw(T):
    """(angle, unit axis, point on axis or None, translation along axis) of a rigid transform."""
    R, t = T[:3, :3], T[:3, 3]
    ang = float(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    if ang < 1e-3:
        n = np.linalg.norm(t)
        return 0.0, (t / n if n > 0 else np.array([0, 0, 1.0])), None, float(n)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if np.linalg.norm(w) < 1e-6:                              # ~180 deg: axis = eigenvector for eigenvalue 1
        ev, evec = np.linalg.eig(R)
        w = np.real(evec[:, np.argmin(np.abs(ev - 1))])
    a = w / np.linalg.norm(w)
    d = float(a @ t)
    q = np.linalg.lstsq(np.eye(3) - R, t - d * a, rcond=None)[0]
    q = q - (q @ a) * a
    return ang, a, q, d


def _knn_vote(P, labels, known, k=15, iters=6):
    """Fill labels where known==False from the majority of known neighbours (propagating)."""
    tree = cKDTree(P)
    _, nn = tree.query(P, k=k)
    lab = labels.copy().astype(float)
    kn = known.copy()
    for _ in range(iters):
        if kn.all():
            break
        nlab = np.where(kn[nn], lab[nn], np.nan)
        with np.errstate(all='ignore'):
            m = np.nanmean(nlab, axis=1)
        fill = (~kn) & ~np.isnan(m)
        lab[fill] = (m[fill] > 0.5)
        kn = kn | fill
    lab[~kn] = 0
    return lab.astype(bool)


def _largest_component(P, mask, radius):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    idx = np.where(mask)[0]
    if len(idx) < 3:
        return mask
    pairs = cKDTree(P[idx]).query_pairs(radius, output_type="ndarray")
    n = len(idx)
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, comp = connected_components(g, directed=False)
    big = np.argmax(np.bincount(comp))
    out = np.zeros_like(mask)
    out[idx[comp == big]] = True
    return out


def _moving_part(ref, P, voxel, tol, seed=0):
    """Rigid motion of the part of `ref` that moved to give `P` (P already aligned on the base)."""
    ptree, rtree = cKDTree(P), cKDTree(ref)
    static = ptree.query(ref)[0] < tol
    moving0 = np.where(~static)[0]
    pose_moved = np.where(rtree.query(P)[0] >= tol)[0]
    if len(moving0) < 120 or len(pose_moved) < 120:
        raise ValueError("this scan shows (almost) no motion relative to the rest scan - "
                         "rotate the joint further (>= 20 degrees) or check alignment")
    src, dst = ref[moving0], P[pose_moved]
    s_pcd, s_f = _prep(src, voxel)
    d_pcd, d_f = _prep(dst, voxel)
    # contact between moved and static part = where the joint is
    cc = _contact(ref, moving0, np.where(static)[0], tol)
    anchors = [(np.eye(4), cc)] if cc is not None else None
    cands = _ransac_rigid(np.asarray(s_pcd.points), np.asarray(s_f.data).T,
                          np.asarray(d_pcd.points), np.asarray(d_f.data).T, tol, n_iter=40000, seed=seed,
                          prior=_joint_prior(anchors, max_angle=np.radians(115)) if anchors else None,
                          sn=np.asarray(s_pcd.normals), dn=np.asarray(d_pcd.normals))
    if cands[0][0] < 0 and anchors:                       # prismatic / unusual: drop the prior
        cands = _ransac_rigid(np.asarray(s_pcd.points), np.asarray(s_f.data).T,
                              np.asarray(d_pcd.points), np.asarray(d_f.data).T, tol, n_iter=40000, seed=seed,
                              sn=np.asarray(s_pcd.normals), dn=np.asarray(d_pcd.normals))
    sp = _pcd(src); sp.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 3, max_nn=40))
    dp = _pcd(P); dp.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 3, max_nn=40))
    dpn = np.asarray(dp.normals)
    refined = []
    Un = P[pose_moved]
    for _, T in cands:
        for th in (tol * 5, tol * 3, tol * 1.5, tol * 1.0):
            T = _icp_refine(sp, dp, T, th, 40)
        f_src = float(_inliers(src, T, ptree, tol * 1.2).mean())
        Q = src @ T[:3, :3].T + T[:3, 3]
        f_dst = float((cKDTree(Q).query(Un)[0] < tol * 1.2).mean())     # moved-side points explained too
        refined.append((0.5 * (f_src + f_dst), T, f_src))
    best3 = max(refined, key=lambda r: r[0])
    best = (best3[2], best3[1])
    return best[1], best[0], moving0, static


def _classify(ref, P, T, voxel, tol):
    """Moving / static label for every reference point (ambiguous ones by neighbour vote)."""
    ptree = cKDTree(P)
    c_move = ptree.query(ref @ T[:3, :3].T + T[:3, 3])[0]
    c_stat = ptree.query(ref)[0]
    clear_move = (c_move < tol) & (c_stat >= tol * 1.5)
    clear_stat = (c_stat < tol) & (c_move >= tol * 1.5)
    known = clear_move | clear_stat
    lab = _knn_vote(ref, clear_move, known)
    if clear_move.any():                          # undecidable (symmetric) points below every moved point
        floor_side = (~known) & (ref[:, 2] < np.percentile(ref[clear_move, 2], 2))
        lab[floor_side] = False
    return _largest_component(ref, lab, voxel * 2.5)


def _moving_scale(src, P, T, tol):
    """Residual scale of the moved part: P ≈ s * (R src + t).  ~1 when the scan scale is right."""
    Q = src @ T[:3, :3].T + T[:3, 3]
    d, idx = cKDTree(P).query(Q)
    keep = d < tol * 2.5
    if keep.sum() < 100:
        return 1.0
    corr = o3d.utility.Vector2iVector(np.c_[np.where(keep)[0], idx[keep]])
    Ts = reg.TransformationEstimationPointToPoint(with_scaling=True).compute_transformation(
        _pcd(Q), _pcd(P), corr)
    return float(np.cbrt(abs(np.linalg.det(Ts[:3, :3]))))


def _explained(ref, P, voxel, tol, seed=0, n_iter=6000):
    """How completely do TWO rigid bodies (static + one moved rigid part) explain scan P?"""
    ptree, rtree = cKDTree(P), cKDTree(ref)
    static_frac = float((rtree.query(P)[0] < tol).mean())
    U = np.where(rtree.query(P)[0] >= tol)[0]                      # scan points not explained by static
    R = np.where(ptree.query(ref)[0] >= tol)[0]
    if len(U) < 150 or len(R) < 150:
        return static_frac, None
    s_pcd, s_f = _prep(ref[R], voxel)
    d_pcd, d_f = _prep(P[U], voxel)
    cands = _ransac_rigid(np.asarray(s_pcd.points), np.asarray(s_f.data).T,
                          np.asarray(d_pcd.points), np.asarray(d_f.data).T, tol, n_iter=n_iter, seed=seed,
                          sn=np.asarray(s_pcd.normals), dn=np.asarray(d_pcd.normals), n_keep=1)
    T = cands[0][1]
    utree = cKDTree(P[U])
    mv = float(_inliers(ref[R], T, utree, tol * 1.5).mean()) * len(U) / len(P)
    return static_frac + mv, T


def _process_scan(ref, S, voxel, tol, i):
    """Align scan S to the rest cloud, find the rigid motion of what moved, and iterate:
    scale and base alignment are re-estimated from the moved part / static part respectively."""
    n = len(ref)
    zr0, hr = ref[:, 2].min(), np.ptp(ref[:, 2])
    cands = align_to_reference(ref, S, tol, return_all=True)
    screened = []
    for st, Ac in cands[:9]:
        Pc = S @ Ac[:3, :3].T + Ac[:3, 3]
        sc, _ = _explained(ref, Pc, voxel, tol, seed=i)
        log.info("    alignment candidate scale %.3f: static %.0f%%, two-body explanation %.0f%%",
                 np.cbrt(abs(np.linalg.det(Ac[:3, :3]))), 100 * st, 100 * sc)
        screened.append((sc, Ac, st))
    # the cheap screening is noisy: re-score the best few with a much longer RANSAC
    best = (-1, None, 0.0)
    for sc0, Ac, st in sorted(screened, key=lambda x: -x[0])[:3]:
        Pc = S @ Ac[:3, :3].T + Ac[:3, 3]
        sc = max(_explained(ref, Pc, voxel, tol, seed=i + 100 + r, n_iter=25000)[0] for r in range(2))
        log.info("    re-score scale %.3f: %.0f%%", np.cbrt(abs(np.linalg.det(Ac[:3, :3]))), 100 * sc)
        if sc > best[0]:
            best = (sc, Ac, st)
    A = best[1]
    P = S @ A[:3, :3].T + A[:3, 3]
    if best[2] >= 0.92:
        # The whole scan coincides with the rest scan after a rotation about the vertical axis: the
        # joint is a yaw joint on a (nearly) rotation-symmetric base, which is invisible as a motion.
        log.info("  scan %d: coincides with the rest scan after a yaw -> vertical-axis (base) joint", i)
        return P, None, None, best[2]
    flipped = False
    for it in range(4):
        try:
            T, inl, moving0, static0 = _moving_part(ref, P, voxel, tol, seed=i)
        except ValueError:
            T, inl, moving0, static0 = None, 0.0, None, None
        if T is None:
            raise ValueError("scan %d shows no usable motion relative to the rest scan" % i)
        sc = _moving_scale(ref[moving0], P, T, tol)
        lab = _classify(ref, P, T, voxel, tol)
        # re-fit the base alignment on the static part only (rigid)
        stat = np.where(~lab)[0]
        if len(stat) > 300:
            sp_ = _pcd(ref[stat]); sp_.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 3, max_nn=40))
            pp_ = _pcd(P); pp_.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 3, max_nn=40))
            Ar = np.eye(4)
            for th in (tol * 3, tol * 1.5):
                Ar = reg.registration_icp(pp_, sp_, th, Ar, reg.TransformationEstimationPointToPlane(),
                                          reg.ICPConvergenceCriteria(max_iteration=30)).transformation
            P = P @ Ar[:3, :3].T + Ar[:3, 3]
        log.info("    refine %d: moved-part scale %.3f", it, sc)
        if abs(sc - 1) > 0.012 and it < 3:
            c = _base_center(P)
            P = (P - np.r_[c, zr0]) / sc + np.r_[c, zr0]
            continue
        T, inl, moving0, static0 = _moving_part(ref, P, voxel, tol, seed=i)
        lab = _classify(ref, P, T, voxel, tol)
        return P, T, lab, inl
    raise ValueError("scan %d: could not converge" % i)


def _yaw_split(ref, other_masks, voxel):
    """Mask of everything above the base for a vertical-axis joint on a round base.  The base top is
    where the radial extent of the never-moving part drops (base is wider than the waist)."""
    zr0, hr = ref[:, 2].min(), np.ptp(ref[:, 2])
    c = _base_center(ref)
    never = np.ones(len(ref), bool)
    for m in other_masks:
        never &= ~m
    rad = np.linalg.norm(ref[:, :2] - c, axis=1)
    nb = 60
    edges = zr0 + np.linspace(0, 0.6 * hr, nb + 1)
    prof = []
    for b in range(nb):
        sel = never & (ref[:, 2] >= edges[b]) & (ref[:, 2] < edges[b + 1])
        prof.append(np.percentile(rad[sel], 90) if sel.sum() > 20 else np.nan)
    prof = np.array(prof)
    r0 = np.nanmax(prof[:8]) if np.isfinite(prof[:8]).any() else np.nanmax(prof)
    z_split = zr0 + 0.12 * hr
    for b in range(nb - 2):
        if np.all(prof[b:b + 3] < 0.8 * r0):
            z_split = edges[b]
            break
    log.info("  yaw joint: base/waist split at z = %.3f (%.0f%% of height), axis through base centre", z_split,
             100 * (z_split - zr0) / hr)
    mask = ref[:, 2] >= z_split
    for m in other_masks:
        mask |= m
    return mask


def estimate(rest: np.ndarray, scans: List[np.ndarray], voxel: Optional[float] = None) -> Result:
    """rest: (N,3) cloud at the zero pose; scans[i]: cloud with only joint i moved.
    All gravity aligned (z up, floor at z=0)."""
    ref = np.asarray(rest, float)
    diag = float(np.linalg.norm(np.ptp(ref, axis=0)))
    voxel = voxel or diag / 90.0
    tol = voxel * 0.9
    log.info("jointscan: %d joint scans, diag %.3f, voxel %.1f mm", len(scans), diag, voxel * 1e3)
    n = len(ref)
    rtree = cKDTree(ref)
    aligned, Ts, masks, infos = [], [], [], []
    yaw_scan = None
    for i, S in enumerate(scans):
        P, T, lab, inl = _process_scan(ref, np.asarray(S, float), voxel, tol, i)
        aligned.append(P)
        if T is None:
            if yaw_scan is not None:
                raise ValueError("two scans look like pure yaw about the base - only one joint may be the base rotation")
            yaw_scan = i
            Ts.append(None); masks.append(None); infos.append(None)
            continue
        ang, a, q, d = _screw(T)
        log.info("  scan %d: moved %.1f%% of points, inlier %.0f%%, angle %.1f°, axial slide %.1f mm", i,
                 100 * lab.mean(), 100 * inl, np.degrees(ang), d * 1e3)
        Ts.append(T); masks.append(lab); infos.append((inl, ang, a, q, d))
    if yaw_scan is not None:
        masks[yaw_scan] = _yaw_split(ref, [m for m in masks if m is not None], voxel)
        infos[yaw_scan] = (1.0, 0.0, np.array([0, 0, 1.0]), None, 0.0)

    # nesting -> tree.   parent of i = smallest strictly larger set containing >= 80% of D_i
    k = len(scans)
    size = [m.sum() for m in masks]
    par = [None] * k
    for i in range(k):
        best = None
        for j in range(k):
            if j == i or size[j] <= size[i] * 1.05:
                continue
            if (masks[i] & masks[j]).sum() >= 0.8 * size[i] and (best is None or size[j] < size[best]):
                best = j
        par[i] = best
    # link per point = smallest distal set containing it (0 = base)
    labels = np.zeros(n, int)
    order_by_size = sorted(range(k), key=lambda i: -size[i])      # big first, smaller overwrite
    for i in order_by_size:
        labels[masks[i]] = i + 1
    # depth-first link order
    ids = ["base"] + [f"link_{i + 1}" for i in range(k)]
    joints = []
    for i in range(k):
        if i == yaw_scan:
            z_split = float(ref[masks[i], 2].min())
            c = _base_center(ref)
            pname = ids[par[i] + 1] if par[i] is not None else "base"
            joints.append(JointGuess(f"joint_{i + 1}", pname, ids[i + 1], "revolute", np.r_[c, z_split],
                                     np.array([0, 0, 1.0]), 0.0, -np.pi / 2, np.pi / 2, 0.0, 1.0, "medium"))
            continue
        inl, ang, a, q, d = infos[i]
        T = Ts[i]
        moving = np.where(masks[i])[0]
        res = cKDTree(aligned[i]).query(ref[moving] @ T[:3, :3].T + T[:3, 3])[0]
        rms = float(np.sqrt((res ** 2).mean()) * 1e3)
        stat = np.where(~masks[i])[0]
        cc = _contact(ref, moving, stat, tol) if len(stat) else None
        if ang < np.radians(2.0):            # translation -> prismatic
            kind, axis, xyz, obs = "prismatic", a, ref[moving].mean(0), d
            lo, hi = min(0, obs) - 0.02, max(0, obs) + 0.02
        else:
            kind, axis, xyz, obs = "revolute", a, q, ang
            if cc is not None:
                xyz = q + ((cc - q) @ a) * a
            lo, hi = min(-np.pi / 2, -obs - 0.1), max(np.pi / 2, obs + 0.1)
        conf = "high" if (inl > 0.9 and rms < 3.0) else ("medium" if inl > 0.75 else "low")
        pname = ids[par[i] + 1] if par[i] is not None else "base"
        joints.append(JointGuess(f"joint_{i + 1}", pname, ids[i + 1], kind, np.asarray(xyz), np.asarray(axis),
                                 float(obs), float(lo), float(hi), rms, float(inl), conf))
    return Result(ref, labels, ids, joints, aligned, masks)
