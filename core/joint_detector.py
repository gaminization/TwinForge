"""
Joint Detector — Full AI pipeline integration
==============================================
Module 2 — BFS/A* for kinematic chain (R&N §3.4, §3.5)
Module 3 — Hill-climbing axis optimisation (R&N §4.1.1)
Module 4 — FOL rules + Forward chaining for joint type (R&N §8, §9.3)
Module 5 — Naive Bayes classifier (R&N §13.2, §20.2)
Module 6 — Planning state updates (R&N §10.1)

For each parent→child component pair:
  1. Neck detection         → joint origin (xyz)
  2. Axis from PCA or ICP   → candidate axis
  3. Hill-climbing          → refined axis (Module 3)
  4. Bayesian classifier    → joint type + confidence (Module 5)
  5. Forward chaining       → type confirmation or override (Module 4)
  6. Limit estimation       → observed from multi-pose or defaults
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class Joint:
    id:             str
    parent_link:    str
    child_link:     str
    type:           str
    axis:           np.ndarray = field(default_factory=lambda: np.array([0.,0.,1.]))
    origin_xyz:     np.ndarray = field(default_factory=lambda: np.zeros(3))
    origin_rpy:     np.ndarray = field(default_factory=lambda: np.zeros(3))
    limit_lower:    float = -1.5708
    limit_upper:    float =  1.5708
    limit_effort:   float = 30.0
    limit_velocity: float = 1.0
    confidence:     float = 1.0
    # Provenance
    type_source:    str = "unknown"   # 'bayesian' | 'forward_chaining' | 'fol' | 'default'
    axis_source:    str = "unknown"   # 'icp_screw' | 'hill_climbing' | 'pca'


def detect(components: list, scan_data=None) -> List[Joint]:
    """
    Full AI-powered joint detection.

    Pipeline per joint:
      PCA/ICP axis → hill-climbing refinement → Bayesian type → FC confirmation
    """
    from core.ai_reasoning import (
        BayesianJointClassifier, ForwardChainer, FOLComponentRules,
        HillClimbingAxisOptimizer, BFSKinematicSearch,
    )

    bayes    = BayesianJointClassifier()
    fc       = ForwardChainer()
    fol      = FOLComponentRules()
    hc       = HillClimbingAxisOptimizer(delta=0.04, max_iter=40)

    comp_map  = {c.id: c for c in components}
    all_n_pts = [c.n_points for c in components]

    # BFS adjacency for depth computation
    pairs     = _candidate_pairs(components)
    adj: Dict[str, List[str]] = {}
    for parent, child in pairs:
        adj.setdefault(parent.id, []).append(child.id)

    root_id   = next((c.id for c in components if c.type == "base"),
                      components[0].id)

    joints: List[Joint] = []
    log.info(f"Detecting joints for {len(pairs)} pair(s) — "
             f"Bayes + HC + FC pipeline")

    for parent, child in pairs:
        jid = f"joint_{parent.id}_{child.id}"

        # ── 1. Joint origin (neck detection) ─────────────────────────────────
        origin = _neck_point(parent, child)

        # ── 2. Initial axis (ICP screw or PCA) ───────────────────────────────
        n_useful = sum(
            1 for pf in getattr(scan_data, "pose_pcds", [])
            if len(np.asarray(pf.points)) > 50
        )
        if n_useful >= 2:
            axis_init, axis_conf, axis_src = _screw_axis(child, scan_data)
        else:
            axis_init, axis_conf, axis_src = _pca_axis(parent, child)

        # ── 3. Hill-climbing axis refinement (Module 3) ───────────────────────
        # Fitness: proxy using how well axis separates parent/child point clouds
        def fitness_fn(ax: np.ndarray) -> float:
            return _axis_separation_score(ax, parent, child)

        axis_hc, hc_score = hc.optimise(axis_init, fitness_fn)
        if hc_score > axis_conf:
            axis      = axis_hc
            axis_conf = hc_score
            axis_src  = "hill_climbing"
        else:
            axis = axis_init

        # Ensure unit vector
        axis = axis / (np.linalg.norm(axis) + 1e-9)

        # ── 4. Bayesian joint type classification (Module 5) ─────────────────
        depth  = BFSKinematicSearch.find_level(root_id, child.id, adj)
        depth  = max(depth, 0)
        jtype_bayes, bayes_conf, posteriors = bayes.classify(
            disc_ratio   = float(fol.disc_ratio(child)),
            rod_ratio    = float(fol.rod_ratio(child)),
            tree_depth   = depth,
            n_points     = child.n_points,
            all_n_points = all_n_pts,
            movable      = child.movable,
        )
        type_src = "bayesian"

        # ── 5. Forward chaining confirmation (Module 4) ───────────────────────
        facts = {
            "child_type":       child.type,
            "child_movable":    child.movable,
            "child_rod_ratio":  float(fol.rod_ratio(child)),
            "joint_type":       jtype_bayes,
            "axis_confidence":  axis_conf,
            "limit_range":      abs(-1.5708 - 1.5708) * 2,
        }
        augmented = fc.infer(facts)
        jtype_fc  = augmented.get("joint_type", jtype_bayes)

        if jtype_fc != jtype_bayes:
            log.debug(f"  FC overrides Bayes: {jtype_bayes} → {jtype_fc}")
            jtype    = jtype_fc
            type_src = "forward_chaining"
        else:
            jtype = jtype_bayes

        # Combined confidence: geometric mean of Bayes + axis
        confidence = float(np.sqrt(bayes_conf * max(axis_conf, 0.5)))

        # ── 6. Limits ────────────────────────────────────────────────────────
        lo, hi = _estimate_limits(jtype, child, scan_data, axis)

        j = Joint(
            id=jid,
            parent_link=parent.id, child_link=child.id,
            type=jtype, axis=axis,
            origin_xyz=origin, origin_rpy=np.zeros(3),
            limit_lower=lo, limit_upper=hi,
            confidence=confidence,
            type_source=type_src, axis_source=axis_src,
        )
        joints.append(j)

        log.info(f"  {jid}: {jtype} [{type_src}]  "
                 f"axis={axis.round(3)} [{axis_src}]  conf={confidence:.2f}")
        log.debug(f"    Posteriors: " +
                  " ".join(f"{k}={v:.2f}" for k,v in posteriors.items()))

    return joints


def get_rescan_hint(low_conf_items: List[str], joints: List[Joint]) -> str:
    jmap = {j.id: j for j in joints}
    hints = []
    for item in low_conf_items:
        for jid, j in jmap.items():
            if jid in item:
                hints.append(
                    f"Rotate joint '{jid.replace('joint_','')}' ~20-30° "
                    f"and capture from {j.axis_source} direction.")
    return " | ".join(hints) if hints else \
           "Walk around the robot and capture additional angles."


# ── Neck detection ─────────────────────────────────────────────────────────────

def _neck_point(parent, child) -> np.ndarray:
    """
    Neck detection: find minimum cross-section between two components.
    Slice along the parent→child axis, find bin with fewest points.
    """
    p0, p1 = parent.centroid, child.centroid
    d      = p1 - p0
    L      = np.linalg.norm(d)
    if L < 1e-6:
        return (p0 + p1) / 2.0
    d_hat  = d / L

    pts_p  = np.asarray(parent.pcd.points)
    pts_c  = np.asarray(child.pcd.points)
    pts    = np.vstack([pts_p, pts_c])
    proj   = (pts - p0) @ d_hat

    n_bins = 20
    counts, bins = np.histogram(proj, bins=n_bins, range=(0, L))
    neck_bin     = int(np.argmin(counts))
    neck_t       = (bins[neck_bin] + bins[neck_bin + 1]) / 2
    return p0 + neck_t * d_hat


# ── Axis estimation ────────────────────────────────────────────────────────────

def _screw_axis(child, scan_data) -> Tuple[np.ndarray, float, str]:
    """
    ICP screw-axis estimation from multi-pose data.
    Returns (axis, confidence, source_label).
    """
    import open3d as o3d

    r = np.max(np.asarray(child.bbox.get_extent())) * 0.8
    pose_subs = []
    for pf in scan_data.pose_pcds:
        pts   = np.asarray(pf.points)
        dists = np.linalg.norm(pts - child.centroid, axis=1)
        idx   = np.where(dists < r)[0]
        if len(idx) < 30:
            continue
        sub = pf.select_by_index(idx.tolist())
        sub.estimate_normals()
        pose_subs.append(sub)

    if len(pose_subs) < 2:
        return _pca_axis(None, child)

    ref    = pose_subs[0]
    axes, fitnesses = [], []
    for sub in pose_subs[1:]:
        try:
            result = o3d.pipelines.registration.registration_icp(
                sub, ref, 0.03, np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPlane())
            R     = result.transformation[:3, :3]
            angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
            if abs(angle) > 0.02:
                ax  = np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])
                ax /= (2 * np.sin(angle) + 1e-9)
                axes.append(ax / (np.linalg.norm(ax) + 1e-9))
                fitnesses.append(result.fitness)
        except Exception:
            pass

    if not axes:
        return _pca_axis(None, child)

    ref_ax  = axes[0]
    aligned = [a if np.dot(a, ref_ax) >= 0 else -a for a in axes]
    mean_ax = np.mean(aligned, axis=0)
    mean_ax /= np.linalg.norm(mean_ax) + 1e-9
    conf    = min(0.95, 0.5 + float(np.mean(fitnesses)) * 0.5)
    return mean_ax, conf, "icp_screw"


def _pca_axis(parent, child) -> Tuple[np.ndarray, float, str]:
    """PCA-based axis estimate from bounding-box geometry."""
    if child is None or child.bbox is None:
        return np.array([0., 0., 1.]), 0.50, "pca"

    extent  = np.sort(child.extent)
    disc    = extent[1] / (extent[0] + 1e-9) > 2.5

    if disc:
        axis = child.pca_axes[:, 0]   # smallest eigenvector = disc normal
    else:
        axis      = child.pca_axes[:, 2]   # largest eigenvector = link direction
        if parent is not None:
            connector = child.centroid - parent.centroid
            connector /= np.linalg.norm(connector) + 1e-9
            # Axis most perpendicular to connector
            dots = np.abs(child.pca_axes.T @ connector)
            axis = child.pca_axes[:, np.argmin(dots)]

    axis = axis / (np.linalg.norm(axis) + 1e-9)
    return axis, 0.60, "pca"


def _axis_separation_score(axis: np.ndarray, parent, child) -> float:
    """
    Hill-climbing fitness: project both clouds onto axis.
    Good axis → separated clusters → high score.
    """
    pts_p  = np.asarray(parent.pcd.points)
    pts_c  = np.asarray(child.pcd.points)
    ax     = axis / (np.linalg.norm(axis) + 1e-9)
    proj_p = pts_p @ ax
    proj_c = pts_c @ ax
    mean_p = float(np.mean(proj_p))
    mean_c = float(np.mean(proj_c))
    sep    = abs(mean_c - mean_p)
    std_p  = float(np.std(proj_p)) + 1e-6
    std_c  = float(np.std(proj_c)) + 1e-6
    # Fisher's criterion: between-class / within-class variance
    return sep / (std_p + std_c)


# ── Limit estimation ───────────────────────────────────────────────────────────

def _estimate_limits(jtype: str, child, scan_data, axis: np.ndarray
                     ) -> Tuple[float, float]:
    if jtype == "fixed":
        return 0.0, 0.0
    if jtype == "continuous":
        return 0.0, 0.0
    if jtype == "prismatic":
        travel = float(np.max(child.extent)) * 0.8
        return -travel / 2, travel / 2

    # Revolute: estimate from observed angles across poses
    angles = _observed_angles(child, scan_data, axis)
    if len(angles) >= 2:
        lo = float(np.min(angles)) - 0.05
        hi = float(np.max(angles)) + 0.05
        return max(lo, -np.pi), min(hi, np.pi)

    return -1.5708, 1.5708


def _observed_angles(child, scan_data, axis: np.ndarray) -> List[float]:
    import open3d as o3d
    if scan_data is None or len(getattr(scan_data, "pose_pcds", [])) < 2:
        return []
    r       = np.max(np.asarray(child.bbox.get_extent())) * 0.8
    ref_pcd = None
    angles  = []
    for pf in scan_data.pose_pcds:
        pts   = np.asarray(pf.points)
        dists = np.linalg.norm(pts - child.centroid, axis=1)
        idx   = np.where(dists < r)[0]
        if len(idx) < 20:
            continue
        sub = pf.select_by_index(idx.tolist())
        sub.estimate_normals()
        if ref_pcd is None:
            ref_pcd = sub
            continue
        try:
            res = o3d.pipelines.registration.registration_icp(
                sub, ref_pcd, 0.03, np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPlane())
            R     = res.transformation[:3, :3]
            angle = float(np.arccos(np.clip((np.trace(R)-1)/2, -1, 1)))
            angles.append(angle)
        except Exception:
            pass
    return angles


# ── Candidate pairs ────────────────────────────────────────────────────────────

def _candidate_pairs(components: list) -> List[Tuple]:
    """Build parent→child pairs from parent_id links, with spatial fallback."""
    comp_map = {c.id: c for c in components}
    pairs    = []
    for comp in components:
        if comp.parent_id and comp.parent_id in comp_map:
            pairs.append((comp_map[comp.parent_id], comp))

    if not pairs:
        # Spatial fallback: chain by centroid height
        sorted_c = sorted(components,
                          key=lambda c: float(np.linalg.norm(c.centroid)))
        for i in range(1, len(sorted_c)):
            pairs.append((sorted_c[i-1], sorted_c[i]))

    # Deduplicate by id pair (Component is unhashable)
    seen  = set()
    dedup = []
    for p, c in pairs:
        key = (p.id, c.id)
        if key not in seen:
            seen.add(key)
            dedup.append((p, c))
    return dedup
