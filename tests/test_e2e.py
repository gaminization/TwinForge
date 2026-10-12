"""
End-to-end tests — no hardware, no API keys, no display required.
Exercises every stage of the real pipeline with synthetic data.

Run:  pytest tests/test_e2e.py -v -s
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import numpy as np
import pytest
from pathlib import Path
from xml.etree import ElementTree as ET


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _require_open3d():
    try:
        import open3d
    except ImportError:
        pytest.skip("open3d not installed")


# ─────────────────────────────────────────────────────────────────────────────
#  Stage 1: Acquisition
# ─────────────────────────────────────────────────────────────────────────────

def test_acquisition_demo():
    _require_open3d()
    from core.acquisition import load
    scan = load(source=None, demo=True, n_poses=3)

    assert scan.pcd is not None
    assert len(scan.pcd.points) > 1000
    assert len(scan.pose_pcds) == 3
    assert hasattr(scan, "_demo_link_pcds")
    assert len(scan._demo_link_pcds) >= 4
    print(f"\n  Demo scan: {len(scan.pcd.points):,} pts, "
          f"{len(scan.pose_pcds)} poses")


def test_acquisition_from_ply(tmp_path):
    _require_open3d()
    import open3d as o3d
    from core.acquisition import load

    # Write a tiny synthetic .ply
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.random.randn(500, 3) * 0.1)
    ply = tmp_path / "test.ply"
    o3d.io.write_point_cloud(str(ply), pcd)

    scan = load(str(ply))
    assert len(scan.pcd.points) > 0
    print(f"\n  PLY scan: {len(scan.pcd.points):,} pts")


# ─────────────────────────────────────────────────────────────────────────────
#  Stage 2: Segmentation
# ─────────────────────────────────────────────────────────────────────────────

def test_segmentation_demo():
    _require_open3d()
    from core.acquisition import load
    from core.segmenter   import segment

    scan       = load(source=None, demo=True, n_poses=1)
    components = segment(scan)

    assert len(components) >= 3, f"Expected ≥3 components, got {len(components)}"

    # Exactly one base
    bases = [c for c in components if c.type == "base"]
    assert len(bases) == 1, f"Expected 1 base, got {len(bases)}"

    # Exactly one end-effector
    ees = [c for c in components if c.type == "end_effector"]
    assert len(ees) == 1

    # Each non-base component has a parent
    for c in components:
        if c.type != "base":
            assert c.parent_id is not None, f"{c.id} missing parent"

    # All centroids are real numbers
    for c in components:
        assert np.all(np.isfinite(c.centroid)), f"{c.id} centroid has NaN"

    print(f"\n  Segmented: {[c.label for c in components]}")


def test_segmentation_pca_axes():
    _require_open3d()
    from core.acquisition import load
    from core.segmenter   import segment

    scan       = load(source=None, demo=True)
    components = segment(scan)

    for c in components:
        # PCA axes must be orthonormal
        V = c.pca_axes            # 3×3
        err = np.max(np.abs(V.T @ V - np.eye(3)))
        assert err < 0.01, f"{c.id} PCA axes not orthonormal (err={err:.4f})"


# ─────────────────────────────────────────────────────────────────────────────
#  Stage 3: Joint Detection
# ─────────────────────────────────────────────────────────────────────────────

def test_joint_detection_count():
    _require_open3d()
    from core.acquisition    import load
    from core.segmenter      import segment
    from core.joint_detector import detect

    scan       = load(source=None, demo=True, n_poses=3)
    components = segment(scan)
    joints     = detect(components, scan_data=scan)

    n_movable = sum(1 for c in components if c.movable)
    assert len(joints) == n_movable, (
        f"Expected {n_movable} joints (one per movable link), got {len(joints)}")
    print(f"\n  Joints: {len(joints)} (expected {n_movable})")


def test_joint_axes_are_unit_vectors():
    _require_open3d()
    from core.acquisition    import load
    from core.segmenter      import segment
    from core.joint_detector import detect

    scan       = load(source=None, demo=True, n_poses=3)
    components = segment(scan)
    joints     = detect(components, scan_data=scan)

    for j in joints:
        norm = float(np.linalg.norm(j.axis))
        assert abs(norm - 1.0) < 0.01, \
            f"Joint {j.id} axis not unit (norm={norm:.4f})"


def test_joint_types_valid():
    _require_open3d()
    from core.acquisition    import load
    from core.segmenter      import segment
    from core.joint_detector import detect

    valid_types = {"revolute","prismatic","continuous","fixed","floating","planar"}
    scan        = load(source=None, demo=True, n_poses=1)
    components  = segment(scan)
    joints      = detect(components)

    for j in joints:
        assert j.type in valid_types, \
            f"Joint {j.id} has invalid type '{j.type}'"
        if j.type in ("revolute","prismatic"):
            assert j.limit_lower <= j.limit_upper, \
                f"Joint {j.id} lower > upper"
    print(f"\n  Joint types: {[j.type for j in joints]}")


def test_neck_detection():
    """Neck point must lie between the two component centroids."""
    _require_open3d()
    from core.acquisition    import load
    from core.segmenter      import segment
    from core.joint_detector import detect

    scan       = load(source=None, demo=True, n_poses=1)
    components = segment(scan)
    joints     = detect(components)

    comp_map = {c.id: c for c in components}
    for j in joints:
        parent = comp_map[j.parent_link]
        child  = comp_map[j.child_link]
        p0, p1 = parent.centroid, child.centroid
        # Project neck onto connecting segment — must be in [0,1]
        seg = p1 - p0
        L   = np.linalg.norm(seg)
        if L < 1e-6:
            continue
        t = np.dot(j.origin_xyz - p0, seg) / (L * L)
        assert -0.1 <= t <= 1.1, \
            f"Neck of {j.id} outside segment (t={t:.3f})"


# ─────────────────────────────────────────────────────────────────────────────
#  Stage 4: URDF Generation
# ─────────────────────────────────────────────────────────────────────────────

def test_urdf_generated(tmp_path):
    _require_open3d()
    from core.acquisition    import load
    from core.segmenter      import segment
    from core.joint_detector import detect
    from core.urdf_builder   import build

    scan       = load(source=None, demo=True, n_poses=1)
    components = segment(scan)
    joints     = detect(components)
    urdf_path  = build(components, joints, "test_arm", tmp_path)

    assert urdf_path.exists(), "URDF file not created"
    assert urdf_path.stat().st_size > 500, "URDF file suspiciously small"
    print(f"\n  URDF: {urdf_path} ({urdf_path.stat().st_size:,} bytes)")


def test_urdf_xml_structure(tmp_path):
    _require_open3d()
    from core.acquisition    import load
    from core.segmenter      import segment
    from core.joint_detector import detect
    from core.urdf_builder   import build

    scan       = load(source=None, demo=True, n_poses=1)
    components = segment(scan)
    joints     = detect(components)
    urdf_path  = build(components, joints, "test_arm", tmp_path)

    root        = ET.parse(str(urdf_path)).getroot()
    link_names  = {el.get("name") for el in root.findall("link")}
    joint_names = {el.get("name") for el in root.findall("joint")}
    comp_ids    = {c.id for c in components}
    joint_ids   = {j.id for j in joints}

    assert link_names  == comp_ids,  f"Link mismatch: {link_names} vs {comp_ids}"
    assert joint_names == joint_ids, f"Joint mismatch: {joint_names} vs {joint_ids}"

    # Every joint references existing links
    for jel in root.findall("joint"):
        p = jel.find("parent").get("link")
        c = jel.find("child").get("link")
        assert p in link_names, f"Unknown parent link '{p}'"
        assert c in link_names, f"Unknown child link '{c}'"

    # Axis elements are unit vectors
    for jel in root.findall("joint"):
        ax = jel.find("axis")
        if ax is not None:
            xyz  = [float(v) for v in ax.get("xyz").split()]
            norm = np.linalg.norm(xyz)
            assert abs(norm - 1.0) < 0.02, \
                f"Joint {jel.get('name')} axis norm={norm:.4f}"

    print(f"\n  Links: {sorted(link_names)}")
    print(f"  Joints: {sorted(joint_names)}")


def test_stl_meshes_generated(tmp_path):
    _require_open3d()
    from core.acquisition    import load
    from core.segmenter      import segment
    from core.joint_detector import detect
    from core.urdf_builder   import build

    scan       = load(source=None, demo=True, n_poses=1)
    components = segment(scan)
    joints     = detect(components)
    urdf_path  = build(components, joints, "test_arm", tmp_path)

    mesh_dir = tmp_path / "meshes"
    assert mesh_dir.exists()
    stl_files = list(mesh_dir.glob("*.stl"))
    assert len(stl_files) >= len(components) - 1, \
        f"Expected ≥{len(components)-1} STL files, got {len(stl_files)}"

    # All STL files must be non-empty
    for stl in stl_files:
        assert stl.stat().st_size > 100, f"Empty STL: {stl}"

    print(f"\n  STL files: {[f.name for f in stl_files]}")


def test_inertia_positive_definite(tmp_path):
    _require_open3d()
    from core.acquisition    import load
    from core.segmenter      import segment
    from core.joint_detector import detect
    from core.urdf_builder   import build

    scan       = load(source=None, demo=True, n_poses=1)
    components = segment(scan)
    joints     = detect(components)
    urdf_path  = build(components, joints, "test_arm", tmp_path)

    root = ET.parse(str(urdf_path)).getroot()
    for link in root.findall("link"):
        inert = link.find("inertial")
        if inert is None:
            continue
        mass = float(inert.find("mass").get("value"))
        assert mass > 0, f"Link {link.get('name')} mass={mass}"
        I_el = inert.find("inertia")
        I = np.array([
            [float(I_el.get("ixx")), float(I_el.get("ixy")), float(I_el.get("ixz"))],
            [float(I_el.get("ixy")), float(I_el.get("iyy")), float(I_el.get("iyz"))],
            [float(I_el.get("ixz")), float(I_el.get("iyz")), float(I_el.get("izz"))],
        ])
        eigs = np.linalg.eigvalsh(I)
        assert np.all(eigs >= -1e-9), \
            f"Link {link.get('name')} inertia not PSD: eigenvalues={eigs}"


# ─────────────────────────────────────────────────────────────────────────────
#  Stage 5: Forward Kinematics
# ─────────────────────────────────────────────────────────────────────────────

def test_fk_root_is_identity(tmp_path):
    _require_open3d()
    from core.acquisition    import load
    from core.segmenter      import segment
    from core.joint_detector import detect
    from core.urdf_builder   import build
    from core.visualizer     import parse_urdf, forward_kinematics

    scan       = load(source=None, demo=True, n_poses=1)
    components = segment(scan)
    joints     = detect(components)
    urdf_path  = build(components, joints, "test_arm", tmp_path)

    links, jmap, root_id = parse_urdf(urdf_path)
    T_world = forward_kinematics(links, jmap, {})

    # Root link must have identity transform (zero joints)
    T_root = T_world[root_id]
    assert np.allclose(T_root, np.eye(4), atol=1e-6), \
        f"Root transform not identity:\n{T_root}"
    print(f"\n  Root link: {root_id}")


def test_fk_moves_child_links(tmp_path):
    _require_open3d()
    from core.acquisition    import load
    from core.segmenter      import segment
    from core.joint_detector import detect
    from core.urdf_builder   import build
    from core.visualizer     import parse_urdf, forward_kinematics

    scan       = load(source=None, demo=True, n_poses=1)
    components = segment(scan)
    joints     = detect(components)
    urdf_path  = build(components, joints, "test_arm", tmp_path)

    links, jmap, root_id = parse_urdf(urdf_path)

    # Set all joints to 45 degrees
    joint_vals_zero = {}
    joint_vals_45   = {}
    for child_id, j in jmap.items():
        if j["type"] == "revolute":
            joint_vals_zero[j["id"]] = 0.0
            joint_vals_45[j["id"]]   = math.pi / 4

    T_zero = forward_kinematics(links, jmap, joint_vals_zero)
    T_45   = forward_kinematics(links, jmap, joint_vals_45)

    # At least one link must have moved
    moved = False
    for link in links:
        lid = link["id"]
        if lid == root_id:
            continue
        if not np.allclose(T_zero[lid], T_45[lid], atol=1e-4):
            moved = True
            break

    assert moved, "No links moved when joints were rotated 45°"
    print(f"\n  FK correctly moves child links when joints change ✔")


def test_fk_rotation_is_orthogonal(tmp_path):
    _require_open3d()
    from core.acquisition    import load
    from core.segmenter      import segment
    from core.joint_detector import detect
    from core.urdf_builder   import build
    from core.visualizer     import parse_urdf, forward_kinematics

    scan       = load(source=None, demo=True, n_poses=1)
    components = segment(scan)
    joints     = detect(components)
    urdf_path  = build(components, joints, "test_arm", tmp_path)

    links, jmap, root_id = parse_urdf(urdf_path)
    joint_vals = {j["id"]: 0.7 for j in jmap.values() if j["type"]=="revolute"}
    T_world    = forward_kinematics(links, jmap, joint_vals)

    for lid, T in T_world.items():
        R   = T[:3, :3]
        err = np.max(np.abs(R.T @ R - np.eye(3)))
        assert err < 1e-6, \
            f"Link {lid} world rotation not orthogonal (err={err:.2e})"


# ─────────────────────────────────────────────────────────────────────────────
#  Full pipeline smoke test
# ─────────────────────────────────────────────────────────────────────────────

def test_full_pipeline_headless(tmp_path):
    """Run the complete pipeline end-to-end in headless mode."""
    _require_open3d()
    from core.acquisition    import load
    from core.segmenter      import segment
    from core.joint_detector import detect
    from core.urdf_builder   import build
    from core.visualizer     import save_frames

    scan       = load(source=None, demo=True, n_poses=2)
    components = segment(scan)
    joints     = detect(components, scan_data=scan)
    urdf_path  = build(components, joints, "smoke_test", tmp_path)

    assert urdf_path.exists()

    frames_dir = tmp_path / "frames"
    saved = save_frames(
        urdf_path=urdf_path,
        output_dir=frames_dir,
        components=components,
        n_frames=10,       # fast
        period=2.0,
    )

    assert len(saved) == 10
    for p in saved:
        assert p.exists()
        assert p.stat().st_size > 1000  # non-trivial PNG

    print(f"\n  Full pipeline ✔")
    print(f"  URDF:   {urdf_path}")
    print(f"  Frames: {len(saved)} PNGs in {frames_dir}")
