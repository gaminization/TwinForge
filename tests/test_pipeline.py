"""
Regression tests for the TwinForge pipeline.

  pytest -q tests

Needs: open3d numpy trimesh yourdfpy scipy pytest
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

trimesh = pytest.importorskip("trimesh")
yourdfpy = pytest.importorskip("yourdfpy")

from core import joint_detector, segmenter, urdf_builder          # noqa: E402
from core.acquisition import load                                  # noqa: E402
from core.demo_arm import _link_world_transforms                   # noqa: E402


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    out = tmp_path_factory.mktemp("urdf")
    scan = load(demo=True)
    comps = segmenter.segment(scan)
    joints = joint_detector.detect(comps, scan)
    urdf = urdf_builder.build(comps, joints, "demo", out)
    return scan, comps, joints, urdf, out


def _robot(urdf, out):
    return yourdfpy.URDF.load(
        str(urdf), load_meshes=False, build_collision_scene_graph=False,
        filename_handler=lambda f: str(out / "meshes" / Path(f).name))


def _world_meshes(urdf, out, cfg):
    u = _robot(urdf, out)
    u.update_cfg(cfg)
    res = {}
    for link in u.robot.links:
        for v in link.visuals:
            m = trimesh.load(out / "meshes" / Path(v.geometry.mesh.filename).name, force="mesh")
            m.apply_transform(u.get_transform(link.name))
            res[link.name] = m
    return res


def test_urdf_structure(demo):
    _, comps, joints, urdf, out = demo
    u = _robot(urdf, out)
    assert len(u.robot.links) == len(comps) == 5
    assert len(u.actuated_joint_names) == len(joints) == 4
    for c in comps:
        assert (out / "meshes" / f"{c.id}.stl").stat().st_size > 1000


def test_link_frames_match_joint_table(demo):
    """URDF forward kinematics reproduces the ground-truth link poses."""
    _, _, _, urdf, out = demo
    u = _robot(urdf, out)
    u.update_cfg(np.zeros(4))
    for lid, T in _link_world_transforms().items():
        assert np.allclose(u.get_transform(lid), T, atol=1e-6), lid


def test_zero_pose_mesh_matches_scan(demo):
    """Link-local STL + FK == the world-frame mesh the pipeline scanned (regression for the
    world-vs-local frame bug that displaced links by up to 266 mm)."""
    _, comps, _, urdf, out = demo
    placed = _world_meshes(urdf, out, np.zeros(4))
    for c in comps:
        got = placed[c.id].bounds
        want = np.asarray([c.mesh.get_min_bound(), c.mesh.get_max_bound()])
        assert np.abs(got - want).max() < 1e-4, c.id


def test_articulation_moves_children_rigidly(demo):
    """Rotating joint k moves links k+1.. and leaves links <= k untouched."""
    _, _, _, urdf, out = demo
    names = ["base", "waist", "arm1", "arm2", "gripper"]
    rest = _world_meshes(urdf, out, np.zeros(4))
    for k in range(4):
        cfg = np.zeros(4)
        cfg[k] = 0.7
        moved = _world_meshes(urdf, out, cfg)
        for i, n in enumerate(names):
            shift = np.linalg.norm(moved[n].centroid - rest[n].centroid)
            if i <= k:
                assert shift < 1e-6, (k, n)
            else:
                assert shift > 1e-3, (k, n)


def test_real_scan_path_world_joints_roundtrip(tmp_path):
    """Real-scan joints are given in world coords; builder must convert to link-relative."""
    scan = load(demo=True)
    pcd = scan.pcd
    comps = segmenter._real_segment(pcd, 5)
    assert len(comps) == 5
    T = _link_world_transforms()
    order = ["base", "waist", "arm1", "arm2", "gripper"]
    import json
    spec = [dict(parent=comps[i - 1].id, child=comps[i].id, type="revolute",
                 axis=(T[order[i]][:3, :3] @ [0, 0, 1]).tolist(),
                 xyz=T[order[i]][:3, 3].tolist()) for i in range(1, 5)]
    p = tmp_path / "j.json"
    p.write_text(json.dumps(spec))
    joints = joint_detector.detect(comps, None, overrides_path=str(p))
    urdf = urdf_builder.build(comps, joints, "scan", tmp_path)
    u = _robot(urdf, tmp_path)
    assert len(u.actuated_joint_names) == 4
    u.update_cfg(np.zeros(4))
    # joint positions in the URDF must land on the supplied world positions
    for i in range(1, 5):
        assert np.allclose(u.get_transform(comps[i].id)[:3, 3], T[order[i]][:3, 3], atol=1e-5)


def test_split_by_chain_assigns_all_triangles():
    """Joint-guided split covers every triangle exactly once and respects chain order."""
    import trimesh
    m = trimesh.creation.box(extents=(0.05, 0.05, 0.4))
    m.apply_translation([0, 0, 0.2])
    import open3d as o3d
    om = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(m.vertices), o3d.utility.Vector3iVector(m.faces))
    J = np.array([[0, 0, 0.1], [0, 0, 0.2], [0, 0, 0.3]])
    parts = segmenter._split_by_chain(om, J)
    assert sum(len(np.asarray(p.triangles)) for p in parts) == len(m.faces)
    zs = [np.asarray(p.vertices)[:, 2].mean() for p in parts]
    assert zs == sorted(zs)


def test_screw_axis_roundtrip():
    from core.jointscan import _screw
    rng = np.random.default_rng(1)
    for _ in range(20):
        a = rng.normal(size=3); a /= np.linalg.norm(a)
        q = rng.normal(size=3)
        ang = rng.uniform(0.2, 2.5)
        K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        R = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = q - R @ q          # rotation about the line (q, a)
        ang2, a2, q2, d = _screw(T)
        assert abs(ang2 - ang) < 1e-6 and abs(abs(a2 @ a) - 1) < 1e-6 and abs(d) < 1e-6
        v = q - q2
        assert np.linalg.norm(v - (v @ a) * a) < 1e-6           # q2 lies on the same line


@pytest.mark.slow
def test_joint_scans_recover_joints(tmp_path):
    """Rest scan + one scan per joint (random yaw/scale each) -> joints within a few degrees / mm.
    Slow (~5 min): pytest -m slow"""
    import open3d as o3d
    import yourdfpy
    from core import articulated
    demo = load(demo=True)
    comps = segmenter.segment(demo)
    joints = joint_detector.detect(comps, demo)
    out = tmp_path / "u"
    urdf = urdf_builder.build(comps, joints, "demo", out)
    u = _robot(urdf, out)
    rest = np.array([0, 0.6, -0.9, 0.5])
    rng = np.random.default_rng(3)

    def scan(cfg, k):
        placed = _world_meshes(urdf, out, cfg)
        m = o3d.geometry.TriangleMesh()
        for p in placed.values():
            m += o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(p.vertices), o3d.utility.Vector3iVector(p.faces))
        s = 1.0 if k == 0 else rng.uniform(0.7, 1.4)
        a = rng.uniform(0, 6.28)
        T = np.eye(4); T[:3, :3] = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]) * s
        m.transform(T)
        return m
    rest_m = scan(rest, 0)
    scans = []
    for j in range(4):
        c = rest.copy(); c[j] += 0.7 if j != 1 else -0.7
        scans.append(scan(c, j + 1))
    comps2, joints2, res = articulated.build_from_scans(rest_m, scans)
    assert len(comps2) == 5 and len(joints2) == 4
    for j, want in zip(sorted(res.joints, key=lambda j: j.name), (None, None, None, None)):
        assert j.confidence in ("high", "medium")
