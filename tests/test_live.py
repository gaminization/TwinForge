"""Live-capture tests: non-blocking uploads, real progress, error surfacing, joint editing."""
import json, sys, time
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import live


def circle_cams(n, radius=1.0, step_deg=20.0, w=640, h=480, f=500.0):
    """n cameras on a circle at height 0.3 looking at the origin, phone upright."""
    cams = []
    for k in range(n):
        a = np.radians(k * step_deg)
        C = np.array([radius * np.cos(a), 0.3, radius * np.sin(a)])
        z = -C / np.linalg.norm(C)
        yd = -(np.array([0, 1.0, 0]) - (np.array([0, 1.0, 0]) @ z) * z); yd /= np.linalg.norm(yd)
        x = np.cross(yd, z)
        R = np.stack([x, yd, z])
        cams.append({"R": R, "t": -R @ C, "K": np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]]), "wh": (w, h)})
    return cams


def scene_points():
    """robot (blob of 0.4 x 0.4 x 0.5 on the table) + table slab + far wall."""
    rng = np.random.default_rng(1)
    robot = np.stack([rng.uniform(-.15, .15, 800), rng.uniform(0, .5, 800), rng.uniform(-.15, .15, 800)], 1)
    table = np.stack([rng.uniform(-.6, .6, 3000), np.full(3000, -0.001) + rng.normal(0, .002, 3000),
                      rng.uniform(-.6, .6, 3000)], 1)
    wall = np.stack([rng.uniform(-3, 3, 2000), rng.uniform(-1, 2, 2000), np.full(2000, -3.0)], 1)
    pts = np.vstack([robot, table, wall])
    rgb = np.vstack([np.tile([30, 120, 255], (800, 1)), np.tile([90, 90, 90], (3000, 1)),
                     np.tile([220, 200, 180], (2000, 1))]).astype(np.uint8)
    return pts, rgb


class FakeEngine:
    """Registers every photo, puts cameras on a circle, and is deliberately slow."""
    name = "fake"; lost = False; waiting_baseline = False
    model_state = {"state": "ready"}

    def __init__(self, delay=0.15, fail_at=None):
        self.n = 0; self.delay = delay; self.fail_at = fail_at

    @property
    def n_registered(self): return self.n

    def add_paths(self, paths):
        time.sleep(self.delay)
        for _ in paths:
            if self.fail_at is not None and self.n == self.fail_at:
                self.fail_at = None
                raise RuntimeError("boom")
            self.n += 1

    def snapshot(self):
        pts, rgb = scene_points()
        return pts, rgb, circle_cams(self.n)


def _run(tmp_path, engine, n):
    ups = []
    ls = live.LiveScan(tmp_path, engine, ups.append)
    t = time.time()
    for _ in range(n):
        ls.submit(b"x")
    submit_time = time.time() - t
    while ls.n_processed < n:
        time.sleep(0.05)
    time.sleep(0.2)
    return ls, ups, submit_time


def test_submit_never_blocks_on_slow_engine(tmp_path):
    ls, ups, submit_time = _run(tmp_path, FakeEngine(delay=0.3), 12)
    assert submit_time < 0.2                      # 12 uploads accepted instantly although inference is slow
    assert ls.state["n_frames"] == 12 and ls.state["n_registered"] == 12


def test_progress_is_real_and_monotone(tmp_path):
    ls, ups, _ = _run(tmp_path, FakeEngine(delay=0.01), 20)
    p = [u["progress"] for u in ups]
    assert p == sorted(p) and p[0] == 0 and p[-1] >= 85
    assert ls.state["coverage"] >= 0.9 and ls.state["ready"]
    n = np.frombuffer((tmp_path / "preview.bin").read_bytes()[:4], np.uint32)[0]
    assert 400 <= n <= 800                      # robot kept; table + wall dropped


def test_errors_are_surfaced_not_swallowed(tmp_path):
    ls, ups, _ = _run(tmp_path, FakeEngine(delay=0.01, fail_at=1), 3)
    assert any(u["error"] and "boom" in u["error"] for u in ups)
    assert any("Reconstruction error" in u["guidance"] for u in ups)


def test_model_loading_state_reported_and_frames_wait(tmp_path):
    eng = FakeEngine(delay=0.01); eng.model_state = {"state": "loading", "msg": "Downloading VGGT weights: 40%"}
    ups = []
    ls = live.LiveScan(tmp_path, eng, ups.append)
    ls.submit(b"x"); time.sleep(0.8)
    assert ls.n_processed == 0 and "Downloading" in ls.state["guidance"]
    eng.model_state = {"state": "ready"}
    time.sleep(1.0)
    assert ls.n_processed == 1                    # queued frame was kept, not dropped


def test_edit_joint_keeps_geometry(tmp_path):
    import xml.etree.ElementTree as ET
    from core.acquisition import load
    from core.segmenter import segment
    from core.joint_detector import detect
    from core.urdf_builder import build
    from core import urdf_builder
    from core.edit import edit_joint
    scan = load(demo=True); comps = segment(scan); joints = detect(comps, scan_data=scan)
    build(comps, joints, "r", tmp_path)
    rel = urdf_builder.build.relative_joints
    (tmp_path / "meta.json").write_text(json.dumps({"joints": [
        {"id": j.id, "type": j.type, "axis": list(map(float, j.axis)), "origin_xyz": list(map(float, j.origin_xyz)),
         "limit_lower": j.limit_lower, "limit_upper": j.limit_upper} for j in rel]}))
    jid = rel[0].id
    edit_joint(tmp_path, jid, flip=True, limit_lower=-1.0, limit_upper=0.5, pivot_shift=[0.01, 0, 0])
    j = next(e for e in ET.parse(next(tmp_path.glob("*.urdf"))).getroot().findall("joint") if e.get("name") == jid)
    assert float(j.find("limit").get("lower")) == -1.0
    edit_joint(tmp_path, jid, type="fixed")
    j = next(e for e in ET.parse(next(tmp_path.glob("*.urdf"))).getroot().findall("joint") if e.get("name") == jid)
    assert j.get("type") == "fixed" and j.find("limit") is None
    with pytest.raises(KeyError):
        edit_joint(tmp_path, "nope")


def test_foreground_keeps_robot_drops_wall_and_table():
    from core.foreground import select_object
    pts, rgb = scene_points()
    m, info = select_object(pts, circle_cams(18), return_info=True)
    robot, table, wall = m[:800], m[800:3800], m[3800:]
    assert robot.mean() > 0.9, info             # the robot survives
    assert table.mean() < 0.05, info            # the support plane is removed
    assert wall.mean() == 0.0, info             # far background is removed


def test_foreground_never_empties_the_model():
    from core.foreground import select_object
    pts = np.random.default_rng(0).normal(size=(40, 3))
    assert select_object(pts, circle_cams(5)).all()      # too little data: keep everything
