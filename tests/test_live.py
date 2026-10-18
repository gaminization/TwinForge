"""Live-capture tests: non-blocking uploads, real progress, error surfacing, joint editing."""
import json, sys, time
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import live


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
        rng = np.random.default_rng(0)
        pts = rng.normal(size=(500, 3)) * 0.1
        k = np.arange(self.n)
        ang = np.radians(k * 20.0)
        C = np.stack([np.cos(ang), np.zeros_like(ang), np.sin(ang)], 1)
        Y = np.tile([0.0, 1.0, 0.0], (self.n, 1))       # camera "down" = +y  -> up = -y
        return pts, np.full((500, 3), 128, np.uint8), C, Y


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
    assert (tmp_path / "preview.bin").stat().st_size == 4 + 500 * 15


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
