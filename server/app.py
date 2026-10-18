"""TwinForge 3D — FastAPI server (HTTP + WebSocket)."""
from __future__ import annotations
import asyncio, json, logging, os, uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict
import numpy as np

from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
import uvicorn

log = logging.getLogger("twinforge.server")

SERVER_DIR = Path(__file__).parent
STATIC_DIR = SERVER_DIR / "static"
from fastapi.staticfiles import StaticFiles
JOBS_DIR   = SERVER_DIR / "jobs"
JOBS_DIR.mkdir(exist_ok=True)

_jobs:    Dict[str, dict]   = {}
_ws:      Dict[str, list]   = {}
_live:    Dict[str, Dict[int, object]] = {}
_executor = ThreadPoolExecutor(max_workers=2)
_loop     = None
PROJECT_ROOT = Path(__file__).parent.parent

import sys; sys.path.insert(0, str(PROJECT_ROOT))
from core.live import ModelHost
_host = ModelHost()
app = FastAPI(title="TwinForge 3D")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.on_event("startup")
async def _startup():
    global _loop
    _loop = asyncio.get_event_loop()
    import sys; sys.path.insert(0, str(PROJECT_ROOT))
    if _backend_choice() == "vggt":
        _host.start()          # background download/load; the phone page shows its progress

@app.get("/", response_class=HTMLResponse)
async def pwa(): return FileResponse(STATIC_DIR / "index.html")

@app.get("/viewer/{job_id}", response_class=HTMLResponse)
async def viewer(job_id: str): return FileResponse(STATIC_DIR / "viewer.html")

@app.get("/live/{job_id}", response_class=HTMLResponse)
async def live_viewer(job_id: str): return FileResponse(STATIC_DIR / "viewer.html")

@app.get("/live", response_class=HTMLResponse)
async def live_latest(): return FileResponse(STATIC_DIR / "viewer.html")

@app.get("/jobs/latest")
async def latest_job():
    return {"job_id": next(reversed(_jobs), None)}

@app.get("/system")
async def system():
    """What the server is using; the phone page polls this before the first capture."""
    return {"backend": _backend_choice(), "model": _host.state if _backend_choice() == "vggt"
            else {"state": "ready", "msg": "CPU incremental SfM (no model download)", "pct": 100}}

@app.post("/jobs/create")
async def create_job():
    jid = str(uuid.uuid4())[:8]
    (JOBS_DIR / jid / "scans").mkdir(parents=True)
    _jobs[jid]    = {"status":"waiting","pct":0,"msg":"Ready","n_frames":0,
                      "guidance":"Point camera at robot and tap Capture",
                      "coverage":0.0,"progress":0,"ready":False}
    _ws[jid]      = []
    _live[jid]    = {}
    log.warning("New scan job %s - live 3D view: /live/%s  (or just open /live)", jid, jid)
    print(f"\n>>> New scan job {jid}. Live 3D view: open /live on this PC (or /live/{jid})\n", flush=True)
    return {"job_id": jid}

def _backend_choice() -> str:
    import sys; sys.path.insert(0, str(PROJECT_ROOT))
    want = os.environ.get("TWINFORGE_BACKEND", "auto").lower()
    from core.live import ModelHost
    if want == "sfm":
        return "sfm"
    if want in ("vggt", "auto") and ModelHost.available():
        return "vggt"
    return "sfm"

def _get_live(jid: str, scan: int):
    from core.live import LiveScan, SfMEngine, VGGTEngine
    scans = _live.setdefault(jid, {})
    if scan not in scans:
        d = JOBS_DIR / jid / "scans" / f"scan_{scan:02d}"
        eng = VGGTEngine(d, _host) if _backend_choice() == "vggt" else SfMEngine(d)
        if _backend_choice() == "vggt":
            _host.start()
        def on_update(state, jid=jid, scan=scan):
            st = dict(state, scan=scan)
            _jobs[jid].update({k: v for k, v in st.items() if k != "type"})
            if _loop:
                asyncio.run_coroutine_threadsafe(_broadcast(jid, st), _loop)
        scans[scan] = LiveScan(d, eng, on_update, label=f"scan {scan}")
    return scans[scan]

@app.post("/jobs/{jid}/frame")
async def upload_frame(jid: str, file: UploadFile = File(...), scan: int = 0):
    """Accept one photo and return immediately; the live worker thread does the reconstruction."""
    if jid not in _jobs:
        return JSONResponse({"error":"unknown job"}, status_code=404)
    if not 0 <= scan <= 32:
        return JSONResponse({"error": "scan index out of range"}, status_code=400)
    data = await file.read()
    try:
        ls = _get_live(jid, scan)
    except Exception as e:
        log.exception("live init failed")
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=500)
    ls.submit(data)
    return ls.state

@app.get("/jobs/{jid}/preview")
async def preview(jid: str, scan: int = 0):
    """Binary live model: uint32 n | float32 xyz[n*3] | uint8 rgb[n*3], display frame (y up, ~1 unit)."""
    from fastapi.responses import Response
    ls = _live.get(jid, {}).get(scan)
    if not ls:
        return Response(b"", media_type="application/octet-stream")
    return Response(ls.preview_bytes(), media_type="application/octet-stream",
                    headers={"Cache-Control": "no-store"})

@app.post("/jobs/{jid}/joints/{joint_id}")
async def edit_joint_ep(jid: str, joint_id: str, body: dict):
    """Post-scan correction: change a joint's type/axis/limits/pivot without rescanning."""
    from core.edit import edit_joint
    try:
        mj = edit_joint(JOBS_DIR / jid, joint_id, **{k: body[k] for k in
                        ("type", "axis", "flip", "limit_lower", "limit_upper", "pivot_shift") if k in body})
    except (KeyError, ValueError, StopIteration) as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    return mj

@app.post("/jobs/{jid}/run")
async def run_job(jid: str, body: dict = None):
    if jid not in _jobs:
        return JSONResponse({"error":"unknown job"}, status_code=404)
    if _jobs[jid]["status"] in ("running","done"):
        return {"status": _jobs[jid]["status"]}
    robot_name = (body or {}).get("robot_name", "scanned_robot")
    _jobs[jid]["status"] = "running"
    if (body or {}).get("articulated"):
        asyncio.get_event_loop().run_in_executor(
            _executor, _run_articulated_sync, jid, robot_name, (body or {}).get("size_m"))
        return {"status": "running", "job_id": jid, "mode": "articulated"}
    asyncio.get_event_loop().run_in_executor(
        _executor, _run_sync, jid, robot_name, (body or {}).get("size_m"))
    return {"status": "running", "job_id": jid}

@app.get("/jobs/{jid}/status")
async def get_status(jid: str):
    if jid not in _jobs:
        return JSONResponse({"error":"unknown job"}, status_code=404)
    return _jobs[jid]

@app.websocket("/jobs/{jid}/ws")
async def ws_ep(ws: WebSocket, jid: str):
    await ws.accept()
    _ws.setdefault(jid, []).append(ws)
    if jid in _jobs:
        await ws.send_json(_jobs[jid])
    try:
        while True:
            await asyncio.sleep(1)
            try:
                await ws.send_json({"type":"ping"})
            except:
                break
    except WebSocketDisconnect:
        pass
    finally:
        if jid in _ws and ws in _ws[jid]:
            _ws[jid].remove(ws)

@app.get("/jobs/{jid}/urdf")
async def get_urdf(jid: str):
    urdfs = list((JOBS_DIR/jid).glob("*.urdf"))
    if not urdfs:
        return JSONResponse({"error":"not ready"}, status_code=404)
    return FileResponse(urdfs[0], filename=urdfs[0].name,
                        media_type="application/xml")

@app.get("/jobs/{jid}/meta")
async def get_meta(jid: str):
    p = (Path(__file__).parent.parent / "output" / "meta.json") if jid == "demo" \
        else JOBS_DIR / jid / "meta.json"
    if not p.exists():
        return JSONResponse({"error":"not ready"}, status_code=404)
    return json.loads(p.read_text())

@app.get("/jobs/{jid}/meshes/{fn}")
async def get_mesh(jid: str, fn: str):
    if jid == "demo":      # generic route would otherwise shadow /jobs/demo/meshes/{fn}
        p = Path(__file__).parent.parent / "output" / "meshes" / Path(fn).name
    else:
        p = JOBS_DIR / jid / "meshes" / Path(fn).name      # Path().name blocks ../ traversal
    if p.exists():
        return FileResponse(str(p), media_type="model/stl")
    return JSONResponse({"error":"not found"}, status_code=404)

@app.get("/jobs/{jid}/frames/{fn}")
async def get_frame(jid: str, fn: str):
    p = JOBS_DIR / jid / "frames" / fn
    if p.exists():
        return FileResponse(str(p), media_type="image/png")
    return JSONResponse({"error":"not found"}, status_code=404)

@app.get("/jobs/{jid}/frames")
async def list_frames(jid: str):
    d = JOBS_DIR / jid / "frames"
    return sorted(p.name for p in d.glob("frame_*.png")) if d.exists() else []

# Serve demo output meshes (created by twinforge.py --demo)
PROJECT_ROOT = Path(__file__).parent.parent
@app.get("/jobs/demo/meshes/{fn}")
async def demo_mesh(fn: str):
    p = PROJECT_ROOT / "output" / "meshes" / fn
    if p.exists():
        return FileResponse(str(p), media_type="model/stl")
    return JSONResponse({"error":"not found"}, status_code=404)


# ── Pipeline worker ───────────────────────────────────────────────────────────

def _prog(jid, pct, stage, msg):
    _jobs[jid].update({"pct": pct, "stage": stage, "msg": msg})
    if _loop:
        asyncio.run_coroutine_threadsafe(
            _broadcast(jid, {"type":"progress","pct":pct,"stage":stage,"msg":msg}),
            _loop)

def _run_sync(jid: str, robot_name: str, size_m=None):
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from core.acquisition    import load
    from core.segmenter      import segment
    from core.joint_detector import detect
    from core.kinematic_tree import KinematicTreeBuilder
    from core.urdf_builder   import build
    from core import urdf_builder
    from core.visualizer     import save_frames

    jdir = JOBS_DIR / jid

    try:
        _prog(jid, 5, "loading", "Loading captured frames…")

        scan = None
        fdir = jdir / "scans" / "scan_00" / "frames"
        saved_imgs = sorted(fdir.glob("*.jpg")) if fdir.exists() else []
        if len(saved_imgs) >= 6:
            # Final model: the heavier offline SfM + dense stereo + TSDF pipeline over every saved photo.
            import open3d as o3d
            from core.articulated import load_mesh
            _prog(jid, 10, "reconstruction", f"Dense reconstruction from {len(saved_imgs)} photos (several minutes on CPU)...")
            work = jdir / "work"; work.mkdir(exist_ok=True)
            mesh = load_mesh(fdir, work, size_m=size_m)
            pcd = o3d.geometry.PointCloud(mesh.vertices)
            if mesh.has_vertex_colors():
                pcd.colors = mesh.vertex_colors
            mesh.compute_vertex_normals(); pcd.normals = mesh.vertex_normals
            if len(pcd.points) >= 500:
                _prog(jid, 20, "reconstruction", f"Reconstructed {len(pcd.points):,} points")
                scan = _make_scan_from_pcd(pcd)
        if scan is None:
            if len(saved_imgs) > 0:
                raise RuntimeError(
                    f"Only {len(saved_imgs)} photos - capture at least 6 overlapping views all around the robot.")
            _prog(jid, 15, "loading", "No photos uploaded - using the built-in demo arm")
            scan = load(demo=True)

        _prog(jid, 15, "loading", f"{len(scan.pcd.points):,} points loaded")

        _prog(jid, 30, "segmenting", "Segmenting robot components…")
        comps = segment(scan)
        _prog(jid, 45, "segmenting", f"Found {len(comps)} component(s)")

        _prog(jid, 50, "joints", "Detecting joints from geometry…")
        joints = detect(comps, scan_data=scan)
        _prog(jid, 65, "joints", f"Found {len(joints)} joint(s)")

        _prog(jid, 68, "tree", "Building kinematic chain…")
        ktb = KinematicTreeBuilder(comps, joints)
        ktb.assign_depths()

        _prog(jid, 72, "urdf", "Generating URDF + STL meshes…")
        urdf_path = build(comps, joints, robot_name, jdir)

        _prog(jid, 82, "rendering", "Rendering animation frames…")
        frames = save_frames(urdf_path, jdir / "frames",
                             components=comps, n_frames=36)
        _prog(jid, 95, "rendering", f"{len(frames)} frames rendered")

        # Build mesh list (only files that actually exist)
        mesh_dir = jdir / "meshes"
        mesh_list = []
        for c in comps:
            stl = mesh_dir / f"{c.id}.stl"
            if stl.exists():
                mesh_list.append({
                    "id":   c.id,
                    "label": c.label,
                    "mesh_url": f"/jobs/{jid}/meshes/{c.id}.stl",
                })

        meta = {
            "mode": "single_scan_heuristic",
            "job_id":    jid,
            "robot_name": robot_name,
            "urdf_url":  f"/jobs/{jid}/urdf",
            "n_links":   len(comps),
            "n_joints":  len(joints),
            "n_frames":  len(frames),
            "frame_url_template": f"/jobs/{jid}/frames/frame_{{:04d}}.png",
            "links":  mesh_list,
            "joints": [{
                "id":          j.id,
                "parent":      j.parent_id,
                "child":       j.child_id,
                "type":        j.type,
                "origin_xyz":  [float(v) for v in j.origin_xyz],
                "origin_rpy":  [float(v) for v in j.origin_rpy],
                "axis":        j.axis.tolist(),
                "limit_lower": j.limit_lower,
                "limit_upper": j.limit_upper,
                "confidence":  j.confidence,
            } for j in urdf_builder.build.relative_joints],
        }
        (jdir / "meta.json").write_text(json.dumps(meta, indent=2))

        _jobs[jid].update({"status":"done","pct":100,"msg":"Complete!","meta":meta})
        if _loop:
            asyncio.run_coroutine_threadsafe(
                _broadcast(jid, {"type":"complete", **meta}), _loop)

    except Exception as e:
        import traceback
        traceback.print_exc()
        _jobs[jid].update({"status":"error","msg":str(e)})
        if _loop:
            asyncio.run_coroutine_threadsafe(
                _broadcast(jid, {"type":"error","msg":str(e)}), _loop)


def _run_articulated_sync(jid: str, robot_name: str, size_m=None):
    """Rest scan + one scan per joint -> automatic joints -> URDF."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core.articulated import load_mesh, build_from_scans
    from core.urdf_builder import build
    from core import urdf_builder
    jdir = JOBS_DIR / jid
    try:
        dirs = sorted(p for p in (jdir / "scans").glob("scan_*") if p.is_dir())
        if len(dirs) < 3:
            raise RuntimeError("Need the rest scan (scan 0) plus at least two joint scans (scan 1, 2, ...)")
        for d in dirs:
            if len(list((d / "frames").glob("*.jpg"))) < 20:
                raise RuntimeError(f"{d.name} has only {len(list((d / 'frames').glob('*.jpg')))} photos; capture 40-80 around the robot")
        work = jdir / "work"; work.mkdir(exist_ok=True)
        meshes = []
        for k, d in enumerate(dirs):
            _prog(jid, 5 + int(60 * k / len(dirs)), "reconstruction", f"Reconstructing {d.name} ({k + 1}/{len(dirs)})...")
            meshes.append(load_mesh(d / "frames", work / d.name, size_m=size_m if k == 0 else None))
        _prog(jid, 70, "joints", "Detecting links and joints from the motion between scans...")
        comps, joints, res = build_from_scans(meshes[0], meshes[1:])
        _prog(jid, 85, "urdf", "Generating URDF + STL meshes...")
        urdf_path = build(comps, joints, robot_name, jdir)
        meta = {
            "mode": "joint_scans",
            "job_id": jid, "robot_name": robot_name, "urdf_url": f"/jobs/{jid}/urdf",
            "n_links": len(comps), "n_joints": len(joints), "n_frames": 0,
            "links": [{"id": c.id, "label": c.label, "mesh_url": f"/jobs/{jid}/meshes/{c.id}.stl"}
                      for c in comps if (jdir / "meshes" / f"{c.id}.stl").exists()],
            "joints": [{"id": j.id, "parent": j.parent_id, "child": j.child_id, "type": j.type,
                        "origin_xyz": [float(v) for v in j.origin_xyz], "origin_rpy": [float(v) for v in j.origin_rpy],
                        "axis": j.axis.tolist(), "limit_lower": j.limit_lower, "limit_upper": j.limit_upper,
                        "confidence": j.confidence} for j in urdf_builder.build.relative_joints],
        }
        (jdir / "meta.json").write_text(json.dumps(meta, indent=2))
        _jobs[jid].update({"status": "done", "pct": 100, "msg": "Complete!", "meta": meta})
        if _loop:
            asyncio.run_coroutine_threadsafe(_broadcast(jid, {"type": "complete", **meta}), _loop)
    except Exception as e:
        import traceback
        traceback.print_exc()
        _jobs[jid].update({"status": "error", "msg": str(e)})
        if _loop:
            asyncio.run_coroutine_threadsafe(_broadcast(jid, {"type": "error", "msg": str(e)}), _loop)


def _make_scan_from_pcd(pcd):
    """Wrap a raw PointCloud into the ScanData duck-type."""
    class ScanData:
        def __init__(self, p):
            self.pcd = p
            self.source = "phone"
            self._demo_link_meta = None
    return ScanData(pcd)


async def _broadcast(jid: str, msg: dict):
    dead = []
    for ws in _ws.get(jid, []):
        try:
            await ws.send_json(msg)
        except:
            dead.append(ws)
    for ws in dead:
        _ws[jid].remove(ws)

def run_server(host="0.0.0.0", port=8000,
               ssl_keyfile=None, ssl_certfile=None):
    uvicorn.run(app, host=host, port=port, log_level="info",
                ssl_keyfile=ssl_keyfile, ssl_certfile=ssl_certfile)
