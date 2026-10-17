"""TwinForge 3D — FastAPI server (HTTP + WebSocket)."""
from __future__ import annotations
import asyncio, json, logging, uuid
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
_reconst: Dict[str, object] = {}
_executor = ThreadPoolExecutor(max_workers=2)
_loop     = None

app = FastAPI(title="TwinForge 3D")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.on_event("startup")
async def _startup():
    global _loop
    _loop = asyncio.get_event_loop()

@app.get("/", response_class=HTMLResponse)
async def pwa(): return FileResponse(STATIC_DIR / "index.html")

@app.get("/viewer/{job_id}", response_class=HTMLResponse)
async def viewer(job_id: str): return FileResponse(STATIC_DIR / "viewer.html")

@app.post("/jobs/create")
async def create_job():
    jid = str(uuid.uuid4())[:8]
    (JOBS_DIR / jid / "input_frames").mkdir(parents=True)
    _jobs[jid]    = {"status":"waiting","pct":0,"msg":"Ready","n_frames":0,
                      "guidance":"Point camera at robot and tap Capture",
                      "coverage":0.0,"ready":False}
    _ws[jid]      = []
    _reconst[jid] = None
    return {"job_id": jid}

@app.post("/jobs/{jid}/frame")
async def upload_frame(jid: str, file: UploadFile = File(...), scan: int = None):
    if jid not in _jobs:
        return JSONResponse({"error":"unknown job"}, status_code=404)

    data = await file.read()

    if scan is not None:
        # Multi-scan capture: scan 0 = rest pose, scan i = only joint i moved.  Photos are stored per
        # scan and reconstructed together when /run is called with {"articulated": true}.
        if not 0 <= scan <= 32:
            return JSONResponse({"error": "scan index out of range"}, status_code=400)
        d = JOBS_DIR / jid / "scans" / f"scan_{scan:02d}"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"frame_{len(list(d.glob('*.jpg'))):04d}.jpg").write_bytes(data)
        counts = {p.name: len(list(p.glob("*.jpg"))) for p in sorted((JOBS_DIR / jid / "scans").iterdir())}
        _jobs[jid]["scan_counts"] = counts
        return {"scan": scan, "scan_counts": counts}

    # Save raw frame to disk
    n    = _jobs[jid]["n_frames"]
    dest = JOBS_DIR / jid / "input_frames" / f"frame_{n:04d}.jpg"
    dest.write_bytes(data)
    _jobs[jid]["n_frames"] += 1

    # Lazy-init reconstructor
    if _reconst[jid] is None:
        try:
            import sys; sys.path.insert(0, str(Path(__file__).parent.parent))
            from core.reconstruction import IncrementalReconstructor
            _reconst[jid] = IncrementalReconstructor(JOBS_DIR / jid)
        except Exception as e:
            log.warning(f"Reconstructor init failed: {e}")

    guidance_dict = {
        "n_frames": _jobs[jid]["n_frames"],
        "guidance": "Analysing…",
        "coverage": 0.0,
        "covered_sectors": [],
        "ready": False,
    }

    if _reconst[jid]:
        try:
            loop   = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                _executor, _reconst[jid].add_frame, data)
            guidance_dict = {
                "n_frames":       _jobs[jid]["n_frames"],
                "guidance":       result.guidance,
                "coverage":       result.coverage,
                "covered_sectors":result.covered_sectors,
                "ready":          result.coverage >= 0.75,
            }
        except Exception as e:
            log.warning(f"Incremental recon error: {e}")

    _jobs[jid].update(guidance_dict)
    await _broadcast(jid, {"type": "frame_update", **guidance_dict})
    return guidance_dict

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
        _executor, _run_sync, jid, robot_name)
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

def _run_sync(jid: str, robot_name: str):
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
    inp  = jdir / "input_frames"

    try:
        _prog(jid, 5, "loading", "Loading captured frames…")

        scan = None
        saved_imgs = sorted(inp.glob("*.jpg"))

        # Use phone reconstruction if enough frames were captured
        if len(saved_imgs) >= 4 and _reconst.get(jid):
            try:
                import open3d as o3d
                _prog(jid, 10, "reconstruction", "Running SfM reconstruction…")
                pcd = _reconst[jid].get_pcd()
                if len(pcd.points) >= 500:
                    log.info(f"Using phone reconstruction: {len(pcd.points):,} points")
                    _prog(jid, 20, "reconstruction",
                          f"Reconstructed {len(pcd.points):,} points")
                    scan = _make_scan_from_pcd(pcd)
            except Exception as e:
                log.warning(f"Phone reconstruction failed: {e}")

        if scan is None:
            if len(saved_imgs) > 0:
                # Never silently swap in the demo arm for a user's own scan.
                try:
                    n_pts = len(_reconst[jid].get_pcd().points) if _reconst.get(jid) else 0
                except Exception:
                    n_pts = 0
                backend = getattr(getattr(_reconst.get(jid), "_impl", None), "__class__", type(None)).__name__
                raise RuntimeError(
                    f"Reconstruction produced only {n_pts} points from {len(saved_imgs)} photos "
                    f"({backend}) - not enough to build a model. Capture more overlapping views "
                    f"all around the robot, or install VGGT/DUSt3R for dense reconstruction.")
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
            if len(list(d.glob("*.jpg"))) < 25:
                raise RuntimeError(f"{d.name} has only {len(list(d.glob('*.jpg')))} photos; capture 50-80 around the robot")
        work = jdir / "work"; work.mkdir(exist_ok=True)
        meshes = []
        for k, d in enumerate(dirs):
            _prog(jid, 5 + int(60 * k / len(dirs)), "reconstruction", f"Reconstructing {d.name} ({k + 1}/{len(dirs)})...")
            meshes.append(load_mesh(d, work, size_m=size_m if k == 0 else None))
        _prog(jid, 70, "joints", "Detecting links and joints from the motion between scans...")
        comps, joints, res = build_from_scans(meshes[0], meshes[1:])
        _prog(jid, 85, "urdf", "Generating URDF + STL meshes...")
        urdf_path = build(comps, joints, robot_name, jdir)
        meta = {
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
