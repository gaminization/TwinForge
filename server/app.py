"""
TwinForge 3D — Server
Phone uploads frames → server reconstructs incrementally →
sends guidance back to phone → full pipeline when coverage is ready.
"""
from __future__ import annotations
import asyncio, json, logging, time, uuid
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
JOBS_DIR   = SERVER_DIR / "jobs"
JOBS_DIR.mkdir(exist_ok=True)

_jobs:      Dict[str, dict] = {}
_ws:        Dict[str, list] = {}
_reconst:   Dict[str, object] = {}   # job_id → IncrementalReconstructor
_executor   = ThreadPoolExecutor(max_workers=2)
_main_loop  = None

app = FastAPI(title="TwinForge 3D")

@app.on_event("startup")
async def _capture_loop():
    global _main_loop
    _main_loop = asyncio.get_event_loop()

# ── Pages ─────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def pwa():         return FileResponse(STATIC_DIR / "index.html")

@app.get("/viewer/{job_id}", response_class=HTMLResponse)
async def viewer(job_id: str): return FileResponse(STATIC_DIR / "viewer.html")

# ── Job lifecycle ─────────────────────────────────────────────────────────────
@app.post("/jobs/create")
async def create_job():
    job_id  = str(uuid.uuid4())[:8]
    job_dir = JOBS_DIR / job_id
    (job_dir / "input_frames").mkdir(parents=True)
    _jobs[job_id]    = {"status":"waiting","pct":0,"msg":"Ready to scan",
                         "n_frames":0,"guidance":"Start scanning!",
                         "confidence":0.0,"coverage":{},
                         "needs_angles":[],"ready":False}
    _ws[job_id]      = []
    _reconst[job_id] = None          # created on first frame
    return {"job_id": job_id}

# ── Frame upload — incremental reconstruction happens here ────────────────────
@app.post("/jobs/{job_id}/frame")
async def upload_frame(job_id: str, file: UploadFile = File(...)):
    """Single-frame upload.  Returns guidance immediately."""
    if job_id not in _jobs:
        return JSONResponse({"error":"unknown job"}, 404)

    import cv2
    data  = await file.read()
    arr   = np.frombuffer(data, np.uint8)
    img   = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error":"bad image"}, 400)
    img   = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Save to disk
    n   = _jobs[job_id]["n_frames"]
    dest = JOBS_DIR / job_id / "input_frames" / f"frame_{n:04d}.jpg"
    cv2.imwrite(str(dest), cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 85])
    _jobs[job_id]["n_frames"] += 1

    # Lazy-create reconstructor
    if _reconst[job_id] is None:
        try:
            import sys; sys.path.insert(0, str(Path(__file__).parent.parent))
            from core.reconstruction import IncrementalReconstructor
            _reconst[job_id] = IncrementalReconstructor()
        except Exception as e:
            log.warning(f"Reconstructor unavailable: {e}")

    # Run incremental reconstruction in thread
    guidance = {"n_frames":_jobs[job_id]["n_frames"],
                "guidance":"Analysing…","confidence":0.0,
                "needs_angles":[],"ready":False,"coverage":{}}

    if _reconst[job_id] is not None:
        try:
            loop     = asyncio.get_event_loop()
            guidance = await loop.run_in_executor(
                _executor, _reconst[job_id].add_frame, img)
        except Exception as e:
            log.warning(f"Incremental reconstruction: {e}")

    # Update job state
    _jobs[job_id].update({
        "guidance":    guidance.get("guidance","Keep scanning…"),
        "confidence":  guidance.get("confidence", 0.0),
        "coverage":    guidance.get("coverage", {}),
        "needs_angles":guidance.get("needs_angles",[]),
        "ready":       guidance.get("ready", False),
        "n_points":    guidance.get("n_points", 0),
    })

    # Push to WebSocket
    await _broadcast(job_id, {"type":"frame_update", **guidance,
                               "n_frames":_jobs[job_id]["n_frames"]})

    return guidance


@app.post("/jobs/{job_id}/run")
async def run_pipeline(job_id: str, body: dict = None):
    if job_id not in _jobs: return JSONResponse({"error":"unknown job"},404)
    if _jobs[job_id]["status"] in ("running","done"):
        return {"status": _jobs[job_id]["status"]}
    robot_name = (body or {}).get("robot_name","scanned_robot")
    _jobs[job_id]["status"] = "running"
    asyncio.get_event_loop().run_in_executor(
        _executor, _run_pipeline_sync, job_id, robot_name)
    return {"status":"running","job_id":job_id}

@app.get("/jobs/{job_id}/status")
async def status(job_id: str):
    if job_id not in _jobs: return JSONResponse({"error":"unknown job"},404)
    return _jobs[job_id]

# ── WebSocket ─────────────────────────────────────────────────────────────────
@app.websocket("/jobs/{job_id}/ws")
async def ws_endpoint(websocket: WebSocket, job_id: str):
    await websocket.accept()
    _ws.setdefault(job_id, []).append(websocket)
    if job_id in _jobs:
        await websocket.send_json(_jobs[job_id])
    try:
        while True:
            await asyncio.sleep(1)
            try: await websocket.send_json({"type":"ping"})
            except: break
    except WebSocketDisconnect:
        pass
    finally:
        if job_id in _ws and websocket in _ws[job_id]:
            _ws[job_id].remove(websocket)

# ── Results ───────────────────────────────────────────────────────────────────
@app.get("/jobs/{job_id}/urdf")
async def get_urdf(job_id: str):
    urdfs = list((JOBS_DIR/job_id).glob("*.urdf"))
    if not urdfs: return JSONResponse({"error":"not ready"},404)
    return FileResponse(urdfs[0], filename=urdfs[0].name, media_type="application/xml")

@app.get("/jobs/{job_id}/meta")
async def get_meta(job_id: str):
    p = JOBS_DIR/job_id/"meta.json"
    if not p.exists(): return JSONResponse({"error":"not ready"},404)
    return json.loads(p.read_text())

@app.get("/jobs/{job_id}/meshes/{filename}")
async def get_mesh(job_id:str, filename:str):
    p = JOBS_DIR/job_id/"meshes"/filename
    return FileResponse(p, media_type="model/stl") if p.exists() else JSONResponse({"error":"not found"},404)

@app.get("/jobs/{job_id}/frames/{filename}")
async def get_frame(job_id:str, filename:str):
    p = JOBS_DIR/job_id/"frames"/filename
    return FileResponse(p, media_type="image/png") if p.exists() else JSONResponse({"error":"not found"},404)

@app.get("/jobs/{job_id}/frames")
async def list_frames(job_id: str):
    d = JOBS_DIR/job_id/"frames"
    return sorted(p.name for p in d.glob("frame_*.png")) if d.exists() else []

# ── Pipeline worker ───────────────────────────────────────────────────────────
def _progress(job_id, pct, stage, msg):
    _jobs[job_id].update({"status":"running","pct":pct,"stage":stage,"msg":msg})
    if _main_loop:
        asyncio.run_coroutine_threadsafe(
            _broadcast(job_id,{"type":"progress","pct":pct,"stage":stage,"msg":msg}),
            _main_loop)

def _run_pipeline_sync(job_id: str, robot_name: str):
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from core.acquisition    import load, ScanData
    from core.reconstruction import to_open3d_pcd
    from core.segmenter      import segment
    from core.joint_detector import detect
    from core.kinematic_tree import KinematicTreeBuilder
    from core.urdf_builder   import build
    from core.visualizer     import save_frames

    job_dir   = JOBS_DIR / job_id
    input_dir = job_dir / "input_frames"

    try:
        _progress(job_id,5,"loading","Loading captured frames …")

        # Use IncrementalReconstructor result if available
        pcd = None
        rec = _reconst.get(job_id)
        if rec is not None:
            pcd = rec.get_pcd()

        # Fallback: reconstruct from all saved frames
        if pcd is None or len(pcd.points) < 200:
            frames = sorted(input_dir.glob("*.jpg"))
            if frames:
                import cv2
                imgs = []
                for fp in frames:
                    img = cv2.imread(str(fp))
                    if img is not None:
                        imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                if len(imgs) >= 2:
                    from core.reconstruction import reconstruct_from_frames
                    result = reconstruct_from_frames(imgs)
                    pcd    = to_open3d_pcd(result)

        # Final fallback: demo mode
        if pcd is None or len(pcd.points) < 100:
            log.warning("Insufficient frames — using demo arm")
            scan = load(source=None, demo=True, n_poses=3)
        else:
            import open3d as o3d
            scan = type('ScanData',(),{
                'pcd':pcd,'pose_pcds':[],'source':'phone',
                '_demo_link_pcds':None,'_demo_meta':None
            })()

        _progress(job_id,20,"loading",
                  f"{len(scan.pcd.points):,} points reconstructed")

        _progress(job_id,30,"segmenting","Detecting robot components …")
        components = segment(scan)
        _progress(job_id,45,"segmenting",f"Found {len(components)} components")

        _progress(job_id,50,"joints","Detecting joints and axes …")
        joints = detect(components, scan_data=scan)
        _progress(job_id,65,"joints",f"Found {len(joints)} joints")

        _progress(job_id,68,"tree","Building kinematic tree …")
        tree = KinematicTreeBuilder().build(components, joints)

        _progress(job_id,72,"urdf","Generating URDF …")
        urdf_path = build(components, joints, robot_name, job_dir)
        _progress(job_id,82,"urdf","URDF generated")

        _progress(job_id,85,"rendering","Rendering animation …")
        frames_out = save_frames(urdf_path, job_dir/"frames",
                                  components=components, n_frames=36, period=4.0)
        _progress(job_id,95,"rendering",f"{len(frames_out)} frames rendered")

        # Build meta
        meta = {
            "job_id":robot_name,"robot_name":robot_name,
            "urdf_url":f"/jobs/{job_id}/urdf",
            "n_links":len(components),"n_joints":len(joints),
            "links":[{"id":c.id,"label":c.label,"type":c.type,
                      "mesh_url":f"/jobs/{job_id}/meshes/{c.id}.stl",
                      "centroid":c.centroid.tolist(),"extent":c.extent.tolist(),
                      "movable":c.movable} for c in components],
            "joints":[{"id":j.id,"parent":j.parent_link,"child":j.child_link,
                       "type":j.type,"axis":j.axis.tolist(),
                       "origin_xyz":j.origin_xyz.tolist(),
                       "origin_rpy":j.origin_rpy.tolist(),
                       "limit_lower":j.limit_lower,"limit_upper":j.limit_upper,
                       "confidence":j.confidence,
                       "type_source":j.type_source,
                       "axis_source":j.axis_source} for j in joints],
            "animation_frames":len(frames_out),
            "frame_url_template":f"/jobs/{job_id}/frames/frame_{{:04d}}.png",
        }
        (job_dir/"meta.json").write_text(json.dumps(meta,indent=2))

        _jobs[job_id].update({"status":"done","pct":100,
                               "stage":"done","msg":"Complete!","meta":meta})
        if _main_loop:
            asyncio.run_coroutine_threadsafe(
                _broadcast(job_id,{"type":"complete",**meta}), _main_loop)
        log.info(f"Job {job_id} done: {len(components)} links, {len(joints)} joints")

    except Exception as e:
        import traceback
        log.error(f"Job {job_id} failed:\n{traceback.format_exc()}")
        _jobs[job_id].update({"status":"error","msg":str(e)})
        if _main_loop:
            asyncio.run_coroutine_threadsafe(
                _broadcast(job_id,{"type":"error","msg":str(e)}),_main_loop)

async def _broadcast(job_id: str, msg: dict):
    dead = []
    for ws in _ws.get(job_id,[]):
        try:   await ws.send_json(msg)
        except: dead.append(ws)
    for ws in dead:
        _ws[job_id].remove(ws)

def run_server(host="0.0.0.0", port=8000):
    uvicorn.run(app, host=host, port=port, log_level="info")
