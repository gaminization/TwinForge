# TwinForge 3D

Point a phone at a robot arm, walk around it, and watch a 3D model of it grow on your PC screen photo by photo.
Then TwinForge turns it into a **URDF digital twin** (one mesh per link, real joint axes) that you can load in a
simulator, with a built-in viewer to **audit and correct** the joints.

Reference robot used for development: https://github.com/smart-methods/arduino_robot_arm

---------------------------------------------------------------------------------------------------------------

## 1. How it works (the 60-second version)

```
 phone browser                       PC (this repo: start_server.py)                      PC browser
 ─────────────                       ───────────────────────────────                      ──────────
 tap Capture / Auto  ──photo──►  POST /jobs/<id>/frame   (returns in ms, never blocks)
                                   │ queue
                                   ▼
                          LIVE WORKER THREAD  (core/live.py)
                          1. add the photo to the 3D model      (SfM: localise + triangulate | VGGT: re-solve)
                          2. progress = 60 % orbit coverage + 40 % registered views
                          3. write preview point cloud
                                   │ WebSocket frame_update
   ring / % / guidance ◄───────────┤───────────────────────────────►  /live/<id>  growing coloured point cloud,
                                   │                                    camera positions, progress bar, stats
 tap Finish ───────────►  POST /jobs/<id>/run
                          FINAL BUILD (offline, slower, better): dense stereo + TSDF → mesh → segment links →
                          detect joints → URDF + STL meshes
                                   │
                                   ▼
                          /viewer/<id>   articulated model + JOINT AUDIT panel (pose-test, flip axis, limits,
                                          move pivot, "not a joint") → saved straight into the URDF
```

Two stages, on purpose:

| Stage | When | What | Quality |
|---|---|---|---|
| **Live** (`core/live.py`) | while you capture | every photo immediately adds to a coloured point cloud you watch grow | preview: sparse (SfM) or dense (VGGT); arbitrary scale, no bundle adjustment |
| **Final** (`core/photo_recon.py`, `core/articulated.py`) | after you tap Finish | all saved photos → COLMAP SfM → dense stereo → TSDF → mesh → links/joints → URDF | the model you export |

The progress ring is **real**: 60 % = how many of 12 orbit directions around the robot have a *registered* camera
(a photo only counts if it was actually localised in the model), 40 % = registered views (target 24). It can only
go up. Photos that fail to register do not count, and the guidance line says "Tracking lost" instead of pretending.

### Live engines

* **`sfm` (CPU, always works, no download)** – real incremental Structure-from-Motion: SIFT features → match to the
  most recent views + keyframes → PnP-RANSAC pose of the new photo against the existing map → triangulate new
  points → map grows. Needs textured surfaces and shots ~5–15° apart. Sparse cloud, no bundle adjustment, so it
  drifts on long sequences – it is a *preview*, the final build re-solves everything.
* **`vggt` (GPU, dense)** – facebook/VGGT-1B. Weights (~5 GB) download in a **background thread when the server starts**;
  the phone shows "Downloading VGGT weights: 43 %" and photos queue until it is ready (nothing is lost, nothing blocks).
  Each new photo re-runs VGGT on the newest + evenly spaced earlier views (max 16), so the whole cloud refines.
* `auto` (default) uses VGGT if `torch` + `vggt` import, otherwise SfM. Force with `--backend vggt|sfm`.

---------------------------------------------------------------------------------------------------------------

## 2. Install

Python 3.10+, Linux/macOS/Windows. GPU optional (live VGGT preview needs one; ~8 GB VRAM is enough for 16 views).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# optional dense live preview (RTX 50-series needs the cu128 wheels):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
git clone https://github.com/facebookresearch/vggt && pip install -e vggt
```

Self-test (≈1 min): `pytest -p no:launch_testing -q tests`
(`-p no:launch_testing` avoids a crash when a ROS install puts its `launch_testing` pytest plugin on your path.)

## 3. Run it

```bash
python3 start_server.py                 # HTTPS on :8443, prints a QR code. --backend sfm|vggt|auto  --port N
```
1. Open the printed `https://<pc-ip>:8443` on the phone (same Wi-Fi). Accept the self-signed certificate warning
   (*Advanced → Proceed*); the camera only works over HTTPS.
2. On the PC open `https://localhost:8443/live` – it waits for the phone to start a scan and then follows the
   newest job automatically (the job id is also printed in the server log; `/live/<job-id>` works directly, and the
   phone's **👁 Live 3D** button shows the same page on the phone).
3. Tap **Start scan**, then **Capture** (or **Auto**: one photo every 1.2 s) while walking slowly around the robot.
4. When the ring is green and the pill says *Ready*, tap **Finish**. Or add joint scans (next section).

### How to photograph (this decides the quality)
* Walk a full circle at ~1 m, camera pointing at the robot, a second circle higher up if you can; keep the robot centred.
* Move **sideways** (translate) – don't stand still and rotate. First two photos need ~10 cm of sideways motion.
* Shots ≤ 10–15° apart, sharp (no motion blur), even light, robot on a textured/patterned surface (newspaper, a
  printed grid) – plain white plastic on a plain table is the hardest case for any photogrammetry.
* Don't move the robot or table during a scan.

## 4. Getting joints automatically: rest scan + one scan per joint

A single static scan cannot show how a robot moves. TwinForge therefore scans the arm in several poses and
**measures** the joints from the motion (no manual joint entry):

1. **Scan 0 – rest pose** (any bent pose; don't stretch the arm straight up along a joint axis). Full circle.
2. Tap **+ Joint scan**. Turn **one** joint by 30–60°, change nothing else, circle again. Repeat for each joint.
3. Tap **Finish**. Each scan is reconstructed; every joint scan is aligned to the rest scan on the base; what did not
   move cancels, what remains is one rigid body whose screw motion gives the axis, pivot and angle; nested moved
   sets give the links and the kinematic tree.

The phone labels the current scan ("Scan 2: move ONE joint ~40°"), each scan gets its own live preview
(`/live/<id>?scan=2`), and **Finish** after joint scans runs the articulated pipeline automatically.
CLI equivalent for photo folders: `python3 twinforge.py --rest r/ --joint-scans j1/ j2/ j3/ --size-m 0.322 --name arm`.

Without joint scans, Finish uses the older single-scan heuristics (segmentation + geometric joint guess): fine for a
first look, not trustworthy for kinematics.

## 5. Audit and correct (after the scan)

Open `/viewer/<job-id>` (opens by itself when the build finishes). The left panel lists every joint with its
confidence (`high / medium / low` – low = recapture suggested). Select one:

| Control | Use |
|---|---|
| **Pose test** slider | rotate/slide the joint to check it moves the right part the right way (yellow arrow = axis) |
| **Flip axis** / axis boxes | fix a reversed or tilted axis |
| **Limits** | joint limits in rad (defaults are a guess around the observed motion) |
| **pivot x/y/z ±** | move the rotation centre in 5 mm steps *without moving the geometry* |
| **Not a joint** | make it `fixed` (a false detection) |
| **Save to URDF** | writes into `jobs/<id>/*.urdf` and `meta.json` (`POST /jobs/<id>/joints/<joint>`) |

Not possible in the editor: *adding* a joint that was never detected, or re-splitting a link. Fix those by
rescanning with that joint moved.

## 6. Outputs

`server/jobs/<id>/`: `<name>.urdf`, `meshes/<link>.stl` (link-local frames, joint origins relative to the parent – loads
in RViz / PyBullet / MuJoCo / Isaac), `meta.json`, `scans/scan_NN/frames/*.jpg` (your photos), `scans/scan_NN/preview.bin`.

## 7. HTTP / WebSocket API

| Endpoint | Purpose |
|---|---|
| `GET /system` | backend + model download state |
| `POST /jobs/create` → `{job_id}` | new job |
| `POST /jobs/<id>/frame?scan=N` (multipart `file`) | queue a photo (instant) → current state |
| `GET /jobs/<id>/status` | latest state: `progress, n_frames, n_registered, n_points, coverage, guidance, ready, lost, error, model` |
| `GET /jobs/<id>/preview?scan=N` | binary cloud: `uint32 n, float32 xyz[3n], uint8 rgb[3n]` (y-up, ~1 unit) |
| `WS /jobs/<id>/ws` | `frame_update`, `progress`, `complete`, `error` |
| `POST /jobs/<id>/run` `{"articulated":bool,"size_m":m}` | final build |
| `GET /jobs/<id>/urdf`, `/meta`, `/meshes/<f>` | results |
| `POST /jobs/<id>/joints/<joint>` | audit edit |

## 8. Project layout

```
start_server.py        HTTPS server + QR          twinforge.py        CLI (demo / photos / scans)
server/app.py          FastAPI, WebSocket          server/static/index.html   phone capture page
core/live.py           live engines + worker       server/static/viewer.html  live view + result + joint audit
core/photo_recon.py    offline SfM+dense+TSDF      core/jointscan.py   joints from rest+joint scans
core/articulated.py    scans → links/joints        core/edit.py        post-scan joint editing
core/urdf_builder.py   URDF + STL writer           eval/               synthetic benchmarks (see eval/README.md)
```

## 9. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Ring stuck at 0 % (old versions) | fixed: UI read a field the server never sent, and the first photo blocked the server while VGGT downloaded |
| "Downloading VGGT weights…" for minutes | first run only (~5 GB, cached in `~/.cache/huggingface`); photos queue meanwhile. Use `--backend sfm` to skip it |
| "Move the phone sideways…" | need parallax: translate the phone ~10 cm between the first two shots |
| "Tracking lost…" | shots too far apart / blur / blank surface: go back to the last good spot, shoot closer together |
| `Reconstruction error: …` pill | the real exception is shown (and in the server log) instead of being swallowed |
| Camera won't open on phone | must use `https://` and accept the certificate |
| `pytest` crashes in `launch_testing` | `pytest -p no:launch_testing -q tests` |
| Port busy | `fuser -k 8443/tcp` |

## 10. What is and isn't verified

All numbers are on SYNTHETIC textured renders of the arduino_robot_arm (ground truth known); see `eval/README.md`.
**Real phone photographs and real VGGT weights have not been tested by the developer** (no weights/phone access in the
build environment) – the VGGT engine is written against the real VGGT API but only the SfM engine was run end to end.

Live flow (SfM engine, sandbox, 36 photos of the rest-pose arm at 10° steps, uploaded over HTTP):
* each upload returns in ~4 ms; progress, registered views, points and coverage rise per photo (0 → 95 %, 1 750 points,
  28/36 views registered; 8 were flagged "tracking lost" and not counted). A headless-browser screenshot of `/live/<id>`
  shows the arm's shape in the growing cloud. The same job then ran the full path (Finish → dense build → URDF, 6 links /
  5 joints, viewer + joint edit saved to the URDF) - but with only 36 photos in one scan the single-scan mesh was rough and
  the joints are heuristic guesses (the viewer labels them `guess`); use the rest + joint-scan protocol for real kinematics. At 15° steps only ~half the views register (the tool says so).

Automatic joints (rest + one scan per joint, 72 photos/scan, SfM + dense stereo, CPU): 5 scans → 4 joints found, right
types and tree; angles 38.7 / 40.0 / 35.2° for a true 40.1°; axis error 1.1–6.1°, pivot error 2–12 mm on a 322 mm robot;
URDF vs original over 6 poses: Chamfer 6.4 mm mean. Noisy-point-cloud repeat test: a shoulder-type joint failed at 3 mm
noise (flagged `low`). With 36 photos/scan one wrist axis was ~25° off (flagged `low`).

Known limits: plain untextured plastic reconstructs poorly; yaw on a rotation-symmetric base is a heuristic (vertical
axis through the base centre is exact, angle/split height are not); prismatic joints coarse; surfaces ~5 mm rough with
holes (visual + kinematic twin, not CAD); masses/inertias are placeholders; the SfM live preview drifts on long sequences.
