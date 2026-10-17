# TwinForge 3D

Photos / point cloud of a robot -> segmented links -> URDF (+ per-link STL meshes) you can load in a simulator.

## Quick start
```bash
pip install -r requirements.txt
python3 twinforge.py --demo                      # arduino arm from its real STL files (no GPU)
pytest -q tests                                  # validates URDF frames + articulation
python3 start_server.py                          # phone capture + web viewer (HTTPS)
```

## Automatic joints: rest scan + one scan per joint (recommended)
A single static scan cannot reveal joint axes, but a few scans can. Capture the robot (same table, same
camera routine, ~70 photos each) in these poses:

1. **rest pose**: any natural *bent* pose (do not stretch the arm straight up along a joint axis);
2. **one scan per joint**: the rest pose with ONLY that joint turned by 30-60 degrees (everything else unchanged).
   Keep the arm above the table in every pose.

```bash
python3 twinforge.py --rest rest_photos/ --joint-scans j1_photos/ j2_photos/ j3_photos/ j4_photos/ \
                     --size-m 0.322 --name myrobot
```
Each argument may be a photo folder or a mesh (`.ply`). `--size-m` is the real length of the rest scan's longest side
(only the rest scan needs a scale: the others are scaled automatically from the part that did not move).
What happens: every joint scan is aligned to the rest scan on the base, what did not move cancels, what is left is one
rigid body whose screw motion gives the joint axis, position and angle; nested moved-sets give the kinematic tree and links.
Joint limits are defaults around the observed motion: edit them if you know the real ones.
Known limits: a yaw joint on a rotation-symmetric base is detected as a vertical axis through the base centre, its
angle and the base/waist split are heuristics; sliding (prismatic) joints are only handled coarsely.

## One scan only (joints must be given by you)
```bash
# 1) photos -> dense cloud (VGGT if installed, else DUSt3R, else sparse SIFT)
python3 twinforge.py --images photos/ --links 5 --size-m 0.40 --name myrobot
# 2) open output/joints_template.json, correct each joint's axis / xyz / limits, then
python3 twinforge.py --images photos/ --links 5 --size-m 0.40 --joints output/joints_template.json
```
Or start from a ready scan: `--input scan.ply`.

* `--links N`   number of rigid links to split the mesh into (use the real count)
* `--size-m M`  real length of the object's longest side in metres (neural/SfM scans have arbitrary scale)
* `--joints F`  user-confirmed joints; **required for a trustworthy URDF** (see below)

Install VGGT (needs HuggingFace access for the 1B weights, ~5 GB):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # RTX 50-series
git clone https://github.com/facebookresearch/vggt && pip install -e vggt
```

## Photo pipeline without neural weights (`--backend sfm`)
```bash
pip install pycolmap scikit-image torch
python twinforge.py --images photos/ --backend sfm --size-m 0.454 --links 5 --joints my_joints.json --name myrobot --headless
```
pycolmap SfM -> GrabCut masks -> visual hull -> plane-sweep stereo -> TSDF fusion -> mesh -> metric scale
(`--size-m` = true height of the robot in metres; wrong value = wrong scale) -> joint-guided segmentation -> URDF.
`--joints` takes a chain file (`{"joints":[{"name","type","xyz","axis","lower","upper"},...]}`, world coordinates,
base to tip); run once without it to get `output/joints_template.json` and edit it.
Note: Open3D 0.20 `ScalableTSDFVolume` returned empty meshes in testing; `UniformTSDFVolume` is used.

## What is and isn't verified
All numbers below are on SYNTHETIC textured renders of the arduino_robot_arm (ground truth known), reproducible
with `eval/README.md`. Real phone photos have not been tested.

Automatic joints (rest scan + one scan per joint, 72 photos per scan, SfM + dense stereo, CPU):
* 5 scans -> 4 joints found, all with the right type and kinematic tree; joint angles measured 38.7 / 40.0 / 35.2 deg
  for a true 40.1 deg. Axis error 1.1-6.1 deg, axis position error 2-12 mm (322 mm robot).
* URDF built from those photo scans vs the original robot over 6 poses: Chamfer 6.4 mm mean (6.1 mm at the rest
  pose, so articulation adds ~0.3 mm), 64% of surface within 5 mm; the residual is reconstruction roughness, not joints.
* Clean meshes with random yaw/scale per scan: Chamfer 1.3 mm, 99% / 99% precision / recall within 5 mm.
* Point-cloud level repeat test, 1.5 mm noise, 5 seeds: 20/20 joints right, axis <= 5.2 deg, position <= 5 mm.
  At 3 mm noise the shoulder-type joint (where most of the robot moves, so little static part to align on) failed in
  3 of 3 seeds - the tool flags it `low` confidence and prints a recapture warning; the other joints stay accurate.
* Web server job flow with photos per scan (`/jobs/{id}/frame?scan=k`, `/run {"articulated": true}`) completes and serves
  the URDF + meshes; with only 36 photos per scan the wrist axis came out ~25 deg off and was flagged low confidence.
* Viewer (headless Chromium) renders the photo-derived URDF and articulates it.
* `pytest -q tests` (7 fast tests) and `pytest -m slow` (rest + 4 joint scans, ~6 min) pass.

NOT verified / known limits:
* VGGT / DUSt3R reconstruction quality (weights could not be downloaded where this was developed).
* Real photographs: plain plastic with little texture will reconstruct worse than the textured renders.
* Yaw joint on a rotation-symmetric base: axis = vertical through the base centre is exact, but its angle and the
  base/waist split height are heuristics. Prismatic joints are handled coarsely. Joint limits are defaults.
* Surfaces are rough (~5 mm) and have holes: this is a usable visual + kinematic twin, not CAD-clean geometry.
  Masses/inertias are bounding-box placeholders.
* The single-scan route (`--images ... --joints`) still needs the joints from you.
