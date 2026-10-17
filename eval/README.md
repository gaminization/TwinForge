# Reproducing the evaluation (synthetic ground truth)

Needs the arduino_robot_arm repo (`git clone https://github.com/smart-methods/arduino_robot_arm`, set `ARM_PKG=<path>/robot_arm_pkg`),
Playwright + Chromium, pycolmap, torch.

```bash
cd eval
python3 -m http.server 8777 &                       # serves render2.html + world_*.stl
python3 gen_poses.py                                 # pose_R/K1..K4.json (rest pose + one joint moved each), gt_joints.json
for p in R K1 K2 K3 K4; do python3 shoot_pose.py $p; done      # 72 photos per scan -> photos_<p>/
for p in R K1 K2 K3 K4; do python3 recon_pose.py $p; done      # SfM + dense stereo -> scans/<p>.ply  (~10-15 min each, CPU)
python3 ../twinforge.py --rest scans/R.ply --joint-scans scans/K1.ply scans/K2.ply scans/K3.ply scans/K4.ply --size-m 0.322 --name photo_auto
python3 eval_art.py ../output photo_auto.urdf        # Chamfer vs original robot over random poses + joint errors
python3 js_eval.py 0.0015 0                          # point-cloud-level test: noise (m), seed; repeat for seeds 0-4
```
