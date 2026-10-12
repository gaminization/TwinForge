#!/usr/bin/env python3
"""
TwinForge 3D — single entry point

Usage examples
--------------
# Demo (no hardware): generates synthetic 4-DOF arm, shows it moving
python twinforge.py --demo

# From a .ply / .pcd scan
python twinforge.py --input scan.ply --name my_arm

# From a folder of RGB-D image pairs
python twinforge.py --input scan_folder/ --name my_arm

# From a video file (phone scan)
python twinforge.py --input robot_video.mp4 --name my_arm

# Headless (saves PNG frames + MP4, no window)
python twinforge.py --demo --headless --output output/
"""

import argparse
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("twinforge")


def run(args):
    from core.acquisition   import load
    from core.segmenter     import segment
    from core.joint_detector import detect
    from core.urdf_builder  import build
    from core.visualizer    import run_interactive, save_frames

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # ── 1. Acquire ────────────────────────────────────────────────────────
    t0 = time.time()
    log.info("Step 1/5 — Acquiring scan data …")
    scan = load(
        source=args.input,
        demo=args.demo,
        n_poses=args.poses,
    )
    log.info(f"  → {len(scan.pcd.points):,} points  ({time.time()-t0:.1f}s)")

    # ── 2. Segment ────────────────────────────────────────────────────────
    t1 = time.time()
    log.info("Step 2/5 — Segmenting components …")
    components = segment(
        scan,
        voxel=args.voxel,
        eps=args.eps,
        min_pts=args.min_pts,
    )
    log.info(f"  → {len(components)} component(s)  ({time.time()-t1:.1f}s)")
    for c in components:
        log.info(f"     {c.id:10s}  {c.type:14s}  pts={c.n_points:5d}  "
                 f"extent={c.extent.round(3)}")

    if len(components) < 2:
        log.error("Need ≥2 components to build a robot. "
                  "Try lowering --eps or --min-pts, or use --demo.")
        sys.exit(1)

    # ── 3. Detect joints ──────────────────────────────────────────────────
    t2 = time.time()
    log.info("Step 3/5 — Detecting joints …")
    joints = detect(components, scan_data=scan)
    log.info(f"  → {len(joints)} joint(s)  ({time.time()-t2:.1f}s)")

    if not joints:
        log.error("No joints detected. Try --demo to verify the pipeline.")
        sys.exit(1)

    # ── 4. Build URDF ─────────────────────────────────────────────────────
    t3 = time.time()
    log.info("Step 4/5 — Building URDF …")
    urdf_path = build(
        components=components,
        joints=joints,
        robot_name=args.name,
        output_dir=out,
    )
    log.info(f"  → {urdf_path}  ({time.time()-t3:.1f}s)")

    # Print summary
    _print_summary(components, joints, urdf_path, time.time()-t0)

    # ── 5. Visualise ──────────────────────────────────────────────────────
    log.info("Step 5/5 — Launching visualiser …")
    if args.headless:
        frames_dir = out / "frames"
        save_frames(
            urdf_path=urdf_path,
            output_dir=frames_dir,
            components=components,
            n_frames=args.frames,
            period=args.period,
        )
        log.info(f"Frames saved to {frames_dir}/")
        log.info("Run without --headless to open the interactive viewer.")
    else:
        run_interactive(
            urdf_path=urdf_path,
            components=components,
            n_frames=args.frames,
            period=args.period,
        )

    return urdf_path


def _print_summary(components, joints, urdf_path, elapsed):
    width = 60
    print(f"\n{'='*width}")
    print(f"  TwinForge 3D — Complete  ({elapsed:.1f}s)")
    print(f"{'='*width}")
    print(f"  URDF:    {urdf_path}")
    print(f"  Meshes:  {urdf_path.parent / 'meshes'}")
    print()
    print(f"  Components ({len(components)}):")
    for c in components:
        print(f"    {c.id:12s}  {c.type:14s}  movable={c.movable}")
    print()
    print(f"  Joints ({len(joints)}):")
    for j in joints:
        print(f"    {j.id}")
        print(f"      type={j.type}  axis={j.axis.round(3)}")
        print(f"      limits=[{j.limit_lower:.3f}, {j.limit_upper:.3f}]  "
              f"conf={j.confidence:.2f}")
    print()
    print(f"  To open in RViz (requires ROS 2):")
    print(f"    ros2 launch {urdf_path.parent}/launch_rviz.py")
    print(f"{'='*width}\n")


def main():
    p = argparse.ArgumentParser(
        description="TwinForge 3D — scan a robot, get a URDF, see it move",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--demo",  action="store_true",
                     help="Run with synthetic 4-DOF arm (no hardware needed)")
    grp.add_argument("--input", metavar="PATH",
                     help=".ply / .pcd / image-folder / video file")

    # Robot
    p.add_argument("--name",   default="robot",
                   help="Robot name (used in URDF and output filenames)")
    p.add_argument("--output", default="output",
                   help="Output directory  [default: output/]")
    p.add_argument("--poses",  type=int, default=3,
                   help="Number of poses to capture for motion-based axis fitting"
                        "  [default: 3]")

    # Segmentation tuning
    p.add_argument("--voxel",   type=float, default=0.005,
                   help="Voxel size for downsampling  [default: 0.005 m]")
    p.add_argument("--eps",     type=float, default=0.025,
                   help="DBSCAN neighbourhood radius  [default: 0.025 m]")
    p.add_argument("--min-pts", type=int,   default=80, dest="min_pts",
                   help="DBSCAN min points per cluster  [default: 80]")

    # Visualiser
    p.add_argument("--headless", action="store_true",
                   help="Save frames instead of opening a window")
    p.add_argument("--frames",  type=int,   default=300,
                   help="Animation frames  [default: 300]")
    p.add_argument("--period",  type=float, default=4.0,
                   help="Animation period in seconds  [default: 4.0]")

    args = p.parse_args()

    if not args.demo and not args.input:
        p.print_help()
        print("\nTip: run  python twinforge.py --demo  to try it right now.\n")
        sys.exit(0)

    run(args)


if __name__ == "__main__":
    main()
