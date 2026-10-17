#!/usr/bin/env python3
"""
TwinForge 3D — CLI
Usage:
  python3 twinforge.py --demo
  python3 twinforge.py --input scan.ply --name my_robot --links 5
  python3 twinforge.py --images photos/ --links 5 --joints joints.json
"""
import argparse, logging, sys, time
from pathlib import Path

logging.basicConfig(level=logging.INFO,
    format="[%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("twinforge")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


def run_articulated(args, out, t0):
    """Rest scan + one scan per joint -> automatically detected joints -> URDF."""
    import json
    from core.articulated import load_mesh, build_from_scans
    from core.urdf_builder import build
    from core import urdf_builder
    import open3d as o3d
    work = out / "work"; work.mkdir(exist_ok=True)
    log.info("Step 1/4 - Reconstructing rest scan and %d joint scans ...", len(args.joint_scans))
    rest = load_mesh(args.rest, work, size_m=args.size_m)
    scans = [load_mesh(s, work) for s in args.joint_scans]
    log.info("Step 2/4 - Detecting links and joints from the motion between scans ...")
    comps, joints, res = build_from_scans(rest, scans)
    o3d.io.write_triangle_mesh(str(out / "scan_mesh.ply"), rest)
    log.info("Step 3/4 - Building URDF ...")
    urdf_path = build(comps, joints, args.name, out)
    (out / "meta.json").write_text(json.dumps({
        "robot_name": args.name, "urdf_url": "/jobs/demo/urdf", "n_links": len(comps), "n_joints": len(joints),
        "links": [{"id": c.id, "label": c.label, "mesh_url": f"/jobs/demo/meshes/{c.id}.stl"} for c in comps],
        "joints": [{"id": j.id, "parent": j.parent_id, "child": j.child_id, "type": j.type,
                    "origin_xyz": [float(v) for v in j.origin_xyz], "origin_rpy": [float(v) for v in j.origin_rpy],
                    "axis": j.axis.tolist(), "limit_lower": j.limit_lower, "limit_upper": j.limit_upper,
                    "confidence": j.confidence} for j in urdf_builder.build.relative_joints]}, indent=2))
    print("\nURDF:", urdf_path, " (%.0fs)" % (time.time() - t0))
    for j in res.joints:
        print(f"  {j.name}: {j.parent}->{j.child} {j.type} axis={j.axis.round(3)} observed={j.observed:.2f} "
              f"rms={j.rms_mm:.1f}mm confidence={j.confidence}")
    low = [j.name for j in res.joints if j.confidence == "low"]
    if low:
        print(f"  WARNING: low confidence for {', '.join(low)} - recapture those scans (rotate the joint 30-60 deg, "
              "keep everything else still, cover the robot from all sides) before trusting the URDF.")
    print("  NOTE: joint limits are defaults around the observed motion; edit them if you know the real ones.")


def run(args):
    from core.acquisition    import load
    from core.segmenter      import segment
    from core.joint_detector import detect
    from core.kinematic_tree import KinematicTreeBuilder
    from core.urdf_builder   import build
    from core.visualizer     import save_frames, show_interactive

    out = ROOT / "output"
    out.mkdir(exist_ok=True)
    t0 = time.time()

    if args.rest:
        return run_articulated(args, out, t0)

    log.info("Step 1/5 — Acquiring scan data …")
    t = time.time()
    scan = load(source=args.input or args.images, demo=args.demo,
                backend=args.backend, size_m=args.size_m)
    if args.size_m and not args.demo:
        import numpy as np
        ext = np.asarray(scan.pcd.get_max_bound()) - np.asarray(scan.pcd.get_min_bound())
        k = args.size_m / float(ext.max())
        scan.pcd.scale(k, center=(0, 0, 0))
        log.info(f"  Rescaled so the longest side = {args.size_m} m (x{k:.4g})")
    if args.images:
        import open3d as o3d
        o3d.io.write_point_cloud(str(out / "scan.ply"), scan.pcd)
        if getattr(scan, "mesh", None) is not None:
            o3d.io.write_triangle_mesh(str(out / "scan_mesh.ply"), scan.mesh)
        log.info(f"  Saved reconstructed scan -> {out/'scan_mesh.ply'} (reuse with --input)")
    log.info(f"  → {len(scan.pcd.points):,} points  ({time.time()-t:.1f}s)")

    log.info("Step 2/5 — Segmenting components …")
    t = time.time()
    comps = segment(scan, n_links=args.links, joints_path=args.joints)
    log.info(f"  → {len(comps)} component(s)  ({time.time()-t:.1f}s)")
    for c in comps:
        log.info(f"     {c.id:<12} {c.label:<16} pts={len(c.pcd.points):>5}  "
                 f"extent={c.extent}")

    log.info("Step 3/5 — Detecting joints …")
    t = time.time()
    joints = detect(comps, scan_data=scan, overrides_path=args.joints)
    if not args.demo and not args.joints:
        from core.joint_detector import write_template
        write_template(joints, out / 'joints_template.json')
        log.warning('Joint axes/limits from a static scan are GUESSES. Edit output/joints_template.json and rerun with --joints to confirm them.')
    log.info(f"  → {len(joints)} joint(s)  ({time.time()-t:.1f}s)")

    log.info("Step 4/5 — Building URDF …")
    t = time.time()
    urdf_path = build(comps, joints, args.name, out)
    log.info(f"  → {urdf_path}  ({time.time()-t:.1f}s)")

    # Metadata for the web viewer (server serves it at /jobs/demo/meta)
    import json
    from core import urdf_builder
    (out / "meta.json").write_text(json.dumps({
        "robot_name": args.name, "urdf_url": "/jobs/demo/urdf",
        "n_links": len(comps), "n_joints": len(joints),
        "links": [{"id": c.id, "label": c.label, "mesh_url": f"/jobs/demo/meshes/{c.id}.stl"} for c in comps],
        "joints": [{"id": j.id, "parent": j.parent_id, "child": j.child_id, "type": j.type,
                    "origin_xyz": [float(v) for v in j.origin_xyz], "origin_rpy": [float(v) for v in j.origin_rpy],
                    "axis": j.axis.tolist(), "limit_lower": j.limit_lower, "limit_upper": j.limit_upper,
                    "confidence": j.confidence} for j in urdf_builder.build.relative_joints]}, indent=2))

    total = time.time() - t0
    print()
    print("=" * 60)
    print(f"  TwinForge 3D — Complete  ({total:.1f}s)")
    print("=" * 60)
    print(f"  URDF:    {urdf_path}")
    print(f"  Meshes:  {out}/meshes")
    print()
    print(f"  Components ({len(comps)}):")
    for c in comps:
        print(f"    {c.id:<12} {c.label:<16} movable={c.movable}")
    print()
    print(f"  Joints ({len(joints)}):")
    for j in joints:
        print(f"    {j.id}")
        print(f"      type={j.type}  axis={j.axis.round(3)}")
        print(f"      limits=[{j.limit_lower:.3f}, {j.limit_upper:.3f}]  "
              f"conf={j.confidence:.2f}")
    print()
    print("  To launch server:")
    print("    python3 start_server.py")
    print("=" * 60)
    print()

    if not args.no_viewer:
        log.info("Step 5/5 — Rendering animation …")
        frames_dir = out / "frames"
        frames = save_frames(urdf_path, frames_dir, components=comps, n_frames=36)
        log.info(f"  → {len(frames)} frames saved to {frames_dir}")

        if not args.headless:
            try:
                show_interactive(comps, joints)
            except Exception as e:
                log.warning(f"Interactive viewer unavailable: {e}")


def main():
    p = argparse.ArgumentParser(description="TwinForge 3D — Robot Digital Twin Generator")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--demo",  action="store_true", help="Use built-in demo arm")
    src.add_argument("--input", metavar="FILE",       help="PLY/PCD scan file")
    src.add_argument("--images", metavar="DIR",       help="Folder of photos to reconstruct (VGGT/DUSt3R/SIFT)")
    p.add_argument("--rest", metavar="DIR_OR_PLY", help="Rest-pose scan (mesh or photo folder); use with --joint-scans")
    p.add_argument("--joint-scans", nargs="+", metavar="DIR_OR_PLY", default=[],
                   help="One scan per joint, in any order, each with ONLY that joint moved >=20 degrees")
    p.add_argument("--backend", choices=["auto", "neural", "sfm", "sift"], default="auto",
                   help="Photo reconstruction backend (see README)")
    p.add_argument("--links",  type=int, default=6,   help="Number of rigid links to segment into")
    p.add_argument("--size-m", type=float, default=None, help="Real longest dimension of the object in metres (fixes scale of neural/SfM scans)")
    p.add_argument("--joints", metavar="JSON",        help="User-confirmed joints (see output/joints_template.json)")
    p.add_argument("--name",       default="robot",  help="Robot name")
    p.add_argument("--headless",   action="store_true")
    p.add_argument("--no-viewer",  action="store_true")
    args = p.parse_args()
    if not args.demo and not args.input and not args.images and not args.rest:
        args.demo = True
    if args.rest and not args.joint_scans:
        p.error('--rest needs --joint-scans')
    run(args)


if __name__ == "__main__":
    main()
