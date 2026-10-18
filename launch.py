#!/usr/bin/env python3
"""
TwinForge 3D - ONE command to run the whole system.

    python3 launch.py                 # check setup, start the server, open the live viewer, show the phone QR
    python3 launch.py --check         # only check the setup and exit
    python3 launch.py --backend sfm   # force the CPU engine (no GPU / no VGGT download)
    python3 launch.py --demo          # no phone: build the built-in demo arm and open the viewer
    python3 launch.py --test          # run the test-suite

What it does for you: preflight check (with the exact fix for anything missing) -> HTTPS cert for the phone
camera -> server (+ VGGT background download/load when available) -> opens the live 3D page in your browser
(it shows a QR code for the phone) -> prints the phone URL. Ctrl+C stops everything.
"""
import argparse
import importlib
import logging
import os
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

OK, BAD, WARN = "\033[32m✓\033[0m", "\033[31m✗\033[0m", "\033[33m!\033[0m"


def _try(mod):
    try:
        return importlib.import_module(mod)
    except Exception:
        return None


def preflight(verbose=True) -> dict:
    """Returns {'ok': bool, 'vggt': bool, 'colmap': bool}. Prints what works and how to fix what doesn't."""
    res = {"ok": True, "vggt": False, "colmap": False}

    def line(sym, msg):
        if verbose:
            print(f"  {sym} {msg}")
    if verbose:
        print("\nTwinForge 3D - setup check")
    if sys.version_info < (3, 9):
        line(BAD, f"Python {sys.version.split()[0]} - need 3.9+"); res["ok"] = False
    else:
        line(OK, f"Python {sys.version.split()[0]}")
    for mod, pipname in (("numpy", "numpy"), ("cv2", "opencv-python"), ("open3d", "open3d"), ("scipy", "scipy"),
                         ("trimesh", "trimesh"), ("fastapi", "fastapi"), ("uvicorn", "uvicorn"),
                         ("skimage", "scikit-image"), ("PIL", "pillow")):
        if _try(mod):
            continue
        line(BAD, f"missing '{pipname}'  ->  pip install {pipname}   (or: pip install -r requirements.txt)")
        res["ok"] = False
    if res["ok"]:
        line(OK, "core libraries (numpy, OpenCV, Open3D, FastAPI ...)")
    if _try("multipart") is None:
        line(BAD, "missing 'python-multipart' (photo upload)  ->  pip install python-multipart"); res["ok"] = False
    if _try("pycolmap"):
        res["colmap"] = True; line(OK, "pycolmap (final photo reconstruction)")
    else:
        line(WARN, "pycolmap missing - the FINAL build from photos needs it  ->  pip install pycolmap")
    torch = _try("torch")
    if torch is None:
        line(WARN, "PyTorch not installed - live preview uses the CPU engine (sparse). For dense: see README 'Install'")
    else:
        gpu = torch.cuda.is_available()
        name = torch.cuda.get_device_name(0) if gpu else "no GPU"
        if _try("vggt") and gpu:
            res["vggt"] = True
            line(OK, f"VGGT + CUDA ({name}) - dense live preview; weights (~5 GB) download in the background on first start")
        elif _try("vggt"):
            line(WARN, "VGGT installed but no CUDA GPU - it would be very slow; using the CPU engine")
        else:
            line(WARN, f"torch OK ({name}) but VGGT not installed -> CPU engine. Dense: git clone https://github.com/facebookresearch/vggt && pip install -e vggt")
    if _try("qrcode") is None:
        line(WARN, "qrcode missing - no QR code (type the URL on the phone)  ->  pip install qrcode")
    return res


def _open_browser_when_up(port, path, scheme):
    import socket
    for _ in range(120):
        try:
            socket.create_connection(("127.0.0.1", port), 0.5).close()
            break
        except OSError:
            time.sleep(0.5)
    time.sleep(0.8)
    url = f"{scheme}://localhost:{port}{path}"
    print(f"\n  >> Opening {url}   (first time: browser warns about the self-signed certificate -> Advanced -> Proceed)\n")
    try:
        webbrowser.open(url)
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser(description="Run the whole TwinForge 3D system")
    ap.add_argument("--port", type=int, default=8443)
    ap.add_argument("--backend", choices=["auto", "vggt", "sfm"], default="auto")
    ap.add_argument("--http", action="store_true", help="plain HTTP (phone camera will not work)")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--check", action="store_true", help="only check the setup")
    ap.add_argument("--demo", action="store_true", help="build the built-in demo arm and open the viewer (no phone)")
    ap.add_argument("--test", action="store_true", help="run the test-suite")
    a = ap.parse_args()

    if a.test:
        sys.exit(subprocess.call([sys.executable, "-m", "pytest", "-p", "no:launch_testing", "-q", "tests"], cwd=ROOT))
    pf = preflight()
    if not pf["ok"]:
        print("\nFix the items marked ✗ above, then run again.\n"); sys.exit(1)
    if a.check:
        print("\nSetup OK.\n"); return
    if a.demo:
        rc = subprocess.call([sys.executable, "twinforge.py", "--demo", "--headless"], cwd=ROOT)
        if rc:
            sys.exit(rc)
        print("\nDemo built. Starting the viewer ...")
        a.backend = "sfm"
    if a.backend == "vggt" and not pf["vggt"]:
        print("\n--backend vggt requested but VGGT + a CUDA GPU are not available (see above)."); sys.exit(1)
    if a.backend == "auto":
        a.backend = "vggt" if pf["vggt"] else "sfm"
    print(f"\n  Live engine: {a.backend.upper()}" + ("  (dense, GPU)" if a.backend == "vggt" else "  (CPU, sparse preview)"))

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("twinforge").setLevel(logging.INFO)
    os.environ["TWINFORGE_BACKEND"] = a.backend
    if not a.no_browser:
        path = "/viewer/demo" if a.demo else "/live"
        threading.Thread(target=_open_browser_when_up,
                         args=(a.port if not a.http else (a.port if a.port != 8443 else 8000), path,
                               "http" if a.http else "https"), daemon=True).start()
    import start_server
    sys.argv = ["start_server.py", "--port", str(a.port), "--backend", a.backend] + (["--http"] if a.http else [])
    start_server.main()


if __name__ == "__main__":
    main()
