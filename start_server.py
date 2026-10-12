#!/usr/bin/env python3
"""
TwinForge 3D — Server Launcher

Usage:
    python start_server.py             # default port 8000
    python start_server.py --port 8080
    python start_server.py --demo      # open browser with demo job

Opens the server on all network interfaces so your phone can connect over WiFi.
Prints the local IP address and a QR code to scan from your phone.
"""

import argparse
import logging
import os
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

# ── Project root on path ─────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO,
                    format="[%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("twinforge")


def get_local_ip() -> str:
    """Best-effort local network IP (the one phones on the same WiFi can reach)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


def print_qr(url: str):
    """Print a QR code to the terminal (uses qrcode if available)."""
    try:
        import qrcode
        qr = qrcode.QRCode(border=1)
        qr.add_data(url)
        qr.make(fit=True)
        qr.print_ascii(invert=True)
    except ImportError:
        print("  (pip install qrcode to show a scannable QR code)")


def print_banner(host: str, port: int):
    ip  = get_local_ip()
    url = f"http://{ip}:{port}"
    width = 62

    print()
    print("=" * width)
    print("  TwinForge 3D — Server Running")
    print("=" * width)
    print(f"  Local:   http://localhost:{port}")
    print(f"  Network: {url}   ← open on your phone")
    print()
    print("  PHONE SETUP:")
    print(f"  1. Connect phone to the SAME Wi-Fi as this PC")
    print(f"  2. Scan the QR code below OR type: {url}")
    print(f"  3. Tap 'Start Scanning' and walk around your robot")
    print()
    print("-" * width)
    print_qr(url)
    print("-" * width)
    print()
    print("  3D VIEWER  (PC browser):")
    print(f"  http://localhost:{port}/viewer/demo")
    print()
    print("  Press Ctrl+C to stop")
    print("=" * width)
    print()


def open_demo_job(port: int):
    """Pre-process a demo job so the 3D viewer is immediately ready."""
    import time as _time
    _time.sleep(1.5)   # wait for server to boot

    try:
        import requests

        # Create job
        r = requests.post(f"http://localhost:{port}/jobs/create")
        job_id = r.json()["job_id"]

        # Start pipeline (no frames → demo mode)
        requests.post(f"http://localhost:{port}/jobs/{job_id}/run",
                      json={"robot_name": "demo_arm"})

        log.info(f"Demo job started: {job_id}")
        log.info(f"3D viewer: http://localhost:{port}/viewer/{job_id}")

        # Open browser
        webbrowser.open(f"http://localhost:{port}/viewer/{job_id}")

    except Exception as e:
        log.debug(f"Demo auto-launch skipped: {e}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host",  default="0.0.0.0")
    p.add_argument("--port",  type=int, default=8000)
    p.add_argument("--demo",  action="store_true",
                   help="Auto-run demo job and open browser")
    args = p.parse_args()

    print_banner(args.host, args.port)

    if args.demo:
        t = threading.Thread(target=open_demo_job, args=(args.port,), daemon=True)
        t.start()

    # Start FastAPI server
    from server.app import run_server
    run_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
