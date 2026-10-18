#!/usr/bin/env python3
"""
TwinForge 3D — Start Server
============================
Generates a self-signed SSL cert so phone camera works (HTTPS required),
then launches the FastAPI server with QR code printed to terminal.

Usage:
    python3 start_server.py [--port 8443] [--http]
"""
import argparse, os, socket, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


def get_local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def gen_ssl_cert(cert_path: Path, key_path: Path, ip: str):
    """Generate a self-signed cert valid for the local IP."""
    try:
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.hazmat.backends import default_backend
        import datetime, ipaddress

        key = rsa.generate_private_key(
            public_exponent=65537, key_size=2048,
            backend=default_backend())

        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, "TwinForge3D"),
        ])
        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.datetime.utcnow())
            .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=365))
            .add_extension(x509.SubjectAlternativeName([
                x509.DNSName("localhost"),
                x509.IPAddress(ipaddress.IPv4Address(ip)),
                x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
            ]), critical=False)
            .sign(key, hashes.SHA256(), default_backend())
        )

        key_path.write_bytes(key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption()))
        cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
        return True

    except ImportError:
        # Fallback: openssl command
        try:
            subprocess.run([
                "openssl", "req", "-x509", "-newkey", "rsa:2048",
                "-keyout", str(key_path), "-out", str(cert_path),
                "-days", "365", "-nodes", "-subj",
                f"/CN=TwinForge3D",
                "-addext", f"subjectAltName=IP:{ip},IP:127.0.0.1,DNS:localhost"
            ], check=True, capture_output=True)
            return True
        except Exception as e:
            print(f"  [warn] Could not generate SSL cert: {e}")
            return False


def print_qr(url: str):
    try:
        import qrcode
        qr = qrcode.QRCode(border=1)
        qr.add_data(url)
        qr.make(fit=True)
        qr.print_ascii(invert=True)
    except ImportError:
        print(f"  (install qrcode for QR: pip install qrcode)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8443)
    parser.add_argument("--backend", choices=["auto", "vggt", "sfm"], default="auto",
                        help="live reconstruction engine: vggt (GPU, dense), sfm (CPU, sparse), auto")
    parser.add_argument("--http", action="store_true",
                        help="HTTP only (camera won't work on phone)")
    args = parser.parse_args()

    ip   = get_local_ip()
    port = args.port
    use_ssl = not args.http

    ssl_cert = ssl_key = None
    cert_dir = ROOT / ".ssl"
    cert_dir.mkdir(exist_ok=True)
    cert_path = cert_dir / "cert.pem"
    key_path  = cert_dir / "key.pem"

    if use_ssl:
        print("  Generating self-signed SSL cert…", end=" ", flush=True)
        ok = gen_ssl_cert(cert_path, key_path, ip)
        if ok:
            ssl_cert = str(cert_path)
            ssl_key  = str(key_path)
            print("done ✓")
        else:
            use_ssl = False
            port    = args.port if args.port != 8443 else 8000
            print("skipped — running HTTP (camera may not work)")

    scheme = "https" if use_ssl else "http"
    net_url = f"{scheme}://{ip}:{port}"

    print()
    print("=" * 60)
    print("  TwinForge 3D — Server")
    print("=" * 60)
    print(f"  Local:   {scheme}://localhost:{port}")
    print(f"  Network: {net_url}   ← open on your phone")
    if use_ssl:
        print()
        print("  ⚠️  PHONE SETUP (one-time only):")
        print(f"  1. Open {net_url} on your phone")
        print("  2. Tap 'Advanced' → 'Proceed to site' (it's safe — self-signed)")
        print("  3. Camera will work after you accept the warning")
    print()
    print("  Scan this QR code with your phone:")
    print("-" * 60)
    print_qr(net_url)
    print("-" * 60)
    print(f"  3D Viewer (PC): {scheme}://localhost:{port}/viewer/demo   (live view opens automatically per scan: /live/<job>)")
    print()
    print("  Press Ctrl+C to stop")
    print("=" * 60)
    print()

    os.environ["TWINFORGE_BACKEND"] = args.backend
    from server.app import run_server
    run_server(host="0.0.0.0", port=port,
               ssl_keyfile=ssl_key, ssl_certfile=ssl_cert)


if __name__ == "__main__":
    main()
