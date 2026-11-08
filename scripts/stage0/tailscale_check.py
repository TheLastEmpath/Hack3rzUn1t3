#!/usr/bin/env python3
"""Tailscale helper: peer status and direct-vs-relayed path check."""
import argparse
import os
import re
import shutil
import subprocess
import sys


def ts(args, timeout=30):
    exe = shutil.which("tailscale")
    app = "/Applications/Tailscale.app/Contents/MacOS/Tailscale"
    if sys.platform == "darwin" and os.path.exists(app):
        exe = app  # the App Store/standalone app CLI; a PATH shim may crash
    if not exe:
        sys.exit("tailscale CLI not found in PATH")
    p = subprocess.run([exe] + args, capture_output=True, text=True, timeout=timeout)
    return p.returncode, (p.stdout + p.stderr).strip()


def classify(ping_output):
    """Return 'direct', 'relayed' or 'unknown' from `tailscale ping` output."""
    # Relayed replies look like "via DERP(nyc)"; direct ones like "via 1.2.3.4:41641".
    pongs = [l for l in ping_output.splitlines() if "pong" in l]
    if not pongs:
        return "unknown"
    last = pongs[-1]
    if re.search(r"via DERP\(", last):
        return "relayed"
    if re.search(r"via \S+:\d+", last):
        return "direct"
    return "unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status", help="run `tailscale status`")
    p = sub.add_parser("ping", help="ping peers and flag relayed vs direct")
    p.add_argument("peers", nargs="+")
    p.add_argument("-c", "--count", type=int, default=10, help="max pings (tailscale ping stops at direct)")
    a = ap.parse_args()
    if a.cmd == "status":
        rc, out = ts(["status"])
        print(out)
        sys.exit(rc)
    bad = 0
    for peer in a.peers:
        rc, out = ts(["ping", "-c", str(a.count), peer], timeout=60)
        kind = classify(out)
        print(f"{peer}: {kind.upper()}")
        if kind != "direct":
            bad += 1
            print(out)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
