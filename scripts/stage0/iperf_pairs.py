#!/usr/bin/env python3
"""iperf3 pairwise bandwidth runner.

  server              run `iperf3 -s` on this node
  run --peers A B C   run a client test against each peer, print Mbit/s
"""
import argparse
import json
import shutil
import subprocess
import sys


def need_iperf():
    exe = shutil.which("iperf3")
    if not exe:
        sys.exit("iperf3 not found. Install it (e.g. `brew install iperf3`).")
    return exe


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("server")
    r = sub.add_parser("run")
    r.add_argument("--peers", nargs="+", required=True)
    r.add_argument("--seconds", type=int, default=10)
    r.add_argument("--streams", type=int, default=4)
    r.add_argument("--port", type=int, default=5201)
    a = ap.parse_args()
    exe = need_iperf()
    if a.cmd == "server":
        sys.exit(subprocess.call([exe, "-s"]))
    results = {}
    for peer in a.peers:
        for direction, extra in (("up", []), ("down", ["-R"])):
            cmd = [exe, "-c", peer, "-p", str(a.port), "-t", str(a.seconds),
                   "-P", str(a.streams), "-J"] + extra
            p = subprocess.run(cmd, capture_output=True, text=True)
            try:
                bps = json.loads(p.stdout)["end"]["sum_received"]["bits_per_second"]
                results[f"{peer}/{direction}"] = round(bps / 1e6, 1)
            except (ValueError, KeyError):
                results[f"{peer}/{direction}"] = None
                print(f"{peer} {direction}: failed: {(p.stderr or p.stdout).strip()[:200]}", file=sys.stderr)
    print(json.dumps({"mbit_per_s": results}, indent=2))


if __name__ == "__main__":
    main()
