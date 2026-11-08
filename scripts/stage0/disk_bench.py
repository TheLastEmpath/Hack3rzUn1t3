#!/usr/bin/env python3
"""Sequential-read benchmark. Uses fio if present, else a plain Python read."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time


def python_read(path, size, block):
    with open(path, "wb") as f:
        chunk = os.urandom(block)
        for _ in range(size // block):
            f.write(chunk)
        f.flush()
        os.fsync(f.fileno())
    t = time.perf_counter()
    done = 0
    with open(path, "rb", buffering=0) as f:
        while True:
            b = f.read(block)
            if not b:
                break
            done += len(b)
    dt = time.perf_counter() - t
    return done / dt / 1e6


def fio_read(fio, path, size, block):
    direct = "0" if sys.platform == "darwin" else "1"
    cmd = [fio, "--name=seqread", f"--filename={path}", f"--size={size}", f"--bs={block}",
           "--rw=read", f"--direct={direct}", "--ioengine=" + ("posixaio" if sys.platform == "darwin" else "libaio"),
           "--iodepth=16", "--output-format=json"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return json.loads(p.stdout)["jobs"][0]["read"]["bw_bytes"] / 1e6
    except (ValueError, KeyError, IndexError):
        sys.exit(f"fio failed: {(p.stderr or p.stdout).strip()[:300]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default=".", help="directory on the disk to test")
    ap.add_argument("--size-gb", type=float, default=2.0)
    ap.add_argument("--block-mb", type=int, default=4)
    ap.add_argument("--no-fio", action="store_true", help="force the Python fallback")
    a = ap.parse_args()
    size = int(a.size_gb * 1024**3)
    block = a.block_mb * 1024**2
    size -= size % block
    free = shutil.disk_usage(a.dir).free
    if size * 2 > free:
        sys.exit("not enough free space for the test file")
    path = os.path.join(a.dir, "fio-testfile.bin")
    fio = None if a.no_fio else shutil.which("fio")
    try:
        if fio:
            # fio lays out the file itself on first run (write pass), then reads.
            method = "fio"
            subprocess.run([fio, "--name=prep", f"--filename={path}", f"--size={size}",
                            f"--bs={block}", "--rw=write", "--output=/dev/null"], check=False)
            mbps = fio_read(fio, path, size, block)
        else:
            method = "python (buffered; page cache may inflate result)"
            mbps = python_read(path, size, block)
    finally:
        if os.path.exists(path):
            os.remove(path)
    print(json.dumps({"method": method, "size_bytes": size, "block_bytes": block,
                      "seq_read_MBps": round(mbps, 1)}, indent=2))


if __name__ == "__main__":
    main()
