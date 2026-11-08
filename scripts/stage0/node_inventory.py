#!/usr/bin/env python3
"""Print a JSON inventory of this node: OS, CPU, RAM, disk, GPU, torch/CUDA."""
import argparse
import json
import os
import platform
import shutil
import subprocess
import sys


def run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=15).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def ram_bytes():
    if sys.platform == "darwin":
        out = run(["sysctl", "-n", "hw.memsize"])
        return int(out) if out.isdigit() else None
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def gpus():
    if not shutil.which("nvidia-smi"):
        return []
    out = run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
    result = []
    for line in out.splitlines():
        name, _, mem = line.partition(",")
        mem = mem.strip()
        result.append({"name": name.strip(), "vram_mib": int(mem) if mem.isdigit() else None})
    return result


def torch_info():
    try:
        import torch  # noqa: WPS433
    except Exception:
        return {"installed": False}
    info = {"installed": True, "version": torch.__version__,
            "cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available()}
    if sys.platform == "darwin":
        info["mps_available"] = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--path", default=os.getcwd(), help="path whose filesystem free space is reported")
    ap.add_argument("--redact-host", action="store_true", help="omit hostname")
    args = ap.parse_args()
    du = shutil.disk_usage(args.path)
    inv = {
        "hostname": None if args.redact_host else platform.node(),
        "os": platform.platform(),
        "arch": platform.machine(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "ram_bytes": ram_bytes(),
        "disk": {"path": args.path, "free_bytes": du.free, "total_bytes": du.total},
        "gpus": gpus(),
        "torch": torch_info(),
    }
    print(json.dumps(inv, indent=2))


if __name__ == "__main__":
    main()
