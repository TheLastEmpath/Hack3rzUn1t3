#!/usr/bin/env python3
"""Fail if a change adds personal info, secrets, weights or large files.

Usage:
  check_safety.py BASE      # check lines/files added since git ref BASE
  check_safety.py --all     # check every tracked file
A line containing the marker `safety-ok` is skipped (for documented examples).
Standard library only.
"""
import os
import re
import subprocess
import sys

MAX_BYTES = 5 * 1024 * 1024
BAD_EXT = (".safetensors", ".pt", ".pth", ".bin", ".ckpt", ".gguf", ".onnx", ".npz")
MARKER = "safety-ok"
PATTERNS = [
    ("private/tailnet IP", re.compile(r"\b(?:192\.168|10\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])|100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7]))\.\d{1,3}\.\d{1,3}\b")),
    ("local user path", re.compile(r"(?:/Users/|/home/)[A-Za-z0-9._-]+/|[A-Za-z]:\\Users\\")),
    ("email address", re.compile(r"\b[\w.+-]+@(?!users\.noreply\.github\.com)[\w-]+\.[\w.-]+\b")),
    ("key or token", re.compile(r"ghp_[A-Za-z0-9]{20,}|github_pat_\w{20,}|\bsk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|\bhf_[A-Za-z0-9]{20,}|BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY")),
]
SELF = "scripts/check_safety.py"


def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout


def scan_line(path, n, line, problems):
    if MARKER in line:
        return
    for name, rx in PATTERNS:
        if rx.search(line):
            problems.append(f"{path}:{n}: {name}")


def main(argv):
    if len(argv) != 2:
        sys.exit(__doc__)
    problems = []
    if argv[1] == "--all":
        paths = [f for f in git("ls-files").splitlines() if f != SELF]
        for f in paths:
            try:
                with open(f, encoding="utf-8") as fh:
                    for n, line in enumerate(fh, 1):
                        scan_line(f, n, line, problems)
            except (UnicodeDecodeError, IsADirectoryError, FileNotFoundError):
                pass
    else:
        base = argv[1]
        paths = [f for f in git("diff", "--name-only", "--diff-filter=AM", f"{base}...HEAD").splitlines() if f != SELF]
        path = None
        n = 0
        for raw in git("diff", "-U0", "--diff-filter=AM", f"{base}...HEAD").splitlines():
            if raw.startswith("+++ "):
                path = raw[6:] if raw.startswith("+++ b/") else None
            elif raw.startswith("@@"):
                m = re.search(r"\+(\d+)", raw)
                n = int(m.group(1)) - 1 if m else 0
            elif raw.startswith("+") and path and path != SELF:
                n += 1
                scan_line(path, n, raw[1:], problems)
    for f in paths:
        if f.lower().endswith(BAD_EXT):
            problems.append(f"{f}: weights/checkpoint file type")
        elif os.path.isfile(f) and os.path.getsize(f) > MAX_BYTES:
            problems.append(f"{f}: larger than 5 MB")
    if problems:
        print("Safety check failed:")
        print("\n".join(problems))
        print("\nRemove the item, or add `safety-ok` on a line that is a deliberate example.")
        return 1
    print("Safety check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
