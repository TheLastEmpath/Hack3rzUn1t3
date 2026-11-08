#!/usr/bin/env python3
"""Hash local files and compare with the SHA-256 (LFS oid) Hugging Face reports.

  shard_check.py --repo ORG/MODEL --dir LOCAL_DIR [--revision main]

Exit code 0 if every checked file matches, 1 otherwise. Set HF_TOKEN for gated repos.
"""
import argparse
import hashlib
import json
import os
import sys
import urllib.parse
import urllib.request


def sha256_file(path, block=16 * 1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def remote_oids(repo, revision, endpoint="https://huggingface.co"):
    url = f"{endpoint}/api/models/{repo}/tree/{urllib.parse.quote(revision, safe='')}?recursive=true"
    req = urllib.request.Request(url)
    tok = os.environ.get("HF_TOKEN")
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    with urllib.request.urlopen(req, timeout=60) as r:
        entries = json.load(r)
    return {e["path"]: e["lfs"]["oid"] for e in entries if e.get("type") == "file" and e.get("lfs")}


def check(local_dir, oids, hasher=sha256_file):
    """Return (ok, bad, missing) lists of relative paths."""
    ok, bad, missing = [], [], []
    for rel, oid in sorted(oids.items()):
        p = os.path.join(local_dir, rel)
        if not os.path.isfile(p):
            missing.append(rel)
        elif hasher(p) == oid:
            ok.append(rel)
        else:
            bad.append(rel)
    return ok, bad, missing


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--revision", default="main")
    ap.add_argument("--allow-missing", action="store_true", help="don't fail on files not present locally (partial shard sets)")
    a = ap.parse_args()
    oids = remote_oids(a.repo, a.revision)
    if not oids:
        sys.exit("no LFS files reported by the Hub for this repo")
    ok, bad, missing = check(a.dir, oids)
    for rel in ok:
        print(f"OK       {rel}")
    for rel in bad:
        print(f"MISMATCH {rel}")
    for rel in missing:
        print(f"MISSING  {rel}")
    print(f"{len(ok)} ok, {len(bad)} mismatched, {len(missing)} missing")
    sys.exit(1 if bad or (missing and not a.allow_missing) else 0)


if __name__ == "__main__":
    main()
