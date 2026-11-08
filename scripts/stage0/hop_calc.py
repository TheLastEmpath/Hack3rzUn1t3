#!/usr/bin/env python3
"""Per-hop payload size and transfer time for a pipeline boundary.

payload = batch * seq * d_model * dtype_bytes
"""
import argparse


def human(n):
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or unit == "TiB":
            return f"{n:.2f} {unit}"
        n /= 1024


def hop(batch, seq, d_model, dtype_bytes, bandwidth_mbps, latency_ms=0.0):
    payload = batch * seq * d_model * dtype_bytes
    seconds = payload * 8 / (bandwidth_mbps * 1e6) + latency_ms / 1000
    return payload, seconds


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--seq", type=int, required=True)
    ap.add_argument("--d-model", type=int, required=True)
    ap.add_argument("--bandwidth-mbps", type=float, required=True, help="measured, megabits per second")
    ap.add_argument("--dtype-bytes", type=int, default=2, help="2 for bf16/fp16, 4 for fp32")
    ap.add_argument("--latency-ms", type=float, default=0.0)
    ap.add_argument("--hops", type=int, default=1, help="number of hops in the pipeline")
    a = ap.parse_args()
    if min(a.batch, a.seq, a.d_model, a.dtype_bytes) <= 0 or a.bandwidth_mbps <= 0:
        ap.error("all sizes and bandwidth must be positive")
    payload, sec = hop(a.batch, a.seq, a.d_model, a.dtype_bytes, a.bandwidth_mbps, a.latency_ms)
    print(f"payload per hop : {human(payload)} ({payload} bytes)")
    print(f"transfer / hop  : {sec:.4f} s at {a.bandwidth_mbps} Mbit/s")
    if a.hops > 1:
        print(f"{a.hops} hops total   : {sec * a.hops:.4f} s per forward pass of the batch")


if __name__ == "__main__":
    main()
