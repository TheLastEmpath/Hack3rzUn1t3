---
created: 2026-11-01T12:42
updated: 2026-11-01T12:42
---
# Stage 0: measure before building

All scripts are in `scripts/stage0/`, standard library only, Python 3.9+,
macOS and Linux. External tools (`tailscale`, `iperf3`, `fio`) are optional and
detected at runtime; a missing tool gives a clear message, not a traceback.

## Scripts

| Script | Purpose |
|---|---|
| `node_inventory.py` | Prints JSON: OS, CPU, RAM, free disk, GPU model/VRAM (nvidia-smi), torch/CUDA versions if installed |
| `tailscale_check.py` | `status` lists peers; `ping <peer>` runs `tailscale ping` and flags **direct** vs **relayed** |
| `iperf_pairs.py` | Pairwise bandwidth: `server` on one node, `run --peers a b c` from another |
| `disk_bench.py` | Sequential-read benchmark with `fio` (falls back to a plain Python read if fio is missing) |
| `shard_check.py` | SHA-256 of local files vs the LFS oid Hugging Face reports |
| `hop_calc.py` | Per-hop payload size and transfer time from batch, seq, d_model, bandwidth |

## Typical flow

```bash
python3 scripts/stage0/node_inventory.py > inventory-$(hostname -s).json   # keep out of git
python3 scripts/stage0/tailscale_check.py status
python3 scripts/stage0/tailscale_check.py ping PEER
python3 scripts/stage0/iperf_pairs.py server            # on node A
python3 scripts/stage0/iperf_pairs.py run --peers A     # on node B
python3 scripts/stage0/disk_bench.py --dir /path/on/nvme --size-gb 4
python3 scripts/stage0/hop_calc.py --batch 64 --seq 256 --d-model 7168 --bandwidth-mbps 800
```

Measured output contains hostnames and IPs: keep it out of the repo (ignored
via `.gitignore` patterns such as `out/`, `results/`).

## Notes

- `disk_bench.py` drops no caches (needs root); fio uses `--direct=1` where
  supported (Linux). On macOS it uses buffered reads and says so: treat numbers
  as an upper bound.
- `shard_check.py` queries the public HF API; set `HF_TOKEN` in the environment
  for gated repos (never commit it).
- `hop_calc.py` assumes 2 bytes/value (bf16/fp16) by default; use `--dtype-bytes`.

## Verification status

Tested locally (macOS): unit tests (`python3 scripts/stage0/test_stage0.py`) for the
calculator, direct/relayed classifier, shard comparison and inventory JSON;
live `shard_check` query against the Hub API and `tailscale status`; Python-fallback
disk bench. **Not yet run:** fio and iperf3 paths (tools not installed on the dev Mac),
`tailscale ping` on real peers, and the Linux/NVIDIA inventory branch.
