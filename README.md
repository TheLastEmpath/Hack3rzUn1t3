# Hack3rzUn1t3

A distributed activation-extraction harness for learning how abliteration
(refusal-direction removal, Arditi et al. 2024) works on large open-weight
models, scaling up to very large MoE models. This is a feasibility and
learning exercise for an AI safety / red-teaming path. The first deliverable is
a research write-up; any released artifact is a separate, later decision.

**Status:** Stage 0 tools, a Heretic run on a 3B model, and a tested
harness core exist. Start at [docs/handoff.md](docs/handoff.md) to pick up the work.

## Idea in one paragraph

The prompt set is small and fixed, so hidden states at a layer boundary are
small. The weights are the big thing. So the design streams layers from NVMe on
one node first, and scales out to a pipeline of nodes that each hold a range of
layers. Custom PyTorch/HF harness, not Petals / exo / llama.cpp RPC. Nodes talk
over Tailscale.

See [docs/plan.md](docs/plan.md) for the full plan and test ladder.

## Setup

Stage 0 scripts need only Python 3.9+ (standard library). Optional external
tools: `tailscale`, `iperf3`, `fio`.

```bash
# macOS
brew install iperf3 fio
# Debian/Ubuntu/Fedora: use your package manager for iperf3 and fio
```

## Usage

See [docs/stage0.md](docs/stage0.md). Quick tour:

```bash
python3 scripts/stage0/node_inventory.py
python3 scripts/stage0/tailscale_check.py status
python3 scripts/stage0/disk_bench.py --dir /path/on/nvme
python3 scripts/stage0/shard_check.py --repo ORG/MODEL --dir /path/to/shards
python3 scripts/stage0/hop_calc.py --batch 64 --seq 256 --d-model 4096 --bandwidth-mbps 900
```

## Docs

Everything longer lives in [docs/](docs/). Docs are written once, here; the
owner's notes app views this folder through a symlink.

## License

Apache License 2.0. See [LICENSE](LICENSE). Third-party components keep their own
licenses: see [NOTICES.md](NOTICES.md).
