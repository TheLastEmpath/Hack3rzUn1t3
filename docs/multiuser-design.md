---
created: 2026-11-01T13:10
updated: 2026-11-01T13:10
---
# Multi-user design (theory)

Design target: **20+ volunteers**, each contributing one consumer GPU machine.
Everything here is theory to be tested by the [test ladder](plan.md). K3 figures
are unverified (see the caveat in the plan).

## Reference node ("slot")

Design to an **RTX 3090 (24 GB VRAM)**. Treat anything smaller (e.g. RTX 3060,
12 GB) as a "small slot" that gets proportionally fewer layers, and anything
larger as a bonus, never a requirement.

Hardware numbers to confirm with `nvidia-smi` before they go in any doc: the
RTX 3060 normally has 12 GB, and the RTX 3090 normally has 24 GB. (16 GB and
32 GB cards exist, but they are other models.) Ampere cards (3060, 3090) share
the same architecture, which helps numeric consistency across nodes.

| Slot class | VRAM | Role |
|---|---|---|
| Small | 8-12 GB | fewer layers, dense models up to ~8B via streaming |
| Reference | 24 GB | default unit for planning |
| Large | 32 GB+ | takes more layers; same protocol |

### Measured: test worker 1 (the single-GPU development machine, 2026-11-01)

| Item | Value | Consequence |
|---|---|---|
| GPU | one 12 GB consumer GPU (Ampere) | small slot; 8B bf16 (about 16 GB) must stream |
| CPU / RAM | 12 threads, **15 GB** RAM (about 10 GB available) | page cache cannot hold an 8B model; expect disk reads every pass |
| Disk | 256 GB **SATA** SSD, about 137 GB free | about 0.5 GB/s, not NVMe; 16 GB read is roughly 30+ s per pass |

The 24 GB reference card is not installed anywhere yet. Re-measure with
`scripts/stage0/node_inventory.py` and `disk_bench.py` for real numbers.

Per-node minimums to publish: VRAM class, free NVMe (about 100 GB), RAM (32 GB+),
sustained upload bandwidth, and a Tailscale-reachable direct path.

## Capacity math: why K3 is a streaming problem

- 20 x 24 GB = 480 GB of pooled VRAM, about **31%** of K3's ~1.56 TB weights.
  The pool cannot hold K3 in VRAM, so weights must stream from disk.
- **Disk is the real pool.** Split the weights by layer range: 1.56 TB / 20 =
  about 78 GB of shards per node, which fits an ordinary NVMe.
- A single K3 layer may exceed one card's VRAM (about 1.56 TB / number of
  layers; the layer count is unverified). So a node must stream **expert groups**
  within a layer: run attention and the router, bucket tokens by expert, then
  load experts in groups that fit and process their buckets.
- Keep all prompt hidden states resident on the node (about 1 GB for
  ~1000 prompts x 64 tokens x d_model 7168 in bf16), so each weight chunk is
  read from disk **once per run**.
- Rough time per node: shard read (78 GB at ~3 GB/s is about 30 s) plus compute,
  so phase 1 is likely minutes of compute. The wall clock is dominated by the
  sequential hops: about 1 GB per hop, roughly 80 s at 100 Mbps or 8 s at
  1 Gbps, times the number of hops. Measure with `scripts/stage0/hop_calc.py`.
- If K3 ships in a lower-precision format, all sizes shrink. Check the model card.

## Pull-based worker model

Home networks mean NAT, changing IPs and uptime gaps. So:

- Workers make **outbound-only** connections and **poll** a coordinator for jobs
  (layer range, prompt-batch id, checkpoint to start from). No inbound ports.
- The coordinator owns tokenization, embeddings, masks, job assignment, and
  boundary checkpoints. Workers are stateless between jobs.
- Hidden states are checkpointed at layer-range boundaries, so a dropped node
  costs one range, not the run (this is test stage 3).
- Assignment is a simple solver over (VRAM, disk, measured bandwidth, uptime),
  giving contiguous layer ranges proportional to capacity.

## Trust and integrity (untrusted volunteers)

A wrong activation silently corrupts the refusal direction, so assume nodes can
be buggy, misconfigured, or dishonest.

1. **Shard hashes** against Hugging Face SHA-256 before a node may join.
2. **Replication factor 2** on a random sample of ranges (higher early on),
   compared with cosine similarity and relative error at a per-dtype tolerance.
   Disagreement triggers a third run and flags the odd node.
3. **Canary prompts** with stored reference activations, mixed into every job
   invisibly.
4. **Signed results** (per-node key registered at onboarding) so a result can be
   attributed.
5. **Pinned environment**: container image with fixed torch, transformers and
   CUDA versions. Tolerances absorb GPU-generation differences, not version drift.
6. **Direction sanity**: the final direction must match a smaller-model
   reference run and be stable across bootstrap resamples of the prompt set.

## Network and access

- One tailnet for the project. ACLs by tag: workers may reach only the
  coordinator on one port; workers cannot reach each other (hops go through the
  coordinator, or via an allowed pairwise rule only if bandwidth requires it).
- Device approval on, key expiry on. Prefer **node sharing** or a dedicated
  tailnet over inviting volunteers into a personal one.
- Check direct vs relayed paths (`scripts/stage0/tailscale_check.py`). A relayed
  node counts as a small slot.

## Privacy and safety of the data

- Only **public prompt datasets** go to workers (harmful and harmless sets are
  already public). No private prompts, ever.
- Workers see weights and activations only, never the final direction, unless
  the coordinator chooses to publish it.
- Volunteers should know exactly what runs on their machine: read-only
  container, no host mounts except the shard directory, resource limits.

## Governance (to decide before onboarding anyone)

- Written contributor terms: what runs, what is logged, how to leave.
- The model's license governs redistribution of weights and any derived
  artifact; settle it before any release (open question in the plan).
- Output policy: write-up first; any released artifact is a separate, explicit
  decision by the owner.

## Build order for a single early node

Start with one node, because most of this design is testable on it:

1. Implement the layer-streaming engine on one GPU, with a dense ~8B model. An
   8B model in bf16 is about 16 GB, which does not fit a 12 GB card, so this is
   a true test of streaming, not a toy.
2. Compare per-layer hidden states and the refusal direction against a normal
   in-memory reference run (on CPU or a cloud GPU if needed).
3. Add the worker/coordinator protocol (pull model, checkpoints) on localhost.
4. Only then add a second machine and exercise Stage 0 networking for real.

## Open design questions

- Replication rate vs. cost: what fraction of ranges to double-run.
- Whether hops should go worker-to-worker or always via the coordinator.
- Onboarding: attestation of GPU model and VRAM (self-reported numbers can lie).
- Incentives and uptime expectations for volunteers.
