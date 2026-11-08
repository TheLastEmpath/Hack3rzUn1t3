---
created: 2026-11-01T12:42
updated: 2026-11-05T06:58
---
# Plan

> **Caveat:** all facts about Kimi K3 in this document (size, active
> parameters, license) are **unverified**. Re-check against the model card
> before relying on any of them.

## Goal

Learn how refusal-direction abliteration works (Arditi et al. 2024) by
extracting activations from progressively larger open-weight models, up to a
~2.8T-parameter MoE (Kimi K3). The first output is a research write-up. Whether
to release any artifact is a separate, later decision.

## Tooling decision

Custom PyTorch / Hugging Face harness, **not** Petals, exo, or llama.cpp RPC.

- **Petals:** swarm model with untrusted peers and a fixed inference API; hard
  to read arbitrary hidden states at chosen layers, weak fit for models it
  doesn't host, and trust is poor for our integrity needs.
- **exo:** aimed at interactive inference across consumer devices; limited
  control over what is captured and where, and its supported model set lags.
- **llama.cpp RPC:** quantized GGUF inference; abliteration research wants
  full-precision residual-stream activations and exact control of hook points.
- **Custom:** we need hidden states at a chosen layer boundary, deterministic
  control over precision, our own integrity checks, and a layer-streaming mode
  none of the above provide. The workload (fixed small prompt set, forward
  pass only) is simple enough that a custom harness is cheaper than bending a
  framework.

## Layer-streaming insight

Weights are huge; activations are tiny. With a fixed, small prompt set, the
hidden state at a layer boundary is `batch x seq x d_model` values, which is
megabytes. So:

1. **One node first:** stream layers from NVMe through GPU/CPU one at a time,
   keep the whole prompt batch's hidden states resident, and run layer by layer.
   Disk read speed, not compute, is the bottleneck.
2. **Scale out:** a pipeline of nodes, each holding a contiguous layer range,
   passing only the small hidden-state tensor between hops over Tailscale.

## Test ladder

| Stage | What | Pass criterion |
|---|---|---|
| 0 | Network and bandwidth, disk speed, node inventory | Direct (not relayed) paths; measured numbers recorded |
| 1 | ~8B dense on 3-4 nodes vs single-GPU reference | Hidden states match reference within tolerance |
| 2 | ~70B on 6-8 nodes | Same, plus throughput numbers |
| 3 | Kill a node mid-run and resume | Resumes from checkpointed boundary, same result |
| 4 | ~200B-class MoE rehearsal | Expert routing handled; activations sane |
| 5 | Kimi K3 with node pool plus offload | Activations extracted for the full prompt set |

Multi-user design (theory): [multiuser-design.md](multiuser-design.md). Model sizing and harness combinations: [model-sizing.md](model-sizing.md). Stage 1 flow (small model, single GPU): [stage1-flow.md](stage1-flow.md). Stage 1a results: [stage1a-results.md](stage1a-results.md). OBLITERATUS comparison: [obliteratus-comparison.md](obliteratus-comparison.md). Published-model comparison: [reference-comparison.md](reference-comparison.md). Heretic notes: [heretic-notes.md](heretic-notes.md). Heretic result: [heretic-results.md](heretic-results.md). Harness design: [harness-design.md](harness-design.md).

Stage 0 tooling: [stage0.md](stage0.md). Do not start Stage 1 before Stage 0
numbers exist.

## Trust model

Nodes may be flaky or (in principle) wrong; results must be checkable.

- **Shard hashes:** hash every local weight shard and compare with the SHA-256
  (LFS oid) that Hugging Face reports for the file. Implemented in Stage 0.
- **Spot-check segments:** re-run randomly chosen layer segments on a second
  node and require agreement within a numeric tolerance (set per dtype).
- **Canary prompts:** include prompts with known expected outputs/activations
  to detect a node that returns garbage or stale data.

## Open questions

- **K3 active parameters:** 104B vs ~50B is unresolved (unverified sources
  disagree).
- **License conflict:** K3's license terms vs. the intended write-up/release
  have not been reconciled.
- **MoE and a single refusal direction:** does one direction suffice, or do
  experts/routing need per-expert or per-layer treatment?
- **Node count and hardware:** how many nodes, which GPUs/VRAM/RAM/NVMe.
- **End goal:** write-up only, or eventually a released artifact.
