---
created: 2026-11-01T14:05
updated: 2026-11-01T14:05
---
# Model sizing and harness combinations

For each target model: what the harness needs, and which hardware combinations
can do it. We start with the largest (Kimi K3), then work down to models that
fit the single dev GPU (see the end).

## Terminology

Abliteration itself has no "nodes". It has three steps:

1. **Collect** residual-stream activations on harmful vs harmless prompts.
2. **Compute** the refusal direction (difference of means, normalized), and pick
   the layer where it works best.
3. **Edit** the weights to project that direction out of everything that writes
   into the residual stream.

"Node" is only our distributed-systems word for one contributed machine. In this
project's docs, prefer **worker** (a machine running a layer range) and
**slot** (its capacity class). Steps 1 and 2 need the model running; step 3 does
not (see [plan.md](plan.md)).

## Kimi K3

Values below come from the model's `config.json` on Hugging Face unless marked.
Sources: [config.json](https://huggingface.co/moonshotai/Kimi-K3/raw/main/config.json),
[NVIDIA model card](https://build.nvidia.com/moonshotai/kimi-k3/modelcard),
[InferenceX](https://inferencex.semianalysis.com/model/kimi-k3).

| Item | Value |
|---|---|
| Layers | 93 (1 dense, 92 MoE) |
| Hidden size (d_model) | 7168 |
| Experts | 896 routed, 16 per token, 2 shared |
| Attention | 69 linear (KDA) + 24 full (MLA-style) layers |
| Vocab / max context | 163,840 / 1,048,576 |
| Weights | MXFP4 (4-bit, group 32) for routed experts; **bf16** for attention, shared experts, dense MLP, lm_head |
| Checkpoint size | about 1.4-1.56 TB, 96 shards (secondary sources) |
| Active params | 104B (secondary sources; **not** in the config, the 50B figure is unresolved) |
| License | conflicting reports; read the repo LICENSE before any release |

Derived (approximate, from checkpoint size divided by layers):

- **Average layer: about 16.8 GB.** A whole average layer fits in a 24 GB card;
  it does not fit in 12 GB, so small slots stream experts in groups.
- **Average expert: about 19 MB** (checkpoint-derived). The config's
  `moe_intermediate_size` x `d_model` would imply a larger expert, so the
  checkpoint probably uses a latent/compressed expert form. **Check the
  tensor shapes in the safetensors index before building the loader.**
- Phase 1 only runs up to the candidate layers. Refusal directions usually peak in
  the middle of the network, so plan on a sweep to about layer 60 of 93:
  60 x 16.8 GB is about **1.0 TB read**, plus embeddings. A full pass is 1.56 TB.

### The key result: VRAM is not the constraint

| Question | Answer |
|---|---|
| Keep K3 in pooled VRAM? | 1.56 TB / 24 GB is about 65 cards, more with overhead. **Not feasible.** |
| Compute needed? | 512 prompts x 64 tokens is about 33k tokens. Through 60 layers at 104B active: about 4e15 FLOPs, minutes on one 3090 even at poor MoE efficiency (allow 10x: under an hour). |
| Activation memory? | 33k tokens x 7168 x 2 B = about **0.5 GB**, resident the whole run. |
| Real bottleneck | **Reading about 1 TB of weights from disk once**, then dequantizing MXFP4 to bf16 in software (Ampere has no FP4 hardware). |

So the question is "how do we get 1 TB past a GPU", not "how many GPUs".

### Time model

- **Disk read** (about 1.0 TB): SATA 0.5 GB/s is about 33 min; NVMe 3 GB/s is about 6 min; fast NVMe 7 GB/s is about 2.5 min. Pooled workers read in parallel only their own shards.
- **Hop** (one tensor, 0.5 GB between pipeline stages): 100 Mbps about 38 s; 1 Gbps about 4 s.
- **Download** of the checkpoint (one-off): 1.56 TB at 1 Gbps is about 3.5 h. Split across workers it is about 1.56 TB / N each.
- Compute is small next to the above, but MoE grouped GEMMs are inefficient. Measure it in stage 1.

### Harness combinations (estimates, not measured)

| Combo | Hardware | Per-worker disk | Rough wall clock for phase 1 | Notes |
|---|---|---|---|---|
| A. Single box | 1 GPU (12 GB+; 24 GB ideal), 64 GB+ RAM, 2 TB NVMe | 1.6 TB | about 15-60 min after the download | Simplest. Download is the pain (hours). Best first K3 attempt. |
| B. Small pool | 4 x 24 GB, 32 GB+ RAM each | about 390 GB | about 20-40 min | 3 hops; each person stores a quarter. |
| C. Medium pool | 8 x 24 GB | about 195 GB | about 20-45 min | 7 hops. |
| D. Target pool | 12-20 x 24 GB (12 GB works with finer expert streaming) | about 80-130 GB | about 15-45 min at 100 Mbps+ links | 11-19 hops. The point of pooling is **storage and download split**, not VRAM. |
| E. Big-RAM box | 1 GPU, 512 GB-1 TB RAM | RAM holds hot layers | faster passes, repeatable sweeps | Good for iterating on the layer sweep; costs more per machine. |

Recommended path: **D as the design target, A as the first real run**, since A
validates the streaming and MXFP4 handling with no networking. Per-worker floor
for D: 12 GB VRAM, 32 GB RAM, 100 GB free NVMe, 100 Mbps+ symmetric, and a direct
Tailscale path.

### Phase 2 (editing) cost, and the hard part

Phase 2 needs disk and CPU, not GPUs, but it is not trivial for K3:

- Edit every matrix that **writes** into the residual stream: embeddings,
  attention output projections, down-projections of the dense MLP, shared
  experts, and **every routed expert**. The router only reads the residual
  stream, so it is not edited.
- Routed experts are MXFP4. Editing means dequantize, project, **re-quantize**,
  which adds quantization error on top of the edit. The alternative, a bf16
  result, is about 5.6 TB. Which output format to use is an open question, and
  the choice changes the quality of the result.
- Validation after the edit reruns the phase 1 harness on the edited weights.

### Risks and unknowns

- Hugging Face `transformers` support for the KDA linear-attention and K3's
  custom layers is unverified; the model may need `trust_remote_code` and
  extra kernels (check Ampere compatibility).
- Whether one refusal direction is enough across experts (the open research
  question) is answered only by running the harness.
- Active-parameter conflict (104B vs about 50B) affects only the compute
  estimate, which is not the bottleneck.

## Smaller models (next; fit the dev GPU)

To be filled in after K3. Planned sizing table columns: params, layers,
d_model, bf16 size, streams on 12 GB?, harness combo, purpose.

| Model class | Role in the ladder |
|---|---|
| About 8B dense | Stage 1: flow logic on the single dev GPU (bf16 is about 16 GB, so it must stream) |
| About 70B dense | Stage 2 |
| About 200B-class MoE | Stage 4 rehearsal for K3's expert handling |
