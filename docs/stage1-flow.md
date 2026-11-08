---
created: 2026-11-01T14:40
updated: 2026-11-01T14:40
---
# Stage 1 flow: abliteration end to end on one small GPU

Goal: run the **whole** abliteration process once, by hand-sized steps, on a small
model, so every step is understood before scaling it ([plan.md](plan.md),
[model-sizing.md](model-sizing.md)). Hardware (the "test worker"): one 12 GB GPU, 15 GB RAM,
SATA SSD (see [multiuser-design.md](multiuser-design.md)).

This is a design for learning. Nothing here is built yet.

## Model ladder for the single GPU

| Step | Model (suggested) | bf16 size | Why |
|---|---|---|---|
| 1a | Qwen2.5-1.5B-Instruct (28 layers, d=1536) | about 3 GB | Fits in the GPU. Run the *ordinary* way to learn the recipe with no streaming. |
| 1b | Same 1.5B through **our streaming engine** | about 3 GB | It also fits normally, so the engine can be checked **exactly** against 1a. |
| 1c | Qwen2.5-7B-Instruct (28 layers, d=3584) | about 15 GB | Too big for 12 GB VRAM and the 10 GB of free RAM, so it must stream. Same code as 1b. |

Alternatives to Qwen2.5-7B: Llama-3.1-8B-Instruct (32 layers, d=4096; gated on
Hugging Face, needs license acceptance). Qwen needs no gate. Model facts here are
from memory; confirm layer count and size from each `config.json`.

The 1b step is the key trick: **validate the streaming engine on a model that
also fits normally**, so any difference is a bug in the engine, not the model.

## The five phases

```
 prompts ──► [1 collect] ──► activations ──► [2 direction + layer pick]
                                                  │
                      refusal direction r̂, layer l*
                                                  ▼
   original weights ──► [3 edit weights] ──► edited weights ──► [4 validate]
                                                  ▲                    │
                                                  └──── [5 compare to reference]
```

### Phase 0: data

- **Harmful prompts:** about 128 train + 32 validation from AdvBench (or
  HarmBench). **Harmless:** matching counts from Alpaca. Public datasets only.
- Apply the model's **chat template** to each prompt, with the assistant turn
  open, then tokenize. Left-pad so the **last token** of every prompt is aligned.
- The position we read is the **last token** (the one that predicts the first
  reply token). That is where the model has decided whether to refuse.

### Phase 1: collect activations (the part that needs the model running)

For every prompt, at the last-token position, record the **residual stream** at
the input of each layer (`resid_pre`, a vector of size d_model). Result:
`acts[class][layer]` of shape `(n_prompts, d_model)`. For 256 prompts this is a
few MB, tiny.

Ordinary mode (step 1a): `model(..., output_hidden_states=True)` on the GPU.

**Streaming mode (steps 1b, 1c):**

1. Embed all prompts once. Keep the full hidden state `(prompts, tokens, d)` for
   all prompts resident, about 0.2-0.3 GB, on the GPU or in RAM.
2. For layer `i` in `0..L-1`:
   1. Read layer `i`'s tensors from the safetensors shard (only that layer's
      byte ranges, via `safe_open`) into GPU memory.
   2. Run the layer over all prompts, in micro-batches that fit.
   3. Save the last-token `resid_pre`; keep the output as the next input.
   4. Free the layer.
3. Cost: the whole model is read from disk **once**: 15 GB at about 0.5 GB/s is
   about 30 s, plus compute. This is the same loop we later spread over
   workers.

Engine pieces to get right (the usual traps): rotary position embeddings,
the causal mask with left padding, and `position_ids` for padded batches. The
exact-match check in 1b catches all of these.

### Phase 2: the refusal direction and which layer

- `r_l = mean(harmful_l) - mean(harmless_l)`, then normalize: `r̂_l = r_l / |r_l|`.
  One candidate direction per layer.
- **Pick a layer.** Run the validation harmful prompts with each candidate
  ablated (projecting `r̂` out of the residual stream at every layer) and measure
  how much refusal drops; run harmless prompts with `r̂` added to see whether it
  induces refusal; check the harmless output distribution barely changes (KL).
  Choose the layer whose direction removes refusal with the least damage,
  skipping the last few layers.
- Refusal measure to start with: probability mass on the first-token refusal
  words (e.g. "I", "Sorry", "As") from one forward pass. Cheap, and works in
  streaming mode because it needs only one pass per candidate set.
- This phase needs the engine to support one **hook**:
  `x := x - (x . r̂) r̂` applied to the residual stream. Same hook idea, two uses.

### Phase 3: edit the weights (no GPU)

Removing the direction from the residual stream for good means orthogonalizing
every matrix that **writes** into it, so the model can no longer write `r̂`:

`W' = W - r̂ r̂ᵀ W` (for matrices whose output lives in the residual stream)

| Matrix | Edited? |
|---|---|
| Token embeddings | yes |
| Attention output projection (`o_proj`) | yes, every layer |
| MLP down projection (`down_proj`) | yes, every layer |
| q/k/v, gate/up projections, norms | no (they only read) |
| `lm_head` | no |

Do it shard by shard: load a shard, edit the matrices it contains, write a new
safetensors shard. Constant memory, CPU only. Keep the original shard names and
index so the result loads like the original.

Gotchas: models with **tied embeddings** (Qwen2.5-1.5B ties input and output
embeddings) would also change `lm_head` when you edit the embedding, so untie
first. Check biases on output projections (Qwen2.5 `o_proj` has none; if a model
had one, it would need the same projection).

### Phase 4: validate the edited model

Reuse the Phase 1 engine on the edited weights:

- **Refusal:** first-token refusal probability on the held-out harmful prompts
  should collapse; spot-check with a short greedy generation.
- **Collateral damage:** KL divergence of the next-token distribution on harmless
  prompts versus the original, plus a few capability sanity prompts.
- **Equivalence check:** the edited model should match "original model with
  the ablation hook switched on" to numeric tolerance. That ties Phase 3 to
  Phase 2 and is the best correctness test of the edit.

### Phase 5: compare against references

- 1b vs 1a: per-layer hidden states and the final direction must match (cosine
  near 1; cosine across layers' directions above 0.99, bf16 tolerance).
- 1c: no in-memory reference fits on this box. Use Hugging Face's own
  `device_map` offload as an independent slow reference, or cross-check with
  known published directions/behaviours for the same model.

## What each step teaches

| After | You understand |
|---|---|
| 1a | the recipe: activations, difference of means, direction, ablation |
| 1b | why the engine is just "one layer at a time, activations stay resident" |
| 1c | that bigger models change cost (disk reads), not the logic |
| Phase 3 | the edit is plain linear algebra on a handful of matrices |
| Phase 4 | how to tell whether it worked and what it broke |

## Environment note

Install the Python stack (torch, transformers, safetensors) in a **virtual
environment** on the user's machine. Do not touch the kernel, bootloader,
initramfs or GPU driver; the existing driver already exposes the GPU.
Check disk: original and edited 7B weights are about 30 GB together; about
137 GB is free.

## Proposed build order

1. `data/`: prompt loaders and chat-template tokenization.
2. `engine/`: ordinary runner (1a) with activation capture and the ablation hook.
3. `directions/`: difference of means, layer selection.
4. `engine/streaming`: layer-by-layer runner; exact-match test against 1a (1b).
5. `edit/`: shard-by-shard orthogonalization and the equivalence test.
6. 7B run (1c), then record timings in `docs/` for the pooled design.
