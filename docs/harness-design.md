---
created: 2026-11-05T09:00
updated: 2026-11-13T20:55
---
# Harness design: first slice

How the research pipeline ([stage1a-results.md](stage1a-results.md)) becomes a
harness many volunteers can run for a very large MoE model. Background:
[plan.md](plan.md), [multiuser-design.md](multiuser-design.md),
[stage1-flow.md](stage1-flow.md). K3 figures stay unverified.

Status: the torch-free core exists and is tested (`src/hack3rzun1t3/harness/`,
`tests/test_harness.py`, `tests/test_coordinator.py`). Real compute and layer streaming exist (checked on the 3B); streaming also ran on a 7B that does not fit the card.

## Two phases, very different costs

| Phase | Work | Bound by | Parallel? |
|---|---|---|---|
| 1. Direction | Run prompts through layers, keep last-token activations, difference of means | Disk reads (about 1 TB to layer ~60 for K3) | Across layer ranges as a pipeline; across prompt batches freely |
| 2. Edit | Orthogonalize embeddings, every `o_proj`, every `down_proj` shard by shard | Disk read and write, no GPU | Fully, one shard per unit |

Phase 2 needs the final direction, so it starts only after phase 1 is combined.
(On a MoE, every expert's `down_proj` is edited too; that is a lot of matrices
but the same operation.)

## Work unit

A unit is **(layer range, prompt batch)**: `WorkUnit` in `harness/plan.py`.

- `unit_id` is a hash of the three fields, so every machine names the same
  unit the same way and a result can be matched to its request.
- Layer ranges are contiguous and sized in proportion to each slot's capacity
  (`split_ranges`); every slot gets at least one layer.
- Prompts are cut into fixed batches (`make_units`), so a slow slot loses at
  most one batch.

A contributor receives the unit, the input hidden states at the range's first
layer boundary (a reference to a checkpoint the coordinator holds), and
returns a `Partial`.

## Result format and combining

A `Partial` (`harness/partial.py`) holds, per layer in the range, the **sum** of
last-token activations for harmful and for harmless prompts, plus the prompt
counts. Sums and counts add, so the combined mean does not depend on how
prompts were batched or which slot ran which batch. The test
`test_split_batches_match_one_big_batch` checks this against one big batch.
Floats are added in a fixed way (`math.fsum` over sorted values), so combining
is reproducible and independent of arrival order.

The refusal direction per layer is `mean(harmful) - mean(harmless)`, then
normalized. Choosing the layer is a later step that needs forward passes
with the direction as a hook ([stage1-flow.md](stage1-flow.md)).

## Resuming

`Store` (`harness/store.py`) is an append-only JSON-lines file, one finished
unit per line with a CRC of its payload. On load, a line that fails its CRC
(cut short by a crash, Ctrl-C or power loss, or edited) is dropped and counted,
and `pending(units)` returns what is left to do. A resume therefore redoes
only the unit in flight. This is the same behavior Heretic's per-trial
checkpoint gave us when a run was interrupted twice.

## Checking untrusted results

Cheap checks run on every result (`validate`): right layers, right vector
length, positive counts, no NaN or infinity. A replica pass runs a unit on a
second slot and compares with `agrees` (cosine and relative error per layer
and class). Disagreement means run a third and flag the odd slot. Shard
hashes, canary prompts, signed results and a pinned environment from
[multiuser-design.md](multiuser-design.md) still apply and are not built yet.

Limit: these checks catch garbage, stale data and honest bugs. A slot that
returns plausible but wrong values is caught only when a replica of that unit
runs on another slot, so the unreplicated fraction is a real risk
(`test_no_replication_does_not_catch_plausible_corruption` shows it).

## Coordinator

`Coordinator` (`harness/coordinator.py`) is the pull-model core, in memory and
single-threaded; a network layer will wrap `claim` and `submit`.

- A unit is leased to one worker at a time; an expired lease is handed out
  again, so a vanished worker costs one unit.
- A worker never receives the same unit twice, so replicas land on different
  workers.
- A fraction of units (chosen by hash, `replicate_fraction`) needs two agreeing
  results. On disagreement a third result decides by majority and the odd
  worker is flagged. Other units need one valid result.
- Accepted results go to the `Store`, so a restart resumes.

Simulated runs (`tests/test_coordinator.py`) show: an honest run reproduces
the ground-truth direction; full replication with one corrupting worker still
gives the right direction and flags that worker; a killed run resumes and
finishes; a stalled lease is reassigned.

## Worker loop

`harness/worker.py`: claim, compute, submit, repeat until the coordinator says
the run is done. `compute(unit) -> Partial` is the only GPU-specific part and is
passed in, so the loop is tested with fake compute (`tests/test_worker.py`) and
the real streaming engine plugs in later. A compute failure (for example out of
memory) is not reported; the lease expires and the unit is handed out again,
the same path as a vanished worker.

## Real compute (built)

`harness/gpu_compute.py` (needs torch, imported by workers only) turns a loaded
HF model into a `compute(unit) -> Partial`. It runs each prompt of the unit's
batch alone (no padding), sums the last-token residual entering each layer in
the range in float64, and stops the forward pass after the last wanted layer.
`scripts/gpu_equiv_check.py` splits Qwen2.5-3B into 3 layer ranges x 2 prompt
batches, combines them, and compares with the whole-model path: max absolute
difference 0 (2026-11-13, one 12 GB GPU). The whole model fits the card here, so
this is not yet layer streaming.

## Layer streaming (built)

`harness/streaming.py` runs a model that does not fit the card. Only the token
embedding and one decoder layer are on the GPU at once. For a prompt batch the
hidden states stay on the GPU while each layer's weights are read from the
safetensors files, applied to every prompt, and freed, so a batch costs one read
of layers 0..end-1. The final norm and LM head are never loaded. It works for
Llama-style models (Qwen2, Llama, Mistral) and plugs in as
`make_streaming_compute(...)`, the same contract as `gpu_compute`.
`scripts/streaming_equiv_check.py` on Qwen2.5-3B (3 ranges x 2 batches): max
absolute difference 0 versus the in-memory path, peak VRAM 0.88 GiB
(2026-11-13).

First run on a model larger than the card (2026-11-13): Qwen2.5-7B-Instruct
(28 layers, d_model 3584, about 15 GB in bf16) on the 12 GB card, 8 harmful + 8
harmless prompts, `scripts/streaming_consistency_check.py`. Peak VRAM 1.90 GiB.
One full-range pass over both batches took 145 s (reading the weights from a SATA
SSD is the cost); the same prompts as three split layer ranges gave per-layer sums
identical to the full pass (max absolute difference 0). The layer with the
largest mean-difference norm was the last one (norm 272), which mostly reflects
residual norms growing with depth; picking the layer that matters needs the
sweep, not this number. Not compared against an in-memory run, since the model
does not fit.

## Not built yet (next steps in order)

1. Command-line entry points for coordinator and worker, with a config file.
2. Phase 2 shard editor with the "edit equals hook" equivalence test.
3. Expert-group streaming for MoE layers too big for one card.

## Tests

`PYTHONPATH=src python -m unittest tests.test_harness` runs anywhere. The
older `tests/test_math.py` needs torch, so it runs on the test worker's
virtual environment, not on a laptop without torch.
