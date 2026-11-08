---
created: 2026-11-01T22:55
updated: 2026-11-01T23:47
---
# OBLITERATUS vs our stage 1a

OBLITERATUS is a third-party, AGPL-3.0 abliteration toolkit. We **read and run**
it to learn; we do not copy its code into this repo (our license is
all-rights-reserved). Ideas we borrow get credited in the notices file.
Telemetry is off by default locally and was left off. Run on the same model as
[stage1a-results.md](stage1a-results.md): Qwen2.5-1.5B-Instruct.

## Its pipeline

SUMMON (load) then PROBE (collect activations) then DISTILL (directions) then
EXCISE (project out) then VERIFY (perplexity, coherence, refusal) then REBIRTH
(save with metadata). Same shape as our five phases, with a heavier verify step.

## Run: `obliterate --method basic` (defaults)

| Setting / result | Value |
|---|---|
| Direction | one difference of means ("Arditi" baseline) |
| Chat template | **off** |
| Layer selection | `knee_cosmic`, which picked layers **23-27** (the last five of 28) |
| Prompts | 842 harmful, 842 harmless |
| Refusal rate after edit | **0.967** (still refuses almost everything) |
| Perplexity | 3.24 to 3.93 |
| KL (first token / sequence) | 0.074 / 0.272 |
| Capability probes | all pass |
| Spectral certification | **RED**: its log says one eigenvalue is above the noise threshold and recommends more directions |

Outcome: the default `basic` preset did **not** remove refusal on this model,
like our first stage 1a run, though for different reasons.

## Run: `obliterate --method advanced` (defaults)

| Setting / result | Value |
|---|---|
| Directions | 4 (SVD), norm-preserving, regularization 0.3 (keeps 30% of the refusal component), biases projected |
| Chat template | **on** |
| Layers edited | again **23-27** (`knee_cosmic`) |
| Refusal rate after edit | **0.933** (still refuses almost everything) |
| Perplexity | 3.24 to 3.27 |
| KL (first token / sequence) | 0.409 / 0.140 |
| Coherence / capability | 0.80 / 0.83 (both dropped) |
| Spectral certification | RED |

More directions plus the chat template did **not** fix it, and damage went up
(coherence and capability fell) while refusal barely moved. The constant across
both runs is the layer set: the same five last layers.

The code explains why. Its layer selection has a guard for small models (hidden
size under 2048 or under 2B parameters) that caps the edit at about 20% of the
layers, and excludes the first two layers. For a 28-layer model that is 5 layers,
so the tool deliberately edits few layers on small models to avoid over-ablation.
On this model that guard leaves most of the refusal pathway untouched.

## Run: `obliterate --method basic --layer-selection all_except_first`

| Setting / result | Value |
|---|---|
| Requested layers | all except layer 0 (27 layers) |
| Layers actually edited | **2-6** (five layers) |
| Refusal rate after edit | **1.00** (no effect) |
| Perplexity | 3.24 to 3.35 |
| Coherence / capability | 0.90 / 0.67 |
| Spectral certification | GREEN |

The requested 27-layer edit was **cut to five layers** by the small-model cap, and
the cap keeps the *first* five of the list, which here are the earliest layers
after the excluded two. So this run did not test "edit all layers"; it tested
"edit layers 2-6", which does nothing. Code check (installed version): when the
model has under 2048 hidden size or under 2B parameters, the selected layers are
trimmed to 20% of the depth (5 of 28) unless the selection is `pinned`, and
`pinned` is not offered by the command-line flag.

So the cap, not the preset, explains all three failures. A fair test needs a
model above the cap thresholds (at least 2B parameters and hidden size at least
2048, for example Qwen2.5-3B) or a way to pin layers.

## Run on Qwen2.5-3B (above the small-model cap): `basic` vs our 1a

Qwen2.5-3B-Instruct: 36 layers, hidden size 2048, about 3.1B parameters, so the
small-model guard does not trigger. Same prompts and procedures as above.

| | OBLITERATUS `basic` | Our 1a (cutoff 0.25) |
|---|---|---|
| Layers edited | **14**: 35-25, 22, 19, 17 (no cap) | embeddings plus all `o_proj` / `down_proj`, one direction from layer 25 |
| Chat template | off | on |
| Refusal after edit | 0.90 (generated-reply rate) | first-token mass 0.996 to **0.114** |
| Damage | perplexity 3.02 to 3.81, sequence KL **1.06**, coherence 1.0, capability 0.83 | harmless first-token KL **0.20** |
| Spectral certification | INCONCLUSIVE | n/a |

With the cap out of the way OBLITERATUS edited 14 layers but still left refusal
at 0.90 while costing noticeably more fidelity (sequence KL above 1). Our edit
removed first-token refusal at a smaller measured cost. The metrics differ (see
caveats), so this is suggestive, not a clean win. Likely contributors on the
OBLITERATUS side: the chat template is off in `basic`, and the edit is spread
over late layers with per-layer directions. Neither is proven.

## What differs from our 1a (and likely explains it)

| | Ours (1a, loose cutoff) | OBLITERATUS `basic` defaults |
|---|---|---|
| Chat template when probing | yes | no, so the probed position is not the assistant turn start |
| Layers edited | embeddings plus every `o_proj` and `down_proj`, all layers | selected layers only (23-27) |
| Layer choice | sweep: ablate, add, KL, pick by a damage cutoff | knee detection plus COSMIC separability |
| Refusal measure | first reply token mass, 32 prompts | generated replies, about 30 shown by its verify sample |
| Damage measure | first-token KL | perplexity, coherence, sequence KL, capability probes |

The tool's own preset list also contains an `all_except_first` option described
as a reproduction of the FailSpy abliterator (all layers except layer 0). Running
`basic` with `--layer-selection all_except_first` isolates the edit-scope effect
and is the obvious next run (the small-model cap may still apply; check the log).

## Takeaways

1. **Where you measure matters.** Direction quality depends on the token
   position and prompt format (chat template) used for probing.
2. **Where you edit matters.** Editing only the last five layers did not suppress
   refusal in either preset (even with four directions and the chat template);
   editing all residual writers did in our run. Defaults tuned to avoid damage can
   under-edit small models.
3. **A single direction can be incomplete.** The spectral check (RED) is a
   direct, cheap signal for the open MoE question: is one direction enough?
4. **Verification is richer than ours.** Perplexity, coherence and capability
   probes are worth adding to our harness.

## Caveats

- Different metrics, prompt sets (842 vs 32) and verify procedures, so the two
  results are not directly comparable numbers.
- The layer-editing behavior is read from its metadata and logs, not traced
  through the code in depth.
- The tool was run from the user's existing source checkout through our own
  virtual environment (the global package set has a version conflict). Nothing
  was installed.
