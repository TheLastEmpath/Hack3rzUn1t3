---
created: 2026-11-03T09:30
updated: 2026-11-03T19:44
---
# Heretic: how it works (read from its README and source)

[Heretic](https://github.com/p-e-w/heretic) (AGPL-3.0-or-later) is a training-free
abliteration tool that **searches** for good settings instead of using fixed
presets. We read it to learn; we do not copy its code into this repo (our license
is all-rights-reserved). Read at version 2.1.0.dev0. Run result: [heretic-results.md](heretic-results.md).

## The loop

1. **Modifier** applies an ablation to the model. Default is *Arbitrary-Rank
   Ablation* (ARA); the other is directional ablation ("abliteration") with
   optional projected / norm-preserving variants (MPOA). Edits are applied as LoRA
   adapters, so each trial can be applied and undone cheaply.
2. **Scorers** measure the result. Defaults: refusal rate (`KeywordRate`) and KL
   divergence from the original model (`KLDivergence`).
3. **Optimizer** (Optuna TPE) proposes the next settings to try, to minimize both
   scores. Default 100 trials, the first 30 random.
4. You pick a trial from the refusals-vs-KL tradeoff, then save, upload, chat with
   it, or benchmark it.

## What gets searched (abliteration modifier)

- `direction_index`: a float that interpolates between per-layer refusal
  directions (or a single global one).
- Per component (`attn.o_proj`, `mlp.down_proj`): a **weight kernel across
  layers**: `max_weight`, `max_weight_position`, `min_weight`,
  `min_weight_distance`. So strength varies by layer instead of being one number.
- Options: winsorization of residuals, orthogonalizing against the harmless
  direction, row-magnitude renormalization.

## Defaults worth knowing

| Item | Default |
|---|---|
| Harmful / harmless prompts | `mlabonne/harmful_behaviors`, `mlabonne/harmless_alpaca` (400 each for directions) |
| Refusal counting | keyword markers ("sorry", "i can't", "i cannot", ...) in the first 100 generated tokens |
| KL | first-token distribution over 100 harmless prompts |
| System prompt | "You are a helpful assistant." |
| Precision | bf16 via `dtypes = ["auto", ...]`; `bnb_4bit` optional |
| Runtime | README: about 30-40 min for a 4B model on an RTX 3090 (100 trials) |
| Telemetry | none found in the source; the only network fetch is for reproducibility manifests |

## How it maps onto our pipeline

| Ours | Heretic |
|---|---|
| one direction from one layer, one global strength | per-layer directions plus a per-layer weight kernel, tuned |
| chosen by a manual sweep and a KL cutoff | chosen by Optuna over refusals and KL together |
| first-token metric and string-match buckets | keyword refusal rate (same weakness as our string matching) and first-token KL (same as ours) |
| blinded grader on saved replies | none; it trusts keyword counts |

Two consequences: (1) Heretic automates the "per-layer directions and strength"
experiment we wanted to try by hand, and (2) it shares our string-matching blind
spot, so our blinded grading is a useful independent check on its output.

## Caveats

- Support: README lists most dense models, many multimodal models, several MoE
  architectures and "many modern hybrid models". Whether K3's mix of linear and
  full attention is covered is untested.
- ARA imports `bitsandbytes` at load, so installing it is a likely friction point
  on the test worker (the single-GPU development machine).
- Its quality claims are the author's own.

## Plan (done)

Run it on Qwen2.5-3B-Instruct on the single GPU (the 3B fits at bf16), then score
the saved model with `eval_gen.py` and blinded grading next to ours and the
published model ([reference-comparison.md](reference-comparison.md)).
