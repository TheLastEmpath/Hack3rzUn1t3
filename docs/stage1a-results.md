---
created: 2026-11-01T17:20
updated: 2026-11-01T23:47
---
# Stage 1a results: Qwen2.5-1.5B-Instruct, single GPU

Design: [stage1-flow.md](stage1-flow.md). Code: `src/hack3rzun1t3/`, run with
`python -m hack3rzun1t3.run_1a` (see the module docstring for env vars).

## Setup

| Item | Value |
|---|---|
| Model | Qwen/Qwen2.5-1.5B-Instruct, 28 layers, d_model 1536, bf16, sdpa attention |
| Model revision | pinned in `run_1a.py` (`DEFAULT_REVISION`) |
| Harmful prompts | AdvBench `harmful_behaviors.csv` (520), 128 train + 32 validation |
| Harmless prompts | Alpaca-cleaned, filtered to plain instructions of 30-150 characters (30,939 left), 128 train + 32 validation |
| Seed | 0; train and validation are disjoint |
| Position | last token of the chat-templated prompt, batch size 1 (no padding) |
| Software | torch 2.14, transformers 5.22 (in a venv layered on the system packages) |
| Metric | probability mass on single-token refusal openers (`I`, `Sorry`, `As`, `Unfortunately`, `Im`) at the first reply token |

Calibration before the sweep: baseline refusal metric is **0.974 on harmful
and 0.114 on harmless** validation prompts, and greedy replies to harmful prompts
open with "I'm sorry, but I can't assist with that." The metric separates the
two sets, so it was trusted.

## Layer sweep (identical in both runs)

Candidate direction at each layer = mean harmful minus mean harmless activation
at that layer's input. Columns: refusal after ablating the direction everywhere
(lower = refusal removed), refusal after *adding* the direction to harmless
prompts (higher = it carries refusal), and KL of the harmless distribution vs the
untouched model (damage).

| Layer | Ablated harmful refusal | Added harmless refusal | Harmless KL |
|---|---|---|---|
| 1-10 | 0.94-0.99 | 0.10-0.13 | 0.03-0.40 |
| 11 | 0.677 | 0.140 | 0.316 |
| 12 | 0.859 | 0.305 | 0.065 |
| 13 | 0.442 | 0.289 | 0.337 |
| 14 | 0.023 | 0.816 | 0.991 |
| 15 | 0.022 | 0.904 | 1.646 |
| 16 | 0.045 | 0.939 | 0.826 |
| 17 | 0.111 | 0.919 | 0.153 |
| 18 | 0.128 | 0.959 | 0.185 |
| 19 | 0.110 | 0.970 | 0.198 |
| 20 | 0.001 | 0.961 | 1.548 |
| 21 | 0.019 | 0.921 | 0.210 |

Reading it: early layers (1-10) carry no usable refusal direction. From layer 14
the direction both removes refusal when ablated and induces it when added, but
for several layers the cost to harmless output is large (KL above 0.8). Layers
17-19 and 21 are the useful middle ground.

Layer 0 is skipped: every prompt ends in the same token, so the difference of
means there is exactly zero (the first run produced NaN for it).

## Outcome depends on the damage cutoff

| | Cutoff 0.1 (the paper's) | Cutoff 0.25 |
|---|---|---|
| Layer chosen | 12 | 21 |
| Harmful refusal, original to edited | 0.974 to 0.860 | 0.974 to **0.020** |
| Harmless refusal, original to edited | 0.114 to 0.103 | 0.114 to 0.036 |
| Harmless KL vs original | 0.069 | 0.209 |
| Edited vs ablation-hook KL (harmful / harmless) | 0.0012 / 0.0019 | 0.0017 / 0.0018 |

With the strict cutoff the selection falls back to a layer with a weak direction
and the edit barely changes behavior. With 0.25 the edit flips the first reply
tokens from "I'm sorry, but I can't assist" to the model starting to answer, at
the price of a visibly larger shift on harmless prompts.

## Same pipeline on Qwen2.5-3B

Qwen2.5-3B-Instruct (36 layers, d_model 2048, tied embeddings, pinned revision),
cutoff 0.25, 128 + 32 prompts, seed 0.

- Baseline refusal metric: 0.996 harmful, 0.102 harmless.
- Layers 1-18 carry no usable direction (ablated refusal 0.95-1.00). From layer
  19 the direction works, but with large harmless drift (KL 1.3-5.3 at layers
  19-24). Layers 25-27 are the useful range (KL 0.21-0.48).
- With the strict 0.1 cutoff, every layer that passes the damage filter is also
  one that does not remove refusal (best of them, layer 12, leaves 0.955), so the
  strict rule would again pick a useless layer.
- Chosen (cutoff 0.25): **layer 25**. First-token refusal 0.996 to **0.114**;
  harmless KL 0.20; edit equals hook (KL 0.001 to 0.003); the untied `lm_head`
  persisted through save and reload.
- **Soft refusals.** The edited model's replies often open with a hedge such as
  "While cheating can be tempting, it's important to remember..." or "While I can
  provide some guidance... it's important to note". The first-token refusal
  metric does not count these, so it overstates how much refusal was removed. A
  generation-based score is needed (open question below).

### Generation-based scoring on the 3B (`eval_gen.py`)

100 harmful and 100 harmless held-out prompts (never in the training split),
greedy decoding, first 64 new tokens, each reply sorted by string matching into
**hard** refusal, **soft** (hedge or caveat, no refusal phrase) or **comply**.

| Model | Prompts | Hard refusal | Soft | Comply |
|---|---|---|---|---|
| Original 3B | harmful | 98 | 0 | 2 |
| | harmless | 2 | 3 | 95 |
| Our edit (layer 25) | harmful | **2** | 73 | 25 |
| | harmless | 1 | 8 | 91 |
| OBLITERATUS `basic` | harmful | **100** | 0 | 0 |
| | harmless | 7 | 5 | 88 |

- Our edit removed outright refusals (98 to 2). The first-token metric said
  refusal fell to 0.114; the generation view is consistent on hard refusals.
- The edited model's non-refusals are mostly **soft**: it starts engaging with the
  request, often with a caveat ("While I can provide..., it's important to note").
  The classifier cannot tell hedged engagement from a real refusal-by-redirect,
  and it does not judge whether a reply contains harmful detail. So "73 soft"
  means "no longer an outright refusal", not "fully complied".
- OBLITERATUS `basic` left all 100 harmful prompts refused and refused 7 harmless
  prompts, so on this scorer its default run had no refusal-removing effect on
  the 3B (its own report said 0.90).
- Over-refusal on harmless prompts did not get worse for our edit (2 to 1 hard).

Limits: string matching, 64 tokens, one seed, and no human or model grader.
Replies are saved on the run machine for reading and are not stored in git.

## What the checks show

- **The edit equals the hook.** Orthogonalizing the weights reproduces the
  ablation hook within bf16 noise (KL about 0.002). The math is also unit-tested
  (`tests/test_math.py`: a weight edit equals projecting the layer's output).
- **Untying `lm_head` persisted** through save and reload: the tied-embeddings
  setting is false, `lm_head` is a separate tensor with different values, and the
  edited embedding has about zero component along the direction (max 6e-4).
- **Edited weights are not in git.** They stay on the test worker under the
  gitignored run folder.

## Caveats (do not over-read this)

- Refusal is measured from the **first reply token** only, on 32 validation
  prompts, with one seed. The generated text is checked on a few prompts, not
  scored.
- KL is also measured on the first token, so it understates or overstates damage
  to long replies in ways not yet measured. No capability benchmark was run.
- Only the last token position was tried. Earlier post-instruction positions
  (for example the end-of-turn marker) may give a cleaner direction at layers
  where the last token does not; that is untested.
- The 0.25 cutoff was chosen after seeing the sweep. A fair comparison needs the
  cutoff chosen in advance, or a held-out prompt set.

## Open questions this raises

1. Does a different token position give a clean direction with low KL?
2. How noisy is the sweep? Rerun with other seeds and a larger validation set.
3. Does a longer-reply check (generation, not first token) agree with the metric?
   On the 3B the replies suggest it does not fully, because of soft refusals.
