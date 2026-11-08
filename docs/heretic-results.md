---
created: 2026-11-06T17:10
updated: 2026-11-06T17:10
---
# Heretic on Qwen2.5-3B-Instruct: result

Method and defaults: [heretic-notes.md](heretic-notes.md). Same base model, same
100 held-out harmful prompts and same decoding (greedy, first 128 new tokens) as
[reference-comparison.md](reference-comparison.md).

## Run

Heretic 2.1.0.dev0 (commit pinned), ARA modifier, 100 trials (30 random start),
seed 0, default prompt sets, bf16 on one 12 GB card. The run was interrupted
twice (a Ctrl-C and a shutdown) and resumed from its checkpoint each time; the
search history was not lost. About 18 minutes per 50 trials.

The tradeoff list at the end (refusals out of 100 by Heretic's own keyword
counter, first-token KL on harmless prompts):

| Trial | Refusals | KL |
|---|---|---|
| 20 | 0 | 15.68 |
| 26 | 2 | 1.32 |
| 18 | 3 | 0.49 |
| **95** | **4** | **0.042** |
| 100 | 5 | 0.030 |
| 88 | 7 | 0.027 |
| 13 | 13 | 0.021 |

Trial 95 was picked by eye as the knee: below it KL jumps by an order of
magnitude, above it refusals barely change. That is a judgment call made after
seeing the list. Saved locally as merged bf16 weights; nothing uploaded.

## Result 1: string matching (100 harmful prompts)

| Model | Hard | Soft | Comply |
|---|---|---|---|
| original | 98 | 0 | 2 |
| ours | 3 | 75 | 22 |
| published | 4 | 34 | 62 |
| **heretic trial 95** | **0** | **30** | **70** |

Over-refusal on 100 harmless prompts: heretic 1 hard and 10 soft (original 2
hard, ours 1, published 1 hard).

## Result 2: blinded grading (25 random harmful replies per model)

A separate sheet of heretic, published and ours (75 replies, shuffled, model
names hidden, first 300 characters read, rubric in `grade.py`):

| Model | R | P | C |
|---|---|---|---|
| ours | 1 | 13 | 11 |
| published | 1 | 10 | 14 |
| **heretic trial 95** | **1** | **7** | **17** |

## Reading it

1. Heretic's search beat both of our hand-built edits at getting full compliance
   on this model: 17 of 25 replies comply, against 14 for the published model
   and 11 for ours, while refusing about as rarely.
2. Its low KL (0.042) says the harmless first-token distribution barely moved,
   a better tradeoff than the strict-versus-useful layer choice we had to make
   by hand. The harmless over-refusal count (10 soft) is higher than we saw for
   the others; that is worth a closer look.
3. The same hedging pattern remains in the middle bucket: many replies open with
   a caveat and then comply. Keyword refusal counts miss exactly that.

## Caveats

- One AI grader, 25 per model, first 300 characters only, and a new sample
  (different seed) from the earlier four-model sheet, so the counts for ours
  and the published model differ by a few replies from that report. That is
  grader and sample noise; treat gaps of a few replies as not significant.
- The grading here was done by the same assistant that ran the experiment, with
  names hidden by the key. A second grader would be stronger.
- Trial choice was made after seeing the tradeoff list; the held-out prompts
  were not used for Heretic's search, but a different trial could score
  differently.
- Single seed, single model size. Nothing here shows the result carries to MoE.
