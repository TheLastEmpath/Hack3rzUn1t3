---
created: 2026-11-08T13:21
updated: 2026-11-13T20:42
---
# Pick up here

Where the project stands and what to do next, for anyone continuing the work.

## What exists

- **Stage 0** measurement scripts (`scripts/stage0/`): see [stage0.md](stage0.md).
- **Heretic run** on Qwen2.5-3B, scored with string-match buckets and blinded
  grading: see [heretic-results.md](heretic-results.md) and
  [heretic-notes.md](heretic-notes.md).
- **Harness core** (`src/hack3rzun1t3/harness/`, torch-free): work units,
  exact-combining partial sums, append-only checkpoint, coordinator with leases
  and replication, HTTP layer with optional HMAC signing, worker loop. Design in
  [harness-design.md](harness-design.md).

Run the tests (no GPU needed):

```bash
PYTHONPATH=src python3 -m unittest tests.test_harness tests.test_coordinator tests.test_net tests.test_worker
```

`tests/test_math.py` needs torch and a GPU machine.

## Machines

Docs call the single-GPU development machine the **test worker** (one 12 GB GPU,
15 GB RAM, SATA SSD). The **coordinator** is whichever machine runs the harness
server. Any contributor machine is a **worker**.

## Next steps (not started unless noted)

1. Layer streaming (load one layer range at a time), then a 7B run. Real compute for
   models that fit one card is done: `harness/gpu_compute.py`, checked by
   `scripts/gpu_equiv_check.py`.
2. Shard editor and layer selection (needs a GPU machine).
3. Per-worker public-key signatures and network access rules.
4. MoE expert-group streaming and a small MoE rehearsal.
5. A second grader, and a look at why Heretic over-refused slightly more on
   harmless prompts.
6. Contributor terms.

## Unverified

Facts about the target model (Kimi K3) and its license are unverified. Check
before relying on them.
