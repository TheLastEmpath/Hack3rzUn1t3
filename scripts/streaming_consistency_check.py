"""GPU check for a model larger than the card: streaming is self-consistent.

    HACK_DATA_DIR=/path/to/data HACK_MODEL_DIR=/path/to/model_snapshot \
      PYTHONPATH=src python scripts/streaming_consistency_check.py

The model need not fit the GPU. Runs the same prompts as one full-range unit per
batch and as three split layer ranges; the per-layer sums must be identical.
Also reports time and peak VRAM, and where the refusal direction is strongest.
"""
import math
import os
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from hack3rzun1t3 import prompts as P
from hack3rzun1t3.harness.partial import combine, mean_diff, validate
from hack3rzun1t3.harness.plan import make_units, split_ranges
from hack3rzun1t3.harness.streaming import make_streaming_compute

D = Path(os.environ.get("HACK_DATA_DIR", "data"))
MD = Path(os.environ["HACK_MODEL_DIR"])
N = int(os.environ.get("HACK_N", "8"))
tok = AutoTokenizer.from_pretrained(str(MD))
h, _ = P.split(P.load_advbench(D / "advbench_harmful_behaviors.csv"), N, 0, 0)
s, _ = P.split(P.load_alpaca(D / "alpaca_data_cleaned.json"), N, 0, 0)

compute = make_streaming_compute(MD, tok, h, s, batch_size=4)
import json
cfg = json.loads((MD / "config.json").read_text())
n_layers, d = cfg["num_hidden_layers"], cfg["hidden_size"]
print(f"model: {n_layers} layers, d_model {d}", flush=True)


def run(ranges, label):
    t0 = time.time()
    parts = {}
    for u in make_units(ranges, N, 4):
        p = compute(u)
        validate(p, range(u.layer_start, u.layer_end), d)
        parts.setdefault((u.layer_start, u.layer_end), []).append(p)
        print(f"  {label} unit {u.layer_start}-{u.layer_end} batch {u.batch} done at {time.time()-t0:.0f}s", flush=True)
    return {r: combine(ps) for r, ps in parts.items()}, time.time() - t0


full, t_full = run([(0, n_layers)], "full")
split, t_split = run(split_ranges([1.0] * n_layers, [1, 1, 1]), "split")
worst = 0.0
for r, p in split.items():
    for L in range(*r):
        for cls in ("harmful", "harmless"):
            a, b = getattr(p, cls)[L], getattr(full[(0, n_layers)], cls)[L]
            worst = max(worst, max(abs(x - y) for x, y in zip(a, b)))
md = mean_diff(full[(0, n_layers)])
norms = {L: math.sqrt(sum(x * x for x in v)) for L, v in md.items()}
best = max(norms, key=norms.get)
print(f"full pass {t_full:.0f}s, split passes {t_split:.0f}s")
print(f"max abs diff full vs split sums: {worst:.3e}")
print(f"peak VRAM: {torch.cuda.max_memory_allocated()/2**30:.2f} GiB (card has {torch.cuda.get_device_properties(0).total_memory/2**30:.1f})")
print(f"largest mean-difference norm at layer {best}: {norms[best]:.2f}")
