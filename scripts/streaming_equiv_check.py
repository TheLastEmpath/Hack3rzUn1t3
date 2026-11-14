"""GPU check: layer streaming equals the whole-model path, and report peak VRAM.

    HACK_DATA_DIR=/path/to/data HACK_MODEL_DIR=/path/to/model_snapshot \
      PYTHONPATH=src python scripts/streaming_equiv_check.py

Needs torch, transformers, safetensors and a CUDA GPU. Streaming is compared
with the in-memory path from gpu_compute on the same prompts.
"""
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from hack3rzun1t3 import prompts as P
from hack3rzun1t3.harness.gpu_compute import make_compute
from hack3rzun1t3.harness.partial import combine, mean_diff, validate
from hack3rzun1t3.harness.plan import make_units, split_ranges
from hack3rzun1t3.harness.streaming import make_streaming_compute

D = Path(os.environ.get("HACK_DATA_DIR", "data"))
MD = Path(os.environ["HACK_MODEL_DIR"])
N = int(os.environ.get("HACK_N", "8"))
tok = AutoTokenizer.from_pretrained(str(MD))
harm = P.load_advbench(D / "advbench_harmful_behaviors.csv")
safe = P.load_alpaca(D / "alpaca_data_cleaned.json")
h, _ = P.split(harm, N, 0, 0)
s, _ = P.split(safe, N, 0, 0)

model = AutoModelForCausalLM.from_pretrained(str(MD), dtype=torch.bfloat16).to("cuda").eval()
n_layers, d = len(model.model.layers), model.config.hidden_size
ref_c = make_compute(model, tok, h, s, batch_size=4)
units = make_units(split_ranges([1.0] * n_layers, [1, 1, 1]), N, 4)
ref = {}
for u in units:
    ref.setdefault((u.layer_start, u.layer_end), []).append(ref_c(u))
ref = {r: mean_diff(combine(ps)) for r, ps in ref.items()}
del model, ref_c
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

st_c = make_streaming_compute(MD, tok, h, s, batch_size=4)
t0 = time.time()
got = {}
for u in units:
    p = st_c(u)
    validate(p, range(u.layer_start, u.layer_end), d)
    got.setdefault((u.layer_start, u.layer_end), []).append(p)
worst = 0.0
for r, ps in got.items():
    md = mean_diff(combine(ps))
    for L in range(*r):
        worst = max(worst, (torch.tensor(md[L]) - torch.tensor(ref[r][L])).abs().max().item())
print(f"units={len(units)} time={time.time()-t0:.1f}s max_abs_diff_vs_in_memory={worst:.3e}")
print(f"peak VRAM during streaming: {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
