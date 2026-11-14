"""GPU check: real compute over split units, combined, equals the whole-model path.

    HACK_DATA_DIR=/path/to/data PYTHONPATH=src python scripts/gpu_equiv_check.py

Needs torch, transformers and a CUDA GPU. Expect max_abs_diff_vs_reference=0.
"""
import sys, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from hack3rzun1t3 import prompts as P
from hack3rzun1t3.hooks import ResidCapture
from hack3rzun1t3.harness.plan import make_units, split_ranges
from hack3rzun1t3.harness.gpu_compute import make_compute
from hack3rzun1t3.harness.partial import combine, mean_diff, validate

import os
D = Path(os.environ.get("HACK_DATA_DIR", "data"))
name = "Qwen/Qwen2.5-3B-Instruct"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16).to("cuda").eval()
n_layers = len(model.model.layers); d = model.config.hidden_size
harm = P.load_advbench(D / "advbench_harmful_behaviors.csv")
safe = P.load_alpaca(D / "alpaca_data_cleaned.json")
N = 8
h, _ = P.split(harm, N, 0, 0); s, _ = P.split(safe, N, 0, 0)

# reference: existing whole-model path
def collect(texts):
    with ResidCapture(model) as cap, torch.inference_mode():
        for t in texts:
            model(**tok(t, return_tensors="pt", add_special_tokens=False).to("cuda"))
    return cap.stacked()
ref = (collect(P.chat_texts(tok, h)).double().mean(1) - collect(P.chat_texts(tok, s)).double().mean(1))

# real compute, 3 layer ranges x 2 prompt batches
compute = make_compute(model, tok, h, s, batch_size=4)
ranges = split_ranges([1.0] * n_layers, [1, 1, 1])
units = make_units(ranges, N, 4)
t0 = time.time()
parts = {}
for u in units:
    p = compute(u); validate(p, range(u.layer_start, u.layer_end), d)
    parts.setdefault((u.layer_start, u.layer_end), []).append(p)
per_range = {r: combine(ps) for r, ps in parts.items()}
worst = 0.0
for (a, b), p in per_range.items():
    md = mean_diff(p)
    for L in range(a, b):
        diff = (torch.tensor(md[L]) - ref[L].cpu()).abs().max().item()
        worst = max(worst, diff)
print(f"units={len(units)} ranges={ranges} time={time.time()-t0:.1f}s max_abs_diff_vs_reference={worst:.3e}")
print("ref scale (max abs):", ref.abs().max().item())
