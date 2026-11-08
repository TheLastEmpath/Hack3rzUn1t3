"""Stage 1a: the whole abliteration recipe on a small model that fits the GPU.

    PYTHONPATH=src python -m hack3rzun1t3.run_1a

Env: HACK_DATA_DIR (downloaded prompt files, default ./data) and
HACK_RUN_DIR (outputs, default ./runs/1a). Both are gitignored.
"""

import argparse
import json
import os
import platform
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import prompts as P
from .directions import mean_diff, unit
from .edit import orthogonalize_model
from .hooks import Ablate, ResidCapture
from .refusal import first_token_logprobs, generate, kl, refusal_prob, refusal_token_ids
from .sweep import choose, sweep

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"


def log(msg):
    print(msg, flush=True)


def collect(model, tokenizer, texts):
    """(layers, n, d) residual stream entering each layer, last token."""
    device = next(model.parameters()).device
    with ResidCapture(model) as cap, torch.inference_mode():
        for t in texts:
            model(**tokenizer(t, return_tensors="pt", add_special_tokens=False).to(device))
    return cap.stacked()


def show_generations(model, tokenizer, texts, label, n=3):
    for t in texts[:n]:
        log(f"  [{label}] {generate(model, tokenizer, t)!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--revision", default=DEFAULT_REVISION)
    ap.add_argument("--n-train", type=int, default=128)
    ap.add_argument("--n-val", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layer-frac", type=float, default=0.8, help="only consider layers below this fraction of depth")
    ap.add_argument("--kl-max", type=float, default=0.1, help="max harmless-prompt KL a candidate layer may cause")
    args = ap.parse_args()

    data_dir = Path(os.environ.get("HACK_DATA_DIR", "data"))
    run_dir = Path(os.environ.get("HACK_RUN_DIR", "runs/1a"))
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = "cuda"

    # ---- data -------------------------------------------------------------
    adv_path = P.fetch(P.ADVBENCH_URL, data_dir / "advbench_harmful_behaviors.csv")
    alp_path = P.fetch(P.ALPACA_URL, data_dir / "alpaca_data_cleaned.json")
    harmful = P.load_advbench(adv_path)
    harmless = P.load_alpaca(alp_path)
    harm_train, harm_val = P.split(harmful, args.n_train, args.n_val, args.seed)
    safe_train, safe_val = P.split(harmless, args.n_train, args.n_val, args.seed)
    log(f"harmful pool {len(harmful)}, harmless pool (filtered) {len(harmless)}")

    # ---- model ------------------------------------------------------------
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, dtype=torch.bfloat16
    ).to(device)
    model.eval()
    n_layers = model.config.num_hidden_layers
    ids = refusal_token_ids(tok)
    log(f"layers {n_layers}, d_model {model.config.hidden_size}, refusal tokens {ids}")

    manifest = {
        "model": args.model, "revision": args.revision, "dtype": "bfloat16",
        "attn_implementation": model.config._attn_implementation,
        "seed": args.seed, "n_train": args.n_train, "n_val": args.n_val,
        "layer_frac": args.layer_frac, "kl_max": args.kl_max, "refusal_tokens": ids,
        "advbench_sha256": P.sha256_file(adv_path), "alpaca_sha256": P.sha256_file(alp_path),
        "torch": torch.__version__, "transformers": transformers.__version__,
        "python": platform.python_version(), "gpu": torch.cuda.get_device_name(0),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    h_tr, s_tr = P.chat_texts(tok, harm_train), P.chat_texts(tok, safe_train)
    h_va, s_va = P.chat_texts(tok, harm_val), P.chat_texts(tok, safe_val)

    # ---- calibrate the refusal metric before trusting it ------------------
    base_h = first_token_logprobs(model, tok, h_va)
    base_s = first_token_logprobs(model, tok, s_va)
    rp_h, rp_s = refusal_prob(base_h, ids).mean().item(), refusal_prob(base_s, ids).mean().item()
    log(f"baseline refusal prob: harmful {rp_h:.3f}, harmless {rp_s:.3f}")
    show_generations(model, tok, h_va, "harmful")
    show_generations(model, tok, s_va, "harmless")
    if rp_h - rp_s < 0.3:
        log("WARNING: the refusal metric barely separates harmful from harmless; results will be unreliable")

    # ---- phase 1+2: activations, directions, layer choice ------------------
    acts_h, acts_s = collect(model, tok, h_tr), collect(model, tok, s_tr)
    dirs = mean_diff(acts_h, acts_s)
    torch.save(dirs, run_dir / "directions.pt")
    rows = sweep(model, tok, dirs, h_va, s_va, base_s, ids, int(args.layer_frac * n_layers))
    (run_dir / "sweep.json").write_text(json.dumps(rows, indent=2))
    log("layer  ablated_harmful_refusal  added_harmless_refusal  harmless_kl")
    for r in rows:
        log(f"{r['layer']:>5}  {r['ablated_harmful_refusal']:>23.3f}  {r['added_harmless_refusal']:>22.3f}  {r['harmless_kl']:>11.4f}")
    best = choose(rows, args.kl_max)
    r_hat = unit(dirs[best])
    log(f"chosen layer {best}")

    # ---- the hook version (what the edit must reproduce) -------------------
    with Ablate(model, r_hat):
        hook_h = first_token_logprobs(model, tok, h_va)
        hook_s = first_token_logprobs(model, tok, s_va)
        show_generations(model, tok, h_va, "ablate-hook harmful")

    # ---- phase 3: edit the weights ----------------------------------------
    orthogonalize_model(model, r_hat)
    ed_h = first_token_logprobs(model, tok, h_va)
    ed_s = first_token_logprobs(model, tok, s_va)
    show_generations(model, tok, h_va, "edited harmful")
    show_generations(model, tok, s_va, "edited harmless")

    # ---- phase 4: validate -------------------------------------------------
    results = {
        "chosen_layer": best,
        "refusal_harmful": {
            "original": rp_h,
            "ablate_hook": refusal_prob(hook_h, ids).mean().item(),
            "edited": refusal_prob(ed_h, ids).mean().item(),
        },
        "refusal_harmless": {
            "original": rp_s,
            "ablate_hook": refusal_prob(hook_s, ids).mean().item(),
            "edited": refusal_prob(ed_s, ids).mean().item(),
        },
        "harmless_kl_vs_original": {
            "ablate_hook": kl(base_s, hook_s).mean().item(),
            "edited": kl(base_s, ed_s).mean().item(),
        },
        "edit_vs_hook_kl": {
            "harmful": kl(hook_h, ed_h).mean().item(),
            "harmless": kl(hook_s, ed_s).mean().item(),
        },
    }

    # ---- save, reload, and check the untie actually persisted --------------
    out = run_dir / "edited"
    model.save_pretrained(out)
    tok.save_pretrained(out)
    del model
    torch.cuda.empty_cache()
    re = AutoModelForCausalLM.from_pretrained(out, dtype=torch.bfloat16)
    emb, head = re.model.embed_tokens.weight, re.lm_head.weight
    results["reload"] = {
        "tie_word_embeddings": bool(re.config.tie_word_embeddings),
        "lm_head_is_separate_tensor": emb.data_ptr() != head.data_ptr() and not torch.equal(emb, head),
        "max_abs_embedding_component_along_r": (emb.float() @ r_hat.cpu()).abs().max().item(),
    }
    (run_dir / "results.json").write_text(json.dumps(results, indent=2))
    log(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
