"""Pick the layer whose refusal direction works best (the Arditi et al. recipe).

For each candidate layer l, with r_l the difference-of-means direction there:
  - ablate r_hat_l everywhere, measure refusal on harmful validation prompts
    (lower is better: the refusal went away);
  - add raw r_l at layer l, measure refusal on harmless prompts (higher means
    the direction really carries refusal);
  - KL of the harmless distribution under ablation vs the untouched model (a
    damage check).
"""

from .directions import unit
from .hooks import Ablate, AddAtLayer
from .refusal import first_token_logprobs, kl, refusal_prob


def sweep(model, tokenizer, dirs, harm_val, safe_val, base_safe_logprobs, ids, max_layer):
    """`dirs` is (layers, d) raw directions. Returns one dict per candidate layer."""
    rows = []
    # Layer 0 is skipped: its input at the last token is the same token for every
    # prompt, so the difference of means is exactly zero (and unit() would be NaN).
    for layer in range(1, max_layer):
        r = dirs[layer]
        r_hat = unit(r)
        with Ablate(model, r_hat):
            harm_abl = refusal_prob(first_token_logprobs(model, tokenizer, harm_val), ids)
            safe_abl_lp = first_token_logprobs(model, tokenizer, safe_val)
        with AddAtLayer(model, layer, r):
            safe_add = refusal_prob(first_token_logprobs(model, tokenizer, safe_val), ids)
        rows.append(
            {
                "layer": layer,
                "ablated_harmful_refusal": harm_abl.mean().item(),
                "added_harmless_refusal": safe_add.mean().item(),
                "harmless_kl": kl(base_safe_logprobs, safe_abl_lp).mean().item(),
            }
        )
    return rows


def choose(rows, kl_max=0.1):
    """Lowest ablated-harmful refusal among layers that did not damage harmless output."""
    ok = [r for r in rows if r["harmless_kl"] < kl_max] or rows
    return min(ok, key=lambda r: r["ablated_harmful_refusal"])["layer"]
