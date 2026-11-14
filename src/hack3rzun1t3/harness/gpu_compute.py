"""Real compute for a worker: run a prompt batch, return per-layer activation sums.

This is the only harness module that needs torch and transformers, so it is not
imported by the package. A worker builds a `compute(unit) -> Partial` with
`make_compute(...)` and hands it to `worker.run`.

For a unit (layer_start, layer_end, batch) it runs each prompt of that batch
through the model one at a time (no padding, so the last token is index -1) and
adds up the residual stream entering each layer in the range, at the last token.
The forward pass stops right after the last wanted layer, so layers past the
range cost nothing.
"""

from __future__ import annotations

from .partial import Partial


class _Stop(Exception):
    """Raised by a hook to end the forward pass early."""


def make_compute(model, tokenizer, harmful: list[str], harmless: list[str], batch_size: int):
    """Return compute(unit) -> Partial for a loaded HF decoder-only model.

    `harmful` and `harmless` are raw prompts (chat template applied here).
    Unit `batch` b covers prompts [b*batch_size, (b+1)*batch_size) of each class.
    """
    import torch

    from ..prompts import chat_texts

    layers = model.model.layers
    device = next(model.parameters()).device
    texts = {
        "harmful": chat_texts(tokenizer, harmful),
        "harmless": chat_texts(tokenizer, harmless),
    }

    def sums_for(class_texts, start, end):
        acc = {i: None for i in range(start, end)}
        count = 0
        handles = []

        def make(i):
            def hook(module, args, kwargs):
                x = args[0] if args else kwargs["hidden_states"]
                v = x[0, -1].detach().double().cpu()
                acc[i] = v if acc[i] is None else acc[i] + v
                if i == end - 1 and end < len(layers):
                    raise _Stop

            return hook

        for i in range(start, end):
            handles.append(layers[i].register_forward_pre_hook(make(i), with_kwargs=True))
        try:
            with torch.inference_mode():
                for t in class_texts:
                    enc = tokenizer(t, return_tensors="pt", add_special_tokens=False).to(device)
                    try:
                        model(**enc)
                    except _Stop:
                        pass
                    count += 1
        finally:
            for h in handles:
                h.remove()
        return {i: v.tolist() for i, v in acc.items()}, count

    def compute(unit) -> Partial:
        lo, hi = unit.batch * batch_size, (unit.batch + 1) * batch_size
        hm = texts["harmful"][lo:hi]
        hl = texts["harmless"][lo:hi]
        if not hm or not hl:
            raise ValueError(f"batch {unit.batch} is empty")
        h_sums, n_h = sums_for(hm, unit.layer_start, unit.layer_end)
        l_sums, n_l = sums_for(hl, unit.layer_start, unit.layer_end)
        return Partial(h_sums, l_sums, n_h, n_l)

    return compute
