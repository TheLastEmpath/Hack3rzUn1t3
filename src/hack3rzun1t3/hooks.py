"""Forward hooks on a HF decoder-only model: capture, ablate, add.

Hooks (not `output_hidden_states`) because the last hidden state HF returns is
taken after the final norm, not the raw residual stream, and because the same
interface is what the stage 1b streaming engine will expose.

All hooks assume batch size 1 (no padding), so "last token" is index -1.
"""

import torch

from .directions import project_out


def decoder_layers(model):
    return model.model.layers


class ResidCapture:
    """Record the residual stream entering each layer, at the last token.

    After the `with` block, `.stacked()` is (layers, n_prompts, d) in fp32 on CPU.
    """

    def __init__(self, model):
        self.layers = decoder_layers(model)
        self.store = [[] for _ in self.layers]
        self.handles = []

    def _make(self, i):
        def hook(module, args, kwargs):
            x = args[0] if args else kwargs["hidden_states"]
            self.store[i].append(x[0, -1].detach().float().cpu())

        return hook

    def __enter__(self):
        for i, layer in enumerate(self.layers):
            self.handles.append(layer.register_forward_pre_hook(self._make(i), with_kwargs=True))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()

    def stacked(self) -> torch.Tensor:
        return torch.stack([torch.stack(s) for s in self.store])


class Ablate:
    """Project r_hat out of everything written into the residual stream.

    Hooks the outputs of the token embedding and every attention `o_proj` and
    MLP `down_proj`. This is mathematically the same as editing those weights
    (phase 3), which is what the equivalence test relies on.
    """

    def __init__(self, model, r_hat: torch.Tensor):
        self.model = model
        self.r_hat = r_hat.detach().float().to(next(model.parameters()).device)
        self.handles = []

    def _hook(self, module, args, output):
        out = project_out(output.float(), self.r_hat)
        return out.to(output.dtype)

    def __enter__(self):
        m = self.model.model
        targets = [m.embed_tokens]
        for layer in m.layers:
            targets += [layer.self_attn.o_proj, layer.mlp.down_proj]
        for t in targets:
            self.handles.append(t.register_forward_hook(self._hook))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()


class AddAtLayer:
    """Add `vector` (raw, not normalized) to the residual stream entering one layer."""

    def __init__(self, model, layer: int, vector: torch.Tensor):
        self.layer = decoder_layers(model)[layer]
        self.vector = vector.detach().float().to(next(model.parameters()).device)
        self.handle = None

    def _hook(self, module, args, kwargs):
        x = args[0] if args else kwargs["hidden_states"]
        x = (x.float() + self.vector).to(x.dtype)
        if args:
            return (x,) + tuple(args[1:]), kwargs
        kwargs = dict(kwargs, hidden_states=x)
        return args, kwargs

    def __enter__(self):
        self.handle = self.layer.register_forward_pre_hook(self._hook, with_kwargs=True)
        return self

    def __exit__(self, *exc):
        self.handle.remove()
