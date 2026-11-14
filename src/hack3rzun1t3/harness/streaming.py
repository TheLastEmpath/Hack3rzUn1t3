"""Layer streaming: run a model that does not fit the GPU, one layer at a time.

Only the token embedding and ONE decoder layer are on the GPU at once. For a
prompt batch the hidden states (small) stay on the GPU while each layer's
weights are read from the safetensors files, applied to every prompt, and freed.
So one batch costs one read of layers 0..end-1, however many prompts it holds.

Needs torch, transformers and safetensors; workers import it, the harness core
does not. Works for Llama-style decoder-only models (Qwen2, Llama, Mistral):
weights named model.embed_tokens.* and model.layers.{i}.*.

The residual entering layer i is captured at the last token (prompts run one at
a time, no padding), matching `gpu_compute` and `hooks.ResidCapture`. Layers
past the wanted range, the final norm and the LM head are never loaded.
"""

from __future__ import annotations

import json
from pathlib import Path

from .partial import Partial


class WeightIndex:
    """Which safetensors file holds each tensor, and a lazy per-tensor reader."""

    def __init__(self, model_dir: Path):
        self.dir = Path(model_dir)
        index = self.dir / "model.safetensors.index.json"
        if index.exists():
            self.files = json.loads(index.read_text())["weight_map"]
        else:
            from safetensors import safe_open

            single = self.dir / "model.safetensors"
            with safe_open(str(single), framework="pt") as f:
                self.files = {k: "model.safetensors" for k in f.keys()}

    def read(self, prefix: str, device) -> dict:
        """All tensors whose name starts with `prefix`, keyed without it, on `device`."""
        from safetensors import safe_open

        by_file: dict[str, list[str]] = {}
        for name, fname in self.files.items():
            if name.startswith(prefix):
                by_file.setdefault(fname, []).append(name)
        out = {}
        for fname, names in by_file.items():
            with safe_open(str(self.dir / fname), framework="pt", device=str(device)) as f:
                for n in names:
                    out[n[len(prefix):]] = f.get_tensor(n)
        if not out:
            raise KeyError(f"no tensors under {prefix!r}")
        return out


class StreamingModel:
    def __init__(self, model_dir, device="cuda", dtype=None):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM

        self.torch = torch
        self.device = torch.device(device)
        self.dtype = dtype or torch.bfloat16
        self.weights = WeightIndex(model_dir)
        config = AutoConfig.from_pretrained(str(model_dir))
        config._attn_implementation = "sdpa"
        with torch.device("meta"):
            skeleton = AutoModelForCausalLM.from_config(config, dtype=self.dtype)
        self.config = config
        self.n_layers = config.num_hidden_layers
        self.d_model = config.hidden_size
        self._layer_cls_args = skeleton.model.layers[0]
        self._skeleton = skeleton
        emb = self.weights.read("model.embed_tokens.", self.device)
        self.embed_weight = emb["weight"].to(self.dtype)
        with torch.device(self.device):
            self.rotary = type(skeleton.model.rotary_emb)(config)

    def load_layer(self, i: int):
        import copy

        layer = copy.deepcopy(self._skeleton.model.layers[i])  # meta tensors, no memory
        layer = layer.to_empty(device=self.device)
        sd = self.weights.read(f"model.layers.{i}.", self.device)
        layer.load_state_dict({k: v.to(self.dtype) for k, v in sd.items()}, strict=True)
        return layer.eval()

    def residual_sums(self, token_ids: list, start: int, end: int):
        """For each prompt (list of token-id lists): last-token residual entering
        layers start..end-1, summed over prompts in float64. Returns ({layer: list}, count)."""
        torch = self.torch
        if not (0 <= start < end <= self.n_layers):
            raise ValueError("bad layer range")
        with torch.inference_mode():
            hs, pos = [], []
            for ids in token_ids:
                t = torch.tensor([ids], device=self.device)
                h = torch.nn.functional.embedding(t, self.embed_weight)
                p = torch.arange(len(ids), device=self.device).unsqueeze(0)
                hs.append(h)
                pos.append((p, self.rotary(h, p)))
            acc = {}
            for i in range(end):
                if i >= start:
                    acc[i] = sum(h[0, -1].double().cpu() for h in hs)
                if i == end - 1:
                    break  # residual entering the last wanted layer is all we need
                layer = self.load_layer(i)
                for j, h in enumerate(hs):
                    p, pe = pos[j]
                    out = layer(h, attention_mask=None, position_ids=p, position_embeddings=pe)
                    hs[j] = out[0] if isinstance(out, tuple) else out
                del layer
        return {i: v.tolist() for i, v in acc.items()}, len(token_ids)


def make_streaming_compute(model_dir, tokenizer, harmful, harmless, batch_size, device="cuda"):
    """compute(unit) -> Partial using layer streaming; same contract as gpu_compute."""
    from ..prompts import chat_texts

    sm = StreamingModel(model_dir, device)
    ids = {
        "harmful": [tokenizer(t, add_special_tokens=False)["input_ids"] for t in chat_texts(tokenizer, harmful)],
        "harmless": [tokenizer(t, add_special_tokens=False)["input_ids"] for t in chat_texts(tokenizer, harmless)],
    }

    def compute(unit) -> Partial:
        lo, hi = unit.batch * batch_size, (unit.batch + 1) * batch_size
        hm, hl = ids["harmful"][lo:hi], ids["harmless"][lo:hi]
        if not hm or not hl:
            raise ValueError(f"batch {unit.batch} is empty")
        h_sums, n_h = sm.residual_sums(hm, unit.layer_start, unit.layer_end)
        l_sums, n_l = sm.residual_sums(hl, unit.layer_start, unit.layer_end)
        return Partial(h_sums, l_sums, n_h, n_l)

    return compute
