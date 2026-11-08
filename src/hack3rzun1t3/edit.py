"""Phase 3: orthogonalize the weights that write into the residual stream."""

import torch
from torch import nn

from .directions import orthogonalize_embedding, orthogonalize_output


def untie_lm_head(model):
    """Give lm_head its own copy of the embedding, so editing the embedding
    does not also change the output projection (Qwen2.5-1.5B ties them)."""
    if model.lm_head.weight is model.model.embed_tokens.weight:
        model.lm_head.weight = nn.Parameter(model.model.embed_tokens.weight.detach().clone())
    model.config.tie_word_embeddings = False


@torch.no_grad()
def orthogonalize_model(model, r_hat: torch.Tensor):
    """Edit in place: token embeddings, every o_proj and every down_proj.

    q/k/v and gate/up only read the residual stream, and lm_head is left alone.
    """
    untie_lm_head(model)
    r_hat = r_hat.to(model.model.embed_tokens.weight.device)
    emb = model.model.embed_tokens.weight
    emb.copy_(orthogonalize_embedding(emb, r_hat))
    for layer in model.model.layers:
        for proj in (layer.self_attn.o_proj, layer.mlp.down_proj):
            assert proj.bias is None, "output projection has a bias; it would need editing too"
            proj.weight.copy_(orthogonalize_output(proj.weight, r_hat))
