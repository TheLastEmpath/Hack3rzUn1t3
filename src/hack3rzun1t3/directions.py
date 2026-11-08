"""The linear algebra of abliteration. Pure tensor functions, no model needed.

d = d_model, r_hat = unit refusal direction of shape (d,).
"""

import torch


def mean_diff(acts_harmful: torch.Tensor, acts_harmless: torch.Tensor) -> torch.Tensor:
    """Refusal direction (unnormalized) per layer.

    Inputs are (layers, n_prompts, d). Output is (layers, d): the mean harmful
    activation minus the mean harmless activation.
    """
    return acts_harmful.mean(dim=1) - acts_harmless.mean(dim=1)


def unit(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm(dim=-1, keepdim=True)


def project_out(x: torch.Tensor, r_hat: torch.Tensor) -> torch.Tensor:
    """x with its component along r_hat removed (last dim is d)."""
    return x - (x @ r_hat).unsqueeze(-1) * r_hat


def orthogonalize_output(weight: torch.Tensor, r_hat: torch.Tensor) -> torch.Tensor:
    """Edit a Linear weight (out=d, in) so its output has no r_hat component.

    W' = (I - r r^T) W, so W' x = project_out(W x, r).
    """
    w = weight.float()
    r = r_hat.float()
    return (w - torch.outer(r, r @ w)).to(weight.dtype)


def orthogonalize_embedding(weight: torch.Tensor, r_hat: torch.Tensor) -> torch.Tensor:
    """Edit an embedding table (vocab, d) so every embedding has no r_hat component."""
    w = weight.float()
    r = r_hat.float()
    return (w - torch.outer(w @ r, r)).to(weight.dtype)
