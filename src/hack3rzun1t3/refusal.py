"""Measuring refusal from the model's first reply token (one forward pass)."""

import torch

# Common first words of a refusal. Only words that are a single token are kept.
CANDIDATE_REFUSAL_WORDS = ["I", "Sorry", "As", "Unfortunately", "Apologies", "Im"]


def refusal_token_ids(tokenizer) -> dict[str, int]:
    ids = {}
    for w in CANDIDATE_REFUSAL_WORDS:
        enc = tokenizer.encode(w, add_special_tokens=False)
        if len(enc) == 1:
            ids[w] = enc[0]
    return ids


@torch.inference_mode()
def first_token_logprobs(model, tokenizer, texts: list[str]) -> torch.Tensor:
    """Log-probabilities of the first reply token, (n, vocab) fp32 on CPU."""
    device = next(model.parameters()).device
    out = []
    for t in texts:
        enc = tokenizer(t, return_tensors="pt", add_special_tokens=False).to(device)
        logits = model(**enc).logits[0, -1].float()
        out.append(torch.log_softmax(logits, dim=-1).cpu())
    return torch.stack(out)


def refusal_prob(logprobs: torch.Tensor, ids: dict[str, int]) -> torch.Tensor:
    """Probability mass on refusal-opening tokens, per prompt."""
    return logprobs[:, list(ids.values())].exp().sum(dim=-1)


def kl(p_logprobs: torch.Tensor, q_logprobs: torch.Tensor) -> torch.Tensor:
    """KL(p || q) per prompt for (n, vocab) log-probabilities."""
    return (p_logprobs.exp() * (p_logprobs - q_logprobs)).sum(dim=-1)


@torch.inference_mode()
def generate(model, tokenizer, text: str, max_new_tokens: int = 24) -> str:
    device = next(model.parameters()).device
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(out[0, enc["input_ids"].shape[1] :], skip_special_tokens=True)
