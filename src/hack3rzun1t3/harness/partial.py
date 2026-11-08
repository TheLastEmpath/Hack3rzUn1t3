"""Per-batch partial sums for the refusal direction, and how they combine.

A contributor returns, for its prompt batch and layers, the SUM of last-token
activations per class plus the prompt count. Sums and counts add across
batches, so the combined mean does not depend on how prompts were split. Sums
are added in sorted batch order with math.fsum so the result is reproducible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class Partial:
    """layer -> summed activation vector, for each class, and the prompt counts."""

    harmful: dict[int, list[float]]
    harmless: dict[int, list[float]]
    n_harmful: int
    n_harmless: int


def validate(p: Partial, layers: range, d_model: int) -> None:
    """Reject results with wrong shape, missing layers, bad counts or non-finite values."""
    if p.n_harmful < 1 or p.n_harmless < 1:
        raise ValueError("counts must be positive")
    for name, sums in (("harmful", p.harmful), ("harmless", p.harmless)):
        if set(sums) != set(layers):
            raise ValueError(f"{name}: layers {sorted(sums)} != expected {list(layers)}")
        for layer, vec in sums.items():
            if len(vec) != d_model:
                raise ValueError(f"{name} layer {layer}: length {len(vec)} != {d_model}")
            if not all(math.isfinite(x) for x in vec):
                raise ValueError(f"{name} layer {layer}: non-finite value")


def combine(partials: list[Partial]) -> Partial:
    """Add partials covering the same layers; order-independent via sorted fsum."""
    if not partials:
        raise ValueError("nothing to combine")
    layers = set(partials[0].harmful)
    out = {}
    for name in ("harmful", "harmless"):
        acc = {}
        for layer in layers:
            vecs = [getattr(p, name)[layer] for p in partials]
            acc[layer] = [math.fsum(sorted(col)) for col in zip(*vecs)]
        out[name] = acc
    return Partial(out["harmful"], out["harmless"],
                   sum(p.n_harmful for p in partials), sum(p.n_harmless for p in partials))


def mean_diff(p: Partial) -> dict[int, list[float]]:
    """Per-layer (mean harmful - mean harmless), the unnormalized refusal direction."""
    return {
        layer: [h / p.n_harmful - l / p.n_harmless for h, l in zip(p.harmful[layer], p.harmless[layer])]
        for layer in p.harmful
    }


def cosine(a: list[float], b: list[float]) -> float:
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def agrees(a: Partial, b: Partial, min_cosine: float = 0.999, max_rel_err: float = 0.01) -> bool:
    """Do two replicas of the same unit agree? Checks every layer and class."""
    for name in ("harmful", "harmless"):
        for layer, va in getattr(a, name).items():
            vb = getattr(b, name)[layer]
            diff = math.sqrt(sum((x - y) ** 2 for x, y in zip(va, vb)))
            scale = math.sqrt(sum(x * x for x in va)) or 1.0
            if cosine(va, vb) < min_cosine or diff / scale > max_rel_err:
                return False
    return (a.n_harmful, a.n_harmless) == (b.n_harmful, b.n_harmless)


def to_json(p: Partial) -> dict:
    """JSON-safe form (layer keys become strings)."""
    return {
        "harmful": {str(k): v for k, v in p.harmful.items()},
        "harmless": {str(k): v for k, v in p.harmless.items()},
        "n_harmful": p.n_harmful,
        "n_harmless": p.n_harmless,
    }


def from_json(d: dict) -> Partial:
    return Partial(
        {int(k): v for k, v in d["harmful"].items()},
        {int(k): v for k, v in d["harmless"].items()},
        d["n_harmful"],
        d["n_harmless"],
    )
