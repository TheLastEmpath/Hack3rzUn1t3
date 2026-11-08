"""Split a model into work units sized to what each contributor can hold.

Layers are split into contiguous ranges (hidden states flow layer to layer, so a
range is the natural unit), in proportion to each slot's capacity. Prompts are
split into fixed batches so a unit is (layer range, prompt batch).
"""

import hashlib
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class WorkUnit:
    layer_start: int  # inclusive
    layer_end: int  # exclusive
    batch: int
    unit_id: str = ""

    def with_id(self) -> "WorkUnit":
        raw = f"{self.layer_start}:{self.layer_end}:{self.batch}".encode()
        return WorkUnit(self.layer_start, self.layer_end, self.batch, hashlib.sha256(raw).hexdigest()[:16])

    def to_json(self) -> dict:
        return asdict(self)


def split_ranges(layer_costs: list[float], capacities: list[float]) -> list[tuple[int, int]]:
    """Contiguous layer ranges, one per slot, with cost proportional to capacity.

    Every slot gets at least one layer. Returns [(start, end), ...] covering all layers.
    """
    n, k = len(layer_costs), len(capacities)
    if k == 0 or n < k:
        raise ValueError(f"need at least one layer per slot (layers={n}, slots={k})")
    if any(c <= 0 for c in capacities) or any(c < 0 for c in layer_costs):
        raise ValueError("capacities must be positive and layer costs non-negative")
    total, cap_total = sum(layer_costs), sum(capacities)
    bounds, start, acc, want = [], 0, 0.0, 0.0
    for slot in range(k):
        want += total * capacities[slot] / cap_total
        if slot == k - 1:
            end = n
        else:
            end = start + 1  # at least one layer
            acc_end = acc + layer_costs[start]
            limit = n - (k - 1 - slot)  # leave one layer for each later slot
            while end < limit and acc_end + layer_costs[end] / 2 <= want:
                acc_end += layer_costs[end]
                end += 1
            acc = acc_end
        bounds.append((start, end))
        start = end
    return bounds


def make_units(ranges: list[tuple[int, int]], n_prompts: int, batch_size: int) -> list[WorkUnit]:
    """One unit per (range, prompt batch), in a stable order."""
    if batch_size < 1 or n_prompts < 1:
        raise ValueError("n_prompts and batch_size must be positive")
    n_batches = -(-n_prompts // batch_size)
    return [WorkUnit(s, e, b).with_id() for (s, e) in ranges for b in range(n_batches)]
