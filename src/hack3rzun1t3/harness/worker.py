"""Worker loop: claim a unit, compute it, submit, repeat until the run is done.

`compute(unit) -> Partial` is the only GPU-specific part and is passed in, so the
loop is tested here with fake compute and the real streaming engine plugs in later.
A compute failure is reported nowhere and the unit's lease simply expires, which
is the same path as a crashed or vanished worker.
"""

from __future__ import annotations

import time

from . import net
from .plan import WorkUnit


def run(base: str, worker: str, compute, key: bytes | None = None,
        idle_seconds: float = 1.0, max_idle: int = 600, sleep=time.sleep) -> dict:
    """Returns counts: {"accepted", "waiting", "rejected", "failed"}."""
    counts = {"accepted": 0, "waiting": 0, "rejected": 0, "failed": 0}
    idle = 0
    while True:
        unit, done = net.claim(base, worker, key)
        if unit is None:
            if done:
                return counts
            idle += 1
            if idle > max_idle:
                return counts
            sleep(idle_seconds)
            continue
        idle = 0
        try:
            result = compute(unit)
        except Exception:
            counts["failed"] += 1
            continue
        counts[net.submit(base, worker, unit.unit_id, result, key)] += 1
