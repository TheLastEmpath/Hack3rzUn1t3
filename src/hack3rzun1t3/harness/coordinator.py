"""Coordinator: hands out units, checks results, replicates a sample, records.

Pull model: workers call `claim` and `submit`; the coordinator never connects out.
In memory and single-threaded here; the network layer will wrap these two calls.

Rules:
- A unit is leased to one worker at a time; an expired lease is handed out again.
- A worker never gets the same unit twice, so replicas land on different workers.
- Units picked for replication (by hash, rate `replicate_fraction`) need two
  agreeing results. On disagreement a third result decides by majority and the
  odd worker is flagged. Other units need one valid result.
- Accepted results are written to the Store, so a restart resumes.
"""

import hashlib
import time
from dataclasses import dataclass, field

from . import partial as PT
from .plan import WorkUnit
from .store import Store


@dataclass
class _State:
    results: dict[str, PT.Partial] = field(default_factory=dict)  # worker -> result
    lease: tuple[str, float] | None = None  # (worker, expires_at)
    need: int = 1
    flagged_workers: set = field(default_factory=set)


class Coordinator:
    def __init__(self, units: list[WorkUnit], store: Store, d_model: int,
                 replicate_fraction: float = 0.0, lease_seconds: float = 600.0,
                 clock=time.monotonic):
        self.units = {u.unit_id: u for u in units}
        self.order = [u.unit_id for u in units]
        self.store, self.d_model = store, d_model
        self.lease_seconds, self.clock = lease_seconds, clock
        self.flagged: set[str] = set()  # workers that returned a minority result
        self.rejected: list[tuple[str, str, str]] = []  # (worker, unit_id, reason)
        self.state = {uid: _State(need=self._need(uid, replicate_fraction)) for uid in self.order}

    @staticmethod
    def _need(unit_id: str, fraction: float) -> int:
        bucket = int(hashlib.sha256(unit_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        return 2 if bucket < fraction else 1

    def done(self) -> bool:
        return all(self.store.has(uid) for uid in self.order)

    def claim(self, worker: str) -> WorkUnit | None:
        now = self.clock()
        for uid in self.order:
            st = self.state[uid]
            if self.store.has(uid) or worker in st.results:
                continue
            if st.lease and st.lease[1] > now and st.lease[0] != worker:
                continue  # someone else is working on it
            st.lease = (worker, now + self.lease_seconds)
            return self.units[uid]
        return None

    def submit(self, worker: str, unit_id: str, result: PT.Partial) -> str:
        """Returns "rejected", "waiting" (needs a replica) or "accepted"."""
        unit = self.units[unit_id]
        st = self.state[unit_id]
        if self.store.has(unit_id):
            return "accepted"
        try:
            PT.validate(result, range(unit.layer_start, unit.layer_end), self.d_model)
        except ValueError as e:
            self.rejected.append((worker, unit_id, str(e)))
            return "rejected"
        st.results[worker] = result
        st.lease = None
        return self._settle(unit_id, st)

    def _settle(self, unit_id: str, st: _State) -> str:
        res = list(st.results.items())
        if st.need == 1:
            return self._accept(unit_id, res[0][1])
        if len(res) < 2:
            return "waiting"
        (w1, a), (w2, b) = res[0], res[1]
        if PT.agrees(a, b):
            return self._accept(unit_id, a)
        if len(res) < 3:
            st.need = 3
            return "waiting"
        for w, r in res:  # majority of three
            others = [x for x_w, x in res if x_w != w]
            if sum(PT.agrees(r, x) for x in others) >= 1:
                odd = [x_w for x_w, x in res if not PT.agrees(r, x)]
                self.flagged.update(odd)
                return self._accept(unit_id, r)
        self.flagged.update(w for w, _ in res)  # no two agree: trust none, redo
        st.results.clear()
        st.need = 3
        return "waiting"

    def _accept(self, unit_id: str, result: PT.Partial) -> str:
        self.store.record(unit_id, PT.to_json(result))
        return "accepted"

    def combined(self) -> PT.Partial:
        """Sum of one accepted result per unit, per layer range (all prompt batches)."""
        parts = [PT.from_json(self.store.done[uid]) for uid in self.order]
        by_range: dict[tuple[int, int], list[PT.Partial]] = {}
        for uid, p in zip(self.order, parts):
            u = self.units[uid]
            by_range.setdefault((u.layer_start, u.layer_end), []).append(p)
        merged = [PT.combine(ps) for _, ps in sorted(by_range.items())]
        return PT.Partial(
            {k: v for m in merged for k, v in m.harmful.items()},
            {k: v for m in merged for k, v in m.harmless.items()},
            merged[0].n_harmful, merged[0].n_harmless)
