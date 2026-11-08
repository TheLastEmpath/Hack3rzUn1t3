"""Simulated runs of the coordinator with honest, faulty and slow workers."""

import random
import tempfile
import unittest
from pathlib import Path

from hack3rzun1t3.harness import partial as PT
from hack3rzun1t3.harness.coordinator import Coordinator
from hack3rzun1t3.harness.plan import make_units
from hack3rzun1t3.harness.store import Store

D, N_PROMPTS, BATCH = 3, 8, 4
RANGES = [(0, 2), (2, 4)]


def truth():
    """Fixed fake last-token activations: acts[class][layer][prompt] -> vector."""
    rng = random.Random(7)
    return {c: {layer: [[rng.gauss(0, 1) for _ in range(D)] for _ in range(N_PROMPTS)]
                for layer in range(4)} for c in ("harmful", "harmless")}


def honest(acts, unit):
    lo, hi = unit.batch * BATCH, min(N_PROMPTS, (unit.batch + 1) * BATCH)
    sums = {c: {layer: [sum(v[i] for v in acts[c][layer][lo:hi]) for i in range(D)]
                for layer in range(unit.layer_start, unit.layer_end)} for c in acts}
    return PT.Partial(sums["harmful"], sums["harmless"], hi - lo, hi - lo)


def drive(coord, workers, acts, bad=None, max_steps=200):
    """Round-robin workers until done. `bad` maps worker -> function altering a result."""
    for step in range(max_steps):
        if coord.done():
            return step
        for w in workers:
            unit = coord.claim(w)
            if unit is None:
                continue
            result = honest(acts, unit)
            if bad and w in bad:
                result = bad[w](result)
            coord.submit(w, unit.unit_id, result)
    raise AssertionError("did not finish")


def expected(acts):
    whole = PT.Partial(
        {l: [sum(v[i] for v in acts["harmful"][l]) for i in range(D)] for l in range(4)},
        {l: [sum(v[i] for v in acts["harmless"][l]) for i in range(D)] for l in range(4)},
        N_PROMPTS, N_PROMPTS)
    return PT.mean_diff(whole)


def corrupt(p):
    return PT.Partial({k: [x + 5.0 for x in v] for k, v in p.harmful.items()}, p.harmless,
                      p.n_harmful, p.n_harmless)


def close(a, b):
    return all(abs(x - y) < 1e-9 for k in a for x, y in zip(a[k], b[k]))


class CoordinatorRuns(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / "done.jsonl"
        self.units = make_units(RANGES, N_PROMPTS, BATCH)
        self.acts = truth()

    def tearDown(self):
        self.dir.cleanup()

    def coord(self, **kw):
        return Coordinator(self.units, Store(self.path), D, **kw)

    def test_honest_run_matches_ground_truth(self):
        c = self.coord()
        drive(c, ["w1", "w2"], self.acts)
        self.assertTrue(close(PT.mean_diff(c.combined()), expected(self.acts)))

    def test_full_replication_catches_one_bad_worker(self):
        c = self.coord(replicate_fraction=1.0)
        drive(c, ["w1", "w2", "w3"], self.acts, bad={"w2": corrupt})
        self.assertTrue(close(PT.mean_diff(c.combined()), expected(self.acts)))
        self.assertEqual(c.flagged, {"w2"})

    def test_no_replication_does_not_catch_plausible_corruption(self):
        c = self.coord(replicate_fraction=0.0)
        drive(c, ["w1", "w2"], self.acts, bad={"w2": corrupt})
        self.assertFalse(close(PT.mean_diff(c.combined()), expected(self.acts)))

    def test_malformed_result_is_rejected_and_redone(self):
        c = self.coord()
        unit = c.claim("w1")
        bad = PT.Partial({0: [1.0]}, {0: [1.0]}, 4, 4)
        self.assertEqual(c.submit("w1", unit.unit_id, bad), "rejected")
        self.assertEqual(c.rejected[0][0], "w1")
        drive(c, ["w1"], self.acts)
        self.assertTrue(c.done())

    def test_resume_after_kill_redoes_only_unfinished(self):
        c = self.coord()
        for _ in range(3):
            u = c.claim("w1")
            c.submit("w1", u.unit_id, honest(self.acts, u))
        c2 = self.coord()  # fresh process, same file
        self.assertEqual(len(c2.store.done), 3)
        drive(c2, ["w1"], self.acts)
        self.assertTrue(close(PT.mean_diff(c2.combined()), expected(self.acts)))

    def test_lease_expiry_reassigns_a_stalled_unit(self):
        now = [0.0]
        c = self.coord(lease_seconds=10, clock=lambda: now[0])
        stalled = c.claim("w1")  # w1 takes it and vanishes
        self.assertNotEqual(c.claim("w2").unit_id, stalled.unit_id)
        now[0] = 11.0
        self.assertEqual(c.claim("w3").unit_id, stalled.unit_id)

    def test_worker_never_gets_same_unit_twice(self):
        c = self.coord(replicate_fraction=1.0)
        seen = set()
        while (u := c.claim("w1")) is not None:
            self.assertNotIn(u.unit_id, seen)
            seen.add(u.unit_id)
            c.submit("w1", u.unit_id, honest(self.acts, u))
        self.assertFalse(c.done())  # needs a second worker for the replicas


if __name__ == "__main__":
    unittest.main()
