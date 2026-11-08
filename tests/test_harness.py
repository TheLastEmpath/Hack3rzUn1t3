"""Tests for the torch-free harness core. Run with:

    PYTHONPATH=src python -m unittest discover -s tests
"""

import tempfile
import unittest
from pathlib import Path

from hack3rzun1t3.harness import partial as PT
from hack3rzun1t3.harness.plan import make_units, split_ranges
from hack3rzun1t3.harness.store import Store


class Ranges(unittest.TestCase):
    def check_cover(self, ranges, n):
        self.assertEqual(ranges[0][0], 0)
        self.assertEqual(ranges[-1][1], n)
        for (_, e), (s, _) in zip(ranges, ranges[1:]):
            self.assertEqual(e, s)
        self.assertTrue(all(e > s for s, e in ranges))

    def test_equal_slots_equal_layers(self):
        r = split_ranges([1.0] * 12, [1, 1, 1])
        self.check_cover(r, 12)
        self.assertEqual([e - s for s, e in r], [4, 4, 4])

    def test_bigger_slot_gets_more(self):
        r = split_ranges([1.0] * 12, [1, 2])
        self.check_cover(r, 12)
        self.assertGreater(r[1][1] - r[1][0], r[0][1] - r[0][0])

    def test_every_slot_gets_a_layer_even_when_skewed(self):
        r = split_ranges([1.0] * 4, [100, 1, 1, 1])
        self.check_cover(r, 4)

    def test_rejects_more_slots_than_layers(self):
        with self.assertRaises(ValueError):
            split_ranges([1.0] * 2, [1, 1, 1])

    def test_units_are_stable_and_unique(self):
        u = make_units([(0, 2), (2, 4)], n_prompts=10, batch_size=4)
        self.assertEqual(len(u), 6)
        self.assertEqual(len({x.unit_id for x in u}), 6)
        self.assertEqual(u, make_units([(0, 2), (2, 4)], 10, 4))


class Checkpoint(unittest.TestCase):
    def test_resume_and_pending(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "done.jsonl"
            units = make_units([(0, 2)], 8, 4)
            s = Store(path)
            s.record(units[0].unit_id, {"ok": 1})
            s2 = Store(path)
            self.assertTrue(s2.has(units[0].unit_id))
            self.assertEqual(s2.pending(units), [units[1]])

    def test_truncated_last_line_is_dropped(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "done.jsonl"
            s = Store(path)
            s.record("a", {"x": 1})
            s.record("b", {"x": 2})
            path.write_text(path.read_text()[:-15])  # cut the last line short
            s2 = Store(path)
            self.assertTrue(s2.has("a"))
            self.assertFalse(s2.has("b"))
            self.assertEqual(s2.dropped, 1)

    def test_corrupted_payload_fails_crc(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "done.jsonl"
            Store(path).record("a", {"x": 1})
            path.write_text(path.read_text().replace('"x": 1', '"x": 9'))
            self.assertFalse(Store(path).has("a"))


def part(h, l, nh=2, nl=2):
    return PT.Partial({0: h}, {0: l}, nh, nl)


class Aggregation(unittest.TestCase):
    def test_split_batches_match_one_big_batch(self):
        # four harmful and four harmless prompts, summed whole vs in two batches
        hs = [[1.0, 2.0], [3.0, 0.0], [5.0, 1.0], [0.0, 4.0]]
        ls = [[1.0, 1.0], [0.0, 2.0], [2.0, 2.0], [1.0, 0.0]]
        whole = PT.Partial({0: [sum(c) for c in zip(*hs)]}, {0: [sum(c) for c in zip(*ls)]}, 4, 4)
        a = part([sum(c) for c in zip(*hs[:2])], [sum(c) for c in zip(*ls[:2])])
        b = part([sum(c) for c in zip(*hs[2:])], [sum(c) for c in zip(*ls[2:])])
        for order in ([a, b], [b, a]):
            got = PT.mean_diff(PT.combine(order))
            want = PT.mean_diff(whole)
            self.assertEqual(got, want)

    def test_validate_rejects_bad_results(self):
        good = part([1.0, 2.0], [0.5, 0.5])
        PT.validate(good, range(0, 1), 2)
        with self.assertRaises(ValueError):
            PT.validate(part([1.0], [0.5, 0.5]), range(0, 1), 2)
        with self.assertRaises(ValueError):
            PT.validate(part([float("nan"), 1.0], [0.5, 0.5]), range(0, 1), 2)
        with self.assertRaises(ValueError):
            PT.validate(good, range(0, 2), 2)
        with self.assertRaises(ValueError):
            PT.validate(part([1.0, 2.0], [0.5, 0.5], nh=0), range(0, 1), 2)

    def test_replica_agreement(self):
        a = part([1.0, 2.0, 3.0], [1.0, 1.0, 1.0])
        close = part([1.0001, 2.0, 3.0], [1.0, 1.0, 1.0])
        far = part([3.0, 2.0, 1.0], [1.0, 1.0, 1.0])
        self.assertTrue(PT.agrees(a, close))
        self.assertFalse(PT.agrees(a, far))
        self.assertFalse(PT.agrees(a, part([1.0, 2.0, 3.0], [1.0, 1.0, 1.0], nh=3)))


if __name__ == "__main__":
    unittest.main()
