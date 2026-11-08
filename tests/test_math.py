"""CPU tests for the abliteration math and prompt handling. Run with:

    PYTHONPATH=src python -m unittest discover -s tests
"""

import unittest

import torch

from hack3rzun1t3 import prompts as P
from hack3rzun1t3.eval_gen import classify, tally
from hack3rzun1t3 import grade as G
from hack3rzun1t3.directions import (
    mean_diff, orthogonalize_embedding, orthogonalize_output, project_out, unit,
)


def rand_unit(d, seed=0):
    g = torch.Generator().manual_seed(seed)
    return unit(torch.randn(d, generator=g))


class DirectionMath(unittest.TestCase):
    def test_mean_diff_per_layer(self):
        harmful = torch.ones(2, 4, 3) * torch.tensor([3.0, 0.0, 0.0])
        harmless = torch.ones(2, 4, 3) * torch.tensor([1.0, 0.0, 0.0])
        out = mean_diff(harmful, harmless)
        self.assertEqual(out.shape, (2, 3))
        self.assertTrue(torch.allclose(out, torch.tensor([[2.0, 0, 0], [2.0, 0, 0]])))

    def test_unit_has_norm_one(self):
        self.assertAlmostEqual(unit(torch.randn(2, 5)).norm(dim=-1)[0].item(), 1.0, places=5)

    def test_project_out_removes_direction_and_is_idempotent(self):
        r, x = rand_unit(16), torch.randn(5, 16)
        y = project_out(x, r)
        self.assertLess((y @ r).abs().max().item(), 1e-5)
        self.assertTrue(torch.allclose(project_out(y, r), y, atol=1e-5))

    def test_linear_edit_equals_projecting_the_output(self):
        # W' x == project_out(W x): the weight edit and the hook are the same thing.
        d, n_in = 16, 8
        r, W, x = rand_unit(d), torch.randn(d, n_in), torch.randn(4, n_in)
        edited = x @ orthogonalize_output(W, r).T
        hooked = project_out(x @ W.T, r)
        self.assertTrue(torch.allclose(edited, hooked, atol=1e-5))

    def test_embedding_edit_equals_projecting_the_lookup(self):
        vocab, d = 10, 16
        r, E = rand_unit(d), torch.randn(vocab, d)
        ids = torch.tensor([1, 4, 7])
        self.assertTrue(
            torch.allclose(orthogonalize_embedding(E, r)[ids], project_out(E[ids], r), atol=1e-5)
        )

    def test_edit_keeps_dtype(self):
        r, W = rand_unit(8), torch.randn(8, 4).bfloat16()
        self.assertEqual(orthogonalize_output(W, r).dtype, torch.bfloat16)

    def test_direction_recovered_from_planted_signal(self):
        d, n = 32, 200
        r = rand_unit(d, seed=1)
        g = torch.Generator().manual_seed(2)
        harmless = torch.randn(1, n, d, generator=g)
        harmful = torch.randn(1, n, d, generator=g) + 5 * r
        found = unit(mean_diff(harmful, harmless)[0])
        self.assertGreater((found @ r).item(), 0.95)


class PromptHandling(unittest.TestCase):
    def test_alpaca_filter(self):
        rows = [
            {"instruction": "Give three tips for staying healthy this year.", "input": ""},
            {"instruction": "Give three tips for staying healthy this year.", "input": ""},
            {"instruction": "Summarize the following text carefully for me.", "input": "some text"},
            {"instruction": "Too short", "input": ""},
            {"instruction": "x" * 300, "input": ""},
        ]
        self.assertEqual(P.filter_alpaca(rows), ["Give three tips for staying healthy this year."])

    def test_split_is_disjoint_and_seeded(self):
        items = [f"p{i}" for i in range(100)]
        tr, va = P.split(items, 30, 10, seed=3)
        self.assertEqual((len(tr), len(va)), (30, 10))
        self.assertFalse(set(tr) & set(va))
        self.assertEqual((tr, va), P.split(items, 30, 10, seed=3))
        self.assertNotEqual(tr, P.split(items, 30, 10, seed=4)[0])

    def test_split_rejects_too_few(self):
        with self.assertRaises(ValueError):
            P.split(["a", "b"], 2, 2, seed=0)


class ReplyClassifier(unittest.TestCase):
    def test_hard_refusal(self):
        self.assertEqual(classify("I'm sorry, but I can't assist with that."), "hard")

    def test_soft_refusal_hedge(self):
        self.assertEqual(classify("While cheating can be tempting, it's a bad idea"), "soft")

    def test_compliance(self):
        self.assertEqual(classify("Step 1: gather the materials you need."), "comply")

    def test_hard_wins_over_soft(self):
        self.assertEqual(classify("However, I cannot help with that."), "hard")

    def test_tally_fractions(self):
        t = tally(["I'm sorry, no.", "Sure, here you go.", "While that is common, be careful."])
        self.assertEqual((t["hard"], t["comply"], t["soft"], t["n"]), (1, 1, 1, 3))


class BlindGrading(unittest.TestCase):
    def rows(self, tag, n=10):
        return [{"prompt": f"{tag}-p{i}", "reply": f"{tag}-reply-{i} " * 3} for i in range(n)]

    def test_sheet_hides_labels_and_keeps_counts(self):
        sheet, key = G.make_sheet({"a": self.rows("a"), "b": self.rows("b")}, n=4, seed=1)
        self.assertEqual(len(sheet), 8)
        self.assertEqual(sorted(key.values()), ["a"] * 4 + ["b"] * 4)
        self.assertTrue(all(set(x) == {"id", "prompt", "reply"} for x in sheet))

    def test_sheet_is_seeded_and_truncates(self):
        a = G.make_sheet({"m": self.rows("m")}, n=3, seed=2, chars=5)
        b = G.make_sheet({"m": self.rows("m")}, n=3, seed=2, chars=5)
        self.assertEqual(a, b)
        self.assertTrue(all(len(x["reply"]) <= 5 for x in a[0]))

    def test_tally_counts_and_validates(self):
        key = {"s0": "a", "s1": "a", "s2": "b"}
        out = G.tally({"s0": "R", "s1": "C", "s2": "P"}, key)
        self.assertEqual(out["a"], {"R": 1, "P": 0, "C": 1, "n": 2})
        with self.assertRaises(ValueError):
            G.tally({"s0": "R"}, key)
        with self.assertRaises(ValueError):
            G.tally({"s0": "R", "s1": "X", "s2": "P"}, key)


if __name__ == "__main__":
    unittest.main()
