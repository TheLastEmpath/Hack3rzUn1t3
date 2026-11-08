#!/usr/bin/env python3
"""Local tests for stage 0 helpers (no network, no external tools needed)."""
import hashlib
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import hop_calc  # noqa: E402
import shard_check  # noqa: E402
import tailscale_check  # noqa: E402


class T(unittest.TestCase):
    def test_hop(self):
        payload, sec = hop_calc.hop(64, 256, 4096, 2, 1000)
        self.assertEqual(payload, 64 * 256 * 4096 * 2)
        self.assertAlmostEqual(sec, payload * 8 / 1e9)

    def test_classify(self):
        self.assertEqual(tailscale_check.classify("pong from a (100.1.1.1) via DERP(nyc) in 40ms"), "relayed")
        self.assertEqual(tailscale_check.classify("pong from a (100.1.1.1) via 192.0.2.5:41641 in 2ms"), "direct")
        self.assertEqual(tailscale_check.classify("timeout"), "unknown")

    def test_shard_check(self):
        with tempfile.TemporaryDirectory() as d:
            data = b"hello"
            open(os.path.join(d, "a.safetensors"), "wb").write(data)
            open(os.path.join(d, "b.safetensors"), "wb").write(b"corrupt")
            good = hashlib.sha256(data).hexdigest()
            oids = {"a.safetensors": good, "b.safetensors": good, "c.safetensors": good}
            ok, bad, missing = shard_check.check(d, oids)
            self.assertEqual((ok, bad, missing), (["a.safetensors"], ["b.safetensors"], ["c.safetensors"]))

    def test_inventory_json(self):
        import json
        out = subprocess.run([sys.executable, os.path.join(HERE, "node_inventory.py"), "--redact-host"],
                             capture_output=True, text=True, check=True).stdout
        self.assertIn("ram_bytes", json.loads(out))


if __name__ == "__main__":
    unittest.main()
