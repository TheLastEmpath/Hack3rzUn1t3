"""Workers talking to the coordinator over real HTTP on loopback."""

import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from hack3rzun1t3.harness import net, partial as PT
from hack3rzun1t3.harness.coordinator import Coordinator
from hack3rzun1t3.harness.plan import make_units
from hack3rzun1t3.harness.store import Store

try:
    from test_coordinator import D, N_PROMPTS, BATCH, RANGES, close, expected, honest, truth
except ImportError:
    from tests.test_coordinator import D, N_PROMPTS, BATCH, RANGES, close, expected, honest, truth


class OverHttp(unittest.TestCase):
    def test_two_threaded_workers_finish_and_match_truth(self):
        acts = truth()
        with tempfile.TemporaryDirectory() as d:
            coord = Coordinator(make_units(RANGES, N_PROMPTS, BATCH), Store(Path(d) / "s.jsonl"), D)
            server = net.make_server(coord)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            base = f"http://127.0.0.1:{server.server_address[1]}"

            def work(name):
                while True:
                    unit, done = net.claim(base, name)
                    if unit is None:
                        if done:
                            return
                        continue
                    net.submit(base, name, unit.unit_id, honest(acts, unit))

            workers = [threading.Thread(target=work, args=(n,)) for n in ("a", "b")]
            [w.start() for w in workers]
            [w.join(timeout=30) for w in workers]
            server.shutdown()
            server.server_close()
            self.assertTrue(coord.done())
            self.assertTrue(close(PT.mean_diff(coord.combined()), expected(acts)))

    def test_bad_request_gets_400(self):
        with tempfile.TemporaryDirectory() as d:
            coord = Coordinator(make_units(RANGES, N_PROMPTS, BATCH), Store(Path(d) / "s.jsonl"), D)
            server = net.make_server(coord)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            base = f"http://127.0.0.1:{server.server_address[1]}"
            with self.assertRaises(urllib.error.HTTPError) as cm:
                net._post(base, "/claim", {})
            server.shutdown()
            server.server_close()
            self.assertEqual(cm.exception.code, 400)


class Signed(unittest.TestCase):
    def serve(self, d, keys):
        coord = Coordinator(make_units(RANGES, N_PROMPTS, BATCH), Store(Path(d) / "s.jsonl"), D)
        server = net.make_server(coord, keys=keys)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(lambda: (server.shutdown(), server.server_close()))
        return coord, f"http://127.0.0.1:{server.server_address[1]}"

    def test_signed_requests_work_and_others_get_401(self):
        acts = truth()
        with tempfile.TemporaryDirectory() as d:
            coord, base = self.serve(d, {"a": b"secret-a", "b": b"secret-b"})
            unit, _ = net.claim(base, "a", b"secret-a")
            self.assertIsNotNone(unit)
            for worker, key in (("a", None), ("a", b"wrong"), ("a", b"secret-b"), ("ghost", b"x")):
                with self.assertRaises(urllib.error.HTTPError) as cm:
                    net.claim(base, worker, key)
                self.assertEqual(cm.exception.code, 401)
            self.assertEqual(net.submit(base, "a", unit.unit_id, honest(acts, unit), b"secret-a"), "accepted")

    def test_cannot_submit_as_another_worker(self):
        with tempfile.TemporaryDirectory() as d:
            _, base = self.serve(d, {"a": b"secret-a", "b": b"secret-b"})
            body = {"worker": "b"}
            with self.assertRaises(urllib.error.HTTPError) as cm:
                raw = json.dumps(body).encode()
                req = urllib.request.Request(base + "/claim", raw, {
                    "X-Worker": "a", "X-Signature": net.sign(b"secret-a", raw)})
                urllib.request.urlopen(req)
            self.assertEqual(cm.exception.code, 401)


if __name__ == "__main__":
    unittest.main()
