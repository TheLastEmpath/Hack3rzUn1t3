import tempfile
import threading
import unittest
from pathlib import Path

from hack3rzun1t3.harness import net, partial as PT, worker
from hack3rzun1t3.harness.coordinator import Coordinator
from hack3rzun1t3.harness.plan import make_units
from hack3rzun1t3.harness.store import Store

try:
    from test_coordinator import D, N_PROMPTS, BATCH, RANGES, close, expected, honest, truth
except ImportError:
    from tests.test_coordinator import D, N_PROMPTS, BATCH, RANGES, close, expected, honest, truth


class WorkerLoop(unittest.TestCase):
    def serve(self, d, **kw):
        coord = Coordinator(make_units(RANGES, N_PROMPTS, BATCH), Store(Path(d) / "s.jsonl"), D, **kw)
        server = net.make_server(coord, keys={"w1": b"k1", "w2": b"k2"})
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(lambda: (server.shutdown(), server.server_close()))
        return coord, f"http://127.0.0.1:{server.server_address[1]}"

    def test_worker_finishes_run_with_signing(self):
        acts = truth()
        with tempfile.TemporaryDirectory() as d:
            coord, base = self.serve(d)
            counts = worker.run(base, "w1", lambda u: honest(acts, u), b"k1")
            self.assertEqual(counts["accepted"], 4)
            self.assertTrue(close(PT.mean_diff(coord.combined()), expected(acts)))

    def test_failing_compute_is_retried_after_lease_expiry(self):
        acts = truth()
        calls = []

        def flaky(unit):
            calls.append(unit.unit_id)
            if len(calls) == 1:
                raise RuntimeError("out of memory")
            return honest(acts, unit)

        with tempfile.TemporaryDirectory() as d:
            coord, base = self.serve(d, lease_seconds=0.0)  # lease expires at once
            counts = worker.run(base, "w1", flaky, b"k1")
            self.assertEqual(counts["failed"], 1)
            self.assertTrue(coord.done())
            self.assertEqual(calls[0], calls[1])  # same unit handed out again


if __name__ == "__main__":
    unittest.main()
