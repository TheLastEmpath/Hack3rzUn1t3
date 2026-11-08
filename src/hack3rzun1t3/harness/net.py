"""HTTP layer over the Coordinator: workers pull with POST /claim and POST /submit.

Standard library only. The server binds to loopback by default; exposing it to
volunteers needs the tailnet ACLs and signed results from the multiuser design,
which are not built yet. Authentication is optional (`keys`); without it anyone who can reach the port can submit.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import partial as PT
from .coordinator import Coordinator
from .plan import WorkUnit


def sign(key: bytes, body: bytes) -> str:
    return hmac.new(key, body, hashlib.sha256).hexdigest()


def make_server(coord: Coordinator, host: str = "127.0.0.1", port: int = 0,
                keys: dict[str, bytes] | None = None) -> ThreadingHTTPServer:
    """`keys` maps worker name -> secret. If given, every request must carry
    X-Worker and X-Signature (HMAC-SHA256 of the body), and the body's worker
    field must match, so a result is attributable to one registered key."""
    lock = threading.Lock()  # the coordinator is single-threaded

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _reply(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            try:
                raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                req = json.loads(raw)
                if keys is not None:
                    who = self.headers.get("X-Worker", "")
                    ok = who in keys and hmac.compare_digest(
                        sign(keys[who], raw), self.headers.get("X-Signature", ""))
                    if not ok or req.get("worker") != who:
                        return self._reply(401, {"error": "bad signature"})
                with lock:
                    if self.path == "/claim":
                        unit = coord.claim(req["worker"])
                        return self._reply(200, {"unit": unit.to_json() if unit else None,
                                                 "done": coord.done()})
                    if self.path == "/submit":
                        status = coord.submit(req["worker"], req["unit_id"], PT.from_json(req["result"]))
                        return self._reply(200, {"status": status})
                self._reply(404, {"error": "unknown path"})
            except (KeyError, ValueError, TypeError) as e:
                self._reply(400, {"error": str(e)})

    return ThreadingHTTPServer((host, port), Handler)


def _post(base: str, path: str, obj: dict, key: bytes | None = None) -> dict:
    body = json.dumps(obj).encode()
    headers = {"Content-Type": "application/json"}
    if key is not None:
        headers.update({"X-Worker": obj["worker"], "X-Signature": sign(key, body)})
    req = urllib.request.Request(base + path, body, headers)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def claim(base: str, worker: str, key: bytes | None = None) -> tuple[WorkUnit | None, bool]:
    r = _post(base, "/claim", {"worker": worker}, key)
    return (WorkUnit(**r["unit"]) if r["unit"] else None), r["done"]


def submit(base: str, worker: str, unit_id: str, result: PT.Partial, key: bytes | None = None) -> str:
    return _post(base, "/submit", {"worker": worker, "unit_id": unit_id,
                                   "result": PT.to_json(result)}, key)["status"]
