"""Append-only checkpoint of finished work units, safe to resume after a kill.

One JSON object per line with a CRC. A line cut short by a crash or Ctrl-C fails
its CRC and is dropped on load, so a resume only redoes the unit in flight.
"""

from __future__ import annotations

import json
import zlib
from pathlib import Path


def _crc(payload: str) -> int:
    return zlib.crc32(payload.encode()) & 0xFFFFFFFF


class Store:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.done: dict[str, dict] = {}
        self.dropped = 0
        self._load()

    def _load(self):
        if not self.path.exists():
            return
        for line in self.path.read_text().splitlines():
            try:
                rec = json.loads(line)
                payload = json.dumps(rec["data"], sort_keys=True)
                if rec["crc"] != _crc(payload):
                    raise ValueError("crc")
                self.done[rec["unit_id"]] = rec["data"]
            except (ValueError, KeyError, TypeError):
                self.dropped += 1

    def has(self, unit_id: str) -> bool:
        return unit_id in self.done

    def record(self, unit_id: str, data: dict):
        payload = json.dumps(data, sort_keys=True)
        line = json.dumps({"unit_id": unit_id, "crc": _crc(payload), "data": data}) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(line)
            f.flush()
        self.done[unit_id] = data

    def pending(self, units) -> list:
        return [u for u in units if not self.has(u.unit_id)]
