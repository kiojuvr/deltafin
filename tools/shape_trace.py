"""Small privacy-minimized request-shape JSONL writer."""
from __future__ import annotations

import datetime as dt
import json
import pathlib
import threading
from typing import Any


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


class ShapeTraceWriter:
    """Append shape metadata without prompt content, hashes, or token IDs."""

    def __init__(
        self,
        path: pathlib.Path | str,
        max_bytes: int = 16 * 1024 * 1024,
    ):
        self.path = pathlib.Path(path).expanduser()
        self.max_bytes = int(max_bytes)
        if self.max_bytes <= 0:
            raise ValueError("shape trace max_bytes must be positive")
        self._lock = threading.Lock()

    def record(
        self,
        *,
        request_id: str,
        mode: str,
        total_positions: int,
        memo_hit: bool,
        activation: dict[str, Any],
    ) -> dict[str, Any]:
        row = {
            "schema": "deltafin.request-shape.v1",
            "created_at": _now(),
            "request_id": str(request_id),
            "mode": str(mode),
            "total_positions": int(total_positions),
            "memo_hit": bool(memo_hit),
            "eligible": bool(activation.get("eligible")),
            "prefix_tokens": int(activation.get("prefix_tokens", 0)),
            "activation_hit_preview": bool(activation.get("hit")),
            "activation_action_preview": str(
                activation.get("action", "none")
            ),
            "resident_shapes_lru": list(
                (activation.get("cache_before") or {}).get(
                    "resident_shapes_lru", ()
                )
            ),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(row, sort_keys=True) + "\n"
        if len(encoded.encode()) > self.max_bytes:
            raise ValueError("one shape trace row exceeds max_bytes")
        with self._lock:
            if (
                self.path.exists()
                and self.path.stat().st_size + len(encoded.encode())
                > self.max_bytes
            ):
                backup = self.path.with_name(self.path.name + ".1")
                if backup.exists():
                    backup.unlink()
                self.path.replace(backup)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(encoded)
                stream.flush()
        return row
