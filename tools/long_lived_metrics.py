"""Optional request-level metrics for a long-lived local K3 process."""

from __future__ import annotations

import datetime as dt
import json
import pathlib
import threading
import time
from collections.abc import Callable
from typing import Any


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _disk_delta(before: dict[str, Any], after: dict[str, Any]) -> int | None:
    try:
        delta_mb = (
            after["disk"]["member_megabytes_total"]
            - before["disk"]["member_megabytes_total"]
        )
    except (KeyError, TypeError):
        return None
    return max(0, round(delta_mb * 1024 * 1024))


def _numeric_delta(
    before: dict[str, Any], after: dict[str, Any]
) -> dict[str, int | float]:
    return {
        key: value - before.get(key, 0)
        for key, value in after.items()
        if isinstance(value, (int, float))
    }


class LongLivedRequestMetrics:
    """Collect exact request boundaries without owning model or cache state."""

    def __init__(
        self,
        path: pathlib.Path | str,
        *,
        snapshot: Callable[[str], dict[str, Any]],
        stats: Callable[[], dict[str, Any]],
        routes: Callable[[], dict[int | str, list[int]]],
        expert_bytes: int,
    ):
        if expert_bytes <= 0:
            raise ValueError("expert_bytes must be positive")
        self.path = pathlib.Path(path).expanduser()
        self.snapshot = snapshot
        self.stats = stats
        self.routes = routes
        self.expert_bytes = expert_bytes
        self._seen: set[tuple[int, int]] = set()
        self._lock = threading.Lock()

    def begin(
        self,
        *,
        request_id: str,
        mode: str,
        input_tokens: int,
        max_new_tokens: int,
        memo_hit: bool,
        prefix_state: dict[str, Any] | None = None,
        prefix_activation: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "request_id": request_id,
            "mode": mode,
            "input_tokens": input_tokens,
            "max_new_tokens": max_new_tokens,
            "memo_hit": memo_hit,
            "prefix_state": dict(prefix_state or {}),
            "prefix_activation": dict(prefix_activation or {}),
            "created_at": _now(),
            "started_ns": time.perf_counter_ns(),
            "before": self.snapshot(f"request-{request_id}-before"),
            "stats_before": self.stats(),
            "routed": set(),
            "first_token": None,
            "finished": False,
        }

    def observe_token(self, session: dict[str, Any]) -> None:
        if session["finished"]:
            return
        for layer, experts in self.routes().items():
            session["routed"].update(
                (int(layer), int(expert)) for expert in experts
            )
        if session["first_token"] is None:
            duration_ns = time.perf_counter_ns() - session["started_ns"]
            session["first_token"] = {
                "duration_ns": duration_ns,
                "snapshot": self.snapshot(
                    f"request-{session['request_id']}-first-token"
                ),
            }

    def finish(
        self,
        session: dict[str, Any],
        *,
        status: str,
        output_tokens: int,
        error: str | None = None,
    ) -> dict[str, Any] | None:
        if session["finished"]:
            return None
        duration_ns = time.perf_counter_ns() - session["started_ns"]
        after = self.snapshot(f"request-{session['request_id']}-after")
        stats_after = self.stats()
        routed = session["routed"]
        new = routed - self._seen
        reused = routed & self._seen
        self._seen.update(routed)
        first = session["first_token"]
        direct_stats = _numeric_delta(
            session["stats_before"], stats_after
        )
        physical_bytes = _disk_delta(session["before"], after)
        logical_bytes = direct_stats.get(
            "demand_bytes", direct_stats.get("pread_bytes")
        )
        inferred_page_cache_bytes = (
            max(0, logical_bytes - physical_bytes)
            if logical_bytes is not None and physical_bytes is not None
            else None
        )
        activation = session["prefix_activation"]
        activation_detail = activation.get("activation", {})
        avoided_experts = int(
            activation_detail.get("skipped_unique_experts", 0)
        )
        record = {
            "schema": "deltafin.long-lived-request.v2",
            "created_at": session["created_at"],
            "completed_at": _now(),
            "request_id": session["request_id"],
            "mode": session["mode"],
            "status": status,
            "error": error,
            "memo_hit": session["memo_hit"],
            "prefix_state": session["prefix_state"],
            "prefix_activation": session["prefix_activation"],
            "input_tokens": session["input_tokens"],
            "max_new_tokens": session["max_new_tokens"],
            "output_tokens": output_tokens,
            "duration_ns": duration_ns,
            "ttft_ns": first["duration_ns"] if first is not None else None,
            "logical_expert_bytes": logical_bytes,
            "physical_member_read_bytes": physical_bytes,
            "inferred_page_cache_bytes": inferred_page_cache_bytes,
            "physical_fraction": (
                physical_bytes / logical_bytes
                if physical_bytes is not None and logical_bytes
                else None
            ),
            "ttft_physical_member_read_bytes": (
                _disk_delta(session["before"], first["snapshot"])
                if first is not None else None
            ),
            "direct_stats": direct_stats,
            "prefix_activation_avoided_experts": avoided_experts,
            "prefix_activation_avoided_expert_bytes": (
                avoided_experts * self.expert_bytes
            ),
            "prefix_activation_skipped_route_edges": int(
                activation_detail.get("skipped_route_edges", 0)
            ),
            "unique_routed_experts": len(routed),
            "unique_expert_working_set_bytes": (
                len(routed) * self.expert_bytes
            ),
            "new_process_experts": len(new),
            "new_process_expert_bytes": len(new) * self.expert_bytes,
            "reused_process_experts": len(reused),
            "reused_process_expert_bytes": (
                len(reused) * self.expert_bytes
            ),
            "cumulative_unique_experts": len(self._seen),
            "cumulative_unique_expert_bytes": (
                len(self._seen) * self.expert_bytes
            ),
            "before": session["before"],
            "first_token": first,
            "after": after,
        }
        session["finished"] = True
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self.path.open("a") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
        return record
