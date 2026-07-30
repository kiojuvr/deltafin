#!/usr/bin/env python3
"""Summarize M12 prefix-activation behavior from server request JSONL."""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
from collections import OrderedDict
from typing import Any, Iterable


def load_records(paths: Iterable[pathlib.Path]) -> list[dict[str, Any]]:
    records = []
    for path in paths:
        with path.expanduser().open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{path}:{line_number}: invalid JSON"
                    ) from exc
                if not isinstance(record, dict):
                    raise ValueError(
                        f"{path}:{line_number}: record is not an object"
                    )
                records.append(record)
    return records


def simulate_lru(shapes: Iterable[int], capacity: int) -> dict[str, Any]:
    capacity = int(capacity)
    if capacity <= 0:
        raise ValueError("LRU capacity must be positive")
    resident: OrderedDict[int, None] = OrderedDict()
    hits = captures = evictions = 0
    compulsory_misses = eviction_misses = 0
    seen = set()
    trace = []
    for shape in (int(value) for value in shapes):
        hit = shape in resident
        if hit:
            hits += 1
            resident.move_to_end(shape)
            action = "replay"
        else:
            captures += 1
            if shape in seen:
                miss_class = "eviction"
                eviction_misses += 1
            else:
                miss_class = "compulsory"
                compulsory_misses += 1
                seen.add(shape)
            resident[shape] = None
            action = "capture"
            if len(resident) > capacity:
                resident.popitem(last=False)
                evictions += 1
        trace.append(
            {
                "shape": shape,
                "action": action,
                "miss_class": None if hit else miss_class,
                "resident_shapes_lru": list(resident),
            }
        )
    requests = hits + captures
    return {
        "capacity": capacity,
        "requests": requests,
        "hits": hits,
        "captures": captures,
        "compulsory_misses": compulsory_misses,
        "eviction_misses": eviction_misses,
        "evictions": evictions,
        "hit_rate": hits / requests if requests else None,
        "resident_shapes_lru": list(resident),
        "trace": trace,
    }


def reuse_distances(shapes: Iterable[int]) -> list[int | None]:
    """Return LRU stack distance; first occurrences have no distance."""
    stack: list[int] = []
    distances: list[int | None] = []
    for shape in (int(value) for value in shapes):
        if shape not in stack:
            distances.append(None)
        else:
            position = stack.index(shape)
            distances.append(len(stack) - position - 1)
            stack.pop(position)
        stack.append(shape)
    return distances


def _sum(rows, key):
    return sum(
        value for row in rows
        if isinstance((value := row.get(key)), (int, float))
    )


def _mean(rows, key):
    values = [
        value for row in rows
        if isinstance((value := row.get(key)), (int, float))
    ]
    return statistics.fmean(values) if values else None


def _phase_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    logical = _sum(rows, "logical_expert_bytes")
    physical = _sum(rows, "physical_member_read_bytes")
    duration = _mean(rows, "duration_ns")
    ttft = _mean(rows, "ttft_ns")
    return {
        "requests": len(rows),
        "mean_duration_seconds": (
            duration / 1e9 if duration is not None else None
        ),
        "mean_ttft_seconds": (
            ttft / 1e9 if ttft is not None else None
        ),
        "logical_expert_bytes": logical,
        "physical_member_read_bytes": physical,
        "inferred_page_cache_bytes": _sum(
            rows, "inferred_page_cache_bytes"
        ),
        "physical_fraction": physical / logical if logical else None,
        "avoided_expert_bytes": _sum(
            rows, "prefix_activation_avoided_expert_bytes"
        ),
        "skipped_route_edges": _sum(
            rows, "prefix_activation_skipped_route_edges"
        ),
    }


def summarize(
    records: list[dict[str, Any]], *, max_simulated_capacity: int = 4
) -> dict[str, Any]:
    eligible = []
    phases: dict[str, list[dict[str, Any]]] = {
        "capture": [],
        "replay": [],
        "ineligible": [],
        "memo": [],
        "error": [],
    }
    shapes: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        activation = record.get("prefix_activation") or {}
        if record.get("status") not in ("ok", "memo-hit"):
            phases["error"].append(record)
            continue
        if record.get("memo_hit"):
            phases["memo"].append(record)
            continue
        if not activation.get("eligible"):
            phases["ineligible"].append(record)
            continue
        eligible.append(record)
        action = activation.get("action")
        if action not in ("capture", "replay"):
            raise ValueError(
                f"eligible request has invalid activation action {action!r}"
            )
        phases[action].append(record)
        shape = int(activation["total_positions"])
        shapes.setdefault(shape, []).append(record)

    shape_rows = {}
    for shape, rows in sorted(shapes.items()):
        captures = sum(
            (row.get("prefix_activation") or {}).get("action") == "capture"
            for row in rows
        )
        replays = len(rows) - captures
        shape_rows[str(shape)] = {
            "requests": len(rows),
            "captures": captures,
            "replays": replays,
            "hit_rate": replays / len(rows),
            **_phase_summary(rows),
        }
    shape_sequence = [
        int(record["prefix_activation"]["total_positions"])
        for record in eligible
    ]
    sweep = {
        str(capacity): simulate_lru(shape_sequence, capacity)
        for capacity in range(1, max_simulated_capacity + 1)
    }
    captures = len(phases["capture"])
    replays = len(phases["replay"])
    final_cache = {}
    for record in reversed(eligible):
        candidate = (
            record.get("prefix_activation", {}).get("cache_after")
        )
        if candidate:
            final_cache = candidate
            break
    return {
        "schema": "deltafin.m13-prefix-workload.v1",
        "records": len(records),
        "eligible_requests": len(eligible),
        "captures": captures,
        "replays": replays,
        "hit_rate": (
            replays / (captures + replays)
            if captures + replays else None
        ),
        "distinct_shapes": len(shapes),
        "shape_sequence": shape_sequence,
        "by_shape": shape_rows,
        "phases": {
            name: _phase_summary(rows) for name, rows in phases.items()
        },
        "capacity_sweep": sweep,
        "final_cache": final_cache,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", nargs="+", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--max-capacity", type=int, default=4)
    args = parser.parse_args(argv)
    records = load_records(args.jsonl)
    report = summarize(
        records, max_simulated_capacity=args.max_capacity
    )
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.expanduser().parent.mkdir(parents=True, exist_ok=True)
        args.output.expanduser().write_text(payload)
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
