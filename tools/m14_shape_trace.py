#!/usr/bin/env python3
"""Tokenize request bodies locally and analyze exact prompt-shape reuse."""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import os
import pathlib
import statistics
import sys
from typing import Any, Iterable

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import k3_official  # noqa: E402
from m13_prefix_workload import (  # noqa: E402
    load_records,
    reuse_distances,
    simulate_lru,
    simulate_repeat_admission,
)
from prefix_state import derive_chat_prefix  # noqa: E402
from validate_m2_serial_token import model_tree_fingerprint  # noqa: E402


ROOT = pathlib.Path(__file__).resolve().parent.parent
ACTIVATION_ENTRY_BYTES = 195_198_976


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def stress_requests() -> list[dict[str, Any]]:
    """A public, deliberately scan-heavy development-command trace."""
    hot_a = "Proceed to the next phase."
    hot_b = "Run the focused tests and report the result."
    contents = [hot_a, hot_b, hot_a, hot_b]
    for detail_count in range(1, 17):
        contents.append(
            "Investigate the component and include "
            + "detail " * detail_count
        )
    contents.extend((hot_a, hot_b))
    for detail_count in range(17, 33):
        contents.append(
            "Validate the subsystem and include "
            + "evidence " * detail_count
        )
    contents.extend((hot_a, hot_b, hot_a, hot_b))
    return [
        {
            "model": "deltafin-kimi-k3",
            "messages": [{"role": "user", "content": content.strip()}],
            "max_tokens": 1,
        }
        for content in contents
    ]


def load_request_bodies(paths: Iterable[pathlib.Path]):
    bodies = []
    for path in paths:
        with path.expanduser().open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{path}:{line_number}: invalid JSON"
                    ) from exc
                if isinstance(value, dict) and isinstance(
                    value.get("body"), dict
                ):
                    value = value["body"]
                if not isinstance(value, dict):
                    raise ValueError(
                        f"{path}:{line_number}: request is not an object"
                    )
                bodies.append(value)
    return bodies


def trace_from_shape_records(records) -> list[dict[str, Any]]:
    trace = []
    for index, record in enumerate(records):
        if record.get("schema") != "deltafin.request-shape.v1":
            raise ValueError(
                f"shape record {index}: unsupported schema "
                f"{record.get('schema')!r}"
            )
        total = int(record.get("total_positions", 0))
        if total <= 0:
            raise ValueError(
                f"shape record {index}: total_positions must be positive"
            )
        trace.append(
            {
                "index": index,
                "mode": str(record.get("mode", "")),
                "message_count": None,
                "content_bytes": None,
                "total_positions": total,
                "prefix_tokens": int(record.get("prefix_tokens", 0)),
                "eligible": bool(record.get("eligible")),
                "memo_hit": bool(record.get("memo_hit")),
            }
        )
    return trace


def trace_requests(tokenizer, bodies, prefix_ids) -> list[dict[str, Any]]:
    """Return shape-only records; prompt text and token IDs are discarded."""
    prefix_ids = tuple(int(token) for token in prefix_ids)
    rows = []
    for index, body in enumerate(bodies):
        messages = body.get("messages")
        prompt = body.get("prompt")
        if isinstance(messages, list) and messages:
            mode = "chat"
            token_ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )
            content_bytes = sum(
                len(str(message.get("content", "")).encode())
                for message in messages
                if isinstance(message, dict)
            )
            message_count = len(messages)
        elif isinstance(prompt, str):
            mode = "completion"
            token_ids = tokenizer.encode(prompt)
            content_bytes = len(prompt.encode())
            message_count = 0
        else:
            raise ValueError(
                f"request {index}: messages or string prompt required"
            )
        token_ids = tuple(int(token) for token in token_ids)
        eligible = (
            mode == "chat"
            and len(token_ids) > len(prefix_ids)
            and token_ids[: len(prefix_ids)] == prefix_ids
        )
        rows.append(
            {
                "index": index,
                "mode": mode,
                "message_count": message_count,
                "content_bytes": content_bytes,
                "total_positions": len(token_ids),
                "prefix_tokens": len(prefix_ids) if eligible else 0,
                "eligible": eligible,
                "memo_hit": False,
            }
        )
    return rows


def calibration_from_records(records) -> dict[str, Any]:
    phases = {"capture": [], "replay": []}
    for record in records:
        activation = record.get("prefix_activation") or {}
        action = activation.get("action")
        if (
            record.get("status") == "ok"
            and not record.get("memo_hit")
            and action in phases
        ):
            phases[action].append(record)

    def mean(rows, key):
        values = [
            row[key] for row in rows
            if isinstance(row.get(key), (int, float))
        ]
        return statistics.fmean(values) if values else None

    return {
        action: {
            "samples": len(rows),
            "ttft_ns": mean(rows, "ttft_ns"),
            "logical_expert_bytes": mean(rows, "logical_expert_bytes"),
            "physical_member_read_bytes": mean(
                rows, "physical_member_read_bytes"
            ),
        }
        for action, rows in phases.items()
    }


def _project(simulation, calibration):
    captures = simulation["captures"]
    replays = simulation["hits"]

    def total(key):
        capture = calibration["capture"].get(key)
        replay = calibration["replay"].get(key)
        if capture is None or replay is None:
            return None
        return captures * capture + replays * replay

    return {
        "coarse_calibration_only": True,
        "ttft_seconds": (
            total("ttft_ns") / 1e9
            if total("ttft_ns") is not None else None
        ),
        "logical_expert_bytes": total("logical_expert_bytes"),
        "physical_member_read_bytes": total(
            "physical_member_read_bytes"
        ),
    }


def analyze_shapes(
    trace: list[dict[str, Any]],
    *,
    max_capacity: int,
    calibration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    memo_requests = sum(bool(row.get("memo_hit")) for row in trace)
    eligible = [
        row for row in trace
        if row["eligible"] and not row.get("memo_hit")
    ]
    shapes = [int(row["total_positions"]) for row in eligible]
    distances = reuse_distances(shapes)
    histogram = collections.Counter(
        "first" if distance is None else str(distance)
        for distance in distances
    )
    frequency = collections.Counter(shapes)
    sweep = {}
    repeat_sweep = {}
    for capacity in range(1, max_capacity + 1):
        simulation = simulate_lru(shapes, capacity)
        repeat = simulate_repeat_admission(shapes, capacity)
        maximum_bytes = (
            min(capacity, len(frequency)) * ACTIVATION_ENTRY_BYTES
        )
        simulation["max_activation_bytes"] = maximum_bytes
        repeat["max_activation_bytes"] = maximum_bytes
        if calibration is not None:
            simulation["projection"] = _project(
                simulation, calibration
            )
            repeat["projection"] = _project(repeat, calibration)
        sweep[str(capacity)] = simulation
        repeat_sweep[str(capacity)] = repeat
    repeats = len(shapes) - len(frequency)
    finite_distances = [
        distance for distance in distances if distance is not None
    ]
    return {
        "requests": len(trace),
        "eligible_requests": len(eligible),
        "ineligible_requests": sum(
            not row["eligible"] for row in trace
        ),
        "memo_requests_excluded": memo_requests,
        "distinct_shapes": len(frequency),
        "shape_frequency": {
            str(shape): count for shape, count in sorted(frequency.items())
        },
        "shape_sequence": shapes,
        "reuse_distance_histogram": dict(
            sorted(
                histogram.items(),
                key=lambda row: (
                    row[0] != "first",
                    int(row[0]) if row[0] != "first" else -1,
                ),
            )
        ),
        "compulsory_misses": len(frequency),
        "repeat_requests": repeats,
        "infinite_capacity_hit_rate": (
            repeats / len(shapes) if shapes else None
        ),
        "capacity_for_all_observed_reuse": (
            max(finite_distances) + 1 if finite_distances else None
        ),
        "capacity_sweep": sweep,
        "repeat_admission_sweep": repeat_sweep,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request_jsonl", nargs="*", type=pathlib.Path)
    parser.add_argument(
        "--shape-jsonl",
        nargs="+",
        type=pathlib.Path,
        help="privacy-minimized JSONL emitted by the ordinary server",
    )
    parser.add_argument(
        "--stress",
        action="store_true",
        help="use the built-in public scan-heavy request sequence",
    )
    parser.add_argument("--max-capacity", type=int, default=16)
    parser.add_argument(
        "--calibration-jsonl",
        type=pathlib.Path,
        help="optional M13 server metrics for coarse cost projection",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=ROOT / "bench-results/m14-shape-trace.json",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.max_capacity <= 0:
        raise ValueError("--max-capacity must be positive")
    sources = sum(
        (bool(args.stress), bool(args.request_jsonl), bool(args.shape_jsonl))
    )
    if sources != 1:
        raise ValueError(
            "select exactly one of --stress, request JSONL, or --shape-jsonl"
        )
    prefix_ids = ()
    tree_unchanged = None
    if args.shape_jsonl:
        trace = trace_from_shape_records(
            load_records(args.shape_jsonl)
        )
        source = "server-shape-jsonl"
    else:
        bodies = (
            stress_requests()
            if args.stress
            else load_request_bodies(args.request_jsonl)
        )
        model_dir = k3_official.model_dir()
        tree_before = model_tree_fingerprint(model_dir)
        tokenizer = k3_official.load_tokenizer(ROOT)
        prefix_ids = derive_chat_prefix(tokenizer)
        trace = trace_requests(tokenizer, bodies, prefix_ids)
        tree_after = model_tree_fingerprint(model_dir)
        if tree_after != tree_before:
            raise AssertionError(
                "official model tree changed during tokenization"
            )
        tree_unchanged = True
        source = "built-in-stress" if args.stress else "request-jsonl"
    calibration = None
    if args.calibration_jsonl:
        calibration = calibration_from_records(
            load_records([args.calibration_jsonl])
        )
    analysis = analyze_shapes(
        trace,
        max_capacity=args.max_capacity,
        calibration=calibration,
    )
    prefix_lengths = {
        int(row["prefix_tokens"]) for row in trace if row["eligible"]
    }
    report = {
        "schema": "deltafin.m14-shape-trace.v1",
        "status": "passed",
        "created_at": now(),
        "source": source,
        "privacy": {
            "prompt_content_stored": False,
            "token_ids_stored": False,
            "request_hash_stored": False,
        },
        "prefix_tokens": (
            next(iter(prefix_lengths))
            if len(prefix_lengths) == 1
            else len(prefix_ids) if prefix_ids else None
        ),
        "activation_entry_bytes": ACTIVATION_ENTRY_BYTES,
        "calibration": calibration,
        "trace": trace,
        "analysis": analysis,
        "official_tree_unchanged": tree_unchanged,
        "model_conversion_performed": False,
    }
    args.output.expanduser().parent.mkdir(parents=True, exist_ok=True)
    args.output.expanduser().write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(f"PASS: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
