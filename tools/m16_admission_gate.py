#!/usr/bin/env python3
"""Decide whether passive shape evidence justifies a real repeat-admission A/B."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import subprocess
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m13_prefix_workload import load_records  # noqa: E402
from m14_shape_trace import (  # noqa: E402
    analyze_shapes,
    calibration_from_records,
    trace_from_shape_records,
)
from validate_m2_serial_token import model_tree_fingerprint  # noqa: E402


ROOT = pathlib.Path(__file__).resolve().parent.parent


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def evaluate_admission_gate(
    analysis: dict[str, Any], *, capacity: int
) -> dict[str, Any]:
    """Require observed scan resistance before paying for a real server A/B."""
    key = str(int(capacity))
    try:
        lru = analysis["capacity_sweep"][key]
        repeat = analysis["repeat_admission_sweep"][key]
    except KeyError as exc:
        raise ValueError(
            f"capacity {capacity} is absent from the analysis sweep"
        ) from exc

    eligible = int(analysis.get("eligible_requests", 0))
    repeat_requests = int(analysis.get("repeat_requests", 0))
    hit_gain = int(repeat["hits"]) - int(lru["hits"])
    capture_reduction = int(lru["captures"]) - int(repeat["captures"])
    checks = {
        "eligible_request_observed": eligible > 0,
        "exact_shape_reuse_observed": repeat_requests > 0,
        "repeat_hit_gain_observed": hit_gain > 0,
        "capture_reduction_observed": capture_reduction > 0,
    }
    candidate = all(checks.values())
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "capacity": int(capacity),
        "eligible_requests": eligible,
        "distinct_shapes": int(analysis.get("distinct_shapes", 0)),
        "repeat_requests": repeat_requests,
        "lru": {
            name: lru[name]
            for name in (
                "hits",
                "captures",
                "eviction_misses",
                "admissions",
                "bypasses",
                "hit_rate",
            )
        },
        "repeat": {
            name: repeat[name]
            for name in (
                "hits",
                "captures",
                "eviction_misses",
                "admissions",
                "bypasses",
                "hit_rate",
                "history_forget_bypasses",
            )
        },
        "hit_gain": hit_gain,
        "capture_reduction": capture_reduction,
        "checks": checks,
        "failed_checks": failed,
        "candidate_for_controlled_server_ab": candidate,
        "next_action": (
            "run-controlled-server-ab"
            if candidate
            else "keep-lru-and-continue-passive-observation"
        ),
    }


def prior_gate_summary(
    m12: dict[str, Any], m13: dict[str, Any]
) -> dict[str, Any]:
    m12_tree_exact = (
        m12.get("official_tree_before") == m12.get("official_tree_after")
    )
    m13_tree_exact = (
        m13.get("official_tree_before") == m13.get("official_tree_after")
    )
    result = {
        "m12_bitwise_replay": {
            "schema": m12.get("schema"),
            "status": m12.get("status"),
            "git_commit": m12.get("git_commit"),
            "parity_passed": bool((m12.get("parity") or {}).get("passed")),
            "activation_immutable": bool(m12.get("activation_immutable")),
            "official_tree_unchanged": m12_tree_exact,
        },
        "m13_ordinary_server": {
            "schema": m13.get("schema"),
            "status": m13.get("status"),
            "git_commit": m13.get("git_commit"),
            "response_exact": bool(m13.get("response_exact")),
            "official_tree_unchanged": m13_tree_exact,
        },
    }
    result["passed"] = all(
        (
            result["m12_bitwise_replay"]["schema"]
            == "deltafin.m12-prefix-activation.v1",
            result["m12_bitwise_replay"]["status"] == "passed",
            result["m12_bitwise_replay"]["parity_passed"],
            result["m12_bitwise_replay"]["activation_immutable"],
            result["m12_bitwise_replay"]["official_tree_unchanged"],
            result["m13_ordinary_server"]["schema"]
            == "deltafin.m13-server-workload.v1",
            result["m13_ordinary_server"]["status"] == "passed",
            result["m13_ordinary_server"]["response_exact"],
            result["m13_ordinary_server"]["official_tree_unchanged"],
        )
    )
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shape-jsonl",
        nargs="+",
        type=pathlib.Path,
        required=True,
        help="privacy-minimized passive server trace, oldest file first",
    )
    parser.add_argument("--capacity", type=int, default=2)
    parser.add_argument("--max-capacity", type=int, default=16)
    parser.add_argument("--repeat-history-entries", type=int, default=4096)
    parser.add_argument(
        "--calibration-jsonl",
        type=pathlib.Path,
        default=ROOT / "bench-results/m13-server-requests.jsonl",
    )
    parser.add_argument(
        "--m12-evidence",
        type=pathlib.Path,
        default=ROOT
        / "bench-results/m12-prefix-activation-chat-final.json",
    )
    parser.add_argument(
        "--m13-evidence",
        type=pathlib.Path,
        default=ROOT / "bench-results/m13-server-workload.json",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=ROOT / "bench-results/m16-admission-gate.json",
    )
    return parser.parse_args(argv)


def read_json(path: pathlib.Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected one JSON object")
    return value


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.capacity <= 0 or args.max_capacity < args.capacity:
        raise ValueError("capacity must be positive and within max-capacity")
    if args.repeat_history_entries <= 0:
        raise ValueError("repeat-history-entries must be positive")

    selected_policy = os.environ.get(
        "K3_PREFIX_ACTIVATION_ADMISSION", "lru"
    ).lower()
    if selected_policy != "lru":
        raise RuntimeError(
            "M16 evidence gate must start from the selected lru policy"
        )

    model_dir_value = os.environ.get("K3_MODEL_DIR")
    if not model_dir_value:
        raise RuntimeError("K3_MODEL_DIR is required for the M16 tree gate")
    model_dir = pathlib.Path(model_dir_value).expanduser().resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"official model directory missing: {model_dir}")
    tree_before = model_tree_fingerprint(model_dir)
    records = load_records(args.shape_jsonl)
    trace = trace_from_shape_records(records)
    calibration = calibration_from_records(
        load_records([args.calibration_jsonl])
    )
    analysis = analyze_shapes(
        trace,
        max_capacity=args.max_capacity,
        repeat_history_entries=args.repeat_history_entries,
        calibration=calibration,
    )
    gate = evaluate_admission_gate(analysis, capacity=args.capacity)
    prior = prior_gate_summary(
        read_json(args.m12_evidence), read_json(args.m13_evidence)
    )
    if not prior["passed"]:
        raise AssertionError("M12/M13 prerequisite evidence did not pass")
    tree_after = model_tree_fingerprint(model_dir)
    tree_unchanged = tree_before == tree_after
    if not tree_unchanged:
        raise AssertionError("official model tree changed during M16 analysis")

    report = {
        "schema": "deltafin.m16-admission-gate.v1",
        "status": "passed",
        "created_at": now(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "selected_policy_before": selected_policy,
        "selected_policy_after": "lru",
        "decision": (
            "controlled-server-ab-required-before-selection"
            if gate["candidate_for_controlled_server_ab"]
            else "keep-lru-insufficient-passive-evidence"
        ),
        "gate": gate,
        "analysis": analysis,
        "prior_exactness_gates": prior,
        "controlled_server_ab_executed": False,
        "controlled_server_ab_skip_reason": (
            None
            if gate["candidate_for_controlled_server_ab"]
            else "passive trace contains no observed repeat-admission benefit"
        ),
        "privacy": {
            "prompt_content_stored": False,
            "token_ids_stored": False,
            "request_hash_stored": False,
        },
        "official_tree_unchanged": tree_unchanged,
        "model_conversion_performed": False,
    }
    output = args.output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        f"PASS: {output} decision={report['decision']}", flush=True
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
