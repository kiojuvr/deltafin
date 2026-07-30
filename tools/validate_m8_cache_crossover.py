#!/usr/bin/env python3
"""Run order-reversed FP32/source resident cache-capacity comparisons."""

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
from validate_m1d_resident import write_evidence  # noqa: E402
from validate_m2_serial_token import EXPERT_BYTES, require_profile  # noqa: E402


ROOT = pathlib.Path(__file__).resolve().parent.parent
PYTHON = ROOT / ".venv/bin/python"
VALIDATOR = ROOT / "tools/validate_m3_overlap.py"
M7_VALIDATOR = ROOT / "tools/validate_m7_resident_scratch.py"


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[81001, 81002],
        help="one new starting token per matched pair",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=ROOT / "bench-results/m8-cache-crossover.json",
    )
    parser.add_argument(
        "--run-dir",
        type=pathlib.Path,
        default=ROOT / "bench-results/m8-cache-crossover-runs",
    )
    return parser.parse_args(argv)


def disk_delta(before: dict[str, Any], after: dict[str, Any]) -> int | None:
    try:
        delta_mb = (
            after["disk"]["member_megabytes_total"]
            - before["disk"]["member_megabytes_total"]
        )
    except (KeyError, TypeError):
        return None
    return max(0, round(delta_mb * 1024 * 1024))


def value_delta(
    before: dict[str, Any],
    after: dict[str, Any],
    section: str,
    key: str,
) -> int | None:
    try:
        return int(after[section][key]) - int(before[section][key])
    except (KeyError, TypeError):
        return None


def summarize_run(path: pathlib.Path) -> dict[str, Any]:
    evidence = json.loads(path.read_text())
    if evidence.get("status") != "passed":
        raise RuntimeError(f"{path}: child evidence did not pass")
    sequence = evidence["sequences"]["serial_first"]
    rows = sequence["tokens"]
    routed = {
        (int(layer), int(expert))
        for row in rows
        for layer, experts in row["routes"].items()
        for expert in experts
    }
    route_selections = sum(
        len(experts)
        for row in rows
        for experts in row["routes"].values()
    )
    logical = sum(row["slab_transfer_bytes"] for row in rows)
    physical = sum(
        row["physical_member_read_bytes"] or 0 for row in rows
    )
    before = sequence["before"]
    after = sequence["after"]
    official_unchanged = (
        evidence["source_before"] == evidence["source_after"]
        and evidence["official_tree_before"]
        == evidence["official_tree_after"]
    )
    if not official_unchanged:
        raise AssertionError(f"{path}: official checkpoint changed")
    if not evidence["release"]["passed"]:
        raise AssertionError(f"{path}: MPS/swap release validation failed")
    return {
        "evidence": str(path.resolve()),
        "git_commit": evidence["git_commit"],
        "resident_bank_dtype": evidence["resident_bank_dtype"],
        "resident_scratch": evidence["resident_scratch"],
        "tokens": len(rows),
        "wall_seconds": sum(row["seconds"] for row in rows),
        "expert_wait_seconds": sum(
            row["phase_seconds"]["expert_fetch"] for row in rows
        ),
        "resident_materialization_seconds": sum(
            row["phase_seconds"]["resident_io"] for row in rows
        ),
        "logical_expert_bytes": logical,
        "route_selections": route_selections,
        "unique_routed_experts": len(routed),
        "unique_expert_working_set_bytes": len(routed) * EXPERT_BYTES,
        "route_selection_reuse_fraction": (
            1.0 - len(routed) / route_selections
            if route_selections else None
        ),
        "physical_member_read_bytes": physical,
        "inferred_page_cache_bytes": max(0, logical - physical),
        "physical_fraction": physical / logical if logical else None,
        "per_token_physical_bytes": [
            row["physical_member_read_bytes"] or 0 for row in rows
        ],
        "per_token_seconds": [row["seconds"] for row in rows],
        "startup_physical_member_read_bytes": disk_delta(
            evidence["baseline"], before
        ),
        "file_backed_bytes_delta": value_delta(
            before, after, "vm", "external_bytes"
        ),
        "free_bytes_delta": value_delta(
            before, after, "vm", "free_bytes"
        ),
        "compressions_delta": value_delta(
            before, after, "vm", "compressions"
        ),
        "compressor_bytes_delta": value_delta(
            before, after, "vm", "compressor_pages_bytes"
        ),
        "pageins_delta": value_delta(
            before, after, "vm", "pageins"
        ),
        "swapouts_delta": value_delta(
            before, after, "vm", "swapouts"
        ),
        "swap_used_bytes_before": before["swap"].get("used_bytes"),
        "swap_used_bytes_after": after["swap"].get("used_bytes"),
        "mps_current_bytes": before["mps"]["current_allocated"],
        "resident_bank": evidence["resident_bank"],
        "resident_scratch_final": evidence.get("resident_scratch_final"),
        "release": evidence["release"],
        "official_unchanged": official_unchanged,
        "token_rows": [
            {
                "step": row["step"],
                "input_token": row["input_token"],
                "output_token": row["output_token"],
                "routes": row["routes"],
                "logits_sha256_f32": row["logits"]["sha256_f32"],
            }
            for row in rows
        ],
    }


def compare_pair(fp32: dict[str, Any], source: dict[str, Any]):
    if len(fp32["token_rows"]) != len(source["token_rows"]):
        raise AssertionError("paired run lengths differ")
    parity = []
    for expected, actual in zip(fp32["token_rows"], source["token_rows"]):
        row = {
            "step": expected["step"],
            "input_token_exact": (
                expected["input_token"] == actual["input_token"]
            ),
            "output_token_exact": (
                expected["output_token"] == actual["output_token"]
            ),
            "routes_exact": expected["routes"] == actual["routes"],
            "logits_exact": (
                expected["logits_sha256_f32"]
                == actual["logits_sha256_f32"]
            ),
        }
        row["passed"] = all(
            value for key, value in row.items() if key.endswith("_exact")
        )
        if not row["passed"]:
            raise AssertionError(f"paired parity failed: {row}")
        parity.append(row)
    return {
        "passed": True,
        "tokens": parity,
        "source_minus_fp32": {
            "wall_seconds": (
                source["wall_seconds"] - fp32["wall_seconds"]
            ),
            "expert_wait_seconds": (
                source["expert_wait_seconds"]
                - fp32["expert_wait_seconds"]
            ),
            "resident_materialization_seconds": (
                source["resident_materialization_seconds"]
                - fp32["resident_materialization_seconds"]
            ),
            "physical_member_read_bytes": (
                source["physical_member_read_bytes"]
                - fp32["physical_member_read_bytes"]
            ),
            "startup_physical_member_read_bytes": (
                source["startup_physical_member_read_bytes"]
                - fp32["startup_physical_member_read_bytes"]
                if (
                    source["startup_physical_member_read_bytes"] is not None
                    and fp32["startup_physical_member_read_bytes"] is not None
                )
                else None
            ),
        },
    }


def aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for mode in ("fp32", "source"):
        selected = [run["summary"] for run in runs if run["mode"] == mode]
        result[mode] = {
            "runs": len(selected),
            "tokens": sum(run["tokens"] for run in selected),
            "wall_seconds": sum(run["wall_seconds"] for run in selected),
            "expert_wait_seconds": sum(
                run["expert_wait_seconds"] for run in selected
            ),
            "resident_materialization_seconds": sum(
                run["resident_materialization_seconds"]
                for run in selected
            ),
            "logical_expert_bytes": sum(
                run["logical_expert_bytes"] for run in selected
            ),
            "physical_member_read_bytes": sum(
                run["physical_member_read_bytes"] for run in selected
            ),
            "startup_physical_member_read_bytes": sum(
                run["startup_physical_member_read_bytes"] or 0
                for run in selected
            ),
            "unique_expert_working_set_bytes": sum(
                run["unique_expert_working_set_bytes"]
                for run in selected
            ),
            "compressions_delta": sum(
                run["compressions_delta"] or 0 for run in selected
            ),
            "swapouts_delta": sum(
                run["swapouts_delta"] or 0 for run in selected
            ),
        }
        logical = result[mode]["logical_expert_bytes"]
        physical = result[mode]["physical_member_read_bytes"]
        result[mode]["physical_fraction"] = (
            physical / logical if logical else None
        )
        result[mode]["seconds_per_token"] = (
            result[mode]["wall_seconds"] / result[mode]["tokens"]
        )
        result[mode]["physical_bytes_per_token"] = (
            physical / result[mode]["tokens"]
        )
    result["source_minus_fp32"] = {
        key: result["source"][key] - result["fp32"][key]
        for key in (
            "wall_seconds",
            "expert_wait_seconds",
            "resident_materialization_seconds",
            "physical_member_read_bytes",
            "startup_physical_member_read_bytes",
            "seconds_per_token",
            "physical_bytes_per_token",
        )
    }
    result["by_position"] = {}
    for position in (0, 1):
        result["by_position"][str(position)] = {}
        for mode in ("fp32", "source"):
            selected = [
                run["summary"] for run in runs
                if run["position"] == position and run["mode"] == mode
            ]
            result["by_position"][str(position)][mode] = {
                "runs": len(selected),
                "wall_seconds": sum(
                    run["wall_seconds"] for run in selected
                ),
                "physical_member_read_bytes": sum(
                    run["physical_member_read_bytes"] for run in selected
                ),
                "startup_physical_member_read_bytes": sum(
                    run["startup_physical_member_read_bytes"] or 0
                    for run in selected
                ),
                "compressions_delta": sum(
                    run["compressions_delta"] or 0 for run in selected
                ),
            }
    return result


def child_command(
    mode: str,
    *,
    token_id: int,
    tokens: int,
    output: pathlib.Path,
) -> list[str]:
    common = [
        "--runtime-profile-off",
        "--sequence-set",
        "serial-only",
        "--token-id",
        str(token_id),
        "--tokens",
        str(tokens),
        "--output",
        str(output),
    ]
    if mode == "fp32":
        return [
            str(PYTHON),
            str(VALIDATOR),
            "--resident-bank-dtype",
            "runtime",
            "--pin-layers",
            "93",
            *common,
        ]
    return [
        str(PYTHON),
        str(M7_VALIDATOR),
        *common,
    ]


def run_child(command: list[str], log_path: pathlib.Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=os.environ.copy(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        returncode = process.wait()
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.tokens < 2:
        raise ValueError("--tokens must be at least 2")
    if len(args.seeds) < 2:
        raise ValueError("at least two seeds are required for order reversal")
    require_profile()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    evidence: dict[str, Any] = {
        "schema": "deltafin.m8-cache-crossover.v1",
        "status": "running",
        "created_at": now(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "tokens_per_run": args.tokens,
        "seeds": args.seeds,
        "orders": [],
        "runs": [],
        "pairs": [],
        "official_model_modified": False,
        "model_conversion_performed": False,
    }
    write_evidence(args.output, evidence)
    try:
        for pair_index, seed in enumerate(args.seeds):
            order = (
                ("fp32", "source")
                if pair_index % 2 == 0
                else ("source", "fp32")
            )
            evidence["orders"].append(
                {"pair": pair_index, "seed": seed, "modes": list(order)}
            )
            pair_runs = {}
            for position, mode in enumerate(order):
                stem = f"pair-{pair_index:02d}-seed-{seed}-{position}-{mode}"
                child_output = args.run_dir / f"{stem}.json"
                log_path = args.run_dir / f"{stem}.log"
                command = child_command(
                    mode,
                    token_id=seed,
                    tokens=args.tokens,
                    output=child_output,
                )
                print(
                    f"[M8 pair {pair_index} position {position}] "
                    f"seed={seed} mode={mode}",
                    flush=True,
                )
                run_child(command, log_path)
                summary = summarize_run(child_output)
                record = {
                    "pair": pair_index,
                    "position": position,
                    "seed": seed,
                    "mode": mode,
                    "command": command,
                    "log": str(log_path.resolve()),
                    "summary": summary,
                }
                evidence["runs"].append(record)
                pair_runs[mode] = summary
                write_evidence(args.output, evidence)
            comparison = compare_pair(
                pair_runs["fp32"], pair_runs["source"]
            )
            evidence["pairs"].append(
                {
                    "pair": pair_index,
                    "seed": seed,
                    "order": list(order),
                    "comparison": comparison,
                }
            )
            write_evidence(args.output, evidence)
        evidence["aggregate"] = aggregate(evidence["runs"])
        evidence["status"] = "passed"
    except BaseException as exc:
        evidence["status"] = "failed"
        evidence["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        raise
    finally:
        evidence["completed_at"] = now()
        write_evidence(args.output, evidence)
    print(f"PASS: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
