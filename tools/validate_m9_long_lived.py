#!/usr/bin/env python3
"""Validate cache convergence across requests in one long-lived K3 runtime."""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import json
import os
import pathlib
import subprocess
import sys
from typing import Any

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validate_m1d_resident import (  # noqa: E402
    memory_snapshot,
    model_fingerprint,
    write_evidence,
)
from validate_m2_serial_token import (  # noqa: E402
    EXPERT_BYTES,
    cleanup_runtime,
    model_tree_fingerprint,
    require_profile,
)
from validate_m3_overlap import (  # noqa: E402
    compare_sequences,
    run_sequence,
)


ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_REQUESTS = (
    "anchor-1:85001",
    "unrelated-b:85002",
    "anchor-2:85001",
    "unrelated-c:85003",
    "anchor-3:85001",
)


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument(
        "--request",
        action="append",
        dest="requests",
        metavar="LABEL:TOKEN_ID",
        help=(
            "request workload; repeat a token ID to require exact replay "
            "(default: anchor/unrelated/anchor/unrelated/anchor)"
        ),
    )
    parser.add_argument(
        "--runtime-profile",
        action="store_true",
        help="retain per-layer synchronization/profiling (default: off)",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=ROOT / "bench-results/m9-long-lived.json",
    )
    return parser.parse_args(argv)


def parse_requests(values: list[str] | None) -> list[dict[str, Any]]:
    requests = []
    labels = set()
    for position, value in enumerate(values or DEFAULT_REQUESTS):
        try:
            label, raw_token = value.rsplit(":", 1)
            token_id = int(raw_token)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid request {value!r}; expected LABEL:TOKEN_ID"
            ) from exc
        label = label.strip()
        if not label:
            raise ValueError("request label must not be empty")
        if label in labels:
            raise ValueError(f"duplicate request label {label!r}")
        if token_id < 0:
            raise ValueError("request token ID must be non-negative")
        labels.add(label)
        requests.append(
            {
                "position": position,
                "label": label,
                "token_id": token_id,
            }
        )
    if len(requests) < 3:
        raise ValueError("M9 needs at least three requests")
    counts = {}
    for request in requests:
        token_id = request["token_id"]
        counts[token_id] = counts.get(token_id, 0) + 1
    if max(counts.values(), default=0) < 2:
        raise ValueError("M9 needs at least one repeated token ID")
    return requests


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


def routed_experts(sequence: dict[str, Any]) -> set[tuple[int, int]]:
    return {
        (int(layer), int(expert))
        for row in sequence["tokens"]
        for layer, experts in row["routes"].items()
        for expert in experts
    }


def summarize_request(
    sequence: dict[str, Any],
    *,
    seen: set[tuple[int, int]],
    previous: set[tuple[int, int]] | None,
) -> tuple[dict[str, Any], set[tuple[int, int]]]:
    rows = sequence["tokens"]
    routed = routed_experts(sequence)
    logical = sum(row["slab_transfer_bytes"] for row in rows)
    physical = sum(
        row["physical_member_read_bytes"] or 0 for row in rows
    )
    new = routed - seen
    seen_before = routed & seen
    prior = routed & previous if previous is not None else set()
    summary = {
        "tokens": len(rows),
        "wall_seconds": sum(row["seconds"] for row in rows),
        "ttft_seconds": rows[0]["seconds"] if rows else None,
        "post_ttft_seconds": sum(row["seconds"] for row in rows[1:]),
        "post_ttft_seconds_per_token": (
            sum(row["seconds"] for row in rows[1:]) / (len(rows) - 1)
            if len(rows) > 1 else None
        ),
        "logical_expert_bytes": logical,
        "physical_member_read_bytes": physical,
        "physical_bytes_per_token": (
            physical / len(rows) if rows else None
        ),
        "physical_fraction": physical / logical if logical else None,
        "inferred_page_cache_bytes": max(0, logical - physical),
        "unique_routed_experts": len(routed),
        "unique_expert_working_set_bytes": len(routed) * EXPERT_BYTES,
        "new_process_experts": len(new),
        "new_process_expert_bytes": len(new) * EXPERT_BYTES,
        "reused_process_experts": len(seen_before),
        "reused_process_expert_bytes": len(seen_before) * EXPERT_BYTES,
        "previous_request_overlap_experts": len(prior),
        "previous_request_overlap_bytes": len(prior) * EXPERT_BYTES,
        "file_backed_bytes_delta": value_delta(
            sequence["before"], sequence["after"], "vm", "external_bytes"
        ),
        "free_bytes_delta": value_delta(
            sequence["before"], sequence["after"], "vm", "free_bytes"
        ),
        "compressions_delta": value_delta(
            sequence["before"], sequence["after"], "vm", "compressions"
        ),
        "compressor_bytes_delta": value_delta(
            sequence["before"],
            sequence["after"],
            "vm",
            "compressor_pages_bytes",
        ),
        "pageins_delta": value_delta(
            sequence["before"], sequence["after"], "vm", "pageins"
        ),
        "swapouts_delta": value_delta(
            sequence["before"], sequence["after"], "vm", "swapouts"
        ),
    }
    return summary, routed


def resident_bank_identity(bank) -> dict[str, Any]:
    digest = hashlib.sha256()
    for name in sorted(bank.names):
        tensor = bank.tensor(name)
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(str(tensor.data_ptr()).encode())
        digest.update(b"\0")
    return {
        "object_id": id(bank),
        "tensors": len(bank),
        "checkpoint_bytes": bank.checkpoint_bytes,
        "materialized_bytes": bank.materialized_bytes,
        "storage_dtype": bank.storage_dtype,
        "name_pointer_sha256": digest.hexdigest(),
    }


def aggregate_requests(requests: list[dict[str, Any]]) -> dict[str, Any]:
    summaries = [request["summary"] for request in requests]
    tokens = sum(summary["tokens"] for summary in summaries)
    physical = sum(
        summary["physical_member_read_bytes"] for summary in summaries
    )
    logical = sum(summary["logical_expert_bytes"] for summary in summaries)
    wall = sum(summary["wall_seconds"] for summary in summaries)
    return {
        "requests": len(requests),
        "tokens": tokens,
        "wall_seconds": wall,
        "seconds_per_token": wall / tokens if tokens else None,
        "logical_expert_bytes": logical,
        "physical_member_read_bytes": physical,
        "physical_bytes_per_token": physical / tokens if tokens else None,
        "physical_fraction": physical / logical if logical else None,
        "compressions_delta": sum(
            summary["compressions_delta"] or 0 for summary in summaries
        ),
        "swapouts_delta": sum(
            summary["swapouts_delta"] or 0 for summary in summaries
        ),
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.tokens < 2:
        raise ValueError("--tokens must be at least 2")
    request_specs = parse_requests(args.requests)
    require_profile()
    if not args.runtime_profile:
        os.environ["K3_PROFILE"] = "0"
    torch.set_grad_enabled(False)
    baseline = memory_snapshot("baseline", detailed=True)
    model_dir = pathlib.Path(os.environ["K3_MODEL_DIR"]).resolve()
    tree_before = model_tree_fingerprint(model_dir)
    evidence: dict[str, Any] = {
        "schema": "deltafin.m9-long-lived.v1",
        "status": "running",
        "created_at": now(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "tokens_per_request": args.tokens,
        "request_specs": request_specs,
        "runtime_profile": args.runtime_profile,
        "baseline": baseline,
        "official_tree_before": tree_before,
        "official_model_modified": False,
        "model_conversion_performed": False,
        "requests": [],
        "replay_parity": [],
    }
    write_evidence(args.output, evidence)
    failure = None
    kr = direct_shard_loader = resident_shard_loader = None
    layers = []
    retained: dict[int, tuple[dict[str, Any], list[torch.Tensor]]] = {}
    seen: set[tuple[int, int]] = set()
    previous: set[tuple[int, int]] | None = None
    try:
        import kimi_run as kr
        import direct_shard_loader
        import resident_shard_loader

        store = direct_shard_loader.store()
        source_before = model_fingerprint(store)
        evidence["source_before"] = source_before
        layers = kr.build_layers()
        if len(layers) != kr.NL:
            raise AssertionError(f"built {len(layers)}/{kr.NL} layers")
        bank = resident_shard_loader.runtime_bank()
        scratch = kr.resident_scratch_arena()
        if bank is None or scratch is None:
            raise AssertionError("selected resident bank/scratch is unavailable")
        evidence["resident_bank_initial"] = resident_bank_identity(bank)
        evidence["resident_scratch_initial"] = scratch.snapshot()
        evidence["runtime_loaded"] = memory_snapshot(
            "runtime-loaded", detailed=True
        )

        def progress(label, row):
            print(
                f"[{label} token {row['step']}] "
                f"{row['seconds']:.3f}s, "
                f"physical "
                f"{(row['physical_member_read_bytes'] or 0)/1e9:.3f} GB",
                flush=True,
            )

        for spec in request_specs:
            label = spec["label"]
            token_id = spec["token_id"]
            print(
                f"[M9 request {spec['position']}] "
                f"{label} token_id={token_id}",
                flush=True,
            )
            sequence, logits = run_sequence(
                kr,
                direct_shard_loader,
                layers,
                label=label,
                token_id=token_id,
                tokens=args.tokens,
                overlap=False,
                progress=progress,
            )
            summary, routed = summarize_request(
                sequence, seen=seen, previous=previous
            )
            seen.update(routed)
            summary["cumulative_unique_experts"] = len(seen)
            summary["cumulative_unique_expert_bytes"] = (
                len(seen) * EXPERT_BYTES
            )
            post_request = memory_snapshot(f"{label}-released-request-state")
            record = {
                **spec,
                "summary": summary,
                "post_request": post_request,
                "sequence": sequence,
            }
            if token_id in retained:
                reference_sequence, reference_logits = retained[token_id]
                parity = compare_sequences(
                    reference_sequence,
                    sequence,
                    reference_logits,
                    logits,
                )
                parity.update(
                    {
                        "token_id": token_id,
                        "reference_label": next(
                            request["label"]
                            for request in evidence["requests"]
                            if request["token_id"] == token_id
                        ),
                        "candidate_label": label,
                    }
                )
                evidence["replay_parity"].append(parity)
                logits.clear()
            else:
                retained[token_id] = (sequence, logits)
            evidence["requests"].append(record)
            previous = routed
            write_evidence(args.output, evidence)

        evidence["aggregate"] = aggregate_requests(evidence["requests"])
        evidence["resident_bank_final"] = resident_bank_identity(bank)
        evidence["resident_scratch_final"] = scratch.snapshot()
        if evidence["resident_bank_final"] != evidence["resident_bank_initial"]:
            raise AssertionError("resident bank identity changed across requests")
        initial_scratch = evidence["resident_scratch_initial"]
        final_scratch = evidence["resident_scratch_final"]
        if (
            final_scratch["slot_pointers"]
            != initial_scratch["slot_pointers"]
            or final_scratch["storage_bytes"]
            != initial_scratch["storage_bytes"]
        ):
            raise AssertionError("resident scratch identity changed")
        expected_preparations = (
            initial_scratch["preparations"]
            + kr.NL * args.tokens * len(request_specs)
        )
        if final_scratch["preparations"] != expected_preparations:
            raise AssertionError(
                "resident scratch preparation count differs: "
                f"{final_scratch['preparations']}/{expected_preparations}"
            )
        if not evidence["replay_parity"]:
            raise AssertionError("no repeated request was parity checked")
        if not all(row["passed"] for row in evidence["replay_parity"]):
            raise AssertionError("repeated request parity failed")
        post_mps = [
            request["post_request"]["mps"]["current_allocated"]
            for request in evidence["requests"][1:]
        ]
        evidence["request_state_stability"] = {
            "samples_after_first": post_mps,
            "range_bytes": max(post_mps) - min(post_mps),
            "limit_bytes": 64 * 1024 * 1024,
        }
        evidence["request_state_stability"]["passed"] = (
            evidence["request_state_stability"]["range_bytes"]
            <= evidence["request_state_stability"]["limit_bytes"]
        )
        if not evidence["request_state_stability"]["passed"]:
            raise AssertionError("post-request MPS allocation did not stabilize")
        source_after = model_fingerprint(store)
        tree_after = model_tree_fingerprint(model_dir)
        evidence["source_after"] = source_after
        evidence["official_tree_after"] = tree_after
        if source_after != source_before or tree_after != tree_before:
            raise AssertionError("official model tree changed")
        evidence["status"] = "passed"
    except BaseException as exc:
        failure = exc
        evidence["status"] = "failed"
        evidence["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
    finally:
        retained.clear()
        gc.collect()
        if kr is not None:
            cleanup_runtime(
                kr, direct_shard_loader, resident_shard_loader, layers
            )
        released = memory_snapshot("released", detailed=True)
        evidence["release"] = {
            "snapshot": released,
            "current_delta_from_baseline": (
                released["mps"]["current_allocated"]
                - baseline["mps"]["current_allocated"]
            ),
            "driver_delta_from_baseline": (
                released["mps"]["driver_allocated"]
                - baseline["mps"]["driver_allocated"]
            ),
            "empty_cache_called": True,
        }
        evidence["release"]["passed"] = (
            abs(evidence["release"]["current_delta_from_baseline"])
            <= 64 * 1024 * 1024
            and evidence["release"]["driver_delta_from_baseline"]
            <= 1024 * 1024 * 1024
            and released["swap"].get("used_bytes")
            == baseline["swap"].get("used_bytes")
        )
        if evidence["status"] == "passed" and not evidence["release"]["passed"]:
            evidence["status"] = "failed"
            evidence["error"] = {
                "type": "ReleaseValidationError",
                "message": "MPS or swap did not return to baseline",
            }
        evidence["completed_at"] = now()
        write_evidence(args.output, evidence)

    if failure is not None:
        raise failure
    if evidence["status"] != "passed":
        raise RuntimeError(evidence["error"]["message"])
    print(f"PASS: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
