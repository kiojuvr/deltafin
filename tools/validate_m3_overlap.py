#!/usr/bin/env python3
"""Measure consecutive K3 tokens and validate direct-slab overlap parity."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import gc
import json
import os
import pathlib
import subprocess
import sys
import time
from typing import Any

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validate_m1d_resident import (  # noqa: E402
    disk_delta_bytes,
    memory_snapshot,
    model_fingerprint,
    write_evidence,
)
from validate_m2_serial_token import (  # noqa: E402
    EXPERT_BYTES,
    cleanup_runtime,
    logits_summary,
    model_tree_fingerprint,
    require_profile,
    reset_runtime_counters,
)


ROOT = pathlib.Path(__file__).resolve().parent.parent


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token-id", type=int, default=1008)
    parser.add_argument("--tokens", type=int, default=4)
    parser.add_argument(
        "--sequence-order",
        choices=("serial-first", "adaptive-first"),
        default="serial-first",
    )
    parser.add_argument(
        "--darwin-nocache",
        action="store_true",
        help="enable F_NOCACHE after validating the normal M3 profile",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=ROOT / "bench-results/m3-overlap.json",
    )
    return parser.parse_args(argv)


def numeric_delta(after, before):
    return {
        key: after[key] - before.get(key, 0)
        for key in after
        if isinstance(after[key], (int, float))
    }


def routes_snapshot(kr) -> dict[str, list[int]]:
    routes = {
        str(layer): [int(expert) for expert in experts]
        for layer, experts in sorted(kr._LAST_SEL.items())
    }
    if {int(layer) for layer in routes} != set(range(1, kr.NL)):
        raise AssertionError("multi-token pass did not route all MoE layers")
    if any(len(experts) != kr.BASE_MOE_TOP_K for experts in routes.values()):
        raise AssertionError("multi-token pass did not select top-16")
    return routes


def locality(
    current: dict[str, list[int]],
    previous: dict[str, list[int]] | None,
) -> dict[str, Any]:
    total = sum(len(ids) for ids in current.values())
    if previous is None:
        return {
            "available": False,
            "hits": 0,
            "misses": total,
            "hit_rate": 0.0,
            "exact_layers": 0,
        }
    hits = sum(
        len(set(ids) & set(previous[layer]))
        for layer, ids in current.items()
    )
    exact_layers = sum(
        set(ids) == set(previous[layer])
        for layer, ids in current.items()
    )
    return {
        "available": True,
        "hits": hits,
        "misses": total - hits,
        "hit_rate": hits / total,
        "exact_layers": exact_layers,
    }


def run_sequence(
    kr,
    direct_shard_loader,
    layers,
    *,
    label: str,
    token_id: int,
    tokens: int,
    overlap: bool,
    progress,
):
    direct_shard_loader.settle_slab_prefetches()
    direct_shard_loader.DIRECT_OVERLAP = overlap
    kr.DIRECT_OVERLAP_ACTIVE = overlap and kr.DIRECT_SLAB_ACTIVE
    reset_runtime_counters(kr, direct_shard_loader)
    direct_shard_loader.reset_demand_signal()
    kr._LAST_SEL.clear()
    kr._PREV_SEL.clear()
    if hasattr(kr, "_LAST_ROUTE_RANK"):
        kr._LAST_ROUTE_RANK.clear()
        kr._PREV_ROUTE_RANK.clear()
    adaptive_policy = getattr(kr, "DIRECT_ADAPTIVE_POLICY", None)
    if adaptive_policy is not None:
        adaptive_policy.reset()
    cache = kr.ml.KimiDynamicCache(kr.config)
    embed = kr.LazyEmbed()
    current_token = token_id
    previous_routes = None
    rows = []
    retained_logits = []
    sequence_started = time.perf_counter()
    sequence_before = memory_snapshot(f"{label}-before")
    for step in range(tokens):
        demand_before = direct_shard_loader.demand_read_snapshot()
        policy_before = (
            adaptive_policy.snapshot()
            if adaptive_policy is not None else None
        )
        stats_before = dict(direct_shard_loader.stats)
        times_before = dict(kr.TIMES)
        profile_start = len(kr.LAYER_PROFILE)
        before = memory_snapshot(f"{label}-token-{step}-before")
        started = time.perf_counter()
        kr._step_ctx["step"] = step
        hidden = embed([current_token])
        logits = kr.forward_pass(
            layers, cache, hidden, step=step, verbose=False
        )
        torch.mps.synchronize()
        elapsed = time.perf_counter() - started
        after = memory_snapshot(f"{label}-token-{step}-after")
        cpu_logits = logits.detach().to(torch.float32).cpu()
        output_token = int(cpu_logits[0, -1].argmax())
        routes = routes_snapshot(kr)
        local = locality(routes, previous_routes)
        policy_after = (
            adaptive_policy.snapshot()
            if adaptive_policy is not None else None
        )
        demand_after = direct_shard_loader.demand_read_snapshot()
        stats_delta = numeric_delta(
            direct_shard_loader.stats, stats_before
        )
        phase_delta = numeric_delta(kr.TIMES, times_before)
        physical = disk_delta_bytes(before, after)
        requested = stats_delta["pread_bytes"]
        demanded = (kr.NL - 1) * kr.BASE_MOE_TOP_K * EXPERT_BYTES
        row = {
            "step": step,
            "input_token": current_token,
            "output_token": output_token,
            "seconds": elapsed,
            "before": before,
            "after": after,
            "physical_member_read_bytes": physical,
            "demanded_expert_bytes": demanded,
            "slab_transfer_bytes": requested,
            "inferred_page_cache_bytes": (
                max(0, requested - physical)
                if physical is not None else None
            ),
            "phase_seconds": phase_delta,
            "direct_stats": stats_delta,
            "locality_vs_previous_token": local,
            "adaptive_policy": policy_after,
            "demand_read_before": demand_before,
            "demand_read_after": demand_after,
            "routes": routes,
            "logits": logits_summary(cpu_logits),
            "layer_profile": copy.deepcopy(
                kr.LAYER_PROFILE[profile_start:]
            ),
        }
        if len(row["layer_profile"]) != kr.NL:
            raise AssertionError("token did not produce 93 layer profiles")
        if stats_delta["http_bytes"] or stats_delta["expert_http"]:
            raise AssertionError("multi-token direct runtime attempted HTTP")
        if not overlap and (
            stats_delta["pread_bytes"] != demanded
            or stats_delta["slab_loads"] != kr.NL - 1
        ):
            raise AssertionError("serial token direct-read totals differ")
        if overlap and step > 0:
            if kr.DIRECT_OVERLAP_POLICY == "full":
                if stats_delta["overlap_layers"] != kr.NL - 1:
                    raise AssertionError(
                        "not every routed layer consumed a prefetch"
                    )
                if stats_delta["overlap_hits"] != local["hits"]:
                    raise AssertionError(
                        "slab reuse differs from route intersection"
                    )
                if stats_delta["overlap_misses"] != local["misses"]:
                    raise AssertionError(
                        "slab misses differ from route intersection"
                    )
            else:
                predicted_hits = (
                    policy_after["predicted_hits"]
                    - policy_before["predicted_hits"]
                )
                predicted_experts = (
                    policy_after["predicted_experts"]
                    - policy_before["predicted_experts"]
                )
                if stats_delta["overlap_hits"] != predicted_hits:
                    raise AssertionError(
                        "adaptive slab hits differ from policy hits"
                    )
                if (
                    stats_delta["overlap_prefetch_experts"]
                    != predicted_experts
                ):
                    raise AssertionError(
                        "adaptive reads differ from policy predictions"
                    )
                if (
                    stats_delta["overlap_layers"]
                    != stats_delta["overlap_prefetches"]
                ):
                    raise AssertionError(
                        "not every adaptive prefetch was consumed"
                    )
        rows.append(row)
        retained_logits.append(cpu_logits)
        progress(label, row)
        previous_routes = routes
        current_token = output_token
        del logits, hidden
    direct_shard_loader.settle_slab_prefetches()
    sequence_after = memory_snapshot(f"{label}-after")
    result = {
        "label": label,
        "overlap": overlap,
        "overlap_policy": kr.DIRECT_OVERLAP_POLICY,
        "seconds": time.perf_counter() - sequence_started,
        "before": sequence_before,
        "after": sequence_after,
        "tokens": rows,
        "final_stats": dict(direct_shard_loader.stats),
        "output_tokens": [row["output_token"] for row in rows],
    }
    del cache, embed
    gc.collect()
    torch.mps.empty_cache()
    torch.mps.synchronize()
    return result, retained_logits


def compare_sequences(reference, candidate, ref_logits, candidate_logits):
    if len(reference["tokens"]) != len(candidate["tokens"]):
        raise AssertionError("sequence lengths differ")
    tokens = []
    for ref_row, got_row, ref_logit, got_logit in zip(
        reference["tokens"],
        candidate["tokens"],
        ref_logits,
        candidate_logits,
    ):
        row = {
            "step": ref_row["step"],
            "input_token_exact": (
                ref_row["input_token"] == got_row["input_token"]
            ),
            "output_token_exact": (
                ref_row["output_token"] == got_row["output_token"]
            ),
            "routes_exact": ref_row["routes"] == got_row["routes"],
            "logits_exact": bool(torch.equal(ref_logit, got_logit)),
            "max_abs": float((ref_logit - got_logit).abs().max()),
        }
        row["passed"] = all(
            value for key, value in row.items()
            if key.endswith("_exact")
        )
        if not row["passed"]:
            raise AssertionError(f"sequence parity failed: {row}")
        tokens.append(row)
    return {"passed": True, "tokens": tokens}


def aggregate(sequence):
    rows = sequence["tokens"]
    result = {
        "tokens": len(rows),
        "wall_seconds": sum(row["seconds"] for row in rows),
        "physical_member_read_bytes": sum(
            row["physical_member_read_bytes"] or 0 for row in rows
        ),
        "demanded_expert_bytes": sum(
            row["demanded_expert_bytes"] for row in rows
        ),
        "slab_transfer_bytes": sum(
            row["slab_transfer_bytes"] for row in rows
        ),
        "critical_expert_wait_seconds": sum(
            row["phase_seconds"]["expert_fetch"] for row in rows
        ),
        "moe_kernel_seconds": sum(
            row["phase_seconds"]["moe_kernel"] for row in rows
        ),
        "route_hits": sum(
            row["locality_vs_previous_token"]["hits"] for row in rows[1:]
        ),
        "route_misses": sum(
            row["locality_vs_previous_token"]["misses"] for row in rows[1:]
        ),
    }
    policy = rows[-1].get("adaptive_policy") if rows else None
    if policy is not None:
        result["adaptive_policy"] = policy
    return result


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.tokens < 2:
        raise ValueError("--tokens must be at least 2")
    require_profile()
    if args.darwin_nocache:
        os.environ["K3_PREAD_NOCACHE"] = "1"
    if os.environ.get("K3_DIRECT_OVERLAP") not in ("0", "1"):
        raise RuntimeError("set K3_DIRECT_OVERLAP=0 or 1 explicitly")
    torch.set_grad_enabled(False)
    baseline = memory_snapshot("baseline", detailed=True)
    model_dir = pathlib.Path(os.environ["K3_MODEL_DIR"]).resolve()
    tree_before = model_tree_fingerprint(model_dir)
    evidence: dict[str, Any] = {
        "schema": (
            "deltafin.m5-cold-gate.v1"
            if os.environ.get("K3_DIRECT_PREFETCH_COLD_ONLY") == "1"
            else (
                "deltafin.m4-adaptive.v1"
                if os.environ.get("K3_DIRECT_OVERLAP_POLICY") == "adaptive"
                else "deltafin.m3-overlap.v1"
            )
        ),
        "status": "running",
        "created_at": now(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "token_id": args.token_id,
        "tokens": args.tokens,
        "sequence_order": args.sequence_order,
        "darwin_nocache": args.darwin_nocache,
        "baseline": baseline,
        "official_tree_before": tree_before,
        "model_files_modified": False,
        "model_conversion_performed": False,
        "sequences": {},
    }
    write_evidence(args.output, evidence)
    failure = None
    kr = direct_shard_loader = resident_shard_loader = None
    layers = []
    all_logits = []

    def progress(label, row):
        print(
            f"[{label} token {row['step']}] "
            f"{row['seconds']:.3f}s, "
            f"physical {(row['physical_member_read_bytes'] or 0)/1e9:.3f} GB, "
            f"route-hit "
            f"{row['locality_vs_previous_token']['hit_rate']*100:.1f}%",
            flush=True,
        )
        write_evidence(args.output, evidence)

    try:
        import kimi_run as kr
        import direct_shard_loader
        import resident_shard_loader

        evidence["overlap_configuration"] = {
            "policy": kr.DIRECT_OVERLAP_POLICY,
            "adaptive": (
                {
                    "max_experts_per_layer": (
                        kr.DIRECT_ADAPTIVE_POLICY.max_experts_per_layer
                    ),
                    "token_budget_bytes": (
                        kr.DIRECT_ADAPTIVE_POLICY.token_budget_bytes
                    ),
                    "warmup_observations": (
                        kr.DIRECT_ADAPTIVE_POLICY.warmup_observations
                    ),
                    "min_wilson_precision": (
                        kr.DIRECT_ADAPTIVE_POLICY.min_wilson_precision
                    ),
                    "cold_only": kr.DIRECT_PREFETCH_COLD_ONLY,
                    "cold_gbps": kr.DIRECT_PREFETCH_COLD_GBPS,
                    "demand_ema_alpha": (
                        direct_shard_loader.DEMAND_EMA_ALPHA
                    ),
                }
                if kr.DIRECT_ADAPTIVE_POLICY is not None else None
            ),
        }
        store = direct_shard_loader.store()
        source_before = model_fingerprint(store)
        evidence["source_before"] = source_before
        layers = kr.build_layers()

        if args.sequence_order == "serial-first":
            sequence_specs = (
                ("serial_first", "serial-first", False),
                ("serial_warm", "serial-warm", False),
                ("overlap_warm", "overlap-warm", True),
            )
        else:
            sequence_specs = (
                ("overlap_first", "overlap-first", True),
                ("serial_warm", "serial-warm", False),
                ("overlap_warm", "overlap-warm", True),
            )
        sequence_logits = {}
        for key, label, overlap in sequence_specs:
            sequence, logits = run_sequence(
                kr,
                direct_shard_loader,
                layers,
                label=label,
                token_id=args.token_id,
                tokens=args.tokens,
                overlap=overlap,
                progress=progress,
            )
            evidence["sequences"][key] = sequence
            sequence_logits[key] = logits
            write_evidence(args.output, evidence)

        if args.sequence_order == "serial-first":
            evidence["serial_repeat_parity"] = compare_sequences(
                evidence["sequences"]["serial_first"],
                evidence["sequences"]["serial_warm"],
                sequence_logits["serial_first"],
                sequence_logits["serial_warm"],
            )
        else:
            evidence["cold_overlap_parity"] = compare_sequences(
                evidence["sequences"]["overlap_first"],
                evidence["sequences"]["serial_warm"],
                sequence_logits["overlap_first"],
                sequence_logits["serial_warm"],
            )
        evidence["overlap_parity"] = compare_sequences(
            evidence["sequences"]["serial_warm"],
            evidence["sequences"]["overlap_warm"],
            sequence_logits["serial_warm"],
            sequence_logits["overlap_warm"],
        )
        evidence["aggregate"] = {
            name: aggregate(sequence)
            for name, sequence in evidence["sequences"].items()
        }
        for logits in sequence_logits.values():
            all_logits.extend(logits)
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
        all_logits.clear()
        if kr is not None:
            cleanup_runtime(
                kr, direct_shard_loader, resident_shard_loader, layers
            )
        released = memory_snapshot("released", detailed=True)
        release = {
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
        release["passed"] = (
            abs(release["current_delta_from_baseline"]) <= 64 * 1024 * 1024
            and release["driver_delta_from_baseline"] <= 1024 * 1024 * 1024
            and released["swap"].get("used_bytes")
            == baseline["swap"].get("used_bytes")
        )
        evidence["release"] = release
        if evidence["status"] == "passed" and not release["passed"]:
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
