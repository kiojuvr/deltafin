#!/usr/bin/env python3
"""Validate bounded direct-shard slabs for multi-token K3 prefill."""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import gc
import hashlib
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
    cleanup_runtime,
    logits_summary,
    model_tree_fingerprint,
    require_profile,
    reset_runtime_counters,
)
from validate_m3_overlap import numeric_delta  # noqa: E402
from validate_m9_long_lived import resident_bank_identity  # noqa: E402


ROOT = pathlib.Path(__file__).resolve().parent.parent
CASE_ORDER = ("two", "eight", "chat")


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        choices=CASE_ORDER,
        nargs="+",
        default=list(CASE_ORDER),
    )
    parser.add_argument("--chat-text", default="Hello")
    parser.add_argument(
        "--position-batch",
        action="store_true",
        help="enable the existing multi-position Metal command-buffer ABI",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=ROOT / "bench-results/m10-prefill.json",
    )
    return parser.parse_args(argv)


def prompt_cases(tokenizer, chat_text: str) -> dict[str, list[int]]:
    eight = tokenizer.encode("The capital of France is Paris, and Europe")[:8]
    cases = {
        "two": tokenizer.encode("The Chrysler"),
        "eight": eight,
        "chat": tokenizer.apply_chat_template(
            [{"role": "user", "content": chat_text}],
            tokenize=True,
            add_generation_prompt=True,
        ),
    }
    expected = {"two": 2, "eight": 8}
    wrong = {
        name: len(cases[name])
        for name, length in expected.items()
        if len(cases[name]) != length
    }
    if wrong:
        raise AssertionError(f"fixed prompt token counts changed: {wrong}")
    if len(cases["chat"]) <= 16:
        raise AssertionError("chat template no longer exercises large prefill")
    return cases


def tensor_digest(tensor: torch.Tensor) -> dict[str, Any]:
    cpu = tensor.detach().contiguous().cpu()
    raw = cpu.view(torch.uint8).numpy().tobytes()
    return {
        "shape": list(cpu.shape),
        "dtype": str(cpu.dtype),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def cache_digest(cache) -> dict[str, Any]:
    tensors = {}
    total = 0
    for attribute in (
        "recurrent_states",
        "conv_states",
        "key_cache",
        "value_cache",
    ):
        values = getattr(cache, attribute)
        for layer, value in enumerate(values):
            parts = value if isinstance(value, tuple) else (value,)
            for part, tensor in enumerate(parts):
                if tensor is None:
                    continue
                name = f"{attribute}.{layer}.{part}"
                summary = tensor_digest(tensor)
                tensors[name] = summary
                total += summary["bytes"]
    return {
        "tensors": len(tensors),
        "bytes": total,
        "entries": tensors,
    }


def route_observer(target: dict[str, Any]):
    def observe(step, layer, rows):
        target[f"{step}:{layer}"] = [list(row) for row in rows]

    return observe


def route_summary(routes: dict[str, list[list[int]]]) -> dict[str, Any]:
    by_step: dict[int, list[int]] = {}
    edges_by_step: dict[int, int] = {}
    for key, rows in routes.items():
        raw_step, _raw_layer = key.split(":", 1)
        step = int(raw_step)
        union = {expert for row in rows for expert in row}
        by_step.setdefault(step, []).append(len(union))
        edges_by_step[step] = edges_by_step.get(step, 0) + sum(
            len(row) for row in rows
        )
    return {
        str(step): {
            "layers": len(unions),
            "union_experts_total": sum(unions),
            "union_experts_min": min(unions),
            "union_experts_max": max(unions),
            "union_experts_mean": sum(unions) / len(unions),
            "active_slab_bytes_max": max(unions) * 17_547_264,
            "route_edges": edges_by_step[step],
        }
        for step, unions in sorted(by_step.items())
        if unions
    }


@contextlib.contextmanager
def legacy_heap_provider(direct_shard_loader, layer, ids, workers=None):
    raw = direct_shard_loader.fetch_experts(
        layer,
        ids,
        workers=workers,
        dequant=False,
    )
    try:
        yield raw
    finally:
        raw = None


def run_mode(
    kr,
    direct_shard_loader,
    layers,
    *,
    case: str,
    token_ids: list[int],
    mode: str,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    if mode not in ("elastic", "legacy-heap"):
        raise ValueError(f"unknown prefill mode {mode}")
    direct_shard_loader.settle_slab_prefetches()
    reset_runtime_counters(kr, direct_shard_loader)
    direct_shard_loader.reset_demand_signal()
    routes = {}
    old_observer = kr.set_route_observer(route_observer(routes))
    old_provider = direct_shard_loader.prefill_slab_experts
    if mode == "legacy-heap":
        direct_shard_loader.prefill_slab_experts = (
            lambda layer, ids, workers=None: legacy_heap_provider(
                direct_shard_loader, layer, ids, workers
            )
        )
    cache = kr.ml.KimiDynamicCache(kr.config)
    embed = kr.LazyEmbed()
    before = memory_snapshot(f"{case}-{mode}-before")
    stats_before = dict(direct_shard_loader.stats)
    metal_before = kr.metal_moe.stats()
    times_before = dict(kr.TIMES)
    try:
        prefill_started = time.perf_counter()
        kr._step_ctx["step"] = 0
        prefill_logits = kr.forward_pass(
            layers,
            cache,
            embed(token_ids),
            step=0,
            verbose=False,
        )
        torch.mps.synchronize()
        prefill_seconds = time.perf_counter() - prefill_started
        prefill_cpu = prefill_logits.detach().to(torch.float32).cpu()
        if list(prefill_cpu.shape[:2]) != [1, len(token_ids)]:
            raise AssertionError(
                f"{case}/{mode}: prefill logits shape "
                f"{list(prefill_cpu.shape)}"
            )
        prefill_cache = cache_digest(cache)
        next_token = int(prefill_cpu[0, -1].argmax())

        decode_started = time.perf_counter()
        kr._step_ctx["step"] = 1
        decode_logits = kr.forward_pass(
            layers,
            cache,
            embed([next_token]),
            step=1,
            verbose=False,
        )
        torch.mps.synchronize()
        decode_seconds = time.perf_counter() - decode_started
        decode_cpu = decode_logits.detach().to(torch.float32).cpu()
        decode_cache = cache_digest(cache)
        after = memory_snapshot(f"{case}-{mode}-after")
        stats_delta = numeric_delta(
            direct_shard_loader.stats, stats_before
        )
        metal_after = kr.metal_moe.stats()
        metal_delta = numeric_delta(metal_after, metal_before)
        phase_delta = numeric_delta(kr.TIMES, times_before)
        if stats_delta["http_bytes"] or stats_delta["expert_http"]:
            raise AssertionError(f"{case}/{mode}: HTTP access detected")
        if mode == "elastic":
            if stats_delta["prefill_slab_loads"] <= 0:
                raise AssertionError(
                    f"{case}: multi-token prefill did not use elastic slab"
                )
            if metal_delta.get("copies", 0) != 0:
                raise AssertionError(
                    f"{case}: elastic slab caused Metal staging copies"
                )
            slab = direct_shard_loader.prefill_slab_snapshot()
            if not slab["page_aligned"]:
                raise AssertionError(f"{case}: prefill slab is not page aligned")
        layer_owned = sum(
            parameter.device.type != "meta"
            for layer in layers
            for parameter in layer.parameters()
        )
        if layer_owned:
            raise AssertionError(
                f"{case}/{mode}: retained {layer_owned} layer tensors"
            )
        evidence = {
            "case": case,
            "mode": mode,
            "prompt_tokens": len(token_ids),
            "input_token_ids": token_ids,
            "next_token": next_token,
            "prefill_seconds": prefill_seconds,
            "decode_seconds": decode_seconds,
            "physical_member_read_bytes": disk_delta_bytes(before, after),
            "before": before,
            "after": after,
            "direct_stats": stats_delta,
            "metal_stats_delta": metal_delta,
            "metal_state": metal_after,
            "phase_seconds": phase_delta,
            "routes": routes,
            "route_summary": route_summary(routes),
            "prefill_logits": logits_summary(prefill_cpu),
            "decode_logits": logits_summary(decode_cpu),
            "prefill_cache": prefill_cache,
            "decode_cache": decode_cache,
            "prefill_slab": (
                direct_shard_loader.prefill_slab_snapshot()
                if mode == "elastic" else None
            ),
            "layer_runtime_owned_tensors": layer_owned,
        }
        artifacts = {
            "prefill_logits": prefill_cpu,
            "decode_logits": decode_cpu,
        }
        return evidence, artifacts
    finally:
        kr.set_route_observer(old_observer)
        direct_shard_loader.prefill_slab_experts = old_provider
        del cache, embed
        gc.collect()
        torch.mps.empty_cache()
        torch.mps.synchronize()


def compare_modes(
    elastic: dict[str, Any],
    reference: dict[str, Any],
    elastic_artifacts: dict[str, torch.Tensor],
    reference_artifacts: dict[str, torch.Tensor],
) -> dict[str, Any]:
    prefill_exact = torch.equal(
        elastic_artifacts["prefill_logits"],
        reference_artifacts["prefill_logits"],
    )
    decode_exact = torch.equal(
        elastic_artifacts["decode_logits"],
        reference_artifacts["decode_logits"],
    )
    result = {
        "next_token_exact": elastic["next_token"] == reference["next_token"],
        "routes_exact": elastic["routes"] == reference["routes"],
        "prefill_logits_exact": prefill_exact,
        "prefill_logits_max_abs": float(
            (
                elastic_artifacts["prefill_logits"]
                - reference_artifacts["prefill_logits"]
            ).abs().max()
        ),
        "decode_logits_exact": decode_exact,
        "decode_logits_max_abs": float(
            (
                elastic_artifacts["decode_logits"]
                - reference_artifacts["decode_logits"]
            ).abs().max()
        ),
        "prefill_cache_exact": (
            elastic["prefill_cache"] == reference["prefill_cache"]
        ),
        "decode_cache_exact": (
            elastic["decode_cache"] == reference["decode_cache"]
        ),
    }
    result["passed"] = all(
        value for key, value in result.items() if key.endswith("_exact")
    )
    if not result["passed"]:
        raise AssertionError(f"elastic/legacy prefill parity failed: {result}")
    return result


def main(argv=None) -> int:
    args = parse_args(argv)
    require_profile()
    os.environ["K3_PROFILE"] = "0"
    os.environ["K3_PREFILL_LAST_LOGIT"] = "0"
    os.environ["K3_METAL_POSITION_BATCH"] = (
        "1" if args.position_batch else "0"
    )
    torch.set_grad_enabled(False)
    baseline = memory_snapshot("baseline", detailed=True)
    model_dir = pathlib.Path(os.environ["K3_MODEL_DIR"]).resolve()
    tree_before = model_tree_fingerprint(model_dir)
    evidence: dict[str, Any] = {
        "schema": "deltafin.m10-prefill.v1",
        "status": "running",
        "created_at": now(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "cases_requested": args.cases,
        "position_batch": args.position_batch,
        "baseline": baseline,
        "official_tree_before": tree_before,
        "official_model_modified": False,
        "model_conversion_performed": False,
        "cases": [],
    }
    write_evidence(args.output, evidence)
    failure = None
    kr = direct_shard_loader = resident_shard_loader = None
    layers = []
    try:
        import kimi_run as kr
        import direct_shard_loader
        import resident_shard_loader

        store = direct_shard_loader.store()
        source_before = model_fingerprint(store)
        evidence["source_before"] = source_before
        tokenizer = kr.k3_official.load_tokenizer(kr.ROOT)
        cases = prompt_cases(tokenizer, args.chat_text)
        evidence["prompt_cases"] = cases
        layers = kr.build_layers()
        bank = resident_shard_loader.runtime_bank()
        scratch = kr.resident_scratch_arena()
        if bank is None or scratch is None:
            raise AssertionError("resident bank/scratch is unavailable")
        evidence["resident_bank_initial"] = resident_bank_identity(bank)
        evidence["resident_scratch_initial"] = scratch.snapshot()
        evidence["runtime_loaded"] = memory_snapshot(
            "runtime-loaded", detailed=True
        )

        for case in CASE_ORDER:
            if case not in args.cases:
                continue
            ids = cases[case]
            print(
                f"[M10 {case}] {len(ids)} prompt tokens: elastic",
                flush=True,
            )
            case_record = {
                "case": case,
                "prompt_tokens": len(ids),
                "status": "running",
            }
            evidence["cases"].append(case_record)
            elastic, elastic_artifacts = run_mode(
                kr,
                direct_shard_loader,
                layers,
                case=case,
                token_ids=ids,
                mode="elastic",
            )
            case_record["elastic"] = elastic
            write_evidence(args.output, evidence)
            print(
                f"[M10 {case}] {len(ids)} prompt tokens: legacy heap oracle",
                flush=True,
            )
            reference, reference_artifacts = run_mode(
                kr,
                direct_shard_loader,
                layers,
                case=case,
                token_ids=ids,
                mode="legacy-heap",
            )
            parity = compare_modes(
                elastic,
                reference,
                elastic_artifacts,
                reference_artifacts,
            )
            case_record["legacy_heap"] = reference
            case_record["parity"] = parity
            case_record["status"] = "passed"
            write_evidence(args.output, evidence)
            del elastic_artifacts, reference_artifacts
            kr.metal_moe.flush()
            gc.collect()

        evidence["resident_bank_final"] = resident_bank_identity(bank)
        evidence["resident_scratch_final"] = scratch.snapshot()
        if evidence["resident_bank_initial"] != evidence["resident_bank_final"]:
            raise AssertionError("resident bank identity changed")
        initial_scratch = evidence["resident_scratch_initial"]
        final_scratch = evidence["resident_scratch_final"]
        if (
            initial_scratch["slot_pointers"]
            != final_scratch["slot_pointers"]
            or initial_scratch["storage_bytes"]
            != final_scratch["storage_bytes"]
        ):
            raise AssertionError("resident scratch identity changed")
        expected_preparations = (
            initial_scratch["preparations"]
            + kr.NL * 2 * 2 * len(evidence["cases"])
        )
        if final_scratch["preparations"] != expected_preparations:
            raise AssertionError(
                "scratch preparation count differs: "
                f"{final_scratch['preparations']}/{expected_preparations}"
            )
        source_after = model_fingerprint(store)
        tree_after = model_tree_fingerprint(model_dir)
        evidence["source_after"] = source_after
        evidence["official_tree_after"] = tree_after
        if source_before != source_after or tree_before != tree_after:
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
        if kr is not None:
            kr.set_route_observer(None)
            kr.metal_moe.flush()
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
