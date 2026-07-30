#!/usr/bin/env python3
"""Validate exact immutable prefix-state reuse against full K3 prefill."""
from __future__ import annotations

import argparse
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
from prefix_state import PrefixStateSnapshot, derive_chat_prefix  # noqa: E402
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
from validate_m10_prefill import (  # noqa: E402
    cache_digest,
    route_observer,
    route_summary,
)


ROOT = pathlib.Path(__file__).resolve().parent.parent


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("smoke", "chat"), default="chat")
    parser.add_argument("--chat-text", default="Hello")
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=ROOT / "bench-results/m11-prefix-state.json",
    )
    return parser.parse_args(argv)


def prompt_and_prefix(tokenizer, case: str, chat_text: str):
    if case == "smoke":
        prompt = tokenizer.encode(
            "The capital of France is Paris, and Europe"
        )[:8]
        prefix = tuple(prompt[:4])
    else:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": chat_text}],
            tokenize=True,
            add_generation_prompt=True,
        )
        prefix = derive_chat_prefix(tokenizer)
    prompt = tuple(int(token) for token in prompt)
    if not 0 < len(prefix) < len(prompt):
        raise AssertionError(
            f"invalid prefix boundary {len(prefix)}/{len(prompt)}"
        )
    if prompt[: len(prefix)] != tuple(prefix):
        raise AssertionError("prompt does not begin with derived prefix")
    return prompt, tuple(prefix)


def phase_delta(kr, direct_shard_loader, before):
    return {
        "direct_stats": numeric_delta(
            direct_shard_loader.stats, before["direct"]
        ),
        "metal_stats": numeric_delta(
            kr.metal_moe.stats(), before["metal"]
        ),
        "runtime_seconds": numeric_delta(kr.TIMES, before["times"]),
    }


def phase_before(kr, direct_shard_loader):
    return {
        "direct": dict(direct_shard_loader.stats),
        "metal": kr.metal_moe.stats(),
        "times": dict(kr.TIMES),
    }


def assert_direct_exact(label: str, phase: dict[str, Any]) -> None:
    direct = phase["direct_stats"]
    metal = phase["metal_stats"]
    if direct["expert_http"] or direct["http_bytes"]:
        raise AssertionError(f"{label}: HTTP access detected")
    if metal.get("copies", 0):
        raise AssertionError(f"{label}: Metal staging copy detected")


def run_full(kr, direct_shard_loader, layers, embed, prompt):
    reset_runtime_counters(kr, direct_shard_loader)
    routes = {}
    observer = kr.set_route_observer(route_observer(routes))
    cache = kr.ml.KimiDynamicCache(kr.config)
    before = memory_snapshot("full-before")
    counters = phase_before(kr, direct_shard_loader)
    try:
        started = time.perf_counter()
        kr._step_ctx["step"] = 0
        logits = kr.forward_pass(
            layers, cache, embed(prompt), step=0, verbose=False
        )
        kr._device_synchronize()
        prefill_seconds = time.perf_counter() - started
        prefill_logits = logits.detach().to(torch.float32).cpu()
        prefill_cache = cache_digest(cache)
        next_token = int(prefill_logits[0, -1].argmax())

        started = time.perf_counter()
        kr._step_ctx["step"] = 1
        decode = kr.forward_pass(
            layers, cache, embed([next_token]), step=1, verbose=False
        )
        kr._device_synchronize()
        decode_seconds = time.perf_counter() - started
        decode_logits = decode.detach().to(torch.float32).cpu()
        decode_cache = cache_digest(cache)
        after = memory_snapshot("full-after")
        delta = phase_delta(kr, direct_shard_loader, counters)
        assert_direct_exact("full", delta)
        evidence = {
            "prefill_seconds": prefill_seconds,
            "decode_seconds": decode_seconds,
            "physical_member_read_bytes": disk_delta_bytes(before, after),
            "before": before,
            "after": after,
            "routes": routes,
            "route_summary": route_summary(routes),
            "prefill_logits": logits_summary(prefill_logits),
            "decode_logits": logits_summary(decode_logits),
            "prefill_cache": prefill_cache,
            "decode_cache": decode_cache,
            "next_token": next_token,
            **delta,
        }
        artifacts = {
            "prefill_logits": prefill_logits,
            "decode_logits": decode_logits,
        }
        return evidence, artifacts
    finally:
        kr.set_route_observer(observer)
        del cache
        gc.collect()
        torch.mps.empty_cache()
        kr._device_synchronize()


def build_prefix(
    kr, direct_shard_loader, layers, embed, prefix
):
    reset_runtime_counters(kr, direct_shard_loader)
    routes = {}
    observer = kr.set_route_observer(route_observer(routes))
    cache = kr.ml.KimiDynamicCache(kr.config)
    before = memory_snapshot("prefix-build-before")
    counters = phase_before(kr, direct_shard_loader)
    try:
        started = time.perf_counter()
        kr._step_ctx["step"] = 0
        logits = kr.forward_pass(
            layers, cache, embed(prefix), step=0, verbose=False
        )
        kr._device_synchronize()
        seconds = time.perf_counter() - started
        logits_cpu = logits.detach().to(torch.float32).cpu()
        snapshot = PrefixStateSnapshot.capture(cache, prefix)
        state = cache_digest(cache)
        after = memory_snapshot("prefix-build-after")
        delta = phase_delta(kr, direct_shard_loader, counters)
        assert_direct_exact("prefix build", delta)
        evidence = {
            "seconds": seconds,
            "physical_member_read_bytes": disk_delta_bytes(before, after),
            "before": before,
            "after": after,
            "routes": routes,
            "route_summary": route_summary(routes),
            "logits": logits_summary(logits_cpu),
            "cache": state,
            "snapshot": snapshot.snapshot(),
            **delta,
        }
        return cache, snapshot, evidence, logits_cpu
    finally:
        kr.set_route_observer(observer)


def run_suffix(
    kr,
    direct_shard_loader,
    layers,
    embed,
    prompt,
    snapshot,
    *,
    cache=None,
    restore: bool,
    label: str,
):
    reset_runtime_counters(kr, direct_shard_loader)
    routes = {}
    observer = kr.set_route_observer(route_observer(routes))
    if cache is None:
        cache = kr.ml.KimiDynamicCache(kr.config)
    before = memory_snapshot(f"{label}-before")
    counters = phase_before(kr, direct_shard_loader)
    try:
        restore_started = time.perf_counter_ns()
        if restore:
            restored_tokens = snapshot.restore(cache)
        else:
            restored_tokens = int(cache.get_seq_length() or 0)
            if restored_tokens != len(snapshot.prefix_ids):
                raise AssertionError(
                    f"{label}: live prefix length {restored_tokens} "
                    f"!= {len(snapshot.prefix_ids)}"
                )
        restore_ns = time.perf_counter_ns() - restore_started
        suffix = snapshot.suffix(prompt)
        started = time.perf_counter()
        kr._step_ctx["step"] = 0
        logits = kr.forward_pass(
            layers, cache, embed(suffix), step=0, verbose=False
        )
        kr._device_synchronize()
        suffix_seconds = time.perf_counter() - started
        suffix_logits = logits.detach().to(torch.float32).cpu()
        prefill_cache = cache_digest(cache)
        next_token = int(suffix_logits[0, -1].argmax())

        started = time.perf_counter()
        kr._step_ctx["step"] = 1
        decode = kr.forward_pass(
            layers, cache, embed([next_token]), step=1, verbose=False
        )
        kr._device_synchronize()
        decode_seconds = time.perf_counter() - started
        decode_logits = decode.detach().to(torch.float32).cpu()
        decode_cache = cache_digest(cache)
        after = memory_snapshot(f"{label}-after")
        delta = phase_delta(kr, direct_shard_loader, counters)
        assert_direct_exact(label, delta)
        snapshot.assert_intact()
        evidence = {
            "restored_tokens": restored_tokens,
            "restore_ns": restore_ns,
            "suffix_tokens": len(suffix),
            "suffix_seconds": suffix_seconds,
            "decode_seconds": decode_seconds,
            "physical_member_read_bytes": disk_delta_bytes(before, after),
            "before": before,
            "after": after,
            "routes": routes,
            "route_summary": route_summary(routes),
            "suffix_logits": logits_summary(suffix_logits),
            "decode_logits": logits_summary(decode_logits),
            "prefill_cache": prefill_cache,
            "decode_cache": decode_cache,
            "next_token": next_token,
            "snapshot_after": snapshot.snapshot(),
            **delta,
        }
        artifacts = {
            "suffix_logits": suffix_logits,
            "decode_logits": decode_logits,
        }
        return evidence, artifacts
    finally:
        kr.set_route_observer(observer)


def combine_routes(prefix_routes, suffix_routes):
    combined = {}
    prefix_keys = {key for key in prefix_routes if key.startswith("0:")}
    suffix_keys = {key for key in suffix_routes if key.startswith("0:")}
    if prefix_keys != suffix_keys:
        raise AssertionError(
            "prefix/suffix routed layer sets differ: "
            f"{sorted(prefix_keys ^ suffix_keys)}"
        )
    for key in prefix_keys:
        combined[key] = prefix_routes[key] + suffix_routes[key]
    return combined


def compare_monolithic(
    full,
    full_artifacts,
    prefix,
    prefix_logits,
    segmented,
    segmented_artifacts,
    prefix_tokens,
    *,
    tolerance=1e-4,
):
    combined = combine_routes(prefix["routes"], segmented["routes"])
    full_prefill_routes = {
        key: rows
        for key, rows in full["routes"].items()
        if key.startswith("0:")
    }
    full_decode_routes = {
        key: rows
        for key, rows in full["routes"].items()
        if key.startswith("1:")
    }
    segmented_decode_routes = {
        key: rows
        for key, rows in segmented["routes"].items()
        if key.startswith("1:")
    }
    prefix_exact = torch.equal(
        full_artifacts["prefill_logits"][:, :prefix_tokens],
        prefix_logits,
    )
    suffix_exact = torch.equal(
        full_artifacts["prefill_logits"][:, prefix_tokens:],
        segmented_artifacts["suffix_logits"],
    )
    decode_exact = torch.equal(
        full_artifacts["decode_logits"],
        segmented_artifacts["decode_logits"],
    )
    prefix_max_abs = float(
        (
            full_artifacts["prefill_logits"][:, :prefix_tokens]
            - prefix_logits
        ).abs().max()
    )
    suffix_max_abs = float(
        (
            full_artifacts["prefill_logits"][:, prefix_tokens:]
            - segmented_artifacts["suffix_logits"]
        ).abs().max()
    )
    decode_max_abs = float(
        (
            full_artifacts["decode_logits"]
            - segmented_artifacts["decode_logits"]
        ).abs().max()
    )
    result = {
        "prefix_routes_exact": all(
            full_prefill_routes[key][:prefix_tokens] == rows
            for key, rows in prefix["routes"].items()
        ),
        "combined_prefill_routes_exact": combined == full_prefill_routes,
        "decode_routes_exact": segmented_decode_routes == full_decode_routes,
        "prefix_logits_exact": prefix_exact,
        "prefix_logits_max_abs": prefix_max_abs,
        "suffix_logits_exact": suffix_exact,
        "suffix_logits_max_abs": suffix_max_abs,
        "next_token_exact": (
            full["next_token"] == segmented["next_token"]
        ),
        "prefill_cache_exact": (
            full["prefill_cache"] == segmented["prefill_cache"]
        ),
        "decode_logits_exact": decode_exact,
        "decode_logits_max_abs": decode_max_abs,
        "decode_cache_exact": (
            full["decode_cache"] == segmented["decode_cache"]
        ),
        "tolerance": tolerance,
    }
    result["passed"] = (
        result["prefix_routes_exact"]
        and result["combined_prefill_routes_exact"]
        and result["decode_routes_exact"]
        and result["next_token_exact"]
        and prefix_max_abs <= tolerance
        and suffix_max_abs <= tolerance
        and decode_max_abs <= tolerance
    )
    if not result["passed"]:
        raise AssertionError(
            f"monolithic compatibility failed: {result}"
        )
    return result


def compare_exact(
    segmented,
    segmented_artifacts,
    reused,
    reused_artifacts,
):
    suffix_exact = torch.equal(
        segmented_artifacts["suffix_logits"],
        reused_artifacts["suffix_logits"],
    )
    decode_exact = torch.equal(
        segmented_artifacts["decode_logits"],
        reused_artifacts["decode_logits"],
    )
    result = {
        "routes_exact": segmented["routes"] == reused["routes"],
        "next_token_exact": (
            segmented["next_token"] == reused["next_token"]
        ),
        "suffix_logits_exact": suffix_exact,
        "suffix_logits_max_abs": float(
            (
                segmented_artifacts["suffix_logits"]
                - reused_artifacts["suffix_logits"]
            ).abs().max()
        ),
        "prefill_cache_exact": (
            segmented["prefill_cache"] == reused["prefill_cache"]
        ),
        "decode_logits_exact": decode_exact,
        "decode_logits_max_abs": float(
            (
                segmented_artifacts["decode_logits"]
                - reused_artifacts["decode_logits"]
            ).abs().max()
        ),
        "decode_cache_exact": (
            segmented["decode_cache"] == reused["decode_cache"]
        ),
    }
    result["passed"] = all(
        value for key, value in result.items() if key.endswith("_exact")
    )
    if not result["passed"]:
        raise AssertionError(f"exact prefix-state parity failed: {result}")
    return result


def main(argv=None) -> int:
    args = parse_args(argv)
    require_profile()
    os.environ["K3_PROFILE"] = "0"
    os.environ["K3_PREFILL_LAST_LOGIT"] = "0"
    os.environ["K3_METAL_POSITION_BATCH"] = "1"
    torch.set_grad_enabled(False)
    baseline = memory_snapshot("baseline", detailed=True)
    model_dir = pathlib.Path(os.environ["K3_MODEL_DIR"]).resolve()
    tree_before = model_tree_fingerprint(model_dir)
    evidence: dict[str, Any] = {
        "schema": "deltafin.m11-prefix-state.v1",
        "status": "running",
        "case": args.case,
        "created_at": now(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "baseline": baseline,
        "official_tree_before": tree_before,
        "official_model_modified": False,
        "model_conversion_performed": False,
    }
    write_evidence(args.output, evidence)
    failure = None
    kr = direct_shard_loader = resident_shard_loader = None
    layers = []
    snapshot = None
    embed = None
    full_artifacts = prefix_logits = None
    segmented_artifacts = reused_artifacts = None
    try:
        import kimi_run as kr
        import direct_shard_loader
        import resident_shard_loader

        store = direct_shard_loader.store()
        source_before = model_fingerprint(store)
        evidence["source_before"] = source_before
        tokenizer = kr.k3_official.load_tokenizer(kr.ROOT)
        prompt, prefix_ids = prompt_and_prefix(
            tokenizer, args.case, args.chat_text
        )
        evidence["prompt_tokens"] = len(prompt)
        evidence["prefix_tokens"] = len(prefix_ids)
        evidence["suffix_tokens"] = len(prompt) - len(prefix_ids)
        evidence["prompt_ids"] = list(prompt)
        evidence["prefix_ids"] = list(prefix_ids)
        layers = kr.build_layers()
        embed = kr.LazyEmbed()
        bank = resident_shard_loader.runtime_bank()
        scratch = kr.resident_scratch_arena()
        evidence["resident_bank_initial"] = resident_bank_identity(bank)
        evidence["resident_scratch_initial"] = scratch.snapshot()
        evidence["runtime_loaded"] = memory_snapshot(
            "runtime-loaded", detailed=True
        )

        print(
            f"[M11 {args.case}] full oracle: {len(prompt)} tokens",
            flush=True,
        )
        full, full_artifacts = run_full(
            kr, direct_shard_loader, layers, embed, prompt
        )
        evidence["full"] = full
        write_evidence(args.output, evidence)

        print(
            f"[M11 {args.case}] build prefix: {len(prefix_ids)} tokens",
            flush=True,
        )
        prefix_cache, snapshot, prefix, prefix_logits = build_prefix(
            kr, direct_shard_loader, layers, embed, prefix_ids
        )
        evidence["prefix_build"] = prefix
        write_evidence(args.output, evidence)

        print(
            f"[M11 {args.case}] uncached segmented suffix: "
            f"{len(prompt) - len(prefix_ids)} tokens",
            flush=True,
        )
        segmented, segmented_artifacts = run_suffix(
            kr,
            direct_shard_loader,
            layers,
            embed,
            prompt,
            snapshot,
            cache=prefix_cache,
            restore=False,
            label="segmented",
        )
        prefix_cache = None
        gc.collect()
        torch.mps.empty_cache()
        kr._device_synchronize()
        evidence["segmented"] = segmented
        write_evidence(args.output, evidence)

        print(
            f"[M11 {args.case}] restore + suffix: "
            f"{len(prompt) - len(prefix_ids)} tokens",
            flush=True,
        )
        reused, reused_artifacts = run_suffix(
            kr,
            direct_shard_loader,
            layers,
            embed,
            prompt,
            snapshot,
            restore=True,
            label="reused",
        )
        gc.collect()
        torch.mps.empty_cache()
        kr._device_synchronize()
        evidence["reused"] = reused
        evidence["exact_reuse_parity"] = compare_exact(
            segmented,
            segmented_artifacts,
            reused,
            reused_artifacts,
        )
        write_evidence(args.output, evidence)
        evidence["monolithic_compatibility"] = compare_monolithic(
            full,
            full_artifacts,
            prefix,
            prefix_logits,
            segmented,
            segmented_artifacts,
            len(prefix_ids),
        )

        pristine = kr.ml.KimiDynamicCache(kr.config)
        snapshot.restore(pristine)
        evidence["snapshot_pristine_cache"] = cache_digest(pristine)
        evidence["snapshot_pristine_exact"] = (
            evidence["snapshot_pristine_cache"] == prefix["cache"]
        )
        if not evidence["snapshot_pristine_exact"]:
            raise AssertionError("prefix snapshot values changed after reuse")
        del pristine

        evidence["resident_bank_final"] = resident_bank_identity(bank)
        evidence["resident_scratch_final"] = scratch.snapshot()
        if evidence["resident_bank_initial"] != evidence["resident_bank_final"]:
            raise AssertionError("resident bank identity changed")
        if (
            evidence["resident_scratch_initial"]["slot_pointers"]
            != evidence["resident_scratch_final"]["slot_pointers"]
        ):
            raise AssertionError("resident scratch identity changed")
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
        snapshot = None
        embed = None
        full_artifacts = prefix_logits = None
        segmented_artifacts = reused_artifacts = None
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
