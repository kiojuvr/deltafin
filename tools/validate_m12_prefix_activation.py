#!/usr/bin/env python3
"""Validate shape-stable fixed-prefix MoE activation reuse for K3."""
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
from prefix_activation import PrefixActivationSession  # noqa: E402
from prefix_state import derive_chat_prefix  # noqa: E402
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
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=ROOT / "bench-results/m12-prefix-activation.json",
    )
    return parser.parse_args(argv)


def prompt_pair(tokenizer, case: str):
    if case == "smoke":
        first = tuple(
            tokenizer.encode(
                "The capital of France is Paris, and Europe"
            )[:8]
        )
        alternate = tuple(
            tokenizer.encode("Germany is Berlin today")[:4]
        )
        if len(first) != 8 or len(alternate) != 4:
            raise AssertionError("smoke tokenizer lengths changed")
        second = first[:4] + alternate
        prefix = first[:4]
    else:
        first = tuple(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": "Hello"}],
                tokenize=True,
                add_generation_prompt=True,
            )
        )
        second = tuple(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": "World"}],
                tokenize=True,
                add_generation_prompt=True,
            )
        )
        prefix = derive_chat_prefix(tokenizer)
    if len(first) != len(second):
        raise AssertionError(
            f"prompt pair shape differs: {len(first)}/{len(second)}"
        )
    if first[: len(prefix)] != prefix or second[: len(prefix)] != prefix:
        raise AssertionError("prompt pair does not share the fixed prefix")
    if first == second:
        raise AssertionError("prompt pair suffix must differ")
    return first, second, tuple(prefix)


def run_pass(
    kr,
    direct_shard_loader,
    layers,
    embed,
    token_ids,
    *,
    label: str,
    activation_session=None,
    decode: bool,
):
    reset_runtime_counters(kr, direct_shard_loader)
    routes = {}
    previous_observer = kr.set_route_observer(route_observer(routes))
    previous_session = kr.set_moe_activation_session(activation_session)
    session_installed = True
    cache = kr.ml.KimiDynamicCache(kr.config)
    before = memory_snapshot(f"{label}-before")
    stats_before = dict(direct_shard_loader.stats)
    metal_before = kr.metal_moe.stats()
    times_before = dict(kr.TIMES)
    try:
        started = time.perf_counter()
        kr._step_ctx["step"] = 0
        logits = kr.forward_pass(
            layers,
            cache,
            embed(token_ids),
            step=0,
            verbose=False,
        )
        kr._device_synchronize()
        prefill_seconds = time.perf_counter() - started
        prefill_logits = logits.detach().to(torch.float32).cpu()
        kr.set_moe_activation_session(previous_session)
        session_installed = False
        prefill_cache = cache_digest(cache)
        next_token = int(prefill_logits[0, -1].argmax())

        decode_seconds = None
        decode_logits = None
        decode_cache = None
        if decode:
            started = time.perf_counter()
            kr._step_ctx["step"] = 1
            decoded = kr.forward_pass(
                layers,
                cache,
                embed([next_token]),
                step=1,
                verbose=False,
            )
            kr._device_synchronize()
            decode_seconds = time.perf_counter() - started
            decode_logits = decoded.detach().to(torch.float32).cpu()
            decode_cache = cache_digest(cache)
        after = memory_snapshot(f"{label}-after")
        direct_delta = numeric_delta(
            direct_shard_loader.stats, stats_before
        )
        metal_delta = numeric_delta(kr.metal_moe.stats(), metal_before)
        if direct_delta["expert_http"] or direct_delta["http_bytes"]:
            raise AssertionError(f"{label}: HTTP access detected")
        if metal_delta.get("copies", 0):
            raise AssertionError(f"{label}: Metal staging copy detected")
        evidence = {
            "label": label,
            "prompt_tokens": len(token_ids),
            "input_token_ids": list(token_ids),
            "prefill_seconds": prefill_seconds,
            "decode_seconds": decode_seconds,
            "physical_member_read_bytes": disk_delta_bytes(before, after),
            "before": before,
            "after": after,
            "direct_stats": direct_delta,
            "metal_stats": metal_delta,
            "runtime_seconds": numeric_delta(kr.TIMES, times_before),
            "routes": routes,
            "route_summary": route_summary(routes),
            "prefill_logits": logits_summary(prefill_logits),
            "prefill_cache": prefill_cache,
            "next_token": next_token,
            "decode_logits": (
                logits_summary(decode_logits)
                if decode_logits is not None else None
            ),
            "decode_cache": decode_cache,
        }
        artifacts = {
            "prefill_logits": prefill_logits,
            "decode_logits": decode_logits,
        }
        return evidence, artifacts
    finally:
        if session_installed:
            kr.set_moe_activation_session(previous_session)
        kr.set_route_observer(previous_observer)
        del cache
        gc.collect()
        torch.mps.empty_cache()
        kr._device_synchronize()


def compare_exact(oracle, oracle_artifacts, replay, replay_artifacts):
    prefill_exact = torch.equal(
        oracle_artifacts["prefill_logits"],
        replay_artifacts["prefill_logits"],
    )
    decode_exact = torch.equal(
        oracle_artifacts["decode_logits"],
        replay_artifacts["decode_logits"],
    )
    result = {
        "routes_exact": oracle["routes"] == replay["routes"],
        "prefill_logits_exact": prefill_exact,
        "prefill_logits_max_abs": float(
            (
                oracle_artifacts["prefill_logits"]
                - replay_artifacts["prefill_logits"]
            ).abs().max()
        ),
        "prefill_cache_exact": (
            oracle["prefill_cache"] == replay["prefill_cache"]
        ),
        "next_token_exact": oracle["next_token"] == replay["next_token"],
        "decode_logits_exact": decode_exact,
        "decode_logits_max_abs": float(
            (
                oracle_artifacts["decode_logits"]
                - replay_artifacts["decode_logits"]
            ).abs().max()
        ),
        "decode_cache_exact": (
            oracle["decode_cache"] == replay["decode_cache"]
        ),
    }
    result["passed"] = all(
        value for key, value in result.items() if key.endswith("_exact")
    )
    if not result["passed"]:
        raise AssertionError(f"prefix activation parity failed: {result}")
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
        "schema": "deltafin.m12-prefix-activation.v1",
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
    embed = None
    session = None
    capture_artifacts = oracle_artifacts = replay_artifacts = None
    try:
        import kimi_run as kr
        import direct_shard_loader
        import resident_shard_loader

        store = direct_shard_loader.store()
        source_before = model_fingerprint(store)
        evidence["source_before"] = source_before
        tokenizer = kr.k3_official.load_tokenizer(kr.ROOT)
        capture_ids, target_ids, prefix_ids = prompt_pair(
            tokenizer, args.case
        )
        evidence["prompt_tokens"] = len(target_ids)
        evidence["prefix_tokens"] = len(prefix_ids)
        evidence["suffix_tokens"] = len(target_ids) - len(prefix_ids)
        evidence["capture_prompt_ids"] = list(capture_ids)
        evidence["target_prompt_ids"] = list(target_ids)
        layers = kr.build_layers()
        embed = kr.LazyEmbed()
        bank = resident_shard_loader.runtime_bank()
        scratch = kr.resident_scratch_arena()
        evidence["resident_bank_initial"] = resident_bank_identity(bank)
        evidence["resident_scratch_initial"] = scratch.snapshot()
        evidence["runtime_loaded"] = memory_snapshot(
            "runtime-loaded", detailed=True
        )

        session = PrefixActivationSession(
            len(prefix_ids), len(target_ids)
        )
        print(
            f"[M12 {args.case}] capture prompt: "
            f"{len(target_ids)} positions / {len(prefix_ids)} prefix",
            flush=True,
        )
        capture, capture_artifacts = run_pass(
            kr,
            direct_shard_loader,
            layers,
            embed,
            capture_ids,
            label="capture",
            activation_session=session,
            decode=False,
        )
        evidence["capture"] = capture
        evidence["capture_activation"] = session.snapshot()
        evidence["capture_activation_digest"] = session.value_digest()
        session.arm_replay(expected_layers=kr.NL - 1)
        write_evidence(args.output, evidence)

        print(f"[M12 {args.case}] target oracle", flush=True)
        oracle, oracle_artifacts = run_pass(
            kr,
            direct_shard_loader,
            layers,
            embed,
            target_ids,
            label="oracle",
            decode=True,
        )
        evidence["oracle"] = oracle
        write_evidence(args.output, evidence)

        print(f"[M12 {args.case}] target activation replay", flush=True)
        replay, replay_artifacts = run_pass(
            kr,
            direct_shard_loader,
            layers,
            embed,
            target_ids,
            label="replay",
            activation_session=session,
            decode=True,
        )
        session.finish_replay(expected_layers=kr.NL - 1)
        evidence["replay"] = replay
        evidence["replay_activation"] = session.snapshot()
        evidence["replay_activation_digest"] = session.value_digest()
        evidence["activation_immutable"] = (
            evidence["capture_activation_digest"]
            == evidence["replay_activation_digest"]
        )
        if not evidence["activation_immutable"]:
            raise AssertionError("captured prefix activations changed")
        evidence["parity"] = compare_exact(
            oracle, oracle_artifacts, replay, replay_artifacts
        )

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
        if kr is not None:
            kr.set_moe_activation_session(None)
        if session is not None:
            session.close()
        session = None
        embed = None
        capture_artifacts = oracle_artifacts = replay_artifacts = None
        if kr is not None:
            kr.set_route_observer(None)
            kr.metal_moe.flush()
            cleanup_runtime(
                kr, direct_shard_loader, resident_shard_loader, layers
            )
        released = memory_snapshot("released", detailed=True)
        swap_delta = (
            released["swap"].get("used_bytes", 0)
            - baseline["swap"].get("used_bytes", 0)
        )
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
            "swap_delta_from_baseline": swap_delta,
            "empty_cache_called": True,
        }
        evidence["release"]["passed"] = (
            abs(evidence["release"]["current_delta_from_baseline"])
            <= 64 * 1024 * 1024
            and evidence["release"]["driver_delta_from_baseline"]
            <= 1024 * 1024 * 1024
            and swap_delta <= 16 * 1024 * 1024
        )
        if evidence["status"] == "passed" and not evidence["release"]["passed"]:
            evidence["status"] = "failed"
            evidence["error"] = {
                "type": "ReleaseValidationError",
                "message": "MPS or swap did not return near baseline",
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
