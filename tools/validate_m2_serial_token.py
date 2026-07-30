#!/usr/bin/env python3
"""Validate one complete serial K3 token through the ordinary direct runtime."""

from __future__ import annotations

import argparse
import contextlib
import copy
import datetime as dt
import gc
import hashlib
import json
import os
import pathlib
import sys
import time
from contextlib import ExitStack
from typing import Any

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validate_m1d_resident import (  # noqa: E402
    disk_delta_bytes,
    memory_snapshot,
    model_fingerprint,
    write_evidence,
)


ROOT = pathlib.Path(__file__).resolve().parent.parent
EXPERT_BYTES = 17_547_264


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def model_tree_fingerprint(model_dir: pathlib.Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(model_dir.rglob("*")):
        if not path.is_file():
            continue
        stat = path.stat()
        rows.append(
            {
                "name": str(path.relative_to(model_dir)),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token-id", type=int, default=1008)
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=ROOT / "bench-results/m2-serial-token.json",
    )
    parser.add_argument(
        "--skip-standard-reference",
        action="store_true",
        help="run only the ordinary direct-slab pass",
    )
    return parser.parse_args(argv)


def require_profile() -> None:
    expected = {
        "K3_EXPERT_SOURCE": "direct-shards",
        "K3_RESIDENT_SOURCE": "direct-shards",
        "K3_RESIDENT_BANK": "1",
        "K3_DIRECT_SLAB": "1",
        "K3_PREAD_NOCACHE": "0",
        "K3_DTYPE": "fp32",
        "K3_APPROX": "0",
        "K3_PRELOAD": "0",
        "K3_EXPERT_PREFETCH": "0",
        "K3_PREFETCH": "0",
        "K3_PILOT": "0",
        "K3_MOE_GROUP_SIZE": "0",
        "K3_TEMPLATES": "0",
        "K3_PIN_LAYERS": "93",
        "K3_PROFILE": "1",
        "K3_TRACE": "off",
        "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "1.0",
        "PYTORCH_MPS_LOW_WATERMARK_RATIO": "0.9",
    }
    wrong = {
        key: {"expected": value, "actual": os.environ.get(key)}
        for key, value in expected.items()
        if os.environ.get(key) != value
    }
    if wrong:
        raise RuntimeError(f"M2 serial profile mismatch: {wrong}")
    if not torch.backends.mps.is_available():
        raise RuntimeError("M2 serial token validation requires MPS")


def reset_runtime_counters(kr, direct_shard_loader) -> None:
    for key in kr.TIMES:
        kr.TIMES[key] = 0.0
    for key in kr.PROF:
        kr.PROF[key] = 0
    kr.LAYER_PROFILE.clear()
    kr._LAST_SEL.clear()
    kr._PREV_SEL.clear()
    for key in direct_shard_loader.stats:
        direct_shard_loader.stats[key] = (
            0.0 if key.endswith("_s") else 0
        )


def tensor_digest(tensor: torch.Tensor) -> str:
    array = tensor.detach().to(torch.float32).contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def logits_summary(logits: torch.Tensor) -> dict[str, Any]:
    values = logits.detach().to(torch.float32).contiguous().cpu()
    final = values[0, -1]
    top = torch.topk(final, 10)
    return {
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "sha256_f32": tensor_digest(values),
        "argmax": int(final.argmax()),
        "top10_ids": top.indices.tolist(),
        "top10_logits": [float(value) for value in top.values],
        "finite": bool(torch.isfinite(values).all()),
    }


def routes_summary(kr) -> dict[str, list[int]]:
    routes = {
        str(layer): [int(expert) for expert in experts]
        for layer, experts in sorted(kr._LAST_SEL.items())
    }
    expected_layers = set(range(1, kr.NL))
    actual_layers = {int(layer) for layer in routes}
    if actual_layers != expected_layers:
        raise AssertionError(
            f"routed layers differ: missing={sorted(expected_layers-actual_layers)}, "
            f"extra={sorted(actual_layers-expected_layers)}"
        )
    wrong = {
        layer: len(experts)
        for layer, experts in routes.items()
        if len(experts) != kr.BASE_MOE_TOP_K
    }
    if wrong:
        raise AssertionError(f"unexpected routed expert counts: {wrong}")
    return routes


def run_pass(kr, direct_shard_loader, layers, token_id: int, label: str):
    reset_runtime_counters(kr, direct_shard_loader)
    cache = kr.ml.KimiDynamicCache(kr.config)
    embed = kr.LazyEmbed()
    hidden = embed([token_id])
    before = memory_snapshot(f"{label}-before")
    started = time.perf_counter()
    kr._step_ctx["step"] = 0
    logits = kr.forward_pass(
        layers, cache, hidden, step=0, verbose=True
    )
    torch.mps.synchronize()
    elapsed = time.perf_counter() - started
    after = memory_snapshot(f"{label}-after")
    cpu_logits = logits.detach().to(torch.float32).cpu()
    routes = routes_summary(kr)
    result = {
        "label": label,
        "seconds": elapsed,
        "before": before,
        "after": after,
        "physical_member_read_bytes": disk_delta_bytes(before, after),
        "phase_seconds": dict(kr.TIMES),
        "layer_profile": copy.deepcopy(kr.LAYER_PROFILE),
        "routes": routes,
        "logits": logits_summary(cpu_logits),
        "direct_stats": dict(direct_shard_loader.stats),
    }
    result["selected_experts"] = sum(len(row) for row in routes.values())
    result["logical_expert_bytes"] = (
        result["selected_experts"] * EXPERT_BYTES
    )
    physical = result["physical_member_read_bytes"]
    if physical is not None:
        result["inferred_page_cache_bytes"] = max(
            0, result["logical_expert_bytes"] - physical
        )
    del logits, hidden, embed, cache
    return cpu_logits, result


class StandardExpertProvider:
    """Provide the routed tensors through safetensors.safe_open."""

    def __init__(self, store):
        self.store = store
        self.calls = 0
        self.experts = 0
        self.bytes = 0
        self.seconds = 0.0

    @contextlib.contextmanager
    def read(self, layer: int, eids, workers=None):
        del workers
        ids = tuple(int(expert) for expert in eids)
        started = time.perf_counter()
        with ExitStack() as stack:
            handles = {}
            result = {}
            for expert in ids:
                layout = self.store.expert_layout(layer, expert)
                prefix = (
                    f"language_model.model.layers.{layer}."
                    f"block_sparse_moe.experts.{expert}."
                )
                tensors = {}
                for span in layout.tensors:
                    handle = handles.get(span.shard)
                    if handle is None:
                        handle = stack.enter_context(
                            safe_open(
                                str(self.store.model_dir / span.shard),
                                framework="np",
                            )
                        )
                        handles[span.shard] = handle
                    tensors[span.name.removeprefix(prefix)] = (
                        handle.get_tensor(span.name)
                    )
                result[expert] = {
                    weight: (
                        np.ascontiguousarray(
                            tensors[f"{weight}.weight_packed"]
                        ),
                        np.ascontiguousarray(
                            tensors[f"{weight}.weight_scale"]
                        ),
                    )
                    for weight in ("w1", "w2", "w3")
                }
                self.bytes += layout.byte_length
            elapsed = time.perf_counter() - started
            self.calls += 1
            self.experts += len(ids)
            self.seconds += elapsed
            yield result

    def report(self):
        return {
            "calls": self.calls,
            "experts": self.experts,
            "bytes": self.bytes,
            "seconds": self.seconds,
        }


def parity_result(
    direct: torch.Tensor,
    reference: torch.Tensor,
    direct_routes: dict[str, list[int]],
    reference_routes: dict[str, list[int]],
) -> dict[str, Any]:
    difference = (direct - reference).abs()
    denominator = reference.abs().clamp_min(1e-12)
    exact = bool(torch.equal(direct, reference))
    routes_exact = direct_routes == reference_routes
    result = {
        "routes_exact": routes_exact,
        "logits_exact": exact,
        "max_abs": float(difference.max()),
        "max_rel": float((difference / denominator).max()),
        "argmax_exact": int(direct[0, -1].argmax())
        == int(reference[0, -1].argmax()),
        "allclose_rtol_1e_5_atol_1e_6": bool(
            torch.allclose(direct, reference, rtol=1e-5, atol=1e-6)
        ),
    }
    if not routes_exact:
        raise AssertionError("direct and standard-reader routes differ")
    if not exact:
        raise AssertionError(
            f"direct and standard-reader logits differ: {result}"
        )
    return result


def cleanup_runtime(kr, direct_shard_loader, resident_shard_loader, layers):
    kr._LM_W = None
    kr._LM_Q = None
    kr._LM_SC = None
    kr._TAIL = None
    layers.clear()
    if hasattr(kr, "release_resident_scratch"):
        kr.release_resident_scratch()
    resident_shard_loader.release_runtime_bank()
    direct_shard_loader.close()
    gc.collect()
    torch.mps.empty_cache()
    torch.mps.synchronize()


def main(argv=None) -> int:
    args = parse_args(argv)
    require_profile()
    torch.set_grad_enabled(False)
    baseline = memory_snapshot("baseline", detailed=True)
    model_dir = pathlib.Path(os.environ["K3_MODEL_DIR"]).expanduser().resolve()
    tree_before = model_tree_fingerprint(model_dir)
    evidence: dict[str, Any] = {
        "schema": "deltafin.m2-serial-token.v1",
        "status": "running",
        "created_at": now(),
        "token_id": args.token_id,
        "environment": {
            "git_commit": None,
            "profile": {
                key: value
                for key, value in sorted(os.environ.items())
                if key.startswith(("K3_", "PYTORCH_"))
            },
        },
        "baseline": baseline,
        "official_tree_before": tree_before,
        "model_conversion_performed": False,
        "model_files_modified": False,
        "serial_controls": {
            "resident_preload": False,
            "expert_prefetch": False,
            "general_prefetch": False,
            "router_pilot": False,
            "grouped_moe": False,
            "pread_nocache": False,
        },
    }
    write_evidence(args.output, evidence)
    failure = None
    kr = direct_shard_loader = resident_shard_loader = None
    layers = []
    source_before = None
    try:
        import subprocess

        evidence["environment"]["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        import kimi_run as kr
        import direct_shard_loader
        import resident_shard_loader

        store = direct_shard_loader.store()
        source_before = model_fingerprint(store)
        evidence["source_before"] = source_before
        tokenizer = kr.k3_official.load_tokenizer(kr.ROOT)
        evidence["tokenizer"] = {
            "class": type(tokenizer).__name__,
            "metadata_dir": str(kr.K3_METADATA_DIR),
            "probe_text": "The capital of France is",
            "probe_ids": tokenizer.encode("The capital of France is"),
        }
        if kr.K3_METADATA_DIR.resolve() != store.model_dir:
            raise AssertionError("runtime metadata is not the official model dir")
        layers = kr.build_layers()
        if len(layers) != kr.NL:
            raise AssertionError(f"built {len(layers)}/{kr.NL} layers")

        direct_logits, direct_result = run_pass(
            kr, direct_shard_loader, layers, args.token_id, "direct-slab"
        )
        evidence["direct"] = direct_result
        expected_expert_bytes = (kr.NL - 1) * kr.BASE_MOE_TOP_K * EXPERT_BYTES
        if direct_result["logical_expert_bytes"] != expected_expert_bytes:
            raise AssertionError("one-token expert-byte total differs")
        if (
            direct_result["direct_stats"]["slab_loads"] != kr.NL - 1
            or direct_result["direct_stats"]["pread_experts"]
            != (kr.NL - 1) * kr.BASE_MOE_TOP_K
            or direct_result["direct_stats"]["pread_bytes"]
            != expected_expert_bytes
        ):
            raise AssertionError("ordinary direct-slab call totals differ")
        if (
            direct_result["direct_stats"]["expert_http"] != 0
            or direct_result["direct_stats"]["http_bytes"] != 0
        ):
            raise AssertionError("direct runtime attempted HTTP access")
        write_evidence(args.output, evidence)

        if not args.skip_standard_reference:
            provider = StandardExpertProvider(store)
            original = direct_shard_loader.slab_experts
            direct_shard_loader.slab_experts = provider.read
            try:
                reference_logits, reference_result = run_pass(
                    kr,
                    direct_shard_loader,
                    layers,
                    args.token_id,
                    "standard-safetensors",
                )
            finally:
                direct_shard_loader.slab_experts = original
            provider_report = provider.report()
            reference_result["standard_reader"] = provider_report
            if (
                provider_report["calls"] != kr.NL - 1
                or provider_report["experts"]
                != (kr.NL - 1) * kr.BASE_MOE_TOP_K
                or provider_report["bytes"] != expected_expert_bytes
            ):
                raise AssertionError("standard-reader call totals differ")
            evidence["reference"] = reference_result
            evidence["parity"] = parity_result(
                direct_logits,
                reference_logits,
                direct_result["routes"],
                reference_result["routes"],
            )
            del reference_logits

        source_after = model_fingerprint(store)
        evidence["source_after"] = source_after
        if source_after != source_before:
            raise AssertionError("official model metadata fingerprint changed")
        tree_after = model_tree_fingerprint(model_dir)
        evidence["official_tree_after"] = tree_after
        if tree_after != tree_before:
            raise AssertionError("official model directory tree changed")
        evidence["status"] = "passed"
        del tokenizer, direct_logits
    except BaseException as exc:
        failure = exc
        evidence["status"] = "failed"
        evidence["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
    finally:
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
        release_ok = (
            abs(evidence["release"]["current_delta_from_baseline"])
            <= 64 * 1024 * 1024
            and evidence["release"]["driver_delta_from_baseline"]
            <= 1024 * 1024 * 1024
            and released["swap"].get("used_bytes")
            == baseline["swap"].get("used_bytes")
        )
        evidence["release"]["passed"] = release_ok
        if evidence["status"] == "passed" and not release_ok:
            evidence["status"] = "failed"
            evidence["error"] = {
                "type": "ReleaseValidationError",
                "message": "MPS or swap did not return to the serial baseline",
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
