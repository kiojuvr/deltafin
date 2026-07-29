#!/usr/bin/env python3
"""M1 validation: resident direct reads, one real layer parity, aligned slabs."""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import datetime as dt
import gc
import importlib
import json
import mmap
import os
import pathlib
import platform
import resource
import subprocess
import sys
import time
import types
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from expert_slab import DoubleExpertSlab, K3_EXPERT_BYTES  # noqa: E402
from local_safetensors import LocalSafetensorsStore  # noqa: E402
from resident_shard_loader import (  # noqa: E402
    DirectResidentLoader,
    MaterializationReport,
    set_parameter,
)


ROOT = pathlib.Path(__file__).resolve().parent.parent
DTYPE_FROM_SAFE = {
    "BOOL": torch.bool,
    "I8": torch.int8,
    "U8": torch.uint8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F64": torch.float64,
}
def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def command_output(argv: list[str]) -> str | None:
    try:
        result = subprocess.run(
            argv, check=False, capture_output=True, text=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def peak_rss_bytes() -> int:
    """Return this process's peak RSS using the platform's native units."""
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Darwin reports bytes; Linux and the BSD-derived macOS ps report KiB.
    return value if sys.platform == "darwin" else value * 1024


def mps_bytes() -> int | None:
    if not torch.backends.mps.is_available():
        return None
    return int(torch.mps.current_allocated_memory())


def load_official_modules(model_dir: pathlib.Path):
    package_name = "deltafin_official_k3"
    package = types.ModuleType(package_name)
    package.__path__ = [str(model_dir)]
    sys.modules[package_name] = package
    modeling = importlib.import_module(f"{package_name}.modeling_kimi_linear")
    configuration = importlib.import_module(
        f"{package_name}.configuration_kimi_k3"
    )
    config_data = json.loads((model_dir / "config.json").read_text())["text_config"]
    config = configuration.KimiLinearConfig(**config_data)
    config._attn_implementation = "eager"
    return modeling, config


def new_meta_layer(modeling, config, layer_index: int):
    with torch.device("meta"):
        return modeling.KimiDecoderLayer(config, layer_index).eval()


def reference_materialize(
    store: LocalSafetensorsStore,
    module,
    prefix: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> MaterializationReport:
    started = time.perf_counter()
    checkpoint_bytes = 0
    materialized_bytes = 0
    count = 0
    transfer_seconds = 0.0
    handles = {}
    try:
        for parameter_name, _parameter in list(module.named_parameters()):
            if ".experts." in parameter_name:
                continue
            full_name = prefix + parameter_name
            span = store.tensor_span(full_name)
            handle = handles.get(span.shard)
            if handle is None:
                handle = safe_open(
                    str(store.model_dir / span.shard),
                    framework="pt",
                    device="cpu",
                )
                handles[span.shard] = handle
            source = handle.get_tensor(full_name)
            transfer_started = time.perf_counter()
            target = source.to(device=device, dtype=dtype, copy=True)
            transfer_seconds += time.perf_counter() - transfer_started
            set_parameter(module, parameter_name, target)
            checkpoint_bytes += span.byte_length
            materialized_bytes += target.numel() * target.element_size()
            count += 1
    finally:
        handles.clear()
    total = time.perf_counter() - started
    return MaterializationReport(
        tensors=count,
        checkpoint_bytes=checkpoint_bytes,
        materialized_bytes=materialized_bytes,
        read_seconds=max(0.0, total - transfer_seconds),
        transfer_seconds=transfer_seconds,
        total_seconds=total,
        device=str(device),
        dtype=str(dtype),
    )


def reference_experts(
    store: LocalSafetensorsStore, layer: int, expert_ids: list[int]
):
    layouts = [store.expert_layout(layer, expert) for expert in expert_ids]
    handles = {}
    output = {}
    try:
        for layout in layouts:
            parts = {}
            for span in layout.tensors:
                handle = handles.get(span.shard)
                if handle is None:
                    handle = safe_open(
                        str(store.model_dir / span.shard),
                        framework="numpy",
                        device="cpu",
                    )
                    handles[span.shard] = handle
                suffix = store._expert_suffix(span.name)
                parts[suffix] = handle.get_tensor(span.name).copy(order="C")
            output[layout.expert] = {
                weight: (
                    parts[f"{weight}.weight_packed"],
                    parts[f"{weight}.weight_scale"],
                )
                for weight in ("w1", "w2", "w3")
            }
    finally:
        handles.clear()
    return output


def direct_experts(
    store: LocalSafetensorsStore, layer: int, expert_ids: list[int]
):
    tensors = store.read_experts(layer, expert_ids, workers=8)
    return {
        expert: {
            weight: (
                values[f"{weight}.weight_packed"],
                values[f"{weight}.weight_scale"],
            )
            for weight in ("w1", "w2", "w3")
        }
        for expert, values in tensors.items()
    }


def capture_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.detach().to(device="cpu", dtype=value.dtype, copy=True)


def run_official_layer(
    modeling,
    layer,
    hidden: torch.Tensor,
    block_residual: torch.Tensor,
    expert_provider: Callable[[int, list[int]], dict],
) -> tuple[dict[str, torch.Tensor], float, dict[str, float]]:
    """Execute the official layer forward while capturing its semantic stages."""
    import fast_moe_batch

    captures: dict[str, torch.Tensor] = {}
    timings = {
        "expert_read_seconds": 0.0,
        "expert_compute_seconds": 0.0,
    }
    hooks = []
    original_apply = modeling._apply_attn_res
    apply_count = 0

    def capture_hook(name):
        def hook(_module, _inputs, output):
            captures[name] = capture_tensor(output)
        return hook

    def wrapped_apply(*args, **kwargs):
        nonlocal apply_count
        result = original_apply(*args, **kwargs)
        name = "attn_res_input" if apply_count == 0 else "moe_res_input"
        captures[name] = capture_tensor(result)
        apply_count += 1
        return result

    gate = layer.block_sparse_moe.gate
    original_gate_forward = gate.forward

    def gate_forward(states):
        flat = states.view(-1, states.shape[-1])
        captures["router_logits"] = capture_tensor(
            F.linear(
                flat.to(torch.float32),
                gate.weight.to(torch.float32),
                None,
            )
        )
        topk_ids, topk_weights = original_gate_forward(states)
        captures["topk_ids"] = capture_tensor(topk_ids)
        captures["topk_weights"] = capture_tensor(topk_weights)
        return topk_ids, topk_weights

    moe = layer.block_sparse_moe
    original_moe_infer = moe.moe_infer

    def native_moe(x, topk_ids, topk_weight):
        captures["latent_input"] = capture_tensor(x)
        ids = [int(expert) for expert in topk_ids[0].tolist()]
        read_started = time.perf_counter()
        raw = expert_provider(layer.layer_idx, ids)
        timings["expert_read_seconds"] += time.perf_counter() - read_started
        compute_started = time.perf_counter()
        output = fast_moe_batch.moe_infer_fast(
            x, topk_ids, topk_weight, raw
        )
        timings["expert_compute_seconds"] += (
            time.perf_counter() - compute_started
        )
        captures["expert_output"] = capture_tensor(output)
        return output

    hooks.extend(
        (
            layer.input_layernorm.register_forward_hook(
                capture_hook("input_norm")
            ),
            layer.self_attn.register_forward_hook(
                capture_hook("attention_output")
            ),
            layer.post_attention_layernorm.register_forward_hook(
                capture_hook("post_attention_norm")
            ),
            moe.routed_expert_norm.register_forward_hook(
                capture_hook("routed_after_norm")
            ),
            moe.routed_expert_up_proj.register_forward_hook(
                capture_hook("routed_up_flat")
            ),
            moe.shared_experts.register_forward_hook(
                capture_hook("shared_output")
            ),
            moe.register_forward_hook(capture_hook("moe_sum")),
        )
    )
    modeling._apply_attn_res = wrapped_apply
    gate.forward = gate_forward
    moe.moe_infer = native_moe
    started = time.perf_counter()
    try:
        with torch.no_grad():
            output, returned_block = layer(
                hidden,
                attention_mask=None,
                past_key_values=None,
                output_attentions=False,
                use_cache=False,
                block_residual=block_residual,
            )
        if returned_block.data_ptr() != block_residual.data_ptr():
            raise AssertionError("layer 1 unexpectedly replaced block residual")
        if hidden.device.type == "mps":
            torch.mps.synchronize()
    finally:
        elapsed = time.perf_counter() - started
        modeling._apply_attn_res = original_apply
        gate.forward = original_gate_forward
        moe.moe_infer = original_moe_infer
        for hook in hooks:
            hook.remove()
    captures["prefix_after_attention"] = (
        capture_tensor(hidden) + captures["attention_output"]
    )
    captures["routed_up"] = captures.pop("routed_up_flat").view_as(
        captures["shared_output"]
    )
    captures["layer_output"] = capture_tensor(output)
    timings["non_expert_layer_seconds"] = max(
        0.0,
        elapsed
        - timings["expert_read_seconds"]
        - timings["expert_compute_seconds"],
    )
    return captures, elapsed, timings


def compare_stages(reference, direct) -> dict[str, Any]:
    if set(reference) != set(direct):
        raise AssertionError(
            f"stage sets differ: {set(reference) ^ set(direct)}"
        )
    results = {}
    for name in sorted(reference):
        expected, actual = reference[name], direct[name]
        if expected.shape != actual.shape or expected.dtype != actual.dtype:
            raise AssertionError(
                f"{name}: {expected.dtype}{tuple(expected.shape)} != "
                f"{actual.dtype}{tuple(actual.shape)}"
            )
        if name == "topk_ids":
            equal = torch.equal(expected, actual)
            results[name] = {"exact": equal, "shape": list(expected.shape)}
            if not equal:
                raise AssertionError(
                    f"top-k IDs differ: {expected.tolist()} != {actual.tolist()}"
                )
            continue
        delta = (expected.to(torch.float64) - actual.to(torch.float64)).abs()
        maximum = float(delta.max()) if delta.numel() else 0.0
        denominator = expected.to(torch.float64).abs().clamp_min(1e-12)
        relative = float((delta / denominator).max()) if delta.numel() else 0.0
        exact = torch.equal(expected, actual)
        passed = torch.allclose(expected, actual, rtol=1e-5, atol=1e-6)
        results[name] = {
            "exact": exact,
            "allclose": bool(passed),
            "max_abs": maximum,
            "max_rel": relative,
            "shape": list(expected.shape),
            "dtype": str(expected.dtype),
        }
        if not passed:
            raise AssertionError(
                f"{name}: parity failed, max_abs={maximum}, max_rel={relative}"
            )
    return results


def validate_resident_bytes(
    store: LocalSafetensorsStore,
    loader: DirectResidentLoader,
    layer: int,
) -> dict[str, Any]:
    names = loader.layer_names(layer)
    shards = {store.tensor_span(name).shard for name in names}
    handles = {
        shard: safe_open(
            str(store.model_dir / shard), framework="pt", device="cpu"
        )
        for shard in shards
    }
    checked_bytes = 0
    try:
        for name in names:
            span = store.tensor_span(name)
            reference = handles[span.shard].get_tensor(name)
            direct = loader.read_bytes(name)
            if tuple(reference.shape) != span.shape:
                raise AssertionError(f"{name}: shape mismatch")
            if reference.dtype != DTYPE_FROM_SAFE[span.dtype]:
                raise AssertionError(f"{name}: dtype mismatch")
            reference_bytes = (
                reference.contiguous().view(torch.uint8).numpy().tobytes()
            )
            if reference_bytes != direct:
                raise AssertionError(f"{name}: raw bytes differ")
            checked_bytes += len(direct)
    finally:
        handles.clear()
    return {
        "status": "passed",
        "layer": layer,
        "tensors": len(names),
        "bytes": checked_bytes,
    }


def validate_embedding_rows(
    store: LocalSafetensorsStore,
    loader: DirectResidentLoader,
) -> dict[str, Any]:
    name = "language_model.model.embed_tokens.weight"
    span = store.tensor_span(name)
    rows = [0, span.shape[0] // 2, span.shape[0] - 1]
    direct = loader.read_rows_torch(name, rows)
    with safe_open(
        str(store.model_dir / span.shard), framework="pt", device="cpu"
    ) as handle:
        tensor_slice = handle.get_slice(name)
        reference = torch.stack([tensor_slice[row] for row in rows])
    if direct.dtype != reference.dtype or direct.shape != reference.shape:
        raise AssertionError("embedding row dtype/shape mismatch")
    if not torch.equal(direct, reference):
        raise AssertionError("embedding rows differ")
    return {
        "status": "passed",
        "rows": rows,
        "shape": list(direct.shape),
        "dtype": str(direct.dtype),
        "bytes_read": direct.numel() * direct.element_size(),
    }


def validate_slab_and_metal(
    store: LocalSafetensorsStore,
    *,
    layer: int,
    expert_ids: list[int],
    latent_input: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> dict[str, Any]:
    import fast_moe_batch
    import metal_moe

    result: dict[str, Any] = {
        "metal_available": metal_moe.available(),
    }
    with DoubleExpertSlab(store) as slabs:
        if not all(bank.page_aligned for bank in slabs.banks):
            raise AssertionError("expert slab is not page-aligned")
        initial_addresses = tuple(
            bank.slot_addresses for bank in slabs.banks
        )
        read_started = time.perf_counter()
        raw_a = slabs.banks[0].load(layer, expert_ids, workers=8)
        read_seconds = time.perf_counter() - read_started
        zero_copy_pointers = []
        for slot, expert in enumerate(expert_ids):
            pointer, _owner = metal_moe._span_ptr(raw_a[expert], slot)
            expected = slabs.banks[0].slot_addresses[slot]
            if pointer != expected or pointer % mmap.PAGESIZE:
                raise AssertionError(
                    f"E{expert} is not structurally zero-copy eligible"
                )
            zero_copy_pointers.append(pointer)
        sample_expert = expert_ids[0]
        standard = direct_experts(store, layer, [sample_expert])[sample_expert]
        for weight in ("w1", "w2", "w3"):
            for part in (0, 1):
                if not np.array_equal(
                    raw_a[sample_expert][weight][part],
                    standard[weight][part],
                ):
                    raise AssertionError(
                        f"slab bytes differ for E{sample_expert} {weight}[{part}]"
                    )
        result.update(
            {
                "status": "passed",
                "bank_count": 2,
                "capacity_per_bank": 16,
                "bytes_per_bank": slabs.banks[0].nbytes,
                "total_bytes": slabs.nbytes,
                "page_size": os.sysconf("SC_PAGE_SIZE"),
                "page_aligned": True,
                "load_seconds": read_seconds,
                "load_gb_s": (
                    len(expert_ids) * K3_EXPERT_BYTES / read_seconds / 1e9
                ),
                "sample_byte_parity": True,
                "metal_zero_copy_contract": {
                    "status": "structurally-eligible",
                    "contiguous_experts": len(zero_copy_pointers),
                    "page_aligned_pointers": len(zero_copy_pointers),
                    "native_span_pointer_matches_slot": True,
                    "runtime_counter_verified": False,
                },
            }
        )
        later_layers = [
            candidate for candidate in store.routed_layers if candidate > layer
        ]
        earlier_layers = [
            candidate for candidate in store.routed_layers if candidate < layer
        ]
        next_layers = later_layers + earlier_layers
        if len(next_layers) < 2:
            raise AssertionError("slab reuse validation needs two other layers")
        next_layer = next_layers[0]
        reuse_layer = next_layers[1]
        if not result["metal_available"]:
            result["metal_skip_reason"] = metal_moe.last_error()
            read_b_started = time.perf_counter()
            raw_b = slabs.banks[1].load(next_layer, expert_ids, workers=8)
            result["bank_b_load_seconds"] = time.perf_counter() - read_b_started
            result["bank_b_loaded_layer"] = next_layer
            del raw_b
            del raw_a
            reuse_started = time.perf_counter()
            raw_reused = slabs.banks[0].load(
                reuse_layer, expert_ids, workers=8
            )
            result["bank_a_reuse_load_seconds"] = (
                time.perf_counter() - reuse_started
            )
            result["bank_a_reused_layer"] = reuse_layer
            result["pointer_reuse_stable"] = (
                initial_addresses
                == tuple(bank.slot_addresses for bank in slabs.banks)
            )
            if not result["pointer_reuse_stable"]:
                raise AssertionError("slab slot address changed after reuse")
            result["metal_zero_copy_contract"]["runtime_reason"] = (
                result["metal_skip_reason"]
            )
            del raw_reused
            return result

        metal_moe.flush()
        before = metal_moe.stats()

        def load_next_bank():
            started = time.perf_counter()
            raw = slabs.banks[1].load(
                next_layer, expert_ids, workers=8
            )
            return raw, time.perf_counter() - started

        with concurrent.futures.ThreadPoolExecutor(
            1, thread_name_prefix="k3-next-layer-slab"
        ) as pool:
            overlap_started = time.perf_counter()
            future_b = pool.submit(load_next_bank)
            metal_started = time.perf_counter()
            metal_output = metal_moe.moe_infer(
                latent_input, topk_ids, topk_weights, raw_a
            )
            metal_seconds = time.perf_counter() - metal_started
            raw_b, bank_b_read_seconds = future_b.result()
            overlap_seconds = time.perf_counter() - overlap_started
        metal_b_started = time.perf_counter()
        metal_output_b = metal_moe.moe_infer(
            latent_input, topk_ids, topk_weights, raw_b
        )
        metal_b_seconds = time.perf_counter() - metal_b_started
        after = metal_moe.stats()
        cpu_output = fast_moe_batch.moe_infer_fast(
            latent_input, topk_ids, topk_weights, raw_a
        )
        cpu_output_b = fast_moe_batch.moe_infer_fast(
            latent_input, topk_ids, topk_weights, raw_b
        )
        delta = {
            key: after[key] - before.get(key, 0)
            for key in ("calls", "zero_copy_wraps", "copies")
        }
        if (
            delta["copies"] != 0
            or delta["zero_copy_wraps"] != 2 * len(expert_ids)
        ):
            raise AssertionError(f"Metal used scratch copies: {delta}")
        difference = (metal_output - cpu_output).abs()
        max_abs = float(difference.max())
        if not torch.allclose(metal_output, cpu_output, rtol=1e-4, atol=1e-5):
            raise AssertionError(f"Metal/CPU expert output differs: {max_abs}")
        difference_b = (metal_output_b - cpu_output_b).abs()
        max_abs_b = float(difference_b.max())
        if not torch.allclose(
            metal_output_b, cpu_output_b, rtol=1e-4, atol=1e-5
        ):
            raise AssertionError(
                f"bank B Metal/CPU expert output differs: {max_abs_b}"
            )
        result["metal"] = {
            "bank_a_seconds": metal_seconds,
            "bank_b_seconds": metal_b_seconds,
            "bank_b_read_seconds": bank_b_read_seconds,
            "bank_a_compute_overlapped_with_bank_b_read_seconds": overlap_seconds,
            "bank_b_loaded_layer": next_layer,
            "counter_delta": delta,
            "bank_a_max_abs_vs_cpu": max_abs,
            "bank_b_max_abs_vs_cpu": max_abs_b,
            "allclose_vs_cpu": True,
        }
        result["metal_zero_copy_contract"]["runtime_counter_verified"] = True
        metal_moe.flush()
        del (
            cpu_output,
            cpu_output_b,
            metal_output,
            metal_output_b,
            raw_a,
            raw_b,
        )
        reuse_started = time.perf_counter()
        raw_reused = slabs.banks[0].load(
            reuse_layer, expert_ids, workers=8
        )
        result["bank_a_reuse_load_seconds"] = (
            time.perf_counter() - reuse_started
        )
        result["bank_a_reused_layer"] = reuse_layer
        result["pointer_reuse_stable"] = (
            initial_addresses
            == tuple(bank.slot_addresses for bank in slabs.banks)
        )
        if not result["pointer_reuse_stable"]:
            raise AssertionError("slab slot address changed after reuse")
        del raw_reused
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=pathlib.Path,
        default=pathlib.Path(
            os.environ.get(
                "K3_MODEL_DIR",
                "/Volumes/USB-SSD-RAID-0/models/moonshotai/Kimi-K3",
            )
        ),
    )
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--sidecar",
        type=pathlib.Path,
        default=ROOT / "bench-results/k3-direct-shards-index.json.gz",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=ROOT / "bench-results/m1-layer-parity.json",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps"),
        default="auto",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    torch.set_grad_enabled(False)
    torch.set_num_threads(args.threads)
    if args.device == "auto":
        device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    else:
        device = torch.device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")

    evidence: dict[str, Any] = {
        "schema": "deltafin.m1-layer-parity.v1",
        "created_at": now(),
        "model_dir": str(args.model_dir.resolve()),
        "layer": args.layer,
        "device": str(device),
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "mps_available": torch.backends.mps.is_available(),
            "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        },
        "model_files_modified": False,
        "model_conversion_performed": False,
    }
    rss_start = peak_rss_bytes()
    mps_start = mps_bytes()
    modeling, config = load_official_modules(args.model_dir)

    with LocalSafetensorsStore(
        args.model_dir, sidecar=args.sidecar
    ) as store:
        loader = DirectResidentLoader(store)
        print("[1/5] resident tensor byte parity", flush=True)
        evidence["resident_validation"] = validate_resident_bytes(
            store, loader, args.layer
        )
        evidence["embedding_validation"] = validate_embedding_rows(store, loader)

        hidden = torch.linspace(
            -0.5, 0.5, config.hidden_size, dtype=torch.float32, device=device
        ).view(1, 1, config.hidden_size)
        block_residual = torch.linspace(
            0.25, -0.25, config.hidden_size,
            dtype=torch.float32,
            device=device,
        ).view(1, 1, config.hidden_size)
        prefix = f"language_model.model.layers.{args.layer}."
        layer_spans = [
            store.tensor_span(name) for name in loader.layer_names(args.layer)
        ]
        max_resident_tensor_bytes = max(
            span.byte_length for span in layer_spans
        )

        print("[2/5] standard safe_open layer forward", flush=True)
        reference_layer = new_meta_layer(modeling, config, args.layer)
        rss_before_reference = peak_rss_bytes()
        mps_before_reference = mps_bytes()
        reference_report = reference_materialize(
            store,
            reference_layer,
            prefix,
            device=device,
            dtype=torch.float32,
        )
        (
            reference_stages,
            reference_forward_s,
            reference_stage_timing,
        ) = run_official_layer(
            modeling,
            reference_layer,
            hidden,
            block_residual,
            lambda layer, ids: reference_experts(store, layer, ids),
        )
        rss_after_reference = peak_rss_bytes()
        mps_after_reference = mps_bytes()
        del reference_layer
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
        mps_after_reference_release = mps_bytes()

        print("[3/5] direct pread layer forward", flush=True)
        direct_layer = new_meta_layer(modeling, config, args.layer)
        rss_before_direct = peak_rss_bytes()
        mps_before_direct = mps_bytes()
        direct_report = loader.materialize_module(
            direct_layer,
            prefix,
            device=device,
            dtype=torch.float32,
        )
        (
            direct_stages,
            direct_forward_s,
            direct_stage_timing,
        ) = run_official_layer(
            modeling,
            direct_layer,
            hidden,
            block_residual,
            lambda layer, ids: direct_experts(store, layer, ids),
        )
        rss_after_direct = peak_rss_bytes()
        mps_after_direct = mps_bytes()

        print("[4/5] stage-by-stage parity", flush=True)
        evidence["stage_parity"] = compare_stages(
            reference_stages, direct_stages
        )
        evidence["layer_forward"] = {
            "status": "passed",
            "standard_seconds": reference_forward_s,
            "direct_seconds": direct_forward_s,
            "standard_stage_timing": reference_stage_timing,
            "direct_stage_timing": direct_stage_timing,
            "topk_ids": direct_stages["topk_ids"].tolist(),
            "standard_materialization": dataclasses.asdict(reference_report),
            "direct_materialization": dataclasses.asdict(direct_report),
        }

        print("[5/5] aligned double slab and Metal copy counters", flush=True)
        expert_ids = [
            int(expert) for expert in direct_stages["topk_ids"][0].tolist()
        ]
        evidence["slab_validation"] = validate_slab_and_metal(
            store,
            layer=args.layer,
            expert_ids=expert_ids,
            latent_input=direct_stages["latent_input"],
            topk_ids=direct_stages["topk_ids"],
            topk_weights=direct_stages["topk_weights"],
        )
        del direct_layer

    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    evidence["memory"] = {
        "rss_start": rss_start,
        "rss_before_reference": rss_before_reference,
        "rss_after_reference": rss_after_reference,
        "rss_before_direct": rss_before_direct,
        "rss_after_direct": rss_after_direct,
        "rss_end": peak_rss_bytes(),
        "mps_start": mps_start,
        "mps_before_reference": mps_before_reference,
        "mps_after_reference": mps_after_reference,
        "mps_after_reference_release": mps_after_reference_release,
        "mps_before_direct": mps_before_direct,
        "mps_after_direct": mps_after_direct,
        "mps_end": mps_bytes(),
        "checkpoint_layer_bytes": direct_report.checkpoint_bytes,
        "materialized_layer_bytes": direct_report.materialized_bytes,
        "max_transient_source_tensor_bytes": max_resident_tensor_bytes,
        "aligned_double_slab_bytes": 2 * 16 * K3_EXPERT_BYTES,
    }
    evidence["status"] = "passed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + f".tmp{os.getpid()}")
    temporary.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n"
    )
    os.replace(temporary, args.output)
    print(f"PASS: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
