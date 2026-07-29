#!/usr/bin/env python3
"""Validate the runtime direct-shard slab lease against CPU and Metal MoE."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import direct_shard_loader  # noqa: E402
import fast_moe_batch  # noqa: E402
import metal_moe  # noqa: E402
from local_safetensors import LocalSafetensorsStore  # noqa: E402


ROOT = pathlib.Path(__file__).resolve().parent.parent
ROUTE = (
    712, 461, 308, 828, 846, 348, 154, 617,
    521, 511, 753, 133, 503, 675, 692, 715,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=pathlib.Path,
        default=pathlib.Path(os.environ["K3_MODEL_DIR"]),
    )
    parser.add_argument(
        "--sidecar",
        type=pathlib.Path,
        default=ROOT / "bench-results/k3-direct-shards-index.json.gz",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=ROOT / "bench-results/direct-slab-adapter.json",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not direct_shard_loader.slab_enabled():
        raise RuntimeError("K3_DIRECT_SLAB=1 is required")
    if not metal_moe.available():
        raise RuntimeError(f"Metal unavailable: {metal_moe.last_error()}")
    direct_shard_loader.close()
    direct_shard_loader._store = LocalSafetensorsStore(
        args.model_dir, sidecar=args.sidecar, darwin_nocache=False
    )
    topk_ids = torch.tensor([ROUTE], dtype=torch.int64)
    topk_weights = torch.linspace(
        1.0, 0.25, len(ROUTE), dtype=torch.float32
    ).view(1, -1)
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
    latent = torch.linspace(-0.5, 0.5, 3584, dtype=torch.float32).view(1, -1)
    evidence = {
        "schema": "deltafin.direct-slab-adapter.v1",
        "model_dir": str(args.model_dir.resolve()),
        "route": list(ROUTE),
        "calls": [],
    }
    pointers = []
    try:
        metal_moe.flush()
        before = metal_moe.stats()
        for layer in (1, 46):
            read_started = time.perf_counter()
            with direct_shard_loader.slab_experts(layer, ROUTE) as raw:
                read_seconds = time.perf_counter() - read_started
                pointers.append(
                    tuple(raw[expert]["w1"][0].ctypes.data for expert in ROUTE)
                )
                metal_started = time.perf_counter()
                metal_output = metal_moe.moe_infer(
                    latent, topk_ids, topk_weights, raw
                )
                metal_seconds = time.perf_counter() - metal_started
                cpu_output = fast_moe_batch.moe_infer_fast(
                    latent, topk_ids, topk_weights, raw
                )
                difference = (metal_output - cpu_output).abs()
                max_abs = float(difference.max())
                if not torch.allclose(
                    metal_output, cpu_output, rtol=1e-4, atol=1e-5
                ):
                    raise AssertionError(
                        f"layer {layer}: Metal/CPU differs by {max_abs}"
                    )
                evidence["calls"].append(
                    {
                        "layer": layer,
                        "read_seconds": read_seconds,
                        "metal_seconds": metal_seconds,
                        "max_abs_vs_cpu": max_abs,
                    }
                )
                del cpu_output, metal_output
            del raw
        with direct_shard_loader.slab_experts(92, ROUTE) as raw:
            reused = tuple(
                raw[expert]["w1"][0].ctypes.data for expert in ROUTE
            )
            evidence["bank_a_pointer_reused"] = reused == pointers[0]
            if not evidence["bank_a_pointer_reused"]:
                raise AssertionError("bank A addresses changed on runtime reuse")
        del raw
        after = metal_moe.stats()
        delta = {
            key: after[key] - before.get(key, 0)
            for key in ("calls", "zero_copy_wraps", "copies")
        }
        evidence["counter_delta"] = delta
        if delta != {"calls": 2, "zero_copy_wraps": 32, "copies": 0}:
            raise AssertionError(f"unexpected Metal counters: {delta}")
        evidence["adapter_stats"] = dict(direct_shard_loader.stats)
        evidence["status"] = "passed"
    finally:
        metal_moe.flush()
        direct_shard_loader.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + f".tmp{os.getpid()}")
    temporary.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(f"PASS: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
