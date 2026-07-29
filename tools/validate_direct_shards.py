#!/usr/bin/env python3
"""Validate and benchmark direct reads from the official local Kimi-K3 shards.

The default invocation performs all structural checks, compares nine experts
against the standard safetensors implementation, builds and reloads a compact
sidecar, and records single/16-expert I/O timings as JSON.

Example:

    set -a; source config/m3ultra512.env; set +a
    .venv/bin/python tools/validate_direct_shards.py \
        --output bench-results/m3ultra512-direct-shards.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import dataclasses
import gc
import json
import os
import pathlib
import platform
import random
import statistics
import subprocess
import sys
import time
from typing import Any

import numpy as np
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from local_safetensors import (  # noqa: E402
    LocalSafetensorsStore,
    UnknownExpertError,
    UnknownTensorError,
)


ROOT = pathlib.Path(__file__).resolve().parent.parent
EXPECTED_SHARDS = 96
EXPECTED_LAYERS = tuple(range(1, 93))
EXPECTED_EXPERTS_PER_LAYER = 896
EXPECTED_TENSORS = {
    "w1.weight_packed": ("U8", (3072, 1792)),
    "w1.weight_scale": ("U8", (3072, 112)),
    "w2.weight_packed": ("U8", (3584, 1536)),
    "w2.weight_scale": ("U8", (3584, 96)),
    "w3.weight_packed": ("U8", (3072, 1792)),
    "w3.weight_scale": ("U8", (3072, 112)),
}
EXPECTED_EXPERT_BYTES = 17_547_264


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def git_output(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def fd_count() -> int | None:
    for location in ("/dev/fd", "/proc/self/fd"):
        try:
            return len(os.listdir(location))
        except OSError:
            continue
    return None


def quantiles(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "median": statistics.median(ordered),
        "max": ordered[-1],
    }


def timed(callable_) -> tuple[Any, float]:
    started = time.perf_counter()
    result = callable_()
    return result, time.perf_counter() - started


def structure_check(store: LocalSafetensorsStore) -> dict[str, Any]:
    """Inspect every routed expert without reading its payload."""
    report = store.report()
    if report.shards != EXPECTED_SHARDS:
        raise AssertionError(f"expected {EXPECTED_SHARDS} shards, found {report.shards}")
    if store.routed_layers != EXPECTED_LAYERS:
        raise AssertionError(
            f"routed layers differ: expected {EXPECTED_LAYERS}, found "
            f"{store.routed_layers}"
        )

    contiguous = 0
    cross_shard = 0
    layouts_by_read_count: dict[int, int] = {}
    shard_expert_counts: dict[str, int] = {}
    span_lengths: dict[int, int] = {}
    checked_tensors = 0
    contiguous_layer_blocks = 0
    lexicographic_layer_blocks = 0
    layer_shards: dict[int, str] = {}
    for layer in EXPECTED_LAYERS:
        expert_ids = store.expert_ids(layer)
        if expert_ids != tuple(range(EXPECTED_EXPERTS_PER_LAYER)):
            raise AssertionError(f"layer {layer}: expert IDs are not 0..895")
        layer_layouts = []
        for expert in expert_ids:
            layout = store.expert_layout(layer, expert)
            layer_layouts.append(layout)
            if layout.byte_length != EXPECTED_EXPERT_BYTES:
                raise AssertionError(
                    f"layer {layer} expert {expert}: {layout.byte_length} bytes"
                )
            suffixes = {
                store._expert_suffix(tensor.name): (tensor.dtype, tensor.shape)
                for tensor in layout.tensors
            }
            if suffixes != EXPECTED_TENSORS:
                raise AssertionError(
                    f"layer {layer} expert {expert}: unexpected tensors {suffixes}"
                )
            grouped_names = [
                name for read_span in layout.read_spans for name in read_span.tensor_names
            ]
            if set(grouped_names) != {tensor.name for tensor in layout.tensors}:
                raise AssertionError(
                    f"layer {layer} expert {expert}: read spans do not cover tensors"
                )
            for read_span in layout.read_spans:
                shard_size, header_len = store._shard_info[read_span.shard]
                if (
                    read_span.offset < 8 + header_len
                    or read_span.byte_length <= 0
                    or read_span.end > shard_size
                ):
                    raise AssertionError(
                        f"layer {layer} expert {expert}: out-of-range read span"
                    )
            contiguous += int(layout.contiguous)
            cross_shard += int(len(layout.shard_names) > 1)
            layouts_by_read_count[len(layout.read_spans)] = (
                layouts_by_read_count.get(len(layout.read_spans), 0) + 1
            )
            span_lengths[layout.byte_length] = span_lengths.get(layout.byte_length, 0) + 1
            for shard in layout.shard_names:
                shard_expert_counts[shard] = shard_expert_counts.get(shard, 0) + 1
            checked_tensors += len(layout.tensors)
        ordered_layouts = sorted(layer_layouts, key=lambda item: item.read_spans[0].offset)
        layer_shard_names = {
            shard for layout in ordered_layouts for shard in layout.shard_names
        }
        if len(layer_shard_names) != 1:
            raise AssertionError(f"layer {layer}: experts cross shard boundaries")
        layer_shards[layer] = next(iter(layer_shard_names))
        if all(
            left.read_spans[-1].end == right.read_spans[0].offset
            for left, right in zip(ordered_layouts, ordered_layouts[1:])
        ):
            contiguous_layer_blocks += 1
        if [layout.expert for layout in ordered_layouts] == sorted(
            range(EXPECTED_EXPERTS_PER_LAYER), key=str
        ):
            lexicographic_layer_blocks += 1

    return {
        "status": "passed",
        "checked_experts": store.routed_expert_count,
        "checked_expert_tensors": checked_tensors,
        "contiguous_experts": contiguous,
        "cross_shard_experts": cross_shard,
        "read_span_count_distribution": {
            str(key): value for key, value in sorted(layouts_by_read_count.items())
        },
        "expert_byte_length_distribution": {
            str(key): value for key, value in sorted(span_lengths.items())
        },
        "shards_containing_experts": len(shard_expert_counts),
        "experts_per_shard_min": min(shard_expert_counts.values()),
        "experts_per_shard_max": max(shard_expert_counts.values()),
        "contiguous_layer_blocks": contiguous_layer_blocks,
        "lexicographic_expert_order_layer_blocks": lexicographic_layer_blocks,
        "layer_shard_examples": {
            str(layer): layer_shards[layer] for layer in (1, 46, 92)
        },
    }


def reference_check(
    store: LocalSafetensorsStore, layers: list[int], experts: list[int]
) -> dict[str, Any]:
    """Compare dtype, shape, raw bytes, and values to safetensors.safe_open."""
    comparisons = []
    total_bytes = 0
    for layer in layers:
        for expert in experts:
            direct = store.read_expert(layer, expert)
            layout = store.expert_layout(layer, expert)
            handles = {}
            try:
                for shard in layout.shard_names:
                    handles[shard] = safe_open(
                        str(store.model_dir / shard), framework="numpy", device="cpu"
                    )
                for span in layout.tensors:
                    suffix = store._expert_suffix(span.name)
                    actual = direct[suffix]
                    reference = handles[span.shard].get_tensor(span.name)
                    if actual.dtype != reference.dtype:
                        raise AssertionError(
                            f"L{layer} E{expert} {suffix}: dtype "
                            f"{actual.dtype} != {reference.dtype}"
                        )
                    if actual.shape != reference.shape:
                        raise AssertionError(
                            f"L{layer} E{expert} {suffix}: shape "
                            f"{actual.shape} != {reference.shape}"
                        )
                    if actual.tobytes() != reference.tobytes():
                        raise AssertionError(
                            f"L{layer} E{expert} {suffix}: raw bytes differ"
                        )
                    if not np.array_equal(actual, reference):
                        raise AssertionError(
                            f"L{layer} E{expert} {suffix}: values differ"
                        )
                    total_bytes += actual.nbytes
                    comparisons.append(
                        {
                            "layer": layer,
                            "expert": expert,
                            "tensor": suffix,
                            "dtype": str(actual.dtype),
                            "shape": list(actual.shape),
                            "bytes": actual.nbytes,
                            "raw_equal": True,
                            "values_equal": True,
                        }
                    )
            finally:
                handles.clear()
    return {
        "status": "passed",
        "layers": layers,
        "experts": experts,
        "expert_count": len(layers) * len(experts),
        "tensor_comparisons": len(comparisons),
        "bytes_compared": total_bytes,
        "comparisons": comparisons,
    }


def error_check(store: LocalSafetensorsStore) -> dict[str, str]:
    checks = {}
    cases = (
        ("invalid_layer", lambda: store.expert_layout(0, 0), UnknownExpertError),
        ("invalid_expert", lambda: store.expert_layout(1, 896), UnknownExpertError),
        ("invalid_tensor", lambda: store.tensor_span("not.a.tensor"), UnknownTensorError),
        (
            "invalid_expert_tensor",
            lambda: store.expert_tensor_span(1, 0, "w9.weight"),
            UnknownTensorError,
        ),
    )
    for name, call, expected in cases:
        try:
            call()
        except expected as exc:
            checks[name] = str(exc)
        else:
            raise AssertionError(f"{name} did not raise {expected.__name__}")
    return checks


def consume_expert(store: LocalSafetensorsStore, layer: int, expert: int) -> int:
    arrays = store.read_expert(layer, expert)
    # The complete pread already forces the data transfer.  Touching both ends
    # also makes accidental lazy/mmap implementations visible to the benchmark.
    return sum(int(array.flat[0]) + int(array.flat[-1]) for array in arrays.values())


def benchmark(
    buffered_store: LocalSafetensorsStore,
    physical_store: LocalSafetensorsStore,
    *,
    layer: int,
    workers: int,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    expert_ids = sorted(rng.sample(list(buffered_store.expert_ids(layer)), 16))
    sample = expert_ids[len(expert_ids) // 2]
    bytes_one = buffered_store.expert_layout(layer, sample).byte_length
    bytes_sixteen = sum(
        buffered_store.expert_layout(layer, expert).byte_length for expert in expert_ids
    )

    # First/second same-expert reads are an observational page-cache comparison.
    # No privileged cache purge is attempted or claimed.
    _, first_s = timed(lambda: consume_expert(buffered_store, layer, sample))
    _, second_s = timed(lambda: consume_expert(buffered_store, layer, sample))

    repeated_s = []
    checksum = 0
    for _ in range(repeats):
        value, elapsed = timed(lambda: consume_expert(buffered_store, layer, sample))
        checksum ^= value
        repeated_s.append(elapsed)

    def sequential():
        value = 0
        for expert in expert_ids:
            value ^= consume_expert(physical_store, layer, expert)
        return value

    checksum_seq, sequential_s = timed(sequential)
    parallel_result, parallel_s = timed(
        lambda: physical_store.read_experts(layer, expert_ids, workers=workers)
    )
    checksum_parallel = 0
    for tensors in parallel_result.values():
        value = sum(
            int(array.flat[0]) + int(array.flat[-1]) for array in tensors.values()
        )
        checksum_parallel ^= value
    del parallel_result
    if checksum_parallel != checksum_seq:
        raise AssertionError(
            "sequential and parallel expert-read checksums differ: "
            f"{checksum_seq} != {checksum_parallel}"
        )

    return {
        "layer": layer,
        "expert_ids": expert_ids,
        "workers": workers,
        "io_policy": {
            "single_and_repeated": "buffered os.pread",
            "sixteen_sequential_and_parallel": (
                "os.pread with Darwin F_NOCACHE on each shard fd"
            ),
        },
        "expert_bytes": bytes_one,
        "single_expert": {
            "first_seconds": first_s,
            "first_gb_s": bytes_one / first_s / 1e9,
            "second_seconds": second_s,
            "second_gb_s": bytes_one / second_s / 1e9,
        },
        "same_expert_repeated": {
            "repetitions": repeats,
            "seconds": repeated_s,
            "summary_seconds": quantiles(repeated_s),
            "median_gb_s": bytes_one / statistics.median(repeated_s) / 1e9,
        },
        "sixteen_sequential": {
            "seconds": sequential_s,
            "bytes": bytes_sixteen,
            "gb_s": bytes_sixteen / sequential_s / 1e9,
        },
        "sixteen_parallel": {
            "seconds": parallel_s,
            "bytes": bytes_sixteen,
            "gb_s": bytes_sixteen / parallel_s / 1e9,
        },
        "page_cache_observation": {
            "cache_drop_performed": False,
            "first_over_second_ratio": first_s / second_s,
            "note": (
                "First and immediate second buffered pread of the same span. "
                "The first read is only a cold candidate; macOS cache state was not reset."
            ),
        },
        "checksums": {
            "repeated": checksum,
            "sequential": checksum_seq,
            "parallel": checksum_parallel,
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
    parser.add_argument(
        "--sidecar",
        type=pathlib.Path,
        default=ROOT / "bench-results/k3-direct-shards-index.json.gz",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=ROOT / "bench-results/m3ultra512-direct-shards.json",
    )
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 46, 92])
    parser.add_argument("--experts", type=int, nargs="+", default=[0, 448, 895])
    # Keep the default benchmark shard disjoint from the reference-check shards
    # so the correctness oracle does not warm the measured payload first.
    parser.add_argument("--benchmark-layer", type=int, default=47)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--skip-reference", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    before_fds = fd_count()
    print(f"[1/6] parsing index and {EXPECTED_SHARDS} shard headers", flush=True)
    store = LocalSafetensorsStore(args.model_dir)
    raw_report = store.report()
    print(
        f"      {raw_report.tensors:,} tensors, "
        f"{raw_report.routed_experts:,} experts in "
        f"{raw_report.parse_seconds:.3f}s",
        flush=True,
    )

    evidence: dict[str, Any] = {
        "schema": "deltafin.k3-direct-shards-validation.v1",
        "created_at": utc_now(),
        "command": [sys.executable, *sys.argv],
        "model_dir": str(args.model_dir.resolve()),
        "model_files_modified": False,
        "model_conversion_performed": False,
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version,
            "git_branch": git_output("branch", "--show-current"),
            "git_commit": git_output("rev-parse", "HEAD"),
        },
        "header_inventory": dataclasses.asdict(raw_report),
    }

    try:
        print("[2/6] checking all routed-expert layouts", flush=True)
        structure, structure_s = timed(lambda: structure_check(store))
        structure["seconds"] = structure_s
        evidence["structure_validation"] = structure
        print(
            f"      {structure['checked_experts']:,} experts passed; "
            f"{structure['contiguous_experts']:,} single-span",
            flush=True,
        )

        print("[3/6] checking invalid requests", flush=True)
        evidence["error_validation"] = error_check(store)

        print("[4/6] comparing with standard safetensors", flush=True)
        if args.skip_reference:
            evidence["reference_validation"] = {"status": "skipped"}
        else:
            reference, reference_s = timed(
                lambda: reference_check(store, args.layers, args.experts)
            )
            reference["seconds"] = reference_s
            evidence["reference_validation"] = reference
            print(
                f"      {reference['tensor_comparisons']} tensors / "
                f"{reference['bytes_compared'] / 1e6:.1f} MB bit-exact",
                flush=True,
            )

        print("[5/6] generating and reloading sidecar", flush=True)
        sidecar_s = store.write_sidecar(args.sidecar)
        sidecar_bytes = args.sidecar.stat().st_size
        restored, sidecar_load_s = timed(
            lambda: LocalSafetensorsStore(args.model_dir, sidecar=args.sidecar)
        )
        try:
            if restored.tensor_count != store.tensor_count:
                raise AssertionError("sidecar tensor count differs")
            if restored.routed_expert_count != store.routed_expert_count:
                raise AssertionError("sidecar expert count differs")
            sample_name = store.expert_layout(1, 0).tensors[0].name
            if restored.tensor_span(sample_name) != store.tensor_span(sample_name):
                raise AssertionError("sidecar sample tensor differs")
        finally:
            restored.close()
        evidence["sidecar"] = {
            "path": str(args.sidecar.resolve()),
            "bytes": sidecar_bytes,
            "generation_seconds": sidecar_s,
            "load_seconds": sidecar_load_s,
            "round_trip": "passed",
        }
        print(
            f"      {sidecar_bytes / 1e6:.1f} MB in {sidecar_s:.3f}s; "
            f"reload {sidecar_load_s:.3f}s",
            flush=True,
        )

        print("[6/6] benchmarking direct expert pread", flush=True)
        if args.skip_benchmark:
            evidence["benchmark"] = {"status": "skipped"}
        else:
            physical_store = LocalSafetensorsStore(
                args.model_dir,
                sidecar=args.sidecar,
                darwin_nocache=True,
            )
            try:
                evidence["benchmark"] = benchmark(
                    store,
                    physical_store,
                    layer=args.benchmark_layer,
                    workers=args.workers,
                    repeats=args.repeats,
                    seed=args.seed,
                )
            finally:
                physical_store.close()
            measured = evidence["benchmark"]
            print(
                f"      single {measured['single_expert']['second_seconds'] * 1e3:.2f}ms; "
                f"16 sequential {measured['sixteen_sequential']['seconds']:.3f}s "
                f"({measured['sixteen_sequential']['gb_s']:.2f} GB/s); "
                f"parallel {measured['sixteen_parallel']['seconds']:.3f}s "
                f"({measured['sixteen_parallel']['gb_s']:.2f} GB/s)",
                flush=True,
            )
    finally:
        store.close()

    gc.collect()
    after_fds = fd_count()
    fd_ok = after_fds is None or before_fds is None or after_fds <= before_fds
    evidence["fd_validation"] = {
        "before": before_fds,
        "after": after_fds,
        "store_open_after_close": store.open_fd_count,
        "status": "passed" if fd_ok and store.open_fd_count == 0 else "failed",
    }
    if evidence["fd_validation"]["status"] != "passed":
        raise AssertionError(f"file descriptor leak: {evidence['fd_validation']}")

    evidence["status"] = "passed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + f".tmp{os.getpid()}")
    temporary.write_text(
        json.dumps(evidence, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )
    os.replace(temporary, args.output)
    print(f"PASS: evidence saved to {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
