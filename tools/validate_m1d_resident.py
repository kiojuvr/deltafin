#!/usr/bin/env python3
"""M1-D: stage all official K3 resident tensors on MPS and audit lifetime."""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import gc
import json
import math
import os
import pathlib
import platform
import re
import subprocess
import sys
import time
from typing import Any

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from expert_slab import DoubleExpertSlab  # noqa: E402
from local_safetensors import LocalSafetensorsStore  # noqa: E402
from resident_shard_loader import (  # noqa: E402
    DirectResidentLoader,
    ResidentTensorBank,
)
from spine_cache import _darwin_vm_snapshot  # noqa: E402
from validate_m1_layer import (  # noqa: E402
    compare_stages,
    direct_experts,
    load_official_modules,
    new_meta_layer,
    reference_experts,
    reference_materialize,
    run_official_layer,
)


ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_ROUTE = (
    712, 461, 308, 828, 846, 348, 154, 617,
    521, 511, 753, 133, 503, 675, 692, 715,
)
PAGE_SIZE = int(os.sysconf("SC_PAGE_SIZE"))


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def command_output(argv: list[str], *, timeout: int = 120) -> str | None:
    try:
        result = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def parse_size(text: str) -> int:
    match = re.fullmatch(r"([0-9.]+)([KMGTP]?)", text.strip())
    if match is None:
        raise ValueError(f"cannot parse size {text!r}")
    scales = {
        "": 1,
        "K": 1024,
        "M": 1024**2,
        "G": 1024**3,
        "T": 1024**4,
        "P": 1024**5,
    }
    return int(float(match[1]) * scales[match[2]])


def swap_snapshot() -> dict[str, Any]:
    raw = command_output(["sysctl", "-n", "vm.swapusage"])
    result: dict[str, Any] = {"raw": raw}
    if raw:
        for key in ("total", "used", "free"):
            match = re.search(rf"{key}\s*=\s*([0-9.]+[KMGTP]?)", raw)
            if match:
                result[f"{key}_bytes"] = parse_size(match[1])
    return result


def process_snapshot() -> dict[str, Any]:
    raw = command_output(
        [
            "ps",
            "-o",
            "pid=,rss=,vsz=,command=",
            "-p",
            str(os.getpid()),
        ]
    )
    result: dict[str, Any] = {"raw": raw}
    if raw:
        parts = raw.split(maxsplit=3)
        if len(parts) >= 3:
            result.update(
                {
                    "pid": int(parts[0]),
                    "rss_bytes": int(parts[1]) * 1024,
                    "vsz_bytes": int(parts[2]) * 1024,
                }
            )
    return result


def disk_snapshot() -> dict[str, Any]:
    raw = command_output(["iostat", "-Id", "disk8", "disk9"])
    result: dict[str, Any] = {"raw": raw, "members": ["disk8", "disk9"]}
    if not raw:
        return result
    lines = [line for line in raw.splitlines() if line.strip()]
    if len(lines) < 3:
        return result
    fields = lines[-1].split()
    if len(fields) != 6:
        return result
    result["disk8"] = {
        "kb_per_transfer": float(fields[0]),
        "transfers": int(fields[1]),
        "megabytes": float(fields[2]),
    }
    result["disk9"] = {
        "kb_per_transfer": float(fields[3]),
        "transfers": int(fields[4]),
        "megabytes": float(fields[5]),
    }
    result["member_megabytes_total"] = (
        result["disk8"]["megabytes"] + result["disk9"]["megabytes"]
    )
    return result


def memory_snapshot(label: str, *, detailed: bool = False) -> dict[str, Any]:
    torch.mps.synchronize()
    vm_pages = _darwin_vm_snapshot() or {}
    vm_selected = {}
    for key in (
        "free",
        "active",
        "inactive",
        "wire",
        "compressor_pages",
        "external",
        "internal",
        "pageins",
        "pageouts",
        "compressions",
        "decompressions",
        "swapins",
        "swapouts",
    ):
        if key in vm_pages:
            value = int(vm_pages[key])
            vm_selected[key] = value
            if key in {
                "free",
                "active",
                "inactive",
                "wire",
                "compressor_pages",
                "external",
                "internal",
            }:
                vm_selected[f"{key}_bytes"] = value * PAGE_SIZE
    snapshot: dict[str, Any] = {
        "label": label,
        "captured_at": now(),
        "monotonic_seconds": time.monotonic(),
        "mps": {
            "current_allocated": int(
                torch.mps.current_allocated_memory()
            ),
            "driver_allocated": int(torch.mps.driver_allocated_memory()),
            "recommended_max": int(torch.mps.recommended_max_memory()),
        },
        "process": process_snapshot(),
        "swap": swap_snapshot(),
        "vm": vm_selected,
        "disk": disk_snapshot(),
    }
    if detailed:
        snapshot["vm_stat_raw"] = command_output(["vm_stat"])
        snapshot["vmmap_summary"] = command_output(
            ["vmmap", "-summary", str(os.getpid())]
        )
    return snapshot


def disk_delta_bytes(before: dict[str, Any], after: dict[str, Any]) -> int | None:
    try:
        delta_mb = (
            after["disk"]["member_megabytes_total"]
            - before["disk"]["member_megabytes_total"]
        )
    except (KeyError, TypeError):
        return None
    return max(0, round(delta_mb * 1024 * 1024))


def write_evidence(path: pathlib.Path, evidence: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp{os.getpid()}")
    temporary.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n"
    )
    os.replace(temporary, path)


def model_fingerprint(store: LocalSafetensorsStore) -> dict[str, Any]:
    files = [store.index_path] + [
        store.model_dir / shard for shard in sorted(store._shard_info)
    ]
    rows = []
    for path in files:
        stat = path.stat()
        rows.append(
            {
                "name": path.name,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return {"files": len(rows), "rows": rows}


def validate_layer_from_bank(
    store: LocalSafetensorsStore,
    bank: ResidentTensorBank,
    modeling,
    config,
    layer_index: int,
) -> dict[str, Any]:
    device = torch.device("mps")
    hidden = torch.linspace(
        -0.5, 0.5, config.hidden_size, dtype=torch.float32, device=device
    ).view(1, 1, config.hidden_size)
    block_residual = torch.linspace(
        0.25,
        -0.25,
        config.hidden_size,
        dtype=torch.float32,
        device=device,
    ).view(1, 1, config.hidden_size)
    prefix = f"language_model.model.layers.{layer_index}."

    baseline = int(torch.mps.current_allocated_memory())
    reference_layer = new_meta_layer(modeling, config, layer_index)
    reference_report = reference_materialize(
        store,
        reference_layer,
        prefix,
        device=device,
        dtype=torch.float32,
    )
    reference_stages, reference_seconds, reference_timing = run_official_layer(
        modeling,
        reference_layer,
        hidden,
        block_residual,
        lambda layer, ids: reference_experts(store, layer, ids),
    )
    del reference_layer
    gc.collect()
    torch.mps.empty_cache()
    torch.mps.synchronize()
    after_reference_release = int(torch.mps.current_allocated_memory())

    resident_layer = new_meta_layer(modeling, config, layer_index)
    bound = bank.bind_module(resident_layer, prefix)
    resident_stages, resident_seconds, resident_timing = run_official_layer(
        modeling,
        resident_layer,
        hidden,
        block_residual,
        lambda layer, ids: direct_experts(store, layer, ids),
    )
    parity = compare_stages(reference_stages, resident_stages)
    for parameter_name, parameter in resident_layer.named_parameters():
        if ".experts." in parameter_name:
            continue
        full_name = prefix + parameter_name
        if parameter.data_ptr() != bank.tensor(full_name).data_ptr():
            raise AssertionError(f"{full_name}: resident layer copied bank storage")
    result = {
        "status": "passed",
        "layer": layer_index,
        "bound_tensors": bound,
        "reference_materialization": dataclasses.asdict(reference_report),
        "reference_seconds": reference_seconds,
        "resident_seconds": resident_seconds,
        "reference_stage_timing": reference_timing,
        "resident_stage_timing": resident_timing,
        "topk_ids": resident_stages["topk_ids"].tolist(),
        "stage_parity": parity,
        "all_stages_exact": all(
            stage.get("exact", False) for stage in parity.values()
        ),
        "mps_before_reference": baseline,
        "mps_after_reference_release": after_reference_release,
        "bank_storage_aliased": True,
    }
    del resident_layer, hidden, block_residual
    gc.collect()
    torch.mps.empty_cache()
    torch.mps.synchronize()
    return result


def stability_result(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        return {"status": "not-run"}
    start, end = samples[0], samples[-1]
    used_start = start["swap"].get("used_bytes")
    used_end = end["swap"].get("used_bytes")
    compressor_start = start["vm"].get("compressor_pages_bytes")
    compressor_end = end["vm"].get("compressor_pages_bytes")
    current_values = [
        sample["mps"]["current_allocated"] for sample in samples
    ]
    result = {
        "samples": len(samples),
        "duration_seconds": (
            end["monotonic_seconds"] - start["monotonic_seconds"]
        ),
        "mps_current_min": min(current_values),
        "mps_current_max": max(current_values),
        "mps_current_growth": current_values[-1] - current_values[0],
    }
    if used_start is not None and used_end is not None:
        result["swap_used_growth_bytes"] = used_end - used_start
    if compressor_start is not None and compressor_end is not None:
        result["compressor_growth_bytes"] = (
            compressor_end - compressor_start
        )
    tail = samples[-min(6, len(samples)):]
    tail_swap_start = tail[0]["swap"].get("used_bytes")
    tail_swap_end = tail[-1]["swap"].get("used_bytes")
    tail_compressor_start = tail[0]["vm"].get(
        "compressor_pages_bytes"
    )
    tail_compressor_end = tail[-1]["vm"].get(
        "compressor_pages_bytes"
    )
    if tail_swap_start is not None and tail_swap_end is not None:
        result["tail_swap_growth_bytes"] = (
            tail_swap_end - tail_swap_start
        )
    if (
        tail_compressor_start is not None
        and tail_compressor_end is not None
    ):
        result["tail_compressor_growth_bytes"] = (
            tail_compressor_end - tail_compressor_start
        )
    # A fixed tensor bank must not change MPS ownership. The final five-minute
    # window must also show that swap/compressor growth has settled. Small
    # unrelated system traffic is tolerated but retained in the evidence.
    tail_swap_ok = result.get("tail_swap_growth_bytes", 0) <= 64 * 1024 * 1024
    tail_compressor_ok = (
        result.get("tail_compressor_growth_bytes", 0)
        <= 256 * 1024 * 1024
    )
    result["status"] = (
        "passed"
        if (
            result["mps_current_max"] - result["mps_current_min"]
            <= 64 * 1024 * 1024
            and tail_swap_ok
            and tail_compressor_ok
        )
        else "failed"
    )
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
    parser.add_argument(
        "--sidecar",
        type=pathlib.Path,
        default=ROOT / "bench-results/k3-direct-shards-index.json.gz",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=ROOT / "bench-results/m1d-resident-mps.json",
    )
    parser.add_argument("--layers-per-stage", type=int, default=8)
    parser.add_argument(
        "--parity-layers", default="1,46,92",
        help="comma-separated routed layers",
    )
    parser.add_argument("--stability-seconds", type=int, default=1800)
    parser.add_argument("--sample-seconds", type=int, default=60)
    parser.add_argument(
        "--stop-after-stage",
        type=int,
        help="smoke-test only: stop after this many stages",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.layers_per_stage <= 0:
        raise ValueError("--layers-per-stage must be positive")
    if args.stability_seconds < 0 or args.sample_seconds <= 0:
        raise ValueError("stability/sample seconds are invalid")
    if not torch.backends.mps.is_available():
        raise RuntimeError("M1-D requires a visible MPS device")
    high = os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO")
    low = os.environ.get("PYTORCH_MPS_LOW_WATERMARK_RATIO")
    if (high, low) != ("1.0", "0.9"):
        raise RuntimeError(
            "set PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0 and "
            "PYTORCH_MPS_LOW_WATERMARK_RATIO=0.9 before importing torch"
        )
    torch.set_grad_enabled(False)
    torch.set_num_threads(8)

    parity_layers = [
        int(value) for value in args.parity_layers.split(",") if value
    ]
    evidence: dict[str, Any] = {
        "schema": "deltafin.m1d-resident-mps.v1",
        "status": "running",
        "created_at": now(),
        "model_dir": str(args.model_dir.resolve()),
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "git_commit": command_output(["git", "rev-parse", "HEAD"]),
            "high_watermark_ratio": high,
            "low_watermark_ratio": low,
            "pread_nocache": os.environ.get("K3_PREAD_NOCACHE"),
        },
        "model_files_modified": False,
        "model_conversion_performed": False,
        "stages": [],
        "parity": {},
        "stability_samples": [],
    }
    bank = None
    slabs = None
    slab_payloads = []
    failure: BaseException | None = None
    source_fingerprint = None
    baseline = memory_snapshot("baseline", detailed=True)
    evidence["baseline"] = baseline
    write_evidence(args.output, evidence)
    try:
        modeling, config = load_official_modules(args.model_dir)
        with LocalSafetensorsStore(
            args.model_dir,
            sidecar=args.sidecar,
            darwin_nocache=False,
        ) as store:
            loader = DirectResidentLoader(store)
            source_fingerprint = model_fingerprint(store)
            evidence["source_before"] = source_fingerprint
            plan = loader.resident_stages(
                layers_per_stage=args.layers_per_stage
            )
            all_names = [
                name for _label, names in plan for name in names
            ]
            spans = [store.tensor_span(name) for name in all_names]
            expected_checkpoint = sum(span.byte_length for span in spans)
            expected_materialized = sum(
                math.prod(span.shape) * 4 for span in spans
            )
            evidence["inventory"] = {
                "tensors": len(all_names),
                "checkpoint_bytes": expected_checkpoint,
                "float32_materialized_bytes": expected_materialized,
                "stage_count": len(plan),
                "layers_per_stage": args.layers_per_stage,
            }
            bank = ResidentTensorBank(
                loader, device="mps", dtype=torch.float32
            )
            for stage_index, (label, names) in enumerate(plan, start=1):
                if (
                    args.stop_after_stage is not None
                    and stage_index > args.stop_after_stage
                ):
                    break
                before = memory_snapshot(f"{label}-before")
                report = bank.load_names(names)
                torch.mps.synchronize()
                after = memory_snapshot(f"{label}-after")
                physical_bytes = disk_delta_bytes(before, after)
                stage = {
                    "index": stage_index,
                    "label": label,
                    "load": dataclasses.asdict(report),
                    "bank_tensors": len(bank),
                    "bank_checkpoint_bytes": bank.checkpoint_bytes,
                    "bank_materialized_bytes": bank.materialized_bytes,
                    "before": before,
                    "after": after,
                    "physical_member_read_bytes": physical_bytes,
                }
                if physical_bytes is not None:
                    stage["inferred_page_cache_bytes"] = max(
                        0, report.checkpoint_bytes - physical_bytes
                    )
                    stage["physical_gb_s"] = (
                        physical_bytes / report.total_seconds / 1e9
                    )
                evidence["stages"].append(stage)
                print(
                    f"[load {stage_index:02d}/{len(plan):02d}] {label}: "
                    f"{len(bank)}/{len(all_names)} tensors, "
                    f"{bank.materialized_bytes / 1e9:.1f}/"
                    f"{expected_materialized / 1e9:.1f} GB MPS, "
                    f"{report.total_seconds:.2f}s",
                    flush=True,
                )
                write_evidence(args.output, evidence)

            full_run = len(bank) == len(all_names)
            evidence["full_run"] = full_run
            if not full_run:
                evidence["status"] = "smoke-passed"
                continue_run = False
            else:
                continue_run = True
                if bank.checkpoint_bytes != expected_checkpoint:
                    raise AssertionError("resident checkpoint-byte total differs")
                if bank.materialized_bytes != expected_materialized:
                    raise AssertionError("resident materialized-byte total differs")
                loaded = memory_snapshot("all-resident-loaded", detailed=True)
                evidence["all_resident_loaded"] = loaded
                expected_current = (
                    baseline["mps"]["current_allocated"]
                    + expected_materialized
                )
                actual_current = loaded["mps"]["current_allocated"]
                if abs(actual_current - expected_current) > 64 * 1024 * 1024:
                    raise AssertionError(
                        f"MPS current {actual_current} differs from expected "
                        f"{expected_current}"
                    )
                write_evidence(args.output, evidence)

            if continue_run:
                for layer_index in parity_layers:
                    print(f"[parity] layer {layer_index}", flush=True)
                    evidence["parity"][str(layer_index)] = (
                        validate_layer_from_bank(
                            store,
                            bank,
                            modeling,
                            config,
                            layer_index,
                        )
                    )
                    write_evidence(args.output, evidence)

                routes = [
                    evidence["parity"].get("1", {}).get(
                        "topk_ids", [list(DEFAULT_ROUTE)]
                    )[0],
                    evidence["parity"].get("46", {}).get(
                        "topk_ids", [list(DEFAULT_ROUTE)]
                    )[0],
                ]
                slabs = DoubleExpertSlab(store)
                slab_payloads.extend(
                    (
                        slabs.banks[0].load(1, routes[0], workers=8),
                        slabs.banks[1].load(46, routes[1], workers=8),
                    )
                )
                evidence["double_slab"] = {
                    "bytes": slabs.nbytes,
                    "banks": 2,
                    "page_aligned": all(
                        slab.page_aligned for slab in slabs.banks
                    ),
                    "committed_by_direct_reads": True,
                }
                stability_start = memory_snapshot(
                    "stability-0", detailed=True
                )
                evidence["stability_samples"].append(stability_start)
                write_evidence(args.output, evidence)
                deadline = time.monotonic() + args.stability_seconds
                sample_index = 0
                while time.monotonic() < deadline:
                    remaining = deadline - time.monotonic()
                    time.sleep(min(args.sample_seconds, remaining))
                    sample_index += 1
                    sample = memory_snapshot(f"stability-{sample_index}")
                    evidence["stability_samples"].append(sample)
                    print(
                        f"[stability] {sample_index}: "
                        f"MPS {sample['mps']['current_allocated'] / 1e9:.1f} GB, "
                        f"swap {sample['swap'].get('used_bytes', 0) / 1e9:.3f} GB, "
                        f"compressor "
                        f"{sample['vm'].get('compressor_pages_bytes', 0) / 1e9:.3f} GB",
                        flush=True,
                    )
                    write_evidence(args.output, evidence)
                evidence["stability"] = stability_result(
                    evidence["stability_samples"]
                )
                if evidence["stability"]["status"] != "passed":
                    raise AssertionError("resident MPS ownership was not stable")
                evidence["source_after"] = model_fingerprint(store)
                if evidence["source_after"] != source_fingerprint:
                    raise AssertionError("official model file metadata changed")
                evidence["status"] = "resident-validated"
    except BaseException as exc:
        failure = exc
        evidence["status"] = "failed"
        evidence["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
    finally:
        slab_payloads.clear()
        gc.collect()
        if slabs is not None:
            slabs.close()
        slabs = None
        if bank is not None:
            released_tensors, released_bytes = bank.release()
        else:
            released_tensors, released_bytes = 0, 0
        bank = None
        gc.collect()
        torch.mps.empty_cache()
        torch.mps.synchronize()
        released = memory_snapshot("released", detailed=True)
        release_current_delta = (
            released["mps"]["current_allocated"]
            - baseline["mps"]["current_allocated"]
        )
        release_driver_delta = (
            released["mps"]["driver_allocated"]
            - baseline["mps"]["driver_allocated"]
        )
        evidence["release"] = {
            "released_tensors": released_tensors,
            "released_materialized_bytes": released_bytes,
            "snapshot": released,
            "current_delta_from_baseline": release_current_delta,
            "driver_delta_from_baseline": release_driver_delta,
            "current_near_baseline": abs(release_current_delta)
            <= 64 * 1024 * 1024,
            "driver_near_baseline": release_driver_delta
            <= 1024 * 1024 * 1024,
            "empty_cache_called": True,
        }
        if (
            evidence["status"] == "resident-validated"
            and (
                not evidence["release"]["current_near_baseline"]
                or not evidence["release"]["driver_near_baseline"]
            )
        ):
            evidence["status"] = "failed"
            evidence["error"] = {
                "type": "ReleaseValidationError",
                "message": "MPS allocation did not return near baseline",
            }
        if evidence["status"] == "resident-validated":
            evidence["status"] = "passed"
        evidence["completed_at"] = now()
        write_evidence(args.output, evidence)

    if failure is not None:
        raise failure
    if evidence["status"] == "failed":
        raise RuntimeError(evidence["error"]["message"])
    print(f"{evidence['status'].upper()}: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
