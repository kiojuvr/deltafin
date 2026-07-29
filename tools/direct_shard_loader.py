"""Adapter from LocalSafetensorsStore to Deltafin's expert-loader contract.

This is intentionally thin: the independent store owns inventory validation,
coalescing, positional I/O, and descriptor lifetime.  The adapter only changes
the six named tensors into the ``w -> (packed, scale)`` shape consumed by the
existing CPU and Metal MoE paths.
"""

from __future__ import annotations

import atexit
import concurrent.futures
import contextlib
import os
import pathlib
import queue
import threading
import time

import numpy as np

from local_safetensors import LocalSafetensorsStore


ROOT = pathlib.Path(__file__).resolve().parent.parent
_MODEL_DIR_TEXT = os.environ.get("K3_MODEL_DIR")
MODEL_DIR = pathlib.Path(_MODEL_DIR_TEXT) if _MODEL_DIR_TEXT else None
SIDECAR = os.environ.get("K3_DIRECT_SHARDS_SIDECAR")
WORKERS = int(os.environ.get("K3_PREAD_WORKERS", "8"))
DIRECT_SLAB = os.environ.get("K3_DIRECT_SLAB", "0") == "1"
DIRECT_OVERLAP = os.environ.get("K3_DIRECT_OVERLAP", "0") == "1"
DARWIN_NOCACHE = os.environ.get(
    "K3_PREAD_NOCACHE", "0"
) == "1"

stats = {
    "expert_http": 0,
    "expert_disk": 0,
    "http_bytes": 0,
    "http_s": 0.0,
    "pread_experts": 0,
    "pread_bytes": 0,
    "pread_s": 0.0,
    "slab_loads": 0,
    "slab_bytes": 0,
    "slab_s": 0.0,
    "overlap_prefetches": 0,
    "overlap_prefetch_skipped": 0,
    "overlap_prefetch_experts": 0,
    "overlap_prefetch_bytes": 0,
    "overlap_prefetch_s": 0.0,
    "overlap_wait_s": 0.0,
    "overlap_layers": 0,
    "overlap_hits": 0,
    "overlap_misses": 0,
    "overlap_miss_s": 0.0,
}
_stats_lock = threading.Lock()
_store_lock = threading.Lock()
_store: LocalSafetensorsStore | None = None
_slab_lock = threading.Lock()
_slabs = None
_slab_available: queue.Queue[int] | None = None
_slab_executor: concurrent.futures.ThreadPoolExecutor | None = None
_slab_pending = {}


def store() -> LocalSafetensorsStore:
    global _store
    if _store is not None:
        return _store
    with _store_lock:
        if _store is None:
            if MODEL_DIR is None:
                raise RuntimeError(
                    "K3_EXPERT_SOURCE=direct-shards requires K3_MODEL_DIR"
                )
            sidecar = pathlib.Path(SIDECAR) if SIDECAR else None
            _store = LocalSafetensorsStore(
                MODEL_DIR,
                sidecar=sidecar,
                darwin_nocache=DARWIN_NOCACHE,
            )
    return _store


def _compatible(
    tensors: dict[int, object],
) -> dict[int, dict[str, tuple[np.ndarray, np.ndarray]]]:
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


def slab_enabled() -> bool:
    return DIRECT_SLAB


def overlap_enabled() -> bool:
    return DIRECT_SLAB and DIRECT_OVERLAP


def _slab_pool():
    global _slabs, _slab_available, _slab_executor
    if _slabs is not None:
        return _slabs, _slab_available
    with _slab_lock:
        if _slabs is None:
            from expert_slab import DoubleExpertSlab

            _slabs = DoubleExpertSlab(store())
            _slab_available = queue.Queue(maxsize=2)
            _slab_available.put(0)
            _slab_available.put(1)
            _slab_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="k3-direct-overlap"
            )
    return _slabs, _slab_available


def _read_bytes(layer: int, ids) -> int:
    return sum(
        store().expert_layout(layer, expert).byte_length
        for expert in ids
    )


def _record_slab_read(
    layer: int, ids, elapsed: float, *, prefetch: bool
) -> None:
    ids = tuple(ids)
    if not ids:
        return
    byte_count = _read_bytes(layer, ids)
    with _stats_lock:
        stats["expert_disk"] += len(ids)
        stats["pread_experts"] += len(ids)
        stats["pread_bytes"] += byte_count
        stats["pread_s"] += elapsed
        stats["slab_loads"] += 1
        stats["slab_bytes"] += byte_count
        stats["slab_s"] += elapsed
        if prefetch:
            stats["overlap_prefetches"] += 1
            stats["overlap_prefetch_experts"] += len(ids)
            stats["overlap_prefetch_bytes"] += byte_count
            stats["overlap_prefetch_s"] += elapsed


def begin_slab_pass() -> None:
    """Start a token pass with no stale speculative bank ownership."""
    settle_slab_prefetches()


def prefetch_slab(layer: int, eids, workers: int | None = None) -> bool:
    """Fill one free bank asynchronously for a predicted next-layer route."""
    ids = tuple(int(expert) for expert in eids)
    if not overlap_enabled() or not ids:
        return False
    if len(ids) > 16:
        raise ValueError(f"direct slab holds 16 experts, got {len(ids)}")
    slabs, available = _slab_pool()
    with _slab_lock:
        if layer in _slab_pending:
            return False
        try:
            bank_index = available.get_nowait()
        except queue.Empty:
            with _stats_lock:
                stats["overlap_prefetch_skipped"] += 1
            return False

        def fill():
            started = time.perf_counter()
            raw = slabs.banks[bank_index].load(
                layer, ids, workers=workers or WORKERS
            )
            elapsed = time.perf_counter() - started
            _record_slab_read(layer, ids, elapsed, prefetch=True)
            return raw

        try:
            future = _slab_executor.submit(fill)
        except BaseException:
            available.put(bank_index)
            raise
        _slab_pending[layer] = {
            "bank_index": bank_index,
            "ids": ids,
            "future": future,
        }
    return True


def settle_slab_prefetches(*, raise_errors: bool = True) -> None:
    """Wait out and release any unconsumed speculative banks."""
    with _slab_lock:
        pending = list(_slab_pending.values())
        _slab_pending.clear()
        available = _slab_available
    failure = None
    for ticket in pending:
        try:
            ticket["future"].result()
        except BaseException as exc:
            if failure is None:
                failure = exc
        finally:
            if available is not None:
                available.put(ticket["bank_index"])
    if failure is not None and raise_errors:
        raise failure


@contextlib.contextmanager
def slab_experts(layer: int, eids, workers: int | None = None):
    """Lease one fixed slab bank through the caller's synchronous MoE compute."""
    ids = tuple(int(expert) for expert in eids)
    if not DIRECT_SLAB:
        raise RuntimeError("K3_DIRECT_SLAB is disabled")
    if len(ids) > 16:
        raise ValueError(f"direct slab holds 16 experts, got {len(ids)}")
    slabs, available = _slab_pool()
    with _slab_lock:
        ticket = _slab_pending.pop(layer, None)
    bank_index = (
        ticket["bank_index"] if ticket is not None else available.get()
    )
    raw = None
    try:
        bank = slabs.banks[bank_index]
        if ticket is None:
            started = time.perf_counter()
            raw = bank.load(layer, ids, workers=workers or WORKERS)
            elapsed = time.perf_counter() - started
            _record_slab_read(layer, ids, elapsed, prefetch=False)
        else:
            waited = time.perf_counter()
            ticket["future"].result()
            wait_seconds = time.perf_counter() - waited
            miss_started = time.perf_counter()
            raw = bank.load_reusing(
                layer, ids, workers=workers or WORKERS
            )
            miss_seconds = time.perf_counter() - miss_started
            misses = bank.last_read_ids
            _record_slab_read(
                layer, misses, miss_seconds, prefetch=False
            )
            with _stats_lock:
                stats["overlap_wait_s"] += wait_seconds
                stats["overlap_layers"] += 1
                stats["overlap_hits"] += len(ids) - len(misses)
                stats["overlap_misses"] += len(misses)
                stats["overlap_miss_s"] += miss_seconds
        yield raw
    finally:
        raw = None
        available.put(bank_index)


def fetch_expert_raw(
    layer: int, expert: int
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    local_store = store()
    started = time.perf_counter()
    layout = local_store.expert_layout(layer, expert)
    result = dict(local_store.read_expert_raw(layer, expert))
    elapsed = time.perf_counter() - started
    with _stats_lock:
        stats["expert_disk"] += 1
        stats["pread_experts"] += 1
        stats["pread_bytes"] += layout.byte_length
        stats["pread_s"] += elapsed
    return result


def fetch_experts(
    layer: int,
    eids,
    workers: int | None = None,
    dequant: bool = True,
    **_ignored,
):
    """Signature-compatible replacement for ``k3loader.fetch_experts``."""
    ids = list(eids)
    local_store = store()
    started = time.perf_counter()
    tensors = local_store.read_experts(layer, ids, workers=workers or WORKERS)
    raw = _compatible(tensors)
    elapsed = time.perf_counter() - started
    byte_count = sum(local_store.expert_layout(layer, expert).byte_length for expert in ids)
    with _stats_lock:
        stats["expert_disk"] += len(ids)
        stats["pread_experts"] += len(ids)
        stats["pread_bytes"] += byte_count
        stats["pread_s"] += elapsed
    if not dequant:
        return raw

    import torch
    from mxfp4 import dequant_mxfp4

    return {
        expert: {
            weight: torch.from_numpy(dequant_mxfp4(packed, scale))
            for weight, (packed, scale) in weights.items()
        }
        for expert, weights in raw.items()
    }


def close() -> None:
    global _store, _slabs, _slab_available, _slab_executor
    settle_slab_prefetches(raise_errors=False)
    with _slab_lock:
        slabs, _slabs = _slabs, None
        _slab_available = None
        executor, _slab_executor = _slab_executor, None
    if executor is not None:
        executor.shutdown(wait=True)
    if slabs is not None:
        slabs.close()
    with _store_lock:
        local_store, _store = _store, None
    if local_store is not None:
        local_store.close()


atexit.register(close)
