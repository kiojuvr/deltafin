"""Adapter from LocalSafetensorsStore to Deltafin's expert-loader contract.

This is intentionally thin: the independent store owns inventory validation,
coalescing, positional I/O, and descriptor lifetime.  The adapter only changes
the six named tensors into the ``w -> (packed, scale)`` shape consumed by the
existing CPU and Metal MoE paths.
"""

from __future__ import annotations

import atexit
import os
import pathlib
import sys
import threading
import time

import numpy as np

from local_safetensors import LocalSafetensorsStore


ROOT = pathlib.Path(__file__).resolve().parent.parent
_MODEL_DIR_TEXT = os.environ.get("K3_MODEL_DIR")
MODEL_DIR = pathlib.Path(_MODEL_DIR_TEXT) if _MODEL_DIR_TEXT else None
SIDECAR = os.environ.get("K3_DIRECT_SHARDS_SIDECAR")
WORKERS = int(os.environ.get("K3_PREAD_WORKERS", "8"))
DARWIN_NOCACHE = os.environ.get(
    "K3_PREAD_NOCACHE", "1" if sys.platform == "darwin" else "0"
) == "1"

stats = {
    "expert_http": 0,
    "expert_disk": 0,
    "http_bytes": 0,
    "http_s": 0.0,
    "pread_experts": 0,
    "pread_bytes": 0,
    "pread_s": 0.0,
}
_stats_lock = threading.Lock()
_store_lock = threading.Lock()
_store: LocalSafetensorsStore | None = None


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
    global _store
    with _store_lock:
        local_store, _store = _store, None
    if local_store is not None:
        local_store.close()


atexit.register(close)
