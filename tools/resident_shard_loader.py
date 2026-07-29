"""Partial resident-weight loader over unmodified local safetensors shards."""

from __future__ import annotations

import atexit
import gc
import math
import re
import threading
import time
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn

from local_safetensors import LocalSafetensorsStore, TensorSpan


_ROUTED_EXPERT = re.compile(
    r"^language_model\.model\.layers\.\d+\.block_sparse_moe\.experts\.\d+\."
)
_LAYER_TENSOR = re.compile(r"^language_model\.model\.layers\.(\d+)\.")
_TORCH_DTYPES = {
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


@dataclass(frozen=True, slots=True)
class MaterializationReport:
    tensors: int
    checkpoint_bytes: int
    materialized_bytes: int
    read_seconds: float
    transfer_seconds: float
    total_seconds: float
    device: str
    dtype: str


@dataclass(frozen=True, slots=True)
class BankLoadReport:
    tensors: int
    checkpoint_bytes: int
    materialized_bytes: int
    read_transfer_seconds: float
    total_seconds: float
    device: str
    dtype: str


def set_parameter(root: nn.Module, dotted_name: str, tensor: torch.Tensor) -> None:
    target = root
    parts = dotted_name.split(".")
    for part in parts[:-1]:
        target = target[int(part)] if part.isdigit() else getattr(target, part)
    setattr(
        target,
        parts[-1],
        nn.Parameter(tensor, requires_grad=False),
    )


class DirectResidentLoader:
    """Header-derived access to non-routed K3 tensors.

    The loader intentionally reads one tensor at a time.  It never constructs a
    second resident tree or retains a layer-sized Python byte dictionary.
    """

    def __init__(self, store: LocalSafetensorsStore):
        self.store = store

    @staticmethod
    def is_routed_expert(name: str) -> bool:
        return _ROUTED_EXPERT.match(name) is not None

    def resident_names(self) -> tuple[str, ...]:
        return tuple(
            name for name in self.store._tensors
            if not self.is_routed_expert(name)
        )

    def layer_names(self, layer: int) -> tuple[str, ...]:
        prefix = f"language_model.model.layers.{layer}."
        names = tuple(
            name for name in self.store._tensors
            if name.startswith(prefix) and not self.is_routed_expert(name)
        )
        if not names:
            raise KeyError(f"no resident tensors found for layer {layer}")
        return names

    def resident_stages(
        self, *, layers_per_stage: int = 8
    ) -> tuple[tuple[str, tuple[str, ...]], ...]:
        if layers_per_stage <= 0:
            raise ValueError("layers_per_stage must be positive")
        globals_: list[str] = []
        by_layer: dict[int, list[str]] = {}
        for name in self.resident_names():
            match = _LAYER_TENSOR.match(name)
            if match:
                by_layer.setdefault(int(match[1]), []).append(name)
            else:
                globals_.append(name)

        def physical_order(items):
            return tuple(
                sorted(
                    items,
                    key=lambda name: (
                        self.store.tensor_span(name).shard,
                        self.store.tensor_span(name).offset,
                    ),
                )
            )

        stages = [("global", physical_order(globals_))]
        layer_ids = sorted(by_layer)
        for start in range(0, len(layer_ids), layers_per_stage):
            group = layer_ids[start:start + layers_per_stage]
            group_names = [
                name for layer in group for name in by_layer[layer]
            ]
            stages.append(
                (
                    f"layers-{group[0]}-{group[-1]}",
                    physical_order(group_names),
                )
            )
        planned = sum(len(names) for _label, names in stages)
        if planned != len(self.resident_names()):
            raise AssertionError("resident stage plan lost or duplicated tensors")
        return tuple(stages)

    def tensor_span(self, name: str) -> TensorSpan:
        span = self.store.tensor_span(name)
        if self.is_routed_expert(name):
            raise ValueError(
                f"{name!r} is a routed-expert tensor; use the expert reader"
            )
        return span

    def read_bytes(self, name: str) -> bytes:
        self.tensor_span(name)
        return self.store.read_tensor_bytes(name)

    def read_torch(
        self,
        name: str,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        span = self.tensor_span(name)
        try:
            source_dtype = _TORCH_DTYPES[span.dtype]
        except KeyError as exc:
            raise TypeError(f"torch conversion unsupported for {span.dtype}") from exc
        payload = bytearray(span.byte_length)
        self.store.read_tensor_into(name, payload)
        tensor = torch.frombuffer(payload, dtype=source_dtype).reshape(span.shape)
        target_device = torch.device(device)
        if dtype is None:
            dtype = source_dtype
        # copy=True detaches the result from the temporary bytearray even on CPU.
        return tensor.to(device=target_device, dtype=dtype, copy=True)

    def read_rows_torch(
        self,
        name: str,
        row_ids: Iterable[int],
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Positionally read selected rows from a contiguous 2-D tensor."""
        span = self.tensor_span(name)
        if len(span.shape) != 2:
            raise ValueError(f"{name!r} is not 2-D: {span.shape}")
        try:
            source_dtype = _TORCH_DTYPES[span.dtype]
        except KeyError as exc:
            raise TypeError(f"torch conversion unsupported for {span.dtype}") from exc
        ids = [int(row) for row in row_ids]
        rows, columns = span.shape
        if any(row < 0 or row >= rows for row in ids):
            raise IndexError(f"row outside [0,{rows}) for {name!r}: {ids}")
        element_size = torch.empty((), dtype=source_dtype).element_size()
        row_bytes = columns * element_size
        payload = bytearray(len(ids) * row_bytes)
        target = memoryview(payload)
        for index, row in enumerate(ids):
            self.store._read_into(
                span.shard,
                span.offset + row * row_bytes,
                target[index * row_bytes:(index + 1) * row_bytes],
                f"{name}[{row}]",
            )
        tensor = torch.frombuffer(payload, dtype=source_dtype).reshape(
            len(ids), columns
        )
        return tensor.to(
            device=torch.device(device),
            dtype=dtype or source_dtype,
            copy=True,
        )

    def materialize_module(
        self,
        module: nn.Module,
        prefix: str,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> MaterializationReport:
        started = time.perf_counter()
        read_seconds = 0.0
        transfer_seconds = 0.0
        checkpoint_bytes = 0
        materialized_bytes = 0
        count = 0
        for parameter_name, _parameter in list(module.named_parameters()):
            if ".experts." in parameter_name:
                continue
            full_name = prefix + parameter_name
            span = self.tensor_span(full_name)
            read_started = time.perf_counter()
            payload = bytearray(span.byte_length)
            self.store.read_tensor_into(full_name, payload)
            read_seconds += time.perf_counter() - read_started
            source_dtype = _TORCH_DTYPES[span.dtype]
            source = torch.frombuffer(payload, dtype=source_dtype).reshape(span.shape)
            transfer_started = time.perf_counter()
            target = source.to(
                device=torch.device(device), dtype=dtype, copy=True
            )
            transfer_seconds += time.perf_counter() - transfer_started
            set_parameter(module, parameter_name, target)
            checkpoint_bytes += span.byte_length
            materialized_bytes += math.prod(span.shape) * target.element_size()
            count += 1
        return MaterializationReport(
            tensors=count,
            checkpoint_bytes=checkpoint_bytes,
            materialized_bytes=materialized_bytes,
            read_seconds=read_seconds,
            transfer_seconds=transfer_seconds,
            total_seconds=time.perf_counter() - started,
            device=str(torch.device(device)),
            dtype=str(dtype),
        )


class ResidentTensorBank:
    """Own selected resident tensors on one device without a CPU weight tree.

    Each checkpoint tensor is read and transferred independently. Only the
    resulting device tensor is retained; the source bytearray is eligible for
    release before the next tensor is loaded.
    """

    def __init__(
        self,
        loader: DirectResidentLoader,
        *,
        device: torch.device | str = "mps",
        dtype: torch.dtype = torch.float32,
    ):
        self.loader = loader
        self.device = torch.device(device)
        self.dtype = dtype
        self._tensors: dict[str, torch.Tensor] = {}
        self._checkpoint_bytes = 0
        self._materialized_bytes = 0

    def __len__(self) -> int:
        return len(self._tensors)

    def __contains__(self, name: str) -> bool:
        return name in self._tensors

    @property
    def checkpoint_bytes(self) -> int:
        return self._checkpoint_bytes

    @property
    def materialized_bytes(self) -> int:
        return self._materialized_bytes

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._tensors)

    def tensor(self, name: str) -> torch.Tensor:
        try:
            return self._tensors[name]
        except KeyError as exc:
            raise KeyError(f"resident tensor {name!r} is not loaded") from exc

    def load_names(self, names: Iterable[str]) -> BankLoadReport:
        requested = tuple(names)
        if len(requested) != len(set(requested)):
            raise ValueError("resident tensor names must be unique")
        already_loaded = [name for name in requested if name in self._tensors]
        if already_loaded:
            raise ValueError(
                f"resident tensors already loaded: {already_loaded[:3]}"
            )
        spans = [self.loader.tensor_span(name) for name in requested]
        started = time.perf_counter()
        materialized = 0
        checkpoint = 0
        for name, span in zip(requested, spans):
            tensor = self.loader.read_torch(
                name, device=self.device, dtype=self.dtype
            )
            device_matches = (
                tensor.device.type == self.device.type
                and (
                    self.device.index is None
                    or tensor.device.index == self.device.index
                )
            )
            if not device_matches or tensor.dtype != self.dtype:
                raise AssertionError(
                    f"{name}: materialized as {tensor.device}/{tensor.dtype}, "
                    f"expected {self.device}/{self.dtype}"
                )
            self._tensors[name] = tensor
            checkpoint += span.byte_length
            materialized += tensor.numel() * tensor.element_size()
            self._checkpoint_bytes += span.byte_length
            self._materialized_bytes += (
                tensor.numel() * tensor.element_size()
            )
        elapsed = time.perf_counter() - started
        return BankLoadReport(
            tensors=len(requested),
            checkpoint_bytes=checkpoint,
            materialized_bytes=materialized,
            read_transfer_seconds=elapsed,
            total_seconds=elapsed,
            device=str(self.device),
            dtype=str(self.dtype),
        )

    def bind_module(self, module: nn.Module, prefix: str) -> int:
        """Alias a meta module's parameters to already resident device tensors."""
        count = 0
        for parameter_name, _parameter in list(module.named_parameters()):
            if ".experts." in parameter_name:
                continue
            full_name = prefix + parameter_name
            tensor = self.tensor(full_name)
            source_pointer = tensor.data_ptr()
            set_parameter(module, parameter_name, tensor)
            rebound = dict(module.named_parameters())[parameter_name]
            if rebound.data_ptr() != source_pointer:
                raise AssertionError(f"{full_name}: module binding copied storage")
            count += 1
        return count

    def release(self) -> tuple[int, int]:
        count = len(self._tensors)
        materialized = self._materialized_bytes
        self._tensors.clear()
        self._checkpoint_bytes = 0
        self._materialized_bytes = 0
        return count, materialized


_loader = None
_runtime_bank: ResidentTensorBank | None = None
_runtime_bank_lock = threading.Lock()


def loader() -> DirectResidentLoader:
    global _loader
    if _loader is None:
        import direct_shard_loader
        _loader = DirectResidentLoader(direct_shard_loader.store())
    return _loader


def load_resident(name: str) -> torch.Tensor:
    """Drop-in CPU/source-dtype replacement for ``k3loader.load_resident``."""
    if _runtime_bank is not None:
        return _runtime_bank.tensor(name)
    return loader().read_torch(name)


def runtime_bank() -> ResidentTensorBank | None:
    return _runtime_bank


def build_runtime_bank(
    *,
    device: torch.device | str,
    dtype: torch.dtype,
    layers_per_stage: int = 8,
    progress=None,
) -> ResidentTensorBank:
    """Load the complete resident inventory once for the ordinary runtime."""
    global _runtime_bank
    if _runtime_bank is not None:
        return _runtime_bank
    with _runtime_bank_lock:
        if _runtime_bank is not None:
            return _runtime_bank
        direct_loader = loader()
        stages = direct_loader.resident_stages(
            layers_per_stage=layers_per_stage
        )
        expected = sum(len(names) for _label, names in stages)
        bank = ResidentTensorBank(
            direct_loader, device=device, dtype=dtype
        )
        try:
            for index, (label, names) in enumerate(stages, start=1):
                report = bank.load_names(names)
                if progress is not None:
                    progress(index, len(stages), label, bank, report)
            if len(bank) != expected:
                raise AssertionError(
                    f"runtime resident bank has {len(bank)}/{expected} tensors"
                )
        except BaseException:
            bank.release()
            gc.collect()
            if torch.device(device).type == "mps":
                torch.mps.empty_cache()
            raise
        _runtime_bank = bank
        return bank


def release_runtime_bank() -> None:
    global _runtime_bank
    with _runtime_bank_lock:
        bank, _runtime_bank = _runtime_bank, None
    if bank is not None:
        bank.release()
        gc.collect()
        if bank.device.type == "mps":
            torch.mps.empty_cache()


atexit.register(release_runtime_bank)
