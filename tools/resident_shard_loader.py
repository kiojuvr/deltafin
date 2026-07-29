"""Partial resident-weight loader over unmodified local safetensors shards."""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn

from local_safetensors import LocalSafetensorsStore, TensorSpan


_ROUTED_EXPERT = re.compile(
    r"^language_model\.model\.layers\.\d+\.block_sparse_moe\.experts\.\d+\."
)
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
        payload = bytearray(self.store.read_tensor_bytes(name))
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
            payload = bytearray(self.store.read_tensor_bytes(full_name))
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


_loader = None


def loader() -> DirectResidentLoader:
    global _loader
    if _loader is None:
        import direct_shard_loader
        _loader = DirectResidentLoader(direct_shard_loader.store())
    return _loader


def load_resident(name: str) -> torch.Tensor:
    """Drop-in CPU/source-dtype replacement for ``k3loader.load_resident``."""
    return loader().read_torch(name)
