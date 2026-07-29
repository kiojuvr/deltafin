"""Reusable page-aligned expert slabs filled directly by ``preadv``."""

from __future__ import annotations

import concurrent.futures
import mmap
from types import MappingProxyType
from typing import Mapping

import numpy as np

from local_safetensors import LocalSafetensorsStore


K3_EXPERT_BYTES = 17_547_264
K3_EXPERT_TENSORS = (
    ("w1", 0, (3072, 1792)),
    ("w1", 1, (3072, 112)),
    ("w2", 0, (3584, 1536)),
    ("w2", 1, (3584, 96)),
    ("w3", 0, (3072, 1792)),
    ("w3", 1, (3072, 112)),
)


def _k3_destinations():
    cursor = 0
    destinations = {}
    for weight, kind, shape in K3_EXPERT_TENSORS:
        length = int(np.prod(shape))
        part = "weight_packed" if kind == 0 else "weight_scale"
        destinations[f"{weight}.{part}"] = (cursor, length)
        cursor += length
    if cursor != K3_EXPERT_BYTES:
        raise AssertionError("K3 tensor shapes do not fill one expert span")
    return destinations


K3_DESTINATIONS = _k3_destinations()


class ExpertSlabBank:
    """One fixed-capacity, page-aligned bank.

    Slot addresses never change.  Loading a new layer overwrites the same
    mappings only after the caller has finished compute on the bank.
    """

    def __init__(
        self,
        store: LocalSafetensorsStore,
        *,
        capacity: int = 16,
        slot_bytes: int = K3_EXPERT_BYTES,
        name: str = "bank",
    ):
        if capacity <= 0 or slot_bytes <= 0:
            raise ValueError("capacity and slot_bytes must be positive")
        if slot_bytes != K3_EXPERT_BYTES:
            raise ValueError(
                f"K3 slab slots must be {K3_EXPERT_BYTES} bytes, got "
                f"{slot_bytes}"
            )
        self.store = store
        self.capacity = capacity
        self.slot_bytes = slot_bytes
        self.name = name
        self._mapping = mmap.mmap(-1, capacity * slot_bytes)
        self._array = np.ndarray(
            (capacity, slot_bytes), dtype=np.uint8, buffer=self._mapping
        )
        self._views: list[dict[str, tuple[np.ndarray, np.ndarray]]] = []
        self._owners: list[object] = []
        self._expert_ids: tuple[int, ...] = ()
        self._slot_experts: list[int | None] = [None] * capacity
        self._last_read_ids: tuple[int, ...] = ()
        self._layer: int | None = None
        self._closed = False
        self._build_k3_views()

    def _build_k3_views(self) -> None:
        offsets = [0]
        for _weight, _kind, shape in K3_EXPERT_TENSORS:
            offsets.append(offsets[-1] + int(np.prod(shape)))
        if offsets[-1] != self.slot_bytes:
            raise AssertionError("K3 expert view sizes do not fill one slab slot")
        for slot in range(self.capacity):
            values: dict[str, list[np.ndarray | None]] = {}
            row = self._array[slot]
            for index, (weight, kind, shape) in enumerate(K3_EXPERT_TENSORS):
                values.setdefault(weight, [None, None])[kind] = row[
                    offsets[index]:offsets[index + 1]
                ].reshape(shape)
            self._views.append(
                {
                    weight: (parts[0], parts[1])
                    for weight, parts in values.items()
                }
            )
            self._owners.append(self._mapping)

    @property
    def nbytes(self) -> int:
        return self.capacity * self.slot_bytes

    @property
    def base_address(self) -> int:
        return self._array.ctypes.data

    @property
    def slot_addresses(self) -> tuple[int, ...]:
        return tuple(self._array[index].ctypes.data for index in range(self.capacity))

    @property
    def page_aligned(self) -> bool:
        page = mmap.PAGESIZE
        return (
            self.base_address % page == 0
            and self.slot_bytes % page == 0
            and all(address % page == 0 for address in self.slot_addresses)
        )

    @property
    def layer(self) -> int | None:
        return self._layer

    @property
    def expert_ids(self) -> tuple[int, ...]:
        return self._expert_ids

    @property
    def last_read_ids(self) -> tuple[int, ...]:
        return self._last_read_ids

    def _layouts(self, layer: int, ids: tuple[int, ...]):
        layouts = [self.store.expert_layout(layer, expert) for expert in ids]
        for layout in layouts:
            if layout.byte_length != self.slot_bytes:
                raise ValueError(
                    f"L{layer} E{layout.expert} is {layout.byte_length} bytes, "
                    f"slot is {self.slot_bytes}"
                )
        return layouts

    def _fill_slot(self, layer: int, layout, slot_index: int) -> None:
        slot = memoryview(self._mapping)[
            slot_index * self.slot_bytes:(slot_index + 1) * self.slot_bytes
        ]
        canonical_contiguous = layout.contiguous and all(
            (
                suffix := self.store._expert_suffix(tensor.name)
            ) in K3_DESTINATIONS
            and tensor.offset - layout.read_spans[0].offset
            == K3_DESTINATIONS[suffix][0]
            and tensor.byte_length == K3_DESTINATIONS[suffix][1]
            for tensor in layout.tensors
        )
        if canonical_contiguous:
            read_span = layout.read_spans[0]
            self.store._read_into(
                read_span.shard,
                read_span.offset,
                slot,
                f"L{layer} E{layout.expert} -> {self.name}[{slot_index}]",
            )
            return
        for tensor in layout.tensors:
            suffix = self.store._expert_suffix(tensor.name)
            try:
                destination, expected_length = K3_DESTINATIONS[suffix]
            except KeyError as exc:
                raise ValueError(
                    f"L{layer} E{layout.expert} has unexpected tensor "
                    f"{suffix!r}"
                ) from exc
            if tensor.byte_length != expected_length:
                raise ValueError(
                    f"L{layer} E{layout.expert} {suffix} is "
                    f"{tensor.byte_length} bytes, expected "
                    f"{expected_length}"
                )
            target = slot[destination:destination + tensor.byte_length]
            self.store._read_into(
                tensor.shard,
                tensor.offset,
                target,
                f"{tensor.name} -> {self.name}[{slot_index}]",
            )

    def _read_assignments(self, layer: int, assignments, workers: int) -> None:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(workers, max(1, len(assignments))),
            thread_name_prefix=f"k3-{self.name}-pread",
        ) as pool:
            list(
                pool.map(
                    lambda row: self._fill_slot(layer, row[1], row[0]),
                    assignments,
                )
            )

    def _mapping_for(self, assigned):
        return MappingProxyType(
            {
                expert: MappingProxyType(self._views[slot])
                for expert, slot in assigned.items()
            }
        )

    def load(
        self, layer: int, expert_ids, *, workers: int = 8
    ) -> Mapping[int, Mapping[str, tuple[np.ndarray, np.ndarray]]]:
        if self._closed:
            raise RuntimeError(f"{self.name} is closed")
        ids = tuple(int(expert) for expert in expert_ids)
        if len(ids) > self.capacity:
            raise ValueError(
                f"{self.name} holds {self.capacity} experts, got {len(ids)}"
            )
        if len(ids) != len(set(ids)):
            raise ValueError("expert IDs must be unique")
        if workers <= 0:
            raise ValueError("workers must be positive")
        layouts = self._layouts(layer, ids)
        assignments = [
            (index, layout) for index, layout in enumerate(layouts)
        ]
        self._read_assignments(layer, assignments, workers)
        self._layer = layer
        self._expert_ids = ids
        self._slot_experts = list(ids) + [None] * (self.capacity - len(ids))
        self._last_read_ids = ids
        return self._mapping_for(
            {expert: index for index, expert in enumerate(ids)}
        )

    def load_reusing(
        self, layer: int, expert_ids, *, workers: int = 8
    ) -> Mapping[int, Mapping[str, tuple[np.ndarray, np.ndarray]]]:
        """Retain same-layer slot hits and read only missing experts."""
        if self._closed:
            raise RuntimeError(f"{self.name} is closed")
        ids = tuple(int(expert) for expert in expert_ids)
        if len(ids) > self.capacity:
            raise ValueError(
                f"{self.name} holds {self.capacity} experts, got {len(ids)}"
            )
        if len(ids) != len(set(ids)):
            raise ValueError("expert IDs must be unique")
        if workers <= 0:
            raise ValueError("workers must be positive")
        if self._layer != layer:
            return self.load(layer, ids, workers=workers)

        old_slots = {
            expert: slot
            for slot, expert in enumerate(self._slot_experts)
            if expert is not None
        }
        assigned = {
            expert: old_slots[expert]
            for expert in ids
            if expert in old_slots
        }
        used_slots = set(assigned.values())
        free_slots = [
            slot for slot in range(self.capacity) if slot not in used_slots
        ]
        misses = tuple(expert for expert in ids if expert not in assigned)
        miss_layouts = self._layouts(layer, misses)
        assignments = []
        for expert, layout, slot in zip(misses, miss_layouts, free_slots):
            assigned[expert] = slot
            assignments.append((slot, layout))
        self._read_assignments(layer, assignments, workers)

        slot_experts: list[int | None] = [None] * self.capacity
        for expert, slot in assigned.items():
            slot_experts[slot] = expert
        self._slot_experts = slot_experts
        self._layer = layer
        self._expert_ids = ids
        self._last_read_ids = misses
        return self._mapping_for(assigned)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._views.clear()
        self._owners.clear()
        self._slot_experts.clear()
        self._array = None
        self._mapping.close()


class DoubleExpertSlab:
    """Two banks: compute can consume one while I/O fills the other."""

    def __init__(
        self,
        store: LocalSafetensorsStore,
        *,
        capacity: int = 16,
        slot_bytes: int = K3_EXPERT_BYTES,
    ):
        self.banks = (
            ExpertSlabBank(
                store, capacity=capacity, slot_bytes=slot_bytes, name="bank-a"
            ),
            ExpertSlabBank(
                store, capacity=capacity, slot_bytes=slot_bytes, name="bank-b"
            ),
        )

    @property
    def nbytes(self) -> int:
        return sum(bank.nbytes for bank in self.banks)

    def close(self) -> None:
        for bank in self.banks:
            bank.close()

    def __enter__(self) -> "DoubleExpertSlab":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
