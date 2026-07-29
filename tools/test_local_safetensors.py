#!/usr/bin/env python3
"""Disk-small tests for the direct local safetensors reader."""

from __future__ import annotations

import concurrent.futures
import json
import os
import pathlib
import queue
import struct
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from local_safetensors import (  # noqa: E402
    LocalSafetensorsStore,
    SidecarMismatchError,
    UnknownExpertError,
    UnknownTensorError,
)
import direct_shard_loader  # noqa: E402
import local_safetensors  # noqa: E402
import resident_shard_loader  # noqa: E402
from resident_shard_loader import (  # noqa: E402
    DirectResidentLoader,
    ResidentTensorBank,
)


PREFIX = "language_model.model.layers"


def expert_name(layer: int, expert: int, suffix: str) -> str:
    return f"{PREFIX}.{layer}.block_sparse_moe.experts.{expert}.{suffix}"


def write_shard(path: pathlib.Path, tensors: list[tuple[str, np.ndarray]]) -> None:
    """Write a valid safetensors file while preserving the requested data order."""
    offset = 0
    header = {}
    blobs = []
    dtype_names = {
        np.dtype(np.uint8): "U8",
        np.dtype(np.float32): "F32",
    }
    for name, array in tensors:
        blob = array.tobytes(order="C")
        header[name] = {
            "dtype": dtype_names[array.dtype],
            "shape": list(array.shape),
            "data_offsets": [offset, offset + len(blob)],
        }
        blobs.append(blob)
        offset += len(blob)
    raw = json.dumps(header, separators=(",", ":")).encode()
    padding = (-len(raw)) % 8
    raw += b" " * padding
    with path.open("wb") as stream:
        stream.write(struct.pack("<Q", len(raw)))
        stream.write(raw)
        for blob in blobs:
            stream.write(blob)


class LocalSafetensorsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temp.name)
        shard_a = "model-00001-of-000002.safetensors"
        shard_b = "model-00002-of-000002.safetensors"
        self.names = {
            "a": expert_name(1, 0, "w1.weight_packed"),
            "b": expert_name(1, 0, "w1.weight_scale"),
            "gap": "unrelated.tensor",
            "c": expert_name(1, 0, "w2.weight_packed"),
            "c_scale": expert_name(1, 0, "w2.weight_scale"),
            "c_w3": expert_name(1, 0, "w3.weight_packed"),
            "c_w3_scale": expert_name(1, 0, "w3.weight_scale"),
            "d": expert_name(1, 1, "w1.weight_packed"),
            "e": expert_name(1, 1, "w1.weight_scale"),
        }
        arrays_a = [
            (self.names["a"], np.arange(12, dtype=np.uint8).reshape(3, 4)),
            (self.names["b"], np.arange(3, dtype=np.uint8)),
            (self.names["gap"], np.array([123.5], dtype=np.float32)),
            (self.names["c"], np.arange(8, dtype=np.uint8).reshape(2, 4)),
            (self.names["c_scale"], np.array([6, 7], dtype=np.uint8)),
            (self.names["c_w3"], np.arange(6, dtype=np.uint8).reshape(2, 3)),
            (self.names["c_w3_scale"], np.array([8, 9], dtype=np.uint8)),
            (self.names["d"], np.array([9, 8, 7, 6], dtype=np.uint8)),
        ]
        arrays_b = [
            (self.names["e"], np.array([5, 4], dtype=np.uint8)),
            ("resident.scalar", np.array([7.25], dtype=np.float32)),
            (
                "resident.matrix",
                np.arange(12, dtype=np.float32).reshape(3, 4),
            ),
        ]
        write_shard(self.root / shard_a, arrays_a)
        write_shard(self.root / shard_b, arrays_b)
        weight_map = {
            name: shard_a for name, _ in arrays_a
        } | {
            name: shard_b for name, _ in arrays_b
        }
        total_size = sum(array.nbytes for _, array in arrays_a + arrays_b)
        (self.root / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": total_size}, "weight_map": weight_map})
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_inventory_contiguous_fallback_and_standard_reader_match(self):
        with LocalSafetensorsStore(self.root) as store:
            self.assertEqual(store.shard_count, 2)
            self.assertEqual(store.tensor_count, 11)
            layout0 = store.expert_layout(1, 0)
            self.assertFalse(layout0.contiguous)
            self.assertEqual(len(layout0.read_spans), 2)
            layout1 = store.expert_layout(1, 1)
            self.assertFalse(layout1.contiguous)
            self.assertEqual(len(layout1.read_spans), 2)
            self.assertEqual(len(layout1.shard_names), 2)

            got = store.read_expert(1, 0)
            self.assertEqual(
                set(got),
                {
                    "w1.weight_packed",
                    "w1.weight_scale",
                    "w2.weight_packed",
                    "w2.weight_scale",
                    "w3.weight_packed",
                    "w3.weight_scale",
                },
            )
            for key, full_name in (
                ("w1.weight_packed", self.names["a"]),
                ("w1.weight_scale", self.names["b"]),
                ("w2.weight_packed", self.names["c"]),
            ):
                span = store.tensor_span(full_name)
                with safe_open(
                    str(self.root / span.shard), framework="numpy", device="cpu"
                ) as handle:
                    expected = handle.get_tensor(full_name)
                np.testing.assert_array_equal(got[key], expected)
                self.assertEqual(got[key].dtype, expected.dtype)
                self.assertEqual(got[key].shape, expected.shape)
            compatible = store.read_expert_raw(1, 0)
            self.assertEqual(set(compatible), {"w1", "w2", "w3"})
            np.testing.assert_array_equal(
                compatible["w1"][0], got["w1.weight_packed"]
            )

            one = store.read_tensor("resident.scalar")
            np.testing.assert_array_equal(one, np.array([7.25], dtype=np.float32))
            destination = bytearray(store.tensor_span("resident.matrix").byte_length)
            self.assertEqual(
                store.read_tensor_into("resident.matrix", destination),
                len(destination),
            )
            np.testing.assert_array_equal(
                np.frombuffer(destination, dtype=np.float32).reshape(3, 4),
                np.arange(12, dtype=np.float32).reshape(3, 4),
            )
            with self.assertRaisesRegex(ValueError, "destination has"):
                store.read_tensor_into("resident.matrix", bytearray(1))
            self.assertGreater(store.open_fd_count, 0)
        self.assertEqual(store.open_fd_count, 0)

    def test_parallel_read_and_clear_errors(self):
        with LocalSafetensorsStore(self.root) as store:
            got = store.read_experts(1, [0, 1], workers=2)
            self.assertEqual(set(got), {0, 1})
            with self.assertRaisesRegex(UnknownExpertError, "unknown routed-expert layer"):
                store.expert_layout(99, 0)
            with self.assertRaisesRegex(UnknownExpertError, "unknown expert 99"):
                store.expert_layout(1, 99)
            with self.assertRaisesRegex(UnknownTensorError, "unknown tensor"):
                store.tensor_span("not.present")
            with self.assertRaisesRegex(UnknownTensorError, "available tensors"):
                store.expert_tensor_span(1, 0, "w9.weight")
            with self.assertRaisesRegex(ValueError, "unique"):
                store.read_experts(1, [0, 0])
        with self.assertRaisesRegex(RuntimeError, "closed"):
            store.read_tensor(self.names["a"])

    def test_positional_reads_chunk_large_requests(self):
        with LocalSafetensorsStore(self.root) as store:
            with mock.patch.object(local_safetensors, "_PREAD_CHUNK", 3):
                payload = store.read_tensor_bytes("resident.matrix")
                expected = np.arange(
                    12, dtype=np.float32
                ).reshape(3, 4).tobytes()
                self.assertEqual(payload, expected)
                destination = bytearray(len(expected))
                store.read_tensor_into("resident.matrix", destination)
                self.assertEqual(bytes(destination), expected)

    def test_sidecar_round_trip_and_fingerprint(self):
        sidecar = self.root / "index.json.gz"
        with LocalSafetensorsStore(self.root) as original:
            original.write_sidecar(sidecar)
            expected = original.read_tensor_bytes(self.names["a"])
        with LocalSafetensorsStore(self.root, sidecar=sidecar) as restored:
            self.assertEqual(restored.report().source, "sidecar")
            self.assertEqual(restored.read_tensor_bytes(self.names["a"]), expected)
        index = self.root / "model.safetensors.index.json"
        payload = json.loads(index.read_text())
        payload["metadata"]["test_mutation"] = True
        index.write_text(json.dumps(payload))
        with self.assertRaisesRegex(SidecarMismatchError, "hash"):
            LocalSafetensorsStore(self.root, sidecar=sidecar)

    def test_deltafin_loader_adapter(self):
        direct_shard_loader.close()
        direct_shard_loader._store = LocalSafetensorsStore(self.root)
        try:
            raw = direct_shard_loader.fetch_experts(1, [0], workers=1, dequant=False)
            self.assertEqual(set(raw), {0})
            self.assertEqual(set(raw[0]), {"w1", "w2", "w3"})
            np.testing.assert_array_equal(
                raw[0]["w1"][0], np.arange(12, dtype=np.uint8).reshape(3, 4)
            )
        finally:
            direct_shard_loader.close()
        self.assertEqual(direct_shard_loader._store, None)

    def test_partial_resident_loader(self):
        with LocalSafetensorsStore(self.root) as store:
            loader = DirectResidentLoader(store)
            self.assertIn("resident.scalar", loader.resident_names())
            scalar = loader.read_torch("resident.scalar")
            self.assertEqual(scalar.dtype, __import__("torch").float32)
            self.assertEqual(scalar.tolist(), [7.25])
            rows = loader.read_rows_torch("resident.matrix", [2, 0])
            np.testing.assert_array_equal(
                rows.numpy(),
                np.array([[8, 9, 10, 11], [0, 1, 2, 3]], dtype=np.float32),
            )
            with self.assertRaisesRegex(IndexError, "row outside"):
                loader.read_rows_torch("resident.matrix", [3])
            with self.assertRaisesRegex(ValueError, "routed-expert"):
                loader.read_bytes(self.names["a"])

    def test_resident_tensor_bank_owns_only_materialized_tensors(self):
        import torch

        with LocalSafetensorsStore(self.root) as store:
            loader = DirectResidentLoader(store)
            bank = ResidentTensorBank(
                loader, device="cpu", dtype=torch.float32
            )
            report = bank.load_names(
                ["resident.scalar", "resident.matrix"]
            )
            self.assertEqual(report.tensors, 2)
            self.assertEqual(len(bank), 2)
            self.assertEqual(bank.tensor("resident.matrix").shape, (3, 4))
            self.assertEqual(bank.materialized_bytes, 13 * 4)
            with self.assertRaisesRegex(ValueError, "already loaded"):
                bank.load_names(["resident.scalar"])
            with self.assertRaisesRegex(KeyError, "is not loaded"):
                bank.tensor("resident.missing")
            self.assertEqual(bank.release(), (2, 13 * 4))
            self.assertEqual(len(bank), 0)

    def test_runtime_resident_bank_reuses_device_storage(self):
        import torch

        resident_shard_loader.release_runtime_bank()
        old_loader = resident_shard_loader._loader
        with LocalSafetensorsStore(self.root) as store:
            loader = DirectResidentLoader(store)
            resident_shard_loader._loader = loader
            try:
                stages = loader.resident_stages(layers_per_stage=8)
                self.assertEqual(stages[0][0], "global")
                bank = resident_shard_loader.build_runtime_bank(
                    device="cpu", dtype=torch.float32
                )
                tensor = resident_shard_loader.load_resident(
                    "resident.matrix"
                )
                self.assertEqual(
                    tensor.data_ptr(),
                    bank.tensor("resident.matrix").data_ptr(),
                )
            finally:
                resident_shard_loader.release_runtime_bank()
                resident_shard_loader._loader = old_loader

    def test_slab_adapter_holds_bank_lease_through_context(self):
        class FakeBank:
            def __init__(self, value):
                self.value = value

            def load(self, layer, ids, workers):
                return {
                    "bank": self.value,
                    "layer": layer,
                    "ids": tuple(ids),
                    "workers": workers,
                }

        class FakeSlabs:
            banks = (FakeBank("a"), FakeBank("b"))

            def close(self):
                pass

        direct_shard_loader.close()
        direct_shard_loader._store = LocalSafetensorsStore(self.root)
        direct_shard_loader._slabs = FakeSlabs()
        direct_shard_loader._slab_available = queue.Queue(maxsize=2)
        direct_shard_loader._slab_available.put(0)
        direct_shard_loader._slab_available.put(1)
        old_enabled = direct_shard_loader.DIRECT_SLAB
        direct_shard_loader.DIRECT_SLAB = True
        try:
            with direct_shard_loader.slab_experts(
                1, [0], workers=1
            ) as raw:
                self.assertEqual(raw["bank"], "a")
                self.assertEqual(
                    direct_shard_loader._slab_available.qsize(), 1
                )
            self.assertEqual(
                direct_shard_loader._slab_available.qsize(), 2
            )
        finally:
            direct_shard_loader.DIRECT_SLAB = old_enabled
            direct_shard_loader.close()

    def test_slab_overlap_reuses_hits_and_reads_misses(self):
        class Layout:
            byte_length = 17

        class FakeStore:
            def expert_layout(self, layer, expert):
                return Layout()

            def close(self):
                pass

        class FakeBank:
            def __init__(self):
                self.layer = None
                self.ids = ()
                self.last_read_ids = ()

            def load(self, layer, ids, workers):
                self.layer = layer
                self.ids = tuple(ids)
                self.last_read_ids = self.ids
                return {expert: {"slot": expert} for expert in ids}

            def load_reusing(self, layer, ids, workers):
                ids = tuple(ids)
                old = set(self.ids) if self.layer == layer else set()
                self.last_read_ids = tuple(
                    expert for expert in ids if expert not in old
                )
                self.layer = layer
                self.ids = ids
                return {expert: {"slot": expert} for expert in ids}

        class FakeSlabs:
            def __init__(self):
                self.banks = (FakeBank(), FakeBank())

            def close(self):
                pass

        direct_shard_loader.close()
        old_slab = direct_shard_loader.DIRECT_SLAB
        old_overlap = direct_shard_loader.DIRECT_OVERLAP
        old_stats = dict(direct_shard_loader.stats)
        direct_shard_loader.DIRECT_SLAB = True
        direct_shard_loader.DIRECT_OVERLAP = True
        direct_shard_loader._store = FakeStore()
        direct_shard_loader._slabs = FakeSlabs()
        direct_shard_loader._slab_available = queue.Queue(maxsize=2)
        direct_shard_loader._slab_available.put(0)
        direct_shard_loader._slab_available.put(1)
        direct_shard_loader._slab_executor = (
            concurrent.futures.ThreadPoolExecutor(max_workers=1)
        )
        try:
            self.assertTrue(
                direct_shard_loader.prefetch_slab(2, [1, 2], workers=1)
            )
            with direct_shard_loader.slab_experts(
                2, [2, 3], workers=1
            ) as raw:
                self.assertEqual(set(raw), {2, 3})
            self.assertEqual(
                direct_shard_loader.stats["overlap_hits"]
                - old_stats["overlap_hits"],
                1,
            )
            self.assertEqual(
                direct_shard_loader.stats["overlap_misses"]
                - old_stats["overlap_misses"],
                1,
            )
        finally:
            direct_shard_loader.DIRECT_SLAB = old_slab
            direct_shard_loader.DIRECT_OVERLAP = old_overlap
            direct_shard_loader.close()
            direct_shard_loader.stats.update(old_stats)

    def test_slab_settle_releases_every_bank_after_failure(self):
        direct_shard_loader.close()
        available = queue.Queue(maxsize=2)
        failed = concurrent.futures.Future()
        failed.set_exception(RuntimeError("prefetch failed"))
        completed = concurrent.futures.Future()
        completed.set_result(None)
        direct_shard_loader._slab_available = available
        direct_shard_loader._slab_pending.update(
            {
                1: {"bank_index": 0, "future": failed},
                2: {"bank_index": 1, "future": completed},
            }
        )
        try:
            with self.assertRaisesRegex(RuntimeError, "prefetch failed"):
                direct_shard_loader.settle_slab_prefetches()
            self.assertEqual(available.qsize(), 2)
            self.assertFalse(direct_shard_loader._slab_pending)
        finally:
            direct_shard_loader._slab_pending.clear()
            direct_shard_loader._slab_available = None

    def test_demand_bandwidth_signal_excludes_reset_state(self):
        old_stats = dict(direct_shard_loader.stats)
        direct_shard_loader.reset_demand_signal()
        try:
            self.assertIsNone(
                direct_shard_loader.demand_read_snapshot()["ema_gbps"]
            )
            direct_shard_loader._record_demand_read(1_000_000_000, 1.0)
            first = direct_shard_loader.demand_read_snapshot()
            self.assertEqual(first["last_gbps"], 1.0)
            self.assertEqual(first["ema_gbps"], 1.0)
            direct_shard_loader._record_demand_read(1_000_000_000, 0.5)
            second = direct_shard_loader.demand_read_snapshot()
            self.assertEqual(second["last_gbps"], 2.0)
            self.assertEqual(
                second["ema_gbps"],
                1.0 + direct_shard_loader.DEMAND_EMA_ALPHA,
            )
        finally:
            direct_shard_loader.reset_demand_signal()
            direct_shard_loader.stats.update(old_stats)


if __name__ == "__main__":
    unittest.main()
