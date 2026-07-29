#!/usr/bin/env python3
"""Disk-small tests for the direct local safetensors reader."""

from __future__ import annotations

import json
import os
import pathlib
import struct
import sys
import tempfile
import unittest

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
            self.assertEqual(store.tensor_count, 10)
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


if __name__ == "__main__":
    unittest.main()
