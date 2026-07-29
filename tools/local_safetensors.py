"""Direct, read-only access to tensors in local safetensors shards.

This module deliberately does not import Deltafin's HTTP fetcher or expert
cache.  It builds its view from ``model.safetensors.index.json`` and the
headers of the referenced local shards, then uses positional I/O for payload
reads.  The source model is never opened for writing.

The Kimi-K3 routed-expert naming convention is recognized, but tensor layout is
always derived from the headers.  A contiguous expert is read with one pread;
gapped or cross-shard experts fall back to one pread per contiguous group.
"""

from __future__ import annotations

import concurrent.futures
import fcntl
import gzip
import hashlib
import json
import math
import os
import pathlib
import re
import struct
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable, Mapping

import numpy as np


INDEX_NAME = "model.safetensors.index.json"
SIDECAR_SCHEMA = "deltafin.local-safetensors-index.v1"
_HEADER_LIMIT = 1 << 30
_DARWIN_F_NOCACHE = 48
_EXPERT_RE = re.compile(
    r"^language_model\.model\.layers\.(\d+)\.block_sparse_moe"
    r"\.experts\.(\d+)\.(.+)$"
)

# Safetensors' storage dtypes.  K3 routed experts are U8, while this complete
# table lets the inventory validate all resident tensors as well.
_ITEMSIZE = {
    "BOOL": 1,
    "I8": 1,
    "U8": 1,
    "I16": 2,
    "U16": 2,
    "I32": 4,
    "U32": 4,
    "I64": 8,
    "U64": 8,
    "F8_E4M3": 1,
    "F8_E4M3FN": 1,
    "F8_E5M2": 1,
    "F8_E4M3FNUZ": 1,
    "F8_E5M2FNUZ": 1,
    "F16": 2,
    "BF16": 2,
    "F32": 4,
    "F64": 8,
}
_NUMPY_DTYPES = {
    "BOOL": np.bool_,
    "I8": np.int8,
    "U8": np.uint8,
    "I16": np.int16,
    "U16": np.uint16,
    "I32": np.int32,
    "U32": np.uint32,
    "I64": np.int64,
    "U64": np.uint64,
    "F16": np.float16,
    "F32": np.float32,
    "F64": np.float64,
}


class SafetensorsStoreError(RuntimeError):
    """Base class for local-store errors."""


class InvalidSafetensorsError(SafetensorsStoreError):
    """A shard, header, index, or tensor range is malformed."""


class SidecarMismatchError(SafetensorsStoreError):
    """A sidecar does not describe the supplied model directory."""


class UnknownTensorError(KeyError, SafetensorsStoreError):
    """The requested tensor is absent."""


class UnknownExpertError(KeyError, SafetensorsStoreError):
    """The requested routed expert is absent."""


@dataclass(frozen=True, slots=True)
class TensorSpan:
    """One tensor's exact payload location."""

    name: str
    shard: str
    offset: int
    byte_length: int
    dtype: str
    shape: tuple[int, ...]

    @property
    def end(self) -> int:
        return self.offset + self.byte_length


@dataclass(frozen=True, slots=True)
class ReadSpan:
    """A single positional read containing one or more tensors."""

    shard: str
    offset: int
    byte_length: int
    tensor_names: tuple[str, ...]

    @property
    def end(self) -> int:
        return self.offset + self.byte_length


@dataclass(frozen=True, slots=True)
class ExpertLayout:
    """Header-derived layout of a routed expert."""

    layer: int
    expert: int
    tensors: tuple[TensorSpan, ...]
    read_spans: tuple[ReadSpan, ...]

    @property
    def contiguous(self) -> bool:
        return len(self.read_spans) == 1

    @property
    def byte_length(self) -> int:
        return sum(t.byte_length for t in self.tensors)

    @property
    def shard_names(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(t.shard for t in self.tensors))


@dataclass(frozen=True, slots=True)
class InventoryReport:
    shards: int
    tensors: int
    routed_layers: int
    routed_experts: int
    contiguous_experts: int
    cross_shard_experts: int
    payload_bytes: int
    parse_seconds: float
    source: str


def _pread_exact(fd: int, length: int, offset: int, label: str) -> bytes:
    """Read exactly ``length`` bytes without changing a shared file pointer."""
    if length == 0:
        return b""
    chunks: list[bytes] = []
    remaining = length
    position = offset
    while remaining:
        chunk = os.pread(fd, remaining, position)
        if not chunk:
            got = length - remaining
            raise IOError(f"short pread for {label}: {got}/{length} bytes")
        chunks.append(chunk)
        position += len(chunk)
        remaining -= len(chunk)
    return chunks[0] if len(chunks) == 1 else b"".join(chunks)


def _preadv_exact(fd: int, target: memoryview, offset: int, label: str) -> int:
    """Fill a writable buffer directly with positional I/O."""
    if target.readonly:
        raise TypeError(f"{label}: destination buffer is read-only")
    target = target.cast("B")
    done = 0
    while done < target.nbytes:
        count = os.preadv(fd, [target[done:]], offset + done)
        if count <= 0:
            raise IOError(
                f"short preadv for {label}: {done}/{target.nbytes} bytes"
            )
        done += count
    return done


def _shape_nbytes(dtype: str, shape: tuple[int, ...]) -> int:
    try:
        itemsize = _ITEMSIZE[dtype]
    except KeyError as exc:
        raise InvalidSafetensorsError(f"unsupported safetensors dtype {dtype!r}") from exc
    return math.prod(shape) * itemsize


def _numpy_dtype(dtype: str) -> np.dtype:
    if dtype == "BF16":
        # ml_dtypes registers bfloat16 with numpy when installed.  Deltafin's
        # environment includes it, but raw-byte APIs remain usable without it.
        try:
            import ml_dtypes
        except ImportError as exc:
            raise TypeError(
                "BF16 conversion needs ml_dtypes; use read_tensor_bytes()"
            ) from exc
        return np.dtype(ml_dtypes.bfloat16)
    try:
        return np.dtype(_NUMPY_DTYPES[dtype])
    except KeyError as exc:
        raise TypeError(
            f"{dtype} has no lossless numpy view; use read_tensor_bytes()"
        ) from exc


def _tensor_array(span: TensorSpan, payload: bytes, relative_offset: int = 0) -> np.ndarray:
    dtype = _numpy_dtype(span.dtype)
    count = math.prod(span.shape)
    array = np.frombuffer(payload, dtype=dtype, count=count, offset=relative_offset)
    return array.reshape(span.shape)


class LocalSafetensorsStore:
    """Read-only local safetensors inventory and positional reader.

    File descriptors are opened lazily, shared safely across ``os.pread`` calls,
    and closed by :meth:`close` or the context-manager exit.  Constructing from
    raw headers is the correctness path.  Passing an existing sidecar avoids
    decoding all 96 JSON headers while retaining source fingerprint checks.
    """

    def __init__(
        self,
        model_dir: os.PathLike[str] | str,
        *,
        sidecar: os.PathLike[str] | str | None = None,
        strict_index: bool = True,
        darwin_nocache: bool = False,
    ):
        started = time.perf_counter()
        self.model_dir = pathlib.Path(model_dir).expanduser().resolve()
        self.index_path = self.model_dir / INDEX_NAME
        self._strict_index = strict_index
        if darwin_nocache and sys.platform != "darwin":
            raise ValueError("darwin_nocache is only available on macOS")
        self._darwin_nocache = darwin_nocache
        self._fd_lock = threading.Lock()
        self._fds: dict[str, int] = {}
        self._closed = False
        self._index_sha256 = ""
        self._shard_info: dict[str, tuple[int, int]] = {}
        self._tensors: dict[str, TensorSpan] = {}
        self._experts: dict[tuple[int, int], ExpertLayout] = {}
        self._layers: dict[int, tuple[int, ...]] = {}
        self._payload_bytes = 0
        self._source = "headers"

        if sidecar is not None and pathlib.Path(sidecar).is_file():
            self._load_sidecar(pathlib.Path(sidecar))
            self._source = "sidecar"
        else:
            self._parse_model()
        self.parse_seconds = time.perf_counter() - started

    def __enter__(self) -> "LocalSafetensorsStore":
        if self._closed:
            raise RuntimeError("LocalSafetensorsStore is closed")
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    @property
    def tensor_count(self) -> int:
        return len(self._tensors)

    @property
    def shard_count(self) -> int:
        return len(self._shard_info)

    @property
    def routed_expert_count(self) -> int:
        return len(self._experts)

    @property
    def routed_layers(self) -> tuple[int, ...]:
        return tuple(sorted(self._layers))

    @property
    def open_fd_count(self) -> int:
        with self._fd_lock:
            return len(self._fds)

    def report(self) -> InventoryReport:
        return InventoryReport(
            shards=self.shard_count,
            tensors=self.tensor_count,
            routed_layers=len(self._layers),
            routed_experts=self.routed_expert_count,
            contiguous_experts=sum(x.contiguous for x in self._experts.values()),
            cross_shard_experts=sum(len(x.shard_names) > 1 for x in self._experts.values()),
            payload_bytes=self._payload_bytes,
            parse_seconds=self.parse_seconds,
            source=self._source,
        )

    def tensor_span(self, name: str) -> TensorSpan:
        try:
            return self._tensors[name]
        except KeyError as exc:
            raise UnknownTensorError(f"unknown tensor {name!r}") from exc

    def expert_ids(self, layer: int) -> tuple[int, ...]:
        try:
            return self._layers[layer]
        except KeyError as exc:
            available = (
                f"{min(self._layers)}..{max(self._layers)}" if self._layers else "none"
            )
            raise UnknownExpertError(
                f"unknown routed-expert layer {layer}; available layers: {available}"
            ) from exc

    def expert_layout(self, layer: int, expert: int) -> ExpertLayout:
        try:
            return self._experts[(layer, expert)]
        except KeyError as exc:
            if layer not in self._layers:
                self.expert_ids(layer)  # raises the more specific layer error
            ids = self._layers[layer]
            raise UnknownExpertError(
                f"unknown expert {expert} in layer {layer}; "
                f"available IDs: {ids[0]}..{ids[-1]}"
            ) from exc

    def expert_tensor_span(self, layer: int, expert: int, tensor_name: str) -> TensorSpan:
        layout = self.expert_layout(layer, expert)
        for span in layout.tensors:
            if tensor_name in (span.name, self._expert_suffix(span.name)):
                return span
        valid = ", ".join(self._expert_suffix(x.name) for x in layout.tensors)
        raise UnknownTensorError(
            f"unknown tensor {tensor_name!r} for layer {layer} expert {expert}; "
            f"available tensors: {valid}"
        )

    def read_tensor_bytes(self, name: str) -> bytes:
        span = self.tensor_span(name)
        return self._read_span(span.shard, span.offset, span.byte_length, span.name)

    def read_tensor(self, name: str) -> np.ndarray:
        span = self.tensor_span(name)
        return _tensor_array(span, self.read_tensor_bytes(name))

    def read_tensor_into(self, name: str, destination) -> int:
        """Read a tensor directly into an exactly-sized writable buffer."""
        span = self.tensor_span(name)
        target = memoryview(destination)
        if target.nbytes != span.byte_length:
            raise ValueError(
                f"{name!r} needs {span.byte_length} bytes, destination has "
                f"{target.nbytes}"
            )
        return self._read_into(
            span.shard, span.offset, target, span.name
        )

    def read_expert(self, layer: int, expert: int) -> Mapping[str, np.ndarray]:
        """Read one expert, coalescing adjacent tensors when the header permits."""
        layout = self.expert_layout(layer, expert)
        output: dict[str, np.ndarray] = {}
        tensor_by_name = {tensor.name: tensor for tensor in layout.tensors}
        for group in layout.read_spans:
            payload = self._read_span(
                group.shard,
                group.offset,
                group.byte_length,
                f"layer {layer} expert {expert}",
            )
            for name in group.tensor_names:
                tensor = tensor_by_name[name]
                relative = tensor.offset - group.offset
                output[self._expert_suffix(name)] = _tensor_array(
                    tensor, payload, relative
                )
        return MappingProxyType(output)

    def read_expert_raw(
        self, layer: int, expert: int
    ) -> Mapping[str, tuple[np.ndarray, np.ndarray]]:
        """Return Deltafin's existing ``w -> (packed, scale)`` expert shape."""
        tensors = self.read_expert(layer, expert)
        return MappingProxyType(
            {
                weight: (
                    tensors[f"{weight}.weight_packed"],
                    tensors[f"{weight}.weight_scale"],
                )
                for weight in ("w1", "w2", "w3")
            }
        )

    def read_expert_tensor(
        self, layer: int, expert: int, tensor_name: str
    ) -> np.ndarray:
        span = self.expert_tensor_span(layer, expert, tensor_name)
        return _tensor_array(
            span,
            self._read_span(span.shard, span.offset, span.byte_length, span.name),
        )

    def read_experts(
        self, layer: int, experts: Iterable[int], *, workers: int = 8
    ) -> dict[int, Mapping[str, np.ndarray]]:
        """Read independent experts in parallel through shared positional fds."""
        ids = list(experts)
        if len(ids) != len(set(ids)):
            raise ValueError("expert IDs must be unique")
        if workers <= 0:
            raise ValueError("workers must be positive")
        for expert in ids:
            self.expert_layout(layer, expert)
        if not ids:
            return {}
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(workers, len(ids)), thread_name_prefix="k3-direct-pread"
        ) as pool:
            futures = {pool.submit(self.read_expert, layer, expert): expert for expert in ids}
            return {futures[f]: f.result() for f in concurrent.futures.as_completed(futures)}

    def write_sidecar(self, path: os.PathLike[str] | str) -> float:
        """Atomically write a compact, gzip-compressed, rebuildable inventory."""
        destination = pathlib.Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": SIDECAR_SCHEMA,
            "index_sha256": self._index_sha256,
            "shards": [
                [name, size, header_len]
                for name, (size, header_len) in sorted(self._shard_info.items())
            ],
            "tensors": [
                [
                    span.name,
                    span.shard,
                    span.offset,
                    span.byte_length,
                    span.dtype,
                    list(span.shape),
                ]
                for span in self._tensors.values()
            ],
        }
        started = time.perf_counter()
        fd, temporary_name = tempfile.mkstemp(
            prefix=destination.name + ".", suffix=".tmp", dir=destination.parent
        )
        try:
            with os.fdopen(fd, "wb") as raw:
                with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
                    compressed.write(
                        json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode(
                            "utf-8"
                        )
                    )
                raw.flush()
                os.fsync(raw.fileno())
            os.replace(temporary_name, destination)
        except BaseException:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise
        return time.perf_counter() - started

    def close(self) -> None:
        with self._fd_lock:
            if self._closed:
                return
            self._closed = True
            fds, self._fds = self._fds, {}
        errors = []
        for shard, fd in fds.items():
            try:
                os.close(fd)
            except OSError as exc:
                errors.append(f"{shard}: {exc}")
        if errors:
            raise OSError("failed to close shard descriptors: " + "; ".join(errors))

    def _parse_model(self) -> None:
        try:
            index_bytes = self.index_path.read_bytes()
        except OSError as exc:
            raise InvalidSafetensorsError(f"cannot read {self.index_path}: {exc}") from exc
        self._index_sha256 = hashlib.sha256(index_bytes).hexdigest()
        try:
            index = json.loads(index_bytes)
            weight_map = index["weight_map"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise InvalidSafetensorsError(f"invalid {self.index_path}: {exc}") from exc
        if not isinstance(weight_map, dict) or not weight_map:
            raise InvalidSafetensorsError("index weight_map must be a non-empty object")

        shard_names = sorted(set(weight_map.values()))
        if not all(isinstance(name, str) and pathlib.PurePath(name).name == name
                   for name in shard_names):
            raise InvalidSafetensorsError("index contains unsafe shard names")

        tensors: dict[str, TensorSpan] = {}
        shard_info: dict[str, tuple[int, int]] = {}
        for shard_name in shard_names:
            shard_path = self.model_dir / shard_name
            try:
                size = shard_path.stat().st_size
                fd = os.open(shard_path, os.O_RDONLY)
            except OSError as exc:
                raise InvalidSafetensorsError(f"cannot open shard {shard_path}: {exc}") from exc
            try:
                prefix = _pread_exact(fd, 8, 0, shard_name)
                header_len = struct.unpack("<Q", prefix)[0]
                if header_len <= 0 or header_len > _HEADER_LIMIT or 8 + header_len > size:
                    raise InvalidSafetensorsError(
                        f"{shard_name}: invalid header length {header_len} for {size}-byte file"
                    )
                raw_header = _pread_exact(fd, header_len, 8, f"{shard_name} header")
            finally:
                os.close(fd)
            try:
                header = json.loads(raw_header)
            except json.JSONDecodeError as exc:
                raise InvalidSafetensorsError(f"{shard_name}: invalid JSON header: {exc}") from exc
            if not isinstance(header, dict):
                raise InvalidSafetensorsError(f"{shard_name}: header is not an object")
            header.pop("__metadata__", None)
            shard_info[shard_name] = (size, header_len)
            local_spans = []
            for name, info in header.items():
                span = self._parse_tensor_info(name, shard_name, header_len, size, info)
                if name in tensors:
                    raise InvalidSafetensorsError(f"duplicate tensor {name!r}")
                tensors[name] = span
                local_spans.append(span)
            local_spans.sort(key=lambda item: item.offset)
            for left, right in zip(local_spans, local_spans[1:]):
                if left.end > right.offset:
                    raise InvalidSafetensorsError(
                        f"{shard_name}: overlapping tensors {left.name!r} and {right.name!r}"
                    )

        if self._strict_index:
            header_names = set(tensors)
            index_names = set(weight_map)
            if header_names != index_names:
                missing = sorted(index_names - header_names)[:3]
                extra = sorted(header_names - index_names)[:3]
                raise InvalidSafetensorsError(
                    f"index/header tensor mismatch: missing={missing}, extra={extra}"
                )
            wrong = [
                name
                for name, shard_name in weight_map.items()
                if tensors[name].shard != shard_name
            ]
            if wrong:
                name = wrong[0]
                raise InvalidSafetensorsError(
                    f"index maps {name!r} to {weight_map[name]!r}, "
                    f"header maps it to {tensors[name].shard!r}"
                )
            declared = index.get("metadata", {}).get("total_size")
            actual = sum(span.byte_length for span in tensors.values())
            if declared is not None and int(declared) != actual:
                raise InvalidSafetensorsError(
                    f"index total_size {declared} != tensor payload bytes {actual}"
                )

        self._install_inventory(tensors, shard_info)

    @staticmethod
    def _parse_tensor_info(
        name: str, shard: str, header_len: int, file_size: int, info: object
    ) -> TensorSpan:
        if not isinstance(name, str) or not name:
            raise InvalidSafetensorsError(f"{shard}: tensor name must be non-empty")
        try:
            dtype = info["dtype"]  # type: ignore[index]
            shape_raw = info["shape"]  # type: ignore[index]
            offsets = info["data_offsets"]  # type: ignore[index]
        except (KeyError, TypeError) as exc:
            raise InvalidSafetensorsError(f"{shard}: malformed tensor {name!r}") from exc
        if (
            not isinstance(dtype, str)
            or not isinstance(shape_raw, list)
            or not all(isinstance(dim, int) and dim >= 0 for dim in shape_raw)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(value, int) for value in offsets)
        ):
            raise InvalidSafetensorsError(f"{shard}: invalid metadata for tensor {name!r}")
        relative_start, relative_end = offsets
        if relative_start < 0 or relative_end < relative_start:
            raise InvalidSafetensorsError(f"{shard}: invalid offsets for tensor {name!r}")
        shape = tuple(shape_raw)
        byte_length = relative_end - relative_start
        expected = _shape_nbytes(dtype, shape)
        if byte_length != expected:
            raise InvalidSafetensorsError(
                f"{shard}: tensor {name!r} has {byte_length} bytes, expected {expected} "
                f"for {dtype}{shape}"
            )
        absolute_offset = 8 + header_len + relative_start
        if absolute_offset < 8 + header_len or absolute_offset + byte_length > file_size:
            raise InvalidSafetensorsError(
                f"{shard}: tensor {name!r} range "
                f"[{absolute_offset},{absolute_offset + byte_length}) outside file size "
                f"{file_size}"
            )
        return TensorSpan(name, shard, absolute_offset, byte_length, dtype, shape)

    def _install_inventory(
        self,
        tensors: dict[str, TensorSpan],
        shard_info: dict[str, tuple[int, int]],
    ) -> None:
        expert_tensors: dict[tuple[int, int], list[TensorSpan]] = {}
        for name, span in tensors.items():
            match = _EXPERT_RE.match(name)
            if match:
                expert_tensors.setdefault((int(match[1]), int(match[2])), []).append(span)

        experts: dict[tuple[int, int], ExpertLayout] = {}
        layers: dict[int, list[int]] = {}
        for (layer, expert), spans in expert_tensors.items():
            ordered = tuple(sorted(spans, key=lambda item: (item.shard, item.offset)))
            read_spans: list[ReadSpan] = []
            current: list[TensorSpan] = []
            for tensor in ordered:
                if current and (
                    tensor.shard != current[-1].shard
                    or tensor.offset != current[-1].end
                ):
                    read_spans.append(self._make_read_span(current))
                    current = []
                current.append(tensor)
            if current:
                read_spans.append(self._make_read_span(current))
            experts[(layer, expert)] = ExpertLayout(
                layer, expert, ordered, tuple(read_spans)
            )
            layers.setdefault(layer, []).append(expert)

        self._tensors = tensors
        self._shard_info = shard_info
        self._experts = experts
        self._layers = {layer: tuple(sorted(ids)) for layer, ids in layers.items()}
        self._payload_bytes = sum(span.byte_length for span in tensors.values())

    @staticmethod
    def _make_read_span(tensors: list[TensorSpan]) -> ReadSpan:
        return ReadSpan(
            tensors[0].shard,
            tensors[0].offset,
            tensors[-1].end - tensors[0].offset,
            tuple(tensor.name for tensor in tensors),
        )

    @staticmethod
    def _expert_suffix(name: str) -> str:
        match = _EXPERT_RE.match(name)
        if match is None:
            raise AssertionError(f"not an expert tensor: {name}")
        return match[3]

    def _get_fd(self, shard: str) -> int:
        with self._fd_lock:
            if self._closed:
                raise RuntimeError("LocalSafetensorsStore is closed")
            fd = self._fds.get(shard)
            if fd is None:
                fd = os.open(self.model_dir / shard, os.O_RDONLY)
                if self._darwin_nocache:
                    try:
                        fcntl.fcntl(fd, _DARWIN_F_NOCACHE, 1)
                    except BaseException:
                        os.close(fd)
                        raise
                self._fds[shard] = fd
            return fd

    def _read_span(self, shard: str, offset: int, length: int, label: str) -> bytes:
        fd = self._get_fd(shard)
        return _pread_exact(fd, length, offset, label)

    def _read_into(
        self, shard: str, offset: int, destination: memoryview, label: str
    ) -> int:
        fd = self._get_fd(shard)
        return _preadv_exact(fd, destination, offset, label)

    def _load_sidecar(self, sidecar_path: pathlib.Path) -> None:
        try:
            with gzip.open(sidecar_path, "rt", encoding="utf-8") as stream:
                sidecar = json.load(stream)
        except (OSError, json.JSONDecodeError) as exc:
            raise SidecarMismatchError(
                f"cannot read sidecar {sidecar_path}: {exc}"
            ) from exc
        if sidecar.get("schema") != SIDECAR_SCHEMA:
            raise SidecarMismatchError(
                f"unsupported sidecar schema {sidecar.get('schema')!r}"
            )
        try:
            index_bytes = self.index_path.read_bytes()
        except OSError as exc:
            raise SidecarMismatchError(f"cannot read {self.index_path}: {exc}") from exc
        self._index_sha256 = hashlib.sha256(index_bytes).hexdigest()
        if sidecar.get("index_sha256") != self._index_sha256:
            raise SidecarMismatchError("sidecar index hash does not match model index")

        shard_info: dict[str, tuple[int, int]] = {}
        try:
            for name, recorded_size, recorded_header_len in sidecar["shards"]:
                shard_path = self.model_dir / name
                size = shard_path.stat().st_size
                fd = os.open(shard_path, os.O_RDONLY)
                try:
                    actual_header_len = struct.unpack(
                        "<Q", _pread_exact(fd, 8, 0, name)
                    )[0]
                finally:
                    os.close(fd)
                if (size, actual_header_len) != (recorded_size, recorded_header_len):
                    raise SidecarMismatchError(
                        f"sidecar fingerprint mismatch for {name}: "
                        f"{(recorded_size, recorded_header_len)} != "
                        f"{(size, actual_header_len)}"
                    )
                shard_info[name] = (size, actual_header_len)
            tensors = {
                row[0]: TensorSpan(
                    row[0], row[1], int(row[2]), int(row[3]), row[4], tuple(row[5])
                )
                for row in sidecar["tensors"]
            }
        except (KeyError, TypeError, ValueError, OSError, struct.error) as exc:
            if isinstance(exc, SidecarMismatchError):
                raise
            raise SidecarMismatchError(
                f"malformed sidecar {sidecar_path}: {exc}"
            ) from exc

        for tensor in tensors.values():
            try:
                size, header_len = shard_info[tensor.shard]
            except KeyError as exc:
                raise SidecarMismatchError(
                    f"sidecar tensor {tensor.name!r} references unknown shard"
                ) from exc
            if (
                tensor.offset < 8 + header_len
                or tensor.end > size
                or tensor.byte_length != _shape_nbytes(tensor.dtype, tensor.shape)
            ):
                raise SidecarMismatchError(
                    f"sidecar tensor {tensor.name!r} has an invalid range or shape"
                )
        self._install_inventory(tensors, shard_info)
