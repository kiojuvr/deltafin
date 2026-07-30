"""Exact immutable cache snapshots for K3's deterministic chat prefix.

The snapshot owns only references to KV/KDA/short-convolution state tensors.
K3 updates those tensors functionally during inference, so restoring a snapshot
into a fresh cache is constant-time and the retained prefix remains immutable.
No model weight or expert data is copied into this component.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch


CACHE_ATTRIBUTES = (
    "recurrent_states",
    "conv_states",
    "key_cache",
    "value_cache",
)


def _as_ids(values: Iterable[int]) -> tuple[int, ...]:
    return tuple(int(value) for value in values)


def longest_common_prefix(
    first: Iterable[int], second: Iterable[int]
) -> tuple[int, ...]:
    left = _as_ids(first)
    right = _as_ids(second)
    length = 0
    for one, two in zip(left, right):
        if one != two:
            break
        length += 1
    return left[:length]


def derive_chat_prefix(tokenizer, *, minimum_tokens: int = 32) -> tuple[int, ...]:
    """Derive only the input-independent prefix of K3's official template."""
    rendered = []
    for content in ("Hello", "Different words entirely"):
        rendered.append(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=True,
                add_generation_prompt=True,
            )
        )
    prefix = longest_common_prefix(*rendered)
    if len(prefix) < minimum_tokens:
        raise RuntimeError(
            "official chat template has no sufficiently long fixed prefix: "
            f"{len(prefix)} < {minimum_tokens}"
        )
    if len(prefix) >= min(len(sequence) for sequence in rendered):
        raise RuntimeError("chat prefix derivation consumed a complete prompt")
    if any(_as_ids(sequence[: len(prefix)]) != prefix for sequence in rendered):
        raise AssertionError("derived chat prefix does not match its templates")
    return prefix


def _parts(value):
    if value is None:
        return ()
    return value if isinstance(value, tuple) else (value,)


def _tensor_signature(tensor: torch.Tensor) -> tuple[Any, ...]:
    storage = tensor.untyped_storage()
    return (
        storage.data_ptr(),
        storage.nbytes(),
        tensor.data_ptr(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        str(tensor.dtype),
        str(tensor.device),
    )


@dataclass(frozen=True)
class PrefixStateDescription:
    prefix_tokens: int
    tensors: int
    logical_bytes: int
    storage_bytes: int
    layers: int

    def as_dict(self) -> dict[str, int]:
        return {
            "prefix_tokens": self.prefix_tokens,
            "tensors": self.tensors,
            "logical_bytes": self.logical_bytes,
            "storage_bytes": self.storage_bytes,
            "layers": self.layers,
        }


class PrefixStateSnapshot:
    """Immutable references to one exact token prefix's recurrent state."""

    def __init__(
        self,
        prefix_ids: Iterable[int],
        *,
        recurrent_states,
        conv_states,
        key_cache,
        value_cache,
    ):
        self.prefix_ids = _as_ids(prefix_ids)
        if not self.prefix_ids:
            raise ValueError("prefix state requires at least one token")
        rows = {
            "recurrent_states": tuple(recurrent_states),
            "conv_states": tuple(conv_states),
            "key_cache": tuple(key_cache),
            "value_cache": tuple(value_cache),
        }
        lengths = {len(value) for value in rows.values()}
        if len(lengths) != 1:
            raise ValueError(f"cache attribute lengths differ: {lengths}")
        self._rows = rows
        tensors = [
            tensor
            for values in rows.values()
            for value in values
            for tensor in _parts(value)
        ]
        if not tensors:
            raise ValueError("prefix cache contains no state tensors")
        self._signatures = tuple(_tensor_signature(tensor) for tensor in tensors)
        self._logical_bytes = sum(
            tensor.numel() * tensor.element_size() for tensor in tensors
        )
        storages = {}
        for tensor in tensors:
            storage = tensor.untyped_storage()
            storages.setdefault(storage.data_ptr(), storage.nbytes())
        self._storage_bytes = sum(storages.values())
        self._tensor_count = len(tensors)
        self.restores = 0

    @classmethod
    def capture(cls, cache, prefix_ids: Iterable[int]):
        prefix = _as_ids(prefix_ids)
        sequence_length = int(cache.get_seq_length() or 0)
        if sequence_length != len(prefix):
            raise ValueError(
                "cache sequence length does not match prefix: "
                f"{sequence_length} != {len(prefix)}"
            )
        return cls(
            prefix,
            **{
                attribute: getattr(cache, attribute)
                for attribute in CACHE_ATTRIBUTES
            },
        )

    def matches(self, token_ids: Iterable[int]) -> bool:
        values = _as_ids(token_ids)
        return (
            len(values) > len(self.prefix_ids)
            and values[: len(self.prefix_ids)] == self.prefix_ids
        )

    def suffix(self, token_ids: Iterable[int]) -> tuple[int, ...]:
        values = _as_ids(token_ids)
        if not self.matches(values):
            raise ValueError("prompt does not begin with the exact cached prefix")
        return values[len(self.prefix_ids) :]

    def assert_intact(self) -> None:
        signatures = tuple(
            _tensor_signature(tensor)
            for values in self._rows.values()
            for value in values
            for tensor in _parts(value)
        )
        if signatures != self._signatures:
            raise RuntimeError("prefix snapshot tensor ownership changed")

    def restore(self, cache) -> int:
        """Install the immutable prefix references into a fresh cache."""
        self.assert_intact()
        occupied = {
            attribute: sum(
                value is not None for value in getattr(cache, attribute)
            )
            for attribute in CACHE_ATTRIBUTES
        }
        if any(occupied.values()):
            raise ValueError(
                f"prefix snapshot restore requires a fresh cache: {occupied}"
            )
        for attribute, values in self._rows.items():
            setattr(cache, attribute, list(values))
        sequence_length = int(cache.get_seq_length() or 0)
        if sequence_length != len(self.prefix_ids):
            raise RuntimeError(
                "restored cache sequence length differs from prefix: "
                f"{sequence_length} != {len(self.prefix_ids)}"
            )
        self.restores += 1
        return sequence_length

    def description(self) -> PrefixStateDescription:
        return PrefixStateDescription(
            prefix_tokens=len(self.prefix_ids),
            tensors=self._tensor_count,
            logical_bytes=self._logical_bytes,
            storage_bytes=self._storage_bytes,
            layers=len(self._rows["key_cache"]),
        )

    def snapshot(self) -> dict[str, Any]:
        result = self.description().as_dict()
        result.update(
            {
                "restores": self.restores,
                "prefix_head": list(self.prefix_ids[:8]),
                "prefix_tail": list(self.prefix_ids[-8:]),
            }
        )
        return result
