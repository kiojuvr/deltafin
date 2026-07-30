#!/usr/bin/env python3
"""Unit tests for exact reusable K3 prefix state."""
from __future__ import annotations

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prefix_state import (
    PrefixStateSnapshot,
    derive_chat_prefix,
    longest_common_prefix,
)


class FakeTokenizer:
    def apply_chat_template(self, messages, **_kwargs):
        content = messages[0]["content"]
        body = [11] if content == "Hello" else [22, 23]
        return [1, 2, 3, 4] + body + [8, 9]


class FakeCache:
    def __init__(self):
        self.recurrent_states = [None, None]
        self.conv_states = [None, None]
        self.key_cache = [None, None]
        self.value_cache = [None, None]

    def get_seq_length(self):
        for value in self.key_cache:
            if value is not None:
                return value.shape[2]
        return 0


def populated_cache():
    cache = FakeCache()
    cache.recurrent_states[0] = torch.arange(4, dtype=torch.float32)
    cache.conv_states[0] = (
        torch.ones(2),
        torch.ones(2) * 2,
        torch.ones(2) * 3,
    )
    cache.key_cache[1] = torch.arange(6, dtype=torch.float32).view(1, 1, 3, 2)
    cache.value_cache[1] = torch.arange(9, dtype=torch.float32).view(1, 1, 3, 3)
    return cache


class PrefixStateTests(unittest.TestCase):
    def test_longest_common_prefix_and_template_derivation(self):
        self.assertEqual(longest_common_prefix([1, 2, 3], [1, 2, 4]), (1, 2))
        self.assertEqual(
            derive_chat_prefix(FakeTokenizer(), minimum_tokens=4),
            (1, 2, 3, 4),
        )

    def test_capture_restore_is_shallow_exact_and_bounded(self):
        source = populated_cache()
        snapshot = PrefixStateSnapshot.capture(source, [5, 6, 7])
        description = snapshot.description()
        self.assertEqual(description.prefix_tokens, 3)
        self.assertEqual(description.tensors, 6)
        self.assertGreater(description.logical_bytes, 0)
        self.assertLessEqual(description.storage_bytes, description.logical_bytes)

        restored = FakeCache()
        self.assertEqual(snapshot.restore(restored), 3)
        self.assertEqual(snapshot.restores, 1)
        self.assertEqual(
            restored.key_cache[1].data_ptr(),
            source.key_cache[1].data_ptr(),
        )
        self.assertTrue(snapshot.matches([5, 6, 7, 8]))
        self.assertEqual(snapshot.suffix([5, 6, 7, 8, 9]), (8, 9))

        # K3 inference replaces state tensors instead of modifying the retained
        # prefix. A later restore therefore recovers the original references.
        restored.recurrent_states[0] = restored.recurrent_states[0] + 10
        snapshot.assert_intact()
        again = FakeCache()
        snapshot.restore(again)
        self.assertTrue(
            torch.equal(again.recurrent_states[0], torch.arange(4).float())
        )

    def test_restore_and_match_fail_closed(self):
        snapshot = PrefixStateSnapshot.capture(populated_cache(), [5, 6, 7])
        with self.assertRaisesRegex(ValueError, "exact cached prefix"):
            snapshot.suffix([5, 6, 8, 9])
        occupied = populated_cache()
        with self.assertRaisesRegex(ValueError, "fresh cache"):
            snapshot.restore(occupied)
        wrong_length = populated_cache()
        with self.assertRaisesRegex(ValueError, "sequence length"):
            PrefixStateSnapshot.capture(wrong_length, [1, 2])


if __name__ == "__main__":
    unittest.main()
