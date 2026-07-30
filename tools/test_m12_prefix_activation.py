#!/usr/bin/env python3
"""Small tests for M12 prompt pairing and exact evidence comparison."""
from __future__ import annotations

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import validate_m12_prefix_activation as m12


class FakeTokenizer:
    def encode(self, text):
        if text.startswith("The capital"):
            return list(range(8))
        return [20, 21, 22, 23]


class M12ValidatorTests(unittest.TestCase):
    def test_smoke_pair_preserves_shape_and_changes_only_suffix(self):
        first, second, prefix = m12.prompt_pair(
            FakeTokenizer(), "smoke"
        )
        self.assertEqual(len(first), 8)
        self.assertEqual(len(second), 8)
        self.assertEqual(first[:4], second[:4])
        self.assertEqual(first[:4], prefix)
        self.assertNotEqual(first[4:], second[4:])

    def test_exact_comparison_requires_every_state_and_logit(self):
        prefill = torch.arange(6, dtype=torch.float32).view(1, 3, 2)
        decode = torch.tensor([[[7.0, 8.0]]])
        base = {
            "routes": {"0:1": [[1]], "1:1": [[2]]},
            "prefill_cache": {"sha": "p"},
            "next_token": 8,
            "decode_cache": {"sha": "d"},
        }
        artifacts = {
            "prefill_logits": prefill,
            "decode_logits": decode,
        }
        result = m12.compare_exact(base, artifacts, base, artifacts)
        self.assertTrue(result["passed"])
        changed = dict(base)
        changed["prefill_cache"] = {"sha": "changed"}
        with self.assertRaisesRegex(AssertionError, "parity failed"):
            m12.compare_exact(base, artifacts, changed, artifacts)


if __name__ == "__main__":
    unittest.main()
