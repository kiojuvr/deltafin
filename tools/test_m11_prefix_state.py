#!/usr/bin/env python3
"""Small tests for M11 validation and prefix partitioning."""
from __future__ import annotations

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import validate_m11_prefix_state as m11


class FakeTokenizer:
    def encode(self, _text):
        return list(range(10))

    def apply_chat_template(self, messages, **_kwargs):
        content = messages[0]["content"]
        body = [7] if content == "Hello" else [8, 9]
        return [1, 2, 3, 4] + body + [10]


class M11ValidatorTests(unittest.TestCase):
    def test_smoke_partition_is_bounded(self):
        prompt, prefix = m11.prompt_and_prefix(
            FakeTokenizer(), "smoke", "ignored"
        )
        self.assertEqual(len(prompt), 8)
        self.assertEqual(len(prefix), 4)
        self.assertEqual(prompt[:4], prefix)

    def test_route_combination_preserves_position_order(self):
        prefix = {"0:1": [[1], [2]], "0:2": [[3], [4]]}
        suffix = {
            "0:1": [[5]],
            "0:2": [[6]],
            "1:1": [[7]],
            "1:2": [[8]],
        }
        self.assertEqual(
            m11.combine_routes(prefix, suffix),
            {
                "0:1": [[1], [2], [5]],
                "0:2": [[3], [4], [6]],
            },
        )

    def test_compare_requires_routes_logits_and_cache_exactness(self):
        full_logits = torch.arange(6, dtype=torch.float32).view(1, 3, 2)
        decode_logits = torch.tensor([[[9.0, 8.0]]])
        segmented = {
            "routes": {"0:1": [[3]], "1:1": [[4]]},
            "next_token": 9,
            "prefill_cache": {"sha": "p"},
            "decode_cache": {"sha": "d"},
        }
        reused = {
            "routes": {"0:1": [[3]], "1:1": [[4]]},
            "next_token": 9,
            "prefill_cache": {"sha": "p"},
            "decode_cache": {"sha": "d"},
        }
        result = m11.compare_exact(
            segmented,
            {
                "suffix_logits": full_logits[:, 2:],
                "decode_logits": decode_logits,
            },
            reused,
            {
                "suffix_logits": full_logits[:, 2:],
                "decode_logits": decode_logits,
            },
        )
        self.assertTrue(result["passed"])
        reused["decode_cache"] = {"sha": "changed"}
        with self.assertRaisesRegex(AssertionError, "parity failed"):
            m11.compare_exact(
                segmented,
                {
                    "suffix_logits": full_logits[:, 2:],
                    "decode_logits": decode_logits,
                },
                reused,
                {
                    "suffix_logits": full_logits[:, 2:],
                    "decode_logits": decode_logits,
                },
            )


if __name__ == "__main__":
    unittest.main()
