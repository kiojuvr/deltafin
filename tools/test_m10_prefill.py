#!/usr/bin/env python3
"""Small tests for M10 multi-token prefill evidence helpers."""

from __future__ import annotations

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import validate_m10_prefill as m10


class FakeTokenizer:
    def encode(self, text):
        if text == "The Chrysler":
            return [1, 2]
        return list(range(9))

    def apply_chat_template(self, *args, **kwargs):
        return list(range(17))


class M10PrefillTests(unittest.TestCase):
    def test_prompt_cases_have_fixed_focused_lengths(self):
        cases = m10.prompt_cases(FakeTokenizer(), "hello")
        self.assertEqual(len(cases["two"]), 2)
        self.assertEqual(len(cases["eight"]), 8)
        self.assertEqual(len(cases["chat"]), 17)

    def test_cache_digest_distinguishes_state_bytes(self):
        class Cache:
            recurrent_states = [torch.arange(4, dtype=torch.float32)]
            conv_states = [(torch.ones(2), torch.zeros(2))]
            key_cache = [None]
            value_cache = [None]

        first = m10.cache_digest(Cache())
        second = m10.cache_digest(Cache())
        self.assertEqual(first, second)
        Cache.recurrent_states = [torch.arange(4, dtype=torch.float32) + 1]
        third = m10.cache_digest(Cache())
        self.assertNotEqual(first, third)

    def test_route_observer_preserves_position_order(self):
        routes = {}
        observe = m10.route_observer(routes)
        observe(0, 4, ((9, 2), (1, 8)))
        self.assertEqual(routes["0:4"], [[9, 2], [1, 8]])

    def test_route_summary_reports_union_and_edge_bounds(self):
        routes = {
            "0:1": [[1, 2], [2, 3]],
            "0:2": [[4, 5], [4, 5]],
            "1:1": [[8, 9]],
        }
        summary = m10.route_summary(routes)
        self.assertEqual(summary["0"]["union_experts_total"], 5)
        self.assertEqual(summary["0"]["union_experts_max"], 3)
        self.assertEqual(summary["0"]["route_edges"], 8)
        self.assertEqual(summary["1"]["union_experts_min"], 2)

    def test_compare_modes_requires_cache_and_logit_exactness(self):
        evidence = {
            "next_token": 4,
            "routes": {"0:1": [[2]]},
            "prefill_cache": {"sha": "a"},
            "decode_cache": {"sha": "b"},
        }
        artifacts = {
            "prefill_logits": torch.tensor([1.0]),
            "decode_logits": torch.tensor([2.0]),
        }
        result = m10.compare_modes(
            evidence, evidence, artifacts, artifacts
        )
        self.assertTrue(result["passed"])
        changed = dict(evidence)
        changed["decode_cache"] = {"sha": "different"}
        with self.assertRaisesRegex(AssertionError, "parity failed"):
            m10.compare_modes(evidence, changed, artifacts, artifacts)


if __name__ == "__main__":
    unittest.main()
