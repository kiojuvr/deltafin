#!/usr/bin/env python3
"""Tests for the direct-shard adaptive route policy."""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adaptive_prefetch import AdaptiveRoutePrefetch  # noqa: E402


class AdaptiveRoutePrefetchTests(unittest.TestCase):
    def policy(self, **overrides):
        options = {
            "max_experts_per_layer": 4,
            "expert_bytes": 10,
            "token_budget_bytes": 80,
            "warmup_observations": 4,
            "min_wilson_precision": 0.3,
        }
        options.update(overrides)
        return AdaptiveRoutePrefetch(**options)

    def test_requires_warmup_then_uses_conservative_prefix(self):
        policy = self.policy()
        previous = {layer: (10, 11, 12, 13) for layer in range(1, 6)}
        policy.begin_pass(previous)
        for layer in range(1, 5):
            self.assertEqual(policy.predict(layer), ())
            policy.observe(layer, {10, 11})
        self.assertEqual(policy.eligible_prefix(), 2)
        self.assertEqual(policy.predict(5), (10, 11))
        policy.observe(5, {10})
        snapshot = policy.snapshot()
        self.assertEqual(snapshot["predicted_experts"], 2)
        self.assertEqual(snapshot["predicted_hits"], 1)

    def test_token_byte_budget_is_hard_limit(self):
        policy = self.policy(
            token_budget_bytes=30,
            warmup_observations=1,
            min_wilson_precision=0.0,
        )
        policy.begin_pass({1: (1, 2, 3, 4)})
        policy.observe(1, {1, 2, 3, 4})
        policy.begin_pass({1: (1, 2, 3, 4), 2: (5, 6, 7, 8)})
        self.assertEqual(policy.predict(1), (1, 2, 3))
        self.assertEqual(policy.predict(2), ())
        self.assertEqual(
            policy.snapshot()["current_token_prefetch_bytes"], 30
        )

    def test_reset_removes_training_and_accounting(self):
        policy = self.policy(
            warmup_observations=1, min_wilson_precision=0.0
        )
        policy.begin_pass({1: (1, 2, 3, 4)})
        policy.observe(1, {1, 2})
        policy.begin_pass({1: (1, 2, 3, 4)})
        self.assertTrue(policy.predict(1))
        policy.reset()
        snapshot = policy.snapshot()
        self.assertEqual(snapshot["passes"], 0)
        self.assertEqual(snapshot["predicted_experts"], 0)
        self.assertEqual(snapshot["eligible_prefix"], 0)

    def test_io_gate_skips_without_spending_budget(self):
        policy = self.policy(
            warmup_observations=1, min_wilson_precision=0.0
        )
        policy.begin_pass({1: (1, 2, 3, 4)})
        policy.observe(1, {1, 2, 3, 4})
        policy.begin_pass(
            {1: (1, 2, 3, 4), 2: (5, 6, 7, 8), 3: (9, 10, 11, 12)}
        )
        self.assertEqual(policy.predict(1, gate="no-signal"), ())
        self.assertEqual(policy.predict(2, gate="closed"), ())
        self.assertEqual(policy.predict(3, gate="open"), (9, 10, 11, 12))
        snapshot = policy.snapshot()
        self.assertEqual(snapshot["gate_no_signal_layers"], 1)
        self.assertEqual(snapshot["gate_closed_layers"], 2)
        self.assertEqual(snapshot["gate_open_layers"], 1)
        self.assertEqual(snapshot["gate_skipped_experts"], 8)
        self.assertEqual(snapshot["current_token_prefetch_bytes"], 40)


if __name__ == "__main__":
    unittest.main()
