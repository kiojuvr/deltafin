#!/usr/bin/env python3
"""Unit tests for shape-stable prefix MoE activation reuse."""
from __future__ import annotations

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prefix_activation import PrefixActivationCache, PrefixActivationSession


class PrefixActivationTests(unittest.TestCase):
    def test_capture_replay_preserves_full_shape_and_prefix_bits(self):
        session = PrefixActivationSession(2, 4)
        x = torch.arange(24, dtype=torch.float32).view(4, 6)
        rows = [[1, 2], [3, 4], [4, 5], [6, 7]]
        weights = [[0.4, 0.6], [0.3, 0.7], [0.2, 0.8], [0.1, 0.9]]
        self.assertEqual(session.prepare(1, x, rows, weights), 0)
        captured = torch.arange(24, dtype=torch.float32).view(4, 6) + 100
        self.assertIs(session.finish(1, captured), captured)
        session.arm_replay(expected_layers=1)

        self.assertEqual(session.prepare(1, x, rows, weights), 2)
        session.record_demand(
            {1, 2, 3, 4, 5, 6, 7},
            {4, 5, 6, 7},
            route_edges_skipped=4,
        )
        computed = torch.zeros_like(captured)
        computed[2:] = captured[2:] + 50
        replayed = session.finish(1, computed)
        self.assertTrue(torch.equal(replayed[:2], captured[:2]))
        self.assertTrue(torch.equal(replayed[2:], computed[2:]))
        session.finish_replay(expected_layers=1)
        summary = session.snapshot()
        self.assertEqual(summary["owned_bytes"], 2 * 2 * 6 * 4)
        self.assertEqual(summary["skipped_unique_experts"], 3)
        self.assertEqual(summary["skipped_route_edges"], 4)

    def test_replay_fails_closed_on_input_route_or_weight_change(self):
        x = torch.arange(12, dtype=torch.float32).view(3, 4)
        rows = [[1], [2], [3]]
        weights = [[0.5], [0.6], [0.7]]

        def captured():
            session = PrefixActivationSession(1, 3)
            session.prepare(0, x, rows, weights)
            session.finish(0, x)
            session.arm_replay(expected_layers=1)
            return session

        with self.assertRaisesRegex(RuntimeError, "latent input changed"):
            captured().prepare(0, x + 1, rows, weights)
        changed_routes = [[9], [2], [3]]
        with self.assertRaisesRegex(RuntimeError, "routes changed"):
            captured().prepare(0, x, changed_routes, weights)
        changed_weights = [[0.4], [0.6], [0.7]]
        with self.assertRaisesRegex(RuntimeError, "weights changed"):
            captured().prepare(0, x, rows, changed_weights)

    def test_invalid_shape_and_lifecycle_are_rejected(self):
        with self.assertRaises(ValueError):
            PrefixActivationSession(3, 3)
        session = PrefixActivationSession(1, 2)
        with self.assertRaisesRegex(ValueError, "latent input"):
            session.prepare(
                0,
                torch.zeros(1, 4),
                [[1], [2]],
                [[0.5], [0.5]],
            )
        with self.assertRaisesRegex(RuntimeError, "captured 0/1"):
            session.arm_replay(expected_layers=1)

    def test_replay_session_can_be_used_repeatedly(self):
        session = PrefixActivationSession(1, 2)
        x = torch.arange(8, dtype=torch.float32).view(2, 4)
        rows = [[1], [2]]
        weights = [[0.5], [0.6]]
        session.prepare(0, x, rows, weights)
        session.finish(0, x + 10)
        session.arm_replay(expected_layers=1)
        for _ in range(2):
            session.begin_replay()
            self.assertEqual(session.prepare(0, x, rows, weights), 1)
            session.finish(0, torch.zeros_like(x))
            session.finish_replay(expected_layers=1)

    def test_shape_cache_is_bounded_and_only_replays_exact_shape(self):
        cache = PrefixActivationCache([10, 11], max_entries=1)

        def complete_capture(total):
            ids = [10, 11] + list(range(total - 2))
            session, plan = cache.begin("chat", ids)
            self.assertEqual(plan["action"], "capture")
            x = torch.zeros(total, 4)
            rows = [[1] for _ in range(total)]
            weights = [[0.5] for _ in range(total)]
            session.prepare(0, x, rows, weights)
            session.finish(0, x)
            cache.complete(session, plan, expected_layers=1)
            return ids

        first = complete_capture(4)
        session, hit = cache.begin("chat", first)
        self.assertEqual(hit["action"], "replay")
        session.prepare(
            0,
            torch.zeros(4, 4),
            [[1]] * 4,
            [[0.5]] * 4,
        )
        session.finish(0, torch.zeros(4, 4))
        cache.complete(session, hit, expected_layers=1)
        self.assertEqual(cache.hits, 1)

        complete_capture(5)
        self.assertEqual(cache.evictions, 1)
        _session, miss = cache.begin("chat", first)
        self.assertEqual(miss["action"], "capture")

    def test_shape_cache_rejects_nonchat_and_wrong_prefix(self):
        cache = PrefixActivationCache([10, 11])
        for mode, ids in (
            ("completion", [10, 11, 12]),
            ("chat", [10, 99, 12]),
            ("chat", [10, 11]),
        ):
            session, plan = cache.begin(mode, ids)
            self.assertIsNone(session)
            self.assertFalse(plan["eligible"])

    def test_failed_replay_is_invalidated_for_future_capture(self):
        cache = PrefixActivationCache([10], max_entries=1)
        session, plan = cache.begin("chat", [10, 20])
        x = torch.zeros(2, 4)
        session.prepare(0, x, [[1], [2]], [[0.5], [0.6]])
        session.finish(0, x)
        cache.complete(session, plan, expected_layers=1)

        replay, replay_plan = cache.begin("chat", [10, 30])
        self.assertEqual(replay_plan["action"], "replay")
        cache.abort(replay, replay_plan)
        replacement, replacement_plan = cache.begin("chat", [10, 40])
        self.assertEqual(replacement_plan["action"], "capture")
        self.assertIsNot(replacement, replay)


if __name__ == "__main__":
    unittest.main()
