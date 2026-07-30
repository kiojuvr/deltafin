#!/usr/bin/env python3
"""Tests for M13 request-shape and I/O aggregation."""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m13_prefix_workload import reuse_distances, simulate_lru, summarize


def record(shape, action, *, logical=100, physical=25, avoided=0):
    return {
        "status": "ok",
        "memo_hit": False,
        "duration_ns": 2_000_000_000,
        "ttft_ns": 1_000_000_000,
        "logical_expert_bytes": logical,
        "physical_member_read_bytes": physical,
        "inferred_page_cache_bytes": logical - physical,
        "prefix_activation_avoided_expert_bytes": avoided,
        "prefix_activation_skipped_route_edges": (
            108_928 if action == "replay" else 0
        ),
        "prefix_activation": {
            "eligible": True,
            "action": action,
            "total_positions": shape,
            "cache_after": {
                "resident_shapes_lru": [shape],
                "owned_bytes": 10,
            },
        },
    }


class M13PrefixWorkloadTests(unittest.TestCase):
    def test_lru_trace_exposes_capacity_crossover(self):
        shapes = [89, 90, 89, 91, 90, 91]
        one = simulate_lru(shapes, 1)
        two = simulate_lru(shapes, 2)
        self.assertEqual(one["hits"], 0)
        self.assertEqual(two["hits"], 2)
        self.assertEqual(two["evictions"], 2)
        self.assertEqual(two["compulsory_misses"], 3)
        self.assertEqual(two["eviction_misses"], 1)
        self.assertEqual(two["resident_shapes_lru"], [90, 91])

    def test_reuse_distance_is_distinct_shape_stack_distance(self):
        self.assertEqual(
            reuse_distances([89, 90, 89, 91, 90, 90]),
            [None, None, 1, None, 2, 0],
        )

    def test_summary_groups_shapes_and_io(self):
        rows = [
            record(89, "capture"),
            record(89, "replay", logical=40, physical=10, avoided=60),
            record(90, "capture"),
        ]
        report = summarize(rows, max_simulated_capacity=2)
        self.assertEqual(report["captures"], 2)
        self.assertEqual(report["replays"], 1)
        self.assertAlmostEqual(report["hit_rate"], 1 / 3)
        self.assertEqual(report["by_shape"]["89"]["replays"], 1)
        self.assertEqual(
            report["phases"]["replay"]["avoided_expert_bytes"], 60
        )
        self.assertEqual(
            report["phases"]["replay"]["physical_fraction"], 0.25
        )

    def test_memo_and_ineligible_do_not_distort_hit_rate(self):
        memo = record(89, "replay")
        memo["memo_hit"] = True
        ineligible = record(0, "capture")
        ineligible["prefix_activation"] = {"eligible": False}
        report = summarize([memo, ineligible])
        self.assertEqual(report["eligible_requests"], 0)
        self.assertIsNone(report["hit_rate"])
        self.assertEqual(report["phases"]["memo"]["requests"], 1)
        self.assertEqual(report["phases"]["ineligible"]["requests"], 1)

    def test_invalid_eligible_action_is_rejected(self):
        broken = record(89, "capture")
        broken["prefix_activation"]["action"] = "none"
        with self.assertRaisesRegex(ValueError, "invalid activation action"):
            summarize([broken])


if __name__ == "__main__":
    unittest.main()
