#!/usr/bin/env python3
"""Small tests for M9 long-lived request accounting."""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import validate_m9_long_lived as m9


def token_row(step, routes, *, seconds=2.0, physical=100, logical=1000):
    return {
        "step": step,
        "seconds": seconds,
        "routes": routes,
        "physical_member_read_bytes": physical,
        "slab_transfer_bytes": logical,
    }


def snapshot(value=0):
    return {
        "vm": {
            "external_bytes": value,
            "free_bytes": value,
            "compressions": value,
            "compressor_pages_bytes": value,
            "pageins": value,
            "swapouts": value,
        }
    }


class M9LongLivedTests(unittest.TestCase):
    def test_default_workload_repeats_anchor_across_unrelated_requests(self):
        requests = m9.parse_requests(None)
        self.assertEqual(len(requests), 5)
        self.assertEqual(requests[0]["token_id"], requests[2]["token_id"])
        self.assertNotEqual(requests[0]["token_id"], requests[1]["token_id"])

    def test_request_parser_rejects_ambiguous_workloads(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            m9.parse_requests(["same:1", "same:1", "third:2"])
        with self.assertRaisesRegex(ValueError, "repeated"):
            m9.parse_requests(["a:1", "b:2", "c:3"])
        with self.assertRaisesRegex(ValueError, "LABEL:TOKEN_ID"):
            m9.parse_requests(["broken", "b:2", "c:2"])

    def test_summary_separates_new_reused_and_previous_experts(self):
        sequence = {
            "before": snapshot(10),
            "after": snapshot(20),
            "tokens": [
                token_row(0, {"1": [1, 2]}),
                token_row(1, {"1": [2, 3]}),
            ],
        }
        seen = {(1, 1), (2, 9)}
        previous = {(1, 2), (3, 8)}
        summary, routed = m9.summarize_request(
            sequence, seen=seen, previous=previous
        )
        self.assertEqual(routed, {(1, 1), (1, 2), (1, 3)})
        self.assertEqual(summary["new_process_experts"], 2)
        self.assertEqual(summary["reused_process_experts"], 1)
        self.assertEqual(summary["previous_request_overlap_experts"], 1)
        self.assertEqual(summary["ttft_seconds"], 2.0)
        self.assertEqual(summary["physical_bytes_per_token"], 100)
        self.assertEqual(summary["compressions_delta"], 10)

    def test_aggregate_keeps_request_and_token_denominators(self):
        requests = [
            {
                "summary": {
                    "tokens": 2,
                    "wall_seconds": 6.0,
                    "logical_expert_bytes": 2000,
                    "physical_member_read_bytes": 400,
                    "compressions_delta": 0,
                    "swapouts_delta": 0,
                }
            },
            {
                "summary": {
                    "tokens": 2,
                    "wall_seconds": 2.0,
                    "logical_expert_bytes": 2000,
                    "physical_member_read_bytes": 0,
                    "compressions_delta": 0,
                    "swapouts_delta": 0,
                }
            },
        ]
        aggregate = m9.aggregate_requests(requests)
        self.assertEqual(aggregate["requests"], 2)
        self.assertEqual(aggregate["tokens"], 4)
        self.assertEqual(aggregate["seconds_per_token"], 2.0)
        self.assertEqual(aggregate["physical_bytes_per_token"], 100)
        self.assertEqual(aggregate["physical_fraction"], 0.1)


if __name__ == "__main__":
    unittest.main()
