#!/usr/bin/env python3
"""Tests for privacy-preserving M14 prompt-shape analysis."""
from __future__ import annotations

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m14_shape_trace import (  # noqa: E402
    analyze_shapes,
    calibration_from_records,
    trace_from_shape_records,
    trace_requests,
)


class FakeTokenizer:
    def apply_chat_template(self, messages, **_kwargs):
        words = str(messages[0]["content"]).split()
        return [10, 11] + list(range(100, 100 + len(words)))

    def encode(self, prompt):
        return list(range(len(prompt.split())))


class M14ShapeTraceTests(unittest.TestCase):
    def test_trace_discards_content_and_token_ids(self):
        body = {
            "messages": [{"role": "user", "content": "secret value"}]
        }
        trace = trace_requests(FakeTokenizer(), [body], [10, 11])
        encoded = json.dumps(trace)
        self.assertNotIn("secret", encoded)
        self.assertNotIn("token_ids", encoded)
        self.assertTrue(trace[0]["eligible"])
        self.assertEqual(trace[0]["total_positions"], 4)

    def test_analysis_separates_compulsory_and_eviction_misses(self):
        trace = [
            {"eligible": True, "total_positions": shape}
            for shape in (89, 90, 89, 91, 90, 91)
        ]
        report = analyze_shapes(trace, max_capacity=2)
        self.assertEqual(report["compulsory_misses"], 3)
        self.assertEqual(report["repeat_requests"], 3)
        self.assertEqual(
            report["reuse_distance_histogram"],
            {"first": 3, "1": 2, "2": 1},
        )
        self.assertEqual(
            report["capacity_sweep"]["2"]["eviction_misses"], 1
        )
        self.assertIn("2", report["repeat_admission_sweep"])
        self.assertEqual(report["capacity_for_all_observed_reuse"], 3)

    def test_calibration_and_projection_are_explicitly_coarse(self):
        records = []
        for action, ttft, logical, physical in (
            ("capture", 10, 100, 80),
            ("replay", 2, 30, 10),
        ):
            records.append(
                {
                    "status": "ok",
                    "memo_hit": False,
                    "ttft_ns": ttft,
                    "logical_expert_bytes": logical,
                    "physical_member_read_bytes": physical,
                    "prefix_activation": {"action": action},
                }
            )
        calibration = calibration_from_records(records)
        trace = [
            {"eligible": True, "total_positions": shape}
            for shape in (89, 89)
        ]
        report = analyze_shapes(
            trace, max_capacity=1, calibration=calibration
        )
        projection = report["capacity_sweep"]["1"]["projection"]
        self.assertTrue(projection["coarse_calibration_only"])
        self.assertEqual(projection["ttft_seconds"], 12 / 1e9)
        self.assertEqual(projection["logical_expert_bytes"], 130)

    def test_completion_is_recorded_but_ineligible(self):
        trace = trace_requests(
            FakeTokenizer(), [{"prompt": "one two"}], [10, 11]
        )
        self.assertEqual(trace[0]["mode"], "completion")
        self.assertFalse(trace[0]["eligible"])

    def test_server_shape_records_need_no_prompt_or_tokenizer(self):
        trace = trace_from_shape_records(
            [
                {
                    "schema": "deltafin.request-shape.v1",
                    "mode": "chat",
                    "total_positions": 89,
                    "prefix_tokens": 74,
                    "eligible": True,
                }
            ]
        )
        self.assertEqual(trace[0]["total_positions"], 89)
        self.assertTrue(trace[0]["eligible"])

    def test_response_memo_hits_are_excluded_from_activation_capacity(self):
        trace = [
            {
                "eligible": True,
                "total_positions": 89,
                "memo_hit": False,
            },
            {
                "eligible": True,
                "total_positions": 89,
                "memo_hit": True,
            },
        ]
        report = analyze_shapes(trace, max_capacity=1)
        self.assertEqual(report["eligible_requests"], 1)
        self.assertEqual(report["memo_requests_excluded"], 1)
        self.assertEqual(report["capacity_sweep"]["1"]["hits"], 0)


if __name__ == "__main__":
    unittest.main()
