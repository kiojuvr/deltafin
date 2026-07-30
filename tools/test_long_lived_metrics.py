#!/usr/bin/env python3
"""Tests for optional long-lived server request metrics."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from long_lived_metrics import LongLivedRequestMetrics


class LongLivedRequestMetricsTests(unittest.TestCase):
    def test_request_records_ttft_io_and_process_reuse(self):
        disk = iter((10.0, 14.0, 16.0, 16.0, 17.0))
        current_routes = {"1": [2, 3]}
        current_stats = {"pread_bytes": 0, "slab_loads": 0}

        def snapshot(label):
            return {
                "label": label,
                "disk": {"member_megabytes_total": next(disk)},
            }

        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "long_lived_metrics.time.perf_counter_ns",
            side_effect=(100, 200, 300, 400, 500),
        ):
            path = Path(temporary) / "requests.jsonl"
            metrics = LongLivedRequestMetrics(
                path,
                snapshot=snapshot,
                stats=lambda: dict(current_stats),
                routes=lambda: current_routes,
                expert_bytes=10,
            )
            first = metrics.begin(
                request_id="one",
                mode="completion",
                input_tokens=1,
                max_new_tokens=2,
                memo_hit=False,
                prefix_state={
                    "eligible": True,
                    "hit": True,
                    "prefix_tokens": 74,
                },
            )
            current_stats["pread_bytes"] = 20
            metrics.observe_token(first)
            record = metrics.finish(
                first, status="ok", output_tokens=2
            )
            self.assertEqual(record["ttft_ns"], 100)
            self.assertEqual(
                record["ttft_physical_member_read_bytes"], 4 * 1024 * 1024
            )
            self.assertEqual(
                record["physical_member_read_bytes"], 6 * 1024 * 1024
            )
            self.assertEqual(record["new_process_experts"], 2)
            self.assertEqual(record["direct_stats"]["pread_bytes"], 20)
            self.assertTrue(record["prefix_state"]["hit"])
            self.assertEqual(record["prefix_state"]["prefix_tokens"], 74)

            second = metrics.begin(
                request_id="two",
                mode="completion",
                input_tokens=1,
                max_new_tokens=1,
                memo_hit=True,
            )
            second_record = metrics.finish(
                second, status="memo-hit", output_tokens=1
            )
            self.assertIsNone(second_record["ttft_ns"])
            self.assertEqual(second_record["unique_routed_experts"], 0)
            self.assertIsNone(
                metrics.finish(second, status="duplicate", output_tokens=0)
            )
            rows = [
                json.loads(line) for line in path.read_text().splitlines()
            ]
            self.assertEqual([row["request_id"] for row in rows], ["one", "two"])


if __name__ == "__main__":
    unittest.main()
