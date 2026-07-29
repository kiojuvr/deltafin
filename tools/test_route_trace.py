#!/usr/bin/env python3
"""Tests for router-trace locality analysis."""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_route_trace import analyze, split_runs  # noqa: E402


class RouteTraceTests(unittest.TestCase):
    def row(self, step, layer, ids):
        return {
            "step": step,
            "layer": layer,
            "ids": ids,
            "w": [float(16 - index) for index in range(16)],
        }

    def test_rank_precision_and_run_split(self):
        first = list(range(16))
        second = [0, 1] + list(range(16, 30))
        rows = [
            self.row(0, 1, first),
            self.row(1, 1, second),
            self.row(0, 1, first),
            self.row(1, 1, second),
        ]
        runs = split_runs(rows)
        self.assertEqual(list(map(len, runs)), [2, 2])
        result = analyze(runs[0])
        self.assertEqual(result["transitions"][0]["hits"], 2)
        self.assertEqual(result["rank_precision"][:3], [1.0, 1.0, 0.0])
        self.assertEqual(result["top_k"][1]["precision"], 1.0)
        self.assertAlmostEqual(result["top_k"][-1]["precision"], 2 / 16)


if __name__ == "__main__":
    unittest.main()
