#!/usr/bin/env python3
"""Tests for the M16 passive-evidence admission gate."""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m14_shape_trace import analyze_shapes  # noqa: E402
from m16_admission_gate import (  # noqa: E402
    evaluate_admission_gate,
    prior_gate_summary,
)


def analysis(shapes):
    return analyze_shapes(
        [
            {"eligible": True, "total_positions": shape}
            for shape in shapes
        ],
        max_capacity=2,
    )


class M16AdmissionGateTests(unittest.TestCase):
    def test_one_passive_request_keeps_lru_without_expensive_ab(self):
        gate = evaluate_admission_gate(analysis([89]), capacity=2)
        self.assertFalse(gate["candidate_for_controlled_server_ab"])
        self.assertEqual(gate["repeat_requests"], 0)
        self.assertEqual(gate["hit_gain"], 0)
        self.assertEqual(
            gate["next_action"],
            "keep-lru-and-continue-passive-observation",
        )

    def test_observed_scan_protection_promotes_only_to_server_ab(self):
        shapes = [89, 90, 89, 90, 100, 89]
        gate = evaluate_admission_gate(analysis(shapes), capacity=2)
        self.assertTrue(gate["candidate_for_controlled_server_ab"])
        self.assertEqual(gate["hit_gain"], 1)
        self.assertEqual(gate["capture_reduction"], 1)
        self.assertEqual(gate["next_action"], "run-controlled-server-ab")

    def test_capacity_must_exist_in_both_sweeps(self):
        with self.assertRaisesRegex(ValueError, "capacity 3"):
            evaluate_admission_gate(analysis([89]), capacity=3)

    def test_prior_exactness_summary_requires_every_boundary(self):
        m12 = {
            "schema": "deltafin.m12-prefix-activation.v1",
            "status": "passed",
            "parity": {"passed": True},
            "activation_immutable": True,
            "official_tree_before": [1],
            "official_tree_after": [1],
        }
        m13 = {
            "schema": "deltafin.m13-server-workload.v1",
            "status": "passed",
            "response_exact": True,
            "official_tree_before": [1],
            "official_tree_after": [1],
        }
        self.assertTrue(prior_gate_summary(m12, m13)["passed"])
        m13["schema"] = "wrong"
        self.assertFalse(prior_gate_summary(m12, m13)["passed"])
        m13["schema"] = "deltafin.m13-server-workload.v1"
        m13["response_exact"] = False
        self.assertFalse(prior_gate_summary(m12, m13)["passed"])


if __name__ == "__main__":
    unittest.main()
