#!/usr/bin/env python3
"""Small tests for the M8 order-reversed process harness."""

from __future__ import annotations

import pathlib
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import validate_m8_cache_crossover as m8


def summary(*, wall: float, physical: int):
    return {
        "tokens": 2,
        "wall_seconds": wall,
        "expert_wait_seconds": wall / 2,
        "resident_materialization_seconds": 0.1,
        "logical_expert_bytes": 1000,
        "physical_member_read_bytes": physical,
        "startup_physical_member_read_bytes": 100,
        "unique_expert_working_set_bytes": 800,
        "compressions_delta": 0,
        "swapouts_delta": 0,
        "token_rows": [
            {
                "step": step,
                "input_token": step,
                "output_token": step + 1,
                "routes": {"1": [step]},
                "logits_sha256_f32": f"hash-{step}",
            }
            for step in range(2)
        ],
    }


class M8CacheCrossoverTests(unittest.TestCase):
    def test_pair_requires_full_token_route_and_logit_parity(self):
        fp32 = summary(wall=4.0, physical=400)
        source = summary(wall=3.0, physical=200)
        result = m8.compare_pair(fp32, source)
        self.assertTrue(result["passed"])
        self.assertEqual(
            result["source_minus_fp32"]["physical_member_read_bytes"],
            -200,
        )
        source["token_rows"][1]["routes"] = {"1": [99]}
        with self.assertRaisesRegex(AssertionError, "paired parity"):
            m8.compare_pair(fp32, source)

    def test_aggregate_keeps_modes_and_matched_delta_separate(self):
        runs = [
            {
                "mode": "fp32",
                "position": 0,
                "summary": summary(wall=4.0, physical=400),
            },
            {
                "mode": "source",
                "position": 1,
                "summary": summary(wall=3.0, physical=200),
            },
        ]
        got = m8.aggregate(runs)
        self.assertEqual(got["fp32"]["tokens"], 2)
        self.assertEqual(got["source"]["physical_bytes_per_token"], 100)
        self.assertEqual(
            got["source_minus_fp32"]["seconds_per_token"], -0.5
        )

    def test_child_modes_are_independent_process_commands(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = pathlib.Path(temporary) / "run.json"
            fp32 = m8.child_command(
                "fp32", token_id=7, tokens=8, output=output
            )
            source = m8.child_command(
                "source", token_id=7, tokens=8, output=output
            )
        self.assertIn("--sequence-set", fp32)
        self.assertIn("serial-only", fp32)
        self.assertIn("--pin-layers", fp32)
        self.assertIn("93", fp32)
        self.assertIn("validate_m7_resident_scratch.py", source[1])

    def test_selected_profile_is_source_plus_one_scratch(self):
        selected = {
            "K3_EXPERT_SOURCE": "direct-shards",
            "K3_RESIDENT_SOURCE": "direct-shards",
            "K3_RESIDENT_BANK": "1",
            "K3_DIRECT_SLAB": "1",
            "K3_PREAD_NOCACHE": "0",
            "K3_DTYPE": "fp32",
            "K3_APPROX": "0",
            "K3_PRELOAD": "0",
            "K3_EXPERT_PREFETCH": "0",
            "K3_PREFETCH": "0",
            "K3_PILOT": "0",
            "K3_MOE_GROUP_SIZE": "0",
            "K3_TEMPLATES": "0",
            "K3_RESIDENT_BANK_DTYPE": "source",
            "K3_RESIDENT_SCRATCH": "1",
            "K3_RESIDENT_SCRATCH_OVERLAP": "0",
            "K3_RESIDENT_SCRATCH_SLOTS": "1",
            "K3_PIN_LAYERS": "0",
            "K3_PROFILE": "1",
            "K3_TRACE": "off",
            "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "1.0",
            "PYTORCH_MPS_LOW_WATERMARK_RATIO": "0.9",
        }
        with mock.patch.dict(os.environ, selected, clear=True), mock.patch(
            "validate_m2_serial_token.torch.backends.mps.is_available",
            return_value=True,
        ):
            m8.require_profile()


if __name__ == "__main__":
    unittest.main()
