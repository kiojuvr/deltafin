#!/usr/bin/env python3
"""Tests for the privacy-minimized server shape trace."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shape_trace import ShapeTraceWriter


class ShapeTraceWriterTests(unittest.TestCase):
    def test_record_contains_shape_state_but_no_prompt_material(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "trace.jsonl"
            writer = ShapeTraceWriter(path)
            row = writer.record(
                request_id="request-1",
                mode="chat",
                total_positions=89,
                memo_hit=False,
                activation={
                    "eligible": True,
                    "prefix_tokens": 74,
                    "hit": True,
                    "action": "replay",
                    "cache_before": {
                        "resident_shapes_lru": [90, 89]
                    },
                    "content": "must-not-leak",
                    "token_ids": [1, 2, 3],
                },
            )
            persisted = json.loads(path.read_text())
            self.assertEqual(persisted, row)
            self.assertEqual(row["resident_shapes_lru"], [90, 89])
            encoded = json.dumps(row)
            self.assertNotIn("must-not-leak", encoded)
            self.assertNotIn("token_ids", encoded)

    def test_trace_rotates_to_one_bounded_backup(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "trace.jsonl"
            writer = ShapeTraceWriter(path, max_bytes=400)
            for index in range(5):
                writer.record(
                    request_id=f"request-{index}",
                    mode="chat",
                    total_positions=89 + index,
                    memo_hit=False,
                    activation={"eligible": True, "prefix_tokens": 74},
                )
            backup = path.with_name(path.name + ".1")
            self.assertTrue(path.exists())
            self.assertTrue(backup.exists())
            self.assertLessEqual(path.stat().st_size, 400)
            self.assertLessEqual(backup.stat().st_size, 400)


if __name__ == "__main__":
    unittest.main()
