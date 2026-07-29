#!/usr/bin/env python3
"""Validate latency-gated adaptive prefetch with adaptive-first ordering."""

from __future__ import annotations

import os
import sys


os.environ["K3_DIRECT_OVERLAP_POLICY"] = "adaptive"
os.environ.setdefault("K3_DIRECT_PREFETCH_MAX_EXPERTS", "4")
os.environ.setdefault("K3_DIRECT_PREFETCH_TOKEN_BUDGET_BYTES", "4000000000")
os.environ.setdefault("K3_DIRECT_PREFETCH_WARMUP", "64")
os.environ.setdefault("K3_DIRECT_PREFETCH_MIN_WILSON_PRECISION", "0.55")
os.environ["K3_DIRECT_PREFETCH_COLD_ONLY"] = "1"
os.environ.setdefault("K3_DIRECT_PREFETCH_COLD_GBPS", "5.0")
os.environ.setdefault("K3_DIRECT_DEMAND_EMA_ALPHA", "0.25")

from validate_m3_overlap import main  # noqa: E402


if __name__ == "__main__":
    arguments = sys.argv[1:]
    if "--sequence-order" not in arguments:
        arguments.extend(["--sequence-order", "adaptive-first"])
    raise SystemExit(main(arguments))
