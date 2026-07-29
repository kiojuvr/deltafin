#!/usr/bin/env python3
"""Run the exact multi-token validator with bounded adaptive slab prefetch."""

from __future__ import annotations

import os


os.environ.setdefault("K3_DIRECT_OVERLAP_POLICY", "adaptive")
os.environ.setdefault("K3_DIRECT_PREFETCH_MAX_EXPERTS", "4")
os.environ.setdefault("K3_DIRECT_PREFETCH_TOKEN_BUDGET_BYTES", "4000000000")
os.environ.setdefault("K3_DIRECT_PREFETCH_WARMUP", "64")
os.environ.setdefault("K3_DIRECT_PREFETCH_MIN_WILSON_PRECISION", "0.55")
os.environ.setdefault("K3_DIRECT_PREFETCH_COLD_ONLY", "0")

from validate_m3_overlap import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
