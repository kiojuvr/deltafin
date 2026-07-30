#!/usr/bin/env python3
"""Validate fixed FP32 scratch slots over checkpoint-native resident weights."""

from __future__ import annotations

import pathlib
import sys

from validate_m3_overlap import main


ROOT = pathlib.Path(__file__).resolve().parent.parent


if __name__ == "__main__":
    user_arguments = sys.argv[1:]
    overlap = "--resident-scratch-overlap" in user_arguments
    arguments = [
        "--resident-bank-dtype",
        "source",
        "--pin-layers",
        "0",
        "--resident-scratch",
        "--resident-scratch-slots",
        "2" if overlap else "1",
        *user_arguments,
    ]
    if not any(
        argument == "--output" or argument.startswith("--output=")
        for argument in arguments
    ):
        arguments.extend(
            ["--output", str(ROOT / "bench-results/m7-resident-scratch.json")]
        )
    raise SystemExit(main(arguments))
