#!/usr/bin/env python3
"""Validate checkpoint-native resident ownership with FP32 layer scratch."""

from __future__ import annotations

import pathlib
import sys

from validate_m3_overlap import main


ROOT = pathlib.Path(__file__).resolve().parent.parent


if __name__ == "__main__":
    arguments = [
        "--resident-bank-dtype",
        "source",
        "--pin-layers",
        "0",
        *sys.argv[1:],
    ]
    if not any(
        argument == "--output" or argument.startswith("--output=")
        for argument in arguments
    ):
        arguments.extend(
            ["--output", str(ROOT / "bench-results/m6-source-resident.json")]
        )
    raise SystemExit(main(arguments))
