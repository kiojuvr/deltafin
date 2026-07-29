#!/usr/bin/env python3
"""Analyze previous-token expert persistence by router-weight rank."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=pathlib.Path)
    parser.add_argument("--run-index", type=int, default=0)
    parser.add_argument("--output", type=pathlib.Path)
    return parser.parse_args(argv)


def split_runs(rows):
    runs = []
    current = []
    for row in rows:
        if (
            current
            and row["step"] == 0
            and row["layer"] == min(item["layer"] for item in current)
        ):
            runs.append(current)
            current = []
        current.append(row)
    if current:
        runs.append(current)
    return runs


def analyze(rows):
    by_key = {(row["step"], row["layer"]): row for row in rows}
    steps = sorted({step for step, _layer in by_key})
    layers = sorted({layer for _step, layer in by_key})
    expected = {(step, layer) for step in steps for layer in layers}
    if set(by_key) != expected:
        raise ValueError("trace does not contain a complete step/layer grid")
    rank_hits = [0] * 16
    rank_observations = [0] * 16
    layer_hits = {layer: 0 for layer in layers}
    layer_observations = {layer: 0 for layer in layers}
    transitions = []
    for previous_step, current_step in zip(steps, steps[1:]):
        transition_hits = 0
        for layer in layers:
            previous = by_key[previous_step, layer]
            actual = set(by_key[current_step, layer]["ids"])
            ranked = sorted(
                zip(previous["ids"], previous["w"]),
                key=lambda pair: pair[1],
                reverse=True,
            )
            if len(ranked) != 16:
                raise ValueError(
                    f"step {previous_step} layer {layer} is not top-16"
                )
            for rank, (expert, _weight) in enumerate(ranked):
                hit = int(expert in actual)
                rank_hits[rank] += hit
                rank_observations[rank] += 1
                layer_hits[layer] += hit
                layer_observations[layer] += 1
                transition_hits += hit
        total = len(layers) * 16
        transitions.append(
            {
                "previous_step": previous_step,
                "current_step": current_step,
                "hits": transition_hits,
                "experts": total,
                "hit_rate": transition_hits / total,
            }
        )
    cumulative_hits = 0
    top_k = []
    layer_transitions = max(0, len(steps) - 1)
    for rank, (hits, observations) in enumerate(
        zip(rank_hits, rank_observations), start=1
    ):
        cumulative_hits += hits
        predicted = layer_transitions * len(layers) * rank
        top_k.append(
            {
                "k": rank,
                "hits": cumulative_hits,
                "predicted": predicted,
                "precision": (
                    cumulative_hits / predicted if predicted else 0.0
                ),
                "recall_of_full_route": (
                    cumulative_hits
                    / (layer_transitions * len(layers) * 16)
                    if layer_transitions else 0.0
                ),
            }
        )
    return {
        "steps": steps,
        "layers": layers,
        "transitions": transitions,
        "rank_hits": rank_hits,
        "rank_observations": rank_observations,
        "rank_precision": [
            hits / observations if observations else 0.0
            for hits, observations in zip(
                rank_hits, rank_observations
            )
        ],
        "top_k": top_k,
        "per_layer": {
            str(layer): {
                "hits": layer_hits[layer],
                "observations": layer_observations[layer],
                "hit_rate": (
                    layer_hits[layer] / layer_observations[layer]
                    if layer_observations[layer] else 0.0
                ),
            }
            for layer in layers
        },
    }


def main(argv=None):
    args = parse_args(argv)
    rows = [
        json.loads(line)
        for line in args.trace.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    runs = split_runs(rows)
    try:
        selected = runs[args.run_index]
    except IndexError as exc:
        raise ValueError(
            f"run index {args.run_index} outside {len(runs)} trace runs"
        ) from exc
    result = {
        "schema": "deltafin.route-locality.v1",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "trace": str(args.trace.resolve()),
        "run_index": args.run_index,
        "run_count": len(runs),
        "analysis": analyze(selected),
    }
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
