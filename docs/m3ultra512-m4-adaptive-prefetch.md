# M3 Ultra 512 GB: M4 adaptive direct-slab prefetch

## Outcome

M4 replaces unconditional previous-route speculation with an online,
confidence-gated, byte-bounded policy. The path remains bitwise exact and
reduces wasted slab traffic by two orders of magnitude relative to M3's full
16-expert prediction.

It is still **disabled by default**. In the measured warm eight-token run,
adaptive prefetch reduced critical expert wait by 1.16%, but total wall time
was 0.87% slower than the serial oracle. This difference is small enough to
require repeated and cold-path measurements before enabling the policy.

## Route study

`K3_TRACE=buffered` captured the router IDs and normalized weights for eight
consecutive tokens. `tools/analyze_route_trace.py` analyzed the first of three
identical runs:

```bash
.venv/bin/python tools/analyze_route_trace.py \
  bench-results/m4-router-study-8.jsonl \
  --run-index 0 \
  --output bench-results/m4-route-locality.json
```

The generated token sequence was:

```text
220, 802, 24, 12, 2975, 15, 3629, 1389
```

Across seven transitions and 92 routed layers, full-route persistence varied
substantially:

| Transition | Persisting experts | Hit rate |
|---:|---:|---:|
| 0 → 1 | 244 / 1,472 | 16.58% |
| 1 → 2 | 329 / 1,472 | 22.35% |
| 2 → 3 | 727 / 1,472 | 49.39% |
| 3 → 4 | 630 / 1,472 | 42.80% |
| 4 → 5 | 713 / 1,472 | 48.44% |
| 5 → 6 | 790 / 1,472 | 53.67% |
| 6 → 7 | 609 / 1,472 | 41.37% |

Router-weight rank was strongly predictive:

| Previous-route candidates | Precision | Current-route recall |
|---:|---:|---:|
| top 1 | 74.38% | 4.65% |
| top 2 | 68.40% | 8.55% |
| top 4 | 60.75% | 15.19% |
| all 16 | 39.23% | 39.23% |

The selected policy therefore learns a conservative prefix of the previous
route ranked by normalized router weight instead of reading all 16 experts.

## Adaptive policy

`tools/adaptive_prefetch.py` maintains hit and observation counts for the first
four router ranks. Every actual route trains all four ranks, including
counterfactual candidates that were not read.

A rank becomes eligible only when:

- it has at least 64 observations;
- its 95% Wilson precision lower bound is at least 0.55;
- every higher rank is also eligible;
- the token's 4,000,000,000-byte speculative-read budget is not exhausted.

The policy is online and begins with zero reads. In this sequence, rank 1
became eligible first and rank 2 only after additional high-locality
transitions. It never selected ranks 3 or 4.

The current experimental controls are:

```text
K3_DIRECT_OVERLAP=1
K3_DIRECT_OVERLAP_POLICY=adaptive
K3_DIRECT_PREFETCH_MAX_EXPERTS=4
K3_DIRECT_PREFETCH_TOKEN_BUDGET_BYTES=4000000000
K3_DIRECT_PREFETCH_WARMUP=64
K3_DIRECT_PREFETCH_MIN_WILSON_PRECISION=0.55
```

The checked-in M3 Ultra profile keeps `K3_DIRECT_OVERLAP=0`.

## Correctness validation

The focused validator runs serial-first, serial-warm, and adaptive-warm in one
228.8 GB resident-bank process:

```bash
set -a
source config/m3ultra512.env
set +a
.venv/bin/python tools/validate_m4_adaptive.py \
  --token-id 1008 \
  --tokens 8 \
  --output bench-results/m4-adaptive-final.json
```

For every token, the serial repeat and adaptive path must have:

- identical input and generated token IDs;
- identical expert routes in all 92 routed layers;
- bitwise-identical full float32 logits;
- zero maximum absolute logit difference.

The validator also checks official checkpoint fingerprints, HTTP counters,
MPS release, and swap. Unit coverage includes Wilson gating, hard byte-budget
enforcement, reset behavior, route-trace analysis, slab hit/miss reuse, and
background-read failure cleanup.

## Warm measurements

The eight-token adaptive smoke produced:

| Path | Wall time | Critical expert wait | Slab transfer |
|---|---:|---:|---:|
| serial-warm | 28.594 s | 4.825 s | 206.637 GB |
| adaptive-warm | 28.843 s | 4.769 s | 208.093 GB |

The adaptive policy issued 479 expert predictions:

- 396 hits;
- 83 misses;
- 82.67% realized precision;
- 1.456 GB additional slab transfer;
- 0.056 seconds less critical expert wait;
- 0.250 seconds more total wall time.

For comparison, unconditional full-route overlap on the same eight-token route
sequence transferred 316.518 GB, 109.881 GB more than serial, and took 33.075
seconds versus the 32.057-second serial repeat. Adaptive selection reduced the
full-overlap regression from 3.17% to 0.87% in these runs.

The initial route-study serial pass incurred 50.849 GB of physical RAID member
reads and spent 61.632 seconds in expert reads. The adaptive sequence ran after
the serial passes and was warm, so this experiment does **not** claim a cold
speedup. It establishes exactness, bounded waste, and a predictor precise
enough for a balanced cold-path experiment.

## Next step

M5 should separate cold benefit from warm overhead:

1. run adaptive-first and serial-first trials on disjoint prompts/routes;
2. gate prefetch on recent demand-read latency so page-cache-hot layers do not
   pay speculative scheduling overhead;
3. record saved critical bytes and physical bytes, not just candidate hits;
4. repeat enough balanced A/B pairs to resolve a sub-1% warm difference;
5. evaluate one-layer-ahead router lookahead only if its gate-cache compute
   cost is lower than the cold bytes it saves.

BF16 resident storage remains independent and deferred until the cold adaptive
policy has a measured end-to-end benefit.
