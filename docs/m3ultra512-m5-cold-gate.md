# M3 Ultra 512 GB: M5 demand-latency cold gate

## Outcome

M5 adds a demand-read latency gate to M4's confidence- and byte-bounded
adaptive prefetch. The gate correctly separates the measured cold candidates
from page-cache-served repeats:

- cold-candidate demand EMA: primarily 1.5–6.4 GB/s;
- normal warm-repeat demand EMA: 29–38 GB/s;
- prefetch threshold: 5.0 GB/s;
- five warm overlap repeats issued zero speculative reads.

The path remains bitwise exact. It is still **disabled by default** because two
disjoint-seed crossover pairs did not establish an end-to-end cold speedup.
The adaptive-first mean was 0.75% slower than the serial-first mean, while
route and physical-byte differences between seeds were larger than the
expected effect.

## Implementation

`tools/direct_shard_loader.py` now records only actual-route slab reads in a
demand signal. Speculative reads never update it. For every demand read it
records:

- bytes and elapsed time;
- instantaneous logical read bandwidth;
- a layer-to-layer bandwidth EMA with alpha 0.25.

`tools/kimi_run.py` opens the adaptive gate only when the EMA is at or below
5.0 GB/s. Router-rank confidence, Wilson lower bound, warmup, and token byte
budget must still pass independently.

`tools/adaptive_prefetch.py` records:

- gate-open and gate-closed layer decisions;
- missing-signal decisions;
- candidates skipped by the I/O gate;
- predictions, hits, misses, and realized precision.

The bandwidth EMA is a demand-latency proxy, not a physical-disk measurement.
Physical bytes continue to come from the two RAID member counters.

## Configuration

The M5 experiment uses:

```text
K3_DIRECT_OVERLAP=1
K3_DIRECT_OVERLAP_POLICY=adaptive
K3_DIRECT_PREFETCH_COLD_ONLY=1
K3_DIRECT_PREFETCH_COLD_GBPS=5.0
K3_DIRECT_DEMAND_EMA_ALPHA=0.25
```

The existing M4 confidence and byte-budget settings remain in effect. The
checked-in M3 Ultra profile continues to set `K3_DIRECT_OVERLAP=0`.

## Validation

The M5 validator supports both sequence orders and optional `F_NOCACHE`:

```bash
set -a
source config/m3ultra512.env
set +a

.venv/bin/python tools/validate_m5_cold_gate.py \
  --token-id 50000 \
  --tokens 8 \
  --output bench-results/m5-cold-gate-smoke-50000.json

.venv/bin/python tools/validate_m5_cold_gate.py \
  --token-id 50001 \
  --tokens 8 \
  --sequence-order serial-first \
  --output bench-results/m5-cold-gate-smoke-50001-serial-first.json
```

Adaptive-first runs use:

1. adaptive overlap on the first, potentially cold sequence;
2. serial replay of the exact same sequence;
3. adaptive replay of the exact same sequence.

Every comparison requires identical generated tokens, all 92 routed-layer
expert sets, and full float32 logits. Official checkpoint fingerprints, HTTP
counters, MPS release, and swap are also checked.

## Cold-candidate crossover

Two adjacent-seed pairs reversed the first-run mode:

| Seed | First-run mode | Wall | Expert wait | Physical member reads |
|---:|---|---:|---:|---:|
| 50000 | adaptive | 125.607 s | 98.605 s | 84.123 GB |
| 50001 | serial | 127.365 s | 99.779 s | 85.075 GB |
| 50002 | adaptive | 126.780 s | 99.668 s | 84.396 GB |
| 50003 | serial | 123.141 s | 96.080 s | 81.596 GB |

The two-mode means were:

| Mode | Mean wall | Mean expert wait | Mean physical reads |
|---|---:|---:|---:|
| adaptive-first | 126.194 s | 99.137 s | 84.259 GB |
| serial-first | 125.253 s | 97.929 s | 83.335 GB |
| adaptive delta | +0.941 s (+0.75%) | +1.207 s (+1.23%) | +0.924 GB |

Across the adaptive-first runs:

- 723 experts were prefetched;
- 506 were used, for 70.0% precision;
- 217 were wasted;
- 8.879 GB of actual-route demand was moved out of the critical read;
- 3.808 GB of extra slab traffic was introduced;
- more than 99.9% of prefetch elapsed time was hidden behind current-layer
  work.

The first pair favored adaptive by 1.38%; the second favored serial by 2.87%.
This spread demonstrates that two different routes are not precise enough to
resolve a roughly 1% effect.

## Warm gate behavior

Five same-sequence adaptive repeats were page-cache-served or partially served.
The gate skipped every eligible candidate:

| Evidence | Gate-open layers | Predictions | Gate-closed layers |
|---|---:|---:|---:|
| seed 50000 | 0 | 0 | 343 |
| seed 50001 | 0 | 0 | 405 |
| seed 50002 | 0 | 0 | 400 |
| seed 50003 | 0 | 0 | 328 |
| seed 70000 with `F_NOCACHE` | 0 | 0 | 400 |

Thus M5 removes M4's known warm extra slab traffic. Runtime variation remains,
but there is no speculative I/O in the warm measurements.

## `F_NOCACHE` result

Seed 70000 was run adaptive-first with `--darwin-nocache`. The first sequence
read 102.104 GB physically and took 150.950 seconds. The identical serial
replay read only 20.283 GB and took 55.277 seconds.

This confirms that macOS `F_NOCACHE` is not a global cache purge and does not
provide two equivalent cold executions. That run is valid parity and gate
evidence, but not a cold performance A/B.

## Correctness and ownership

All five focused validators passed:

- adaptive-first and serial routes were identical;
- generated tokens and full logits were bitwise identical;
- no HTTP path was used;
- official checkpoint fingerprints were unchanged;
- MPS current allocation returned to baseline;
- swap was unchanged.

No model conversion, checkpoint write, model copy, or large derived cache was
created.

## Next step

M5 shows that prediction plumbing is no longer the main uncertainty. The gate
can prevent warm waste, but a sub-1% cold effect cannot be isolated without
stronger cache control or many more randomized seeds.

The higher-leverage next phase is BF16 resident ownership:

1. retain the exact float32 path as the oracle;
2. materialize resident tensors directly to BF16 MPS storage without retaining
   an FP32 CPU or MPS copy;
3. recover approximately 114 GB for macOS file cache;
4. repeat full-logit/token comparisons with explicit error bounds;
5. measure whether the larger file cache reduces physical expert bytes/token.

If exact-token parity is unacceptable under BF16, M5 remains available for
future cold trials after controlled reboot or system-level cache flushing.
