# M3 Ultra 512 GB: M8 resident/cache crossover

M8 establishes the selected resident representation for the ordinary
M3 Ultra 512 GB runtime. The checkpoint-native 114.4 GB resident bank plus one
fixed 4.68 GB FP32 layer scratch slot is bitwise equivalent to the 228.8 GB
FP32 resident bank, while preserving another 109.7 GB of unified memory for
macOS's file-backed expert cache.

Across two independent, order-reversed 16-token pairs, the source-plus-scratch
path reduced mean decode time from 16.263 to 8.119 seconds/token and physical
RAID-member reads from 9.541 to 4.025 GB/token. It is therefore promoted in
`config/m3ultra512.env`.

## Harness

`tools/validate_m8_cache_crossover.py` launches each resident mode in a fresh
process. Each matched pair starts from the same token and must produce the
same:

- input and output token ID at every step;
- 92-layer routed-expert selection at every step;
- SHA-256 digest of all 163,840 FP32 logits at every step.

The first pair runs FP32 then source; the second runs source then FP32. A
single serial sequence is used in each process so the older validator's
in-process repeat cannot pre-warm the measured run. The harness records:

- logical expert bytes and unique routed-expert working set;
- per-token RAID-member bytes from `iostat`;
- resident-startup and decode physical bytes separately;
- file-backed/free pages, pageins, compressor, and swap;
- MPS ownership before decode and after explicit release.

The run intentionally does not issue `F_NOCACHE` or claim a globally cold
cache. These are natural-cache, order-reversed measurements of the exact
runtime choices.

## Validation command

```sh
set -a
source config/m3ultra512.env
set +a

.venv/bin/python tools/validate_m8_cache_crossover.py \
  --tokens 16 \
  --seeds 83001 83002 \
  --output bench-results/m8-cache-crossover-final.json \
  --run-dir bench-results/m8-cache-crossover-final-runs
```

The top-level evidence and all four child records passed. Both matched pairs
had exact token, route, and FP32-logit parity for all 16 steps. The official
checkpoint tree was unchanged, no conversion was performed, every child
released MPS current allocation to baseline, and swap use did not increase.

Focused tests:

```sh
.venv/bin/python -m unittest \
  tools.test_m8_cache_crossover \
  tools.test_local_safetensors
```

All 17 tests passed.

## Results

Each mode processed 32 tokens and 826,546,323,456 logical expert bytes over
the two runs.

| Metric | FP32 resident | Source + one scratch | Difference |
|---|---:|---:|---:|
| Permanent MPS payload | 228.764 GB | 119.086 GB | -109.678 GB |
| Mean decode time | 16.263 s/token | 8.119 s/token | -8.144 s/token |
| Decode RAID-member reads | 9.541 GB/token | 4.025 GB/token | -5.516 GB/token |
| Physical/logical expert ratio | 36.94% | 15.58% | -21.35 points |
| Total decode wall time | 520.425 s | 259.815 s | -260.609 s |
| Total decode physical reads | 305.314 GB | 128.809 GB | -176.505 GB |
| Compressor operations | 19,994,401 | 0 | -19,994,401 |
| Swapouts | 0 | 0 | equal |

`iostat` reports both RAID members, so the figures above are summed member
traffic and are the metric used consistently by the harness. They are not a
claim of application payload bandwidth or globally cold SSD throughput.

### Order-reversed pairs

| Pair | Order | FP32 s/token | Source s/token | FP32 GB/token | Source GB/token |
|---|---|---:|---:|---:|---:|
| seed 83001 | FP32 → source | 15.792 | 4.157 | 9.212 | 0.273 |
| seed 83002 | source → FP32 | 16.734 | 12.082 | 9.870 | 7.777 |

The second position naturally inherits some pages from the first, which
explains the very warm source result in pair one. The reverse order is more
important: constructing the 228.8 GB FP32 bank after the source run took
78.9 seconds, read 50.35 GB during startup, and the following decode still
read 157.92 GB. The larger anonymous/MPS resident allocation displaced a
substantial part of the expert working set that the immediately preceding
source run had populated.

The first-position comparison also favors source despite using different
seeds and a larger unique expert working set:

| First-position run | Unique expert bytes | Time/token | Physical GB/token |
|---|---:|---:|---:|
| FP32, seed 83001 | 178.385 GB | 15.792 | 9.212 |
| Source, seed 83002 | 190.037 GB | 12.082 | 7.777 |

This comparison is not matched by route and is supporting evidence only. The
two exact matched-pair comparisons are the acceptance result.

## Selected profile

The M3 Ultra profile now uses:

```text
K3_RESIDENT_BANK_DTYPE=source
K3_RESIDENT_SCRATCH=1
K3_RESIDENT_SCRATCH_SLOTS=1
K3_RESIDENT_SCRATCH_OVERLAP=0
K3_PIN_LAYERS=0
```

The two-slot resident worker and expert adaptive prefetch remain disabled.
M8 changes resident ownership, not expert fetch semantics: ordinary buffered
positional reads continue to rely on the macOS file cache and never create a
second anonymous expert cache.

## Bottleneck and next phase

Source plus scratch removes resident capacity as the dominant problem, but a
16-token first-position run still read 124.44 GB from the RAID members and
spent 12.08 seconds/token. The next phase should instrument the ordinary
long-lived generation server rather than another isolated validator:

1. retain the selected resident bank across multiple requests;
2. report per-request and rolling expert physical bytes, cache fraction,
   unique expert working set, and token latency;
3. separate resident startup from time-to-first-token;
4. exercise repeated and unrelated prompts without a cache purge;
5. only then reconsider adaptive prefetch or compute/I/O overlap using
   measured page-cache misses as the gate.

The acceptance target is stable multi-request exact inference with no
unbounded anonymous cache, no compressor/swap growth, and physical
bytes/token that improve as the natural routed working set converges.
