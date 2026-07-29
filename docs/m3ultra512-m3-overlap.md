# M3 Ultra 512 GB: M3 multi-token cache and slab overlap

## Outcome

M3 establishes an exact multi-token decode path over the unmodified official
Kimi-K3 checkpoint and separates logical expert demand from physical RAID
traffic. It also implements the first real layer `N` compute / layer `N+1`
read overlap experiment.

The overlap path is bitwise exact, but it is deliberately **disabled by
default**. On the measured three-token sequence, using the previous token's
route as the next token's prediction reused only 573 of 2,944 expert slots
(19.46%). Reading all predicted experts therefore added 41.60 GB of slab
traffic and made the warm sequence 4.93% slower than serial execution.

This is a useful negative performance result: the double slab and ownership
model are sound, while route prediction quality—not I/O plumbing—is the
condition that must improve before overlap should become the runtime default.

## Implementation

- `tools/expert_slab.py`
  - retains same-layer experts at fixed slot addresses;
  - fills only route misses after an asynchronous predicted fill;
  - returns the actual expert-to-slot mapping, so route order and slot order
    need not match;
  - preserves the contiguous one-`preadv`-per-expert fast path and the
    non-contiguous tensor fallback.
- `tools/direct_shard_loader.py`
  - owns a bounded single-worker prefetch queue over the existing two banks;
  - leases one free bank without blocking the compute thread;
  - records speculative bytes, wait time, hits, misses, and miss-refill time;
  - returns every bank even when a background read fails.
- `tools/kimi_run.py`
  - snapshots the previous token's 92 routed-layer selections;
  - starts layer `N+1`'s predicted fill immediately before layer `N` compute;
  - treats the current router result as authoritative and refills every miss
    before Metal consumes the bank.
- `tools/validate_m3_overlap.py`
  - runs `serial-first`, identical `serial-warm`, and identical
    `overlap-warm` sequences in one resident-bank process;
  - records every token's full logits digest, all 92 routes, 93 layer profiles,
    logical bytes, slab bytes, physical RAID-member bytes, MPS memory, swap,
    and official-tree fingerprints;
  - requires bitwise logits, routes, and generated-token parity.

The M3 Ultra profile explicitly sets:

```text
K3_DIRECT_OVERLAP=0
```

Set it to `1` only for controlled experiments.

## Validation

The measured smoke used token 1008 and decoded three consecutive tokens:

```bash
set -a
source config/m3ultra512.env
set +a
.venv/bin/python tools/validate_m3_overlap.py \
  --token-id 1008 \
  --tokens 3 \
  --output bench-results/m3-overlap-smoke.json
```

All three modes generated `[220, 802, 24]`. For each step, serial-repeat and
overlap execution had:

- identical input and output token IDs;
- identical selected expert IDs in all 92 routed layers;
- bitwise-identical full float32 logits;
- zero maximum absolute logit difference.

The logit SHA-256 values were:

| Step | Output | Full float32 logits SHA-256 |
|---:|---:|---|
| 0 | 220 | `4065d212e8885a89bb5635145801f653bc2f6eeb31b93c6c90f66c7e1d7715d6` |
| 1 | 802 | `6fe4e4b434911bfa14341f9deeb36b787795a3d3ec68e558711d5d8d985b4e24` |
| 2 | 24 | `97b070fe9752d168de931f4c3ad4cf32f61b0ec12b4b67539ba104c86bbbc869` |

The official model index, all 96 shards, and the complete official model tree
retained identical fingerprints. No conversion, write, or model copy was
performed. After cleanup, MPS current allocation returned exactly to baseline,
driver allocation was 562,200,576 bytes above baseline, and swap remained
unchanged.

The disk-small suite also covers slab hit/miss reuse and background-read
failure cleanup:

```bash
.venv/bin/python -m unittest \
  tools.test_local_safetensors \
  tools.test_k3_official
```

## Measurements

Each token logically demands 1,472 experts, or 25,829,572,608 bytes. RAID
physical bytes below are deltas of the two member disks. They are not inferred
from process-side `preadv` throughput.

| Sequence | Wall time | Critical expert wait | Logical demand | Slab transfer | Physical member reads |
|---|---:|---:|---:|---:|---:|
| serial-first | 58.913 s | 48.607 s | 77.489 GB | 77.489 GB | 41.239 GB |
| serial-warm | 10.740 s | 1.814 s | 77.489 GB | 77.489 GB | 0.001 GB |
| overlap-warm | 11.269 s | 1.707 s | 77.489 GB | 119.093 GB | 0 GB |

The first serial step was already present in the macOS file cache from earlier
validation, so `serial-first` is not a global-cache-purge benchmark. Its next
two steps nevertheless caused 21.531 GB and 19.708 GB of member reads. The
corresponding warm repeat was served almost entirely from RAM. This confirms
why process-side multi-GB/s reads must not be described as physical USB SSD
bandwidth.

For the two transitions, route locality was:

| Transition | Reused experts | Total experts | Hit rate | Exact layers |
|---|---:|---:|---:|---:|
| token 0 → 1 | 244 | 1,472 | 16.58% | 0 |
| token 1 → 2 | 329 | 1,472 | 22.35% | 0 |
| combined | 573 | 2,944 | 19.46% | 0 |

Overlap hid almost all wait for the speculative fill and reduced critical
expert wait by 0.107 seconds across the three-token warm sequence. However,
the 41.60 GB of extra slab writes contended for unified-memory bandwidth.
Total wall time increased by 0.529 seconds (4.93%); considering only tokens
that had a previous route, the regression was 7.28%.

## Current bottleneck and next step

The cold path remains dominated by physical expert misses: about 20 GB of the
25.83 GB logical demand reached the RAID on the two newly routed steps in this
sample. The page cache can make the same work fast, but the previous route
alone is too weak a predictor to justify filling 16 speculative slots per
layer.

The next phase should retain serial direct-shards as the default oracle and:

1. collect longer-sequence per-layer and per-expert transition statistics;
2. prefetch only high-confidence experts, with a byte budget and an adaptive
   disable threshold;
3. account for physical bytes/token and saved critical bytes, not apparent
   `preadv` bandwidth;
4. repeat full-logit parity and compare total token latency before enabling any
   predictor;
5. only then evaluate lower-cost router prediction or BF16 resident storage.

The implemented overlap mechanism is ready for those experiments; M3 shows
that unconditional previous-route speculation is not the policy to ship.
