# M3 Ultra 512 GB: M14 prompt-shape trace and capacity falsifier

M14 establishes the measurement needed to choose the fixed-prefix activation
LRU capacity without retaining prompt material. It does not change the selected
two-entry cache: the repository had only four prior server request records, of
which only the two M13 records contained activation shape data. That sample
cannot justify a capacity change.

The official Kimi-K3 tokenizer and checkpoint remain read-only. No inference,
resident-bank construction, model conversion, or weight derivative is needed
for offline request-shape analysis.

## Lightweight server trace

The selected M3 Ultra profile now writes
`bench-results/m14-server-shapes.jsonl` through
`K3_SERVER_SHAPE_TRACE_JSONL`. Each row contains only:

- timestamp and random server request ID;
- chat/completion mode and total position count;
- activation eligibility and fixed-prefix length;
- memo status, activation hit/action preview, and resident shape LRU.

It does not store prompt content, token IDs, decoded tokens, or a prompt hash.
The absence of even a content hash avoids dictionary recovery of predictable
short prompts.

The active file is capped at 16 MiB by
`K3_SERVER_SHAPE_TRACE_MAX_BYTES`. One rotated `.1` backup is retained, bounding
the complete trace to about 32 MiB.

This path is separate from `K3_SERVER_METRICS_JSONL`: it does not call
`vm_stat`, `vmmap`, or `iostat`, and is suitable for passive shape collection
during ordinary server use.

## Analysis

`tools/m14_shape_trace.py` accepts:

- the lightweight server shape JSONL;
- OpenAI request bodies, tokenized locally with the official tokenizer; or
- a built-in public stress sequence.

For request bodies, the output discards content and token IDs immediately after
tokenization. The report includes:

- exact-shape frequency;
- LRU stack reuse distance;
- compulsory versus eviction misses;
- exclusion of response-memo hits that never enter activation prefill;
- capacity sweep and activation ownership;
- the infinite-capacity exact-shape hit ceiling;
- an optional, explicitly coarse cost projection calibrated from M13.

The M13 `simulate_lru` helper now classifies every miss and exposes the same
reuse-distance contract.

## Validation commands

Public scan-heavy tokenizer trace:

```bash
set -a
source config/m3ultra512.env
set +a

.venv/bin/python tools/m14_shape_trace.py \
  --stress \
  --max-capacity 16 \
  --calibration-jsonl bench-results/m13-server-requests.jsonl \
  --output bench-results/m14-shape-trace-final.json
```

Analyze passive server data after ordinary use:

```bash
.venv/bin/python tools/m14_shape_trace.py \
  --shape-jsonl \
    bench-results/m14-server-shapes.jsonl.1 \
    bench-results/m14-server-shapes.jsonl \
  --max-capacity 16 \
  --calibration-jsonl bench-results/m13-server-requests.jsonl \
  --output bench-results/m14-observed-shapes.json
```

The analyzer also passed a focused shape-JSONL ingestion test with sequence
`[89, 90, 89, 91, 90, 91]`, correctly reporting three compulsory misses and
one capacity-two eviction miss.

## Stress result

The built-in 42-request trace deliberately warms two short development-command
shapes, scans increasing prompt lengths, revisits the hot shapes, scans again,
and then revisits them. It is a falsifier for scan-heavy behavior, not a claim
about real user traffic.

| Metric | Result |
|---|---:|
| eligible requests | `42 / 42` |
| fixed prefix | `74 tokens` |
| distinct total shapes | `32` |
| compulsory misses | `32` |
| repeat requests | `10` |
| infinite-capacity hit ceiling | `23.81%` |
| capacity needed for every observed reuse | `18` |
| official model tree | unchanged |
| prompt content/token IDs/hashes in artifact | none |

Observed reuse distances were:

| Distance | Requests |
|---:|---:|
| first occurrence | 32 |
| 1 | 4 |
| 2 | 2 |
| 14 | 1 |
| 16 | 1 |
| 17 | 2 |

## Capacity sweep

| Entries | Hit rate | Hits | Eviction misses | Max ownership |
|---:|---:|---:|---:|---:|
| 1 | `0%` | 0 | 10 | `186.2 MiB` |
| 2 | `9.52%` | 4 | 6 | `372.3 MiB` |
| 3 | `14.29%` | 6 | 4 | `558.5 MiB` |
| 4–14 | `14.29%` | 6 | 4 | `744.6–2,606.2 MiB` |
| 15–16 | `16.67%` | 7 | 3 | `2,792.3–2,978.5 MiB` |
| infinite | `23.81%` | 10 | 0 | — |

Capacity three gains two hits over the selected capacity two in this trace.
Capacities four through fourteen gain nothing. Reaching every observed repeat
would require eighteen entries, about 3.27 GiB, but even then 32 compulsory
misses remain.

Using the single M13 capture/replay pair as a coarse calibration, capacity two
projects 7.73 hours and 23.18 TB of summed member traffic for this synthetic
trace, while capacity three projects 7.40 hours and 22.20 TB. These are not
performance predictions: the calibration is T=89 and the stress shapes range
from 94 to 125 positions. The values only verify cost propagation through the
capacity model.

## Decision

Keep `K3_PREFIX_ACTIVATION_ENTRIES=2`.

The stress trace confirms that exact-shape reuse can be fragmented enough that
blindly adding entries buys long plateaus of no benefit. Conversely, a
deliberately scan-heavy public trace cannot establish that capacity two is
optimal for real traffic. The lightweight bounded trace is therefore selected,
while cache admission and capacity remain unchanged.

## Next phase

M15 should use the accumulated passive trace to compare:

1. exact LRU capacities 1–16;
2. compulsory versus eviction cost under the observed reuse distances;
3. a scan-resistant admission policy that does not let one-off shapes evict
   repeatedly used shapes;
4. retained activation memory versus hit-weighted TTFT and physical bytes.

Any admission-policy change must first pass model-independent trace tests and
then the M12 monolithic bitwise parity gate through the ordinary server.
