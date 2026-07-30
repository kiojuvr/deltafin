# M3 Ultra 512 GB: M15 scan-resistant prefix admission

M15 implements and falsifies a scan-resistant admission policy for the M12
shape-stable prefix activation cache. The selected runtime remains exact
two-entry LRU because no passive real-traffic trace has accumulated yet.

The policy is available experimentally as:

```text
K3_PREFIX_ACTIVATION_ADMISSION=repeat
K3_PREFIX_ACTIVATION_HISTORY=4096
```

The M3 Ultra profile explicitly keeps:

```text
K3_PREFIX_ACTIVATION_ADMISSION=lru
```

## Motivation

M14's public stress trace contains 32 distinct shapes in 42 requests. At
capacity two, ordinary LRU admits every completed capture, so a scan of
one-off shapes repeatedly evicts previously useful activations. Increasing
capacity did not solve the compulsory-miss term and showed long plateaus with
no extra hits.

## Repeat-aware policy

The experimental policy keeps a count of observed total-position shapes but no
prompt content or token IDs. The observation table is an LRU-bounded map whose
default maximum is 4096 shapes. The selected `lru` policy does not populate the
map, so it adds no shape-history growth to the production path.

For an eligible request:

1. a resident shape replays normally and becomes most-recently used;
2. a miss always runs the complete monolithic capture path;
3. while free slots exist, the completed capture is admitted;
4. once full, a first-seen shape is closed and bypasses retention;
5. a shape seen previously is admitted and may evict the LRU entry.

Thus admission never changes the current request's logits, routes, cache state,
or expert reads. It changes only whether the completed 195.2 MB activation
session survives for a future request. A bypassed session is explicitly closed;
it does not become an anonymous expert cache or retain model weights.

Telemetry now reports admission policy, observations, capture completions,
admissions, bypasses, and the existing build/hit/eviction counters.
The focused tests also force a two-entry observation bound and verify that the
oldest shape loses its repeat status after eviction.

## Model-independent lifecycle gate

A focused cache test uses two hot shapes followed by four one-off shapes:

```text
[2, 3, 2, 3, 4, 5, 6, 7, 2, 3]
```

With capacity two and repeat admission:

| Check | Result |
|---|---:|
| hits | `4` |
| capture completions | `6` |
| retained builds | `2` |
| bypasses | `4` |
| final LRU | `[2, 3]` |
| bypassed activation ownership before close | structurally valid |

Invalid policies are rejected, and default construction remains `lru`.

## M14 stress comparison

Command:

```bash
set -a
source config/m3ultra512.env
set +a

.venv/bin/python tools/m14_shape_trace.py \
  --stress \
  --max-capacity 16 \
  --calibration-jsonl bench-results/m13-server-requests.jsonl \
  --output bench-results/m15-admission-stress-final.json
```

At the selected capacity two:

| Metric | LRU | Repeat-aware |
|---|---:|---:|
| hits | `4` | `7` |
| hit rate | `9.52%` | `16.67%` |
| captures | `38` | `35` |
| eviction misses | `6` | `3` |
| retained admissions | `38` | `5` |
| first-seen bypasses | `0` | `30` |
| maximum ownership | `390.4 MB` | `390.4 MB` |

Repeat-aware admission produced 75% more hits at the same activation capacity.
The stress trace has only 32 distinct shapes, below the 4096-entry runtime
history bound, so the runtime and offline admission decisions are equivalent
for this comparison.
Across other capacities:

| Entries | LRU hits | Repeat hits |
|---:|---:|---:|
| 1 | 0 | 2 |
| 2 | 4 | 7 |
| 3 | 6 | 9 |
| 4 | 6 | 9 |
| 8 | 6 | 9 |
| 16 | 7 | 9 |

Using the single M13 T=89 pair as a deliberately coarse calibration, the
capacity-two synthetic projection changes as follows:

| Metric | LRU | Repeat-aware | Reduction |
|---|---:|---:|---:|
| TTFT total | `7.726 h` | `7.241 h` | `6.28%` |
| logical expert delivery | `23.399 TB` | `22.101 TB` | `5.55%` |
| physical member traffic | `23.183 TB` | `21.708 TB` | `6.36%` |

These are cost-model propagation checks, not K3 performance predictions. The
calibration has one capture and one replay at T=89, while the public stress
trace ranges from 94 to 125 positions.

## Correctness boundary

M15 does not alter:

- prompt position dimension;
- activation input/route/weight bitwise validation;
- Metal empty-prefix-row ABI;
- routed output splice;
- official checkpoint reads;
- resident bank or expert slab ownership.

The M12/M13 real-server parity therefore remains the replay correctness gate.
Before `repeat` can become selected, it still needs a real server workload that
contains both a one-off scan and a later hot-shape replay, followed by the same
bitwise oracle comparison.

## Decision

Keep exact LRU selected. Retain repeat-aware admission as an explicit
experimental option.

The stress result is strong enough to justify implementation, but not selection:
the passive trace file did not yet exist, so there is no representative
evidence that real traffic contains the same one-off scan pattern.

## Next phase

M16 should analyze the bounded passive trace after ordinary server use. If it
contains sufficient reuse:

1. compare LRU and repeat-aware hit/cost results at capacities 1–16;
2. select a workload containing hot shapes separated by one-off scans;
3. run that sequence through the ordinary server with both policies;
4. compare final logits/tokens, physical bytes, bypass release, and LRU state;
5. select `repeat` only if it improves real hit-weighted TTFT without changing
   the M12 bitwise contract.
