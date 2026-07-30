# M3 Ultra 512 GB: M13 long-lived prefix workload telemetry

M13 moves M12 from a focused forward-pass validator into an observed,
long-lived API workload. The ordinary OpenAI-compatible server now reports
which complete prompt shapes are resident, whether each request captured or
replayed, how much activation memory the LRU owns, how many expert spans were
avoided, and how logical expert delivery divides into physical RAID traffic
and inferred macOS page-cache delivery.

The official Kimi-K3 checkpoint remains read-only and unconverted.

## Implementation

`PrefixActivationCache.snapshot()` exposes compact state without retaining any
new tensors:

- shapes in least- to most-recently-used order;
- entry count and capacity;
- total activation ownership;
- builds, hits, evictions, invalidations, and failed captures.

Every request record written through `K3_SERVER_METRICS_JSONL` now uses
`deltafin.long-lived-request.v2` and adds:

- logical demand-read expert bytes;
- summed physical RAID-member bytes;
- `max(0, logical - physical)` as an observational page-cache lower bound;
- physical/logical fraction;
- layer-local expert spans and bytes avoided by activation replay;
- skipped route edges;
- cache state before and after the request.

`tools/m13_prefix_workload.py` aggregates one or more JSONL logs by activation
action and total prompt shape. It also replays the observed shape sequence
through configurable LRU capacities, so a longer real trace can answer whether
one, two, or more 195.2 MB entries are justified.

`tools/validate_m13_server_workload.py` starts the real server with response
memoization disabled, sends `Hello` followed by `World`, validates
capture-to-replay behavior, analyzes the emitted JSONL, fingerprints the model
tree, terminates the server, and checks swap.

## Command

```bash
set -a
source config/m3ultra512.env
set +a

.venv/bin/python tools/validate_m13_server_workload.py --overwrite
```

The run used normal buffered positional reads and did not purge the global page
cache. Request counters begin only after the 114.404 GB resident bank and
4.360 GiB scratch slot are ready, so startup I/O is excluded.

Artifacts:

- `bench-results/m13-server-requests.jsonl`
- `bench-results/m13-prefix-workload-summary.json`
- `bench-results/m13-server-workload.json`

## Real server result

Both official chat-template prompts contain 89 positions and share the fixed
74-token prefix.

| Request | Action | TTFT | Wall | Logical expert | Physical members | Inferred page cache |
|---|---|---:|---:|---:|---:|---:|
| `Hello` | capture | `717.654 s` | `717.777 s` | `598.344 GB` | `598.809 GB` | `0 GB` |
| `World` | replay | `135.694 s` | `135.817 s` | `165.576 GB` | `107.127 GB` | `58.449 GB` |

Observed TTFT improved by `5.29×`. Logical expert delivery fell by 72.33% and
summed physical member traffic fell by 82.11%.

The capture physical count is 0.08% larger than the process's logical expert
bytes. This is not negative caching: `iostat` reports system-wide cumulative
member counters and therefore includes concurrent device traffic and reporting
precision. The page-cache estimate is deliberately clamped at zero.

## Activation and ownership

| Check | Result |
|---|---:|
| capture / replay requests | `1 / 1` |
| observed hit rate | `50%` |
| replayed route edges | `108,928` |
| avoided layer-local experts | `25,401` |
| avoided expert-span bytes | `445.718 GB` |
| final LRU | `[89]` |
| activation entries | `1 / 2` |
| activation ownership | `195,198,976 bytes` |
| builds / hits / evictions / invalidations | `1 / 1 / 0 / 0` |
| one-token generated content | exact |
| official model tree | unchanged |
| server process after validation | exited |
| swap delta | `0` |

The 445.718 GB avoided figure is computed from the `World` route union, not by
subtracting the different `Hello` route union. The M12 oracle established that
the corresponding replay is bitwise exact for all prefill/decode logits and
cache tensors; M13 establishes that the same mechanism and accounting work
through the ordinary HTTP server.

## Interpretation

This two-request run proves the server integration and the accounting boundary.
It does not prove that two LRU entries cover real chat traffic. Capacity one
and capacity two are indistinguishable for the observed shape sequence
`[89, 89]`; both incur one capture followed by one hit.

The dominant remaining risk is shape fragmentation. Any different total token
count must capture its own 195.2 MB entry, and its first request retains full
prefill cost. Increasing the LRU blindly would consume little memory relative
to 512 GB, but it would not improve traffic whose prompt lengths rarely repeat.

## Next phase

M14 should collect a longer, non-memoized server trace from representative chat
traffic before changing the selected two-entry capacity:

1. record total-position frequency and reuse distance;
2. run the offline capacity sweep over capacities 1–16;
3. report capture cost, hit-weighted TTFT, physical bytes, and activation
   ownership for each capacity;
4. distinguish exact-shape misses from evictions;
5. only then consider prompt-shape bucketing or another reuse boundary, guarded
   by the same monolithic bitwise parity gate.
