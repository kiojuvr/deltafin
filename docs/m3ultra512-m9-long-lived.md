# M3 Ultra 512 GB: M9 long-lived request cache

M9 validates repeated requests in one long-lived source-resident runtime and
adds optional request telemetry to the ordinary OpenAI-compatible server.
Resident tensors and the fixed scratch arena are constructed once; every
request gets a fresh KV/recurrent state and reset route/prefetch state.

The selected five-request workload was:

```text
anchor-1(85001, 8 tokens)
unrelated-b(85002, 8 tokens)
anchor-2(85001, 8 tokens)
unrelated-c(85003, 8 tokens)
anchor-3(85001, 8 tokens)
```

Both anchor replays matched the first anchor bit-for-bit at every token:
input token, output token, all 92 routed-layer selections, and all 163,840
FP32 logits had `max_abs == 0`.

## Implementation

`tools/validate_m9_long_lived.py` reuses the established serial direct-slab
sequence inside one process. At every request boundary it records:

- TTFT and post-TTFT token time;
- logical expert bytes and RAID-member physical reads;
- per-request unique experts, experts new to the process, and cumulative
  routed working set;
- file-backed/free pages, pageins, compressor, swap, process RSS, and MPS;
- resident-bank tensor pointer digest and scratch pointer/capacity;
- fresh-state replay parity and final allocation release.

`tools/long_lived_metrics.py` exposes the same boundary accounting as a small
model-independent component. `tools/serve_openai.py` enables it only when
`K3_SERVER_METRICS_JSONL` is set. One JSON object is appended per request,
including memo hits and error/disconnect status. With the variable unset,
the server does not take MPS/VM/iostat snapshots.

The server now also obtains its tokenizer from the official local model
metadata path used by the direct-shard runtime instead of assuming a generated
`k3-meta` directory.

## Validation

```sh
set -a
source config/m3ultra512.env
set +a

.venv/bin/python tools/validate_m9_long_lived.py \
  --tokens 8 \
  --output bench-results/m9-long-lived-final.json
```

The run used normal buffered positional reads, did not purge the global page
cache, and disabled per-layer profile synchronizations after validating the
selected profile. Therefore these values describe natural cache evolution,
not globally cold SSD bandwidth.

## Request results

| Request | Unique working set | New process experts | Time/token | TTFT | Physical GB/token | Physical/logical |
|---|---:|---:|---:|---:|---:|---:|
| anchor-1 | 104.459 GB | 5,953 | 12.844 s | 9.451 s | 8.385 | 32.46% |
| unrelated-b | 120.286 GB | 5,224 | 13.683 s | 7.801 s | 9.161 | 35.47% |
| anchor-2 | 104.459 GB | 0 | 4.051 s | 4.033 s | 0.160 | 0.62% |
| unrelated-c | 124.094 GB | 4,103 | 12.900 s | 7.139 s | 8.494 | 32.88% |
| anchor-3 | 104.459 GB | 0 | 4.105 s | 4.060 s | 0.158 | 0.61% |

The first anchor read 67.084 GB. After unrelated-b expanded the cumulative
working set to 196.126 GB, anchor-2 read only 1.278 GB. Unrelated-c then
expanded the cumulative set to 268.122 GB, yet anchor-3 still read only
1.263 GB.

Previous-request route overlap does not explain this result. Anchor-2 shared
only 1,631 experts with the immediately preceding request and anchor-3 shared
1,561, while each anchor used 5,953 unique experts. The older anchor pages
remained in macOS's file cache across unrelated work.

Across all 40 tokens:

| Metric | Result |
|---|---:|
| Logical expert bytes | 1,033.183 GB |
| RAID-member physical reads | 210.862 GB |
| Physical/logical ratio | 20.41% |
| Mean physical reads | 5.272 GB/token |
| Mean wall time | 9.517 s/token |
| Cumulative unique expert set | 15,280 experts / 268.122 GB |

The member-read counter is the sum reported for both RAID devices. It is used
to distinguish cached reads from device traffic and is not an application
payload-bandwidth claim.

## Ownership and pressure

| Check | Result |
|---|---:|
| Runtime-loaded MPS current | 119,086,045,184 bytes |
| Post-request MPS current, requests 1–5 | 123,783,772,160 bytes each |
| Post-first request range | 0 bytes |
| Scratch preparations | 3,720 = 93 × 8 × 5 |
| Scratch pointer/capacity | unchanged |
| All 2,628 resident tensor pointers | unchanged |
| Compressor operations during requests | 0 |
| Swapouts during requests | 0 |
| Final MPS current delta from baseline | 0 bytes |
| Final MPS driver delta from baseline | +562,200,576 bytes |
| Swap before/after | 262,144 / 262,144 bytes |

No anonymous expert cache was introduced. The cache benefit comes from
ordinary file-backed pages belonging to the unchanged official safetensors.

## Server telemetry smoke

The ordinary server was started with:

```sh
K3_SERVER_METRICS_JSONL=bench-results/m9-server-smoke.jsonl \
K3_SERVER_MAX_TOKENS=1 \
K3_RESPONSE_MEMO_ENTRIES=0 \
.venv/bin/python tools/serve_openai.py --port 18080
```

The text `" Chrysler"` round-trips to the single official token ID 85001. A
one-token completion succeeded and emitted one telemetry row:

| Metric | Result |
|---|---:|
| Prompt / completion tokens | 1 / 1 |
| TTFT | 6.744 s |
| Request wall time | 6.861 s |
| Logical direct-slab bytes | 25.830 GB |
| RAID-member physical reads | 1.936 GB |
| Unique routed experts | 1,472 |
| MPS before / after | 119.086 / 123.784 GB |

This smoke retained `K3_PROFILE=1`, so it includes the profile
synchronizations and is connectivity evidence rather than the selected
performance result.

## Tests

```sh
.venv/bin/python -m unittest \
  tools.test_long_lived_metrics \
  tools.test_m9_long_lived \
  tools.test_m8_cache_crossover \
  tools.test_local_safetensors \
  tools.test_adaptive_prefetch \
  tools.test_expert_read \
  tools.test_prefetch_cancel \
  tools.test_apple_silicon
```

## Decision and M10

The natural file cache is effective enough that adaptive prefetch remains
disabled. A repeated 104.5 GB routed set reaches approximately 4.1
seconds/token and survives at least 163.7 GB of additional unique expert pages.
Speculative reads would consume storage bandwidth that is needed by the
currently cold unrelated request.

The next correctness boundary is multi-token prompt prefill. The selected
direct slab has a 16-expert layer capacity; a multi-position prompt can select
more than 16 unique experts in one layer and currently leaves the focused
direct-slab path. M10 should:

1. implement an official-shard-only prefill path for expert unions larger than
   16 without creating a persistent expert cache;
2. compare position-serial and bounded elastic-slab designs;
3. establish prompt-logit, route, KV/KDA state, and first decode-token parity
   for 2, 8, and chat-template-length prompts;
4. bound temporary MPS/CPU storage and release it after prefill;
5. then measure TTFT and physical bytes per prompt token.
