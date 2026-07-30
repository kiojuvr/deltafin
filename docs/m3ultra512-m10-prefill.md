# M3 Ultra 512 GB: M10 bounded direct-shard prefill

M10 keeps multi-token prompt prefill on the official-shard path. Routed layers
whose position union exceeds the 16-expert decode bank now use one lazy,
page-aligned, fixed-address prefill slab. No expert is converted, cached by
identity, or copied into a persistent heap tree.

The official 89-token `"Hello"` chat template completed through all 93 layers,
then produced its first decode logits. Its position routes, prompt logits,
decode logits, and every KV/KDA/conv state tensor were bit-for-bit identical to
the former official-shard heap path.

## Design

The existing two 16-expert banks remain the T=1 decode path. For a
multi-position routed layer:

1. the router's unique expert union is enumerated;
2. each official 17,547,264-byte expert span is read directly into a fixed
   page-aligned slot;
3. Metal wraps the slot with no staging copy;
4. all positions consume the mapping synchronously;
5. the next layer overwrites the same slots.

The bank has 896 slots because K3 has 896 routed experts per layer. Its virtual
capacity is 15,722,348,544 bytes, but anonymous pages are committed only for
slots actually touched. The runtime records both the current active count and
the high-water count. This is bounded layer scratch, not an expert cache:
slots do not retain `(layer, expert)` identity across loads.

`K3_METAL_POSITION_BATCH=1` submits the existing position-major Metal ABI for
T>1. The M3 Ultra profile enables it after exact cross-mode validation.

## Correctness oracle

`tools/validate_m10_prefill.py` executes each prompt twice:

- `elastic`: the new page-aligned prefill slab;
- `legacy-heap`: the previous direct official-shard reader and its temporary
  heap buffers.

Both modes use the same official files and model code. The validator compares:

- all position-ordered top-16 routes in all 92 routed layers;
- every FP32 prompt logit, not only the final position;
- the predicted first decode token and all its logits;
- SHA-256 digests, dtype, shape, and byte length of all recurrent, convolution,
  MLA key, and MLA value cache tensors after prefill and after decode;
- resident-bank and fixed-scratch pointer identity;
- official checkpoint fingerprints and final allocation release.

`K3_PREFILL_LAST_LOGIT=0` is used only in the validator so all prompt logits
remain observable. The ordinary runtime retains its final-logit-only output
head optimization.

## Commands

```sh
set -a
source config/m3ultra512.env
set +a

.venv/bin/python tools/validate_m10_prefill.py \
  --cases two eight \
  --position-batch \
  --output bench-results/m10-prefill-two-eight-final.json

.venv/bin/python tools/validate_m10_prefill.py \
  --cases chat \
  --position-batch \
  --output bench-results/m10-prefill-chat-final.json
```

All three cases passed with:

- exact position routes;
- exact prompt and first-decode logits with `max_abs == 0`;
- exact prefill and decode cache digests;
- zero HTTP requests;
- zero Metal staging copies on the elastic path;
- no compressor or swap growth;
- unchanged official checkpoint files.

## Routed union and slab size

| Prompt | Positions | Route edges | Union experts across 92 layers | Mean union/layer | Max union/layer | Max active slab |
|---|---:|---:|---:|---:|---:|---:|
| `The Chrysler` | 2 | 2,944 | 2,798 | 30.41 | 32 | 0.562 GB |
| Eight-token completion | 8 | 11,776 | 7,385 | 80.27 | 111 | 1.948 GB |
| Official `Hello` chat template | 89 | 131,008 | 34,099 | 370.64 | 550 | 9.651 GB |

The chat prompt read 598.344 GB of unique-per-layer prefill expert spans instead
of reading all 131,008 route edges separately. The union is 26.0% of the
un-deduplicated edge count.

## I/O and time

Each focused run includes one prompt prefill and one T=1 decode pass. Physical
bytes are natural-cache RAID-member deltas for that pair and are not globally
cold bandwidth measurements.

| Prompt | Elastic prefill | Total logical expert bytes | Total physical member reads |
|---|---:|---:|---:|
| 2 tokens | 36.059 s | 74.927 GB | 40.321 GB |
| 8 tokens | 98.393 s | 155.416 GB | 84.710 GB |
| 89-token chat | 655.657 s | 624.174 GB | 555.634 GB |

The chat prefill alone accounts for 598.344 GB logical. Its measured time
corresponds to roughly 0.91 GB/s of logical demand, consistent with the
physical USB RAID limit rather than the page-cache bandwidth seen by short
warm requests.

The legacy chat oracle ran second but still took 720.317 seconds and read
569.850 GB. Its large staging ownership displaced file-backed pages and caused
re-reads, so the two timings are useful as end-to-end ownership evidence but
not a controlled cold/warm A/B.

## Copy and memory ownership

| Prompt | Elastic Metal copies | Legacy Metal copies |
|---|---:|---:|
| 2 tokens | 0 | 2,944 |
| 8 tokens | 0 | 11,776 |
| 89-token chat | 0 | 131,008 |

For the chat case:

| Metric after prompt + decode | Elastic | Legacy heap |
|---|---:|---:|
| Process RSS | 15.487 GB | 46.756 GB |
| MPS current | 125.392 GB | 125.370 GB |
| Metal driver allocation | 136.855 GB | 153.419 GB |
| Cache state after prefill | 324 tensors / 0.737 GB | exact same |
| Cache state after decode | 324 tensors / 0.740 GB | exact same |
| Compressor operations | 0 | 0 |
| Swapouts | 0 | 0 |

The old path's route-edge staging grew RSS by more than 30 GB and Metal driver
allocation by another 16.6 GB relative to elastic. The elastic process stayed
near the 9.651 GB slab high-water plus ordinary runtime ownership throughout
the layer scan.

After explicit scratch, bank, wrapper, and MPS cache release:

| Check | Result |
|---|---:|
| MPS current delta from baseline | 0 bytes |
| MPS driver delta from baseline | +42,205,184 bytes |
| Swap before / after | 262,144 / 262,144 bytes |

## Ordinary server smoke

The promoted profile was exercised through `tools/serve_openai.py` with the
two-token raw prompt `"The Chrysler"` and one generated token. The request
completed successfully:

| Metric | Result |
|---|---:|
| Prompt / completion tokens | 2 / 1 |
| TTFT | 5.893 s |
| Request wall time | 6.020 s |
| Prefill direct-shard bytes | 49.097 GB |
| RAID-member physical reads | 0.888 GB |
| Prefill slab loads | 92 |
| Routed union experts | 2,798 |
| HTTP bytes / requests | 0 / 0 |

The server telemetry is saved at
`bench-results/m10-server-prefill-smoke.jsonl`.

## Selected profile

```text
K3_DIRECT_SLAB=1
K3_DIRECT_PREFILL_SLAB_EXPERTS=896
K3_METAL_POSITION_BATCH=1
```

The capacity is a hard bound. An invalid value or a union beyond the selected
capacity raises a clear error instead of falling back to HTTP, a persistent
expert cache, or an unbounded allocation.

## Bottleneck and M11

M10 removes file-format fallback and staging ownership from prefill. The first
chat request is now correct and bounded, but its 598 GB expert demand still
takes about eleven minutes on the current RAID.

Most of the 89-token chat template is a deterministic system prefix shared by
requests. M11 should validate an exact reusable prefix-state snapshot:

1. precompute the fixed system prefix once with the M10 slab;
2. retain only its approximately sub-GB KV/KDA/conv state, not expert weights;
3. clone/restore that state for each request and process only the user suffix;
4. compare full prompt logits, routes, first token, and state against the
   uncached 89-token path;
5. measure clone cost, state ownership, TTFT, compressor, and swap.

This targets the 598 GB repeated demand directly while preserving the official
checkpoint and natural file cache.
