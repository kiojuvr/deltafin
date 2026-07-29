# M3 Ultra 512 GB: M2 serial one-token parity

## Outcome

One real token now traverses all 93 official Kimi-K3 text layers through the
ordinary Deltafin runtime with:

- all 2,628 resident tensors held in the exact float32 MPS bank;
- 92 routed layers and 16 experts per routed layer;
- 1,472 selected experts and 25,829,572,608 logical expert bytes;
- the reusable direct-shard double slab and Metal MoE backend;
- no resident preload, expert prefetch, router pilot, grouped MoE, speculation,
  or next-layer overlap.

The direct pass was repeated with every selected expert supplied by
`safetensors.safe_open` instead of the direct slab reader. All 92 routing sets
and all 163,840 final logits matched bit-for-bit. The maximum absolute and
relative errors were both zero.

The input token was `1008` (`The`). Both paths produced token `220` (a space).
The float32 logit payload SHA-256 was:

```text
4065d212e8885a89bb5635145801f653bc2f6eeb31b93c6c90f66c7e1d7715d6
```

## Official-local bootstrap

The direct runtime no longer requires copied files under `k3-meta` or
`tools/k3pkg`. It mounts `K3_MODEL_DIR` as a read-only Python package and loads
Moonshot's configuration, modeling code, tokenizer code, and `tiktoken.model`
from that directory.

The legacy `k3loader.INV` interface is a lazy compatibility view over the same
header-derived `LocalSafetensorsStore` already owned by the direct reader. This
avoids expanding a second 497,220-entry inventory. Direct mode also skips the
legacy expert-cache scan and does not create an expert-cache directory.

The tokenizer is instantiated directly as `TikTokenTokenizer`; it does not use
Transformers' dynamic-module cache and therefore does not copy official Python
files into `~/.cache/huggingface`.

## Reproduction

```sh
set -a
source config/m3ultra512.env
set +a

.venv/bin/python tools/validate_m2_serial_token.py \
  --token-id 1008 \
  --output bench-results/m2-serial-token.json
```

The JSON includes all routed expert IDs, 93 per-layer timing rows, MPS/VM/swap
snapshots, RAID-member counters, final logits, source fingerprints, and release
counters. It is machine-specific, rebuildable, and ignored by Git.

## Strict warm run

The strict direct and standard-reader passes ran back-to-back after the first
physical-I/O run had populated the macOS page cache.

| Measurement | Direct slab | Standard reader |
|---|---:|---:|
| Full forward wall | 4.336 s | 6.640 s |
| Sum of 93 layer walls | 3.856 s | 6.587 s |
| Expert read time | 0.619 s | 2.326 s |
| Non-expert-read layer time | 3.237 s | 4.261 s |
| Metal MoE time | 0.314 s | 1.182 s |
| Physical RAID bytes | 0 | 0 |
| Logical expert bytes | 25,829,572,608 | 25,829,572,608 |

The direct reader delivered the warm logical expert stream at 42.0 GB/s. This
is RAM/page-cache-to-slab bandwidth and is not physical SSD throughput.
The standard reader is intentionally retained as a correctness oracle, not a
runtime performance path.

## Cold candidate and page-cache effect

The immediately preceding first full-token pass transferred
25,790,219,551 bytes on RAID members for 25,829,572,608 requested expert bytes.
Expert read wait was 29.86 seconds, or 0.865 GB/s. This agrees with the earlier
physical storage measurements.

That first run still had the legacy router-pilot default enabled, but the pilot
issued zero speculative reads because the direct runtime has no `fetch_v2`
queue. Its physical expert-byte and expert-wait measurements remain usable;
its total wall and memory figures are not the final strict serial baseline.
The profile now explicitly sets `K3_PILOT=0`, `K3_PREFETCH=0`, and
`K3_MOE_GROUP_SIZE=0`.

The two runs demonstrate the two operating regimes:

- physical miss: about 30 seconds of expert I/O per token;
- complete page-cache hit: about 0.6 seconds of expert delivery and 4.3 seconds
  for the complete forward.

## Memory ownership and release

Before the strict direct pass, MPS current allocation was 228,764,056,576
bytes. The complete one-token pass increased it by 494,272,512 bytes. Swap and
compressor occupancy did not grow.

After deleting layer/cache references, releasing the resident bank and slabs,
and calling `torch.mps.empty_cache()`:

- MPS current allocation returned to zero;
- driver allocation was 842,956,800 bytes above baseline;
- process RSS was 1.39 GB;
- swap remained 262,144 bytes.

The index and all 96 official shards retained identical size and modification
time fingerprints. No official file was written, converted, or copied.

## Current bottleneck and next step

Cold expert I/O is now the dominant term: 29.86 seconds versus about 3.24
seconds of warm non-expert-read layer time. The next performance experiment
should retain this exact serial result as the oracle and measure:

1. multiple consecutive decode tokens with physical bytes and cache hits per
   token;
2. layer `N` compute overlapped with layer `N+1` direct-slab reads;
3. output-logit parity after overlap is enabled;
4. routing locality and page-cache survival under the actual resident working
   set.

BF16 resident storage remains deferred until the overlapped float32 path is
exact.
