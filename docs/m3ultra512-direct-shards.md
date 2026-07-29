# M3 Ultra 512 GB: official Kimi-K3 direct-shard reader

## Scope and outcome

This phase establishes a read-only path from Kimi-K3 router selections to the
official, unmodified safetensors payloads. It does not attempt complete K3
inference. The new path does not use HTTP range requests, `k3-experts`, or a
converted weight tree.

`LocalSafetensorsStore` parses `model.safetensors.index.json` and every
referenced shard header, records each tensor's shard, absolute file offset, byte
length, dtype, and shape, and services payload reads with `os.pread`. A routed
expert is coalesced into one read when its tensors are adjacent. A header-derived
multi-span fallback handles gaps and shard boundaries.

The `direct_shard_loader` adapter exposes the existing
`k3loader.fetch_experts` contract. `config/m3ultra512.env` selects it through
`K3_EXPERT_SOURCE=direct-shards`; other profiles retain the existing
cache/HTTP path.

## Components

- `tools/local_safetensors.py`: independent inventory, sidecar, positional I/O,
  coalescing, fallback, parallel reads, dtype/shape restoration, explicit
  descriptor lifecycle, and clear lookup errors.
- `tools/direct_shard_loader.py`: lazy singleton and compatibility adapter for
  Deltafin's existing CPU/Metal raw-MXFP4 consumers.
- `tools/test_local_safetensors.py`: synthetic standard-format shards covering
  contiguous groups, a gap, a cross-shard expert, standard-reader equality,
  parallel reads, adapter output, sidecar validation, errors, and FD closure.
- `tools/validate_direct_shards.py`: full K3 structural validation,
  `safetensors.safe_open` comparison, sidecar timing, FD audit, and I/O
  benchmarks.

## Observed official Kimi-K3 layout

The local official release contains 96 shards, 497,220 tensors, and
1,560,860,324,864 tensor payload bytes. Routed MoE occupies layers 1 through 92;
layer 0 is dense. Every routed layer contains expert IDs 0 through 895, for
82,432 experts and 494,592 expert tensors.

Each expert has this exact physical order:

| Tensor | dtype | shape | bytes |
|---|---:|---:|---:|
| `w1.weight_packed` | U8 | 3072 × 1792 | 5,505,024 |
| `w1.weight_scale` | U8 | 3072 × 112 | 344,064 |
| `w2.weight_packed` | U8 | 3584 × 1536 | 5,505,024 |
| `w2.weight_scale` | U8 | 3584 × 96 | 344,064 |
| `w3.weight_packed` | U8 | 3072 × 1792 | 5,505,024 |
| `w3.weight_scale` | U8 | 3072 × 112 | 344,064 |

All six tensors are adjacent with zero gaps, giving one 17,547,264-byte span
per expert. All 82,432 experts satisfy this property and none crosses a shard.
Each routed layer occupies one contiguous 896-expert block in one shard:
layer 1 is in shard 2, and this continues through layer 92 in shard 93. Expert
blocks are physically ordered by the tensor-name string (`0, 1, 10, 100, ...`),
not numeric expert ID.

The implementation does not depend on those regularities for correctness. The
synthetic tests prove the multi-read fallback for a gapped expert and for an
expert split over two shards.

## Reproduction and correctness

Run:

```sh
set -a
source config/m3ultra512.env
set +a
.venv/bin/python -m unittest tools/test_local_safetensors.py -v
.venv/bin/python tools/validate_direct_shards.py
```

The real-model validation checks all 82,432 layouts and all 494,592 expert
tensors structurally. It compares layers 1, 46, and 92 and expert IDs 0, 448,
and 895: 54 tensors and 157.9 MB total. For every tensor, dtype, shape, raw
bytes, and reconstructed NumPy values match `safetensors.safe_open`.

The validation also checks index/header agreement, declared total size,
non-overlapping tensor ranges, every range against its shard size, invalid
layer/expert/tensor errors, sidecar round-trip, and process/store FD counts.
The observed FD count was 4 before and 4 after, with zero descriptors retained
by the closed store.

The generated sidecar contains the complete inventory and source fingerprint,
is 4.9 MB gzip-compressed, and can be regenerated from the official files. It
contains no weight bytes. Raw header/index parsing is currently faster than
sidecar loading, so the fixed profile does not require the sidecar.

## M3 Ultra measurement

Evidence is written to
`bench-results/m3ultra512-direct-shards.json`; the sidecar is
`bench-results/k3-direct-shards-index.json.gz`. These generated files are
ignored by Git.

The retained run uses a shard not touched by the standard-reader comparison.
Single-expert figures use ordinary buffered `pread`; 16-expert storage figures
use Darwin `F_NOCACHE` on the shared shard descriptor. No privileged global
cache purge was performed, so “first” is explicitly a cold candidate.

| Measurement | Result | Effective bandwidth |
|---|---:|---:|
| Index + 96 headers + complete inventory | 2.42 s | — |
| Sidecar generation | 0.786 s | — |
| Sidecar reload | 2.58 s | — |
| Single expert, first buffered read | 21.4 ms | 0.820 GB/s |
| Single expert, immediate second read | 1.75 ms | 10.00 GB/s |
| Same expert, repeated median | 1.16 ms | 15.17 GB/s |
| 16 experts, sequential `F_NOCACHE` | 322 ms | 0.871 GB/s |
| 16 experts, 8-way parallel `F_NOCACHE` | 41.5 ms | 6.77 GB/s |

The immediate second read was 12.2 times faster than the first candidate,
demonstrating why buffered and physical-read measurements must not be mixed.

## Current bottleneck and next phase

The direct storage path reaches 6.77 GB/s with eight positional reads, while a
single sequential stream reaches only 0.871 GB/s. Physical I/O parallelism is
therefore still essential. Warm page-cache reads are much faster, but the
1.45 TB expert set cannot be treated as resident even on a 512 GB machine.
The correctness-first reader also returns Python-owned byte buffers. They are
contiguous but not guaranteed to be page-aligned or reusable, so the current
Metal wrapper may take its safe scratch-copy path. Recyclable, page-aligned
expert slabs are a runtime optimization for the next phase, not part of the
validated on-disk index.

The next implementation sequence should be:

1. Build a read-only resident-spine loader over the same official inventory,
   preserving BF16 and without creating `k3-resident`.
2. Allocate the M3 Ultra memory budget explicitly: resident spine, recurrent
   KDA state, reusable 16-expert double buffers, Metal arenas, and OS headroom.
3. Feed `direct_shard_loader` buffers into the existing MXFP4 CPU and Metal MoE
   paths and validate one real routed layer against the current cache path.
4. Validate every resident tensor against `safetensors.safe_open`, then execute
   a single layer with recorded router IDs.
5. Assemble one-token inference with profiling enabled, first serially for
   correctness and then with next-layer read overlap. Keep HTTP/cache as an A/B
   oracle until token/logit parity is established.

The sidecar should remain optional unless its representation is changed to load
faster than the validated raw-header path.
