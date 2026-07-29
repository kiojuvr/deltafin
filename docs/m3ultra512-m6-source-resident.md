# M3 Ultra 512 GB: M6 checkpoint-native resident ownership

M6 adds an opt-in resident representation that keeps every non-routed tensor
in its official checkpoint dtype on MPS. The FP32 exact execution path is
unchanged: a layer's BF16 tensors are converted to FP32 immediately before that
layer runs and the layer module is returned to meta storage immediately after
it runs.

This reduces the permanent resident payload from 228,764,027,904 bytes to
114,404,258,816 bytes without quantizing or rewriting the checkpoint. The
saved 114,359,769,088 bytes remain available to macOS for the official shard
page cache.

The mode remains opt-in while its page-cache crossover is measured over longer,
realistic routing sequences. The established FP32 bank remains the
`config/m3ultra512.env` default.

## Implementation

- `ResidentTensorBank(dtype=None)` preserves the source dtype independently for
  every tensor and reports the cumulative load time and owned byte count.
- `K3_RESIDENT_BANK_DTYPE=source` selects the checkpoint-native bank.
- Source mode requires `K3_PIN_LAYERS=0`. A positive pin count is rejected
  before the bank is loaded because it would retain a second FP32 layer tree.
- The ordinary non-template runtime materializes one layer into FP32, executes
  it, and dematerializes it. The M6 validator checks after every token that all
  93 layer modules own zero non-meta parameter tensors.
- `LazyEmbed` keeps the complete embedding in BF16 and converts only selected
  rows to the runtime dtype.
- The LM head and small tail tensors retain their existing exact FP32 runtime
  behavior.

The official checkpoint is still read-only. No converted weights, resident
tensor files, or expert cache are created.

## Official resident inventory

| Source dtype | Tensors | Bytes |
|---|---:|---:|
| BF16 | 2,122 | 114,359,769,088 |
| F32 | 506 | 44,489,728 |
| Total | 2,628 | 114,404,258,816 |

The old forced-FP32 representation is 228,764,027,904 bytes. Source mode's
materialized byte count exactly equals the checkpoint byte count.

## Validation

The final run used the normal M3 Ultra profile, overrode only resident storage
and layer pinning, and compared against the saved eight-token M4 FP32 evidence:

```sh
set -a
source config/m3ultra512.env
set +a

.venv/bin/python tools/validate_m6_source_resident.py \
  --tokens 8 \
  --token-id 1008 \
  --oracle-evidence bench-results/m4-adaptive-final.json \
  --output bench-results/m6-source-resident-final.json
```

The validator ran serial-first, serial-warm, and overlap-warm sequences. For
all eight generated tokens it checked:

- exact input and output token IDs;
- all 92 routed-layer expert selections;
- the SHA-256 of all 163,840 FP32 logits;
- in-process serial repeat and overlap parity with `max_abs == 0`;
- zero layer-owned runtime tensors after every token;
- resident bank byte ownership and source dtype counts;
- MPS allocator and swap return-to-baseline behavior;
- before/after fingerprints of every official checkpoint file.

The generated sequence was:

```text
220, 802, 24, 12, 2975, 15, 3629, 1389
```

The focused unit suite passed 22/22 tests:

```sh
.venv/bin/python -m unittest \
  tools.test_local_safetensors \
  tools.test_adaptive_prefetch \
  tools.test_expert_read \
  tools.test_prefetch_cancel \
  tools.test_apple_silicon
```

Repository-wide `unittest discover` is not currently a clean gate: it ran 69
tests but reported 11 collection errors because several legacy tests require
the intentionally absent converted `k3-meta` tree or vendored `tools/k3pkg`,
and two script-style checks call `sys.exit(0)` during import. No M6-focused test
failed.

Results:

| Check | Result |
|---|---:|
| Saved FP32 oracle parity | 8/8 exact |
| Serial repeat parity | 8/8, max abs 0 |
| Serial/overlap parity | 8/8, max abs 0 |
| Layer-owned tensors after each token | 0 |
| Official model tree changed | no |
| Model conversion performed | no |
| MPS current allocation after release vs baseline | 0 bytes |
| MPS driver allocation after `empty_cache()` vs baseline | +562,200,576 bytes |
| Swap used before / after | 262,144 / 262,144 bytes |

The final process peak physical footprint reported by `vmmap` was about
123.0 GB. The first sequence began with exactly 114,404,258,816 bytes in
`current_allocated_memory()`. After the persistent exact LM head and tail had
been created, steady current allocation was about 119.10 GB. Per-token
activation/output storage temporarily raised the end-of-token snapshot to
about 120.06 GB; no layer weight tree remained.

## Performance

This run started with checkpoint pages already warm. Its 114.404 GB source bank
load took 21.017 seconds, an apparent 5.443 GB/s. This is a logical,
page-cache-assisted rate, not physical USB RAID bandwidth.

Warm serial decode:

| Metric | FP32 bank (M4 evidence) | Source bank (M6) |
|---|---:|---:|
| Mean wall time/token | 3.642 s | 4.029 s |
| Resident materialization/token | ~0 s | 0.379 s |
| Expert wait/token | 0.605 s | 0.617 s |
| Physical RAID bytes, 8 tokens | 1.16 MB | 4.58 MB |

The measured warm cost is 0.387 seconds/token, or about 10.6%. Nearly all of
that cost is the 93 BF16/F32-to-FP32 layer materializations. This trade saves
114.36 GB of permanent MPS weight storage.

Process-visible slab bandwidth and physical reads must continue to be reported
separately. This run is not a controlled global-cold comparison and is not used
to claim a cache hit-rate crossover.

## Current bottleneck and M7

M6 changes the primary optimization target. Resident capacity is no longer the
constraint; the fixed per-token source-to-FP32 conversion is.

M7 should add a reusable, shape-classed FP32 layer scratch arena:

1. allocate only the largest KDA and MLA resident layouts, not 93 copies;
2. copy/convert checkpoint-native bank tensors into stable offsets;
3. bind layer parameters as views without allocator churn;
4. overlap preparation of layer N+1 with compute of layer N;
5. retain the M6 post-token zero-ownership and eight-token oracle checks;
6. run long paired FP32-bank/source-bank routing traces while recording
   physical RAID bytes, logical slab bytes, page-cache bytes, and wall time.

The source bank should become the 512 GB profile default only after that paired
test shows that recovered page-cache capacity repays the remaining conversion
cost under representative routing.
