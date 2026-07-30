# M3 Ultra 512 GB: M7 fixed resident scratch

M7 replaces M6's per-tensor FP32 allocations with a fixed, reusable layer
arena. The official resident bank remains checkpoint-native BF16/F32 and is
still the sole permanent owner of weights. The selected serial path uses one
4,681,786,368-byte FP32 slot.

The arena removes allocator churn and reduces measured materialization enqueue
time by about 91%. An experimental two-slot worker path successfully prepared
layer N+1 without blocking the main thread, but did not improve end-to-end time
and is retained default-off.

## Implementation

`ResidentScratchArena` provides:

- a 256-byte-aligned layout sized to the largest real K3 layer;
- fixed MPS storage whose pointer and capacity do not change;
- cached source tensor references and per-layer layouts;
- pre-created FP32 views and `nn.Parameter` objects;
- reusable meta parameters for dematerialization without allocation;
- one selected serial slot or two alternating overlap slots;
- cumulative preparation, source-byte, and materialized-byte counters;
- explicit release before the source resident bank is released.

The ordinary runtime selects the arena with:

```text
K3_RESIDENT_BANK_DTYPE=source
K3_PIN_LAYERS=0
K3_RESIDENT_SCRATCH=1
K3_RESIDENT_SCRATCH_SLOTS=1
K3_RESIDENT_SCRATCH_OVERLAP=0
```

Invalid combinations are rejected before loading the 114 GB bank. Scratch mode
requires the direct official checkpoint, source-dtype bank, BF16 spine, FP32
runtime, private layer modules, and no pinned layers.

## Validation

The selected one-slot run:

```sh
set -a
source config/m3ultra512.env
set +a

.venv/bin/python tools/validate_m7_resident_scratch.py \
  --tokens 8 \
  --token-id 1008 \
  --oracle-evidence bench-results/m4-adaptive-final.json \
  --output bench-results/m7-resident-scratch-serial-final.json
```

The same eight-token check was repeated without the per-layer synchronizations
inserted by `K3_PROFILE=1`:

```sh
.venv/bin/python tools/validate_m7_resident_scratch.py \
  --runtime-profile-off \
  --tokens 8 \
  --token-id 1008 \
  --oracle-evidence bench-results/m4-adaptive-final.json \
  --output bench-results/m7-resident-scratch-profile-off-final.json
```

The rejected overlap experiment:

```sh
.venv/bin/python tools/validate_m7_resident_scratch.py \
  --resident-scratch-overlap \
  --tokens 8 \
  --token-id 1008 \
  --oracle-evidence bench-results/m4-adaptive-final.json \
  --output bench-results/m7-resident-scratch-overlap-final.json
```

Both runs passed:

- eight exact input and output token IDs;
- all 92 routed-layer expert selections for every token;
- SHA-256 parity for all 163,840 FP32 logits;
- serial-repeat and direct-expert-overlap parity with `max_abs == 0`;
- zero layer-owned non-meta tensors after every token;
- exactly 2,232 preparations: 93 layers × 8 tokens × 3 sequences;
- unchanged slot pointers and storage size;
- unchanged official checkpoint fingerprints;
- MPS current allocation returning exactly to baseline;
- unchanged swap use.

The profile-off run also passed all eight oracle tokens. This verifies that
one-slot reuse is ordered correctly by the ordinary MPS command path and does
not depend on the validator's per-layer synchronization.

Focused tests passed 23/23:

```sh
.venv/bin/python -m unittest \
  tools.test_local_safetensors \
  tools.test_adaptive_prefetch \
  tools.test_expert_read \
  tools.test_prefetch_cancel \
  tools.test_apple_silicon
```

## Ownership and memory

| Item | Bytes |
|---|---:|
| Source resident bank | 114,404,258,816 |
| Selected FP32 scratch slot | 4,681,786,368 |
| Combined permanent weight/scratch payload | 119,086,045,184 |
| Recovered vs full FP32 bank | 109,677,982,720 |

The one-slot pointer was unchanged across all 2,232 preparations. The final
counter recorded 2,611,485,057,024 source bytes and 5,221,902,360,576 FP32
materialized bytes passing through that same slot.

After the exact LM head and tail were created, the serial-warm sequence began
at 123,783,772,160 bytes of MPS current allocation and ended each token near
124,634,880,000 bytes. The process peak physical footprint was about 123.6 GB.
After explicit scratch and bank release plus `torch.mps.empty_cache()`:

| Check | Result |
|---|---:|
| MPS current delta from baseline | 0 bytes |
| MPS driver delta from baseline | +562,200,576 bytes |
| Swap before / after | 262,144 / 262,144 bytes |

## Performance

All values below are page-cache-warm, profile-off serial sequences over the
same eight tokens. They do not represent physical cold RAID bandwidth.

| Metric | M6 dynamic source | M7 fixed serial |
|---|---:|---:|
| Mean wall time/token | 3.969 s | 3.863 s |
| Eight-token wall time | 31.754 s | 30.901 s |
| Resident enqueue/token | 0.386 s | 0.034 s |
| Physical RAID bytes | 3.71 MB | 1.28 MB |

The fixed arena reduced recorded resident enqueue time by 91.1% and improved
warm wall time by 2.7%. The smaller wall improvement shows that completion of
the BF16-to-FP32 device conversion is largely charged at the following device
synchronization/command dependency rather than at the Python enqueue call.

With diagnostic per-layer profiling enabled, M6 measured 4.029 seconds/token
and one-slot M7 measured 3.956 seconds/token, a 1.8% improvement.

The two-slot worker experiment produced:

| Metric | One-slot serial | Two-slot worker |
|---|---:|---:|
| Mean wall time/token | 3.956 s | 3.962 s |
| Main-thread scratch wait/token | 0 | 0.00016 s |
| Scratch storage | 4.682 GB | 9.364 GB |

The worker removed visible preparation waits but competed with layer compute
for Metal/unified-memory bandwidth. It was 0.1% slower in this paired run while
consuming another 4.682 GB, so one-slot serial is the selected M7 result.

## Current bottleneck and M8

M7 establishes stable ownership and removes the Python/MPS allocator path.
The remaining source-bank penalty is device conversion bandwidth and the
synchronization boundary, not allocation.

M8 should now measure the actual cache-capacity crossover:

1. run longer, identical routing traces with the 228.8 GB FP32 bank and the
   119.1 GB source-plus-scratch path;
2. record logical expert bytes, physical RAID bytes, cache-inferred bytes,
   wall time, compressor, and swap continuously;
3. separate warm, naturally cold, and repeated-route windows without claiming
   that `F_NOCACHE` globally purges macOS pages;
4. determine how many physical bytes/token the additional 109.7 GB absorbs;
5. promote source-plus-scratch to `config/m3ultra512.env` only if the saved
   physical I/O repays its remaining warm compute cost.
