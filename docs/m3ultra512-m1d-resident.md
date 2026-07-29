# M3 Ultra 512 GB: M1-D complete resident MPS ownership

## Outcome

The complete official Kimi-K3 non-routed inventory can coexist on MPS in the
current exact float32 path. The validator loads one source tensor at a time from
the unmodified official shards, retains only the MPS tensor, and advances in
eight-layer stages.

The validated inventory is:

- 2,628 resident tensors;
- 114,404,258,816 checkpoint bytes;
- 228,764,027,904 float32 MPS tensor bytes;
- 2,122 BF16 and 506 F32 source tensors.

Layer 1, layer 46, and layer 92 were rebound directly to bank storage without a
copy. All recorded intermediate stages and final layer outputs matched an
independent `safetensors.safe_open` materialization bit-for-bit. A committed
561,512,448-byte aligned double expert slab coexisted with the complete bank.

No official model file was opened for writing, converted, copied, or replaced.
Metadata fingerprints for the index and all 96 shards were unchanged.

## Components

- `ResidentTensorBank` in `tools/resident_shard_loader.py` owns device tensors,
  stages the complete inventory, aliases layer parameters to bank storage, and
  releases all ownership explicitly.
- `tools/validate_m1d_resident.py` records every load stage, MPS allocator
  counters, process RSS/VSZ, Darwin VM counters, swap, RAID-member I/O,
  `vm_stat`, and `vmmap -summary`.
- `tools/direct_shard_loader.py` leases one of two aligned expert banks through
  synchronous MoE completion.
- `tools/kimi_run.py` can build the complete resident bank at startup and uses
  the resident embedding directly instead of creating another weight copy.

The fixed profile selects buffered positional I/O:

```sh
K3_PREAD_NOCACHE=0
K3_RESIDENT_BANK=1
K3_DIRECT_SLAB=1
K3_DTYPE=fp32
K3_APPROX=0
K3_INT8_LM_HEAD=0
K3_SPINE_PACK=0
K3_SPEC_DEPTH=1
PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0
PYTORCH_MPS_LOW_WATERMARK_RATIO=0.9
K3_TEMPLATES=0
K3_PIN_LAYERS=93
K3_PRELOAD=0
K3_EXPERT_PREFETCH=0
```

`F_NOCACHE` remains available as a measurement control but is not the normal
runtime path. It does not evict pages already present in RAM. The correctness
profile also disables template copies and speculative preload: all 93 ordinary
layer objects alias the bank, and the first serial-token run will have no
next-layer expert overlap.

## Reproduction

```sh
set -a
source config/m3ultra512.env
set +a

.venv/bin/python tools/validate_m1d_resident.py \
  --model-dir /Volumes/USB-SSD-RAID-0/models/moonshotai/Kimi-K3 \
  --sidecar bench-results/k3-direct-shards-index.json.gz \
  --output bench-results/m1d-resident-mps-30min.json \
  --stability-seconds 1800 \
  --sample-seconds 60
```

The generated JSON is machine-specific, rebuildable, and ignored by Git.

## Full-load ownership

The first complete run was cold-dominant for the language layers:

| Measurement | Result |
|---|---:|
| MPS tensor bytes | 228,764,027,904 |
| MPS driver allocation | 234,634,035,200 |
| Metal recommended maximum | 498,216,206,336 |
| Process RSS | 5,163,679,744 |
| `vmmap` physical footprint | 223.6 GiB |
| Double expert slab | 561,512,448 bytes |
| Swap used | 262,144 bytes |
| Compressor occupied | 308,051,968 bytes |
| Free pages after load | 114,849,218,560 bytes |
| File-backed pages after load | 185,664,094,208 bytes |

`current_allocated_memory()` exactly equaled the computed tensor payload.
The ordinary RSS was only 5.16 GB; `vmmap` attributed the large ownership to
IOAccelerator/graphics regions. There was no retained 114 GB or 228 GB CPU
weight tree.

After releasing 2,628 tensors and calling `torch.mps.empty_cache()`:

- current allocation returned to zero;
- driver allocation was only 344,064 bytes above the original baseline;
- the official checkpoint fingerprint remained unchanged.

## Physical I/O versus page cache

The RAID members are `disk8` and `disk9`, each behind a 5 Gbit/s USB link.
The validator samples their cumulative `iostat -Id` counters around every
stage.

For the cold-dominant complete load:

| Measurement | Result |
|---|---:|
| Logical checkpoint bytes requested | 114,404,258,816 |
| RAID-member bytes transferred | 108,111,750,759 |
| Inferred page-cache contribution | 6,292,508,057 |
| Total tensor load time | 136.55 s |
| Process-observed logical rate | 0.838 GB/s |
| Physical member rate | 0.792 GB/s |

This is consistent with the USB RAID limit and corrects the earlier
interpretation of 6.89 GB/s. The latter, and the 20–27 GB/s slab observations,
are RAM/page-cache delivery rates rather than physical SSD throughput.

An immediate repeated complete load took 27.47 seconds at an apparent
4.16 GB/s while the RAID members transferred only about 2 MB. That run is a
direct demonstration of the macOS page-cache effect, not a storage benchmark.

## Layer parity and storage aliasing

| Layer | Resident tensors | Result |
|---:|---:|---|
| 1 | 28 | all stages bitwise exact |
| 46 | 28 | all stages bitwise exact |
| 92 | 22 | all stages bitwise exact |

Every bound parameter retained the exact `data_ptr()` of its tensor in
`ResidentTensorBank`; constructing the layer introduced no second MPS weight
copy. Independent reference-layer allocations were released back to the bank
baseline after each comparison.

## Stability and next step

The 30-minute run retains all resident tensors and both committed expert banks,
sampling once per minute. It produced 31 samples over 1,807.97 seconds:

| Counter | Start | End | Delta |
|---|---:|---:|---:|
| MPS current allocation | 228,764,027,904 | 228,764,027,904 | 0 |
| MPS driver allocation | 234,634,379,264 | 234,634,379,264 | 0 |
| Swap used | 262,144 | 262,144 | 0 |
| Compressor occupied | 308,051,968 | 305,283,072 | -2,768,896 |
| Free pages | 117,669,904,384 | 117,995,388,928 | +325,484,544 |
| File-backed pages | 186,218,496,000 | 186,299,252,736 | +80,756,736 |

There was no continuing swap or compressor growth. After release,
`current_allocated_memory()` returned exactly to baseline and driver allocation
was 344,064 bytes above baseline. The process RSS after release was 1.46 GB.

The ordinary direct-shard adapter was also exercised through its bank lease,
not only through the focused slab class. Two Metal calls recorded 32 zero-copy
wraps and zero copies, both matched the CPU kernel, and bank A retained the same
addresses when reused for layer 92. The retained evidence is
`bench-results/direct-slab-adapter.json`.

The overlap-disabled serial one-token milestone is now complete. See the
[M2 serial-token report](m3ultra512-m2-serial-token.md): the ordinary direct
runtime and a standard safetensors expert reader produced identical routes and
bitwise-identical final logits. The next experiment is multi-token
page-cache accounting followed by next-layer read/compute overlap. BF16
resident storage remains deferred until the overlapped float32 path is exact.
