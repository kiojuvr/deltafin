# M3 Ultra 512 GB: M1 direct resident and one-layer parity

## Outcome

M1-A through M1-C now have a correctness path over the unmodified official
Kimi-K3 checkpoint:

- non-routed tensors are located with the M0 header inventory and read with
  positional I/O;
- a meta-constructed official decoder layer can be materialized one tensor at a
  time, without a second resident weight tree;
- selected embedding rows can be read without loading the 2.35 GB embedding;
- one real routed decoder layer runs through the official Kimi-K3 modeling code
  and the existing native MXFP4 expert kernel;
- two fixed 16-expert anonymous-memory banks are filled directly by `preadv`;
- every expert slot is contiguous and 16 KiB page-aligned, and reloading a bank
  leaves every slot address unchanged.

`config/m3ultra512.env` selects both expert and resident direct-shard adapters.
It also sets `K3_SPINE=bf16` so an unrelated converted int8 tree cannot silently
take precedence.

## Components

- `tools/resident_shard_loader.py` classifies resident versus routed tensors,
  performs full and row-selective reads, and materializes a requested module
  from meta storage.
- `tools/expert_slab.py` owns the fixed-capacity page-aligned bank and its
  packed/scale NumPy views. A contiguous official expert is one positional
  read; the header-derived multi-span fallback remains correct.
- `tools/validate_m1_layer.py` compares direct reads with
  `safetensors.safe_open`, runs the official layer twice, compares semantic
  stages, measures memory and stage timings, and checks the slab/Metal buffer
  contract.
- `tools/kimi_run.py` accepts `K3_RESIDENT_SOURCE=direct-shards` as a
  compatibility adapter. The old resident and HTTP/cache paths remain the
  default for other profiles.

No official model file is opened for writing. No weight conversion, model
copy, expert cache, or other large derived artifact is produced. The only
generated M1 evidence is a small JSON file under `bench-results/`.

## Reproduction

```sh
set -a
source config/m3ultra512.env
set +a

.venv/bin/python -m unittest -v tools.test_local_safetensors

.venv/bin/python tools/validate_m1_layer.py \
  --model-dir /Volumes/USB-SSD-RAID-0/models/moonshotai/Kimi-K3 \
  --sidecar bench-results/k3-direct-shards-index.json.gz \
  --output bench-results/m1-layer-parity.json \
  --layer 1 \
  --threads 8
```

The retained Metal-visible result is
`bench-results/m1-layer-parity-metal.json`; the sandboxed CPU fallback is
`bench-results/m1-layer-parity.json`. Both are ignored by Git because they are
machine-specific and reproducible.

## Real layer 1 layout and parity

Layer 1 has 28 non-routed tensors, all in shard 2. Their checkpoint payload is
1,267,744,256 bytes (1.181 GiB): 21 BF16 tensors and seven F32 tensors. A
float32 CPU materialization is 2,534,844,928 bytes. The set contains the router
and score bias, latent down/up projections and norm, attention/KDA and AttnRes
weights, the shared expert, and the layer norms.

The validation also positionally reads embedding rows 0, 81,920, and 163,839.
The resulting BF16 tensor has shape `3 × 7168`, consumes 43,008 bytes, and
matches `safe_open` exactly.

For the deterministic one-token input, the router chose:

```text
712, 461, 308, 828, 846, 348, 154, 617,
521, 511, 753, 133, 503, 675, 692, 715
```

The standard and direct-shard runs matched bit-for-bit at every recorded point:
attention residual input, input norm, attention output, post-attention prefix,
MoE residual input, post-attention norm, router logits, top-k IDs and weights,
latent input, routed-expert output and projection, shared-expert output, MoE
sum, and final layer output. Every floating comparison had maximum absolute and
relative error zero.

## Retained measurements

These values are a correctness-run snapshot, not a cold-storage benchmark.
Earlier validation and each stage can warm different pages, so the M0
`F_NOCACHE` measurements remain the physical-I/O reference.

| Measurement | Result |
|---|---:|
| Direct layer checkpoint reads | 192.0 ms |
| Direct layer CPU-to-MPS transfer | 118.7 ms |
| Direct layer materialization total | 315.0 ms |
| Direct official MPS layer forward | 44.9 ms |
| Expert read inside direct forward | 7.97 ms |
| Native CPU MXFP4 parity expert compute | 13.2 ms |
| Other MPS layer work | 23.7 ms |
| Double-slab allocation | 561,512,448 bytes |
| Warm bank A fill, 16 experts | 10.2 ms / 27.5 GB/s |
| Bank B fill, layer 2 | 12.5 ms |
| Bank A Metal expert compute overlapped with bank B read | 5.75 ms compute / 12.54 ms wall |
| Reused bank A fill, layer 3 | 8.65 ms |

One direct float32 layer owns 2,534,844,928 bytes of weights. PyTorch reported
2,534,902,272 MPS bytes after materialization and 57,344 bytes after releasing
the layer, so the measured MPS overhead above the weight payload was 57,344
bytes. The direct loader retains no layer-sized CPU checkpoint dictionary: its
largest transient source tensor is 176,160,768 bytes. The two long-lived expert
banks add 561,512,448 bytes of CPU-addressable unified memory, and their native
Metal wraps add no scratch copy. Peak process RSS rose from 213 MB to 3.474 GB.
The checkpoint representation of the layer is 1,267,744,256 bytes. Peak RSS is
monotonic and allocator reuse is involved, so these figures must not be added
to peak RSS as if they were independent snapshots.

The complete non-routed inventory contains 2,628 tensors: 2,122 BF16 and 506
F32. Its official checkpoint payload is 114,404,258,816 bytes. Materializing
every tensor into the current exact float32 runtime representation would require
228,764,027,904 bytes before recurrent state, allocator overhead, Metal arenas,
expert slabs, or file cache. M1-D therefore must budget approximately 228.8 GB,
not only the 114.4 GB on-disk payload, unless the resident compute path gains a
verified BF16 representation.

## Metal status

All 16 bank A experts passed the existing `metal_moe._span_ptr` contract:
the six arrays form one 17,547,264-byte span, the returned native span pointer
equals the reusable slot address, and every pointer is 16 KiB page-aligned.
After loading another layer into bank A, all 16 addresses remained unchanged.

The ordinary sandbox hides Metal, so a second otherwise identical run was
executed with MPS visibility. Two Metal calls, one from each bank, recorded 32
zero-copy wraps and zero copies. Bank A and bank B outputs matched the CPU
native kernel, with maximum absolute errors `5.65e-9` and `4.31e-9`
respectively. Bank A Metal compute took 5.75 ms while the next-layer bank B
read took 12.45 ms; overlapped wall time was 12.54 ms. Thus the current warm
case is read-bound and the overlap mechanism hides nearly all expert compute.

## Bottleneck and next work

In the warm overlap probe, the 12.45 ms next-layer read is longer than the
5.75 ms Metal expert compute and determines the 12.54 ms wall time. In the
official direct layer run, resident/non-expert work was 23.7 ms of the 44.9 ms
forward. These are one-position warm measurements, but they identify both
resident compute and storage state as first-order variables. The next milestone
must continue to report I/O, resident compute, expert compute, transfer, and
synchronization separately.

The next sequence is:

1. Extend the verified one-layer ownership model to the complete resident
   spine while recording allocated, wired, compressor, and file-cache memory.
2. Feed the double slab through the ordinary routed-layer runtime rather than
   only the focused validator, preserving the zero-copy counter assertion.
3. Build a serial one-token run first, retain stage-level parity and timing,
   then enable next-layer overlap.
