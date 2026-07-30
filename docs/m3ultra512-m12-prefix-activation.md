# M3 Ultra 512 GB: M12 shape-stable prefix activation reuse

M12 replaces the rejected M11 segmented-prefix idea with a stricter reuse
boundary. K3 still evaluates one monolithic prompt with its original position
dimension. Only the routed-MoE result for the fixed 74-token chat prefix is
retained and spliced back into the same rows on later prompts of the same total
length.

The official Kimi-K3 checkpoint remains unmodified and unconverted. Resident
tensors and suffix experts still come directly from its safetensors shards.

## Why M11 could not be reused

M11 proved that a 74-token state snapshot is exact when both runs use segmented
prefill, but `74 + 15` and monolithic `89` produce different floating-point
results. Reusing that state would therefore change the selected exact runtime.

M12 never changes `T`. A retained entry is keyed by:

- the exact fixed chat-prefix token IDs;
- the complete monolithic position count;
- the running model process and its exact runtime configuration.

A 74-token entry cannot be applied to a different full shape.

## Runtime design

`PrefixActivationSession` captures, for every routed layer:

- the prefix latent input `[74, 3584]` on CPU;
- prefix router expert IDs and weights;
- the prefix routed-expert output `[74, 3584]` on MPS.

Before reuse it compares the latent input, route IDs, and route weights
bit-for-bit. A mismatch raises a layer-specific error; it is never treated as a
cache hit. On a valid replay, the Metal position-batch ABI still receives the
full `T` rows, but the 74 prefix rows contain empty expert lists. The native
kernel already defines empty rows as zero output. Deltafin reads and computes
only suffix experts, then concatenates the retained prefix output before routed
normalization, shared-expert evaluation, and residual aggregation.

This is activation reuse, not a weight cache. It retains no routed expert
weights and creates no checkpoint derivative.

`PrefixActivationCache` adds the normal server policy:

- only official chat prompts with the exact fixed prefix are eligible;
- entries are keyed by total prompt length;
- LRU capacity is controlled by `K3_PREFIX_ACTIVATION_ENTRIES`;
- capture failures are discarded;
- replay failures invalidate that shape instead of causing a persistent hit;
- replay sessions can be reused repeatedly;
- `K3_PREFIX_STATE` and `K3_PREFIX_ACTIVATION` are mutually exclusive.

The selected M3 Ultra profile enables two entries, about 390.4 MB total.

## Focused validation

The validator uses two official chat-template prompts:

- capture: `Hello`
- target: `World`

Both render to 89 tokens. Their first 74 tokens are identical and their last 15
tokens differ. Three passes run against the same official checkpoint:

1. capture the `Hello` prefix activations;
2. run an unmodified monolithic `World` oracle;
3. replay the `Hello` fixed-prefix activations during monolithic `World`.

The comparison includes all 92 routed layers, all prompt logits, all 324
prefill cache tensors, selected next token, one-token decode logits, the decode
cache, routes, retained activation hashes, resident-bank identity, scratch
addresses, source-tree fingerprints, release behavior, and swap.

Command:

```bash
set -a
source config/m3ultra512.env
set +a
.venv/bin/python tools/validate_m12_prefix_activation.py \
  --case chat \
  --output bench-results/m12-prefix-activation-chat-final.json
```

Focused short-shape coverage is available with `--case smoke`.

The final source state passed 36 focused unit/regression tests covering the
session lifecycle, repeated replay, bounded shape LRU, invalid prefixes and
shapes, M11/M10 compatibility helpers, direct-shard metadata, request metrics,
and empty-prefix Metal position rows. A complete legacy `tools/test_*.py`
discovery is not a valid gate for this direct-official fork: several unrelated
tests import the intentionally empty `tools/k3pkg` package or require the
unbuilt legacy `k3-meta` tree. The discovery ran 99 tests and reported those 11
collection/import errors separately.

## Exactness result

`bench-results/m12-prefix-activation-chat-final.json` passed:

| Check | Result |
|---|---:|
| routes | exact |
| all prefill logits | bitwise exact, max abs `0.0` |
| all prefill cache tensors | exact |
| next token | exact (`1008`) |
| decode logits | bitwise exact, max abs `0.0` |
| decode cache | exact |
| retained activation digest | unchanged |
| official checkpoint/tree | unchanged |
| MPS copies reported by Metal MoE | `0` |
| release current-allocation delta | `0` |
| release driver-allocation delta | `42,205,184` bytes |
| swap delta | `524,288` bytes |

The 92-layer entry owns `195,198,976` bytes:

- input validation copies: `97,599,488` bytes;
- retained routed outputs: `97,599,488` bytes.

Replay skipped all `74 × 16 × 92 = 108,928` fixed-prefix route edges. Across
layer-local route unions, the oracle used 34,837 unique experts and replay
demanded 9,436, so 25,401 expert loads were avoided.

## Measured performance

These numbers are one natural-cache run, not a globally purged cold-storage
benchmark. Physical bytes are the sum of the two RAID members.

| Pass | Prefill | Decode | Direct demand | Physical member reads |
|---|---:|---:|---:|---:|
| target oracle | `710.321 s` | `15.922 s` | `637.124 GB` | `600.506 GB` |
| target replay | `115.160 s` | `5.165 s` | `191.406 GB` | `87.533 GB` |

Prefill improved by `6.17×`; prefill plus one decode improved from `726.243 s`
to `120.325 s` (`6.04×`). Observed physical member reads fell by 85.4%.
Input validation itself took `0.075 s`, which is negligible beside expert I/O.

The capture pass remains a normal full prefill (`663.704 s`). A shape becomes
faster only after one successful request of that exact total prompt length.

## Limits and next work

The cache hit dimension is exact total token count, so requests with different
lengths do not share an entry. This is required for the established bitwise
contract. The two-entry LRU is intentionally small and should be measured on a
real long-lived workload before increasing it.

The next phase should:

1. collect server metrics for activation hit rate by total prompt length;
2. separate physical RAID bytes from page-cache bytes over repeated requests;
3. measure whether two LRU shapes cover real chat traffic;
4. reduce the 97.6 MB CPU validation copy by hashing or comparing through a
   reusable pinned buffer, only if exactness and latency evidence justify it;
5. investigate safe suffix-length bucketing only through a new monolithic
   parity gate—never by reviving segmented prefill.
