# M3 Ultra 512 GB: M11 prefix-state reuse falsifier

M11 tested whether K3's fixed chat-template prefix can be evaluated once and
reused as an immutable KV/KDA/short-convolution state snapshot. The mechanism
works exactly when both paths use the same segmented execution, and it removes
most repeated prompt I/O. It does **not** preserve the selected monolithic
long-prompt numerics, so it is not enabled in `config/m3ultra512.env`.

This is a correctness-gate rejection, not an I/O or ownership failure.

## Fixed prefix

Two official chat templates with different user text have an identical
74-token prefix. It contains:

- the built-in thinking-effort system message;
- the complete system-message terminator;
- the opening user-message header and separator.

The first user-content token is position 74. The `"Hello"` template is 89
tokens, leaving a 15-token suffix.

`tools/prefix_state.py` derives this boundary from the official tokenizer
instead of embedding token IDs. Reuse is fail-closed: every prefix token must
match and the prompt must contain at least one suffix token.

## Snapshot ownership

`PrefixStateSnapshot` retains references to:

- KDA recurrent states;
- the three short-convolution states per KDA layer;
- MLA key and value states.

K3 inference replaces these tensors functionally. Installing the references
into a fresh `KimiDynamicCache` is therefore constant-time and does not clone
the state. The validator hashes the retained state again after both suffix
passes to detect mutation.

The real 74-token snapshot contained:

| Metric | Result |
|---|---:|
| State tensors | 324 |
| Logical tensor bytes | 693,043,200 |
| Retained storage bytes | 780,337,152 |
| Restore time | 0.747 ms |
| Expert weights retained | 0 |
| HTTP or converted weights | 0 |

The extra storage over logical bytes comes from tensor backing allocations,
not an expert cache.

## Validator

`tools/validate_m11_prefix_state.py` evaluates three paths:

1. the selected monolithic full-prompt oracle;
2. an uncached canonical `prefix` then `suffix` execution;
3. a fresh cache restored from the prefix snapshot, then the same suffix.

It compares position-ordered routes, all observable logits, the first decode
token, decode logits, and SHA-256/dtype/shape/length for all 324 state tensors.
It also checks official-model fingerprints, resident-bank identity, scratch
identity, Metal copy counts, HTTP counts, and final release.

The short falsifier is:

```sh
set -a
source config/m3ultra512.env
set +a

.venv/bin/python tools/validate_m11_prefix_state.py \
  --case smoke \
  --output bench-results/m11-prefix-state-smoke.json
```

The 8-token prompt split 4+4 passed:

- canonical segmented versus restored suffix: bit-for-bit exact;
- monolithic versus segmented routes and next token: exact;
- maximum monolithic logit difference: `6.20e-6`;
- state restore: 0.56 ms;
- MPS current allocation after release: baseline.

## Official 89-token result

The full run is saved at
`bench-results/m11-prefix-state-chat-final.json`. Its monolithic compatibility
gate intentionally failed.

### Segmented versus restored

The serialized evidence was independently re-audited after the gate stopped:

| Exact check | Result |
|---|---:|
| All suffix and decode routes | exact |
| Next token | exact |
| All suffix logits SHA-256 | exact |
| Prefill state, all 324 tensors | exact |
| Decode logits SHA-256 | exact |
| Decode state, all 324 tensors | exact |

This proves the snapshot ownership and restore mechanism. The restored path is
deterministic and does not mutate the retained prefix.

### Monolithic versus 74+15

| Check | Result |
|---|---:|
| First generated token | exact (`1008`) |
| Second-token argmax | exact (`2742`) |
| Prefix-logit maximum difference | 8.1924 |
| Suffix-logit maximum difference | 3.4330 |
| First-decode-logit maximum difference | 1.0147 |
| Differing routed position/layer rows | 3,207 / 8,188 |
| First suffix-route difference | layer 18, position 77 |
| First prefix-route difference | layer 32, position 21 |

The first suffix difference was initially only a rank-order swap inside the
same top-16 set, but the perturbation amplified through later layers.

K3's MPS projections, attention, router, and expert kernels receive different
position dimensions for `T=89` versus `T=74` plus `T=15`. Their floating-point
reduction order is shape-dependent. The 8-token falsifier exposed only a few
micro-units of error; over 93 real layers and 89 positions it crossed router
boundaries and became a materially different internal execution.

The result is not caused by:

- a shared file pointer or wrong shard offset;
- snapshot mutation;
- stale expert weights;
- HTTP fallback;
- Metal staging copies;
- missing causal history.

## Performance potential

These figures are natural-cache measurements, not controlled cold bandwidth.

| Phase | Compute/prefill | Decode | Logical expert bytes | Physical member reads |
|---|---:|---:|---:|---:|
| Monolithic 89-token oracle | 661.358 s | 15.415 s | 624.174 GB | 560.393 GB |
| One-time 74-token prefix build | 631.177 s | — | 559.933 GB | 525.042 GB |
| Uncached 15-token suffix | 127.782 s | 10.758 s | 180.842 GB | 106.934 GB |
| Restored 15-token suffix | 11.758 s | 4.811 s | 180.842 GB | 5.192 GB |

The restored run came last and was page-cache warm, so 11.758 seconds is not a
cold suffix guarantee. The logical-demand reduction is still real: repeated
prefill no longer requests the 74-token prefix's expert spans. The candidate
therefore has large performance value, but the selected exact profile cannot
trade output semantics for it.

## Selection decision

`K3_PREFIX_STATE` remains default-off and is absent from
`config/m3ultra512.env`. The ordinary server and CLI remain on monolithic
prefill unless a developer explicitly opts into this experimental segmented
mode.

The profile was not silently redefined around the faster result.

## Recommended next step

The next design must preserve the monolithic position dimension while avoiding
fixed-prefix expert I/O. The strongest candidate is shape-stable activation
reuse:

1. keep the full prompt position dimension used by the selected runtime;
2. capture fixed-prefix per-layer MoE outputs or the minimal equivalent
   activation state, not expert weights;
3. compute the variable suffix while substituting only proven
   input-independent prefix activations;
4. key any reusable activation by execution shape when kernel numerics depend
   on total position count;
5. require bitwise routes, final logits, and cache state against monolithic
   execution before enabling it.

If shape-stable substitution cannot be made exact, prefix reuse must remain an
explicit approximate mode rather than entering the M3 Ultra exact profile.
