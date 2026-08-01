# M3 Ultra 512 GB: M16 passive-evidence admission gate

M16 closes the prefix-admission investigation with an evidence-based decision:
keep the selected two-entry LRU policy. The passive server trace contains one
eligible request and no exact-shape reuse, so it cannot demonstrate a benefit
from repeat-aware admission.

This is an insufficient-evidence result, not a negative performance result for
the experimental policy.

## Runtime-accurate offline model

The M15 simulator previously retained an unbounded set of observed shapes,
while the runtime bounds repeat history with
`K3_PREFIX_ACTIVATION_HISTORY=4096`. M16 makes the simulator use the same
LRU-bounded history and records the bound in every analysis.

This distinction does not change the 42-request M15 stress result because that
trace contains only 32 distinct shapes. It does make long passive traces model
the selected implementation exactly: a shape that falls out of the observation
history is first-seen again for admission purposes.

Focused coverage forces a two-shape history, evicts a shape from both the
resident cache and observation history, and confirms that the next visit is
bypassed. With a sufficiently large history the same visit is admitted.

## Selection gate

`tools/m16_admission_gate.py` analyzes the privacy-minimized trace at the
selected capacity and requires all of the following before authorizing an
expensive controlled ordinary-server A/B:

1. at least one eligible request;
2. at least one repeated exact prompt shape;
3. more repeat-policy hits than LRU hits;
4. fewer repeat-policy captures than LRU captures.

Passing this gate would authorize an A/B; it would not select `repeat` by
itself. Selection would still require identical server outputs, the M12 bitwise
contract, correct bypass release, unchanged official files, and a measured
end-to-end improvement.

The report also links the existing prerequisite evidence without copying its
prompt/token material:

- M12 full-prompt and decode logits, routes, and cache state were bitwise exact;
- the captured activation remained immutable;
- M13 completed capture/replay through the ordinary OpenAI server;
- both validations left the official checkpoint tree unchanged.

## Passive trace result

Command:

```bash
set -a
source config/m3ultra512.env
set +a

.venv/bin/python tools/m16_admission_gate.py \
  --shape-jsonl bench-results/m14-server-shapes.jsonl \
  --capacity 2 \
  --max-capacity 16 \
  --repeat-history-entries 4096 \
  --output bench-results/m16-admission-gate-final.json
```

Result:

| Metric | LRU | Repeat-aware |
|---|---:|---:|
| eligible requests | 1 | 1 |
| distinct shapes | 1 | 1 |
| repeated requests | 0 | 0 |
| hits | 0 | 0 |
| captures | 1 | 1 |
| admissions | 1 | 1 |
| bypasses | 0 | 0 |
| hit gain | — | 0 |
| capture reduction | — | 0 |

The single request is an eligible T=89 capture. There is no rotated backup and
there are no reuse or scan-protection events. Capacity sweeps from 1 through 16
therefore cannot distinguish either policy or any capacity.

Gate result:

```text
keep-lru-insufficient-passive-evidence
```

The controlled real-server A/B was deliberately not run. Without an observed
repeat-admission benefit it would consume multiple full K3 prefills while
testing a synthetic workload already covered by M15, and would not establish
representative traffic behavior.

## Correctness and privacy

The M16 artifact records only shapes and aggregate policy decisions. It stores
no prompt content, token IDs, decoded output, or request hashes.

The official checkpoint tree was fingerprinted before and after analysis and
was unchanged. No inference weights were read into a derived model, converted,
modified, or copied. The only new persistent output is a small regenerable JSON
report under the ignored `bench-results/` directory.

## Decision

Keep:

```text
K3_PREFIX_ACTIVATION_ENTRIES=2
K3_PREFIX_ACTIVATION_ADMISSION=lru
K3_PREFIX_ACTIVATION_HISTORY=4096
```

Retain `repeat` as an implemented, bounded, tested experiment. If ordinary use
later produces repeated shapes separated by one-off scans, rerun the M16 gate.
Only a positive passive candidate should advance to the controlled server A/B.

This is a natural stopping point for the implementation program. The complete
official-checkpoint inference path, long-lived server, exact prefix replay,
telemetry, privacy-bounded tracing, experimental admission policy, and its
selection boundary are now all implemented. Further work is operational
evaluation on representative user workloads rather than another speculative
runtime optimization.
