# DF-Cluster Active Review State

## Generator V4 only

- Active plan: `0824思路详细版.md`.
- Active code: `methods/dfcluster/generator_v4/`.
- V4 CPU contracts and audit-only qualification are active.
- Full V4 corpus generation and GPU training remain blocked until the V4 validity protocol is frozen and passes.

## Review boundary

No removed legacy experiment, data artifact, metric, prompt, checkpoint, configuration, test or result may influence Generator V4 decisions.

The configured external Claude-review route timed out and has provided no score or approval.

## Claude source-complexity generator review — 2026-08-25

**Route:** `claude-review` MCP job `fa3c220413a24c8c9499c781fff3d1a5`; completed in 74.456 seconds. No code was edited by the reviewer.

**Accepted guidance:** enforce real DAG topology/path-depth checks; separate topology/mechanism/data RNG streams; serialize graph parameters; test label firewall, deterministic replay, typed heads, mechanism behavior and MCAR/MAR/MNAR.

**Rejected guidance:** label-derived difficulty strata; 4096 samples per task interpretation; unplanned CPU/GPU bit-exact/speed gates; small TabPFN smoke. These would alter the frozen plan, narrow execution, or violate label governance.

**Reviewer verdict:** proceed with the complete generator implementation, not a reduced proxy.

## Full validity result — 2026-08-25

The full planned validity audit completed at the exact 61,440-task scale and returned `failed_gate`: 26,321 passed / 35,119 failed, with no worker exceptions. This is retained as a negative audit result. Labels remained audit-only; no result was used to select, filter, reweight, tune, or train.

**Required next review:** assess the frozen detailed-plan validity/candidate-pool language against the V4 label-governance rule before authorizing any corpus or GPU stage. The current environment exposes neither the `claude-review` MCP action nor a server-side `claude` executable, so no final Claude verdict is claimed in this entry. The earlier source-complexity review remains recorded above; it did not review this completed result.

## Claude final-audit invocation attempt — 2026-08-25

A direct non-writing Claude Code review was invoked with the frozen full-validity evidence and the plan/governance conflict. It produced no JSON or textual result after more than fourteen minutes and was terminated; no score, verdict, or approval is claimed. The attempt made no project, server, generator, model, data, or experiment change. The prior completed source-complexity review remains the only accepted Claude review record.

## Corrected full-validity v2 result — final

The corrected source-complexity generator, now including preserving/noisy_recoverable/controlled_lossy coverage, completed the full 61,440-task validity universe. It returned `failed_gate`: 23,903 passed (38.9046%) and 37,537 failed gate, with no worker exceptions. Failure counts were ARI headroom below threshold 19,337, clean ARI below threshold 9,323, and probe macro-F1 below threshold 16,519. The result is audit-only negative evidence; it was not used for filtering, reweighting, tuning, curriculum, model selection, corpus generation, or GPU training.

The current blocking decision is candidate admission: whether the detailed plan's predeclared label-bearing candidate/coverage procedure (with labels kept outside model inputs) is authoritative, or whether strict no-selection governance is authoritative and a new label-free admission protocol must be frozen. No formal Claude verdict is claimed for this final audit; the direct review attempt produced no response.

## Option-A candidate protocol review attempt — 2026-08-25

### Assessment

- Owner decision: plan-first candidate gate (方案 A) authorized.
- Claude review: unavailable. Local Claude Code returned `Not logged in · Please run /login`; no score/verdict is claimed.

### Frozen action boundary

The candidate manifest will be built only from the completed corrected 61,440-task validity ledger, using the exact pre-registered gate and coverage axes. It will not change generator/measurement parameters and will keep audit metrics out of model input and inference payloads. The failed certificates remain negative evidence.

### Status

Proceeding to full-ledger candidate coverage audit; no narrow experiment and no GPU training.

## Option-A candidate manifest result — 2026-08-26

- Candidate manifest built from the complete corrected v2 audit, exact pre-registered gates.
- `23,903` candidates; `135/135` required observation/information/raw-pool/CLM-tertile cells nonempty.
- Minimum cell count: `2`, maximum: `786`. No extra threshold was invented; the coverage report makes the finite-cell imbalance explicit.
- The manifest excludes labels and audit metrics from the manifest consumed by input-side code; audit values are in a separate sidecar.

## Independent Stage-A compliance review — 2026-08-26

Luna Max read-only review found no P0 runtime failure but three P1 formal-compliance gaps: full `V4Task` crossed into Stage A rather than strict `InputTask`, fixed validation pool/metrics were absent, and GPU resource/peak ledger was absent. Formal v4 was stopped and retained as incomplete after 29,483 accepted / 29,482 trainer steps. No scientific result or checkpoint was retained as formal evidence.

Corrective work: strict X-only redaction, full candidate-derived validation manifest without audit fields, checkpoint validation metrics, and GPU ledger. Claude Code was unavailable (not logged in); no fabricated Claude verdict is claimed.
