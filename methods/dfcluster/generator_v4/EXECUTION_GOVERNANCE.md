# Generator V4 execution governance

This file freezes the user-directed execution policy for `0824思路详细版.md`.

## Mainline only

The V4 mainline is the full source-complexity generator specified in the plan:

- actual 3–8 root-to-leaf DAGs;
- TabICL Graph/MLP/Tree-derived adapters with clean/nuisance root injection;
- DT/ET/RF/GB/direct-RF MITRA-inspired TBP;
- label-free flow and analytic strata;
- full mixed-type and MCAR/MAR/MNAR generation;
- actual complexity certificates;
- 64-task contract and then 4096 valid tasks × 3 generator seeds × each observation stratum.

Early 960-task qualification and 9-case validity artifacts are engineering/audit prototypes only. They are not corpus evidence, training evidence, model-selection evidence, or a license to reduce the planned execution scale.

## No ad-hoc narrow experiments

Before any experiment not explicitly specified in the plan, write an `Experiment Scope Record` into `CHANGELOG.md` with:

1. hypothesis and exact planned-stage link;
2. why the setting is not too narrow;
3. whether it changes a planned measurement, scale, model, generator family, or gate;
4. label/K/CLM access boundary;
5. predeclared output path and stop condition.

If it narrows the planned scale or changes a measurement/gate without an explicit user instruction, do not run it.

## Immutable boundaries

- Labels/K/CLM never enter the observation map, model inputs, task selection, curriculum, thresholding, architecture choice, or model selection.
- Audit-only labels may measure a frozen artifact but cannot select tasks or tune V4.
- GPU 0 and GPU 7 are forbidden.
- V3 and its thin-generator branch were removed and must not influence V4.

## Claude cross-review, 2026-08-25

The Claude review accepted the full complex-generator direction and highlighted true DAG validation, independent RNG streams, explicit graph serialization, label-firewall checks and mechanism-level deterministic tests. The following suggestions were rejected because they would deviate from the plan or label governance:

- label-derived difficulty/entropy strata;
- interpreting 4096×3 as samples per task instead of valid tasks per seed/stratum;
- unplanned CPU/GPU bit-exact parity and speed thresholds;
- an unplanned small TabPFN training smoke.

## Frozen full-validity result — 2026-08-25

The exact planned 61,440-task source-complexity validity audit is complete and its aggregate status is `failed_gate` (26,321 passed; 35,119 failed gate; no worker exceptions). This factual audit outcome must be retained, but it must not be used as a source of label/K/CLM/ARI-based task selection, filtering, curriculum, threshold adjustment, generator tuning, architecture choice, or model selection.

It is a major precondition failure for corpus generation and GPU training. Until an independent high-rigor review resolves the compatibility of the detailed plan's label-bearing validity/candidate-pool text with the immutable label-governance boundary, no V4 corpus or GPU training run is authorized. Any proposed change must be a documented, user-visible decision rather than an ad-hoc response to the audit.

## Section-12 information-stratum correction — 2026-08-25

The active generator was found to omit the detailed plan's explicit information-stratum axis. The correction adds only the preplanned `preserving`, `noisy_recoverable`, and `controlled_lossy` mechanisms and full-qualification coverage accounting. It does not alter the frozen measurement protocol or permit label-derived selection. All prior qualification/validity artifacts are superseded for current-generator evidence; the exact full qualification must be rerun before validity.

## Owner decision — plan-first candidate gate (方案 A) — 2026-08-25

The project owner explicitly authorized the **plan-first candidate-pool interpretation**. This is a documented priority decision for the active execution plan, not an ad-hoc response to a small experiment:

- Synthetic `Y`-derived audit values (`ARI_clean`, `ARI_headroom`, supervised probe macro-F1, `CLM_observed`, and `CLM` tertile labels) may be used only for the pre-registered §15 candidate gate and §15.7–§15.8 coverage manifest.
- Candidate admission uses the frozen gates exactly as written: `ARI_clean≥0.90`, `ARI_headroom≥0.20`, probe macro-F1 `≥0.75`, finite/shape/cluster/metadata contracts.
- Raw-ARI pools are exactly `easy=[0.50,0.80]`, `medium=[0.15,0.50)`, and `hard-but-recoverable<0.15`; within each pool, CLM_observed empirical tertiles are computed deterministically from the gate-passing candidate pool.
- These audit fields may control candidate-manifest coverage/admission only. They must never enter the observation map, model input, inference payload, curriculum, early stopping, threshold tuning, hyperparameter/architecture selection, checkpoint/model selection, or real-data evaluation.
- Synthetic `Y_train` may be opened only in the one pre-registered offline meta-pretraining co-assignment loss allowed by `.cursor/rules/DF-Cluster-structure.mdc`; it is not a candidate-admission or model-selection signal.
- Synthetic held-out/development/test labels and all real benchmark labels remain scorer-only after prediction/configuration freeze. V3 remains removed and cannot influence V4.

The previous strict no-selection hold is superseded only for the above pre-registered candidate gate/coverage use. No generator parameter, measurement parameter, task scale, threshold, model dimension, or GPU restriction is changed by this decision.

## Option-A target-label protocol — 2026-08-26

The owner-authorized option-A exception is now reflected in storage: privileged synthetic targets may contain `Y` and bit-packed `A_pair` only for the single pre-registered offline meta-pretraining co-assignment loss. The X-only input artifact and inference payload remain label-free; no general loader may open target labels. This is not permission to use labels for curriculum, early stopping, tuning, or model selection.

## Candidate v2 and online stream boundary — 2026-08-26

Candidate manifest v2 freezes numeric CLM q1/q2 cutpoints within each gate-passing raw-ARI pool. The online candidate stream may use those cutpoints only in the pre-registered option-A admission/coverage audit. It has a fixed 5,000,000 accepted-task target and must retain every rejection in an array-free ledger. The stream is not a finite replay shortcut; it has not been launched.

## Section-24.5 trainability evidence — 2026-08-26

The maximum-envelope full-default AE/DDBM engineering check passed on physical GPU 2 with finite Stage-A/Stage-B updates and gradient checkpointing. This evidence only authorizes continuing implementation/preflight; it does not authorize treating one step as formal training or changing the 5,000,000-task exposure requirement.

## Formal Stage-A compliance correction — 2026-08-26

Formal v4 was stopped after independent review identified strict-loader, fixed-validation, and GPU-ledger gaps. The corrected run must use `InputTask` redaction before Stage-A, a frozen full candidate-derived validation pool with no audit fields, checkpoint validation metrics, and an atomic GPU ledger. This correction does not change the 5,000,000 task target, generator, gates, measurements, or model dimensions.

## Corrected Stage-A v5 launch boundary — 2026-08-26

The corrected relaunch must use strict `InputTask` redaction, the full 23,903-row fixed validation manifest, validation metrics at the frozen checkpoints, and `gpu_ledger.jsonl`. The 5,000,000 accepted-task target, generator source SHA, option-A candidate proposal distribution, model dimensions, corruption parameters, and all scientific measurements remain unchanged. The previous v4 run is historical incomplete evidence and must not be resumed.

## Stage-B engineering contracts — 2026-08-26

Stage B now has explicit row permutation/inverse mapping, fixed cell-budget accumulation at the maximum planned task cell envelope, bounded loss-summary statistics, GPU ledger, and strict input/privileged-target separation. These are implementation/provenance contracts only; no Stage-B exposure is authorized before a completed compliant Stage-A checkpoint.
