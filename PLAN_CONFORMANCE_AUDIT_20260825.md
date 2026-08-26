# Generator V4 plan-conformance audit — 2026-08-25

## Scope and method

This is a static conformance audit of the active V4 mainline only. It reads the detailed execution plan, V4 source/configuration/tests, CodeGraph index, and frozen full-scale artifacts. It is **not** an experiment, does not open task labels, does not modify the generator/model/configuration, and does not authorize corpus or GPU training.

Authoritative plan: `0824思路详细版.md`.

Authoritative active implementation: `methods/dfcluster/generator_v4/`.

CodeGraph was synchronized for `/home/luolie/DF-Cluster` and indexed 757 files, 14,169 nodes, and 28,273 edges before this audit.

## Frozen evidence

| Evidence | SHA-256 / status |
|---|---|
| Generator source component SHA (`core.py`, `mechanisms.py`, `source_complexity_graph.py`, `full_sampler.py`, `provenance.yaml`) | `d21964b8bbb25b215a34aeeb732790ad906c1914f072ecc18802b832cec99f35` |
| Full qualification v2 report | `e766167a7e4e80415499c0afe8f10bd1c1b24e393d6820445966b9a955d6554a`; passed, 61,440 unique tasks |
| Full validity report | `19020959beeeaadfe2e4a50f533c24d6e7f586af0a1e0e2cf67c962183e629d4`; failed_gate, 61,440 completed tasks |
| Full validity array-free ledger | `83ffaa73bea9933ef8ac852fd257572c5c4faa3e6a1f1dd5891bcbbf0a23ff38` |

## Requirement-to-evidence matrix

| Detailed-plan area | Current status | Direct evidence |
|---|---|---|
| §§4–14: source-complexity generator, clean geometry, nuisance roots, DAG depth/roles, five observation strata, mixed types, MCAR/MAR/MNAR, and explicit information strata | **Implemented; corrected implementation requires fresh full-scale qualification** | `core.py`, `mechanisms.py`, `source_complexity_graph.py`, `full_sampler.py`, `provenance.yaml`; targeted tests pass; full corrected qualification is queued at `full_qualification_source_complexity_v3/` |
| §15: full validity gate at 3 × 5 × 4096 scale | **Executed at exact scale; failed** | `full_validity.py`, `validity.py`; `full_validity_source_complexity_v1/report.json` |
| §17: input/target/audit split and artifact isolation | **Contract implemented; full corpus/replay layout and privileged pair-target protocol not implemented or generated** | `storage.py` explicitly states it is contract code and not invoked for full corpus generation; its target artifact currently contains only `clean_latent`; `/data/luolie/DF-Cluster/data/generator_v4/` is absent |
| §§18–19: default AE architecture and type-aware reconstruction | **Architecture contract implemented; no Stage-A trainer/checkpoint/validation run** | `GlobalAEConfig`, `GlobalAE`, `mixed_type_reconstruction_loss` in `models.py`; no V4 training launcher exists |
| §§20–21: default 16-block DDBM architecture | **Architecture contract implemented; no formal Stage-B trainer/run** | `DDBMConfig`, `DatasetContextDDBM` in `models.py`; no V4 training launcher or output/checkpoint root |
| §22: complete DDBM objective | **Not implemented as specified** | Current `ddbm_geometry_loss` implements normalized Gram and sampled distance only. `v4_train_step` is explicitly a one-step smoke path and jointly updates AE+DDBM. It lacks the specified bridge/noise-prediction term, neighborhood loss, balanced co-assignment loss, fixed five-term weighting, per-term history/quantiles, and frozen-AE Stage-B semantics. The repository rule permits synthetic `Y_train` only for one pre-registered offline meta-pretraining clustering loss; the plan's co-assignment auxiliary loss can fit that exception, but no isolated target pipeline has been implemented. |
| §23: staged training/freeze/validation accounting | **Not implemented or run** | `objective.py` says “not a training launcher”; no Stage A/B runner, checkpoint ledger, validation task pool, task-exposure accounting, or frozen-AE workflow is present |
| §24.1–24.4: 64-task contract and full 4,096×3×stratum qualification | **Prior v2 completed but superseded; corrected v3 is required** | `qualification.py`, `full_qualification.py`; v2 is retained as superseded evidence; v3 must include equal information-stratum coverage |
| §24.5: short end-to-end trainability check under frozen AE+DDBM configuration | **Not run; no formal runner exists** | No V4 training launcher; `v4_train_step` is only an engineering smoke API, not a defined plan-scale runnable protocol |
| §25: ≥5,000,000 online tasks × 3 training seeds | **Not started** | No V4 corpus/online trainer/checkpoints/manifests exist |
| §26: synthetic evaluation | **Not started** | No V4 synthetic evaluation runner/results for H_obs/B_hat/geometry/metrics |
| §27: V4-only real known-K evaluation | **Not started** | No frozen V4 real manifest/scorer/result; CLUBench remains source asset only |
| §28: GPU resource protocol | **Not entered** | No V4 GPU training was launched; GPU 0/7 remain forbidden |

## Major precondition failure and governance conflict

The full validity audit completed without worker exceptions, but only 26,321 of 61,440 certificates passed all three frozen gates. The aggregate status is therefore `failed_gate`.

The detailed plan (§6.8 and §15.6–§15.8) specifies label-bearing clean-ARI/headroom/probe/CLM gates to form a candidate pool and then cover raw-ARI/CLM difficulty strata. The subsequent V4 governance freezes a stricter boundary: synthetic labels/K/CLM/ARI cannot perform task selection, curriculum, threshold adjustment, architecture choice, or model selection; the full-validity runner consequently treats all 61,440 tasks as needing to pass and does no selection.

These two rules cannot both determine the next corpus protocol when the complete universe does not satisfy every gate. This is a **major decision-level incompatibility**, not evidence that a single observation family implementation crashed: each stratum has approximately 42% all-gate pass rate.

## Consequences

1. The source-complexity implementation and its full qualification evidence are real, but they do **not** prove that the planned training distribution is valid under the present no-selection rule.
2. Even if the governance ambiguity were resolved, formal corpus writing, Stage A/B training, the complete §22 objective, 5-million-task online exposure, and evaluation pipelines remain work to be implemented and validated at their prescribed scale.
3. The failed ledger is negative evidence that must be retained. It cannot be converted into a training set by selecting/reweighting the passing 26,321 tasks, or used to tune parameters, thresholds, task frequencies, architectures, or losses.
4. No GPU training is authorized until the decision conflict is explicitly resolved and the missing formal training implementation is reviewed/implemented under a frozen, plan-consistent protocol.

## Clarification on synthetic pair targets

After re-reading `.cursor/rules/DF-Cluster-structure.mdc`, the active repository rule permits synthetic `Y_train` only for **one pre-registered offline meta-pretraining clustering loss**, while forbidding it from model inputs, generator observation maps, curriculum, early stopping, thresholds, hyperparameters, or variant selection. The planned §22 co-assignment auxiliary loss can be implemented only under that narrow exception: as an isolated synthetic-train target, never as an input or task-selection signal. This does **not** resolve the separate candidate-pool conflict below.

## Required owner decision before a new corpus or training run

Choose exactly one governing interpretation for **candidate admission/coverage**, and record it before a new corpus or training run:

- **A. Detailed-plan candidate-pool interpretation:** synthetic audit labels/ARI/CLM may be used solely in the predeclared §6/§15 candidate-gate and coverage protocol; they remain forbidden from model inputs, inference, real-data evaluation, architecture/loss tuning, and post-hoc model selection. The §22 pair target remains only the one pre-registered synthetic-train auxiliary loss permitted by the repository rule.
- **B. Strict no-selection interpretation:** labels/K/ARI/CLM remain audit-only and cannot curate the corpus; then the project must first freeze a fully label-free corpus-admission/validity protocol. The existing all-universe gate has failed and must not be silently replaced or tuned. The §22 pair target may still be considered only under the repository's one-loss synthetic-train exception; it cannot be repurposed for admission, curriculum, or model selection.

Neither option authorizes an ad-hoc narrow experiment, a change to the planned N/D/K/missingness/strata/metrics, or reuse of V3.
