# DF-Cluster

面向通用表格的 zero-shot clean-geometry clustering 研究项目。

## Active research line

- **Plan**: `0824思路详细版.md`
- **Code**: `methods/dfcluster/generator_v4/`
- **Status**: Generator V4 implementation and CPU contracts are active. Full corpus generation and GPU training are gated on the V4 validity protocol.

## Directories

- `data/`: project data root (linked to `/data/luolie/DF-Cluster/data/`)
- `outputs/`: artifact root (linked to `/data/luolie/DF-Cluster/outputs/`)
- `baseline/`: frozen external sources and baselines
- `methods/dfcluster/generator_v4/`: active method implementation
- `papers/`: writing, figures and references
- `.cursor/rules/`: project governance rules

## Governance

- Model inputs are features, missing masks and feature types only.
- Synthetic clean geometry is a privileged training/audit target, never an inference payload.
- Labels, K and CLM are audit-only unless a separately frozen protocol explicitly authorizes their isolated use.
- Audit controls and validity reports do not select tasks or tune the active method.

The removed V3 legacy branch is not part of this project state and must not be recreated or used in V4 decisions.

## ZEUS geometry target probe

The current ZEUS comparison implementation is snapshotted in
`methods/zeus_geometry_probe/`. Its complete formal Phase-1 report and metrics
are in `results/zeus_geometry_probe/phase1_formal_resume_v4/`.
