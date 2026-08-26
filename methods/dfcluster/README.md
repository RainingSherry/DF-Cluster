# DF-Cluster method workspace

## Active line: Generator V4

The active implementation is `generator_v4/`, following
`/home/luolie/DF-Cluster/0824思路详细版.md`.

Current V4 modules:

- `core.py`: 128-D clean geometry, label-free observation graph, mixed-type heads and masks.
- `mechanisms.py`: local TabICL-inspired MLP/Tree and MITRA-inspired TBP adapters with explicit provenance boundaries.
- `counterfactual.py`: paired observation mechanisms sharing clean/nuisance roots.
- `controls.py`: audit-only impossible controls that cannot enter training.
- `models.py` / `objective.py`: Global AE and DDBM architecture plus label-free one-step contract.
- `qualification.py`, `validity.py`, `storage.py`: replay, audit-only validity and input/target/audit separation contracts.

Generator V4 has not started full-corpus generation or GPU training. Those stages remain gated on the frozen V4 generator validity protocol.

## Removed legacy line

The V1/V3 thin-generator corpus, V3 DSCPF experiment code/configuration/tests, and V3 outputs were physically removed on 2026-08-25 at the user's request. They must not be used for V4 decisions or reproduced without an explicit new instruction.

The external CLUBench raw benchmark snapshot is retained as a source asset for a future V4-only real evaluation; it is not a V3 experiment artifact.
