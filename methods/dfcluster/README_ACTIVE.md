# DF-Cluster Active Development Guide

## Generator V4 only

**Active design:** `/home/luolie/DF-Cluster/0824思路详细版.md`  
**Active code:** `methods/dfcluster/generator_v4/`

V4 is currently in generator implementation, contract qualification and audit-only validity coverage. Full V4 corpus generation, GPU AE/DDBM training, and real evaluation remain disabled until the V4 gates are frozen and passed.

### Use

```text
python -m methods.dfcluster.generator_v4 --output-root <new-output>
python -m methods.dfcluster.run_generator_v4_validity_grid --output-root <new-output>
pytest methods/dfcluster/tests/test_generator_v4*.py -q
```

### Boundary rules

- Training input is only `features`, `missing_mask`, and `feature_types`.
- Clean geometry is a synthetic target, never an inference payload.
- Labels, K and CLM are audit-only unless a separately pre-registered training loss explicitly authorizes them.
- Audit grids and impossible controls must not select tasks or enter training.
- Do not recreate or use removed V1/V3 experiments, corpora, checkpoints, code, or conclusions.
