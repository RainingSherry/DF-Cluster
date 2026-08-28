# ZEUS Geometry Target Probe

This directory contains the reproducible core of the ZEUS target-only comparison.
It keeps the upstream ZEUS generator and Transformer, pairs identical model
initialization and synthetic task streams, and compares:

- `zeus`: `gmm_loss_with_regularizes`;
- `geometry`: `1 - centered_linear_cka(H, X_ref)`.

The runner is `geometry_probe/phase_runner.py`. The copied `zeus/` package is
the runtime source used by the probe, including the PyTorch compatibility fix
in `zeus/model/layer.py`.

## Formal Phase 1

The complete result snapshot is under
`results/zeus_geometry_probe/phase1_formal_resume_v4/` and is based on one
seed (`42`), 100,000 updates, fixed Gaussian and transformed evaluation sets
(100 tasks per mode), and checkpoints at:

```text
5000, 20000, 50000, 75000, 90000, 100000
```

At 100k, ZEUS has higher ARI while geometry has substantially higher recovery
of the reference geometry. The machine-readable report is the source of truth:
`results/zeus_geometry_probe/phase1_formal_resume_v4/report.json`.

The paired training-task digest is recorded in `report.json` and
`completion.json`. No Phase-2 `proceed=true` gate was created.

## Running

From the repository root, install the dependencies listed in `requirements.txt`
and run the probe package with its directory on `PYTHONPATH`, for example:

```bash
PYTHONPATH=methods/zeus_geometry_probe \
python methods/zeus_geometry_probe/geometry_probe/phase_runner.py \
  --phase phase1 \
  --output /path/to/output \
  --device cpu
```

The formal GPU run used a legal physical GPU and the frozen configuration in
`results/zeus_geometry_probe/phase1_formal_resume_v4/resolved_config.json`.

## Checkpoints

The twelve PyTorch checkpoints are each about 305 MiB and are intentionally not
stored in this GitHub snapshot. They remain in the experiment data volume at:

```text
/data/luolie/DF-Cluster/outputs/zeus_geometry_probe/phase1_formal_resume_v4/
```

The report, metrics, histories, configuration, evaluation manifest, completion
marker, and error/status records are included here. The checkpoint paths and
reuse provenance are recorded in `report.json`.
