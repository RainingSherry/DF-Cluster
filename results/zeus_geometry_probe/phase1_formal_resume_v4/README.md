# Formal Phase-1 Result Snapshot

This is the completed Phase-1 ZEUS target-only comparison. `report.json` has
`status: complete`; both arms reached 100,000 updates and share the same
training-task SHA-256 recorded in `completion.json`.

| Evaluation mode | Arm | ARI | CKA to X_ref | Distance Spearman | kNN overlap |
| --- | --- | ---: | ---: | ---: | ---: |
| Gaussian | ZEUS | 0.869319 | 0.580276 | 0.546299 | 0.087386 |
| Gaussian | Geometry | 0.789216 | 0.998246 | 0.997226 | 0.927809 |
| Gaussian transformed | ZEUS | 0.816172 | 0.445129 | 0.388972 | 0.063199 |
| Gaussian transformed | Geometry | 0.644737 | 0.633056 | 0.611552 | 0.267792 |

The uploaded snapshot includes the fixed evaluation manifest, resolved
configuration, six per-arm metrics files, 100k-step histories, report, status,
errors, and completion marker. The 12 serialized model checkpoints are not in
GitHub because each is approximately 305 MiB. Their expected relative paths,
steps, and provenance remain recorded in `report.json`; the experiment volume
retains the binaries.
