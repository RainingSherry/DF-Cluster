"""CPU-only Generator V4 qualification contract.

This is a validity/coverage/replay audit, not a performance experiment.  It
writes only small provenance and task-ledger artifacts; labels and clean arrays
are never materialized into the ledger or report.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .core import (
    CLEAN_FAMILIES,
    OBSERVATION_FAMILIES,
    V4Config,
    generate_v4_task,
    validate_training_payload,
)


@dataclass(frozen=True)
class QualificationConfig:
    seeds: Tuple[int, ...] = (11, 22, 33)
    tasks_per_observation_family: int = 64
    n_samples: int = 128
    n_features: int = 10
    n_clusters: int = 2
    intrinsic_dim: int = 8
    missing_rate: float = 0.10

    def validate(self) -> None:
        if not self.seeds:
            raise ValueError("qualification requires at least one seed")
        if self.tasks_per_observation_family <= 0:
            raise ValueError("tasks_per_observation_family must be positive")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("qualification seeds must be unique")


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _task_config(
    q: QualificationConfig, seed: int, index: int, family: str
) -> V4Config:
    clean_family = CLEAN_FAMILIES[index % len(CLEAN_FAMILIES)]
    family_offset = list(OBSERVATION_FAMILIES).index(family) * q.tasks_per_observation_family
    global_index = family_offset + index
    missingness = ("mcar", "mar", "mnar")[index % 3]
    return V4Config(
        n_samples=q.n_samples,
        n_features=q.n_features,
        n_clusters=q.n_clusters,
        intrinsic_dim=q.intrinsic_dim,
        clean_family=clean_family,
        observation_family=family,
        missingness=missingness,
        missing_rate=q.missing_rate,
        categorical_fraction=0.20,
        ordinal_fraction=0.10,
        count_fraction=0.10,
        bounded_fraction=0.10,
        graph_depth=3 + (index % 3),
        split="qualification",
        task_index=global_index,
        seed=int(seed * 1_000_000 + global_index),
    )


def run_qualification(
    config: QualificationConfig | None = None,
    output_root: Path | None = None,
) -> Dict[str, Any]:
    """Run the frozen CPU qualification and optionally atomically write artifacts."""

    config = config or QualificationConfig()
    config.validate()
    records: List[Dict[str, Any]] = []
    seen_task_ids = set()
    coverage = {family: 0 for family in OBSERVATION_FAMILIES}
    source_hashes = set()
    failures: List[str] = []
    for seed in config.seeds:
        for family in OBSERVATION_FAMILIES:
            for index in range(config.tasks_per_observation_family):
                task_config = _task_config(config, seed, index, family)
                try:
                    first = generate_v4_task(task_config)
                    second = generate_v4_task(task_config)
                    for name in ("features", "clean_latent", "missing_mask", "feature_types"):
                        if not np.array_equal(getattr(first, name), getattr(second, name)):
                            raise RuntimeError("replay mismatch in %s" % name)
                    if first.metadata != second.metadata:
                        raise RuntimeError("metadata replay mismatch")
                    validate_training_payload(first.training_payload())
                    if not np.isfinite(first.features).all() or not np.isfinite(first.clean_latent).all():
                        raise RuntimeError("non-finite task")
                    if not np.all(first.features[first.missing_mask.astype(bool)] == 0.0):
                        raise RuntimeError("masked sentinel violation")
                    task_id = first.metadata["task_id"]
                    if task_id in seen_task_ids:
                        raise RuntimeError("duplicate task_id: %s" % task_id)
                    seen_task_ids.add(task_id)
                    source_hashes.add(first.metadata["source_sha256"])
                    coverage[family] += 1
                    records.append(
                        {
                            "task_id": task_id,
                            "artifact_sha256": first.metadata["artifact_sha256"],
                            "config_sha256": first.metadata["config_sha256"],
                            "source_sha256": first.metadata["source_sha256"],
                            "seed": int(seed),
                            "observation_family": family,
                            "clean_family": first.metadata["clean_family"],
                            "missingness": first.metadata["missingness"],
                            "finite": True,
                            "replay": True,
                            "labels_opened": False,
                            "training_forbidden_fields_opened": False,
                        }
                    )
                except Exception as exc:  # pragma: no cover - failure is reported atomically
                    failures.append("%s/%s/%d: %s" % (seed, family, index, exc))

    expected = len(config.seeds) * len(OBSERVATION_FAMILIES) * config.tasks_per_observation_family
    report: Dict[str, Any] = {
        "schema_version": "dfcluster.generator_v4.qualification.v1",
        "status": "passed" if not failures and len(records) == expected else "failed",
        "performance_claim": False,
        "config": asdict(config),
        "expected_task_count": expected,
        "completed_task_count": len(records),
        "coverage": coverage,
        "unique_task_count": len(seen_task_ids),
        "source_sha256_count": len(source_hashes),
        "failures": failures,
        "labels_opened": False,
        "clean_arrays_written_to_ledger": False,
        "training_forbidden_fields_opened": False,
    }
    if report["status"] != "passed":
        raise RuntimeError(json.dumps(report, sort_keys=True))

    if output_root is not None:
        output_root = Path(output_root)
        if output_root.exists():
            raise FileExistsError("qualification output already exists: %s" % output_root)
        partial = output_root.with_name(output_root.name + ".partial")
        if partial.exists():
            raise FileExistsError("qualification partial output already exists: %s" % partial)
        partial.mkdir(parents=True, exist_ok=False)
        _json_dump(partial / "resolved_config.json", asdict(config))
        _json_dump(partial / "status.json", {"status": "running", "stage": "qualification"})
        with (partial / "task_ledger.jsonl").open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
        _json_dump(partial / "report.json", report)
        _json_dump(partial / "status.json", {"status": "completed", "stage": "qualification"})
        partial.rename(output_root)
    return report
