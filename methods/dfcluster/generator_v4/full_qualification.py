"""Plan-scale Generator V4 contract and qualification runner.

The 64-task contract and 4096×3×stratum qualification use the same frozen
full-range sampler. CPU worker parallelism changes only scheduling: it does
not alter task identities, distributions, generator mechanisms or metrics.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import itertools
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Tuple

import numpy as np

from .core import OBSERVATION_FAMILIES, generate_v4_task, validate_training_payload
from .full_sampler import FullSamplerConfig, sample_full_task_config
from .source_complexity_graph import INFORMATION_STRATA


CONTRACT_TASKS_PER_STRATUM = 64
FULL_TASKS_PER_STRATUM = 4096
FULL_GENERATOR_SEEDS = (20260825, 20260826, 20260827)
CPU_WORKERS = 16
BATCH_SIZE = 128
THREAD_LIMITS = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


@dataclass(frozen=True)
class FullQualificationConfig:
    stage: str = "contract"
    generator_seeds: Tuple[int, ...] = FULL_GENERATOR_SEEDS
    tasks_per_stratum: int = CONTRACT_TASKS_PER_STRATUM
    sampler: FullSamplerConfig = FullSamplerConfig()
    cpu_workers: int = CPU_WORKERS

    def validate(self) -> None:
        if self.stage not in {"contract", "qualification"}:
            raise ValueError("stage must be contract or qualification")
        expected = CONTRACT_TASKS_PER_STRATUM if self.stage == "contract" else FULL_TASKS_PER_STRATUM
        if self.tasks_per_stratum != expected:
            raise ValueError("tasks_per_stratum may not be reduced or overridden")
        if self.stage == "contract" and self.generator_seeds != (20260825,):
            raise ValueError("plan contract uses the frozen generator seed 20260825")
        if self.stage == "qualification" and tuple(self.generator_seeds) != FULL_GENERATOR_SEEDS:
            raise ValueError("full qualification requires the three frozen generator seeds")
        if self.cpu_workers != CPU_WORKERS:
            raise ValueError("CPU worker count is frozen for reproducible resource scheduling")
        self.sampler.validate()


def _json(path: Path, value: Any) -> None:
    partial = path.with_name(path.name + ".tmp")
    partial.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(partial, path)


def _jobs(config: FullQualificationConfig) -> Iterator[tuple[int, str, int, FullSamplerConfig, str]]:
    for generator_seed in config.generator_seeds:
        for stratum in OBSERVATION_FAMILIES:
            for task_index in range(config.tasks_per_stratum):
                yield generator_seed, stratum, task_index, config.sampler, config.stage


def _chunks(items: Iterable[tuple[int, str, int, FullSamplerConfig, str]], size: int) -> Iterator[list[tuple[int, str, int, FullSamplerConfig, str]]]:
    iterator = iter(items)
    while True:
        batch = list(itertools.islice(iterator, size))
        if not batch:
            return
        yield batch


def _qualify_one(job: tuple[int, str, int, FullSamplerConfig, str]) -> Dict[str, Any]:
    generator_seed, stratum, task_index, sampler, stage = job
    spec = sample_full_task_config(
        generator_seed=generator_seed,
        task_index=task_index,
        split=stage,
        observation_stratum=stratum,
        sampler=sampler,
    )
    try:
        first = generate_v4_task(spec)
        second = generate_v4_task(spec)
        for name in ("features", "clean_latent", "missing_mask", "feature_types"):
            if not np.array_equal(getattr(first, name), getattr(second, name)):
                raise RuntimeError("replay mismatch: %s" % name)
        if first.metadata != second.metadata:
            raise RuntimeError("metadata replay mismatch")
        validate_training_payload(first.training_payload())
        if not np.isfinite(first.features).all() or not np.all(first.features[first.missing_mask.astype(bool)] == 0.0):
            raise RuntimeError("feature/missing contract violation")
        certificate = first.metadata
        return {
            "ok": True,
            "task_id": certificate["task_id"],
            "artifact_sha256": certificate["artifact_sha256"],
            "source_sha256": certificate["source_sha256"],
            "config_sha256": certificate["config_sha256"],
            "generator_seed": generator_seed,
            "observation_stratum": stratum,
            "information_stratum": spec.information_stratum,
            "N": spec.n_samples,
            "D": spec.n_features,
            "K": spec.n_clusters,
            "d_int": spec.intrinsic_dim,
            "missing_rate": spec.missing_rate,
            "labels_opened": False,
            "task_selection_performed": False,
            "node_count": certificate["node_count"],
            "edge_count": certificate["edge_count"],
            "node_depth_max": certificate["node_depth_max"],
            "informative_leaf_path_depths": certificate["informative_leaf_path_depths"],
            "operation_counts": certificate["operation_counts"],
            "effective_rank": certificate["effective_rank"],
            "condition_number": certificate["condition_number"],
            "distance_correlation": certificate["clean_raw_distance_correlation"],
            "knn_preservation": certificate["knn_neighborhood_preservation"],
        }
    except Exception as exc:
        return {
            "ok": False,
            "generator_seed": generator_seed,
            "observation_stratum": stratum,
            "task_index": task_index,
            "failure": "%s: %s" % (type(exc).__name__, exc),
        }


def _range_summary(values: List[float | int]) -> Dict[str, float | int | None]:
    return {"min": min(values) if values else None, "max": max(values) if values else None}


def run_full_qualification(config: FullQualificationConfig, output_root: Path) -> Dict[str, Any]:
    config.validate()
    output_root = Path(output_root)
    if output_root.exists() or output_root.with_name(output_root.name + ".partial").exists():
        raise FileExistsError("qualification destination already exists")
    for name, value in THREAD_LIMITS.items():
        os.environ.setdefault(name, value)
    partial = output_root.with_name(output_root.name + ".partial")
    partial.mkdir(parents=True, exist_ok=False)
    _json(partial / "resolved_config.json", asdict(config))
    _json(partial / "status.json", {"status": "running", "stage": config.stage, "completed_task_count": 0, "expected_task_count": len(config.generator_seeds) * len(OBSERVATION_FAMILIES) * config.tasks_per_stratum, "cpu_workers": config.cpu_workers, "thread_limits": THREAD_LIMITS})
    coverage = {stratum: 0 for stratum in OBSERVATION_FAMILIES}
    information_coverage = {stratum: 0 for stratum in INFORMATION_STRATA}
    joint_information_coverage = {
        observation: {information: 0 for information in INFORMATION_STRATA}
        for observation in OBSERVATION_FAMILIES
    }
    ranges: Dict[str, List[float | int]] = {"N": [], "D": [], "K": [], "d_int": [], "missing_rate": [], "node_count": [], "edge_count": [], "node_depth_max": [], "effective_rank": [], "condition_number": [], "distance_correlation": [], "knn_preservation": []}
    operation_union: set[str] = set()
    informative_depth_union: set[int] = set()
    source_hashes: set[str] = set()
    seen_task_ids: set[str] = set()
    failures: List[str] = []
    expected = len(config.generator_seeds) * len(OBSERVATION_FAMILIES) * config.tasks_per_stratum
    completed = 0
    with (partial / "task_ledger.jsonl").open("w", encoding="utf-8") as ledger:
        with ProcessPoolExecutor(max_workers=config.cpu_workers) as executor:
            for batch in _chunks(_jobs(config), BATCH_SIZE):
                for result in executor.map(_qualify_one, batch, chunksize=1):
                    if not result["ok"]:
                        failures.append("%s/%s/%s: %s" % (result["generator_seed"], result["observation_stratum"], result["task_index"], result["failure"]))
                        continue
                    task_id = str(result["task_id"])
                    if task_id in seen_task_ids:
                        failures.append("duplicate task_id: %s" % task_id)
                        continue
                    seen_task_ids.add(task_id)
                    completed += 1
                    observation = str(result["observation_stratum"])
                    information = str(result["information_stratum"])
                    coverage[observation] += 1
                    information_coverage[information] += 1
                    joint_information_coverage[observation][information] += 1
                    source_hashes.add(str(result["source_sha256"]))
                    for name in ranges:
                        ranges[name].append(result[name])
                    operation_union.update(result["operation_counts"])
                    informative_depth_union.update(int(value) for value in result["informative_leaf_path_depths"])
                    record = {key: value for key, value in result.items() if key != "ok"}
                    ledger.write(json.dumps(record, sort_keys=True) + "\n")
                ledger.flush()
                _json(partial / "status.json", {"status": "running", "stage": config.stage, "completed_task_count": completed, "expected_task_count": expected, "failure_count": len(failures), "cpu_workers": config.cpu_workers, "thread_limits": THREAD_LIMITS})
    information_coverage_ok = all(count > 0 for count in information_coverage.values())
    if config.stage == "qualification":
        expected_per_joint = len(config.generator_seeds) * config.tasks_per_stratum // len(INFORMATION_STRATA)
        information_coverage_ok = information_coverage_ok and all(
            count == expected_per_joint for per_observation in joint_information_coverage.values()
            for count in per_observation.values()
        )
        if not information_coverage_ok:
            failures.append("information-stratum coverage is not equal in the full qualification universe")
    report = {
        "schema_version": "dfcluster.generator_v4.full_qualification.v3",
        "stage": config.stage,
        "status": "passed" if not failures and completed == expected else "failed",
        "performance_claim": False,
        "task_selection_performed": False,
        "labels_opened": False,
        "expected_task_count": expected,
        "completed_task_count": completed,
        "coverage": coverage,
        "information_coverage": information_coverage,
        "joint_information_coverage": joint_information_coverage,
        "information_coverage_ok": information_coverage_ok,
        "ranges": {name: _range_summary(values) for name, values in ranges.items()},
        "source_sha256_count": len(source_hashes),
        "source_sha256": sorted(source_hashes),
        "operation_union": sorted(operation_union),
        "informative_leaf_depth_union": sorted(informative_depth_union),
        "unique_task_count": len(seen_task_ids),
        "cpu_workers": config.cpu_workers,
        "thread_limits": THREAD_LIMITS,
        "failures": failures,
    }
    _json(partial / "report.json", report)
    _json(partial / "status.json", {"status": "completed" if report["status"] == "passed" else "failed", "stage": config.stage, "completed_task_count": completed, "expected_task_count": expected, "failure_count": len(failures), "cpu_workers": config.cpu_workers, "thread_limits": THREAD_LIMITS})
    partial.rename(output_root)
    if report["status"] != "passed":
        raise RuntimeError("full qualification failed: %s" % report["failures"][:3])
    return report
