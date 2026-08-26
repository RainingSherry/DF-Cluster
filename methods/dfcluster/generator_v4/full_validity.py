"""Full-scale audit-only validity gate for the V4 qualification universe.

This runner evaluates exactly the 3×5×4096 qualification universe. It never
filters tasks, writes a training manifest, or changes any generator/model
configuration. Labels are opened only inside the isolated validity worker.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import itertools
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Tuple

from .core import OBSERVATION_FAMILIES, generate_v4_task
from .full_qualification import CPU_WORKERS, FULL_GENERATOR_SEEDS, FULL_TASKS_PER_STRATUM, THREAD_LIMITS
from .full_sampler import FullSamplerConfig, sample_full_task_config
from .validity import ValidityConfig, compute_validity_certificate
from .source_complexity_graph import INFORMATION_STRATA

FULL_VALIDITY_TASK_COUNT = len(FULL_GENERATOR_SEEDS) * len(OBSERVATION_FAMILIES) * FULL_TASKS_PER_STRATUM
BATCH_SIZE = 64


@dataclass(frozen=True)
class FullValidityConfig:
    generator_seeds: Tuple[int, ...] = FULL_GENERATOR_SEEDS
    tasks_per_stratum: int = FULL_TASKS_PER_STRATUM
    cpu_workers: int = CPU_WORKERS
    sampler: FullSamplerConfig = FullSamplerConfig()
    validity: ValidityConfig = ValidityConfig()

    def validate(self) -> None:
        if tuple(self.generator_seeds) != FULL_GENERATOR_SEEDS:
            raise ValueError("full validity requires the three frozen generator seeds")
        if self.tasks_per_stratum != FULL_TASKS_PER_STRATUM:
            raise ValueError("full validity task scale may not be reduced")
        if self.cpu_workers != CPU_WORKERS:
            raise ValueError("full validity worker count is frozen")
        self.sampler.validate()
        self.validity.validate()


def _json(path: Path, value: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _jobs(config: FullValidityConfig) -> Iterator[tuple[int, str, int, FullSamplerConfig]]:
    for seed in config.generator_seeds:
        for stratum in OBSERVATION_FAMILIES:
            for task_index in range(config.tasks_per_stratum):
                yield seed, stratum, task_index, config.sampler


def _chunks(items: Iterable[tuple[int, str, int, FullSamplerConfig]], size: int) -> Iterator[list[tuple[int, str, int, FullSamplerConfig]]]:
    iterator = iter(items)
    while True:
        batch = list(itertools.islice(iterator, size))
        if not batch:
            return
        yield batch


def _audit_one(job: tuple[int, str, int, FullSamplerConfig]) -> Dict[str, Any]:
    seed, stratum, task_index, sampler = job
    try:
        spec = sample_full_task_config(
            generator_seed=seed,
            task_index=task_index,
            split="qualification",
            observation_stratum=stratum,
            sampler=sampler,
        )
        task = generate_v4_task(spec)
        certificate = compute_validity_certificate(task, FullValidityConfig().validity)
        result = certificate.as_dict()
        result.update({
            "generator_seed": seed,
            "observation_stratum": stratum,
            "information_stratum": spec.information_stratum,
            "schedule_task_index": task_index,
            "N": spec.n_samples,
            "D": spec.n_features,
            "K": spec.n_clusters,
            "d_int": spec.intrinsic_dim,
            "missing_rate": spec.missing_rate,
            "task_selection_performed": False,
        })
        return result
    except Exception as exc:
        return {
            "task_id": "unknown/%d/%s/%d" % (seed, stratum, task_index),
            "generator_seed": seed,
            "observation_stratum": stratum,
            "information_stratum": spec.information_stratum,
            "schedule_task_index": task_index,
            "status": "incomplete_validity_audit",
            "failure_reasons": ["worker_exception"],
            "error": "%s: %s" % (type(exc).__name__, exc),
            "labels_opened_only_in_audit": True,
            "task_selection_performed": False,
        }


def run_full_validity(config: FullValidityConfig, output_root: Path) -> Dict[str, Any]:
    config.validate()
    output_root = Path(output_root)
    if output_root.exists() or output_root.with_name(output_root.name + ".partial").exists():
        raise FileExistsError("full validity destination already exists")
    for name, value in THREAD_LIMITS.items():
        os.environ.setdefault(name, value)
    partial = output_root.with_name(output_root.name + ".partial")
    partial.mkdir(parents=True, exist_ok=False)
    _json(partial / "resolved_config.json", asdict(config))
    expected = FULL_VALIDITY_TASK_COUNT
    _json(partial / "status.json", {"status": "running", "expected_task_count": expected, "completed_task_count": 0, "cpu_workers": config.cpu_workers, "thread_limits": THREAD_LIMITS})
    counts: Dict[str, int] = {}
    information_coverage: Dict[str, int] = {stratum: 0 for stratum in INFORMATION_STRATA}
    failures: Dict[str, int] = {}
    reasons: Dict[str, int] = {}
    completed = 0
    passed = 0
    seen = set()
    with (partial / "validity_ledger.jsonl").open("w", encoding="utf-8") as ledger:
        with ProcessPoolExecutor(max_workers=config.cpu_workers) as executor:
            for batch in _chunks(_jobs(config), BATCH_SIZE):
                for result in executor.map(_audit_one, batch, chunksize=1):
                    completed += 1
                    task_id = str(result.get("task_id"))
                    if task_id in seen:
                        result["status"] = "incomplete_validity_audit"
                        result.setdefault("failure_reasons", []).append("duplicate_task_id")
                    seen.add(task_id)
                    status = str(result.get("status", "incomplete_validity_audit"))
                    counts[status] = counts.get(status, 0) + 1
                    information = result.get("information_stratum")
                    if information in information_coverage:
                        information_coverage[str(information)] += 1
                    if status == "passed":
                        passed += 1
                    for reason in result.get("failure_reasons", []) or []:
                        reasons[str(reason)] = reasons.get(str(reason), 0) + 1
                    if status == "incomplete_validity_audit":
                        failures["incomplete_validity_audit"] = failures.get("incomplete_validity_audit", 0) + 1
                    # No labels/arrays are present in the certificate record.
                    ledger.write(json.dumps(result, sort_keys=True) + "\n")
                ledger.flush()
                _json(partial / "status.json", {"status": "running", "expected_task_count": expected, "completed_task_count": completed, "passed_task_count": passed, "certificate_status_counts": counts, "failure_reasons": reasons, "cpu_workers": config.cpu_workers, "thread_limits": THREAD_LIMITS})
    report = {
        "schema_version": "dfcluster.generator_v4.full_validity.v1",
        "status": "passed" if completed == expected and not failures and passed == expected else "failed_gate",
        "performance_claim": False,
        "selection_policy": "audit_only",
        "task_selection_performed": False,
        "labels_opened_only_in_audit": True,
        "labels_written_to_report": False,
        "expected_task_count": expected,
        "completed_task_count": completed,
        "passed_task_count": passed,
        "certificate_status_counts": counts,
        "information_coverage": information_coverage,
        "failure_reasons": reasons,
        "cpu_workers": config.cpu_workers,
        "thread_limits": THREAD_LIMITS,
        "unique_task_count": len(seen),
        "sampler": asdict(config.sampler),
        "validity": asdict(config.validity),
    }
    _json(partial / "report.json", report)
    _json(partial / "status.json", {"status": "completed", "expected_task_count": expected, "completed_task_count": completed, "passed_task_count": passed, "certificate_status_counts": counts, "failure_reasons": reasons, "cpu_workers": config.cpu_workers, "thread_limits": THREAD_LIMITS})
    partial.rename(output_root)
    return report
