"""Generate the frozen recoverable-v2 corpus and isolated CLM sidecars.

The corpus is intentionally restricted to the known paired-view forward
family in :mod:`recoverable_generator`.  The analytic oracle is used only by
the pre-method recoverability audit and is never written to these shards.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import json
import logging
import os
from pathlib import Path
import shutil
import time
import traceback
from typing import Any, Iterator

import h5py
import numpy as np
import yaml

from .audit_recoverable_v2 import GATES as RECOVERABILITY_GATES
from .clm_audit import CLM_COMMIT, CLM_LOGISTIC_K, compute_clm_audit
from .dataset_io import TaskPayload, stable_sha256, write_multi_task_shards
from .generate_corpus import (
    MIN_FREE_AFTER_GIB,
    THREAD_LIMITS,
    _atomic_json,
    _atomic_jsonl,
    _environment_report,
    _file_sha256,
    _git_value,
    _orphan_count,
    _quantiles,
    _storage_breakdown,
)
from .recoverable_generator import (
    RecoverableV2Config,
    generate_recoverable_v2_task,
    seed_for_recoverable_v2_task,
)


CORPUS_NAME = "dfhybrid_v2_recoverable"
TOTAL_TASKS = 65_536
SPLIT_COUNTS = {"train": 57_344, "validation": 4_096, "test": 4_096}
TASKS_PER_SHARD = 256
SMOKE_TASKS = 512
DEFAULT_ROOT = Path(
    "/data/luolie/DF-Cluster/data/synthetic/dfhybrid_v2_recoverable"
)
DEFAULT_CLM_REPOSITORY = Path("/home/luolie/DF-Cluster/baseline/external/clm")
DEFAULT_ACCEPTANCE_REPORT = Path(
    "/data/luolie/DF-Cluster/outputs/data_audit/"
    "dfhybrid_v2_train512_recoverability_v1/report.json"
)
MAX_PROJECTED_GIB = 100.0
MAX_DEGENERATE_FRACTION = 0.0
MAX_CLM_FAILURE_FRACTION = 0.005

for _name, _value in THREAD_LIMITS.items():
    os.environ.setdefault(_name, _value)


@dataclass(frozen=True)
class ShardJob:
    split: str
    shard_index: int
    start_index: int
    task_count: int
    root: str
    clm_repository: str
    compression: str


def _load_and_validate_protocol(module_root: Path) -> dict[str, Any]:
    path = module_root / "configs" / "dfhybrid_v2_recoverable.yaml"
    protocol = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = RecoverableV2Config()
    expected = {
        "protocol.name": config.name,
        "protocol.version": config.protocol_version,
        "sampling.tasks": SMOKE_TASKS,
        "sampling.split": "train",
        "sampling.base_seed": config.base_seed,
        "sampling.n_samples": [256, 512, 1024],
        "sampling.intrinsic_dims": [8, 16, 32],
        "sampling.minimum_cluster_size": config.minimum_cluster_size,
        "sampling.n_clusters_max": config.max_clusters,
        "observation.cubic_strength": config.cubic_strength,
        "observation.additive_noise_std": config.additive_noise_std,
        "observation.row_view_pattern.both": config.both_view_fraction,
        "observation.row_view_pattern.view_1_only": config.view_one_only_fraction,
        "observation.row_view_pattern.view_2_only": config.view_two_only_fraction,
    }
    for dotted, value in expected.items():
        actual: Any = protocol
        for part in dotted.split("."):
            actual = actual[part]
        if actual != value:
            raise RuntimeError(
                f"YAML/code contract mismatch for {dotted}: {actual!r} != {value!r}"
            )
    return protocol


def _acceptance_certificate() -> dict[str, Any]:
    if not DEFAULT_ACCEPTANCE_REPORT.is_file():
        raise RuntimeError(
            "formal train-512 recoverability certificate is missing: "
            f"{DEFAULT_ACCEPTANCE_REPORT}"
        )
    report = json.loads(DEFAULT_ACCEPTANCE_REPORT.read_text(encoding="utf-8"))
    audit_source = Path(__file__).resolve().with_name("audit_recoverable_v2.py")
    if report.get("code_sha256") != _file_sha256(audit_source):
        raise RuntimeError("recoverability certificate audit-code hash mismatch")
    if report.get("config") != asdict(RecoverableV2Config()):
        raise RuntimeError("recoverability certificate generator-config mismatch")
    if report.get("gates_frozen") != RECOVERABILITY_GATES:
        raise RuntimeError("recoverability certificate frozen-gate mismatch")
    if report.get("status") != "passed":
        raise RuntimeError("formal train-512 recoverability certificate did not pass")
    if report.get("split") != "train" or report.get("task_count") != SMOKE_TASKS:
        raise RuntimeError("recoverability certificate split/task count mismatch")
    if report.get("label_arrays_consumed_by_acceptance") is not False:
        raise RuntimeError("recoverability certificate consumed label arrays")
    if report.get("K_ARI_NMI_CLM_consumed_by_acceptance") is not False:
        raise RuntimeError("recoverability certificate consumed forbidden fields")
    expected_gate_names = {
        "raw_cka_median",
        "oracle_cka_median",
        "oracle_cka_q10",
        "delta_cka_median",
        "delta_cka_q10",
        "oracle_stress_median",
        "regeneration",
        "view_coverage",
    }
    gates = report.get("gates")
    if not isinstance(gates, dict) or set(gates) != expected_gate_names:
        raise RuntimeError("recoverability certificate gate set is incomplete")
    if not all(gate.get("pass") is True for gate in gates.values()):
        raise RuntimeError("recoverability certificate contains a failed gate")
    return report


@contextmanager
def _exclusive_run_lock(root: Path) -> Iterator[None]:
    if os.name != "posix":
        raise RuntimeError("formal corpus locking requires a POSIX host")
    import fcntl

    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".generation.lock"
    with lock_path.open("a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another corpus generation process holds the lock") from exc
        stream.seek(0)
        stream.truncate()
        stream.write(f"pid={os.getpid()} started_unix={time.time()}\n")
        stream.flush()
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _resolved_config(
    root: Path, clm_repository: Path, compression: str
) -> dict[str, Any]:
    module_root = Path(__file__).resolve().parent
    project_root = module_root.parents[1]
    protocol_yaml = _load_and_validate_protocol(module_root)
    acceptance = _acceptance_certificate()
    clm_status = _git_value(
        clm_repository, "status", "--porcelain", "--untracked-files=all"
    )
    if clm_status:
        raise RuntimeError("CLM repository is dirty; frozen audit code is required")
    clm_source = clm_repository / "measures" / "calinski_harabasz.py"
    code_files = [
        module_root / "recoverable_generator.py",
        module_root / "audit_recoverable_v2.py",
        module_root / "clm_audit.py",
        module_root / "dataset_io.py",
        module_root / "generate_corpus.py",
        module_root / "generate_recoverable_corpus.py",
        module_root / "configs" / "dfhybrid_v2_recoverable.yaml",
    ]
    code_hashes = {
        str(path.relative_to(project_root)): _file_sha256(path)
        for path in code_files
    }
    sources = {
        "clm": {
            "path": str(clm_repository.resolve()),
            "commit": CLM_COMMIT,
            "head": _git_value(clm_repository, "rev-parse", "HEAD"),
            "remote": _git_value(
                clm_repository, "remote", "get-url", "origin"
            ),
            "working_tree_clean": True,
            "official_source_sha256": _file_sha256(clm_source),
        }
    }
    payload: dict[str, Any] = {
        "corpus": {
            "name": CORPUS_NAME,
            "root": str(root.resolve()),
            "total_tasks": TOTAL_TASKS,
            "split_counts": SPLIT_COUNTS,
            "split_claim": "independent_rng_streams_per_split_and_task_index",
            "tasks_per_shard": TASKS_PER_SHARD,
            "compression": compression,
        },
        "protocol": asdict(RecoverableV2Config()),
        "protocol_yaml": protocol_yaml,
        "recoverability_certificate": {
            "path": str(DEFAULT_ACCEPTANCE_REPORT),
            "sha256": _file_sha256(DEFAULT_ACCEPTANCE_REPORT),
            "split": acceptance["split"],
            "task_count": acceptance["task_count"],
            "status": acceptance["status"],
        },
        "claim_boundary": (
            "known paired-view orthogonal/componentwise-cubic family only; "
            "not generic tabular or nonlinear-ICA identifiability"
        ),
        "storage_contract": {
            "training_arrays": ["features", "missing_mask", "feature_types"],
            "audit_only_feature_array": "clean_signal",
            "label_sidecar_arrays": ["labels"],
            "label_sidecar_metadata": ["K", "CLM"],
            "analytic_oracle_written": False,
        },
        "clm": {
            "measure": "official adjusted Calinski-Harabasz CH_A",
            "upstream_commit": CLM_COMMIT,
            "logistic_k": CLM_LOGISTIC_K,
            "primary": "clm_cha_observed",
            "positive_control": "clm_cha_clean_control",
            "storage": "label_sidecar_only",
        },
        "sources": sources,
        "code_hashes": code_hashes,
        "environment": _environment_report(),
        "hard_stops": {
            "projected_corpus_gib_max": MAX_PROJECTED_GIB,
            "required_free_space_after_projection_gib": MIN_FREE_AFTER_GIB,
            "maximum_degenerate_task_fraction": MAX_DEGENERATE_FRACTION,
            "maximum_incomplete_clm_fraction": MAX_CLM_FAILURE_FRACTION,
            "nonfinite_tasks": 0,
            "cross_worker_hash_mismatches": 0,
            "split_seed_overlap": 0,
            "orphan_shards": 0,
            "partial_files": 0,
        },
    }
    payload["resolved_config_sha256"] = stable_sha256(payload)
    return payload


def _task_is_degenerate(task: Any) -> tuple[bool, list[str]]:
    metadata = task.metadata
    reasons: list[str] = []
    if metadata["clean_rank"] < metadata["intrinsic_dim"]:
        reasons.append("clean_rank_deficient")
    counts = np.bincount(task.labels.astype(np.int64))
    if (
        len(counts) != metadata["n_clusters"]
        or int(counts.min()) < RecoverableV2Config().minimum_cluster_size
    ):
        reasons.append("invalid_cluster_counts")
    if metadata["rows_without_view"] != 0:
        reasons.append("row_without_observation_view")
    observed_per_column = (~task.missing_mask).sum(axis=0)
    if int(observed_per_column.min()) == 0:
        reasons.append("fully_missing_observation_column")
    # The frozen generator stores masks as uint8 for HDF5 compatibility;
    # cast explicitly because uint8 array indexing is positional, not boolean.
    if not np.all(task.features[task.missing_mask.astype(bool)] == 0.0):
        reasons.append("masked_cell_sentinel_violation")
    return bool(reasons), reasons


def _guard_existing_shards(root: Path, resolved: dict[str, Any]) -> None:
    existing = list((root / "features").rglob("*.h5")) + list(
        (root / "labels").rglob("*.h5")
    )
    if not existing:
        return
    config_path = root / "reports" / "resolved_config.json"
    if not config_path.is_file():
        raise RuntimeError("existing shards have no resolved config")
    previous = json.loads(config_path.read_text(encoding="utf-8"))
    if previous.get("resolved_config_sha256") != resolved.get(
        "resolved_config_sha256"
    ):
        raise RuntimeError(
            "existing shards were produced by different code/config; "
            "use a fresh versioned corpus root"
        )


def _recover_incomplete_pairs(root: Path) -> list[str]:
    candidates: list[Path] = []
    for partial in list((root / "features").rglob("*.partial")) + list(
        (root / "labels").rglob("*.partial")
    ):
        candidates.append(partial)
    feature_root = root / "features"
    label_root = root / "labels"
    feature_rel = {
        path.relative_to(feature_root): path for path in feature_root.rglob("*.h5")
    }
    label_rel = {
        path.relative_to(label_root): path for path in label_root.rglob("*.h5")
    }
    for relative in feature_rel.keys() ^ label_rel.keys():
        candidates.append(feature_rel.get(relative) or label_rel[relative])
    if not candidates:
        return []
    quarantine = (
        root.parent / f"{root.name}.recovery" / f"incomplete-{time.time_ns()}"
    )
    moved: list[str] = []
    for source in sorted(set(candidates)):
        relative = source.relative_to(root)
        target = quarantine / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))
        moved.append(f"{source} -> {target}")
    return moved


def _expected_shard_paths(root: Path) -> tuple[set[Path], set[Path]]:
    features: set[Path] = set()
    labels: set[Path] = set()
    for split, count in SPLIT_COUNTS.items():
        for shard in range(count // TASKS_PER_SHARD):
            stem = f"shard-{shard:05d}.h5"
            features.add(root / "features" / split / stem)
            labels.add(root / "labels" / split / stem)
    return features, labels


def _build_payload(
    task_index: int, split: str, clm_repository: str
) -> tuple[TaskPayload, dict[str, Any]]:
    task = generate_recoverable_v2_task(task_index, split)  # type: ignore[arg-type]
    audit = compute_clm_audit(task, clm_repository)
    if audit.status != "complete" or audit.observed is None:
        raise RuntimeError(
            f"CLM failed for {task.task_id}: {audit.error or audit.status}"
        )
    degenerate, reasons = _task_is_degenerate(task)
    feature_attrs = {
        "generator": task.metadata["generator"],
        "generator_protocol_version": task.metadata[
            "generator_protocol_version"
        ],
        "split": split,
        "clean_signal_usage": "audit_only_do_not_load_for_training",
        "analytic_oracle_written": False,
    }
    task_contract_sha256 = stable_sha256(
        {
            "task_metadata": task.metadata,
            "degenerate": degenerate,
            "degeneracy_reasons": reasons,
            "feature_attrs": feature_attrs,
        }
    )
    feature_attrs["task_contract_sha256"] = task_contract_sha256
    audit_metadata = {
        **task.metadata,
        "degenerate": degenerate,
        "degeneracy_reasons": reasons,
        "analytic_oracle_written": False,
        "task_contract_sha256": task_contract_sha256,
    }
    payload = TaskPayload(
        task_id=task.task_id,
        features=task.features,
        clean_signal=task.clean_signal,
        missing_mask=task.missing_mask,
        feature_types=task.feature_types,
        labels=task.labels,
        K=int(task.metadata["n_clusters"]),
        CLM=float(audit.observed),
        clm_cha_observed=float(audit.observed),
        clm_cha_clean_control=audit.clean_control,
        clm_status=audit.status,
        clm_error=audit.error,
        feature_attrs=feature_attrs,
        audit_attrs=audit_metadata,
    )
    summary = {
        "task_id": task.task_id,
        "task_index": task_index,
        "split": split,
        "seed": int(task.metadata["seed"]),
        "artifact_sha256": task.metadata["artifact_sha256"],
        "task_contract_sha256": task_contract_sha256,
        "n_samples": int(task.metadata["n_samples"]),
        "n_features": int(task.metadata["n_features"]),
        "intrinsic_dim": int(task.metadata["intrinsic_dim"]),
        "n_clusters": int(task.metadata["n_clusters"]),
        "clm_cha_observed": float(audit.observed),
        "clm_cha_clean_control": float(audit.clean_control),
        "degenerate": degenerate,
        "degeneracy_reasons": reasons,
        "view_pattern_counts": task.metadata["view_pattern_counts"],
    }
    return payload, summary


def _run_shard(job: ShardJob) -> dict[str, Any]:
    root = Path(job.root)
    started = time.perf_counter()
    payloads: list[TaskPayload] = []
    summaries: list[dict[str, Any]] = []
    for task_index in range(job.start_index, job.start_index + job.task_count):
        payload, summary = _build_payload(
            task_index, job.split, job.clm_repository
        )
        payloads.append(payload)
        summaries.append(summary)
    stem = f"shard-{job.shard_index:05d}.h5"
    result = write_multi_task_shards(
        payloads,
        feature_path=root / "features" / job.split / stem,
        labels_path=root / "labels" / job.split / stem,
        worker_id=f"shard-{job.split}-{job.shard_index:05d}",
        compression=job.compression,
        resume=True,
    )
    return {
        "split": job.split,
        "shard_index": job.shard_index,
        "start_index": job.start_index,
        "task_count": job.task_count,
        "features_path": str(result.features_path),
        "labels_path": str(result.labels_path),
        "resumed": result.resumed,
        "elapsed_seconds": time.perf_counter() - started,
        "training_manifest_records": list(result.training_manifest_records),
        "audit_manifest_records": list(result.audit_manifest_records),
        "task_summaries": summaries,
    }


def _jobs(
    mode: str, root: Path, clm_repository: Path, compression: str
) -> list[ShardJob]:
    del root
    if mode == "smoke":
        return [
            ShardJob(
                "train",
                shard,
                shard * TASKS_PER_SHARD,
                TASKS_PER_SHARD,
                str(DEFAULT_ROOT),
                str(clm_repository),
                compression,
            )
            for shard in range(SMOKE_TASKS // TASKS_PER_SHARD)
        ]
    jobs: list[ShardJob] = []
    for split, count in SPLIT_COUNTS.items():
        for shard in range(count // TASKS_PER_SHARD):
            jobs.append(
                ShardJob(
                    split,
                    shard,
                    shard * TASKS_PER_SHARD,
                    TASKS_PER_SHARD,
                    str(DEFAULT_ROOT),
                    str(clm_repository),
                    compression,
                )
            )
    return jobs


def _cross_process_hash_mismatches(
    summaries: list[dict[str, Any]], checks_per_call: int = 4
) -> int:
    if not summaries:
        return 0
    indices = np.linspace(
        0, len(summaries) - 1, min(checks_per_call, len(summaries)), dtype=int
    )
    mismatches = 0
    for index in np.unique(indices):
        summary = summaries[int(index)]
        regenerated = generate_recoverable_v2_task(
            summary["task_index"], summary["split"]
        )
        if regenerated.metadata["artifact_sha256"] != summary["artifact_sha256"]:
            mismatches += 1
    return mismatches


def _split_overlap_count() -> int:
    seeds: dict[str, set[int]] = {}
    for split, count in SPLIT_COUNTS.items():
        seeds[split] = {
            seed_for_recoverable_v2_task(index, split)  # type: ignore[arg-type]
            for index in range(count)
        }
    return sum(
        len(seeds[left] & seeds[right])
        for left, right in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        )
    )


def _projection_report(
    root: Path, results: list[dict[str, Any]]
) -> dict[str, Any]:
    task_count = sum(result["task_count"] for result in results)
    paths = [
        Path(result[key])
        for result in results
        for key in ("features_path", "labels_path")
    ]
    file_bytes = sum(path.stat().st_size for path in paths)
    projected_bytes = int(np.ceil(file_bytes / task_count * TOTAL_TASKS))
    remaining = max(0, projected_bytes - file_bytes)
    free_after = shutil.disk_usage(root).free - remaining
    passed = (
        projected_bytes / 2**30 < MAX_PROJECTED_GIB
        and free_after / 2**30 >= MIN_FREE_AFTER_GIB
    )
    return {
        "status": "passed" if passed else "failed",
        "task_count": task_count,
        "file_bytes": file_bytes,
        "bytes_per_task": file_bytes / task_count,
        "projected_corpus_gib": projected_bytes / 2**30,
        "free_after_projection_gib": free_after / 2**30,
        "projected_limit_gib": MAX_PROJECTED_GIB,
        "free_after_limit_gib": MIN_FREE_AFTER_GIB,
    }


def _verify_smoke_certificate(
    summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    report = _acceptance_certificate()
    certified = {
        item["task_id"]: item["artifact_sha256"] for item in report["per_task"]
    }
    generated = {
        item["task_id"]: item["artifact_sha256"] for item in summaries
    }
    mismatches = sorted(
        task_id
        for task_id in certified.keys() | generated.keys()
        if certified.get(task_id) != generated.get(task_id)
    )
    return {
        "status": "passed" if not mismatches else "failed",
        "report_path": str(DEFAULT_ACCEPTANCE_REPORT),
        "report_sha256": _file_sha256(DEFAULT_ACCEPTANCE_REPORT),
        "certified_task_count": len(certified),
        "generated_task_count": len(generated),
        "artifact_mismatch_count": len(mismatches),
        "first_mismatches": mismatches[:10],
    }


def _smoke_report(
    root: Path, results: list[dict[str, Any]], elapsed: float
) -> dict[str, Any]:
    summaries = [
        summary for result in results for summary in result["task_summaries"]
    ]
    feature_paths = [Path(result["features_path"]) for result in results]
    label_paths = [Path(result["labels_path"]) for result in results]
    storage = _storage_breakdown(feature_paths, label_paths)
    projection = _projection_report(root, results)
    clm_failures = sum(
        not np.isfinite(item["clm_cha_observed"]) for item in summaries
    )
    degenerate = sum(item["degenerate"] for item in summaries)
    hash_mismatches = _cross_process_hash_mismatches(
        summaries, checks_per_call=64
    )
    split_overlap = _split_overlap_count()
    orphan_count = _orphan_count(root)
    partial_count = len(list(root.rglob("*.partial")))
    certificate = _verify_smoke_certificate(summaries)
    hard_stops = {
        "recoverability_certificate": {
            "value": certificate["artifact_mismatch_count"],
            "limit": 0,
            "pass": certificate["status"] == "passed",
        },
        "projected_corpus_gib": {
            "value": projection["projected_corpus_gib"],
            "limit": MAX_PROJECTED_GIB,
            "pass": projection["projected_corpus_gib"] < MAX_PROJECTED_GIB,
        },
        "free_after_projection_gib": {
            "value": projection["free_after_projection_gib"],
            "limit": MIN_FREE_AFTER_GIB,
            "pass": projection["free_after_projection_gib"]
            >= MIN_FREE_AFTER_GIB,
        },
        "degenerate_fraction": {
            "value": degenerate / len(summaries),
            "limit": MAX_DEGENERATE_FRACTION,
            "pass": degenerate == 0,
        },
        "incomplete_clm_fraction": {
            "value": clm_failures / len(summaries),
            "limit": MAX_CLM_FAILURE_FRACTION,
            "pass": clm_failures / len(summaries)
            <= MAX_CLM_FAILURE_FRACTION,
        },
        "cross_process_hash_mismatches": {
            "value": hash_mismatches,
            "limit": 0,
            "pass": hash_mismatches == 0,
        },
        "split_seed_overlap": {
            "value": split_overlap,
            "limit": 0,
            "pass": split_overlap == 0,
        },
        "orphan_shards": {
            "value": orphan_count,
            "limit": 0,
            "pass": orphan_count == 0,
        },
        "partial_files": {
            "value": partial_count,
            "limit": 0,
            "pass": partial_count == 0,
        },
    }
    return {
        "status": (
            "passed" if all(item["pass"] for item in hard_stops.values())
            else "failed"
        ),
        "task_count": len(summaries),
        "elapsed_seconds": elapsed,
        "tasks_per_second": len(summaries) / elapsed,
        "storage": {
            **storage,
            "bytes_per_task": storage["file_bytes"] / len(summaries),
            "projected_corpus_bytes": int(
                np.ceil(storage["file_bytes"] / len(summaries) * TOTAL_TASKS)
            ),
        },
        "projection": projection,
        "recoverability_certificate": certificate,
        "clm_cha_observed": _quantiles(
            [item["clm_cha_observed"] for item in summaries]
        ),
        "clm_cha_clean_control": _quantiles(
            [item["clm_cha_clean_control"] for item in summaries]
        ),
        "n_samples_counts": {
            str(value): sum(item["n_samples"] == value for item in summaries)
            for value in (256, 512, 1024)
        },
        "intrinsic_dim_counts": {
            str(value): sum(
                item["intrinsic_dim"] == value for item in summaries
            )
            for value in (8, 16, 32)
        },
        "hard_stops": hard_stops,
        "smoke_is_performance_evidence": False,
    }


def _require_passing_smoke(root: Path, resolved: dict[str, Any]) -> None:
    smoke_path = root / "reports" / "smoke_report.json"
    config_path = root / "reports" / "resolved_config.json"
    if not smoke_path.is_file() or not config_path.is_file():
        raise RuntimeError("full generation requires completed smoke artifacts")
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    previous = json.loads(config_path.read_text(encoding="utf-8"))
    if smoke.get("status") != "passed":
        raise RuntimeError("full generation is blocked because smoke did not pass")
    if previous.get("resolved_config_sha256") != resolved.get(
        "resolved_config_sha256"
    ):
        raise RuntimeError(
            "resolved config/code changed after smoke; rerun smoke first"
        )


def _training_record_has_forbidden_key(value: Any) -> bool:
    if isinstance(value, dict):
        for key, nested in value.items():
            normalized = "".join(character for character in str(key).lower() if character.isalnum())
            if normalized in {
                "y",
                "k",
                "clm",
                "labels",
                "labelpath",
                "labelspath",
                "labelsgrouppath",
            }:
                return True
            if _training_record_has_forbidden_key(nested):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_training_record_has_forbidden_key(item) for item in value)
    return False


def _final_integrity_report(
    root: Path,
    results: list[dict[str, Any]],
    training_records: list[dict[str, Any]],
    audit_records: list[dict[str, Any]],
) -> dict[str, Any]:
    expected_features, expected_labels = _expected_shard_paths(root)
    actual_features = set((root / "features").rglob("*.h5"))
    actual_labels = set((root / "labels").rglob("*.h5"))
    summaries = [
        summary for result in results for summary in result["task_summaries"]
    ]
    summary_by_id = {summary["task_id"]: summary for summary in summaries}
    training_ids = [record["task_id"] for record in training_records]
    audit_ids = [record["task_id"] for record in audit_records]
    hdf_task_ids: list[str] = []
    contract_mismatches = 0
    artifact_mismatches = 0
    schema_violations = 0
    stored_degenerate = 0
    nonfinite_clm = 0
    for feature_path in sorted(expected_features & actual_features):
        relative = feature_path.relative_to(root / "features")
        label_path = root / "labels" / relative
        if label_path not in actual_labels:
            continue
        with h5py.File(feature_path, "r") as feature_file, h5py.File(
            label_path, "r"
        ) as label_file:
            feature_ids = set(feature_file["tasks"].keys())
            label_ids = set(label_file["tasks"].keys())
            if feature_ids != label_ids:
                schema_violations += len(feature_ids ^ label_ids) or 1
            for task_id in sorted(feature_ids & label_ids):
                hdf_task_ids.append(task_id)
                feature_group = feature_file["tasks"][task_id]
                label_group = label_file["tasks"][task_id]
                if set(feature_group.keys()) != {
                    "features",
                    "clean_signal",
                    "missing_mask",
                    "feature_types",
                }:
                    schema_violations += 1
                if set(label_group.keys()) != {"labels"}:
                    schema_violations += 1
                summary = summary_by_id.get(task_id)
                if summary is None:
                    contract_mismatches += 1
                    continue
                expected_contract = summary["task_contract_sha256"]
                if str(feature_group.attrs.get("task_contract_sha256", "")) != expected_contract:
                    contract_mismatches += 1
                if str(label_group.attrs.get("task_contract_sha256", "")) != expected_contract:
                    contract_mismatches += 1
                if str(label_group.attrs.get("artifact_sha256", "")) != summary["artifact_sha256"]:
                    artifact_mismatches += 1
                if bool(label_group.attrs.get("degenerate", True)):
                    stored_degenerate += 1
                clm_value = float(label_group.attrs.get("clm_cha_observed", np.nan))
                if not np.isfinite(clm_value):
                    nonfinite_clm += 1

    split_counts = {
        split: sum(summary["split"] == split for summary in summaries)
        for split in SPLIT_COUNTS
    }
    hard_stops = {
        "exact_feature_shards": {
            "value": len(actual_features),
            "expected": len(expected_features),
            "pass": actual_features == expected_features,
        },
        "exact_label_shards": {
            "value": len(actual_labels),
            "expected": len(expected_labels),
            "pass": actual_labels == expected_labels,
        },
        "training_manifest_records": {
            "value": len(training_records),
            "expected": TOTAL_TASKS,
            "pass": len(training_records) == TOTAL_TASKS,
        },
        "audit_manifest_records": {
            "value": len(audit_records),
            "expected": TOTAL_TASKS,
            "pass": len(audit_records) == TOTAL_TASKS,
        },
        "unique_training_task_ids": {
            "value": len(set(training_ids)),
            "expected": TOTAL_TASKS,
            "pass": len(set(training_ids)) == TOTAL_TASKS,
        },
        "unique_audit_task_ids": {
            "value": len(set(audit_ids)),
            "expected": TOTAL_TASKS,
            "pass": len(set(audit_ids)) == TOTAL_TASKS,
        },
        "manifest_task_id_alignment": {
            "value": len(set(training_ids) ^ set(audit_ids)),
            "expected": 0,
            "pass": set(training_ids) == set(audit_ids),
        },
        "hdf_task_id_alignment": {
            "value": len(set(hdf_task_ids) ^ set(training_ids)),
            "expected": 0,
            "pass": set(hdf_task_ids) == set(training_ids)
            and len(hdf_task_ids) == TOTAL_TASKS,
        },
        "split_counts": {
            "value": split_counts,
            "expected": SPLIT_COUNTS,
            "pass": split_counts == SPLIT_COUNTS,
        },
        "training_forbidden_records": {
            "value": sum(
                _training_record_has_forbidden_key(record)
                for record in training_records
            ),
            "expected": 0,
            "pass": not any(
                _training_record_has_forbidden_key(record)
                for record in training_records
            ),
        },
        "schema_violations": {
            "value": schema_violations,
            "expected": 0,
            "pass": schema_violations == 0,
        },
        "task_contract_mismatches": {
            "value": contract_mismatches,
            "expected": 0,
            "pass": contract_mismatches == 0,
        },
        "artifact_hash_mismatches": {
            "value": artifact_mismatches,
            "expected": 0,
            "pass": artifact_mismatches == 0,
        },
        "degenerate_tasks": {
            "value": sum(summary["degenerate"] for summary in summaries)
            + stored_degenerate,
            "expected": 0,
            "pass": not any(summary["degenerate"] for summary in summaries)
            and stored_degenerate == 0,
        },
        "nonfinite_clm_tasks": {
            "value": nonfinite_clm,
            "expected": 0,
            "pass": nonfinite_clm == 0,
        },
        "partial_files": {
            "value": len(list(root.rglob("*.partial"))),
            "expected": 0,
            "pass": not list(root.rglob("*.partial")),
        },
    }
    return {
        "status": (
            "passed" if all(item["pass"] for item in hard_stops.values())
            else "failed"
        ),
        "hard_stops": hard_stops,
        "feature_file_bytes": sum(path.stat().st_size for path in actual_features),
        "label_file_bytes": sum(path.stat().st_size for path in actual_labels),
    }


def run(
    mode: str,
    root: Path,
    clm_repository: Path,
    workers: int,
    compression: str,
) -> dict[str, Any]:
    root = root.resolve()
    if root != DEFAULT_ROOT.resolve():
        raise ValueError(f"formal corpus root is frozen at {DEFAULT_ROOT}")
    root.mkdir(parents=True, exist_ok=True)
    resolved = _resolved_config(root, clm_repository, compression)
    if resolved["sources"]["clm"]["head"] != CLM_COMMIT:
        raise RuntimeError("CLM repository HEAD differs from frozen commit")
    _guard_existing_shards(root, resolved)
    if mode == "full":
        _require_passing_smoke(root, resolved)
    recovered_files = _recover_incomplete_pairs(root)
    if recovered_files:
        logging.warning("quarantined incomplete shard artifacts: %s", recovered_files)
    _atomic_json(root / "reports" / "resolved_config.json", resolved)
    _atomic_json(root / "reports" / "environment.json", resolved["environment"])

    jobs = _jobs(mode, root, clm_repository, compression)
    batches = (
        [jobs]
        if mode == "smoke"
        else [jobs[index : index + 20] for index in range(0, len(jobs), 20)]
    )
    started_wall = time.time()
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    completed = 0
    early_report: dict[str, Any] | None = None
    scale_checkpoints: list[dict[str, Any]] = []
    for batch_index, batch in enumerate(batches):
        with ProcessPoolExecutor(max_workers=min(workers, len(batch))) as pool:
            futures = {pool.submit(_run_shard, job): job for job in batch}
            for future in as_completed(futures):
                job = futures[future]
                result = future.result()
                if _cross_process_hash_mismatches(result["task_summaries"]):
                    raise RuntimeError(
                        "cross-process artifact mismatch in "
                        f"{job.split} shard {job.shard_index}"
                    )
                results.append(result)
                completed += 1
                logging.info(
                    "[%d/%d] %s shard %05d tasks=%d resumed=%s seconds=%.2f",
                    completed,
                    len(jobs),
                    job.split,
                    job.shard_index,
                    job.task_count,
                    result["resumed"],
                    result["elapsed_seconds"],
                )
        if mode == "full":
            checkpoint = _projection_report(root, results)
            checkpoint["completed_shards"] = len(results)
            checkpoint["completed_tasks"] = sum(
                result["task_count"] for result in results
            )
            scale_checkpoints.append(checkpoint)
            _atomic_json(
                root / "reports" / "scale_checkpoint_report.json",
                {"checkpoints": scale_checkpoints},
            )
            if batch_index == 0:
                early_report = checkpoint
                _atomic_json(
                    root / "reports" / "early_5120_report.json", early_report
                )
            if checkpoint["status"] != "passed":
                raise RuntimeError(
                    f"scale hard stop failed after {checkpoint['completed_tasks']} tasks"
                )

    results.sort(
        key=lambda item: (
            list(SPLIT_COUNTS).index(item["split"]),
            item["shard_index"],
        )
    )
    elapsed = time.perf_counter() - started
    training_records = [
        record
        for result in results
        for record in result["training_manifest_records"]
    ]
    audit_records = [
        record
        for result in results
        for record in result["audit_manifest_records"]
    ]
    suffix = ".smoke" if mode == "smoke" else ""
    training_path = root / "manifests" / f"training_manifest{suffix}.jsonl"
    audit_path = root / "manifests" / f"audit_manifest{suffix}.jsonl"
    _atomic_jsonl(training_path, training_records)
    _atomic_jsonl(audit_path, audit_records)

    if mode == "smoke":
        report = _smoke_report(root, results, elapsed)
        _atomic_json(root / "reports" / "smoke_report.json", report)
        status = (
            "completed_smoke"
            if report["status"] == "passed"
            else "failed_smoke_gate"
        )
    else:
        integrity = _final_integrity_report(
            root, results, training_records, audit_records
        )
        report = {
            "status": "completed" if integrity["status"] == "passed" else "failed_integrity",
            "task_count": len(training_records),
            "elapsed_seconds": elapsed,
            "tasks_per_second": len(training_records) / elapsed,
            "resumed_shards": sum(result["resumed"] for result in results),
            "new_shards": sum(not result["resumed"] for result in results),
            "orphan_shards": _orphan_count(root),
            "partial_files": len(list(root.rglob("*.partial"))),
            "cross_process_hash_mismatches": 0,
            "early_5120_report": early_report,
            "scale_checkpoints": scale_checkpoints,
            "final_integrity": integrity,
            "training_manifest_sha256": _file_sha256(training_path),
            "audit_manifest_sha256": _file_sha256(audit_path),
            "analytic_oracle_written": False,
        }
        _atomic_json(root / "reports" / "corpus_report.json", report)
        if integrity["status"] != "passed":
            raise RuntimeError("final corpus integrity hard stop failed")
        status = "completed"
    status_record = {
        "status": status,
        "mode": mode,
        "started_unix": started_wall,
        "finished_unix": time.time(),
        "elapsed_seconds": elapsed,
        "task_count": len(training_records),
        "resolved_config_sha256": resolved["resolved_config_sha256"],
        "error": None,
    }
    _atomic_json(root / "reports" / "run_status.json", status_record)
    return {"status": status_record, "report": report}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--clm-repository", type=Path, default=DEFAULT_CLM_REPOSITORY
    )
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--compression", choices=("lzf", "gzip1", "none"), default="lzf"
    )
    args = parser.parse_args()
    workers = args.workers or (2 if args.mode == "smoke" else 48)
    if workers < 1 or workers > 96:
        parser.error("workers must be in [1, 96]")
    root = args.root.resolve()
    log_path = root / "reports" / "generation.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    try:
        with _exclusive_run_lock(root):
            result = run(
                args.mode,
                root,
                args.clm_repository.resolve(),
                workers,
                args.compression,
            )
        logging.info("generation finished: %s", result["status"])
        print(json.dumps(result["status"], sort_keys=True), flush=True)
        if result["status"]["status"] == "failed_smoke_gate":
            raise SystemExit(2)
    except Exception as exc:
        error_record = {
            "status": "incomplete_compute",
            "mode": args.mode,
            "finished_unix": time.time(),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        _atomic_json(root / "reports" / "run_status.json", error_record)
        logging.exception("generation failed")
        raise


if __name__ == "__main__":
    main()
