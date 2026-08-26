# ⚠️  DEPRECATED: Part of terminated V1/V2/V3 research line. See DEPRECATION_NOTICE.md
"""Run the frozen no-label V2-P0 recoverability acceptance audit."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
import time
import traceback
from typing import Any

import numpy as np

from .recoverable_generator import (
    SPLIT_OFFSETS,
    RecoverableV2Config,
    Split,
    generate_recoverable_v2_task,
)


DEFAULT_OUTPUT = Path(
    "/data/luolie/DF-Cluster/outputs/data_audit/"
    "dfhybrid_v2_train512_recoverability_v1"
)
GATES = {
    "max_raw_cka_median": 0.80,
    "min_oracle_cka_median": 0.95,
    "min_oracle_cka_q10": 0.90,
    "min_oracle_minus_raw_cka_median": 0.15,
    "min_oracle_minus_raw_cka_q10": 0.10,
    "max_oracle_stress_median": 0.10,
}


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(f"{path}.partial")
    partial.write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def _standardize(values: np.ndarray, missing: np.ndarray) -> np.ndarray:
    valid = ~missing
    counts = valid.sum(axis=0).clip(min=1)
    means = (values * valid).sum(axis=0) / counts
    centered = np.where(valid, values - means, 0.0)
    std = np.sqrt((centered**2).sum(axis=0) / counts).clip(min=1.0e-4)
    return np.where(valid, np.clip((values - means) / std, -10.0, 10.0), 0.0)


def _normalize_signal(values: np.ndarray) -> np.ndarray:
    centered = values.astype(np.float64) - values.mean(axis=0, keepdims=True)
    return centered / max(float(np.sqrt(np.mean(centered**2))), 1.0e-8)


def _metrics(
    predicted: np.ndarray,
    clean: np.ndarray,
    task_id: str,
    landmarks: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(
        int.from_bytes(sha256(task_id.encode()).digest()[:8], "little")
    )
    count = min(landmarks, predicted.shape[0])
    rows = rng.choice(predicted.shape[0], size=count, replace=False)
    predicted = predicted[rows].astype(np.float64)
    clean = clean[rows].astype(np.float64)
    predicted -= predicted.mean(axis=0, keepdims=True)
    clean -= clean.mean(axis=0, keepdims=True)
    predicted_gram = predicted @ predicted.T
    clean_gram = clean @ clean.T
    predicted_gram /= max(float(np.linalg.norm(predicted_gram)), 1.0e-12)
    clean_gram /= max(float(np.linalg.norm(clean_gram)), 1.0e-12)
    cka = float(np.sum(predicted_gram * clean_gram))
    pairs = min(4096, count * (count - 1) // 2)
    first = rng.integers(0, count, size=pairs)
    second = rng.integers(0, count, size=pairs)
    keep = first != second
    predicted_distance = np.linalg.norm(predicted[first[keep]] - predicted[second[keep]], axis=1)
    clean_distance = np.linalg.norm(clean[first[keep]] - clean[second[keep]], axis=1)
    predicted_distance /= max(float(np.median(predicted_distance)), 1.0e-8)
    clean_distance /= max(float(np.median(clean_distance)), 1.0e-8)
    stress = float(np.mean(np.abs(predicted_distance - clean_distance)))
    return cka, stress


def _summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "median": float(np.median(array)),
        "q10": float(np.quantile(array, 0.10)),
        "q90": float(np.quantile(array, 0.90)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def run(
    output: Path,
    task_count: int,
    landmarks: int,
    split: Split = "train",
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    _atomic_json(output / "status.json", {"status": "running", "error": None})
    started = time.perf_counter()
    config = RecoverableV2Config()
    records: list[dict[str, Any]] = []
    try:
        task_ids: set[str] = set()
        for index in range(task_count):
            task = generate_recoverable_v2_task(index, split, config)
            raw = _standardize(task.features.astype(np.float64), task.missing_mask.astype(bool))
            clean = _normalize_signal(task.clean_signal)
            oracle = _normalize_signal(task.oracle_recovered)
            raw_cka, raw_stress = _metrics(raw, clean, task.task_id, landmarks)
            oracle_cka, oracle_stress = _metrics(oracle, clean, task.task_id, landmarks)
            if task.task_id in task_ids:
                raise RuntimeError("duplicate task id")
            task_ids.add(task.task_id)
            records.append(
                {
                    "task_id": task.task_id,
                    "artifact_sha256": task.metadata["artifact_sha256"],
                    "n_samples": task.metadata["n_samples"],
                    "intrinsic_dim": task.metadata["intrinsic_dim"],
                    "n_features": task.metadata["n_features"],
                    "raw_cka": raw_cka,
                    "raw_stress": raw_stress,
                    "oracle_cka": oracle_cka,
                    "oracle_stress": oracle_stress,
                    "oracle_minus_raw_cka": oracle_cka - raw_cka,
                    "rows_without_view": task.metadata["rows_without_view"],
                    "clean_separation_over_std": task.metadata[
                        "clean_separation_over_std"
                    ],
                }
            )

        regeneration_mismatches = 0
        for index in np.linspace(0, task_count - 1, num=min(64, task_count), dtype=np.int64):
            regenerated = generate_recoverable_v2_task(int(index), split, config)
            if regenerated.metadata["artifact_sha256"] != records[int(index)]["artifact_sha256"]:
                regeneration_mismatches += 1

        raw_cka = _summary([row["raw_cka"] for row in records])
        raw_stress = _summary([row["raw_stress"] for row in records])
        oracle_cka = _summary([row["oracle_cka"] for row in records])
        oracle_stress = _summary([row["oracle_stress"] for row in records])
        delta = _summary([row["oracle_minus_raw_cka"] for row in records])
        gates = {
            "raw_cka_median": {
                "value": raw_cka["median"],
                "threshold": GATES["max_raw_cka_median"],
                "operator": "<=",
                "pass": raw_cka["median"] <= GATES["max_raw_cka_median"],
            },
            "oracle_cka_median": {
                "value": oracle_cka["median"],
                "threshold": GATES["min_oracle_cka_median"],
                "operator": ">=",
                "pass": oracle_cka["median"] >= GATES["min_oracle_cka_median"],
            },
            "oracle_cka_q10": {
                "value": oracle_cka["q10"],
                "threshold": GATES["min_oracle_cka_q10"],
                "operator": ">=",
                "pass": oracle_cka["q10"] >= GATES["min_oracle_cka_q10"],
            },
            "delta_cka_median": {
                "value": delta["median"],
                "threshold": GATES["min_oracle_minus_raw_cka_median"],
                "operator": ">=",
                "pass": delta["median"]
                >= GATES["min_oracle_minus_raw_cka_median"],
            },
            "delta_cka_q10": {
                "value": delta["q10"],
                "threshold": GATES["min_oracle_minus_raw_cka_q10"],
                "operator": ">=",
                "pass": delta["q10"] >= GATES["min_oracle_minus_raw_cka_q10"],
            },
            "oracle_stress_median": {
                "value": oracle_stress["median"],
                "threshold": GATES["max_oracle_stress_median"],
                "operator": "<=",
                "pass": oracle_stress["median"]
                <= GATES["max_oracle_stress_median"],
            },
            "regeneration": {
                "value": regeneration_mismatches,
                "threshold": 0,
                "operator": "==",
                "pass": regeneration_mismatches == 0,
            },
            "view_coverage": {
                "value": max(row["rows_without_view"] for row in records),
                "threshold": 0,
                "operator": "==",
                "pass": max(row["rows_without_view"] for row in records) == 0,
            },
        }
        passed = all(gate["pass"] for gate in gates.values())
        report = {
            "status": "passed" if passed else "failed_gate",
            "purpose": "pre-method benchmark acceptance; not method performance",
            "split": split,
            "task_count": task_count,
            "landmarks_per_task_max": landmarks,
            "config": asdict(config),
            "gates_frozen": GATES,
            "gates": gates,
            "raw_cka": raw_cka,
            "raw_stress": raw_stress,
            "oracle_cka": oracle_cka,
            "oracle_stress": oracle_stress,
            "oracle_minus_raw_cka": delta,
            "labels_generated_for_future_sidecar": True,
            "label_arrays_consumed_by_acceptance": False,
            "K_ARI_NMI_CLM_consumed_by_acceptance": False,
            "oracle_reads_labels": False,
            "forward_reads_labels": False,
            "claim_scope": "analytic recovery for this known paired-view family only",
            "regeneration_checks": min(64, task_count),
            "regeneration_mismatches": regeneration_mismatches,
            "per_task": records,
            "elapsed_seconds": time.perf_counter() - started,
            "code_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        _atomic_json(output / "report.json", report)
        _atomic_json(
            output / "status.json",
            {
                "status": "completed" if passed else "completed_failed_gate",
                "error": None,
                "elapsed_seconds": report["elapsed_seconds"],
            },
        )
        return report
    except Exception as exc:
        _atomic_json(
            output / "status.json",
            {
                "status": "incomplete_compute",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--task-count", type=int, default=512)
    parser.add_argument("--landmarks", type=int, default=256)
    parser.add_argument("--split", choices=tuple(SPLIT_OFFSETS), default="train")
    args = parser.parse_args()
    report = run(
        args.output.resolve(),
        args.task_count,
        args.landmarks,
        args.split,
    )
    print(json.dumps({"output": str(args.output.resolve()), "status": report["status"]}))
    if report["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
