"""Fixed Stage-A validation manifest and X-only reconstruction evaluator.

The validation pool is built once from the full option-A candidate structural
manifest. It contains every 135 coverage cell and all 23,903 gate-passing
reference rows, but no labels or audit metrics. Validation workers generate a
fresh validation task and redact it to ``InputTask`` before returning to the
training process. A fixed per-task corruption seed makes milestone metrics
reproducible without opening any privileged target.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple

import torch

from .core import generate_v4_task, source_sha256
from .full_sampler import FullSamplerConfig, sample_full_task_config
from .input_loader import InputTask, redact_v4_task
from .models import GlobalAE, mixed_type_reconstruction_loss
from .training import StageACorruptionConfig, corrupt_table

VALIDATION_SEED = 2026082601
VALIDATION_SCHEMA = "dfcluster.generator_v4.stage_a.validation.v3"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _seed_from_row(validation_seed: int, proposal_task_id: str, index: int) -> int:
    raw = f"{int(validation_seed)}:{int(index)}:{proposal_task_id}".encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:16], 16) % (2**63 - 1)


def build_stage_a_validation_manifest(
    *,
    candidate_manifest_path: Path,
    output_root: Path,
    validation_seed: int = VALIDATION_SEED,
    tasks_per_cell: int | None = None,
) -> Dict[str, Any]:
    """Create the fixed full candidate-pool validation manifest."""

    candidate_manifest_path = Path(candidate_manifest_path)
    output_root = Path(output_root)
    if tasks_per_cell is not None and tasks_per_cell <= 0:
        raise ValueError("tasks_per_cell must be positive when provided")
    if output_root.exists():
        raise FileExistsError(output_root)
    proposals: List[Dict[str, Any]] = []
    seen: set[str] = set()
    with candidate_manifest_path.open(encoding="utf-8") as handle:
        for line in handle:
            proposal = json.loads(line)
            proposal_id = str(proposal["task_id"])
            if proposal_id in seen:
                raise ValueError(f"duplicate proposal task id: {proposal_id}")
            seen.add(proposal_id)
            proposals.append(proposal)
    groups: Dict[Tuple[str, str, str, int], List[Dict[str, Any]]] = {}
    for proposal in proposals:
        cell = (
            str(proposal["observation_stratum"]),
            str(proposal["information_stratum"]),
            str(proposal["raw_difficulty_pool"]),
            int(proposal["clm_tertile"]),
        )
        groups.setdefault(cell, []).append(proposal)
    rows: List[Dict[str, Any]] = []
    for cell in sorted(groups):
        ordered = sorted(groups[cell], key=lambda row: str(row["task_id"]))
        selected = ordered if tasks_per_cell is None else ordered[:tasks_per_cell]
        for proposal in selected:
            index = len(rows)
            proposal_id = str(proposal["task_id"])
            task_seed = _seed_from_row(validation_seed, proposal_id, index)
            task_id = f"generator_v4/validation/{task_seed}/{index}"
            rows.append({
                "task_id": task_id,
                "proposal_task_id": proposal_id,
                "validation_seed": int(validation_seed),
                "validation_task_index": int(index),
                "validation_task_seed": int(task_seed),
                "observation_stratum": proposal["observation_stratum"],
                "information_stratum": proposal["information_stratum"],
                "generator_seed": int(proposal["generator_seed"]),
                "schedule_task_index": int(proposal["schedule_task_index"]),
                "n_samples": int(proposal["n_samples"]),
                "n_features": int(proposal["n_features"]),
                "n_clusters": int(proposal["n_clusters"]),
                "intrinsic_dim": int(proposal["intrinsic_dim"]),
                "missing_rate": float(proposal["missing_rate"]),
                "source_sha256": str(proposal["source_sha256"]),
                "labels_in_manifest": False,
                "audit_metrics_in_manifest": False,
                "validation_protocol": "fixed_x_only_reconstruction_pool",
            })
    if not rows:
        raise ValueError("candidate manifest is empty")
    output_root.mkdir(parents=True, exist_ok=False)
    manifest_path = output_root / "validation_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    resolved = {
        "schema_version": VALIDATION_SCHEMA,
        "validation_seed": int(validation_seed),
        "tasks_per_cell": None if tasks_per_cell is None else int(tasks_per_cell),
        "candidate_reference_row_count": len(proposals),
        "task_count": len(rows),
        "candidate_manifest_path": str(candidate_manifest_path),
        "candidate_manifest_sha256": _sha256_file(candidate_manifest_path),
        "validation_manifest_sha256": _sha256_file(manifest_path),
        "generator_source_sha256": source_sha256(),
        "labels_in_manifest": False,
        "audit_metrics_in_manifest": False,
        "coverage_cell_count": len({
            (r["observation_stratum"], r["information_stratum"])
            for r in rows
        }),
        "fixed_corruption_seed_rule": "sha256(validation_task_id) modulo 2^63-1",
        "protocol": "fixed_validation_pool_no_selection",
    }
    (output_root / "resolved_config.json").write_text(
        json.dumps(resolved, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return resolved


def _validation_input_worker(
    row: Mapping[str, Any], sampler: FullSamplerConfig,
) -> InputTask:
    base = sample_full_task_config(
        generator_seed=int(row["generator_seed"]),
        task_index=int(row["schedule_task_index"]),
        split="qualification",
        observation_stratum=str(row["observation_stratum"]),
        sampler=sampler,
    )
    spec = replace(
        base,
        information_stratum=str(row["information_stratum"]),
        split="validation",
        task_index=int(row["validation_task_index"]),
        seed=int(row["validation_task_seed"]),
    )
    task = generate_v4_task(spec)
    if task.metadata["task_id"] != str(row["task_id"]):
        raise ValueError("validation task identity mismatch")
    return redact_v4_task(task, safe_metadata={
        "observation_stratum": row["observation_stratum"],
        "information_stratum": row["information_stratum"],
    })


def iter_validation_inputs(
    manifest_path: Path,
    *,
    sampler: FullSamplerConfig | None = None,
    cpu_workers: int = 16,
) -> Iterator[InputTask]:
    if cpu_workers < 1:
        raise ValueError("cpu_workers must be positive")
    sampler = sampler or FullSamplerConfig()
    sampler.validate()
    rows = [json.loads(line) for line in Path(manifest_path).open(encoding="utf-8")]
    with ProcessPoolExecutor(max_workers=cpu_workers) as executor:
        yield from executor.map(_validation_input_worker, rows, [sampler] * len(rows), chunksize=1)


def _fixed_corruption_generator(task_id: str) -> torch.Generator:
    seed = int(hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:16], 16) % (2**63 - 1)
    return torch.Generator(device="cpu").manual_seed(seed)


def evaluate_stage_a_validation(
    model: GlobalAE,
    manifest_path: Path,
    *,
    device: torch.device,
    corruption: StageACorruptionConfig | None = None,
    cpu_workers: int = 16,
    use_bf16: bool = True,
) -> Dict[str, Any]:
    """Evaluate fixed X-only validation reconstruction metrics."""

    corruption = corruption or StageACorruptionConfig()
    corruption.validate()
    model.eval()
    values: List[float] = []
    mask_values: List[float] = []
    nonfinite_count = 0
    task_count = 0
    cell_count = 0
    with torch.no_grad():
        for input_task in iter_validation_inputs(manifest_path, cpu_workers=cpu_workers):
            model_input = input_task.inference_payload()["model_input"]
            table = torch.as_tensor(model_input["features"], dtype=torch.float32, device=device)
            missing = torch.as_tensor(model_input["missing_mask"], dtype=torch.bool, device=device)
            feature_types = torch.as_tensor(model_input["feature_types"], dtype=torch.long, device=device)
            corrupted = corrupt_table(table, missing, _fixed_corruption_generator(input_task.task_id), corruption)
            autocast_enabled = bool(use_bf16 and device.type == "cuda")
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
                output = model(corrupted.values, corrupted.input_missing_mask, feature_types)
                loss, metrics = mixed_type_reconstruction_loss(
                    output, table, corrupted.input_missing_mask, feature_types,
                    target_missing_mask=corrupted.target_missing_mask,
                )
            task_count += 1
            cell_count += int(table.shape[0] * table.shape[1])
            loss_value = float(loss.detach().float().cpu())
            mask_value = float(metrics["mask"].detach().float().cpu())
            if not math.isfinite(loss_value) or not math.isfinite(mask_value):
                nonfinite_count += 1
            else:
                values.append(loss_value)
                mask_values.append(mask_value)
    if not values:
        raise FloatingPointError("validation produced no finite metrics")
    values_sorted = sorted(values)
    masks_sorted = sorted(mask_values)
    def quantile(sorted_values: List[float], fraction: float) -> float:
        position = (len(sorted_values) - 1) * fraction
        low = int(position); high = min(low + 1, len(sorted_values) - 1)
        return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (position - low)
    return {
        "schema_version": "dfcluster.generator_v4.stage_a.validation_metrics.v1",
        "manifest_path": str(manifest_path),
        "task_count": task_count,
        "cell_count": cell_count,
        "finite_task_count": len(values),
        "nonfinite_count": nonfinite_count,
        "reconstruction_mean": float(sum(values) / len(values)),
        "reconstruction_median": quantile(values_sorted, 0.5),
        "reconstruction_p10": quantile(values_sorted, 0.1),
        "reconstruction_p90": quantile(values_sorted, 0.9),
        "mask_mean": float(sum(mask_values) / len(mask_values)),
        "mask_median": quantile(masks_sorted, 0.5),
        "mask_p90": quantile(masks_sorted, 0.9),
        "labels_opened": False,
        "audit_metrics_opened": False,
        "performance_claim": False,
    }