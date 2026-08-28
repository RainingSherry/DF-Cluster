"""Streaming, preregistered Phase 1/2 runner for the ZEUS geometry probe.

The runner deliberately keeps the original ZEUS generator, Transformer and
loss intact.  It changes only the objective between the paired arms and
streams training tasks so a 100k/300k update run does not retain all tasks in
memory.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import signal
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

try:  # package import
    from .core import (
        ProbeConfig,
        TaskRecord,
        build_paired_models,
        evaluate_arm,
        generate_task,
        set_seed,
        summarize,
        sample_tokens,
    )
    from .metrics import centered_linear_cka
except ImportError:  # direct ``python phase_runner.py`` execution
    ZEUS_ROOT = Path(__file__).resolve().parents[1]
    if str(ZEUS_ROOT) not in sys.path:
        sys.path.insert(0, str(ZEUS_ROOT))
    from geometry_probe.core import (  # type: ignore[no-redef]
        ProbeConfig,
        TaskRecord,
        build_paired_models,
        evaluate_arm,
        generate_task,
        set_seed,
        summarize,
        sample_tokens,
    )
    from geometry_probe.metrics import centered_linear_cka  # type: ignore[no-redef]

from zeus.model.model_utils import get_cosine_schedule_with_warmup
from zeus.utils import gmm_loss_with_regularizes


SCHEMA_VERSION = "zeus-geometry-phase-v1"
TRAIN_SEED_OFFSET = 1_000_000
EVAL_SEED_OFFSETS = {"gaussian": 10_000_000, "gaussian_transformed": 20_000_000}


@dataclass(frozen=True)
class PhaseConfig(ProbeConfig):
    """Configuration for one frozen phase.

    ``ProbeConfig.train_steps`` and ``ProbeConfig.eval_tasks`` are retained for
    backwards compatibility with the original small probe.  This runner uses
    ``total_updates`` and ``eval_task_count`` instead.
    """

    phase: str = "phase1"
    # Formal defaults follow ZEUS's published/configured synthetic prior.  The
    # small values in ProbeConfig remain available only to the legacy smoke CLI.
    num_gaussians: int = 10
    min_points: int = 50
    max_points: int = 500
    dim: int = 30
    embed_dim: int = 512
    n_head: int = 4
    hid_dim: int = 1024
    n_layers: int = 12
    total_updates: int = 100_000
    checkpoint_steps: tuple[int, ...] = (5_000, 20_000, 50_000, 75_000, 90_000, 100_000)
    seeds: tuple[int, ...] = (42,)
    eval_task_count: int = 100
    phase1_gate: str | None = None
    protocol_version: str = "zeus-target-only-v3"
    formal: bool = True
    legal_gpu_pool: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
    gpu_memory_budget_gib: float = 16.0
    concurrency: int = 1
    resume_from: str | None = None
    resume_step: int = 0


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, default=_json_default, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    paths = [
        Path(__file__),
        root / "geometry_probe" / "core.py",
        root / "geometry_probe" / "metrics.py",
        root / "zeus" / "datasets.py",
        root / "zeus" / "configs.py",
        root / "zeus" / "initialziation.py",
        root / "zeus" / "utils.py",
        root / "zeus" / "model" / "zeus.py",
        root / "zeus" / "model" / "layer.py",
    ]
    return {str(path.relative_to(root.parent)): _sha256_file(path) for path in paths}


def _git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _nvidia_snapshot() -> list[str]:
    if not torch.cuda.is_available():
        return []
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        return []


def _environment(config: PhaseConfig) -> dict[str, Any]:
    logical_index, physical_index = _device_indices(config.device)
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "pid": os.getpid(),
        "git_revision": _git_revision(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "device": config.device,
        "logical_device_index": logical_index,
        "physical_device_index": physical_index,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_smi": _nvidia_snapshot(),
        "legal_gpu_pool": list(config.legal_gpu_pool),
        "declared_gpu_memory_budget_gib": config.gpu_memory_budget_gib,
        "declared_concurrency": config.concurrency,
    }


def _device_indices(device: str) -> tuple[int | None, int | None]:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return None, None
    logical = int(device.rsplit(":", 1)[1]) if ":" in device else torch.cuda.current_device()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        entries = [entry.strip() for entry in visible.split(",") if entry.strip()]
        if logical >= len(entries):
            return logical, None
        selected = entries[logical]
        physical = int(selected) if selected.isdigit() else None
    else:
        physical = logical
    return logical, physical


def _validate_device(device: str, legal_gpu_pool: tuple[int, ...] = (1, 2, 3, 4, 5, 6)) -> None:
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but CUDA is unavailable")
        logical, physical = _device_indices(device)
        if logical is not None and logical >= torch.cuda.device_count():
            raise ValueError(f"CUDA logical device {logical} is unavailable")
        if physical in {0, 7}:
            raise ValueError("Physical GPU 0 and 7 are forbidden by the project rule")
        if physical not in legal_gpu_pool:
            raise ValueError(f"Physical GPU {physical} is outside legal_gpu_pool")


def _validate_config(config: PhaseConfig) -> None:
    if config.phase not in {"phase1", "phase2"}:
        raise ValueError("phase must be phase1 or phase2")
    if config.total_updates < 1:
        raise ValueError("total_updates must be positive")
    checkpoints = tuple(sorted(set(int(step) for step in config.checkpoint_steps)))
    if checkpoints != config.checkpoint_steps:
        raise ValueError("checkpoint_steps must be strictly increasing")
    if not checkpoints or checkpoints[-1] != config.total_updates:
        raise ValueError("the final checkpoint must equal total_updates")
    if any(step < 1 or step > config.total_updates for step in checkpoints):
        raise ValueError("checkpoint_steps must be within the update range")
    expected_seeds = 1 if config.phase == "phase1" else 3
    if len(config.seeds) != expected_seeds:
        raise ValueError(f"{config.phase} requires exactly {expected_seeds} seed(s)")
    if len(set(config.seeds)) != len(config.seeds):
        raise ValueError("seeds must be unique")
    if config.eval_task_count < 1:
        raise ValueError("eval_task_count must be positive")
    if config.gpu_memory_budget_gib <= 0:
        raise ValueError("gpu_memory_budget_gib must be positive")
    if config.concurrency < 1:
        raise ValueError("concurrency must be positive")
    if any(gpu in {0, 7} for gpu in config.legal_gpu_pool):
        raise ValueError("legal_gpu_pool contains forbidden physical GPU 0 or 7")
    if config.formal:
        expected = {
            "phase1": {
                "total_updates": 100_000,
                "checkpoint_steps": (5_000, 20_000, 50_000, 75_000, 90_000, 100_000),
                "eval_task_count": 100,
            },
            "phase2": {
                "total_updates": 300_000,
                "checkpoint_steps": (50_000, 100_000, 200_000, 300_000),
                "eval_task_count": 200,
            },
        }[config.phase]
        for name, expected_value in expected.items():
            if getattr(config, name) != expected_value:
                raise ValueError(f"formal {config.phase} requires {name}={expected_value}")
    if config.phase == "phase2" and not config.phase1_gate:
        raise ValueError("phase2 requires --phase1-gate with proceed=true")
    if config.resume_from is None and config.resume_step:
        raise ValueError("resume_step requires --resume-from")
    if config.resume_from is not None:
        if config.resume_step < 1 or config.resume_step >= config.total_updates:
            raise ValueError("resume_step must be between 1 and total_updates - 1")
        if config.resume_step not in checkpoints:
            raise ValueError("resume_step must be one of checkpoint_steps")
    if config.knn_k < 1:
        raise ValueError("knn_k must be positive")
    _validate_device(config.device, config.legal_gpu_pool)


def _task_digest(digest: hashlib._Hash, task: TaskRecord) -> None:
    """Hash exact task content without retaining the task stream."""
    digest.update(f"{task.index}:{task.seed}:{task.generator_mode}".encode("utf-8"))
    for tensor in (task.x_obs, task.labels, task.x_ref, task.probabilities):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())


def _materialize_eval(config: PhaseConfig) -> tuple[dict[str, list[TaskRecord]], str]:
    tasks: dict[str, list[TaskRecord]] = {}
    digest = hashlib.sha256()
    for mode in ("gaussian", "gaussian_transformed"):
        requested = "gaussian" if mode == "gaussian" else "transformed"
        seed_base = EVAL_SEED_OFFSETS[mode]
        mode_tasks = [
            generate_task(
                index=index,
                seed=seed_base + index,
                requested_mode=requested,
                config=config,
            )
            for index in range(config.eval_task_count)
        ]
        for task in mode_tasks:
            _task_digest(digest, task)
        tasks[mode] = mode_tasks
    return tasks, digest.hexdigest()


def _write_eval_manifest(output: Path, tasks: dict[str, list[TaskRecord]], digest: str) -> None:
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "sha256": digest,
        "known_k": True,
        "clustering_stage": "evaluation_only",
        "tasks": {
            mode: [task.audit_metadata() for task in mode_tasks]
            for mode, mode_tasks in tasks.items()
        },
    }
    _write_json(output / "evaluation_manifest.json", manifest)


def _read_phase1_gate(path: str) -> dict[str, Any]:
    gate_path = Path(path)
    if not gate_path.is_file():
        raise FileNotFoundError(f"Phase-1 gate does not exist: {gate_path}")
    with gate_path.open("r", encoding="utf-8") as handle:
        gate = json.load(handle)
    if gate.get("proceed") is not True:
        raise ValueError("Phase-1 gate must contain proceed=true")
    report_path = gate.get("phase1_report")
    if not report_path:
        raise ValueError("Phase-1 gate must identify phase1_report")
    report = Path(report_path)
    if not report.is_absolute():
        report = gate_path.parent / report
    if not report.is_file():
        raise FileNotFoundError(f"Referenced Phase-1 report does not exist: {report}")
    with report.open("r", encoding="utf-8") as handle:
        phase1_report = json.load(handle)
    if phase1_report.get("status") != "complete":
        raise ValueError("Phase-2 gate can reference only a complete Phase-1 report")
    if phase1_report.get("phase") != "phase1":
        raise ValueError("Phase-2 gate must reference a Phase-1 report")
    config = phase1_report.get("config", {})
    if (
        config.get("formal") is not True
        or config.get("total_updates") != 100_000
        or tuple(config.get("checkpoint_steps", ())) != (5_000, 20_000, 50_000, 75_000, 90_000, 100_000)
        or config.get("eval_task_count") != 100
        or tuple(config.get("seeds", ())) != (42,)
    ):
        raise ValueError("Phase-1 report does not match the frozen formal protocol")
    expected_arms = {"zeus", "geometry"}
    seed_result = phase1_report.get("arms", {}).get("42", {})
    if set(seed_result) != expected_arms:
        raise ValueError("Phase-1 report must contain both ZEUS and geometry arms for seed 42")
    expected_steps = (5_000, 20_000, 50_000, 75_000, 90_000, 100_000)
    report_root = report.parent
    for arm in sorted(expected_arms):
        checkpoints = seed_result[arm].get("checkpoints", [])
        if tuple(item.get("step") for item in checkpoints) != expected_steps:
            raise ValueError(f"Phase-1 report has incomplete {arm} checkpoints")
        for item in checkpoints:
            for key in ("checkpoint", "metrics"):
                artifact = report_root / item[key]
                if not artifact.is_file() or artifact.stat().st_size == 0:
                    raise ValueError(f"Phase-1 report references missing {key}: {artifact}")
    evaluation = phase1_report.get("evaluation_manifest", {})
    if evaluation.get("task_count_per_mode") != 100:
        raise ValueError("Phase-1 report has the wrong evaluation task count")
    return {
        "path": str(gate_path.resolve()),
        "sha256": _sha256_file(gate_path),
        "content": gate,
        "phase1_report": str(report.resolve()),
        "phase1_report_sha256": _sha256_file(report),
    }


def _aggregate_seed_summaries(report: dict[str, Any]) -> dict[str, Any]:
    """Aggregate checkpoint means/stds across independent training seeds."""
    aggregate: dict[str, Any] = {}
    for seed_result in report.get("arms", {}).values():
        for arm, arm_result in seed_result.items():
            for checkpoint in arm_result.get("checkpoints", []):
                step_key = str(checkpoint["step"])
                destination = aggregate.setdefault(arm, {}).setdefault(step_key, {})
                for mode, summary in checkpoint.get("summary", {}).items():
                    mode_destination = destination.setdefault(mode, {})
                    for metric, stats in summary.items():
                        mode_destination.setdefault(metric, {}).setdefault("seed_means", []).append(
                            float(stats["mean"])
                        )
    for arm_steps in aggregate.values():
        for step_summaries in arm_steps.values():
            for mode_summaries in step_summaries.values():
                for metric_stats in mode_summaries.values():
                    values = metric_stats.pop("seed_means")
                    metric_stats["mean"] = float(np.mean(values))
                    metric_stats["std"] = float(np.std(values, ddof=0))
                    metric_stats["n_seeds"] = len(values)
    return aggregate


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    seed: int,
    arm: str,
    step: int,
    phase: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "phase": phase,
        "arm": arm,
        "seed": seed,
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _load_resume_checkpoint(
    source: Path,
    output: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    seed: int,
    arm: str,
    step: int,
    phase: str,
) -> list[dict[str, Any]]:
    """Load one checkpoint and preserve its completed artifacts in ``output``."""
    source_checkpoint = source / "runs" / f"seed_{seed}" / arm / "checkpoints" / f"checkpoint_{step}.pt"
    source_metrics = source / "runs" / f"seed_{seed}" / arm / "metrics" / f"checkpoint_{step}.json"
    if not source_checkpoint.is_file() or not source_metrics.is_file():
        raise FileNotFoundError(f"resume checkpoint/metrics missing for {arm} at step {step}")
    checkpoint = torch.load(source_checkpoint, map_location=next(model.parameters()).device)
    if checkpoint.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"resume checkpoint schema mismatch for {arm}")
    if checkpoint.get("phase") != phase or checkpoint.get("arm") != arm:
        raise ValueError(f"resume checkpoint metadata mismatch for {arm}")
    if int(checkpoint.get("seed", -1)) != seed or int(checkpoint.get("step", -1)) != step:
        raise ValueError(f"resume checkpoint seed/step mismatch for {arm}")
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])

    destination_checkpoint = output / "runs" / f"seed_{seed}" / arm / "checkpoints" / source_checkpoint.name
    destination_metrics = output / "runs" / f"seed_{seed}" / arm / "metrics" / source_metrics.name
    destination_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    destination_metrics.parent.mkdir(parents=True, exist_ok=True)
    if not destination_checkpoint.exists():
        try:
            destination_checkpoint.hardlink_to(source_checkpoint)
        except OSError:
            shutil.copy2(source_checkpoint, destination_checkpoint)
    if not destination_metrics.exists():
        try:
            destination_metrics.hardlink_to(source_metrics)
        except OSError:
            shutil.copy2(source_metrics, destination_metrics)
    with source_metrics.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    return [{
        "step": step,
        "checkpoint": str(destination_checkpoint.relative_to(output)),
        "metrics": str(destination_metrics.relative_to(output)),
        "summary": metrics.get("summary", {}),
        "resumed_from": str(source_checkpoint.resolve()),
    }]


def _copy_history_prefix(source: Path, output: Path, seed: int, arm: str, step: int) -> Path:
    """Copy only the verified prefix of a source history into the resumed run."""
    source_history = source / "runs" / f"seed_{seed}" / arm / "training_history.jsonl"
    destination_history = output / "runs" / f"seed_{seed}" / arm / "training_history.jsonl"
    destination_history.parent.mkdir(parents=True, exist_ok=True)
    with destination_history.open("w", encoding="utf-8") as destination:
        if source_history.is_file():
            with source_history.open("r", encoding="utf-8") as source_handle:
                for line in source_handle:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        break
                    if int(record.get("update", 0)) > step:
                        break
                    destination.write(line)
    return destination_history


def _train_arm_stream(
    model: torch.nn.Module,
    config: PhaseConfig,
    seed: int,
    arm: str,
    evaluation_tasks: dict[str, list[TaskRecord]],
    output: Path,
) -> dict[str, Any]:
    objective = arm
    arm_dir = output / "runs" / f"seed_{seed}" / arm
    history_path = arm_dir / "training_history.jsonl"
    arm_dir.mkdir(parents=True, exist_ok=True)
    # Both arms see the same stochastic stream.  The objective is the only
    # intended difference, including when a non-zero dropout is configured.
    set_seed(seed + 30_000)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = get_cosine_schedule_with_warmup(optimizer, 0, config.total_updates)
    task_digest = hashlib.sha256()
    checkpoint_records: list[dict[str, Any]] = []
    checkpoint_set = set(config.checkpoint_steps)
    model.train()

    with history_path.open("w", encoding="utf-8") as history:
        for update in range(1, config.total_updates + 1):
            task_seed = seed + TRAIN_SEED_OFFSET + update - 1
            task = generate_task(update - 1, task_seed, config.train_mode, config)
            _task_digest(task_digest, task)
            x_obs = task.x_obs.to(config.device)
            labels = task.labels.to(config.device)
            x_ref = task.x_ref.to(config.device)
            optimizer.zero_grad(set_to_none=True)
            representation = sample_tokens(model, x_obs, config.num_gaussians)
            if objective == "zeus":
                loss = gmm_loss_with_regularizes(
                    representation.unsqueeze(1), labels, probs=task.probabilities.to(config.device)
                )
            elif objective == "geometry":
                loss = 1.0 - centered_linear_cka(representation, x_ref)
            else:  # pragma: no cover - arm list is fixed by this module
                raise ValueError(f"Unknown arm: {arm}")
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite {arm} loss at update {update}")
            loss.backward()
            optimizer.step()
            scheduler.step()
            history.write(json.dumps({"update": update, "loss": float(loss.detach().cpu())}) + "\n")
            if update in checkpoint_set:
                checkpoint_path = arm_dir / "checkpoints" / f"checkpoint_{update}.pt"
                _save_checkpoint(
                    checkpoint_path, model, optimizer, scheduler, seed, arm, update, config.phase
                )
                per_mode: dict[str, list[dict[str, Any]]] = {}
                summaries: dict[str, dict[str, dict[str, float]]] = {}
                for mode, mode_tasks in evaluation_tasks.items():
                    records = evaluate_arm(model, mode_tasks, config)
                    per_mode[mode] = records
                    summaries[mode] = summarize(records)
                metrics_path = arm_dir / "metrics" / f"checkpoint_{update}.json"
                _write_json(
                    metrics_path,
                    {
                        "schema_version": SCHEMA_VERSION,
                        "seed": seed,
                        "arm": arm,
                        "step": update,
                        "known_k": True,
                        "per_mode": per_mode,
                        "summary": summaries,
                    },
                )
                checkpoint_records.append(
                    {
                        "step": update,
                        "checkpoint": str(checkpoint_path.relative_to(output)),
                        "metrics": str(metrics_path.relative_to(output)),
                        "summary": summaries,
                    }
                )
                model.train()
        history.flush()

    return {
        "training_task_count": config.total_updates,
        "training_task_seed_first": seed + TRAIN_SEED_OFFSET,
        "training_task_seed_last": seed + TRAIN_SEED_OFFSET + config.total_updates - 1,
        "training_task_sha256": task_digest.hexdigest(),
        "history": str(history_path.relative_to(output)),
        "checkpoints": checkpoint_records,
        "peak_memory_allocated": (
            int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
        ),
        "peak_memory_reserved": (
            int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else 0
        ),
    }


def _capture_rng_state() -> tuple[torch.Tensor, list[torch.Tensor] | None]:
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    return torch.random.get_rng_state(), cuda_state


def _restore_rng_state(state: tuple[torch.Tensor, list[torch.Tensor] | None]) -> None:
    cpu_state, cuda_state = state
    torch.random.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)


def _train_paired_stream(
    models: dict[str, torch.nn.Module],
    config: PhaseConfig,
    seed: int,
    evaluation_tasks: dict[str, list[TaskRecord]],
    output: Path,
    resume_from: Path | None = None,
    resume_step: int = 0,
) -> dict[str, Any]:
    """Train both arms from one generated task per update.

    Interleaving avoids generating the same expensive ZEUS task twice while
    preserving a strict paired stream.  RNG is restored before each arm's
    forward pass so stochastic model layers observe identical random draws.
    """
    arm_state: dict[str, dict[str, Any]] = {}
    optimizers: dict[str, torch.optim.Optimizer] = {}
    schedulers: dict[str, Any] = {}
    histories: dict[str, Any] = {}
    task_digest = hashlib.sha256()
    checkpoint_set = set(config.checkpoint_steps)
    try:
        set_seed(seed + 30_000)
        for arm in ("zeus", "geometry"):
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            optimizers[arm] = torch.optim.AdamW(
                models[arm].parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
            )
            schedulers[arm] = get_cosine_schedule_with_warmup(
                optimizers[arm], 0, config.total_updates
            )
            arm_dir = output / "runs" / f"seed_{seed}" / arm
            arm_dir.mkdir(parents=True, exist_ok=True)
            prior_checkpoints: list[dict[str, Any]] = []
            history_path = arm_dir / "training_history.jsonl"
            if resume_from is not None:
                prior_checkpoints = _load_resume_checkpoint(
                    resume_from,
                    output,
                    models[arm],
                    optimizers[arm],
                    schedulers[arm],
                    seed,
                    arm,
                    resume_step,
                    config.phase,
                )
                _copy_history_prefix(resume_from, output, seed, arm, resume_step)
            histories[arm] = history_path.open("a" if resume_from is not None else "w", encoding="utf-8")
            arm_state[arm] = {
                "checkpoints": prior_checkpoints,
                "peak_memory_allocated": 0,
                "peak_memory_reserved": 0,
            }
            models[arm].train()

        for update in range(1, config.total_updates + 1):
            task_seed = seed + TRAIN_SEED_OFFSET + update - 1
            task = generate_task(update - 1, task_seed, config.train_mode, config)
            _task_digest(task_digest, task)
            if resume_from is not None and update <= resume_step:
                continue
            x_obs = task.x_obs.to(config.device)
            labels = task.labels.to(config.device)
            x_ref = task.x_ref.to(config.device)
            # Capture the post-generator RNG state once and replay it for both arms.
            forward_rng = _capture_rng_state()
            for arm in ("zeus", "geometry"):
                _restore_rng_state(forward_rng)
                optimizer = optimizers[arm]
                optimizer.zero_grad(set_to_none=True)
                representation = sample_tokens(models[arm], x_obs, config.num_gaussians)
                if arm == "zeus":
                    loss = gmm_loss_with_regularizes(
                        representation.unsqueeze(1), labels, probs=task.probabilities.to(config.device)
                    )
                else:
                    loss = 1.0 - centered_linear_cka(representation, x_ref)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite {arm} loss at update {update}")
                loss.backward()
                optimizer.step()
                schedulers[arm].step()
                histories[arm].write(
                    json.dumps({"update": update, "loss": float(loss.detach().cpu())}) + "\n"
                )
                if torch.cuda.is_available():
                    arm_state[arm]["peak_memory_allocated"] = max(
                        arm_state[arm]["peak_memory_allocated"], int(torch.cuda.max_memory_allocated())
                    )
                    arm_state[arm]["peak_memory_reserved"] = max(
                        arm_state[arm]["peak_memory_reserved"], int(torch.cuda.max_memory_reserved())
                    )

            if update in checkpoint_set:
                for arm in ("zeus", "geometry"):
                    arm_dir = output / "runs" / f"seed_{seed}" / arm
                    checkpoint_path = arm_dir / "checkpoints" / f"checkpoint_{update}.pt"
                    _save_checkpoint(
                        checkpoint_path,
                        models[arm],
                        optimizers[arm],
                        schedulers[arm],
                        seed,
                        arm,
                        update,
                        config.phase,
                    )
                    per_mode: dict[str, list[dict[str, Any]]] = {}
                    summaries: dict[str, dict[str, dict[str, float]]] = {}
                    for mode, mode_tasks in evaluation_tasks.items():
                        records = evaluate_arm(models[arm], mode_tasks, config)
                        per_mode[mode] = records
                        summaries[mode] = summarize(records)
                    metrics_path = arm_dir / "metrics" / f"checkpoint_{update}.json"
                    _write_json(
                        metrics_path,
                        {
                            "schema_version": SCHEMA_VERSION,
                            "seed": seed,
                            "arm": arm,
                            "step": update,
                            "known_k": True,
                            "per_mode": per_mode,
                            "summary": summaries,
                        },
                    )
                    arm_state[arm]["checkpoints"].append(
                        {
                            "step": update,
                            "checkpoint": str(checkpoint_path.relative_to(output)),
                            "metrics": str(metrics_path.relative_to(output)),
                            "summary": summaries,
                        }
                    )
                    models[arm].train()
        for history in histories.values():
            history.flush()
    finally:
        for history in histories.values():
            history.close()

    return {
        arm: {
            "training_task_count": config.total_updates,
            "training_task_seed_first": seed + TRAIN_SEED_OFFSET,
            "training_task_seed_last": seed + TRAIN_SEED_OFFSET + config.total_updates - 1,
            "training_task_sha256": task_digest.hexdigest(),
            "history": str(
                (output / "runs" / f"seed_{seed}" / arm / "training_history.jsonl").relative_to(output)
            ),
            "checkpoints": arm_state[arm]["checkpoints"],
            "peak_memory_allocated": arm_state[arm]["peak_memory_allocated"],
            "peak_memory_reserved": arm_state[arm]["peak_memory_reserved"],
        }
        for arm in ("zeus", "geometry")
    }


def run_phase(config: PhaseConfig, output: Path) -> dict[str, Any]:
    """Run one phase and persist a complete or ``incomplete_compute`` report."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    errors: list[dict[str, str]] = []
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "phase": config.phase,
        "protocol_version": config.protocol_version,
        "status": "incomplete_compute",
        "config": asdict(config),
        "protocol": {
            "comparison": "same ZEUS generator, Transformer initialization, task seed stream, optimizer, scheduler, and update count; objective only differs",
            "zeus_objective": "gmm_loss_with_regularizes",
            "geometry_objective": "1 - centered_linear_cka(H, X_ref)",
            "training_data_source": "methods/zeus/zeus/datasets.py::dataset_generator(mode=\"random\")",
            "reference_source": "dataset_generator X_ref (pre-transformation reference)",
            "known_k_source": "synthetic evaluation labels only; evaluation clustering stage",
            "labels_in_training": "only the pre-registered ZEUS objective; never geometry objective/control/model selection",
        },
        "source_sha256": {},
        "environment": {},
        "errors": errors,
        "arms": {},
    }
    def _persist_interrupt(signum: int, _frame: Any) -> None:
        errors.append({"type": "SignalInterrupt", "message": f"received signal {signum}"})
        raise KeyboardInterrupt

    previous_handlers = (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM))
    signal.signal(signal.SIGINT, _persist_interrupt)
    signal.signal(signal.SIGTERM, _persist_interrupt)
    try:
        report["source_sha256"] = _source_hashes()
        report["environment"] = _environment(config)
        _validate_config(config)
        if config.phase == "phase2":
            report["phase1_gate"] = _read_phase1_gate(config.phase1_gate or "")

        evaluation_tasks, evaluation_digest = _materialize_eval(config)
        _write_eval_manifest(output, evaluation_tasks, evaluation_digest)
        report["evaluation_manifest"] = {
            "path": "evaluation_manifest.json",
            "sha256": evaluation_digest,
            "task_count_per_mode": config.eval_task_count,
            "modes": ["gaussian", "gaussian_transformed"],
            "known_k": True,
        }

        for seed in config.seeds:
            seed_config = replace(config, seed=seed)
            models = build_paired_models(seed_config)
            seed_result = _train_paired_stream(
                models,
                seed_config,
                seed,
                evaluation_tasks,
                output,
                resume_from=Path(config.resume_from) if config.resume_from else None,
                resume_step=config.resume_step,
            )
            hashes = {seed_result[arm]["training_task_sha256"] for arm in ("zeus", "geometry")}
            if len(hashes) != 1:
                raise RuntimeError(f"paired training task streams differ for seed {seed}")
            report["arms"][str(seed)] = seed_result

        report["status"] = "complete"
        report["aggregate_summary"] = _aggregate_seed_summaries(report)
        if config.resume_from:
            report["resume_provenance"] = {
                "source_output": str(Path(config.resume_from).resolve()),
                "source_step": config.resume_step,
                "mode": "checkpoint_resume",
            }
    except BaseException as exc:  # persist evidence before surfacing the failure
        errors.append({"type": type(exc).__name__, "message": str(exc)})
    finally:
        signal.signal(signal.SIGINT, previous_handlers[0])
        signal.signal(signal.SIGTERM, previous_handlers[1])
    _write_json(output / "resolved_config.json", asdict(config))
    _write_json(output / "status.json", {"schema_version": SCHEMA_VERSION, "status": report["status"], "errors": errors})
    _write_json(output / "errors.json", {"schema_version": SCHEMA_VERSION, "errors": errors})
    _write_json(output / "report.json", report)
    return report


def _parse_steps(value: str) -> tuple[int, ...]:
    return tuple(int(piece.strip()) for piece in value.split(",") if piece.strip())


def _parse_seeds(value: str) -> tuple[int, ...]:
    return tuple(int(piece.strip()) for piece in value.split(",") if piece.strip())


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the formal ZEUS geometry target-only phase.")
    parser.add_argument("--phase", choices=("phase1", "phase2"), default="phase1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--total-updates", type=int, default=None)
    parser.add_argument("--checkpoint-steps", default=None)
    parser.add_argument("--seeds", default=None, help="Comma-separated seed list")
    parser.add_argument("--eval-tasks", type=int, default=None)
    parser.add_argument("--phase1-gate", default=None)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--resume-step", type=int, default=0)
    for name, default in (
        ("train-mode", "random"),
        ("num-gaussians", 10),
        ("min-points", 50),
        ("max-points", 500),
        ("dim", 30),
        ("min-distance", 0.5),
        ("eigenvalue-p1", 0.005),
        ("eigenvalue-p2", 0.05),
        ("start-distance", 1.0),
        ("max-blocks", 3),
        ("num-categorical", 5),
        ("max-categories", 5),
        ("categorical-chance", 0.3),
        ("embed-dim", 512),
        ("n-head", 4),
        ("hid-dim", 1024),
        ("n-layers", 12),
        ("dropout", 0.0),
        ("learning-rate", 3e-5),
        ("weight-decay", 1e-5),
        ("knn-k", 5),
        ("device", "cpu"),
    ):
        parser.add_argument(f"--{name}", default=default, type=type(default))
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    phase2 = args.phase == "phase2"
    config = PhaseConfig(
        phase=args.phase,
        total_updates=args.total_updates or (300_000 if phase2 else 100_000),
        checkpoint_steps=args.checkpoint_steps
        and _parse_steps(args.checkpoint_steps)
        or ((50_000, 100_000, 200_000, 300_000) if phase2 else (5_000, 20_000, 50_000, 75_000, 90_000, 100_000)),
        seeds=args.seeds and _parse_seeds(args.seeds) or ((42, 43, 44) if phase2 else (42,)),
        eval_task_count=args.eval_tasks or (200 if phase2 else 100),
        phase1_gate=args.phase1_gate,
        resume_from=str(args.resume_from) if args.resume_from else None,
        resume_step=args.resume_step,
        train_mode=args.train_mode,
        num_gaussians=args.num_gaussians,
        min_points=args.min_points,
        max_points=args.max_points,
        dim=args.dim,
        min_distance=args.min_distance,
        eigenvalue_p1=args.eigenvalue_p1,
        eigenvalue_p2=args.eigenvalue_p2,
        start_distance=args.start_distance,
        max_blocks=args.max_blocks,
        num_categorical=args.num_categorical,
        max_categories=args.max_categories,
        categorical_chance=args.categorical_chance,
        embed_dim=args.embed_dim,
        n_head=args.n_head,
        hid_dim=args.hid_dim,
        n_layers=args.n_layers,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        knn_k=args.knn_k,
        device=args.device,
    )
    report = run_phase(config, args.output)
    print(json.dumps({"phase": config.phase, "status": report["status"], "output": str(args.output)}))
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
