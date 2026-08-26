"""Formal Stage-A Global AE training infrastructure for Generator V4.

This module is a runnable trainer contract, not a reduced performance smoke.
It accepts only ``V4Task.training_payload()``-shaped objects, applies the
plan's masked-column/masked-cell/noise/scale/feature-dropout corruption, and
records task exposure and atomic checkpoints. It never receives labels, K,
CLM, ARI, generator-family controls, or target geometry.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import torch

from .core import source_sha256, validate_training_payload
from .models import GlobalAE, GlobalAEOutput, mixed_type_reconstruction_loss


@dataclass(frozen=True)
class StageACorruptionConfig:
    """Required Stage-A corruption design choices from plan §19.6."""

    masked_column_rate: float = 0.15
    masked_cell_rate: float = 0.10
    noise_std: float = 0.05
    scale_min: float = 0.80
    scale_max: float = 1.20
    feature_dropout_rate: float = 0.05

    def validate(self) -> None:
        for name in ("masked_column_rate", "masked_cell_rate", "feature_dropout_rate"):
            value = float(getattr(self, name))
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be in [0,1)")
        if self.noise_std < 0.0:
            raise ValueError("noise_std must be non-negative")
        if self.scale_min <= 0.0 or self.scale_max < self.scale_min:
            raise ValueError("invalid scale range")


@dataclass(frozen=True)
class StageAConfig:
    learning_rate: float = 1.0e-4
    weight_decay: float = 0.01
    warmup_steps: int = 10_000
    final_learning_rate: float = 1.0e-6
    gradient_clip_norm: float = 100.0
    task_exposure_target: int = 5_000_000
    checkpoint_milestones: Tuple[int, ...] = (500_000, 1_000_000, 2_000_000, 5_000_000)
    device: str = "cuda"
    use_bf16: bool = True
    physical_gpu_id: Optional[int] = None
    cpu_workers: int = 64
    validation_manifest_path: Optional[str] = None
    validation_cpu_workers: int = 16
    corruption: StageACorruptionConfig = StageACorruptionConfig()

    def validate(self) -> None:
        if self.learning_rate <= 0.0 or self.final_learning_rate <= 0.0:
            raise ValueError("learning rates must be positive")
        if self.final_learning_rate > self.learning_rate:
            raise ValueError("final learning rate cannot exceed initial rate")
        if self.weight_decay < 0.0 or self.warmup_steps < 0:
            raise ValueError("invalid optimizer schedule")
        if self.task_exposure_target <= 0:
            raise ValueError("task_exposure_target must be positive")
        if tuple(self.checkpoint_milestones) != (500_000, 1_000_000, 2_000_000, 5_000_000):
            raise ValueError("Stage-A checkpoint milestones are frozen by plan")
        if any(value <= 0 for value in self.checkpoint_milestones):
            raise ValueError("checkpoint milestones must be positive")
        if self.physical_gpu_id in {0, 7}:
            raise ValueError("physical GPU 0 and 7 are forbidden")
        if self.cpu_workers < 16 or self.cpu_workers > 64:
            raise ValueError("Stage-A CPU workers must be within [16,64]")
        if self.validation_cpu_workers < 1:
            raise ValueError("validation_cpu_workers must be positive")
        if self.validation_manifest_path is not None and not Path(self.validation_manifest_path).is_file():
            raise FileNotFoundError(self.validation_manifest_path)
        self.corruption.validate()


def config_sha256(config: StageAConfig) -> str:
    payload = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


TRAINING_SOURCE_COMPONENTS = ("models.py", "training.py", "stage_b.py", "storage.py", "replay.py", "online.py", "input_loader.py", "validation.py")

def implementation_sha256() -> str:
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in TRAINING_SOURCE_COMPONENTS:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"missing training source component: {path}")
        digest.update(name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class CorruptedTable:
    values: torch.Tensor
    input_missing_mask: torch.Tensor
    target_missing_mask: torch.Tensor
    corruption_mask: torch.Tensor


def _random(shape: Tuple[int, ...], generator: torch.Generator, device: torch.device) -> torch.Tensor:
    return torch.rand(shape, generator=generator, device="cpu", dtype=torch.float32).to(device)


def _normal(shape: Tuple[int, ...], generator: torch.Generator, device: torch.device) -> torch.Tensor:
    return torch.randn(shape, generator=generator, device="cpu", dtype=torch.float32).to(device)


def corrupt_table(
    values: torch.Tensor,
    missing_mask: torch.Tensor,
    generator: torch.Generator,
    config: StageACorruptionConfig | None = None,
) -> CorruptedTable:
    """Apply plan-prescribed denoising corruption without changing targets."""

    config = config or StageACorruptionConfig()
    config.validate()
    if values.ndim != 2 or missing_mask.shape != values.shape:
        raise ValueError("values and missing_mask must have matching [N,D] shapes")
    if not torch.isfinite(values).all():
        raise ValueError("values must be finite")
    native_missing = missing_mask.to(dtype=torch.bool)
    observed = ~native_missing
    n_rows, n_features = values.shape
    column_mask = _random((n_features,), generator, values.device) < config.masked_column_rate
    cell_mask = _random((n_rows, n_features), generator, values.device) < config.masked_cell_rate
    feature_dropout = _random((n_features,), generator, values.device) < config.feature_dropout_rate
    corruption = observed & (cell_mask | column_mask.unsqueeze(0) | feature_dropout.unsqueeze(0))

    scale = config.scale_min + (config.scale_max - config.scale_min) * _random((n_features,), generator, values.device)
    noisy = values * scale.unsqueeze(0)
    if config.noise_std:
        noisy = noisy + config.noise_std * _normal((n_rows, n_features), generator, values.device)
    corrupted = noisy.masked_fill(native_missing | corruption, 0.0)
    return CorruptedTable(
        values=corrupted,
        input_missing_mask=native_missing | corruption,
        target_missing_mask=native_missing,
        corruption_mask=corruption,
    )


def _payload(task: Any) -> tuple[str, Mapping[str, Any], Mapping[str, Any]]:
    # Stage A accepts only a redacted InputTask or a plain X-only mapping.
    # Reject privileged V4Task objects even if they expose inference_payload().
    if hasattr(task, "labels") or hasattr(task, "clean_latent") or hasattr(task, "observation_graph"):
        raise TypeError("Stage-A requires a strict redacted InputTask, not a privileged V4Task")
    if hasattr(task, "inference_payload"):
        task = task.inference_payload()
    if not isinstance(task, Mapping):
        raise TypeError("Stage-A task must be a training payload mapping")
    model_input = task.get("model_input")
    if not isinstance(model_input, Mapping):
        raise ValueError("training payload lacks model_input")
    model_payload = {"task_id": task.get("task_id", "unknown"), "model_input": model_input}
    validate_training_payload(model_payload)
    task_id = str(task.get("task_id", "unknown"))
    # Do not inspect geometry_target here. Stage A has no geometry target.
    return task_id, model_input, {}


def _to_batch(model_input: Mapping[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    required = ("features", "missing_mask", "feature_types")
    if any(key not in model_input for key in required):
        raise ValueError("model_input lacks required fields")
    values = torch.as_tensor(model_input["features"], dtype=torch.float32, device=device)
    missing = torch.as_tensor(model_input["missing_mask"], dtype=torch.bool, device=device)
    types = torch.as_tensor(model_input["feature_types"], dtype=torch.long, device=device)
    return values, missing, types


def _lr_factor(step: int, config: StageAConfig) -> float:
    if config.warmup_steps and step <= config.warmup_steps:
        return max(step, 1) / float(config.warmup_steps)
    remaining = max(config.task_exposure_target - config.warmup_steps, 1)
    progress = min(max((step - config.warmup_steps) / float(remaining), 0.0), 1.0)
    floor = config.final_learning_rate / config.learning_rate
    return floor + 0.5 * (1.0 - floor) * (1.0 + math.cos(math.pi * progress))


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(partial, path)


def _sha256_path(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _gpu_query(physical_gpu_id: Optional[int]) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "physical_gpu_id": physical_gpu_id,
    }
    if physical_gpu_id is None:
        result["status"] = "no_gpu_declared"
        return result
    try:
        query = subprocess.run(
            ["nvidia-smi", "-i", str(physical_gpu_id), "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=False, timeout=10,
        )
        apps = subprocess.run(
            ["nvidia-smi", "-i", str(physical_gpu_id), "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=False, timeout=10,
        )
        result.update({
            "status": "complete" if query.returncode == 0 else "query_failed",
            "gpu_query": query.stdout.strip(),
            "gpu_query_stderr": query.stderr.strip(),
            "compute_apps_query": apps.stdout.strip(),
            "compute_apps_query_stderr": apps.stderr.strip(),
        })
    except Exception as exc:
        result.update({"status": "query_error", "error": "%s: %s" % (type(exc).__name__, exc)})
    return result


class StageATrainer:
    """Formal Stage-A trainer; task streams are supplied by the corpus layer."""

    def __init__(
        self,
        model: GlobalAE,
        config: StageAConfig,
        output_root: Path,
        *,
        seed: int = 20260824,
    ) -> None:
        config.validate()
        self.model = model
        self.config = config
        self.output_root = Path(output_root)
        if self.output_root.exists():
            raise FileExistsError(f"Stage-A output already exists: {self.output_root}")
        self.output_root.mkdir(parents=True, exist_ok=False)
        self.device = torch.device(config.device if config.device != "cuda" or torch.cuda.is_available() else "cpu")
        self.gpu_ledger_path = self.output_root / "gpu_ledger.jsonl"
        self.validation_manifest_path = Path(config.validation_manifest_path) if config.validation_manifest_path else None
        self.validation_manifest_sha256 = _sha256_path(self.validation_manifest_path)
        self.validation_history_path = self.output_root / "validation_history.jsonl"
        self.last_validation_metrics: Optional[Dict[str, Any]] = None
        self._append_gpu_ledger("startup_before_model_transfer")
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda step: _lr_factor(max(int(step), 1), config)
        )
        self.generator = torch.Generator(device="cpu").manual_seed(int(seed))
        self.step = 0
        self.task_exposure = 0
        self.source_sha = source_sha256()
        self.implementation_sha = implementation_sha256()
        self.config_sha = config_sha256(config)
        self.history_path = self.output_root / "history.jsonl"
        self.ledger_path = self.output_root / "task_ledger.jsonl"
        self.status_path = self.output_root / "status.json"
        _atomic_json(self.output_root / "resolved_config.json", {
            "schema_version": "dfcluster.generator_v4.stage_a.v1",
            "stage": "A_global_ae",
            "config": asdict(config),
            "config_sha256": self.config_sha,
            "source_sha256": self.source_sha,
            "implementation_sha256": self.implementation_sha,
            "seed": int(seed),
            "physical_gpu_id": config.physical_gpu_id,
            "cpu_workers": config.cpu_workers,
            "validation_manifest_path": str(self.validation_manifest_path) if self.validation_manifest_path else None,
            "validation_manifest_sha256": self.validation_manifest_sha256,
            "labels_in_model_input": False,
            "geometry_target_used": False,
            "status": "initialized",
        })
        _atomic_json(self.status_path, {"status": "initialized", "step": 0, "task_exposure": 0})

    def _append_gpu_ledger(self, event: str, **extra: Any) -> None:
        record = _gpu_query(self.config.physical_gpu_id)
        record.update({
            "event": event,
            "device": str(self.device),
            "torch_memory_allocated": int(torch.cuda.memory_allocated(self.device)) if self.device.type == "cuda" else 0,
            "torch_memory_reserved": int(torch.cuda.memory_reserved(self.device)) if self.device.type == "cuda" else 0,
            "torch_peak_memory_allocated": int(torch.cuda.max_memory_allocated(self.device)) if self.device.type == "cuda" else 0,
            "torch_peak_memory_reserved": int(torch.cuda.max_memory_reserved(self.device)) if self.device.type == "cuda" else 0,
        })
        record.update(extra)
        with self.gpu_ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    def _run_validation(self) -> Optional[Dict[str, Any]]:
        if self.validation_manifest_path is None:
            return None
        from .validation import evaluate_stage_a_validation
        metrics = evaluate_stage_a_validation(
            self.model,
            self.validation_manifest_path,
            device=self.device,
            corruption=self.config.corruption,
            cpu_workers=self.config.validation_cpu_workers,
            use_bf16=self.config.use_bf16,
        )
        metrics.update({
            "step": self.step,
            "task_exposure": self.task_exposure,
            "validation_manifest_sha256": self.validation_manifest_sha256,
        })
        with self.validation_history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics, ensure_ascii=False, sort_keys=True) + "\n")
        self.last_validation_metrics = metrics
        return metrics

    def _checkpoint(self, metrics: Mapping[str, Any], *, final: bool = False) -> Path:
        validation_metrics = self._run_validation()
        checkpoint_metrics = dict(metrics)
        if validation_metrics is not None:
            checkpoint_metrics["validation"] = validation_metrics
        name = "stage_a_final.pt" if final else f"stage_a_step_{self.step:07d}.pt"
        path = self.output_root / name
        if path.exists():
            raise FileExistsError(f"checkpoint already exists: {path}")
        partial = path.with_name(path.name + ".partial")
        torch.save({
            "schema_version": "dfcluster.generator_v4.stage_a.checkpoint.v2",
            "stage": "A_global_ae",
            "step": self.step,
            "task_exposure": self.task_exposure,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "source_sha256": self.source_sha,
            "implementation_sha256": self.implementation_sha,
            "config_sha256": self.config_sha,
            "validation_manifest_sha256": self.validation_manifest_sha256,
            "metrics": checkpoint_metrics,
        }, partial)
        os.replace(partial, path)
        self._append_gpu_ledger("checkpoint", checkpoint=str(path), step=self.step, task_exposure=self.task_exposure)
        return path

    def step_one(self, task: Any, *, metadata: Optional[Mapping[str, Any]] = None) -> Dict[str, float]:
        if metadata is None and hasattr(task, "safe_ledger_metadata"):
            metadata = task.safe_ledger_metadata()
        task_id, model_input, _ = _payload(task)
        values, missing, types = _to_batch(model_input, self.device)
        self.model.train()
        corruption = corrupt_table(values, missing, self.generator, self.config.corruption)
        self.optimizer.zero_grad(set_to_none=True)
        autocast_enabled = self.config.use_bf16 and self.device.type == "cuda"
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
            output: GlobalAEOutput = self.model(
                corruption.values, corruption.input_missing_mask, types
            )
            loss, loss_metrics = mixed_type_reconstruction_loss(
                output,
                values,
                corruption.input_missing_mask,
                types,
                target_missing_mask=corruption.target_missing_mask,
            )
        if not torch.isfinite(loss):
            raise FloatingPointError("Stage-A loss is non-finite")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError("Stage-A gradient is non-finite")
        self.optimizer.step()
        self.step += 1
        self.task_exposure += 1
        self.scheduler.step()
        metrics = {
            "step": float(self.step),
            "task_exposure": float(self.task_exposure),
            "loss": float(loss.detach()),
            "reconstruction_loss": float(loss_metrics["reconstruction"]),
            "mask_loss": float(loss_metrics["mask"]),
            "gradient_norm": float(grad_norm),
            "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
        }
        with self.history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics, sort_keys=True) + "\n")
        ledger_row = {
            "task_id": task_id,
            "step": self.step,
            "task_exposure": self.task_exposure,
            "n_samples": int(values.shape[0]),
            "n_features": int(values.shape[1]),
            "cell_count": int(values.shape[0] * values.shape[1]),
            "labels_opened_by_trainer": False,
            "geometry_target_opened_by_stage_a": False,
        }
        if metadata:
            for key in ("observation_stratum", "information_stratum", "raw_difficulty_pool", "clm_tertile"):
                if key in metadata:
                    ledger_row[key] = metadata[key]
        with self.ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(ledger_row, sort_keys=True) + "\n")
        if self.task_exposure in self.config.checkpoint_milestones and self.task_exposure < self.config.task_exposure_target:
            self._checkpoint(metrics)
        _atomic_json(self.status_path, {
            "status": "running",
            "step": self.step,
            "task_exposure": self.task_exposure,
            "last_metrics": metrics,
            "source_sha256": self.source_sha,
            "implementation_sha256": self.implementation_sha,
            "config_sha256": self.config_sha,
        })
        return metrics

    def run(self, task_stream: Iterable[Any]) -> Dict[str, Any]:
        for task in task_stream:
            if self.task_exposure >= self.config.task_exposure_target:
                break
            self.step_one(task)
        if self.task_exposure != self.config.task_exposure_target:
            _atomic_json(self.status_path, {
                "status": "incomplete_compute",
                "step": self.step,
                "task_exposure": self.task_exposure,
                "required_task_exposure": self.config.task_exposure_target,
            })
            raise RuntimeError("Stage-A task stream ended before the frozen exposure target")
        final_metrics = {"step": self.step, "task_exposure": self.task_exposure}
        checkpoint = self._checkpoint(final_metrics, final=True)
        report = {
            "schema_version": "dfcluster.generator_v4.stage_a.report.v1",
            "status": "completed",
            "stage": "A_global_ae",
            "task_exposure": self.task_exposure,
            "checkpoint": str(checkpoint),
            "source_sha256": self.source_sha,
            "config_sha256": self.config_sha,
            "implementation_sha256": self.implementation_sha,
            "validation_manifest_sha256": self.validation_manifest_sha256,
            "last_validation_metrics": self.last_validation_metrics,
            "gpu_ledger": str(self.gpu_ledger_path),
            "performance_claim": False,
        }
        _atomic_json(self.output_root / "report.json", report)
        _atomic_json(self.status_path, report)
        return report