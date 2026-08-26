"""Stage-B DDBM bridge and geometry objective for Generator V4.

The model input remains X-only. The clean geometry and synthetic labels are
read from the privileged target side only: labels participate exclusively in
the one pre-registered offline co-assignment auxiliary loss permitted by the
project rules. No audit metric, CLM, ARI, family ID, or difficulty value enters
the DDBM.
"""

from __future__ import annotations

from collections import defaultdict, deque
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

from .core import source_sha256
from .training import implementation_sha256, _gpu_query
from .input_loader import InputTask, PrivilegedTarget
from .models import DatasetContextDDBM, GlobalAE, centered_normalized_gram_torch


@dataclass(frozen=True)
class StageBConfig:
    diffusion_steps: int = 512
    beta_start: float = 1.0e-4
    beta_end: float = 0.02
    pair_count: int = 4096
    neighborhood_k: int = 10
    pair_temperature: float = 1.0
    bridge_weight: float = 1.0
    gram_weight: float = 1.0
    distance_weight: float = 0.5
    neighborhood_weight: float = 0.25
    pair_weight: float = 0.10
    gradient_clip_norm: float = 100.0
    row_permutation: bool = True

    def validate(self) -> None:
        if self.diffusion_steps != 512:
            raise ValueError("Stage-B diffusion_steps must be 512")
        if not 0.0 < self.beta_start < self.beta_end < 1.0:
            raise ValueError("invalid diffusion beta schedule")
        if self.pair_count <= 0 or self.neighborhood_k <= 0:
            raise ValueError("pair_count and neighborhood_k must be positive")
        if self.pair_temperature <= 0.0:
            raise ValueError("pair_temperature must be positive")
        expected = (1.0, 1.0, 0.5, 0.25, 0.10)
        actual = (
            self.bridge_weight,
            self.gram_weight,
            self.distance_weight,
            self.neighborhood_weight,
            self.pair_weight,
        )
        if actual != expected:
            raise ValueError("Stage-B loss weights are frozen by plan §22")
        if self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be positive")
        if self.row_permutation is not True:
            raise ValueError("Stage-B row permutation is frozen on")


def diffusion_schedule(config: StageBConfig, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    config.validate()
    betas = torch.linspace(
        config.beta_start,
        config.beta_end,
        config.diffusion_steps,
        dtype=torch.float32,
        device=device,
    )
    alphas = 1.0 - betas
    cumulative = torch.cumprod(alphas, dim=0)
    return torch.sqrt(cumulative), torch.sqrt(1.0 - cumulative)


def _normalized_factor(values: torch.Tensor) -> torch.Tensor:
    centered = values - values.mean(dim=0, keepdim=True)
    rms = torch.sqrt(centered.square().mean().clamp_min(1.0e-6))
    return centered / rms


def row_permutation(
    n_rows: int,
    *,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a deterministic row permutation and its inverse mapping."""

    if n_rows <= 0:
        raise ValueError("n_rows must be positive")
    permutation = torch.randperm(n_rows, generator=generator, device="cpu").to(device)
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(n_rows, device=device)
    return permutation, inverse


def apply_row_permutation(values: torch.Tensor, permutation: torch.Tensor) -> torch.Tensor:
    if values.ndim < 1 or permutation.shape != (values.shape[0],):
        raise ValueError("permutation must index the first dimension")
    return values[permutation]


def inverse_row_permutation(values: torch.Tensor, inverse: torch.Tensor) -> torch.Tensor:
    if values.ndim < 1 or inverse.shape != (values.shape[0],):
        raise ValueError("inverse permutation must index the first dimension")
    return values[inverse]


def _sample_indices(
    n_rows: int,
    count: int,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    first = torch.randint(n_rows, (count,), generator=generator, device="cpu")
    second = torch.randint(n_rows, (count,), generator=generator, device="cpu")
    same = second == first
    second[same] = (second[same] + 1) % n_rows
    return first.to(device), second.to(device)


def sampled_distance_loss(
    predicted: torch.Tensor,
    clean: torch.Tensor,
    *,
    pair_count: int,
    generator: torch.Generator,
) -> torch.Tensor:
    first, second = _sample_indices(predicted.shape[0], min(pair_count, max(1, predicted.shape[0] * 4)), generator, predicted.device)
    predicted_distance = (predicted[first] - predicted[second]).norm(dim=-1)
    clean_distance = (clean[first] - clean[second]).norm(dim=-1)
    predicted_scale = predicted_distance.detach().median().clamp_min(1.0e-6)
    clean_scale = clean_distance.detach().median().clamp_min(1.0e-6)
    return torch.mean(torch.abs(predicted_distance / predicted_scale - clean_distance / clean_scale))


def neighborhood_consistency_loss(
    predicted: torch.Tensor,
    clean: torch.Tensor,
    *,
    neighborhood_k: int,
    generator: torch.Generator,
) -> torch.Tensor:
    n_rows = predicted.shape[0]
    if n_rows <= 2:
        return predicted.sum() * 0.0
    k = min(neighborhood_k, n_rows - 1)
    clean_distances = torch.cdist(clean, clean)
    clean_distances.fill_diagonal_(float("inf"))
    positive = torch.topk(clean_distances, k=k, largest=False, dim=1).indices
    negative = torch.randint(n_rows, (n_rows, k), generator=generator, device="cpu").to(predicted.device)
    rows = torch.arange(n_rows, device=predicted.device).unsqueeze(1).expand(-1, k)
    negative = torch.where(negative == rows, (negative + 1) % n_rows, negative)
    positive_distance = (predicted[rows] - predicted[positive]).norm(dim=-1)
    negative_distance = (predicted[rows] - predicted[negative]).norm(dim=-1)
    return torch.relu(positive_distance - negative_distance + 0.05).mean()


def coassignment_pair_loss(
    predicted: torch.Tensor,
    labels: torch.Tensor,
    *,
    pair_count: int,
    temperature: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Balanced positive/negative pair loss using privileged Y only."""

    labels = labels.to(device=predicted.device).long()
    if labels.ndim != 1 or labels.shape[0] != predicted.shape[0]:
        raise ValueError("coassignment labels must have shape [N]")
    upper = torch.triu_indices(predicted.shape[0], predicted.shape[0], offset=1, device=predicted.device)
    pair_labels = labels[upper[0]] == labels[upper[1]]
    positive = torch.nonzero(pair_labels, as_tuple=False).flatten().to("cpu")
    negative = torch.nonzero(~pair_labels, as_tuple=False).flatten().to("cpu")
    if positive.numel() == 0 or negative.numel() == 0:
        raise ValueError("coassignment target must contain both positive and negative pairs")
    per_side = max(1, min(pair_count // 2, int(positive.numel()), int(negative.numel())))
    pos_choice = positive[torch.randperm(positive.numel(), generator=generator)[:per_side]]
    neg_choice = negative[torch.randperm(negative.numel(), generator=generator)[:per_side]]
    chosen = torch.cat((pos_choice, neg_choice), dim=0).to(predicted.device)
    target = torch.cat((torch.ones(per_side), torch.zeros(per_side)), dim=0).to(predicted.device)
    distances = (predicted[upper[0][chosen]] - predicted[upper[1][chosen]]).norm(dim=-1)
    logits = -distances / temperature
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, target)


def ddbm_stage_b_loss(
    predicted_noise: torch.Tensor,
    noisy_state: torch.Tensor,
    clean_latent: torch.Tensor,
    labels: torch.Tensor,
    *,
    timestep: int,
    schedule: tuple[torch.Tensor, torch.Tensor],
    config: StageBConfig,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute the exact five-term Stage-B objective."""

    config.validate()
    if predicted_noise.shape != clean_latent.shape or predicted_noise.shape[1] != 128:
        raise ValueError("Stage-B tensors must have shape [N,128]")
    sqrt_alpha, sqrt_one_minus = schedule
    if not 0 <= timestep < config.diffusion_steps:
        raise ValueError("timestep outside diffusion schedule")
    bridge_loss = torch.nn.functional.mse_loss(predicted_noise, (noisy_state - sqrt_alpha[timestep] * clean_latent) / sqrt_one_minus[timestep].clamp_min(1.0e-6))
    recovered = (noisy_state - sqrt_one_minus[timestep] * predicted_noise) / sqrt_alpha[timestep].clamp_min(1.0e-6)
    recovered = _normalized_factor(recovered)
    clean = _normalized_factor(clean_latent)
    gram_loss = torch.mean((centered_normalized_gram_torch(recovered) - centered_normalized_gram_torch(clean)) ** 2)
    distance_loss = sampled_distance_loss(recovered, clean, pair_count=config.pair_count, generator=generator)
    neighborhood_loss = neighborhood_consistency_loss(recovered, clean, neighborhood_k=config.neighborhood_k, generator=generator)
    pair_loss = coassignment_pair_loss(
        recovered,
        labels,
        pair_count=config.pair_count,
        temperature=config.pair_temperature,
        generator=generator,
    )
    total = (
        config.bridge_weight * bridge_loss
        + config.gram_weight * gram_loss
        + config.distance_weight * distance_loss
        + config.neighborhood_weight * neighborhood_loss
        + config.pair_weight * pair_loss
    )
    metrics = {
        "bridge_loss": bridge_loss.detach(),
        "gram_loss": gram_loss.detach(),
        "distance_loss": distance_loss.detach(),
        "neighborhood_loss": neighborhood_loss.detach(),
        "pair_loss": pair_loss.detach(),
        "total_loss": total.detach(),
    }
    return total, metrics


def freeze_ae_for_stage_b(ae: GlobalAE) -> None:
    ae.eval()
    for parameter in ae.parameters():
        parameter.requires_grad_(False)


def stage_b_train_step(
    ae: GlobalAE,
    ddbm: DatasetContextDDBM,
    values: torch.Tensor,
    missing_mask: torch.Tensor,
    feature_types: torch.Tensor,
    clean_latent: torch.Tensor,
    labels: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    *,
    config: StageBConfig | None = None,
    generator: torch.Generator | None = None,
    use_bf16: bool = False,
    zero_grad: bool = True,
    optimizer_step: bool = True,
    clip_grad: bool = True,
) -> Dict[str, float]:
    """One formal Stage-B update with frozen AE and privileged pair target."""

    config = config or StageBConfig()
    config.validate()
    if generator is None:
        generator = torch.Generator(device="cpu").manual_seed(20260824)
    if values.ndim != 2 or missing_mask.shape != values.shape or clean_latent.shape[0] != values.shape[0] or labels.shape[0] != values.shape[0]:
        raise ValueError("Stage-B input/target row shapes do not match")
    # Section 21.8: permute complete rows before both AE context and DDBM;
    # the inverse permutation is implicit when the row-aligned output is
    # evaluated. The permutation is not a label-bearing control.
    row_perm, _inverse_perm = row_permutation(
        values.shape[0], generator=generator, device=values.device
    )
    values = apply_row_permutation(values, row_perm)
    missing_mask = apply_row_permutation(missing_mask, row_perm)
    clean_latent = apply_row_permutation(clean_latent, row_perm)
    labels = apply_row_permutation(labels, row_perm)
    freeze_ae_for_stage_b(ae)
    ddbm.train()
    with torch.no_grad():
        observation, _, _ = ae.encode(values, missing_mask, feature_types)
    device = clean_latent.device
    sqrt_alpha, sqrt_one_minus = diffusion_schedule(config, device)
    timestep = int(torch.randint(config.diffusion_steps, (1,), generator=generator, device="cpu").item())
    noise = torch.randn(clean_latent.shape, generator=generator, device="cpu", dtype=clean_latent.dtype).to(device)
    noisy_state = sqrt_alpha[timestep] * clean_latent + sqrt_one_minus[timestep] * noise
    if zero_grad:
        optimizer.zero_grad(set_to_none=True)
    autocast_enabled = bool(use_bf16 and device.type == "cuda")
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
        predicted_noise = ddbm.predict_noise(noisy_state, observation, timestep)
        total, metrics = ddbm_stage_b_loss(
            predicted_noise,
            noisy_state,
            clean_latent,
            labels,
            timestep=timestep,
            schedule=(sqrt_alpha, sqrt_one_minus),
            config=config,
            generator=generator,
        )
    if not torch.isfinite(total):
        raise FloatingPointError("Stage-B total loss is non-finite")
    total.backward()
    if clip_grad:
        gradient_norm = torch.nn.utils.clip_grad_norm_(ddbm.parameters(), config.gradient_clip_norm)
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError("Stage-B gradient is non-finite")
    else:
        gradient_norm = torch.zeros((), device=total.device)
    if optimizer_step:
        optimizer.step()
    result = {key: float(value) for key, value in metrics.items()}
    result.update({"gradient_norm": float(gradient_norm), "timestep": float(timestep)})
    return result

@dataclass(frozen=True)
class StageBTrainingConfig:
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
    gradient_accumulation_cell_budget: int = 327680
    objective: StageBConfig = StageBConfig()

    def validate(self) -> None:
        if self.learning_rate <= 0.0 or self.final_learning_rate <= 0.0:
            raise ValueError("learning rates must be positive")
        if self.final_learning_rate > self.learning_rate or self.weight_decay < 0.0:
            raise ValueError("invalid optimizer configuration")
        if self.warmup_steps < 0 or self.task_exposure_target <= 0:
            raise ValueError("invalid Stage-B schedule")
        if self.physical_gpu_id in {0, 7}:
            raise ValueError("physical GPU 0 and 7 are forbidden")
        if self.cpu_workers < 16 or self.cpu_workers > 64:
            raise ValueError("Stage-B CPU workers must be within [16,64]")
        if self.gradient_accumulation_cell_budget != 327680:
            raise ValueError("Stage-B cell budget is frozen at the planned envelope")
        if tuple(self.checkpoint_milestones) != (500_000, 1_000_000, 2_000_000, 5_000_000):
            raise ValueError("Stage-B checkpoint milestones are frozen by plan")
        self.objective.validate()


def stage_b_config_sha256(config: StageBTrainingConfig) -> str:
    payload = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _stage_b_lr_factor(step: int, config: StageBTrainingConfig) -> float:
    if config.warmup_steps and step <= config.warmup_steps:
        return max(step, 1) / float(config.warmup_steps)
    remaining = max(config.task_exposure_target - config.warmup_steps, 1)
    progress = min(max((step - config.warmup_steps) / float(remaining), 0.0), 1.0)
    floor = config.final_learning_rate / config.learning_rate
    return floor + 0.5 * (1.0 - floor) * (1.0 + math.cos(math.pi * progress))


def _stage_b_atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(partial, path)


class StageBTrainer:
    """Formal Stage-B trainer with frozen AE and privileged pair target."""

    def __init__(
        self,
        ae: GlobalAE,
        ddbm: DatasetContextDDBM,
        config: StageBTrainingConfig,
        output_root: Path,
        *,
        ae_checkpoint_sha256: str,
        seed: int = 20260824,
    ) -> None:
        config.validate()
        if not ae_checkpoint_sha256:
            raise ValueError("Stage-B requires a frozen AE checkpoint SHA")
        self.ae = ae
        self.ddbm = ddbm
        self.config = config
        self.output_root = Path(output_root)
        if self.output_root.exists():
            raise FileExistsError(f"Stage-B output already exists: {self.output_root}")
        self.output_root.mkdir(parents=True, exist_ok=False)
        self.device = torch.device(config.device if config.device != "cuda" or torch.cuda.is_available() else "cpu")
        self.ae.to(self.device)
        self.ddbm.to(self.device)
        freeze_ae_for_stage_b(self.ae)
        self.optimizer = torch.optim.AdamW(
            self.ddbm.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda step: _stage_b_lr_factor(max(int(step), 1), config),
        )
        self.generator = torch.Generator(device="cpu").manual_seed(int(seed))
        self.step = 0
        self.task_exposure = 0
        self.optimizer_updates = 0
        self.pending_task_count = 0
        self.pending_cell_count = 0
        self.source_sha = source_sha256()
        self.implementation_sha = implementation_sha256()
        self.config_sha = stage_b_config_sha256(config)
        self.ae_checkpoint_sha256 = str(ae_checkpoint_sha256)
        self.history_path = self.output_root / "history.jsonl"
        self.loss_summary_path = self.output_root / "loss_summary.jsonl"
        self.ledger_path = self.output_root / "task_ledger.jsonl"
        self.gpu_ledger_path = self.output_root / "gpu_ledger.jsonl"
        self.status_path = self.output_root / "status.json"
        self._metric_count: Dict[str, int] = {}
        self._metric_sum: Dict[str, float] = {}
        self._metric_nonfinite: Dict[str, int] = {}
        self._metric_reservoir: Dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=4096))
        self._append_gpu_ledger("startup_before_model_transfer")
        _stage_b_atomic_json(self.output_root / "resolved_config.json", {
            "schema_version": "dfcluster.generator_v4.stage_b.v2",
            "stage": "B_ddbm",
            "config": asdict(config),
            "config_sha256": self.config_sha,
            "source_sha256": self.source_sha,
            "implementation_sha256": self.implementation_sha,
            "ae_checkpoint_sha256": self.ae_checkpoint_sha256,
            "seed": int(seed),
            "ae_frozen": True,
            "labels_used_only_for_pair_loss": True,
            "gradient_accumulation_cell_budget": config.gradient_accumulation_cell_budget,
            "gpu_ledger": str(self.gpu_ledger_path),
            "performance_claim": False,
        })
        _stage_b_atomic_json(self.status_path, {"status": "initialized", "step": 0, "task_exposure": 0})

    def _append_gpu_ledger(self, event: str, **extra: Any) -> None:
        record = _gpu_query(self.config.physical_gpu_id)
        record.update({
            "event": event,
            "device": str(self.device),
            "torch_memory_allocated": int(torch.cuda.memory_allocated(self.device)) if self.device.type == "cuda" else 0,
            "torch_memory_reserved": int(torch.cuda.memory_reserved(self.device)) if self.device.type == "cuda" else 0,
            "torch_peak_memory_allocated": int(torch.cuda.max_memory_allocated(self.device)) if self.device.type == "cuda" else 0,
            "torch_peak_memory_reserved": int(torch.cuda.max_memory_reserved(self.device)) if self.device.type == "cuda" else 0,
            "optimizer_updates": self.optimizer_updates,
            "pending_task_count": self.pending_task_count,
            "pending_cell_count": self.pending_cell_count,
        })
        record.update(extra)
        with self.gpu_ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    def _update_metric_summary(self, metrics: Mapping[str, float]) -> None:
        for name, value in metrics.items():
            if name in {"timestep", "step", "task_exposure", "learning_rate", "optimizer_update"}:
                continue
            value = float(value)
            self._metric_count[name] = self._metric_count.get(name, 0) + 1
            if math.isfinite(value):
                self._metric_sum[name] = self._metric_sum.get(name, 0.0) + value
                self._metric_reservoir[name].append(value)
            else:
                self._metric_nonfinite[name] = self._metric_nonfinite.get(name, 0) + 1

    def _loss_summary(self) -> Dict[str, Any]:
        summary: Dict[str, Any] = {}
        for name in sorted(self._metric_count):
            values = sorted(self._metric_reservoir[name])
            if not values:
                summary[name] = {
                    "count": self._metric_count[name],
                    "nonfinite_count": self._metric_nonfinite.get(name, 0),
                }
                continue
            def q(frac: float) -> float:
                pos = (len(values) - 1) * frac
                low = int(pos); high = min(low + 1, len(values) - 1)
                return values[low] + (values[high] - values[low]) * (pos - low)
            summary[name] = {
                "count": self._metric_count[name],
                "finite_count": len(values),
                "nonfinite_count": self._metric_nonfinite.get(name, 0),
                "mean": self._metric_sum.get(name, 0.0) / max(len(values), 1),
                "p10": q(0.10), "median": q(0.50), "p90": q(0.90),
            }
        return {
            "optimizer_updates": self.optimizer_updates,
            "pending_task_count": self.pending_task_count,
            "pending_cell_count": self.pending_cell_count,
            "metrics": summary,
        }

    def _flush_optimizer(self) -> float:
        if self.pending_task_count == 0:
            return 0.0
        scale = float(self.pending_task_count)
        for parameter in self.ddbm.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(scale)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.ddbm.parameters(), self.config.objective.gradient_clip_norm
        )
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError("Stage-B accumulated gradient is non-finite")
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer_updates += 1
        self.pending_task_count = 0
        self.pending_cell_count = 0
        return float(gradient_norm)

    def _checkpoint(self, metrics: Dict[str, Any], final: bool = False) -> Path:
        summary = self._loss_summary()
        name = "stage_b_final.pt" if final else f"stage_b_step_{self.step:07d}.pt"
        path = self.output_root / name
        if path.exists():
            raise FileExistsError(f"checkpoint already exists: {path}")
        partial = path.with_name(path.name + ".partial")
        torch.save({
            "schema_version": "dfcluster.generator_v4.stage_b.checkpoint.v2",
            "stage": "B_ddbm",
            "step": self.step,
            "task_exposure": self.task_exposure,
            "optimizer_updates": self.optimizer_updates,
            "ddbm_state_dict": self.ddbm.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "source_sha256": self.source_sha,
            "implementation_sha256": self.implementation_sha,
            "config_sha256": self.config_sha,
            "ae_checkpoint_sha256": self.ae_checkpoint_sha256,
            "loss_summary": summary,
            "metrics": metrics,
        }, partial)
        os.replace(partial, path)
        with self.loss_summary_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"step": self.step, "task_exposure": self.task_exposure, **summary}, ensure_ascii=False, sort_keys=True) + "\n")
        self._append_gpu_ledger("checkpoint", checkpoint=str(path), step=self.step, task_exposure=self.task_exposure)
        return path

    def step_one(self, input_task: InputTask, target: PrivilegedTarget, *, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        if not isinstance(input_task, InputTask) or not isinstance(target, PrivilegedTarget):
            raise TypeError("Stage-B requires strict InputTask plus PrivilegedTarget; privileged V4Task is forbidden")
        input_task.validate()
        target.validate()
        if input_task.task_id != target.task_id:
            raise ValueError("Stage-B input/target task_id mismatch")
        model_input = input_task.inference_payload()["model_input"]
        values = torch.as_tensor(model_input["features"], dtype=torch.float32, device=self.device)
        missing = torch.as_tensor(model_input["missing_mask"], dtype=torch.bool, device=self.device)
        feature_types = torch.as_tensor(model_input["feature_types"], dtype=torch.long, device=self.device)
        clean = torch.as_tensor(target.clean_latent, dtype=torch.float32, device=self.device)
        labels = torch.as_tensor(target.labels, dtype=torch.long, device=self.device)
        metrics = stage_b_train_step(
            self.ae, self.ddbm, values, missing, feature_types, clean, labels, self.optimizer,
            config=self.config.objective, generator=self.generator, use_bf16=self.config.use_bf16,
            zero_grad=self.pending_task_count == 0, optimizer_step=False, clip_grad=False,
        )
        self.pending_task_count += 1
        self.pending_cell_count += int(values.shape[0] * values.shape[1])
        self.step += 1
        self.task_exposure += 1
        optimizer_update = 0
        gradient_norm = 0.0
        if self.pending_cell_count >= self.config.gradient_accumulation_cell_budget:
            gradient_norm = self._flush_optimizer()
            optimizer_update = 1
        metrics["gradient_norm"] = gradient_norm
        metrics["optimizer_update"] = float(optimizer_update)
        metrics["step"] = float(self.step)
        metrics["task_exposure"] = float(self.task_exposure)
        metrics["learning_rate"] = float(self.optimizer.param_groups[0]["lr"])
        self._update_metric_summary(metrics)
        with self.history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics, sort_keys=True) + "\n")
        row = {
            "task_id": str(input_task.task_id),
            "step": self.step,
            "task_exposure": self.task_exposure,
            "n_samples": int(values.shape[0]),
            "n_features": int(values.shape[1]),
            "cell_count": int(values.shape[0] * values.shape[1]),
            "labels_used_only_for_pair_loss": True,
            "labels_in_model_input": False,
            "geometry_target_used": True,
            "row_permutation_applied": True,
            "inverse_row_alignment_required_for_evaluation": True,
            "optimizer_update": bool(optimizer_update),
        }
        if metadata:
            for key in ("observation_stratum", "information_stratum"):
                if key in metadata:
                    row[key] = metadata[key]
        with self.ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        if self.task_exposure in self.config.checkpoint_milestones:
            self._flush_optimizer()
            self._checkpoint(metrics)
        _stage_b_atomic_json(self.status_path, {
            "status": "running", "step": self.step, "task_exposure": self.task_exposure,
            "optimizer_updates": self.optimizer_updates,
            "pending_task_count": self.pending_task_count,
            "pending_cell_count": self.pending_cell_count,
            "last_metrics": metrics, "source_sha256": self.source_sha,
            "implementation_sha256": self.implementation_sha,
            "config_sha256": self.config_sha, "ae_checkpoint_sha256": self.ae_checkpoint_sha256,
        })
        return metrics

    def run(self, task_stream: Iterable[Tuple[InputTask, PrivilegedTarget]]) -> Dict[str, Any]:
        for item in task_stream:
            if self.task_exposure >= self.config.task_exposure_target:
                break
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError("Stage-B stream must yield (InputTask, PrivilegedTarget)")
            self.step_one(item[0], item[1])
        if self.task_exposure != self.config.task_exposure_target:
            _stage_b_atomic_json(self.status_path, {
                "status": "incomplete_compute", "step": self.step,
                "task_exposure": self.task_exposure,
                "required_task_exposure": self.config.task_exposure_target,
                "loss_summary": self._loss_summary(),
            })
            raise RuntimeError("Stage-B task stream ended before the frozen exposure target")
        self._flush_optimizer()
        metrics = {"step": self.step, "task_exposure": self.task_exposure, "optimizer_updates": self.optimizer_updates}
        checkpoint = self._checkpoint(metrics, final=True)
        report = {
            "schema_version": "dfcluster.generator_v4.stage_b.report.v2",
            "status": "completed", "stage": "B_ddbm",
            "task_exposure": self.task_exposure, "checkpoint": str(checkpoint),
            "source_sha256": self.source_sha, "implementation_sha256": self.implementation_sha,
            "config_sha256": self.config_sha, "ae_checkpoint_sha256": self.ae_checkpoint_sha256,
            "optimizer_updates": self.optimizer_updates,
            "loss_summary": self._loss_summary(),
            "gpu_ledger": str(self.gpu_ledger_path),
            "performance_claim": False,
        }
        _stage_b_atomic_json(self.output_root / "report.json", report)
        _stage_b_atomic_json(self.status_path, report)
        return report
