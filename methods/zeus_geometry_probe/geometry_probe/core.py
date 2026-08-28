"""Paired target-only training and evaluation for ZEUS."""

from __future__ import annotations

import copy
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from zeus.configs import GMMConfig
from zeus.initialziation import model_initialization
from zeus.model.model_utils import get_cosine_schedule_with_warmup
from zeus.utils import gmm_loss_with_regularizes

try:
    from .metrics import centered_linear_cka, geometry_metrics, known_k_kmeans_ari
except ImportError:  # pragma: no cover - exercised by direct CLI execution
    from geometry_probe.metrics import centered_linear_cka, geometry_metrics, known_k_kmeans_ari


@dataclass(frozen=True)
class ProbeConfig:
    seed: int = 42
    train_steps: int = 2
    eval_tasks: int = 1
    train_mode: str = "random"
    num_gaussians: int = 4
    min_points: int = 8
    max_points: int = 12
    dim: int = 10
    min_distance: float = 0.5
    eigenvalue_p1: float = 0.005
    eigenvalue_p2: float = 0.05
    start_distance: float = 1.0
    max_blocks: int = 3
    num_categorical: int = 5
    max_categories: int = 5
    categorical_chance: float = 0.3
    embed_dim: int = 512
    n_head: int = 4
    hid_dim: int = 1024
    n_layers: int = 12
    dropout: float = 0.0
    learning_rate: float = 3e-5
    weight_decay: float = 1e-5
    knn_k: int = 5
    device: str = "cpu"


@dataclass
class TaskRecord:
    index: int
    seed: int
    requested_mode: str
    generator_mode: str
    source_generator_mode: str
    x_obs: torch.Tensor
    labels: torch.Tensor
    x_ref: torch.Tensor
    probabilities: torch.Tensor

    def audit_metadata(self) -> dict[str, object]:
        return {
            "index": self.index,
            "seed": self.seed,
            "requested_mode": self.requested_mode,
            "generator_mode": self.generator_mode,
            "source_generator_mode": self.source_generator_mode,
            "n_samples": int(self.labels.numel()),
            "n_clusters": int(torch.unique(self.labels).numel()),
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _generator_kwargs(config: ProbeConfig, mode: str) -> dict[str, object]:
    return {
        "num_gaussians": config.num_gaussians,
        "min_points": config.min_points,
        "max_points": config.max_points,
        "dim": config.dim,
        "min_distance": config.min_distance,
        "p1": config.eigenvalue_p1,
        "p2": config.eigenvalue_p2,
        "mode": mode,
        "max_blocks": config.max_blocks,
        "num_categorical": config.num_categorical,
        "max_categories": config.max_categories,
        "start_distance": config.start_distance,
        "categorical_chance": config.categorical_chance,
    }


def generate_task(index: int, seed: int, requested_mode: str, config: ProbeConfig) -> TaskRecord:
    """Materialize a ZEUS task with a seed independent of global call order."""
    # datasets.py also exposes real OpenML loaders; defer that optional dependency
    # until a synthetic task is actually requested.
    if not hasattr(np, "sctypes"):
        # openml 0.12.x still reads this NumPy <2 compatibility attribute while
        # importing its unused real-data extension.
        np.sctypes = {
            "int": [np.int8, np.int16, np.int32, np.int64],
            "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
            "float": [np.float16, np.float32, np.float64],
            "complex": [np.complex64, np.complex128],
            "others": [bool, object, bytes, str],
        }
    from zeus.datasets import dataset_generator

    set_seed(seed)
    x_obs, labels, generator_mode, x_ref, probabilities = next(
        dataset_generator(**_generator_kwargs(config, requested_mode))
    )
    canonical_mode = "gaussian_transformed" if "transformed" in generator_mode else "gaussian"
    return TaskRecord(
        index=index,
        seed=seed,
        requested_mode=canonical_mode if requested_mode in {"gaussian", "transformed"} else requested_mode,
        generator_mode=canonical_mode if requested_mode in {"gaussian", "transformed"} else generator_mode,
        source_generator_mode=generator_mode,
        x_obs=x_obs.cpu(),
        labels=labels.cpu(),
        x_ref=x_ref.cpu(),
        probabilities=probabilities.cpu(),
    )


def materialize_tasks(seeds: Iterable[int], requested_mode: str, config: ProbeConfig) -> list[TaskRecord]:
    return [generate_task(index, seed, requested_mode, config) for index, seed in enumerate(seeds)]


def _model_config(config: ProbeConfig) -> GMMConfig:
    return GMMConfig(
        device=config.device,
        num_gaussians=config.num_gaussians,
        min_points=config.min_points,
        max_points=config.max_points,
        dim=config.dim,
        start_distance=config.start_distance,
        end_distance=config.min_distance,
        eigenvalue_p1=config.eigenvalue_p1,
        eigenvalue_p2=config.eigenvalue_p2,
        max_blocks=config.max_blocks,
        num_categorical=config.num_categorical,
        max_categories=config.max_categories,
        categorical_chance=config.categorical_chance,
        embed_dim=config.embed_dim,
        n_head=config.n_head,
        hid_dim=config.hid_dim,
        n_layers=config.n_layers,
        dropout=config.dropout,
    )


def sample_tokens(model: torch.nn.Module, x_obs: torch.Tensor, num_gaussians: int) -> torch.Tensor:
    output = model(x_obs.unsqueeze(1))
    return output[:-num_gaussians].squeeze(1)


def build_paired_models(config: ProbeConfig) -> dict[str, torch.nn.Module]:
    """Create independent ZEUS models that start from one cloned state dict."""
    set_seed(config.seed)
    template = model_initialization(_model_config(config))
    initial_state = copy.deepcopy(template.state_dict())
    models: dict[str, torch.nn.Module] = {}
    for arm in ("zeus", "geometry"):
        model = model_initialization(_model_config(config))
        model.load_state_dict(copy.deepcopy(initial_state), strict=True)
        models[arm] = model
    return models


def train_arm(model: torch.nn.Module, tasks: list[TaskRecord], config: ProbeConfig, objective: str) -> list[float]:
    """Train one arm on pre-materialized tasks; this is the only arm-specific path."""
    # Keep stochastic layers (if enabled) identical across arms as well as the
    # materialized task stream. The objective remains the only intended change.
    set_seed(config.seed + 10_000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 0, len(tasks))
    losses: list[float] = []
    model.train()

    for task in tasks:
        x_obs = task.x_obs.to(config.device)
        labels = task.labels.to(config.device)
        x_ref = task.x_ref.to(config.device)
        optimizer.zero_grad(set_to_none=True)
        representation = sample_tokens(model, x_obs, config.num_gaussians)
        if objective == "zeus":
            probabilities = task.probabilities.to(config.device)
            loss = gmm_loss_with_regularizes(representation.unsqueeze(1), labels, probs=probabilities)
        elif objective == "geometry":
            loss = 1.0 - centered_linear_cka(representation, x_ref)
        else:
            raise ValueError(f"Unknown objective: {objective}")
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.detach().cpu().item()))
    return losses


def evaluate_arm(model: torch.nn.Module, tasks: list[TaskRecord], config: ProbeConfig) -> list[dict[str, object]]:
    model.eval()
    records: list[dict[str, object]] = []
    with torch.no_grad():
        for task in tasks:
            x_obs = task.x_obs.to(config.device)
            x_ref = task.x_ref.to(config.device)
            labels = task.labels.to(config.device)
            n_clusters = int(torch.unique(labels).numel())
            representation = sample_tokens(model, x_obs, config.num_gaussians)
            metrics = {
                "ari_x_obs": known_k_kmeans_ari(x_obs, labels, n_clusters, task.seed),
                "ari_representation": known_k_kmeans_ari(representation, labels, n_clusters, task.seed),
                "ari_x_ref": known_k_kmeans_ari(x_ref, labels, n_clusters, task.seed),
            }
            metrics.update(geometry_metrics(representation, x_ref, config.knn_k))
            records.append({"task": task.audit_metadata(), "metrics": metrics})
    return records


def summarize(records: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    values: dict[str, list[float]] = defaultdict(list)
    for record in records:
        for name, value in record["metrics"].items():
            values[name].append(float(value))
    return {
        name: {"mean": float(np.mean(scores)), "std": float(np.std(scores, ddof=0))}
        for name, scores in values.items()
    }


def run_probe(config: ProbeConfig, output: Path) -> dict[str, object]:
    """Run paired training and write a reproducible, per-task JSON report."""
    if config.train_steps < 1 or config.eval_tasks < 1:
        raise ValueError("train_steps and eval_tasks must both be positive.")
    if config.knn_k < 1:
        raise ValueError("knn_k must be positive.")

    train_seeds = [config.seed + offset for offset in range(config.train_steps)]
    eval_seeds = {
        "gaussian": [config.seed + 100_000 + offset for offset in range(config.eval_tasks)],
        "gaussian_transformed": [config.seed + 200_000 + offset for offset in range(config.eval_tasks)],
    }
    train_tasks = materialize_tasks(train_seeds, config.train_mode, config)
    evaluation_tasks = {
        "gaussian": materialize_tasks(eval_seeds["gaussian"], "gaussian", config),
        "gaussian_transformed": materialize_tasks(eval_seeds["gaussian_transformed"], "transformed", config),
    }

    models = build_paired_models(config)
    arms: dict[str, object] = {}
    for arm, objective in (("zeus", "zeus"), ("geometry", "geometry")):
        losses = train_arm(models[arm], train_tasks, config, objective)
        per_mode = {mode: evaluate_arm(models[arm], tasks, config) for mode, tasks in evaluation_tasks.items()}
        arms[arm] = {
            "train_losses": losses,
            "per_task": per_mode,
            "summary": {mode: summarize(records) for mode, records in per_mode.items()},
        }

    result: dict[str, object] = {
        "protocol": {
            "comparison": "Same ZEUS generator, Transformer state, task stream, optimizer, and steps; objective only differs.",
            "zeus_objective": "gmm_loss_with_regularizes",
            "geometry_objective": "1 - centered_linear_cka(H, X_ref)",
        },
        "config": asdict(config),
        "training_tasks": [task.audit_metadata() for task in train_tasks],
        "evaluation_tasks": {
            mode: [task.audit_metadata() for task in tasks] for mode, tasks in evaluation_tasks.items()
        },
        "arms": arms,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        import json

        json.dump(result, handle, indent=2)
    return result
