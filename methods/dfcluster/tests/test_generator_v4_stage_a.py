from pathlib import Path

import numpy as np
import torch

from methods.dfcluster.generator_v4 import (
    DDBMConfig,
    GlobalAE,
    GlobalAEConfig,
    StageAConfig,
    StageACorruptionConfig,
    StageATrainer,
    V4Config,
    generate_v4_task,
    redact_v4_task,
)


def test_stage_a_trainer_one_engineering_step_is_label_free(tmp_path: Path):
    task = generate_v4_task(V4Config(
        n_samples=128, n_features=10, n_clusters=2, intrinsic_dim=8, seed=2727
    ))
    input_task = redact_v4_task(task)
    model = GlobalAE(GlobalAEConfig(
        cell_hidden_dim=32, heads=4, column_layers=1, row_layers=1,
        ffn_dim=64, perceiver_queries=2,
    ))
    config = StageAConfig(
        device="cpu", use_bf16=False, task_exposure_target=5_000_000,
        corruption=StageACorruptionConfig(
            masked_column_rate=0.15, masked_cell_rate=0.10, noise_std=0.05,
            scale_min=0.80, scale_max=1.20, feature_dropout_rate=0.05,
        ),
    )
    trainer = StageATrainer(model, config, tmp_path / "stage_a", seed=17)
    metrics = trainer.step_one(input_task)
    assert all(np.isfinite(value) for value in metrics.values())
    assert trainer.task_exposure == 1
    assert not (tmp_path / "stage_a" / "report.json").exists()
    assert (tmp_path / "stage_a" / "gpu_ledger.jsonl").is_file()


def test_stage_a_rejects_privileged_v4_task(tmp_path: Path):
    task = generate_v4_task(V4Config(n_samples=128, n_features=10, n_clusters=2, intrinsic_dim=8, seed=2728))
    model = GlobalAE(GlobalAEConfig(cell_hidden_dim=32, heads=4, column_layers=1, row_layers=1, ffn_dim=64, perceiver_queries=2))
    trainer = StageATrainer(model, StageAConfig(device="cpu", use_bf16=False), tmp_path / "reject", seed=18)
    try:
        trainer.step_one(task)
    except TypeError:
        return
    raise AssertionError("Stage-A accepted a privileged V4Task")


def test_stage_a_checkpoint_writes_validation_metrics(tmp_path: Path):
    source = Path("/data/luolie/DF-Cluster/data/generator_v4/manifests/v4_stage_a_validation_v2/validation_manifest.jsonl")
    row = source.open(encoding="utf-8").readline()
    manifest = tmp_path / "validation_manifest.jsonl"
    manifest.write_text(row, encoding="utf-8")
    task = generate_v4_task(V4Config(n_samples=128, n_features=10, n_clusters=2, intrinsic_dim=8, seed=2730))
    input_task = redact_v4_task(task)
    model = GlobalAE(GlobalAEConfig(cell_hidden_dim=16, heads=4, column_layers=1, row_layers=1, ffn_dim=32, perceiver_queries=1))
    config = StageAConfig(device="cpu", use_bf16=False, validation_manifest_path=str(manifest), validation_cpu_workers=1)
    trainer = StageATrainer(model, config, tmp_path / "stage_a_validation", seed=19)
    trainer.step_one(input_task)
    metrics = trainer._run_validation()
    assert metrics["task_count"] == 1
    assert metrics["labels_opened"] is False
    assert (tmp_path / "stage_a_validation" / "validation_history.jsonl").is_file()
