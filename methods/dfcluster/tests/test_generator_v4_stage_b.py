from pathlib import Path

import numpy as np
import torch

from methods.dfcluster.generator_v4 import (
    DDBMConfig,
    DatasetContextDDBM,
    GlobalAE,
    GlobalAEConfig,
    StageBConfig,
    StageBTrainer,
    StageBTrainingConfig,
    V4Config,
    generate_v4_task,
    stage_b_train_step,
    row_permutation,
    apply_row_permutation,
    inverse_row_permutation,
    redact_v4_target,
    redact_v4_task,
)


def test_stage_b_objective_has_all_five_terms_and_freezes_ae():
    task = generate_v4_task(V4Config(
        n_samples=128, n_features=10, n_clusters=2, intrinsic_dim=8, seed=2828
    ))
    ae = GlobalAE(GlobalAEConfig(
        cell_hidden_dim=32, heads=4, column_layers=1, row_layers=1,
        ffn_dim=64, perceiver_queries=2,
    ))
    ddbm = DatasetContextDDBM(DDBMConfig(hidden_dim=32, heads=4, layers=1, ffn_dim=64))
    optimizer = torch.optim.AdamW(ddbm.parameters(), lr=1e-4)
    values = torch.from_numpy(task.features.astype(np.float32))
    missing = torch.from_numpy(task.missing_mask.astype(bool))
    types = torch.from_numpy(task.feature_types.astype(np.int64))
    clean = torch.from_numpy(task.clean_latent.astype(np.float32))
    labels = torch.from_numpy(task.labels.astype(np.int64))
    metrics = stage_b_train_step(
        ae, ddbm, values, missing, types, clean, labels, optimizer,
        config=StageBConfig(pair_count=256, neighborhood_k=5),
        generator=torch.Generator(device="cpu").manual_seed(29),
    )
    assert set(metrics) == {
        "bridge_loss", "gram_loss", "distance_loss", "neighborhood_loss",
        "pair_loss", "total_loss", "gradient_norm", "timestep",
    }
    assert all(np.isfinite(value) for value in metrics.values())
    assert all(parameter.requires_grad is False for parameter in ae.parameters())


def test_stage_b_trainer_contract_records_frozen_ae_and_pair_boundary(tmp_path: Path):
    task = generate_v4_task(V4Config(
        n_samples=128, n_features=10, n_clusters=2, intrinsic_dim=8, seed=2929
    ))
    ae = GlobalAE(GlobalAEConfig(
        cell_hidden_dim=32, heads=4, column_layers=1, row_layers=1,
        ffn_dim=64, perceiver_queries=2,
    ))
    ddbm = DatasetContextDDBM(DDBMConfig(hidden_dim=32, heads=4, layers=1, ffn_dim=64))
    config = StageBTrainingConfig(
        device="cpu", use_bf16=False,
        objective=StageBConfig(pair_count=256, neighborhood_k=5),
    )
    trainer = StageBTrainer(
        ae, ddbm, config, tmp_path / "stage_b", ae_checkpoint_sha256="frozen-ae-sha", seed=31
    )
    metrics = trainer.step_one(redact_v4_task(task), redact_v4_target(task))
    assert all(np.isfinite(value) for value in metrics.values())
    assert trainer.task_exposure == 1
    assert all(parameter.requires_grad is False for parameter in ae.parameters())
    assert not (tmp_path / "stage_b" / "report.json").exists()


def test_stage_b_rejects_privileged_v4_task(tmp_path: Path):
    task = generate_v4_task(V4Config(n_samples=128, n_features=10, n_clusters=2, intrinsic_dim=8, seed=2930))
    ae = GlobalAE(GlobalAEConfig(cell_hidden_dim=16, heads=4, column_layers=1, row_layers=1, ffn_dim=32, perceiver_queries=1))
    ddbm = DatasetContextDDBM(DDBMConfig(hidden_dim=32, heads=4, layers=1, ffn_dim=64))
    trainer = StageBTrainer(ae, ddbm, StageBTrainingConfig(device="cpu", use_bf16=False, objective=StageBConfig(pair_count=128, neighborhood_k=3)), tmp_path / "stage_b_reject", ae_checkpoint_sha256="sha", seed=32)
    try:
        trainer.step_one(task, redact_v4_target(task))
    except TypeError:
        return
    raise AssertionError("Stage-B accepted a privileged V4Task")


def test_stage_b_row_permutation_has_inverse_alignment():
    values = torch.arange(24).reshape(6, 4)
    permutation, inverse = row_permutation(6, generator=torch.Generator(device="cpu").manual_seed(33), device=torch.device("cpu"))
    permuted = apply_row_permutation(values, permutation)
    restored = inverse_row_permutation(permuted, inverse)
    torch.testing.assert_close(restored, values)
