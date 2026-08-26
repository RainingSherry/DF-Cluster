import numpy as np
import torch

from methods.dfcluster.generator_v4 import (
    DDBMConfig,
    DatasetContextDDBM,
    GlobalAE,
    GlobalAEConfig,
    V4Config,
    generate_v4_task,
    v4_train_step,
)


def test_v4_label_free_train_step_has_finite_loss_and_gradients():
    task = generate_v4_task(
        V4Config(n_samples=128, n_features=10, n_clusters=2, intrinsic_dim=8, seed=1515)
    )
    ae = GlobalAE(GlobalAEConfig(
        cell_hidden_dim=32, heads=4, column_layers=1, row_layers=1,
        ffn_dim=64, perceiver_queries=2,
    ))
    ddbm = DatasetContextDDBM(DDBMConfig(hidden_dim=32, heads=4, layers=1, ffn_dim=64))
    optimizer = torch.optim.AdamW(list(ae.parameters()) + list(ddbm.parameters()), lr=1e-4)
    values = torch.from_numpy(task.features.astype(np.float32))
    mask = torch.from_numpy(task.missing_mask.astype(bool))
    types = torch.from_numpy(task.feature_types.astype(np.int64))
    clean = torch.from_numpy(task.clean_latent.astype(np.float32))
    metrics = v4_train_step(
        ae, ddbm, values, mask, types, clean, optimizer,
        timestep=128, generator=torch.Generator(device="cpu").manual_seed(7),
    )
    assert set(metrics) == {
        "total_loss", "reconstruction_loss", "geometry_loss",
        "gram_loss", "distance_loss", "gradient_norm",
    }
    assert all(np.isfinite(value) for value in metrics.values())
