import inspect

import numpy as np
import pytest
import torch

from methods.dfcluster.generator_v4 import (
    DDBMConfig,
    DatasetContextDDBM,
    GlobalAE,
    GlobalAEConfig,
    V4Config,
    centered_normalized_gram_torch,
    ddbm_geometry_loss,
    generate_v4_task,
    mixed_type_reconstruction_loss,
)


def _tiny_table():
    task = generate_v4_task(V4Config(
        n_samples=128, n_features=10, n_clusters=2, intrinsic_dim=8,
        observation_family="mixed", seed=909,
    ))
    return (
        torch.from_numpy(task.features.astype(np.float32)),
        torch.from_numpy(task.missing_mask.astype(bool)),
        torch.from_numpy(task.feature_types.astype(np.int64)), task,
    )


def _tiny_ae():
    return GlobalAE(GlobalAEConfig(
        cell_hidden_dim=32, heads=4, column_layers=1, row_layers=1,
        ffn_dim=64, perceiver_queries=2,
    )).eval()


def test_global_ae_contract_and_default_plan_dimensions():
    GlobalAEConfig().validate()
    model = _tiny_ae()
    values, mask, types, _ = _tiny_table()
    with torch.no_grad():
        output = model(values, mask, types)
    assert output.observation.shape == (128, 128)
    assert output.schema_tokens.shape == (10, 32)
    assert output.reconstruction.shape == values.shape
    assert output.mask_logits.shape == values.shape
    assert output.category_logits.shape == (128, 10, 16)
    assert output.scale.shape == values.shape
    loss, metrics = mixed_type_reconstruction_loss(output, values, mask, types)
    assert torch.isfinite(loss) and torch.isfinite(metrics["mask"])
    assert "labels" not in inspect.signature(model.forward).parameters


def test_global_ae_is_row_and_column_permutation_equivariant():
    torch.manual_seed(4)
    model = _tiny_ae()
    values, mask, types, _ = _tiny_table()
    with torch.no_grad():
        original = model(values, mask, types)
        row_perm = torch.randperm(values.shape[0])
        row_view = model(values[row_perm], mask[row_perm], types)
        col_perm = torch.randperm(values.shape[1])
        col_view = model(values[:, col_perm], mask[:, col_perm], types[col_perm])
    torch.testing.assert_close(row_view.observation, original.observation[row_perm], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(row_view.reconstruction, original.reconstruction[row_perm], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(row_view.schema_tokens, original.schema_tokens, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(col_view.observation, original.observation, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(col_view.schema_tokens, original.schema_tokens[col_perm], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(col_view.reconstruction, original.reconstruction[:, col_perm], atol=1e-5, rtol=1e-5)


def test_ddbm_contract_is_contextual_and_label_free():
    DDBMConfig().validate()
    model = DatasetContextDDBM(DDBMConfig(hidden_dim=32, heads=4, layers=2, ffn_dim=64)).eval()
    torch.manual_seed(5)
    noisy, context = torch.randn(32, 128), torch.randn(32, 128)
    with torch.no_grad():
        output = model(noisy, context, timestep=17)
        permutation = torch.randperm(noisy.shape[0])
        permuted = model(noisy[permutation], context[permutation], timestep=17)
    assert output.shape == noisy.shape and torch.isfinite(output).all()
    torch.testing.assert_close(permuted, output[permutation], atol=1e-5, rtol=1e-5)
    assert "labels" not in inspect.signature(model.forward).parameters


def test_geometry_loss_respects_orthogonal_clean_geometry():
    torch.manual_seed(6)
    clean = torch.randn(32, 128)
    q, r = torch.linalg.qr(torch.randn(128, 128))
    predicted = clean @ (q * torch.where(torch.diag(r) < 0, -1.0, 1.0))
    loss, metrics = ddbm_geometry_loss(
        predicted, clean, generator=torch.Generator(device="cpu").manual_seed(9), pair_count=256
    )
    assert float(loss) < 1e-5 and float(metrics["gram_loss"]) < 1e-8
    torch.testing.assert_close(
        centered_normalized_gram_torch(predicted), centered_normalized_gram_torch(clean),
        atol=1e-5, rtol=1e-5,
    )


def test_invalid_ddbm_timestep_is_rejected():
    model = DatasetContextDDBM(DDBMConfig(hidden_dim=32, heads=4, layers=1, ffn_dim=64))
    with pytest.raises(ValueError, match="timestep"):
        model(torch.zeros(4, 128), torch.zeros(4, 128), timestep=512)
