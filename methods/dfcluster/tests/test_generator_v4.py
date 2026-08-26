import inspect

import numpy as np
import pytest

from methods.dfcluster.generator_v4 import (
    CLEAN_FAMILIES,
    MISSINGNESS_MODES,
    OBSERVATION_FAMILIES,
    V4Config,
    centered_normalized_gram,
    generate_observation,
    generate_v4_task,
    source_sha256,
    validate_training_payload,
)


@pytest.mark.parametrize("clean_family", CLEAN_FAMILIES)
@pytest.mark.parametrize("observation_family", OBSERVATION_FAMILIES)
def test_v4_contract_is_finite_replayable_and_label_isolated(clean_family, observation_family):
    config = V4Config(
        n_samples=128,
        n_features=10,
        n_clusters=2,
        intrinsic_dim=8,
        clean_family=clean_family,
        observation_family=observation_family,
        seed=101,
    )
    first = generate_v4_task(config)
    second = generate_v4_task(config)

    for name in ("features", "clean_latent", "missing_mask", "feature_types", "labels"):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
    assert first.metadata == second.metadata
    assert first.metadata["labels_isolated"] is True
    assert first.metadata["task_id"].startswith("generator_v4/qualification/101/0")
    assert first.metadata["observation_map_reads_labels"] is False
    assert first.training_payload().keys() == {"task_id", "model_input", "geometry_target"}
    assert first.inference_payload().keys() == {"task_id", "model_input"}
    assert "labels" not in repr(first.training_payload()).lower()
    assert "labels" not in repr(first.inference_payload()).lower()
    assert np.isfinite(first.features).all()
    assert np.isfinite(first.clean_latent).all()
    assert np.all(first.features[first.missing_mask.astype(bool)] == 0.0)
    assert first.metadata["nonfinite_count"] == 0


def test_observation_signature_has_no_label_bearing_argument():
    parameters = inspect.signature(generate_observation).parameters
    assert "labels" not in parameters
    assert "y" not in parameters
    assert "K" not in parameters
    assert "CLM" not in parameters


def test_row_permutation_equivariance_is_exact_for_frozen_graph():
    task = generate_v4_task(
        V4Config(
            n_samples=128,
            n_features=10,
            n_clusters=2,
            intrinsic_dim=8,
            observation_family="mixed",
            missingness="mar",
            seed=202,
        )
    )
    permutation = np.random.default_rng(7).permutation(task.features.shape[0])
    permuted = generate_observation(
        task.clean_latent[permutation],
        task.nuisance_roots[permutation],
        task.observation_graph,
        task.row_ids[permutation],
    )
    np.testing.assert_array_equal(permuted.features, task.features[permutation])
    np.testing.assert_array_equal(
        permuted.missing_mask, task.missing_mask[permutation]
    )
    np.testing.assert_array_equal(permuted.feature_types, task.feature_types)


def test_clean_geometry_is_rotation_invariant_and_certificate_is_complete():
    task = generate_v4_task(
        V4Config(
            n_samples=128,
            n_features=10,
            n_clusters=2,
            intrinsic_dim=8,
            clean_family="factor",
            seed=303,
        )
    )
    rng = np.random.default_rng(9)
    q, r = np.linalg.qr(rng.normal(size=(128, 128)))
    q *= np.where(np.diag(r) < 0.0, -1.0, 1.0)
    rotated = task.clean_latent @ q
    np.testing.assert_allclose(
        centered_normalized_gram(task.clean_latent),
        centered_normalized_gram(rotated),
        atol=1e-5,
    )
    required = {
        "node_count",
        "edge_count",
        "max_depth",
        "feature_type_counts",
        "column_role_counts",
        "missing_cell_count",
        "stochastic_node_count",
        "many_to_one_node_count",
    }
    assert required.issubset(task.metadata)
    assert task.metadata["observation_family"] == "tabicl_graph_mlp"
    assert set(task.metadata["column_role_counts"]) == {"informative", "redundant", "nuisance", "irrelevant"}
    targets = task.geometry_targets(pair_seed=5, num_pairs=64)
    assert targets["clean_gram"].shape == (128, 128)
    assert targets["clean_distances"].shape == (64,)
    assert len(task.metadata["source_sha256"]) == 64
    assert source_sha256() == task.metadata["source_sha256"]


@pytest.mark.parametrize("missingness", MISSINGNESS_MODES)
def test_missingness_modes_are_replayable_and_leave_each_row_observable(missingness):
    task = generate_v4_task(
        V4Config(
            n_samples=128,
            n_features=10,
            n_clusters=2,
            intrinsic_dim=8,
            missingness=missingness,
            missing_rate=0.20,
            seed=404,
        )
    )
    assert task.missing_mask.dtype == np.uint8
    assert not task.missing_mask.all(axis=1).any()
    assert task.metadata["missingness"] == missingness


def test_config_rejects_forbidden_shapes_and_invalid_ranges():
    with pytest.raises(ValueError, match="ambient_dim"):
        generate_v4_task(V4Config(ambient_dim=64))
    with pytest.raises(ValueError, match="n_features"):
        generate_v4_task(V4Config(n_features=9))
    with pytest.raises(ValueError, match="fractions"):
        generate_v4_task(V4Config(categorical_fraction=1.0, ordinal_fraction=0.2))


def test_training_payload_contract_fails_closed_on_forbidden_fields():
    task = generate_v4_task(
        V4Config(n_samples=128, n_features=10, n_clusters=2, intrinsic_dim=8)
    )
    payload = task.training_payload()
    validate_training_payload(payload)
    payload["model_input"]["K"] = 2
    with pytest.raises(ValueError, match="forbidden training field"):
        validate_training_payload(payload)
