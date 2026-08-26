import inspect

import numpy as np
import pytest

from methods.dfcluster.generator_v4 import V4Config, generate_observation, generate_v4_task
from methods.dfcluster.generator_v4.mechanisms import flow_iresnet_node
from methods.dfcluster.generator_v4.source_complexity_graph import OBSERVATION_STRATA


@pytest.mark.parametrize("family", OBSERVATION_STRATA)
def test_source_complexity_graph_is_deep_replayable_and_label_free(family: str):
    config = V4Config(
        n_samples=128,
        n_features=10,
        n_clusters=2,
        intrinsic_dim=8,
        observation_family=family,
        missingness="mnar",
        missing_rate=0.10,
        seed=2300 + OBSERVATION_STRATA.index(family),
    )
    first = generate_v4_task(config)
    second = generate_v4_task(config)
    np.testing.assert_array_equal(first.features, second.features)
    np.testing.assert_array_equal(first.missing_mask, second.missing_mask)
    assert first.observation_graph == second.observation_graph
    certificate = first.metadata
    assert all(3 <= depth <= 8 for depth in certificate["informative_leaf_path_depths"])
    assert certificate["node_depth_max"] >= 8 or max(certificate["informative_leaf_path_depths"]) >= 6
    assert certificate["node_role_counts"]["nuisance"] > 0
    assert certificate["parent_count_summary"]["max"] >= 2
    assert len(certificate["operation_counts"]) >= 2
    assert certificate["observation_map_reads_labels"] is False
    assert np.isfinite(first.features).all()
    assert np.all(first.features[first.missing_mask.astype(bool)] == 0.0)


def test_source_complexity_dag_is_exactly_row_permutation_equivariant():
    task = generate_v4_task(
        V4Config(
            n_samples=128,
            n_features=10,
            n_clusters=2,
            intrinsic_dim=8,
            observation_family="mixed",
            missingness="mar",
            seed=2401,
        )
    )
    permutation = np.random.default_rng(8).permutation(task.features.shape[0])
    permuted = generate_observation(
        task.clean_latent[permutation],
        task.nuisance_roots[permutation],
        task.observation_graph,
        task.row_ids[permutation],
    )
    np.testing.assert_array_equal(permuted.features, task.features[permutation])
    np.testing.assert_array_equal(permuted.missing_mask, task.missing_mask[permutation])


def test_mitra_tbp_task_contains_all_five_tree_families():
    task = generate_v4_task(
        V4Config(
            n_samples=128,
            n_features=10,
            n_clusters=2,
            intrinsic_dim=8,
            observation_family="mitra_inspired_tbp",
            seed=2501,
        )
    )
    operations = set(task.observation_graph.node_ops)
    assert {"tbp_dt", "tbp_et", "tbp_rf", "tbp_gb", "tbp_direct_rf"}.issubset(operations)
    assert task.metadata["tree_node_count"] >= 5


def test_flow_is_strictly_monotone_in_primary_coordinate_with_fixed_conditioning():
    primary = np.linspace(-3.0, 3.0, 257)
    values = np.column_stack((primary, np.zeros_like(primary), np.zeros_like(primary)))
    output = flow_iresnet_node(values, np.arange(primary.size, dtype=np.uint64), 1, 2601)
    assert np.all(np.diff(output) > 0.0)


def test_full_graph_observation_signature_has_no_label_parameter():
    parameters = inspect.signature(generate_observation).parameters
    assert set(parameters).isdisjoint({"labels", "y", "K", "CLM", "ARI"})
