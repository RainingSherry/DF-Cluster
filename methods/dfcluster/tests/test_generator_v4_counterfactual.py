import numpy as np

from methods.dfcluster.generator_v4 import V4Config, generate_counterfactual_tasks


def test_counterfactual_tasks_share_roots_but_have_distinct_observation_maps():
    config = V4Config(
        n_samples=128, n_features=10, n_clusters=2, intrinsic_dim=8, seed=1616
    )
    families = ("flow_analytic", "tabicl_tree", "mitra_inspired_tbp")
    first = generate_counterfactual_tasks(config, families)
    second = generate_counterfactual_tasks(config, families)
    assert set(first) == set(families)
    parent_ids = {task.metadata["parent_task_id"] for task in first.values()}
    assert len(parent_ids) == 1
    for family in families:
        task = first[family]
        replay = second[family]
        np.testing.assert_array_equal(task.clean_latent, first["flow_analytic"].clean_latent)
        np.testing.assert_array_equal(task.nuisance_roots, first["flow_analytic"].nuisance_roots)
        np.testing.assert_array_equal(task.labels, first["flow_analytic"].labels)
        np.testing.assert_array_equal(task.features, replay.features)
        assert task.metadata == replay.metadata
        assert "labels" not in repr(task.training_payload()).lower()
    assert any(
        not np.array_equal(first["flow_analytic"].features, first[family].features)
        for family in families[1:]
    )
