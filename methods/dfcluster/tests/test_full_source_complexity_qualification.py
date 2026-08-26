from dataclasses import replace
from pathlib import Path

import pytest

from methods.dfcluster.generator_v4 import (
    FullQualificationConfig,
    FullSamplerConfig,
    run_full_qualification,
    sample_full_task_config,
)
from methods.dfcluster.generator_v4.source_complexity_graph import OBSERVATION_STRATA


def test_full_sampler_uses_plan_ranges_and_forces_envelope_boundaries():
    sampler = FullSamplerConfig()
    first = sample_full_task_config(
        generator_seed=20260825,
        task_index=0,
        split="contract",
        observation_stratum=OBSERVATION_STRATA[0],
        sampler=sampler,
    )
    maximum = sample_full_task_config(
        generator_seed=20260825,
        task_index=1,
        split="contract",
        observation_stratum=OBSERVATION_STRATA[-1],
        sampler=sampler,
    )
    assert (first.n_samples, first.n_features, first.n_clusters, first.intrinsic_dim) == (128, 10, 2, 2)
    assert (maximum.n_samples, maximum.n_features, maximum.n_clusters, maximum.intrinsic_dim) == (2048, 160, 20, 128)
    assert 0.0 <= first.missing_rate <= 0.30
    assert first.n_clusters * max(8, int(__import__("math").ceil(0.01 * first.n_samples))) <= first.n_samples
    assert maximum.n_clusters * max(8, int(__import__("math").ceil(0.01 * maximum.n_samples))) <= maximum.n_samples
    assert first.observation_family in OBSERVATION_STRATA


def test_full_qualification_rejects_scale_reduction():
    with pytest.raises(ValueError, match="may not be reduced"):
        FullQualificationConfig(stage="contract", generator_seeds=(20260825,), tasks_per_stratum=63).validate()
    with pytest.raises(ValueError, match="three frozen"):
        FullQualificationConfig(stage="qualification", generator_seeds=(20260825,), tasks_per_stratum=4096).validate()


def test_plan_contract_runner_uses_64_tasks_per_stratum(tmp_path: Path):
    config = FullQualificationConfig(stage="contract", generator_seeds=(20260825,), tasks_per_stratum=64)
    report = run_full_qualification(config, tmp_path / "contract")
    assert report["status"] == "passed"
    assert report["expected_task_count"] == 64 * len(OBSERVATION_STRATA)
    assert report["coverage"] == {stratum: 64 for stratum in OBSERVATION_STRATA}
    assert report["ranges"]["N"] == {"min": 128, "max": 2048}
    assert report["ranges"]["D"] == {"min": 10, "max": 160}
    assert report["labels_opened"] is False
    assert report["task_selection_performed"] is False
