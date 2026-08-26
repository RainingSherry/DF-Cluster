import pytest

from methods.dfcluster.generator_v4 import FullValidityConfig
from methods.dfcluster.generator_v4.full_qualification import FULL_GENERATOR_SEEDS, FULL_TASKS_PER_STRATUM


def test_full_validity_scale_and_policy_are_frozen():
    config = FullValidityConfig()
    config.validate()
    assert config.generator_seeds == FULL_GENERATOR_SEEDS
    assert config.tasks_per_stratum == FULL_TASKS_PER_STRATUM
    assert config.validity.selection_policy == "audit_only"
    assert config.validity.n_estimators == 256


def test_full_validity_rejects_scale_reduction():
    with pytest.raises(ValueError, match="may not be reduced"):
        FullValidityConfig(tasks_per_stratum=4095).validate()
