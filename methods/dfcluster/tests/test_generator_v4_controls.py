import numpy as np
import pytest

from methods.dfcluster.generator_v4 import (
    V4Config,
    generate_v4_task,
    make_impossible_row_misalignment,
)


def test_impossible_control_is_audit_only_and_row_misaligned():
    task = generate_v4_task(
        V4Config(n_samples=128, n_features=10, n_clusters=2, intrinsic_dim=8, seed=1717)
    )
    control = make_impossible_row_misalignment(task)
    assert control.metadata["audit_only"] is True
    assert control.metadata["training_forbidden"] is True
    assert not np.array_equal(control.row_permutation, np.arange(128))
    assert np.array_equal(control.clean_latent, task.clean_latent)
    assert np.array_equal(control.labels, task.labels)
    assert not np.array_equal(control.features, task.features)
    with pytest.raises(RuntimeError, match="audit-only"):
        control.training_payload()
    assert control.audit_payload()["audit_only"] is True
