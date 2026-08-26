import numpy as np
import pytest

from methods.dfcluster.clm_audit import prepare_clm_matrix


def test_clm_preprocessing_imputes_and_standardizes_deterministically():
    values = np.asarray(
        [[1.0, 0.0, np.nan], [3.0, 5.0, 2.0], [100.0, 10.0, 4.0]],
        dtype=np.float64,
    )
    mask = np.asarray(
        [[False, True, False], [False, False, False], [True, False, False]],
        dtype=bool,
    )
    first = prepare_clm_matrix(values, mask)
    second = prepare_clm_matrix(values, mask)
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()
    assert np.allclose(first.mean(axis=0), 0.0)


def test_clm_preprocessing_rejects_bad_masks():
    with pytest.raises(ValueError):
        prepare_clm_matrix(np.ones((3, 2)), np.zeros((3, 1)))

