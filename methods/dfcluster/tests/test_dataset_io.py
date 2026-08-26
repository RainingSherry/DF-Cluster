import json

import h5py
import numpy as np
import pytest

from methods.dfcluster.dataset_io import (
    ShardConflictError,
    ShardValidationError,
    TaskPayload,
    is_completed_pair,
    stable_sha256,
    validate_paired_shards,
    validate_multi_paired_shards,
    write_multi_task_shards,
    write_task_shards,
)


def _task(task_id: str = "hetero-0", *, clean_dim: int = 3) -> TaskPayload:
    features = np.arange(12, dtype=np.float64).reshape(4, 3) / 10.0
    clean_signal = np.arange(4 * clean_dim, dtype=np.float64).reshape(4, clean_dim)
    missing_mask = np.array(
        [[False, True, False], [False, False, False], [True, False, False], [False, False, True]],
        dtype=np.bool_,
    )
    return TaskPayload(
        task_id=task_id,
        features=features,
        clean_signal=clean_signal,
        missing_mask=missing_mask,
        feature_types=np.array([0, 1, 2], dtype=np.uint8),
        labels=np.array([0, 1, 1, 0], dtype=np.int16),
        K=2,
        CLM=0.75,
        feature_attrs={"generator": "test", "seed": 4},
    )


def test_paired_schema_and_training_isolation(tmp_path):
    result = write_task_shards(_task(), tmp_path, worker_id="worker-3", compression="gzip1")

    assert result.features_path.exists()
    assert result.labels_path.exists()
    assert not list(tmp_path.glob("*.partial"))
    assert "labels_path" not in result.training_manifest
    assert "K" not in result.training_manifest
    assert "CLM" not in result.training_manifest
    assert "Y" not in result.training_manifest
    assert "labels_path" in result.audit_manifest
    # JSON serialization is an easy guard against accidentally returning Path
    # or numpy objects in a manifest record.
    json.dumps(result.training_manifest)
    json.dumps(result.audit_manifest)

    with h5py.File(result.features_path, "r") as features_file:
        assert set(features_file.keys()) == {"features"}
        assert set(features_file["features"].keys()) == {
            "features",
            "clean_signal",
            "missing_mask",
            "feature_types",
        }
        assert features_file["features/features"].dtype == np.float32
        assert features_file["features/clean_signal"].dtype == np.float32
        assert features_file["features/missing_mask"].dtype == np.bool_
        assert features_file["features/feature_types"].dtype == np.uint8
        feature_blob = repr(dict(features_file.attrs)) + repr(dict(features_file["features"].attrs))
        assert "CLM" not in feature_blob
        assert "K" not in feature_blob

    with h5py.File(result.labels_path, "r") as labels_file:
        assert set(labels_file.keys()) == {"labels"}
        assert labels_file["labels/labels"].dtype == np.int32
        assert set(labels_file["labels"].attrs.keys()) == {
            "K",
            "CLM",
            "clm_cha_observed",
            "clm_status",
        }

    validation = validate_paired_shards(result.features_path, result.labels_path)
    assert validation.task_id == "hetero-0"
    assert validation.clean_signal_dim == 3


def test_atomic_completion_and_one_sided_pair_is_not_resume(tmp_path):
    result = write_task_shards(_task(), tmp_path)
    assert not result.paths.features_partial.exists()
    assert not result.paths.labels_partial.exists()
    assert is_completed_pair(result.features_path, result.labels_path, expected_task_id="hetero-0")

    result.labels_path.unlink()
    assert not is_completed_pair(result.features_path, result.labels_path)
    with pytest.raises(ShardConflictError, match="one side"):
        write_task_shards(_task(), tmp_path)


def test_stable_sha256_is_independent_of_mapping_order():
    left = {"b": [1, {"z": 2, "a": 3}], "a": np.array([1, 2], dtype=np.int16)}
    right = {"a": np.array([1, 2], dtype=np.int16), "b": [1, {"a": 3, "z": 2}]}
    assert stable_sha256(left) == stable_sha256(right)


def test_resume_and_intentional_task_mismatch(tmp_path):
    original = _task()
    first = write_task_shards(original, tmp_path)
    resumed = write_task_shards(original, tmp_path)
    assert resumed.resumed is True
    assert resumed.training_manifest == first.training_manifest

    with pytest.raises(ShardConflictError, match="different task content"):
        write_task_shards(_task(clean_dim=2), tmp_path)

    with h5py.File(first.labels_path, "r+") as labels_file:
        labels_file.attrs["task_id"] = "wrong-task"
    assert not is_completed_pair(first.features_path, first.labels_path)
    with pytest.raises(ShardConflictError, match="invalid"):
        write_task_shards(original, tmp_path)


def test_post_write_shape_and_finite_checks(tmp_path):
    bad_mask = _task()
    bad_mask = TaskPayload(
        **{**bad_mask.__dict__, "missing_mask": np.zeros((4, 2), dtype=np.uint8)}
    )
    with pytest.raises(ShardValidationError, match="missing_mask shape"):
        write_task_shards(bad_mask, tmp_path)

    bad_values = _task()
    bad_values = TaskPayload(
        **{**bad_values.__dict__, "features": np.array([[np.nan, 0, 0]] * 4)}
    )
    with pytest.raises(ShardValidationError, match="non-finite"):
        write_task_shards(bad_values, tmp_path)


def test_multi_task_pair_stores_heterogeneous_groups_and_isolates_manifest(tmp_path):
    tasks = [
        _task("task-0", clean_dim=2),
        _task("task-1", clean_dim=4),
        _task("task-2", clean_dim=1),
    ]
    feature_path = tmp_path / "features" / "shard-00000.h5"
    label_path = tmp_path / "labels" / "shard-00000.h5"
    result = write_multi_task_shards(
        tasks,
        feature_path=feature_path,
        labels_path=label_path,
        worker_id="worker-0",
    )
    assert result.task_ids == ("task-0", "task-1", "task-2")
    assert not list(tmp_path.rglob("*.partial"))
    validation = validate_multi_paired_shards(
        feature_path, label_path, expected_task_ids=result.task_ids
    )
    assert validation.task_count == 3
    assert [record.clean_signal_dim for record in validation.records] == [2, 4, 1]
    for record in result.training_manifest_records:
        serialized = json.dumps(record)
        assert "labels_path" not in record
        assert '"K"' not in serialized
        assert "clm_cha" not in serialized
    with h5py.File(feature_path, "r") as handle:
        assert tuple(handle["tasks"].keys()) == result.task_ids
        assert "labels" not in repr(dict(handle.attrs)).lower()
    with h5py.File(label_path, "r") as handle:
        assert int(handle.attrs["task_count"]) == 3
        assert "clm_cha_observed" in handle["tasks/task-0"].attrs


def test_multi_task_resume_requires_both_sides_and_identical_content(tmp_path):
    tasks = [_task("task-0"), _task("task-1")]
    feature_path = tmp_path / "features.h5"
    label_path = tmp_path / "labels.h5"
    first = write_multi_task_shards(
        tasks,
        feature_path=feature_path,
        labels_path=label_path,
        worker_id="worker-0",
    )
    second = write_multi_task_shards(
        tasks,
        feature_path=feature_path,
        labels_path=label_path,
        worker_id="worker-0",
    )
    assert first.resumed is False
    assert second.resumed is True
    label_path.unlink()
    with pytest.raises(ShardConflictError, match="one side"):
        write_multi_task_shards(
            tasks,
            feature_path=feature_path,
            labels_path=label_path,
            worker_id="worker-0",
        )
