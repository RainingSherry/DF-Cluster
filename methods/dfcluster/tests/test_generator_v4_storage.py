from pathlib import Path

from methods.dfcluster.generator_v4 import (
    V4Config,
    generate_v4_task,
    validate_audit_artifact,
    validate_input_artifact,
    validate_target_artifact,
    write_task_artifacts,
)


def test_v4_storage_separates_input_target_and_audit(tmp_path: Path):
    task = generate_v4_task(
        V4Config(n_samples=128, n_features=10, n_clusters=2, intrinsic_dim=8, seed=1414)
    )
    manifest = write_task_artifacts(tmp_path / "task", task)
    assert manifest["labels_in_input"] is False
    assert manifest["labels_in_target"] is True
    assert manifest["labels_in_privileged_target"] is True
    assert manifest["labels_in_audit_sidecar"] is True
    assert manifest["target_label_use_protocol"] == "offline_meta_pretraining_coassignment_loss_only"
    assert validate_input_artifact(Path(manifest["input_path"]))["labels_opened"] is False
    target = validate_target_artifact(Path(manifest["target_path"]))
    assert target["labels_present"] is True
    assert target["label_use_protocol"] == "offline_meta_pretraining_coassignment_loss_only"
    assert validate_audit_artifact(Path(manifest["audit_path"]))["audit_only"] is True
