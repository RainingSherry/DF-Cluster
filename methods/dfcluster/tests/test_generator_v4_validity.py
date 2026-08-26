from pathlib import Path

import pytest

from methods.dfcluster.generator_v4 import (
    V4Config,
    ValidityConfig,
    compute_validity_certificate,
    generate_v4_task,
    run_validity_audit,
)


def test_validity_audit_is_outer_only_and_array_free(tmp_path: Path):
    task = generate_v4_task(
        V4Config(n_samples=128, n_features=10, n_clusters=2, intrinsic_dim=8, seed=1212)
    )
    certificate = compute_validity_certificate(task, ValidityConfig(n_estimators=256))
    assert certificate.task_id == task.metadata["task_id"]
    assert certificate.labels_opened_only_in_audit is True
    report = run_validity_audit(
        [task], ValidityConfig(n_estimators=256), tmp_path / "validity"
    )
    assert report["performance_claim"] is False
    assert report["selection_policy"] == "audit_only"
    assert report["task_selection_performed"] is False
    assert report["labels_opened_only_in_audit"] is True
    assert report["labels_written_to_report"] is False
    assert (tmp_path / "validity" / "report.json").exists()
    report_text = (tmp_path / "validity" / "report.json").read_text()
    assert '"labels":' not in report_text
    assert '"Y":' not in report_text


def test_validity_config_rejects_non_audit_selection_policy():
    from methods.dfcluster.generator_v4 import ValidityConfig
    with pytest.raises(ValueError, match="task selection"):
        ValidityConfig(selection_policy="filter_tasks").validate()
