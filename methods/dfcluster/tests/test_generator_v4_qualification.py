from pathlib import Path

from methods.dfcluster.generator_v4 import QualificationConfig, run_qualification


def test_small_qualification_is_cpu_only_and_label_free(tmp_path: Path):
    report = run_qualification(
        QualificationConfig(seeds=(11,), tasks_per_observation_family=2),
        tmp_path / "qualification",
    )
    assert report["status"] == "passed"
    assert report["performance_claim"] is False
    assert report["labels_opened"] is False
    assert report["training_forbidden_fields_opened"] is False
    assert (tmp_path / "qualification" / "report.json").exists()
    assert (tmp_path / "qualification" / "task_ledger.jsonl").exists()
    ledger = (tmp_path / "qualification" / "task_ledger.jsonl").read_text()
    assert '"labels":' not in ledger
    assert '"Y":' not in ledger
    assert "clean_latent" not in ledger
