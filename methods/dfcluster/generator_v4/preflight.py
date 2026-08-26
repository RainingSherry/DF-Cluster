"""Non-training preflight for the owner-authorized V4 option-A pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict

from .replay import load_candidate_manifest
from .core import source_sha256
from .training import StageAConfig, implementation_sha256
from .stage_b import StageBTrainingConfig


@dataclass(frozen=True)
class PreflightConfig:
    qualification_report: Path
    validity_report: Path
    candidate_coverage_report: Path
    candidate_manifest: Path
    output_root: Path
    validation_manifest_path: Path | None = None
    physical_gpu_id: int | None = None

    def validate(self) -> None:
        for path in (self.qualification_report, self.validity_report, self.candidate_coverage_report, self.candidate_manifest):
            if not Path(path).is_file():
                raise FileNotFoundError(path)
        if self.output_root.exists():
            raise FileExistsError(self.output_root)
        if self.validation_manifest_path is not None and not Path(self.validation_manifest_path).is_file():
            raise FileNotFoundError(self.validation_manifest_path)
        if self.physical_gpu_id in {0, 7}:
            raise ValueError("physical GPU 0 and 7 are forbidden")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_preflight(config: PreflightConfig) -> Dict[str, Any]:
    """Verify frozen evidence and code contracts without starting training."""

    config.validate()
    qualification = json.loads(Path(config.qualification_report).read_text(encoding="utf-8"))
    validity = json.loads(Path(config.validity_report).read_text(encoding="utf-8"))
    coverage = json.loads(Path(config.candidate_coverage_report).read_text(encoding="utf-8"))
    manifest_rows = load_candidate_manifest(config.candidate_manifest)
    validation_info = None
    if config.validation_manifest_path is not None:
        validation_path = Path(config.validation_manifest_path)
        validation_rows = [json.loads(line) for line in validation_path.open(encoding="utf-8")]
        validation_info = {
            "path": str(validation_path),
            "sha256": _sha256(validation_path),
            "task_count": len(validation_rows),
            "schema_version": json.loads((validation_path.parent / "resolved_config.json").read_text(encoding="utf-8")).get("schema_version") if (validation_path.parent / "resolved_config.json").is_file() else None,
            "labels_in_manifest": any(bool(row.get("labels_in_manifest", True)) for row in validation_rows),
            "audit_metrics_in_manifest": any(bool(row.get("audit_metrics_in_manifest", True)) for row in validation_rows),
        }
    current_generator_sha = source_sha256()
    current_implementation_sha = implementation_sha256()

    checks = {
        "qualification_passed": qualification.get("status") == "passed" and int(qualification.get("completed_task_count", -1)) == 61440,
        "qualification_unique": int(qualification.get("unique_task_count", -1)) == 61440,
        "qualification_information_coverage": qualification.get("information_coverage_ok") is True,
        "qualification_source_sha_current": qualification.get("source_sha256") == [current_generator_sha],
        "validity_complete": int(validity.get("completed_task_count", -1)) == 61440 and int(validity.get("unique_task_count", -1)) == 61440,
        "validity_audit_only": validity.get("labels_opened_only_in_audit") is True and validity.get("task_selection_performed") is False,
        "candidate_coverage_passed": coverage.get("status") == "passed" and int(coverage.get("nonempty_cell_count", 0)) == int(coverage.get("required_cell_count", -1)),
        "candidate_manifest_nonempty": len(manifest_rows) > 0,
        "candidate_manifest_no_labels": all(row.get("labels_in_manifest") is False for row in manifest_rows),
        "candidate_manifest_no_audit_metrics": all(not any(key in row for key in ("ari_raw", "ari_clean", "ari_headroom", "clm_observed", "probe_macro_f1")) for row in manifest_rows),
        "validation_manifest_fixed": validation_info is not None and validation_info["task_count"] == 23903 and validation_info["schema_version"] == "dfcluster.generator_v4.stage_a.validation.v3",
        "validation_manifest_no_labels": validation_info is not None and validation_info["labels_in_manifest"] is False,
        "validation_manifest_no_audit_metrics": validation_info is not None and validation_info["audit_metrics_in_manifest"] is False,
    }
    # Validate frozen default contracts without constructing the large models.
    StageAConfig().validate()
    StageBTrainingConfig().validate()
    if not all(checks.values()):
        failed = [name for name, value in checks.items() if not value]
        raise RuntimeError("V4 preflight failed: " + ", ".join(failed))

    report = {
        "schema_version": "dfcluster.generator_v4.preflight.v1",
        "status": "passed",
        "option": "A",
        "checks": checks,
        "qualification_report_sha256": _sha256(config.qualification_report),
        "validity_report_sha256": _sha256(config.validity_report),
        "candidate_coverage_report_sha256": _sha256(config.candidate_coverage_report),
        "candidate_manifest_sha256": _sha256(config.candidate_manifest),
        "generator_source_sha256": current_generator_sha,
        "implementation_sha256": current_implementation_sha,
        "candidate_count": len(manifest_rows),
        "validation_manifest": validation_info,
        "physical_gpu_id": config.physical_gpu_id,
        "gpu_training_started": False,
        "performance_claim": False,
    }
    config.output_root.mkdir(parents=True, exist_ok=False)
    partial = config.output_root / "report.json.partial"
    partial.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(partial, config.output_root / "report.json")
    return report