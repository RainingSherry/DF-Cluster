"""Plan-registered V4 candidate admission and coverage manifest.

This module implements the owner-authorized option-A boundary: audit-bearing
synthetic values can admit a task to the pre-registered §15 candidate pool and
assign its raw-ARI/CLM coverage cell, but the emitted training manifest and
input payload contain no labels or audit metric values. Failed audit rows are
never silently discarded from the source ledger; they remain in the frozen
full-validity ledger and rejection summary.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from .core import source_sha256
from .source_complexity_graph import INFORMATION_STRATA, OBSERVATION_STRATA

EXPECTED_TASK_COUNT = 3 * len(OBSERVATION_STRATA) * 4096
REQUIRED_RAW_POOLS = ("easy", "medium", "hard-but-recoverable")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_new(path: Path, text: str) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"candidate artifact already exists: {path}")
    partial = path.with_name(path.name + ".partial")
    if partial.exists():
        raise FileExistsError(f"candidate partial artifact already exists: {partial}")
    partial.write_text(text, encoding="utf-8")
    os.replace(partial, path)


def _plan_raw_pool(ari_raw: float) -> str:
    value = float(ari_raw)
    if 0.50 <= value <= 0.80:
        return "easy"
    if 0.15 <= value < 0.50:
        return "medium"
    if value < 0.15:
        return "hard-but-recoverable"
    return "very_easy_above_0.80"


def _empirical_quantile(values: List[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot compute a quantile from an empty pool")
    position = (len(ordered) - 1) * float(fraction)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] + weight * (ordered[upper] - ordered[lower])


def _assign_clm_tertiles(rows: Sequence[Mapping[str, Any]]) -> tuple[Dict[str, int | None], Dict[str, List[float]]]:
    """Assign deterministic empirical tertiles and freeze numeric cutpoints.

    Cutpoints are computed only from gate-passing candidates within each
    raw-ARI pool. A future independent task can therefore receive the same
    tertile without knowing the rank of all future tasks. Intervals are
    ``[min,q1)``, ``[q1,q2)``, and ``[q2,max]``.
    """

    groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        pool = str(row["plan_raw_difficulty_pool"])
        if pool in REQUIRED_RAW_POOLS:
            groups[pool].append(row)
    result: Dict[str, int | None] = {}
    cutpoints: Dict[str, List[float]] = {}
    for pool, group in groups.items():
        values = [float(row["clm_observed"]) for row in group]
        q1 = _empirical_quantile(values, 1.0 / 3.0)
        q2 = _empirical_quantile(values, 2.0 / 3.0)
        cutpoints[pool] = [q1, q2]
        for row in group:
            value = float(row["clm_observed"])
            result[str(row["task_id"])] = 0 if value < q1 else 1 if value < q2 else 2
    for row in rows:
        result.setdefault(str(row["task_id"]), None)
    return result, cutpoints


def _jsonl_text(rows: Iterable[Mapping[str, Any]]) -> str:
    return "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows)


def _require_full_reports(
    validity_report: Mapping[str, Any],
    qualification_report: Mapping[str, Any],
) -> None:
    if int(validity_report.get("completed_task_count", -1)) != EXPECTED_TASK_COUNT:
        raise ValueError("validity report is not the complete 61,440-task audit")
    if int(validity_report.get("unique_task_count", -1)) != EXPECTED_TASK_COUNT:
        raise ValueError("validity report has incomplete or duplicate task coverage")
    if int(qualification_report.get("completed_task_count", -1)) != EXPECTED_TASK_COUNT:
        raise ValueError("qualification report is not the complete 61,440-task qualification")
    if int(qualification_report.get("unique_task_count", -1)) != EXPECTED_TASK_COUNT:
        raise ValueError("qualification report has incomplete or duplicate task coverage")
    if qualification_report.get("status") != "passed":
        raise ValueError("candidate admission requires passed corrected qualification")
    if qualification_report.get("information_coverage_ok") is not True:
        raise ValueError("candidate admission requires equal information-stratum coverage")
    current_source = source_sha256()
    source_list = qualification_report.get("source_sha256") or []
    if source_list != [current_source]:
        raise ValueError("qualification source SHA does not match current generator")


def build_candidate_manifest(
    *,
    validity_report_path: Path,
    validity_ledger_path: Path,
    qualification_report_path: Path,
    output_root: Path,
) -> Dict[str, Any]:
    """Build the complete option-A candidate manifest from frozen evidence."""

    validity_report_path = Path(validity_report_path)
    validity_ledger_path = Path(validity_ledger_path)
    qualification_report_path = Path(qualification_report_path)
    output_root = Path(output_root)
    if output_root.exists():
        raise FileExistsError(f"candidate output already exists: {output_root}")
    if not validity_report_path.is_file() or not validity_ledger_path.is_file() or not qualification_report_path.is_file():
        raise FileNotFoundError("candidate manifest source evidence is incomplete")

    validity_report = json.loads(validity_report_path.read_text(encoding="utf-8"))
    qualification_report = json.loads(qualification_report_path.read_text(encoding="utf-8"))
    _require_full_reports(validity_report, qualification_report)

    rows: List[Dict[str, Any]] = []
    seen: set[str] = set()
    with validity_ledger_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            task_id = str(row.get("task_id"))
            if task_id in seen:
                raise ValueError(f"duplicate task_id in validity ledger at line {line_number}: {task_id}")
            seen.add(task_id)
            rows.append(row)
    if len(rows) != EXPECTED_TASK_COUNT:
        raise ValueError("validity ledger does not contain exactly 61,440 rows")

    candidates = [
        row for row in rows
        if row.get("status") == "passed"
        and bool(row.get("clean_ari_gate"))
        and bool(row.get("headroom_gate"))
        and bool(row.get("probe_gate"))
        and bool(row.get("clean_finite"))
        and bool(row.get("observed_finite"))
    ]
    for row in candidates:
        row["plan_raw_difficulty_pool"] = _plan_raw_pool(float(row["ari_raw"]))
    tertiles, clm_cutpoints = _assign_clm_tertiles(candidates)

    audit_rows: List[Dict[str, Any]] = []
    manifest_rows: List[Dict[str, Any]] = []
    for row in sorted(candidates, key=lambda item: (int(item["generator_seed"]), str(item["observation_stratum"]), int(item["schedule_task_index"]))):
        task_id = str(row["task_id"])
        pool = str(row["plan_raw_difficulty_pool"])
        tertile = tertiles[task_id]
        if pool in REQUIRED_RAW_POOLS and tertile is None:
            raise ValueError(f"missing CLM tertile for candidate {task_id}")
        manifest_rows.append({
            "task_id": task_id,
            "generator_seed": int(row["generator_seed"]),
            "schedule_task_index": int(row["schedule_task_index"]),
            "split": "qualification",
            "observation_stratum": str(row["observation_stratum"]),
            "information_stratum": str(row["information_stratum"]),
            "raw_difficulty_pool": pool,
            "clm_tertile": tertile,
            "n_samples": int(row["N"]),
            "n_features": int(row["D"]),
            "n_clusters": int(row["K"]),
            "intrinsic_dim": int(row["d_int"]),
            "missing_rate": float(row["missing_rate"]),
            "source_sha256": str(row.get("source_sha256", source_sha256())),
            "candidate_gate": "plan_v4_section_15_option_A",
            "model_input_uses_audit_fields": False,
            "labels_in_manifest": False,
        })
        audit_rows.append({
            "task_id": task_id,
            "generator_seed": int(row["generator_seed"]),
            "schedule_task_index": int(row["schedule_task_index"]),
            "observation_stratum": str(row["observation_stratum"]),
            "information_stratum": str(row["information_stratum"]),
            "raw_difficulty_pool": pool,
            "clm_tertile": tertile,
            "ari_raw": float(row["ari_raw"]),
            "ari_clean": float(row["ari_clean"]),
            "ari_headroom": float(row["ari_headroom"]),
            "clm_observed": float(row["clm_observed"]),
            "clm_clean": float(row["clm_clean"]),
            "clm_headroom": float(row["clm_headroom"]),
            "probe_macro_f1": float(row["probe_macro_f1"]),
            "clean_ari_gate": True,
            "headroom_gate": True,
            "probe_gate": True,
            "audit_only": True,
            "labels_written": False,
        })

    required_cells: Dict[str, int] = {}
    for observation in OBSERVATION_STRATA:
        for information in INFORMATION_STRATA:
            for pool in REQUIRED_RAW_POOLS:
                for tertile in range(3):
                    key = f"{observation}/{information}/{pool}/clm_tertile_{tertile}"
                    required_cells[key] = sum(
                        row["observation_stratum"] == observation
                        and row["information_stratum"] == information
                        and row["raw_difficulty_pool"] == pool
                        and row["clm_tertile"] == tertile
                        for row in manifest_rows
                    )
    nonempty = sum(value > 0 for value in required_cells.values())
    coverage_report: Dict[str, Any] = {
        "schema_version": "dfcluster.generator_v4.candidate_coverage.v2",
        "status": "passed" if nonempty == len(required_cells) else "failed_coverage",
        "candidate_count": len(manifest_rows),
        "full_audit_task_count": len(rows),
        "rejected_audit_task_count": len(rows) - len(manifest_rows),
        "required_raw_pools": list(REQUIRED_RAW_POOLS),
        "required_information_strata": list(INFORMATION_STRATA),
        "required_observation_strata": list(OBSERVATION_STRATA),
        "required_cell_count": len(required_cells),
        "nonempty_cell_count": nonempty,
        "minimum_required_cell_count": min(required_cells.values()) if required_cells else 0,
        "maximum_cell_count": max(required_cells.values()) if required_cells else 0,
        "cell_counts": required_cells,
        "candidate_counts_by_raw_pool": dict(Counter(row["raw_difficulty_pool"] for row in manifest_rows)),
        "candidate_counts_by_information_stratum": dict(Counter(row["information_stratum"] for row in manifest_rows)),
        "candidate_counts_by_observation_stratum": dict(Counter(row["observation_stratum"] for row in manifest_rows)),
        "clm_tertile_definition": "Within each gate-passing raw-ARI pool, compute empirical q1/q2 cutpoints and assign [min,q1), [q1,q2), [q2,max].",
        "clm_tertile_cutpoints": clm_cutpoints,
        "very_easy_above_0.80_candidate_count": sum(row["raw_difficulty_pool"] == "very_easy_above_0.80" for row in manifest_rows),
        "selection_policy": "pre_registered_plan_v4_section_15_option_A",
        "labels_opened_only_in_audit": True,
        "labels_written_to_manifest": False,
        "audit_metrics_written_to_manifest": False,
        "model_input_uses_audit_fields": False,
        "source_sha256": source_sha256(),
        "qualification_report_sha256": _sha256_file(qualification_report_path),
        "validity_report_sha256": _sha256_file(validity_report_path),
        "validity_ledger_sha256": _sha256_file(validity_ledger_path),
    }
    if coverage_report["status"] != "passed":
        raise RuntimeError("option-A candidate coverage has empty required cells")

    output_root.mkdir(parents=True, exist_ok=False)
    manifest_text = _jsonl_text(manifest_rows)
    audit_text = _jsonl_text(audit_rows)
    _write_new(output_root / "candidate_manifest.jsonl", manifest_text)
    _write_new(output_root / "candidate_audit.jsonl", audit_text)
    coverage_report["candidate_manifest_sha256"] = _sha256_file(output_root / "candidate_manifest.jsonl")
    coverage_report["candidate_audit_sha256"] = _sha256_file(output_root / "candidate_audit.jsonl")
    _write_new(output_root / "coverage_report.json", json.dumps(coverage_report, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    resolved = {
        "schema_version": "dfcluster.generator_v4.candidate_manifest.v2",
        "protocol": "plan_v4_section_15_option_A",
        "source_plan": "/home/luolie/DF-Cluster/0824思路详细版.md",
        "generator_source_sha256": source_sha256(),
        "candidate_gate": {
            "required_clean_ari": 0.90,
            "required_ari_headroom": 0.20,
            "required_probe_macro_f1": 0.75,
            "finite_shape_metadata_required": True,
        },
        "raw_ari_pools": {
            "easy": [0.50, 0.80],
            "medium": [0.15, 0.50],
            "hard-but-recoverable": [-1.0, 0.15],
        },
        "clm_tertile_scope": "gate_passing_candidates_within_each_raw_ari_pool",
        "clm_tertile_cutpoints": coverage_report["clm_tertile_cutpoints"],
        "full_audit_task_count": EXPECTED_TASK_COUNT,
        "candidate_count": len(manifest_rows),
        "labels_in_model_input": False,
        "audit_metrics_in_model_input": False,
        "synthetic_train_pair_loss_exception": "one_pre_registered_offline_meta_pretraining_loss_only",
        "v3_influence": False,
    }
    _write_new(output_root / "resolved_config.json", json.dumps(resolved, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    return coverage_report