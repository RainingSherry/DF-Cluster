"""Run the fixed, audit-only Generator V4 validity grid.

This runner never filters tasks or emits a training manifest. It writes a
small certificate/rejection report so coverage is measured without using
labels to choose a corpus.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import yaml

from .generator_v4 import ValidityConfig, V4Config, generate_v4_task, run_validity_audit


DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "generator_v4_validity_grid.yaml"
DEFAULT_OUTPUT = Path("/data/luolie/DF-Cluster/outputs/generator_v4/validity_grid_v1")


def run(config_path: Path = DEFAULT_CONFIG, output_root: Path = DEFAULT_OUTPUT) -> Dict[str, Any]:
    if output_root.exists() or output_root.with_name(output_root.name + ".partial").exists():
        raise FileExistsError("validity grid output already exists: %s" % output_root)
    spec = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cases = spec.get("fixed_cases") or []
    if not cases:
        raise ValueError("validity grid has no fixed_cases")
    tasks = []
    case_rows = []
    for index, case in enumerate(cases):
        task = generate_v4_task(
            V4Config(
                n_samples=128,
                n_features=10,
                n_clusters=2,
                intrinsic_dim=8,
                clean_family=str(case["clean_family"]),
                observation_family=str(case["observation_family"]),
                noise_scale=float(case["noise_scale"]),
                missingness=str(case["missingness"]),
                missing_rate=float(case["missing_rate"]),
                split="qualification",
                task_index=index,
                seed=int(case["seed"]),
            )
        )
        tasks.append(task)
        case_rows.append({
            "case_id": str(case["case_id"]),
            "expected_profile": str(case["expected_profile"]),
            "task_id": task.metadata["task_id"],
            "observation_family": str(case["observation_family"]),
            "clean_family": str(case["clean_family"]),
            "noise_scale": float(case["noise_scale"]),
            "missingness": str(case["missingness"]),
            "missing_rate": float(case["missing_rate"]),
            "seed": int(case["seed"]),
            "task_selection_performed": False,
        })
    output_root.mkdir(parents=True, exist_ok=False)
    report = run_validity_audit(
        tasks,
        ValidityConfig(n_estimators=256, selection_policy="audit_only"),
        output_root / "metrics",
    )
    resolved = {
        "config_path": str(config_path),
        "selection_policy": "audit_only",
        "task_selection_performed": False,
        "performance_claim": False,
        "case_count": len(case_rows),
        "cases": case_rows,
        "metrics_report": str(output_root / "metrics" / "report.json"),
    }
    (output_root / "resolved_config.json").write_text(
        json.dumps(resolved, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    (output_root / "report.json").write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    (output_root / "status.json").write_text(
        json.dumps({"status": "completed", "stage": "validity_grid"}, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = run(args.config, args.output_root)
    print(json.dumps({key: report[key] for key in ("status", "task_count", "certificate_status_counts", "selection_policy", "task_selection_performed")}, sort_keys=True))


if __name__ == "__main__":
    main()
