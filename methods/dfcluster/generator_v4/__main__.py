from __future__ import annotations

import argparse
from pathlib import Path

from .qualification import QualificationConfig, run_qualification


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CPU-only DF-Cluster Generator V4 qualification contract")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data/luolie/DF-Cluster/outputs/generator_v4/qualification_p0_p1_v1"),
    )
    parser.add_argument("--tasks-per-observation-family", type=int, default=64)
    args = parser.parse_args()
    report = run_qualification(
        QualificationConfig(tasks_per_observation_family=args.tasks_per_observation_family),
        args.output_root,
    )
    print("status=%s tasks=%d output=%s" % (report["status"], report["completed_task_count"], args.output_root))


if __name__ == "__main__":
    main()
