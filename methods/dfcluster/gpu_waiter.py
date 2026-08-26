"""Wait for a truly idle legal GPU, then launch one pre-registered command."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
from typing import Any


ALLOWED = {1, 2, 3, 4, 5, 6}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(f"{path}.partial")
    partial.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(partial, path)


def _gpu_rows() -> list[dict[str, Any]]:
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    active = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    active_uuids = {line.strip() for line in active.stdout.splitlines() if line.strip()}
    rows = []
    for parsed in csv.reader(io.StringIO(query.stdout), skipinitialspace=True):
        index, uuid, used, total, utilization = parsed
        rows.append(
            {
                "index": int(index),
                "uuid": uuid,
                "memory_used_mib": int(used),
                "memory_total_mib": int(total),
                "utilization_percent": int(utilization),
                "has_compute_process": uuid in active_uuids,
            }
        )
    return rows


def _candidate(rows: list[dict[str, Any]], memory_limit_mib: int, util_limit: int) -> int | None:
    eligible = [
        row
        for row in rows
        if row["index"] in ALLOWED
        and row["memory_used_mib"] < memory_limit_mib
        and row["utilization_percent"] < util_limit
        and not row["has_compute_process"]
    ]
    if not eligible:
        return None
    eligible.sort(key=lambda row: (row["memory_used_mib"], row["utilization_percent"], row["index"]))
    return int(eligible[0]["index"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status-dir", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--timeout-hours", type=float, default=168.0)
    parser.add_argument("--memory-limit-mib", type=int, default=4096)
    parser.add_argument("--util-limit", type=int, default=10)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("a command is required after --")
    status_dir = args.status_dir.resolve()
    status_dir.mkdir(parents=True, exist_ok=True)
    status_path = status_dir / "launcher_status.json"
    samples_path = status_dir / "gpu_samples.jsonl"
    started = time.time()
    consecutive_gpu: int | None = None
    consecutive_count = 0
    try:
        while time.time() - started < args.timeout_hours * 3600:
            rows = _gpu_rows()
            candidate = _candidate(rows, args.memory_limit_mib, args.util_limit)
            sample = {"timestamp": time.time(), "candidate": candidate, "gpus": rows}
            with samples_path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(sample, sort_keys=True) + "\n")
                stream.flush()
            if candidate is not None and candidate == consecutive_gpu:
                consecutive_count += 1
            elif candidate is not None:
                consecutive_gpu = candidate
                consecutive_count = 1
            else:
                consecutive_gpu = None
                consecutive_count = 0
            _atomic_json(
                status_path,
                {
                    "status": "waiting",
                    "candidate": consecutive_gpu,
                    "consecutive_samples": consecutive_count,
                    "command": command,
                    "started_unix": started,
                    "last_sample_unix": sample["timestamp"],
                },
            )
            if consecutive_gpu is not None and consecutive_count >= 2:
                break
            time.sleep(args.poll_seconds)
        else:
            _atomic_json(
                status_path,
                {
                    "status": "timeout",
                    "started_unix": started,
                    "finished_unix": time.time(),
                    "command": command,
                },
            )
            raise SystemExit(3)

        assert consecutive_gpu in ALLOWED
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(consecutive_gpu)
        # Required by PyTorch deterministic CUDA matmul/einsum kernels.  Set
        # this in the launcher so it is present before the child imports
        # torch or creates a CUDA context.
        environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        launched_command = command + ["--physical-gpu", str(consecutive_gpu)]
        _atomic_json(
            status_path,
            {
                "status": "running",
                "physical_gpu": consecutive_gpu,
                "command": launched_command,
                "cublas_workspace_config": environment["CUBLAS_WORKSPACE_CONFIG"],
                "started_unix": started,
                "launched_unix": time.time(),
            },
        )
        completed = subprocess.run(launched_command, env=environment, check=False)
        _atomic_json(
            status_path,
            {
                "status": "completed" if completed.returncode == 0 else "incomplete_compute",
                "physical_gpu": consecutive_gpu,
                "command": launched_command,
                "cublas_workspace_config": environment["CUBLAS_WORKSPACE_CONFIG"],
                "returncode": completed.returncode,
                "started_unix": started,
                "finished_unix": time.time(),
            },
        )
        raise SystemExit(completed.returncode)
    except BaseException as exc:
        if isinstance(exc, SystemExit):
            raise
        _atomic_json(
            status_path,
            {
                "status": "incomplete_compute",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "started_unix": started,
                "finished_unix": time.time(),
            },
        )
        raise


if __name__ == "__main__":
    main()
