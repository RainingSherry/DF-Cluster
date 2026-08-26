"""Frozen candidate replay stream for V4 validation and replay shards.

The option-A candidate manifest is a qualification-derived admission index. This
module reconstructs exactly the corresponding V4 tasks from generator seed,
observation family and schedule index, verifies identity/source hashes, and
returns the label-free training payload plus non-model manifest metadata.
It is intentionally separate from the future 5M-task online train sampler:
replay rows are finite fixed evidence, not formal training exposure.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from .core import V4Task, generate_v4_task, validate_training_payload
from .full_sampler import FullSamplerConfig, sample_full_task_config


@dataclass(frozen=True)
class CandidateReplayConfig:
    manifest_path: Path
    split: str = "qualification"
    source_sha256: Optional[str] = None

    def validate(self) -> None:
        if self.split not in {"qualification", "validation", "test"}:
            raise ValueError("candidate replay split must be qualification, validation, or test")
        if not Path(self.manifest_path).is_file():
            raise FileNotFoundError(f"candidate manifest not found: {self.manifest_path}")


def load_candidate_manifest(path: Path) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: List[Dict[str, Any]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            task_id = str(row.get("task_id"))
            if task_id in seen:
                raise ValueError(f"duplicate candidate task_id at line {line_number}: {task_id}")
            seen.add(task_id)
            rows.append(row)
    if not rows:
        raise ValueError("candidate manifest is empty")
    return rows


def replay_candidate_task(
    row: Mapping[str, Any],
    *,
    sampler: FullSamplerConfig | None = None,
    expected_source_sha256: Optional[str] = None,
) -> V4Task:
    """Reconstruct one candidate task and verify its frozen identity."""

    required = ("generator_seed", "schedule_task_index", "observation_stratum", "task_id")
    if any(key not in row for key in required):
        raise ValueError("candidate manifest row lacks replay identity fields")
    split = str(row.get("split", "qualification"))
    spec = sample_full_task_config(
        generator_seed=int(row["generator_seed"]),
        task_index=int(row["schedule_task_index"]),
        split=split,
        observation_stratum=str(row["observation_stratum"]),
        sampler=sampler or FullSamplerConfig(),
    )
    task = generate_v4_task(spec)
    if task.metadata["task_id"] != str(row["task_id"]):
        raise ValueError(
            "candidate replay task_id mismatch: expected %s got %s"
            % (row["task_id"], task.metadata["task_id"])
        )
    source_sha = str(task.metadata["source_sha256"])
    if expected_source_sha256 is not None and source_sha != expected_source_sha256:
        raise ValueError("candidate replay source SHA mismatch")
    if row.get("source_sha256") is not None and source_sha != str(row["source_sha256"]):
        raise ValueError("candidate row source SHA mismatch")
    if row.get("information_stratum") is not None and task.metadata.get("information_stratum") != row["information_stratum"]:
        raise ValueError("candidate replay information-stratum mismatch")
    return task


def iter_candidate_replay(
    config: CandidateReplayConfig,
    *,
    sampler: FullSamplerConfig | None = None,
    expected_source_sha256: Optional[str] = None,
    rows: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Iterator[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """Yield `(label-free training payload, manifest metadata)` pairs."""

    config.validate()
    selected = list(rows) if rows is not None else load_candidate_manifest(config.manifest_path)
    for row in selected:
        task = replay_candidate_task(
            row,
            sampler=sampler,
            expected_source_sha256=expected_source_sha256,
        )
        payload = task.training_payload()
        validate_training_payload(payload)
        metadata = {
            key: row[key]
            for key in (
                "task_id", "observation_stratum", "information_stratum",
                "raw_difficulty_pool", "clm_tertile", "generator_seed",
                "schedule_task_index", "n_samples", "n_features", "n_clusters",
                "intrinsic_dim", "missing_rate", "source_sha256",
            )
            if key in row
        }
        metadata["labels_opened_by_replay_iterator"] = False
        metadata["audit_metrics_opened_by_replay_iterator"] = False
        yield payload, metadata


def candidate_cell_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    observation_stratum: str,
    information_stratum: str,
    raw_difficulty_pool: str,
    clm_tertile: int,
) -> List[Dict[str, Any]]:
    """Return deterministic rows for one frozen coverage cell."""

    selected = [
        dict(row)
        for row in rows
        if row.get("observation_stratum") == observation_stratum
        and row.get("information_stratum") == information_stratum
        and row.get("raw_difficulty_pool") == raw_difficulty_pool
        and int(row.get("clm_tertile", -1)) == int(clm_tertile)
    ]
    return sorted(selected, key=lambda row: str(row["task_id"]))