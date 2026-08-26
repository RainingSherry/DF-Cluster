"""Option-A online candidate structural-proposal stream.

The full corrected qualification/validity reference is the candidate gate.
Formal train exposure uses its frozen structural proposal distribution and
fresh train seeds, with cyclic observation/information/raw-ARI/CLM coverage.
It does not rerun expensive per-task validity audits. Generation failures are
retained in an array-free ledger, and the yielded object is a strict X-only
``InputTask`` with no privileged target or labels.
"""

from __future__ import annotations

from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
import hashlib
import itertools
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple

from .core import OBSERVATION_FAMILIES, generate_v4_task, source_sha256
from .full_sampler import FullSamplerConfig, sample_full_task_config
from .input_loader import InputTask, redact_v4_task
from .replay import load_candidate_manifest
from .source_complexity_graph import INFORMATION_STRATA
from .validity import ValidityConfig, compute_validity_certificate

REQUIRED_RAW_POOLS = ("easy", "medium", "hard-but-recoverable")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(partial, path)


def _raw_pool(ari_raw: float) -> str:
    value = float(ari_raw)
    if 0.50 <= value <= 0.80:
        return "easy"
    if 0.15 <= value < 0.50:
        return "medium"
    if value < 0.15:
        return "hard-but-recoverable"
    return "very_easy_above_0.80"


@dataclass(frozen=True)
class CandidateCoveragePolicy:
    coverage_report_path: Path
    source_sha256: str
    required_clean_ari: float = 0.90
    required_headroom: float = 0.20
    required_probe_macro_f1: float = 0.75

    def __post_init__(self) -> None:
        report = json.loads(Path(self.coverage_report_path).read_text(encoding="utf-8"))
        if report.get("status") != "passed":
            raise ValueError("candidate coverage report is not passed")
        if report.get("schema_version") != "dfcluster.generator_v4.candidate_coverage.v2":
            raise ValueError("online stream requires candidate coverage v2 with frozen cutpoints")
        if report.get("source_sha256") != self.source_sha256:
            raise ValueError("coverage report source SHA does not match current generator")
        cutpoints = report.get("clm_tertile_cutpoints")
        if not isinstance(cutpoints, dict) or set(cutpoints) != set(REQUIRED_RAW_POOLS):
            raise ValueError("coverage report lacks all frozen CLM cutpoints")
        for pool in REQUIRED_RAW_POOLS:
            values = cutpoints[pool]
            if len(values) != 2 or not float(values[0]) <= float(values[1]):
                raise ValueError("invalid CLM cutpoints for %s" % pool)

    @property
    def report(self) -> Dict[str, Any]:
        return json.loads(Path(self.coverage_report_path).read_text(encoding="utf-8"))

    @property
    def clm_cutpoints(self) -> Dict[str, Tuple[float, float]]:
        raw = self.report["clm_tertile_cutpoints"]
        return {pool: (float(values[0]), float(values[1])) for pool, values in raw.items()}

    def raw_pool(self, ari_raw: float) -> str:
        return _raw_pool(ari_raw)

    def clm_tertile(self, raw_pool: str, clm_observed: float) -> Optional[int]:
        if raw_pool not in REQUIRED_RAW_POOLS:
            return None
        q1, q2 = self.clm_cutpoints[raw_pool]
        value = float(clm_observed)
        return 0 if value < q1 else 1 if value < q2 else 2

    def cell_schedule(self) -> Tuple[Tuple[str, str, str, int], ...]:
        return tuple(
            (observation, information, pool, tertile)
            for observation in OBSERVATION_FAMILIES
            for information in INFORMATION_STRATA
            for pool in REQUIRED_RAW_POOLS
            for tertile in range(3)
        )


def _fresh_train_seed(training_seed: int, attempt_index: int) -> int:
    raw = f"{int(training_seed)}:{int(attempt_index)}".encode("ascii")
    return int(hashlib.sha256(raw).hexdigest()[:16], 16) % (2**63 - 1)


def _online_candidate_spec_from_proposal(
    training_seed: int,
    attempt_index: int,
    proposal: Mapping[str, Any],
    sampler: FullSamplerConfig,
) -> Any:
    """Rebuild a qualification structural proposal with a fresh train seed."""

    base = sample_full_task_config(
        generator_seed=int(proposal["generator_seed"]),
        task_index=int(proposal["schedule_task_index"]),
        split="qualification",
        observation_stratum=str(proposal["observation_stratum"]),
        sampler=sampler,
    )
    return replace(
        base,
        information_stratum=str(proposal["information_stratum"]),
        split="train",
        task_index=int(attempt_index * len(OBSERVATION_FAMILIES) + OBSERVATION_FAMILIES.index(str(proposal["observation_stratum"]))),
        seed=_fresh_train_seed(training_seed, attempt_index),
    )


def _online_attempt(job: Tuple[int, int, Mapping[str, Any], FullSamplerConfig]) -> Tuple[int, Optional[InputTask], Dict[str, Any], Optional[Dict[str, str]]]:
    training_seed, attempt_index, proposal, sampler = job
    try:
        spec = _online_candidate_spec_from_proposal(training_seed, attempt_index, proposal, sampler)
        task = generate_v4_task(spec)
        input_task = redact_v4_task(task, safe_metadata={
            "observation_stratum": proposal.get("observation_stratum"),
            "information_stratum": proposal.get("information_stratum"),
        })
        return attempt_index, input_task, dict(proposal), None
    except Exception as exc:
        return attempt_index, None, dict(proposal), {"type": type(exc).__name__, "message": str(exc)}


class OnlineCandidateStream:
    """Fresh-task structural-proposal stream for the formal exposure target."""

    def __init__(
        self,
        *,
        policy: CandidateCoveragePolicy,
        output_root: Path,
        training_seed: int,
        task_exposure_target: int = 5_000_000,
        sampler: FullSamplerConfig | None = None,
        validity: ValidityConfig | None = None,
        proposal_manifest_path: Path | None = None,
        cpu_workers: int = 16,
        prefetch_attempts: int = 32,
    ) -> None:
        if task_exposure_target != 5_000_000:
            raise ValueError("formal online candidate stream target is frozen at 5,000,000 tasks")
        self.policy = policy
        self.output_root = Path(output_root)
        if self.output_root.exists():
            raise FileExistsError(f"online candidate output already exists: {self.output_root}")
        self.output_root.mkdir(parents=True, exist_ok=False)
        self.training_seed = int(training_seed)
        self.task_exposure_target = int(task_exposure_target)
        self.sampler = sampler or FullSamplerConfig()
        self.validity = validity or ValidityConfig()
        if cpu_workers < 16 or cpu_workers > 64:
            raise ValueError("online admission CPU workers must be within the approved [16,64] resource range")
        if prefetch_attempts <= 0:
            raise ValueError("prefetch_attempts must be positive")
        self.cpu_workers = int(cpu_workers)
        self.prefetch_attempts = int(prefetch_attempts)
        self.sampler.validate()
        self.validity.validate()
        if proposal_manifest_path is None:
            raise ValueError("formal online stream requires a frozen candidate structural proposal manifest")
        self.proposal_manifest_path = Path(proposal_manifest_path)
        self.proposal_rows = load_candidate_manifest(self.proposal_manifest_path)
        self.source_sha = source_sha256()
        self.proposals_by_cell: Dict[Tuple[str, str, str, int], List[Dict[str, Any]]] = defaultdict(list)
        for proposal in self.proposal_rows:
            cell = (
                str(proposal["observation_stratum"]),
                str(proposal["information_stratum"]),
                str(proposal["raw_difficulty_pool"]),
                int(proposal["clm_tertile"]),
            )
            self.proposals_by_cell[cell].append(dict(proposal))
        missing_cells = [cell for cell in self.policy.cell_schedule() if not self.proposals_by_cell.get(cell)]
        if missing_cells:
            raise ValueError("proposal manifest has empty coverage cells")
        self.rejection_path = self.output_root / "rejection_ledger.jsonl"
        self.accepted_path = self.output_root / "accepted_ledger.jsonl"
        _atomic_json(self.output_root / "resolved_config.json", {
            "schema_version": "dfcluster.generator_v4.online_candidate_stream.v2",
            "protocol": "plan_v4_section_15_option_A",
            "admission_mode": "frozen_candidate_structural_proposal_distribution",
            "training_seed": self.training_seed,
            "task_exposure_target": self.task_exposure_target,
            "cpu_workers": self.cpu_workers,
            "prefetch_attempts": self.prefetch_attempts,
            "sampler": self.sampler.__dict__,
            "validity": self.validity.__dict__,
            "source_sha256": self.source_sha,
            "coverage_report_sha256": _sha256_file(policy.coverage_report_path),
            "proposal_manifest_path": str(self.proposal_manifest_path),
            "proposal_manifest_sha256": _sha256_file(self.proposal_manifest_path),
            "proposal_mode": "frozen_candidate_structural_config_plus_fresh_train_seed",
            "candidate_gate_reference_frozen": True,
            "per_task_validity_reaudit": False,
            "labels_opened_in_online_stream": False,
            "labels_in_model_input": False,
            "performance_claim": False,
        })
        _atomic_json(self.output_root / "status.json", {"status": "initialized", "accepted": 0, "attempts": 0})

    def _write_row(self, path: Path, row: Mapping[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")

    def __iter__(self) -> Iterator[InputTask]:
        cells = self.policy.cell_schedule()
        attempts = 0
        accepted = 0
        buffers: Dict[Tuple[str, str, str, int], deque[Tuple[InputTask, Dict[str, Any]]]] = defaultdict(deque)
        try:
            with ProcessPoolExecutor(max_workers=self.cpu_workers) as executor:
                for accepted_index in range(self.task_exposure_target):
                    target_cell = cells[accepted_index % len(cells)]
                    observation, information, target_pool, target_tertile = target_cell
                    while not buffers[target_cell]:
                        proposal_rows = self.proposals_by_cell[target_cell]
                        jobs = [
                            (
                                self.training_seed,
                                attempts + offset,
                                proposal_rows[(attempts + offset) % len(proposal_rows)],
                                self.sampler,
                            )
                            for offset in range(self.prefetch_attempts)
                        ]
                        attempts += len(jobs)
                        for attempt_index, task, proposal, error in executor.map(_online_attempt, jobs, chunksize=1):
                            if error is not None or task is None:
                                self._write_row(self.rejection_path, {
                                    "attempt_index": int(attempt_index),
                                    "accepted_index": accepted_index,
                                    "proposal_task_id": proposal.get("task_id"),
                                    "target_cell": list(target_cell),
                                    "status": "generation_failure",
                                    "error": error,
                                    "labels_opened_by_stream": False,
                                })
                                continue
                            row = {
                                "attempt_index": int(attempt_index),
                                "accepted_index": accepted_index,
                                "task_id": task.task_id,
                                "proposal_task_id": proposal.get("task_id"),
                                "observation_stratum": observation,
                                "information_stratum": information,
                                "raw_difficulty_pool": target_pool,
                                "clm_tertile": target_tertile,
                                "status": "accepted_structural_proposal",
                                "labels_opened_by_stream": False,
                                "labels_in_model_input": False,
                                "per_task_validity_reaudit": False,
                            }
                            buffers[target_cell].append((task, row))
                    task, row = buffers[target_cell].popleft()
                    row["accepted_index"] = accepted_index
                    self._write_row(self.accepted_path, row)
                    accepted += 1
                    _atomic_json(self.output_root / "status.json", {
                        "status": "running", "accepted": accepted, "attempts": attempts,
                        "target": self.task_exposure_target, "cpu_workers": self.cpu_workers,
                        "admission_mode": "frozen_candidate_structural_proposal_distribution",
                    })
                    yield replace(
                        task,
                    )
            _atomic_json(self.output_root / "status.json", {
                "status": "completed", "accepted": accepted, "attempts": attempts,
                "target": self.task_exposure_target, "source_sha256": self.source_sha,
                "admission_mode": "frozen_candidate_structural_proposal_distribution",
            })
        except Exception as exc:
            _atomic_json(self.output_root / "status.json", {
                "status": "incomplete_compute", "accepted": accepted, "attempts": attempts,
                "target": self.task_exposure_target, "error": "%s: %s" % (type(exc).__name__, exc),
            })
            raise

