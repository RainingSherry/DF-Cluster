from pathlib import Path

import numpy as np

from methods.dfcluster.generator_v4 import (
    CandidateReplayConfig,
    candidate_cell_rows,
    iter_candidate_replay,
    load_candidate_manifest,
    replay_candidate_task,
)


MANIFEST = Path("/data/luolie/DF-Cluster/data/generator_v4/manifests/v4_candidate_pool_v2/candidate_manifest.jsonl")


def test_candidate_manifest_replay_is_identity_stable():
    rows = load_candidate_manifest(MANIFEST)
    assert len(rows) == 23_903
    row = rows[0]
    task = replay_candidate_task(row, expected_source_sha256=row["source_sha256"])
    assert task.metadata["task_id"] == row["task_id"]
    payload = task.training_payload()
    assert set(payload["model_input"]) == {"features", "missing_mask", "feature_types"}
    assert "labels" not in payload
    assert np.isfinite(payload["model_input"]["features"]).all()


def test_candidate_cell_index_is_nonempty_and_deterministic():
    rows = load_candidate_manifest(MANIFEST)
    first = candidate_cell_rows(
        rows,
        observation_stratum="flow_analytic",
        information_stratum="preserving",
        raw_difficulty_pool="hard-but-recoverable",
        clm_tertile=0,
    )
    second = candidate_cell_rows(
        rows,
        observation_stratum="flow_analytic",
        information_stratum="preserving",
        raw_difficulty_pool="hard-but-recoverable",
        clm_tertile=0,
    )
    assert len(first) == 2
    assert [row["task_id"] for row in first] == [row["task_id"] for row in second]


def test_replay_iterator_returns_payload_and_non_model_metadata():
    rows = load_candidate_manifest(MANIFEST)[:1]
    pairs = list(iter_candidate_replay(CandidateReplayConfig(MANIFEST), rows=rows))
    payload, metadata = pairs[0]
    assert "labels" not in payload
    assert metadata["labels_opened_by_replay_iterator"] is False
    assert metadata["audit_metrics_opened_by_replay_iterator"] is False
