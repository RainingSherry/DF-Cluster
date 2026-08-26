from pathlib import Path

from methods.dfcluster.generator_v4 import CandidateCoveragePolicy


REPORT = Path("/data/luolie/DF-Cluster/data/generator_v4/manifests/v4_candidate_pool_v2/coverage_report.json")
SOURCE = "8f830c96125c67fcd42526fb04a49f236f14005d6bbe27a37f49d04e030d8920"


def test_online_candidate_policy_uses_frozen_cutpoints_and_schedule():
    policy = CandidateCoveragePolicy(REPORT, SOURCE)
    assert len(policy.cell_schedule()) == 135
    assert policy.raw_pool(0.50) == "easy"
    assert policy.raw_pool(0.80) == "easy"
    assert policy.raw_pool(0.15) == "medium"
    assert policy.raw_pool(0.149999) == "hard-but-recoverable"
    assert policy.clm_tertile("easy", 0.0) == 0
    assert policy.clm_tertile("easy", 1.0) == 2


def test_online_stream_yields_strict_input_task(tmp_path: Path):
    from methods.dfcluster.generator_v4 import OnlineCandidateStream
    policy = CandidateCoveragePolicy(REPORT, SOURCE)
    stream = OnlineCandidateStream(
        policy=policy,
        output_root=tmp_path / "stream",
        training_seed=20260824,
        task_exposure_target=5_000_000,
        proposal_manifest_path=Path("/data/luolie/DF-Cluster/data/generator_v4/manifests/v4_candidate_pool_v2/candidate_manifest.jsonl"),
        cpu_workers=16,
        prefetch_attempts=1,
    )
    task = next(iter(stream))
    assert task.__class__.__name__ == "InputTask"
    assert not hasattr(task, "labels")
    assert not hasattr(task, "clean_latent")
