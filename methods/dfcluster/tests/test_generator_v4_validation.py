from pathlib import Path

from methods.dfcluster.generator_v4 import build_stage_a_validation_manifest, iter_validation_inputs


def test_fixed_validation_manifest_and_redacted_worker():
    manifest = Path("/data/luolie/DF-Cluster/data/generator_v4/manifests/v4_stage_a_validation_v3/validation_manifest.jsonl")
    rows = manifest.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 23903
    task = next(iter_validation_inputs(manifest, cpu_workers=1))
    assert not hasattr(task, "labels")
    assert not hasattr(task, "clean_latent")
    assert task.inference_payload()["model_input"]["features"].ndim == 2
