import json
from pathlib import Path

import pytest
import torch

from geometry_probe.phase_runner import (
    PhaseConfig,
    _device_indices,
    _read_phase1_gate,
    _validate_config,
    run_phase,
)


def _tiny_config(**overrides) -> PhaseConfig:
    values = dict(
        total_updates=1,
        checkpoint_steps=(1,),
        eval_task_count=1,
        num_gaussians=3,
        min_points=4,
        max_points=5,
        dim=6,
        categorical_chance=0.0,
        max_blocks=1,
        embed_dim=16,
        n_head=4,
        hid_dim=32,
        n_layers=1,
        knn_k=3,
        device="cpu",
        formal=False,
    )
    values.update(overrides)
    return PhaseConfig(**values)


def test_phase_config_requires_final_checkpoint() -> None:
    with pytest.raises(ValueError, match="final checkpoint"):
        _validate_config(_tiny_config(total_updates=2, checkpoint_steps=(1,)))


def test_formal_phase1_uses_requested_checkpoints() -> None:
    config = PhaseConfig()
    _validate_config(config)
    assert config.checkpoint_steps == (5_000, 20_000, 50_000, 75_000, 90_000, 100_000)


def test_phase2_requires_three_seeds_and_gate() -> None:
    with pytest.raises(ValueError, match="exactly 3"):
        _validate_config(_tiny_config(phase="phase2", total_updates=1, checkpoint_steps=(1,)))


def test_duplicate_seeds_are_rejected() -> None:
    with pytest.raises(ValueError, match="unique"):
        _validate_config(_tiny_config(phase="phase2", seeds=(42, 42, 43)))


def test_phase1_gate_resolves_report_relative_to_gate(tmp_path: Path) -> None:
    report = tmp_path / "phase1.json"
    expected_steps = (5_000, 20_000, 50_000, 75_000, 90_000, 100_000)
    arms = {}
    for arm in ("zeus", "geometry"):
        checkpoints = []
        for step in expected_steps:
            checkpoint = tmp_path / f"{arm}_{step}.pt"
            metrics = tmp_path / f"{arm}_{step}.json"
            checkpoint.write_bytes(b"checkpoint")
            metrics.write_text("{}", encoding="utf-8")
            checkpoints.append({"step": step, "checkpoint": checkpoint.name, "metrics": metrics.name})
        arms[arm] = {"checkpoints": checkpoints}
    report.write_text(
        json.dumps(
            {
                "status": "complete",
                "phase": "phase1",
                "config": {
                    "formal": True,
                    "total_updates": 100_000,
                    "checkpoint_steps": list(expected_steps),
                    "eval_task_count": 100,
                    "seeds": [42],
                },
                "arms": {"42": arms},
                "evaluation_manifest": {"task_count_per_mode": 100},
            }
        ),
        encoding="utf-8",
    )
    gate = tmp_path / "gate.json"
    gate.write_text(
        json.dumps({"proceed": True, "phase1_report": "phase1.json"}), encoding="utf-8"
    )
    resolved = _read_phase1_gate(str(gate))
    assert resolved["phase1_report"] == str(report.resolve())


def test_device_rule_uses_physical_index_when_visible_devices_remap(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,3")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    assert _device_indices("cuda:0") == (0, 1)
    assert _device_indices("cuda:1") == (1, 3)


def test_phase_runner_writes_paired_stream_artifacts(tmp_path: Path) -> None:
    report = run_phase(_tiny_config(), tmp_path)
    assert report["status"] == "complete"
    assert report["errors"] == []
    assert "aggregate_summary" in report
    seed_result = report["arms"]["42"]
    assert seed_result["zeus"]["training_task_sha256"] == seed_result["geometry"]["training_task_sha256"]
    assert seed_result["zeus"]["checkpoints"][0]["step"] == 1
    assert (tmp_path / "evaluation_manifest.json").is_file()
    assert (tmp_path / "errors.json").is_file()


def test_phase_runner_can_resume_from_a_completed_checkpoint(tmp_path: Path) -> None:
    source = tmp_path / "source"
    first = run_phase(_tiny_config(total_updates=1, checkpoint_steps=(1,)), source)
    assert first["status"] == "complete"

    resumed = tmp_path / "resumed"
    config = _tiny_config(
        total_updates=2,
        checkpoint_steps=(1, 2),
        resume_from=str(source),
        resume_step=1,
    )
    report = run_phase(config, resumed)
    assert report["status"] == "complete"
    assert [item["step"] for item in report["arms"]["42"]["zeus"]["checkpoints"]] == [1, 2]
    assert report["resume_provenance"]["source_step"] == 1
    history = (resumed / "runs" / "seed_42" / "zeus" / "training_history.jsonl").read_text()
    assert [json.loads(line)["update"] for line in history.splitlines()] == [1, 2]


def test_phase1_gate_rejects_nonformal_smoke_report(tmp_path: Path) -> None:
    source = tmp_path / "smoke"
    report = run_phase(_tiny_config(), source)
    assert report["status"] == "complete"
    gate = tmp_path / "gate.json"
    gate.write_text(
        json.dumps({"proceed": True, "phase1_report": str(source / "report.json")}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="frozen formal protocol"):
        _read_phase1_gate(str(gate))
