"""Audited Generator V4 P0/P1 public API."""

from .core import (
    CLEAN_FAMILIES,
    FEATURE_TYPE_NAMES,
    MISSINGNESS_MODES,
    INFORMATION_STRATA,
    OBSERVATION_FAMILIES,
    ObservationGraphSpec,
    ObservationResult,
    V4Config,
    V4Task,
    build_observation_graph,
    centered_normalized_gram,
    generate_observation,
    generate_v4_task,
    source_sha256,
    validate_training_payload,
)

__all__ = [
    "CLEAN_FAMILIES",
    "FEATURE_TYPE_NAMES",
    "MISSINGNESS_MODES",
    "INFORMATION_STRATA",
    "OBSERVATION_FAMILIES",
    "ObservationGraphSpec",
    "ObservationResult",
    "V4Config",
    "V4Task",
    "build_observation_graph",
    "centered_normalized_gram",
    "generate_observation",
    "generate_v4_task",
    "source_sha256",
    "validate_training_payload",
]

from .qualification import QualificationConfig, run_qualification

__all__ += ["QualificationConfig", "run_qualification"]

from .models import (
    DDBMConfig,
    DatasetContextDDBM,
    GlobalAE,
    GlobalAEConfig,
    GlobalAEOutput,
    centered_normalized_gram_torch,
    ddbm_geometry_loss,
    mixed_type_reconstruction_loss,
)

__all__ += [
    "DDBMConfig",
    "DatasetContextDDBM",
    "GlobalAE",
    "GlobalAEConfig",
    "GlobalAEOutput",
    "centered_normalized_gram_torch",
    "ddbm_geometry_loss",
    "mixed_type_reconstruction_loss",
]

from .validity import ValidityCertificate, ValidityConfig, compute_validity_certificate, run_validity_audit

__all__ += [
    "ValidityCertificate",
    "ValidityConfig",
    "compute_validity_certificate",
    "run_validity_audit",
]

from .storage import (
    validate_audit_artifact,
    validate_input_artifact,
    validate_target_artifact,
    write_task_artifacts,
)

__all__ += [
    "validate_audit_artifact",
    "validate_input_artifact",
    "validate_target_artifact",
    "write_task_artifacts",
]

from .objective import v4_train_step

__all__ += ["v4_train_step"]

from .counterfactual import generate_counterfactual_tasks

__all__ += ["generate_counterfactual_tasks"]

from .controls import ImpossibleControl, make_impossible_row_misalignment

__all__ += ["ImpossibleControl", "make_impossible_row_misalignment"]

from .full_qualification import FullQualificationConfig, run_full_qualification
from .full_sampler import FullSamplerConfig, sample_full_task_config

__all__ += [
    "FullQualificationConfig",
    "FullSamplerConfig",
    "run_full_qualification",
    "sample_full_task_config",
]

from .full_validity import FullValidityConfig, run_full_validity

__all__ += ["FullValidityConfig", "run_full_validity"]

from .candidate_manifest import build_candidate_manifest

__all__ += ["build_candidate_manifest"]

from .training import StageAConfig, StageACorruptionConfig, StageATrainer, CorruptedTable, corrupt_table, implementation_sha256

__all__ += ["StageAConfig", "StageACorruptionConfig", "StageATrainer", "CorruptedTable", "corrupt_table", "implementation_sha256"]

from .replay import CandidateReplayConfig, candidate_cell_rows, iter_candidate_replay, load_candidate_manifest, replay_candidate_task

__all__ += ["CandidateReplayConfig", "candidate_cell_rows", "iter_candidate_replay", "load_candidate_manifest", "replay_candidate_task"]

from .stage_b import StageBConfig, StageBTrainer, StageBTrainingConfig, apply_row_permutation, coassignment_pair_loss, ddbm_stage_b_loss, freeze_ae_for_stage_b, inverse_row_permutation, row_permutation, stage_b_train_step

__all__ += ["StageBConfig", "StageBTrainer", "StageBTrainingConfig", "apply_row_permutation", "coassignment_pair_loss", "ddbm_stage_b_loss", "freeze_ae_for_stage_b", "inverse_row_permutation", "row_permutation", "stage_b_train_step"]

from .online import CandidateCoveragePolicy, OnlineCandidateStream

__all__ += ["CandidateCoveragePolicy", "OnlineCandidateStream"]

from .preflight import PreflightConfig, run_preflight

__all__ += ["PreflightConfig", "run_preflight"]

from .input_loader import InputTask, PrivilegedTarget, redact_v4_task, redact_v4_target

__all__ += ["InputTask", "PrivilegedTarget", "redact_v4_task", "redact_v4_target"]

from .validation import VALIDATION_SEED, build_stage_a_validation_manifest, evaluate_stage_a_validation, iter_validation_inputs

__all__ += ["VALIDATION_SEED", "build_stage_a_validation_manifest", "evaluate_stage_a_validation", "iter_validation_inputs"]
