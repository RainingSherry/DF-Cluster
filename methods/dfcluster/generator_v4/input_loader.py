"""Strict X-only input task and loader boundary for V4.

Objects in this module intentionally have no ``labels``, ``clean_latent``,
``nuisance_roots`` or observation-graph fields. Generator workers may create a
privileged V4Task internally, but they must redact it into InputTask before it
crosses into the Stage-A parent/trainer process.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass(frozen=True)
class InputTask:
    """The only task object allowed to cross into Stage-A."""

    task_id: str
    features: np.ndarray
    missing_mask: np.ndarray
    feature_types: np.ndarray
    row_ids: np.ndarray
    source_sha256: str
    config_sha256: str
    artifact_sha256: str
    observation_stratum: Optional[str] = None
    information_stratum: Optional[str] = None

    def validate(self) -> None:
        values = np.asarray(self.features)
        mask = np.asarray(self.missing_mask)
        types = np.asarray(self.feature_types)
        rows = np.asarray(self.row_ids)
        if values.ndim != 2 or mask.shape != values.shape:
            raise ValueError("InputTask features/missing_mask must match [N,D]")
        if types.shape != (values.shape[1],) or rows.shape != (values.shape[0],):
            raise ValueError("InputTask feature_types/row_ids shape mismatch")
        if not np.isfinite(values).all():
            raise ValueError("InputTask features must be finite")
        if not np.all(values[mask.astype(bool)] == 0.0):
            raise ValueError("InputTask missing cells must use the zero sentinel")
        if not self.task_id or not self.source_sha256 or not self.config_sha256 or not self.artifact_sha256:
            raise ValueError("InputTask lacks immutable identity/provenance")

    def inference_payload(self) -> Dict[str, Any]:
        """Return only the public model input; no privileged target exists here."""

        self.validate()
        return {
            "task_id": self.task_id,
            "model_input": {
                "features": self.features,
                "missing_mask": self.missing_mask,
                "feature_types": self.feature_types,
            },
        }

    def safe_ledger_metadata(self) -> Dict[str, Any]:
        """Return non-audit structural metadata for bookkeeping only."""

        return {
            key: value
            for key, value in {
                "observation_stratum": self.observation_stratum,
                "information_stratum": self.information_stratum,
            }.items()
            if value is not None
        }



@dataclass(frozen=True)
class PrivilegedTarget:
    """Target-side object for Stage B; never passed to the encoder input."""

    task_id: str
    clean_latent: np.ndarray
    labels: np.ndarray
    row_ids: np.ndarray
    source_sha256: str
    config_sha256: str
    artifact_sha256: str
    protocol: str = "offline_meta_pretraining_coassignment_loss_only"

    def validate(self) -> None:
        clean = np.asarray(self.clean_latent)
        labels = np.asarray(self.labels)
        rows = np.asarray(self.row_ids)
        if clean.ndim != 2 or clean.shape[1] != 128:
            raise ValueError("PrivilegedTarget clean_latent must have shape [N,128]")
        if labels.shape != (clean.shape[0],) or rows.shape != (clean.shape[0],):
            raise ValueError("PrivilegedTarget row/label shapes do not match clean_latent")
        if not np.isfinite(clean).all():
            raise ValueError("PrivilegedTarget clean_latent must be finite")
        if self.protocol != "offline_meta_pretraining_coassignment_loss_only":
            raise ValueError("PrivilegedTarget protocol is not approved")
        if not self.task_id or not self.source_sha256 or not self.config_sha256 or not self.artifact_sha256:
            raise ValueError("PrivilegedTarget lacks immutable identity/provenance")


def redact_v4_target(task: Any) -> PrivilegedTarget:
    """Copy only clean geometry and Y for the approved Stage-B pair loss."""

    required = ("clean_latent", "labels", "row_ids", "metadata")
    if any(not hasattr(task, name) for name in required):
        raise TypeError("redact_v4_target requires a privileged V4Task-like object")
    metadata = dict(getattr(task, "metadata"))
    target = PrivilegedTarget(
        task_id=str(metadata["task_id"]),
        clean_latent=np.ascontiguousarray(np.asarray(task.clean_latent).copy()),
        labels=np.ascontiguousarray(np.asarray(task.labels).copy()),
        row_ids=np.ascontiguousarray(np.asarray(task.row_ids).copy()),
        source_sha256=str(metadata["source_sha256"]),
        config_sha256=str(metadata["config_sha256"]),
        artifact_sha256=str(metadata["artifact_sha256"]),
    )
    target.validate()
    return target

def redact_v4_task(task: Any, *, safe_metadata: Optional[Dict[str, Any]] = None) -> InputTask:
    """Copy only X-side arrays/provenance from a privileged V4Task."""

    required = ("features", "missing_mask", "feature_types", "row_ids", "metadata")
    if any(not hasattr(task, name) for name in required):
        raise TypeError("redact_v4_task requires a V4Task-like generated object")
    metadata = dict(getattr(task, "metadata"))
    safe_metadata = safe_metadata or {}
    result = InputTask(
        task_id=str(metadata["task_id"]),
        features=np.ascontiguousarray(np.asarray(task.features).copy()),
        missing_mask=np.ascontiguousarray(np.asarray(task.missing_mask).copy()),
        feature_types=np.ascontiguousarray(np.asarray(task.feature_types).copy()),
        row_ids=np.ascontiguousarray(np.asarray(task.row_ids).copy()),
        source_sha256=str(metadata["source_sha256"]),
        config_sha256=str(metadata["config_sha256"]),
        artifact_sha256=str(metadata["artifact_sha256"]),
        observation_stratum=safe_metadata.get("observation_stratum"),
        information_stratum=safe_metadata.get("information_stratum"),
    )
    result.validate()
    return result