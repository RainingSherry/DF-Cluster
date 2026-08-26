"""Audit-only impossible controls for Generator V4."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from .core import V4Task


@dataclass(frozen=True)
class ImpossibleControl:
    """A deliberately misaligned audit control that cannot enter training."""

    features: np.ndarray
    missing_mask: np.ndarray
    feature_types: np.ndarray
    clean_latent: np.ndarray
    labels: np.ndarray
    row_permutation: np.ndarray
    metadata: Dict[str, Any]

    def training_payload(self) -> Dict[str, Any]:
        raise RuntimeError("impossible controls are audit-only and cannot enter training")

    def audit_payload(self) -> Dict[str, Any]:
        return {
            "task_id": self.metadata["task_id"],
            "parent_task_id": self.metadata["parent_task_id"],
            "labels": self.labels,
            "control_kind": self.metadata["control_kind"],
            "audit_only": True,
        }


def make_impossible_row_misalignment(
    task: V4Task,
    *,
    seed: int = 20260825,
) -> ImpossibleControl:
    """Permute observed rows while retaining the clean/label row order."""

    rng = np.random.default_rng(seed)
    permutation = rng.permutation(task.features.shape[0]).astype(np.int64)
    if np.array_equal(permutation, np.arange(permutation.size)):
        permutation = np.roll(permutation, 1)
    metadata = {
        "task_id": task.metadata["task_id"] + "/impossible_row_misalignment",
        "parent_task_id": task.metadata["task_id"],
        "control_kind": "impossible_row_misalignment",
        "audit_only": True,
        "training_forbidden": True,
        "labels_isolated": True,
    }
    return ImpossibleControl(
        features=np.ascontiguousarray(task.features[permutation]),
        missing_mask=np.ascontiguousarray(task.missing_mask[permutation]),
        feature_types=np.ascontiguousarray(task.feature_types),
        clean_latent=np.ascontiguousarray(task.clean_latent),
        labels=np.ascontiguousarray(task.labels),
        row_permutation=permutation,
        metadata=metadata,
    )
