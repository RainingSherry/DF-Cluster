"""Outer-only Generator V4 validity audit.

This module may read synthetic labels only in an isolated audit call. It never
returns labels to a model, writes labels to the task ledger, or selects a task
for training. It reports the pre-registered raw/clean/CLM/recoverability gates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

from ..clm_audit import (
    CLM_LOGISTIC_K,
    load_official_cha,
    prepare_clm_matrix,
)
from .core import V4Task


@dataclass(frozen=True)
class ValidityConfig:
    random_state: int = 20260824
    selection_policy: str = "audit_only"
    n_estimators: int = 256
    kmeans_n_init: int = 20
    kmeans_max_iter: int = 300
    required_clean_ari: float = 0.90
    required_headroom: float = 0.20
    required_probe_macro_f1: float = 0.75
    clm_repository: str = "/home/luolie/DF-Cluster/baseline/external/clm"

    def validate(self) -> None:
        if self.selection_policy != "audit_only":
            raise ValueError("validity audit cannot be used for task selection")
        if self.n_estimators != 256:
            raise ValueError("validity gate requires the frozen n_estimators=256")


@dataclass(frozen=True)
class ValidityCertificate:
    task_id: str
    status: str
    ari_raw: Optional[float]
    ari_clean: Optional[float]
    ari_headroom: Optional[float]
    clm_observed: Optional[float]
    clm_clean: Optional[float]
    clm_headroom: Optional[float]
    probe_macro_f1: Optional[float]
    raw_difficulty_pool: Optional[str]
    clean_finite: bool
    observed_finite: bool
    labels_opened_only_in_audit: bool
    clean_ari_gate: bool
    headroom_gate: bool
    probe_gate: bool
    failure_reasons: Tuple[str, ...]
    error: Optional[str]

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _difficulty_pool(ari_raw: float) -> str:
    if ari_raw >= 0.50:
        return "easy"
    if ari_raw >= 0.15:
        return "medium"
    return "hard-but-recoverable"


def compute_validity_certificate(
    task: V4Task,
    config: ValidityConfig | None = None,
) -> ValidityCertificate:
    """Compute all validity metrics in an isolated, label-bearing audit."""

    config = config or ValidityConfig()
    config.validate()
    try:
        labels = np.asarray(task.labels, dtype=np.int32)
        if labels.ndim != 1 or labels.shape[0] != task.features.shape[0]:
            raise ValueError("invalid audit labels")
        n_clusters = int(np.unique(labels).size)
        raw_matrix = prepare_clm_matrix(task.features, task.missing_mask)
        clean_matrix = np.asarray(task.clean_latent, dtype=np.float64)
        if not np.isfinite(clean_matrix).all():
            raise ValueError("clean geometry contains non-finite values")
        raw_prediction = KMeans(
            n_clusters=n_clusters,
            init="k-means++",
            n_init=config.kmeans_n_init,
            max_iter=config.kmeans_max_iter,
            random_state=config.random_state,
            algorithm="lloyd",
        ).fit_predict(raw_matrix)
        clean_prediction = KMeans(
            n_clusters=n_clusters,
            init="k-means++",
            n_init=config.kmeans_n_init,
            max_iter=config.kmeans_max_iter,
            random_state=config.random_state,
            algorithm="lloyd",
        ).fit_predict(clean_matrix)
        ari_raw = float(adjusted_rand_score(labels, raw_prediction))
        ari_clean = float(adjusted_rand_score(labels, clean_prediction))
        features = np.concatenate(
            [raw_matrix, np.asarray(task.missing_mask, dtype=np.float64)], axis=1
        )
        folds = StratifiedKFold(
            n_splits=5, shuffle=True, random_state=config.random_state
        )
        probe = ExtraTreesClassifier(
            n_estimators=config.n_estimators,
            max_features="sqrt",
            class_weight="balanced",
            random_state=config.random_state,
            n_jobs=1,
        )
        probe_macro_f1 = float(
            cross_val_score(probe, features, labels, cv=folds, scoring="f1_macro", n_jobs=1).mean()
        )
        official_cha = load_official_cha(config.clm_repository)
        observed_clm = float(
            official_cha(
                prepare_clm_matrix(task.features, task.missing_mask),
                labels,
                CLM_LOGISTIC_K,
            )
        )
        clean_clm = float(
            official_cha(
                prepare_clm_matrix(task.clean_latent),
                labels,
                CLM_LOGISTIC_K,
            )
        )
        if not np.isfinite(observed_clm) or not np.isfinite(clean_clm):
            raise FloatingPointError("official CH_A returned a non-finite value")
        headroom = ari_clean - ari_raw
        clm_headroom = clean_clm - observed_clm
        clean_ok = ari_clean >= config.required_clean_ari
        headroom_ok = headroom >= config.required_headroom
        probe_ok = probe_macro_f1 >= config.required_probe_macro_f1
        reasons = []
        if not clean_ok:
            reasons.append("clean_ari_below_threshold")
        if not headroom_ok:
            reasons.append("ari_headroom_below_threshold")
        if not probe_ok:
            reasons.append("probe_macro_f1_below_threshold")
        status = "passed" if not reasons else "failed_gate"
        return ValidityCertificate(
            task_id=task.metadata["task_id"],
            status=status,
            ari_raw=ari_raw,
            ari_clean=ari_clean,
            ari_headroom=float(headroom),
            clm_observed=observed_clm,
            clm_clean=clean_clm,
            clm_headroom=clm_headroom,
            probe_macro_f1=probe_macro_f1,
            raw_difficulty_pool=_difficulty_pool(ari_raw),
            clean_finite=True,
            observed_finite=True,
            labels_opened_only_in_audit=True,
            clean_ari_gate=clean_ok,
            headroom_gate=headroom_ok,
            probe_gate=probe_ok,
            failure_reasons=tuple(reasons),
            error=None,
        )
    except Exception as exc:
        return ValidityCertificate(
            task_id=task.metadata.get("task_id", "unknown"),
            status="incomplete_validity_audit",
            ari_raw=None,
            ari_clean=None,
            ari_headroom=None,
            clm_observed=None,
            clm_clean=None,
            clm_headroom=None,
            probe_macro_f1=None,
            raw_difficulty_pool=None,
            clean_finite=False,
            observed_finite=False,
            labels_opened_only_in_audit=True,
            clean_ari_gate=False,
            headroom_gate=False,
            probe_gate=False,
            failure_reasons=("incomplete_validity_audit",),
            error="%s: %s" % (type(exc).__name__, exc),
        )


def run_validity_audit(
    tasks: Sequence[V4Task],
    config: ValidityConfig | None = None,
    output_root: Path | None = None,
) -> Dict[str, Any]:
    """Audit a frozen in-memory task list and optionally write array-free reports."""

    config = config or ValidityConfig()
    config.validate()
    certificates = [compute_validity_certificate(task, config) for task in tasks]
    report = {
        "schema_version": "dfcluster.generator_v4.validity_audit.v1",
        "status": "passed" if certificates and all(item.status == "passed" for item in certificates) else "failed_gate",
        "performance_claim": False,
        "selection_policy": "audit_only",
        "task_selection_performed": False,
        "task_count": len(certificates),
        "certificate_status_counts": {
            status: sum(item.status == status for item in certificates)
            for status in sorted({item.status for item in certificates})
        },
        "config": asdict(config),
        "labels_opened_only_in_audit": True,
        "labels_written_to_report": False,
        "certificates": [item.as_dict() for item in certificates],
    }
    if output_root is not None:
        output_root = Path(output_root)
        if output_root.exists():
            raise FileExistsError("validity output already exists: %s" % output_root)
        partial = output_root.with_name(output_root.name + ".partial")
        if partial.exists():
            raise FileExistsError("validity partial output already exists: %s" % partial)
        partial.mkdir(parents=True, exist_ok=False)
        (partial / "resolved_config.json").write_text(
            json.dumps(asdict(config), sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        (partial / "report.json").write_text(
            json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        (partial / "status.json").write_text(
            json.dumps({"status": "completed", "stage": "validity_audit"}, indent=2) + "\n",
            encoding="utf-8",
        )
        partial.rename(output_root)
    return report
