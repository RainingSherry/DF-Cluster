"""Label-isolated Cluster Label Matching audit for generated tasks.

The canonical per-task CLM value is the official adjusted Calinski-Harabasz
measure (``CH_A``) at the frozen upstream commit.  It is computed after one
fixed preprocessing path and must never be exposed to a training loader.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib
from pathlib import Path
import sys
from typing import Any, Callable, Protocol
import warnings

import numpy as np



CLM_COMMIT = "927927f8e3cd8874fef056d06defbee215a55173"
CLM_MEASURE = "CH_A"
CLM_LOGISTIC_K = 4.432010535838295
CLM_PREPROCESSING = (
    "mask-aware column median imputation; numeric-code representation for "
    "quantized/categorical columns; per-column mean/std standardization; "
    "constant columns mapped to zero"
)


class _CLMAuditTask(Protocol):
    """Minimal task view accepted by the isolated CLM audit."""

    features: np.ndarray
    missing_mask: np.ndarray
    labels: np.ndarray


@dataclass(frozen=True)
class CLMAuditResult:
    observed: float | None
    clean_control: float | None
    status: str
    error: str | None
    measure: str = CLM_MEASURE
    upstream_commit: str = CLM_COMMIT
    logistic_k: float = CLM_LOGISTIC_K
    preprocessing: str = CLM_PREPROCESSING

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def prepare_clm_matrix(
    values: np.ndarray,
    missing_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Apply the one frozen, deterministic representation used by CH_A."""

    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("CLM input must be a two-dimensional matrix")
    if matrix.shape[0] < 2 or matrix.shape[1] < 1:
        raise ValueError("CLM input matrix is too small")
    if missing_mask is None:
        mask = np.zeros(matrix.shape, dtype=bool)
    else:
        mask = np.asarray(missing_mask, dtype=bool)
        if mask.shape != matrix.shape:
            raise ValueError("missing mask must match the value matrix")

    prepared = matrix.copy()
    for column in range(prepared.shape[1]):
        observed = prepared[~mask[:, column], column]
        observed = observed[np.isfinite(observed)]
        fill = float(np.median(observed)) if observed.size else 0.0
        invalid = mask[:, column] | ~np.isfinite(prepared[:, column])
        prepared[invalid, column] = fill

    means = prepared.mean(axis=0, keepdims=True)
    stds = prepared.std(axis=0, keepdims=True)
    stds = np.where(stds > 1e-8, stds, 1.0)
    prepared = (prepared - means) / stds
    if not np.isfinite(prepared).all():
        raise RuntimeError("CLM preprocessing produced non-finite values")
    return np.ascontiguousarray(prepared, dtype=np.float64)


def load_official_cha(
    clm_repository: str | Path,
) -> Callable[[np.ndarray, np.ndarray, float], float]:
    """Load CH_A from the frozen external repository without vendoring it."""

    repository = Path(clm_repository).resolve()
    source = repository / "measures" / "calinski_harabasz.py"
    if not source.is_file():
        raise FileNotFoundError(f"official CLM source not found: {source}")
    repository_text = str(repository)
    if repository_text not in sys.path:
        sys.path.insert(0, repository_text)
    module = importlib.import_module("measures.calinski_harabasz")
    loaded_source = Path(module.__file__).resolve()
    if repository not in loaded_source.parents:
        raise RuntimeError(
            f"loaded CLM from unexpected location: {loaded_source}"
        )
    return module.calinski_harabasz_adjusted


def compute_clm_audit(
    task: _CLMAuditTask,
    clm_repository: str | Path,
) -> CLMAuditResult:
    """Compute observed CH_A and a clean-prior positive control.

    ``observed`` is the corresponding CLM value delivered for each task.
    ``clean_control`` is explicitly a generator sanity check, not a training
    label, difficulty controller, or model-selection signal.
    """

    try:
        official_cha = load_official_cha(clm_repository)
        observed_matrix = prepare_clm_matrix(task.features, task.missing_mask)
        clean_values = getattr(task, "clean_signal", getattr(task, "clean_latent", None))
        if clean_values is None:
            raise ValueError("CLM audit task lacks clean_signal/clean_latent")
        clean_matrix = prepare_clm_matrix(clean_values)
        labels = np.asarray(task.labels, dtype=np.int32)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            observed = float(official_cha(observed_matrix, labels, CLM_LOGISTIC_K))
            clean = float(official_cha(clean_matrix, labels, CLM_LOGISTIC_K))
        if not np.isfinite(observed) or not np.isfinite(clean):
            raise FloatingPointError("official CH_A returned a non-finite value")
        return CLMAuditResult(
            observed=observed,
            clean_control=clean,
            status="complete",
            error=None,
        )
    except Exception as exc:  # audit failures are recorded, then gated in aggregate
        return CLMAuditResult(
            observed=None,
            clean_control=None,
            status="incomplete_clm",
            error=f"{type(exc).__name__}: {exc}",
        )

