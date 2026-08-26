"""Frozen full-range sampler for the Generator V4 plan.

This sampler implements the task-scale distributions in sections 5--13 of
`0824思路详细版.md`. It does not inspect labels, raw ARI, CLM, model output,
or any audit result when sampling a task.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Dict, Tuple

import numpy as np

from .core import CLEAN_FAMILIES, V4Config
from .source_complexity_graph import INFORMATION_STRATA, OBSERVATION_STRATA


SPLIT_SALTS = {"contract": 101, "qualification": 202, "train": 303, "validation": 404, "test": 505}


@dataclass(frozen=True)
class FullSamplerConfig:
    version: int = 2
    n_samples_range: Tuple[int, int] = (128, 2048)
    n_features_range: Tuple[int, int] = (10, 160)
    n_clusters_range: Tuple[int, int] = (2, 20)
    intrinsic_dim_range: Tuple[int, int] = (2, 128)
    missing_rates: Tuple[float, ...] = (0.0, 0.05, 0.10, 0.20, 0.30)
    mixed_type_target_rates: Tuple[float, ...] = (0.0, 0.10, 0.25, 0.50)
    observation_strata: Tuple[str, ...] = OBSERVATION_STRATA
    information_strata: Tuple[str, ...] = INFORMATION_STRATA

    def validate(self) -> None:
        if self.n_samples_range != (128, 2048) or self.n_features_range != (10, 160):
            raise ValueError("full sampler must use planned N/D ranges")
        if self.n_clusters_range != (2, 20) or self.intrinsic_dim_range != (2, 128):
            raise ValueError("full sampler must use planned K/d_int ranges")
        if tuple(self.missing_rates) != (0.0, 0.05, 0.10, 0.20, 0.30):
            raise ValueError("full sampler missing-rate grid is frozen by plan")
        if tuple(self.mixed_type_target_rates) != (0.0, 0.10, 0.25, 0.50):
            raise ValueError("full sampler mixed-type grid is frozen by plan")
        if tuple(self.observation_strata) != OBSERVATION_STRATA:
            raise ValueError("full sampler observation strata do not match active source-complexity generator")
        if tuple(self.information_strata) != INFORMATION_STRATA:
            raise ValueError("full sampler information strata do not match the frozen plan")


def _log_uniform_int(rng: np.random.Generator, low: int, high: int) -> int:
    value = int(round(math.exp(rng.uniform(math.log(low), math.log(high)))))
    return int(np.clip(value, low, high))


def _intrinsic_dim(rng: np.random.Generator) -> int:
    values = np.arange(2, 129, dtype=np.int64)
    weights = 1.0 / np.sqrt(values.astype(np.float64))
    weights /= weights.sum()
    return int(rng.choice(values, p=weights))


def _mixed_fractions(rng: np.random.Generator) -> tuple[float, float, float, float]:
    total = float(rng.choice(np.asarray((0.0, 0.10, 0.25, 0.50))))
    if total == 0.0:
        return 0.0, 0.0, 0.0, 0.0
    parts = rng.dirichlet(np.ones(4)) * total
    return tuple(float(value) for value in parts)


def _boundary_override(task_index: int) -> dict[str, int] | None:
    """Force planned envelope boundaries inside every frozen schedule cycle."""

    boundary = task_index % 64
    if boundary == 0:
        return {"n_samples": 128, "n_features": 10, "n_clusters": 2, "intrinsic_dim": 2}
    if boundary == 1:
        return {"n_samples": 2048, "n_features": 160, "n_clusters": 20, "intrinsic_dim": 128}
    return None


def sample_full_task_config(
    *,
    generator_seed: int,
    task_index: int,
    split: str,
    observation_stratum: str,
    sampler: FullSamplerConfig | None = None,
) -> V4Config:
    """Return a fully resolved task without opening labels or audit sidecars."""

    sampler = sampler or FullSamplerConfig()
    sampler.validate()
    if split not in SPLIT_SALTS:
        raise ValueError("unknown full-sampler split")
    if observation_stratum not in sampler.observation_strata:
        raise ValueError("unknown source-complexity observation stratum")
    stratum_index = sampler.observation_strata.index(observation_stratum)
    rng = np.random.default_rng(np.random.SeedSequence([generator_seed, SPLIT_SALTS[split], task_index, stratum_index]))
    n_samples = _log_uniform_int(rng, *sampler.n_samples_range)
    n_features = _log_uniform_int(rng, *sampler.n_features_range)
    minimum_cluster_size = max(8, int(math.ceil(0.01 * n_samples)))
    maximum_legal_k = min(20, n_samples // minimum_cluster_size)
    if maximum_legal_k < 2:
        raise RuntimeError("planned N range cannot satisfy the minimum cluster-size contract")
    n_clusters = int(rng.integers(2, maximum_legal_k + 1))
    intrinsic_dim = _intrinsic_dim(rng)
    override = _boundary_override(task_index)
    if override:
        n_samples = override["n_samples"]
        n_features = override["n_features"]
        n_clusters = override["n_clusters"]
        intrinsic_dim = override["intrinsic_dim"]
    minimum_cluster_size = max(8, int(math.ceil(0.01 * n_samples)))
    if n_clusters * minimum_cluster_size > n_samples:
        raise RuntimeError("sampled K violates planned minimum cluster-size contract")
    categorical, ordinal, count, bounded = _mixed_fractions(rng)
    # With the frozen three consecutive full-generator seeds and 4,096 tasks
    # per seed/observation family, this cyclic phase makes every information
    # stratum appear exactly 4,096 times per observation family in the full
    # 61,440-task qualification universe.  No label/audit field is inspected.
    information_stratum = sampler.information_strata[(task_index + generator_seed) % len(sampler.information_strata)]
    local_index = task_index * len(sampler.observation_strata) + stratum_index
    return V4Config(
        n_samples=n_samples,
        n_features=n_features,
        n_clusters=n_clusters,
        intrinsic_dim=intrinsic_dim,
        clean_family=CLEAN_FAMILIES[(task_index + stratum_index) % len(CLEAN_FAMILIES)],
        observation_family=observation_stratum,
        information_stratum=information_stratum,
        missingness=("mcar", "mar", "mnar")[task_index % 3],
        missing_rate=float(rng.choice(np.asarray(sampler.missing_rates))),
        categorical_fraction=categorical,
        ordinal_fraction=ordinal,
        count_fraction=count,
        bounded_fraction=bounded,
        graph_depth=3,
        max_parents=int(rng.integers(2, 6)),
        nuisance_dim=int(rng.integers(8, 33)),
        noise_scale=float(math.exp(rng.uniform(math.log(0.01), math.log(0.30)))),
        center_scale=float(math.exp(rng.uniform(math.log(1.5), math.log(6.0)))),
        split="qualification" if split in {"contract", "qualification"} else split,
        task_index=local_index,
        seed=int(np.random.SeedSequence([generator_seed, SPLIT_SALTS[split], task_index, stratum_index, 999]).generate_state(1, dtype=np.uint64)[0]),
    )
