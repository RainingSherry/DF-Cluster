# ⚠️  DEPRECATED: Part of terminated V1/V2/V3 research line. See DEPRECATION_NOTICE.md
"""Frozen paired-view generator for the dfhybrid_v2 necessity test.

This is deliberately a restricted, known forward family.  It does not claim
generic nonlinear-ICA identifiability or arbitrary tabular recovery.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from typing import Any, Literal

import numpy as np


Split = Literal["smoke", "train", "validation", "test"]
N_VALUES = np.asarray([256, 512, 1024], dtype=np.int64)
D_VALUES = np.asarray([8, 16, 32], dtype=np.int64)
SPLIT_OFFSETS = {
    "smoke": 0x1A2B3C4D,
    "train": 0x31415926,
    "validation": 0x27182818,
    "test": 0x16180339,
}


@dataclass(frozen=True)
class RecoverableV2Config:
    name: str = "dfhybrid_v2_redundant_views"
    protocol_version: int = 1
    base_seed: int = 20260824
    minimum_cluster_size: int = 16
    max_clusters: int = 8
    center_pair_distance: float = 6.0
    cluster_std: float = 0.6
    cubic_strength: float = 0.75
    additive_noise_std: float = 0.02
    both_view_fraction: float = 0.25
    view_one_only_fraction: float = 0.375
    view_two_only_fraction: float = 0.375

    def validate(self) -> None:
        if not math.isclose(
            self.both_view_fraction
            + self.view_one_only_fraction
            + self.view_two_only_fraction,
            1.0,
        ):
            raise ValueError("view-pattern probabilities must sum to one")
        if self.minimum_cluster_size < 2 or self.max_clusters < 2:
            raise ValueError("invalid clean-mixture cluster bounds")
        if self.center_pair_distance <= 0 or self.cluster_std <= 0:
            raise ValueError("clean geometry scales must be positive")
        if self.cubic_strength <= 0 or self.additive_noise_std < 0:
            raise ValueError("invalid observation parameters")


@dataclass(frozen=True)
class RecoverableV2Task:
    task_id: str
    features: np.ndarray
    clean_signal: np.ndarray
    missing_mask: np.ndarray
    feature_types: np.ndarray
    labels: np.ndarray
    oracle_recovered: np.ndarray
    metadata: dict[str, Any]

    def training_payload(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "features": self.features,
            "missing_mask": self.missing_mask,
            "feature_types": self.feature_types,
        }

    def audit_payload(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "labels": self.labels,
            "metadata": self.metadata,
        }


def seed_for_recoverable_v2_task(
    task_index: int,
    split: Split,
    config: RecoverableV2Config | None = None,
) -> int:
    """Return the frozen, split-isolated RNG seed without generating arrays."""

    config = config or RecoverableV2Config()
    sequence = np.random.SeedSequence(
        [config.base_seed, SPLIT_OFFSETS[split], int(task_index)]
    )
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def _orthogonal(size: int, rng: np.random.Generator) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(size, size)))
    signs = np.where(np.diag(r) < 0, -1.0, 1.0)
    return q * signs[np.newaxis, :]


def _simplex_centers(
    n_clusters: int,
    dimension: int,
    pair_distance: float,
    rng: np.random.Generator,
) -> np.ndarray:
    centering = np.eye(n_clusters) - np.ones((n_clusters, n_clusters)) / n_clusters
    eigenvalues, eigenvectors = np.linalg.eigh(centering)
    coordinates = eigenvectors[:, eigenvalues > 0.5]
    coordinates *= pair_distance / math.sqrt(2.0)
    padded = np.zeros((n_clusters, dimension), dtype=np.float64)
    padded[:, : coordinates.shape[1]] = coordinates
    return padded @ _orthogonal(dimension, rng)


def _balanced_counts(
    n_samples: int,
    n_clusters: int,
    minimum: int,
    rng: np.random.Generator,
) -> np.ndarray:
    counts = np.full(n_clusters, minimum, dtype=np.int64)
    remainder = n_samples - int(counts.sum())
    probabilities = rng.dirichlet(np.full(n_clusters, 5.0))
    return counts + rng.multinomial(remainder, probabilities)


def _clean_mixture(
    n_samples: int,
    dimension: int,
    n_clusters: int,
    config: RecoverableV2Config,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    counts = _balanced_counts(
        n_samples, n_clusters, config.minimum_cluster_size, rng
    )
    labels = np.repeat(np.arange(n_clusters, dtype=np.int16), counts)
    rng.shuffle(labels)
    centers = _simplex_centers(
        n_clusters, dimension, config.center_pair_distance, rng
    )
    clean = centers[labels] + config.cluster_std * rng.normal(
        size=(n_samples, dimension)
    )
    clean -= clean.mean(axis=0, keepdims=True)
    pairwise = np.linalg.norm(
        centers[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=-1
    )
    positive = pairwise[pairwise > 0]
    metadata = {
        "minimum_cluster_size": int(counts.min()),
        "clean_center_pair_distance_min": float(positive.min()),
        "clean_separation_over_std": float(positive.min() / config.cluster_std),
        "clean_rank": int(np.linalg.matrix_rank(clean)),
    }
    return clean, labels, metadata


def _forward_view(
    clean: np.ndarray,
    rotation: np.ndarray,
    cubic_strength: float,
    noise_std: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    rotated = clean @ rotation
    scale = rotated.std(axis=0, keepdims=True)
    scale = np.maximum(scale, 1.0e-6)
    normalized = rotated / scale
    observed = normalized + cubic_strength * normalized**3
    observed += noise_std * rng.normal(size=observed.shape)
    return observed, scale


def _inverse_view(
    observed: np.ndarray,
    rotation: np.ndarray,
    scale: np.ndarray,
    cubic_strength: float,
) -> np.ndarray:
    estimate = np.cbrt(observed / cubic_strength)
    for _ in range(12):
        residual = estimate + cubic_strength * estimate**3 - observed
        derivative = 1.0 + 3.0 * cubic_strength * estimate**2
        estimate -= residual / derivative
    rotated = estimate * scale
    return rotated @ rotation.T


def _observe(
    clean: np.ndarray,
    config: RecoverableV2Config,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    dimension = clean.shape[1]
    rotations = [_orthogonal(dimension, rng), _orthogonal(dimension, rng)]
    views: list[np.ndarray] = []
    scales: list[np.ndarray] = []
    for rotation in rotations:
        view, scale = _forward_view(
            clean,
            rotation,
            config.cubic_strength,
            config.additive_noise_std,
            rng,
        )
        views.append(view)
        scales.append(scale)

    pattern = rng.choice(
        3,
        size=clean.shape[0],
        p=(
            config.both_view_fraction,
            config.view_one_only_fraction,
            config.view_two_only_fraction,
        ),
    )
    missing = np.zeros((clean.shape[0], 2 * dimension), dtype=bool)
    missing[pattern == 1, dimension:] = True
    missing[pattern == 2, :dimension] = True
    features = np.concatenate(views, axis=1)
    features[missing] = 0.0

    recovered_views = [
        _inverse_view(view, rotation, scale, config.cubic_strength)
        for view, rotation, scale in zip(views, rotations, scales)
    ]
    oracle = np.empty_like(clean)
    for row in range(clean.shape[0]):
        available = []
        if pattern[row] != 2:
            available.append(recovered_views[0][row])
        if pattern[row] != 1:
            available.append(recovered_views[1][row])
        oracle[row] = np.mean(available, axis=0)

    metadata = {
        "observation_family": "paired_orthogonal_componentwise_cubic_block_missing",
        "view_pattern_counts": {
            "both": int(np.sum(pattern == 0)),
            "view_1_only": int(np.sum(pattern == 1)),
            "view_2_only": int(np.sum(pattern == 2)),
        },
        "rows_without_view": int(
            np.sum(missing[:, :dimension].all(axis=1) & missing[:, dimension:].all(axis=1))
        ),
        "forward_reads_labels": False,
        "oracle_reads_labels": False,
        "recoverability_claim": (
            "analytic inverse for this known paired-view family only; additive "
            "noise makes recovery approximate"
        ),
        "rotation_hashes": [
            sha256(np.ascontiguousarray(rotation).tobytes()).hexdigest()
            for rotation in rotations
        ],
    }
    return features, missing, oracle, metadata


def _artifact_hash(
    config: RecoverableV2Config,
    task_index: int,
    split: Split,
    arrays: tuple[np.ndarray, ...],
) -> str:
    digest = sha256(
        json.dumps(
            {"config": asdict(config), "task_index": task_index, "split": split},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def generate_recoverable_v2_task(
    task_index: int,
    split: Split = "smoke",
    config: RecoverableV2Config | None = None,
) -> RecoverableV2Task:
    config = config or RecoverableV2Config()
    config.validate()
    seed = seed_for_recoverable_v2_task(task_index, split, config)
    rng = np.random.default_rng(seed)
    n_samples = int(rng.choice(N_VALUES))
    dimension = int(rng.choice(D_VALUES))
    max_clusters = min(config.max_clusters, dimension + 1, n_samples // config.minimum_cluster_size)
    n_clusters = int(rng.integers(2, max_clusters + 1))
    clean, labels, clean_metadata = _clean_mixture(
        n_samples, dimension, n_clusters, config, rng
    )
    features, missing, oracle, observation_metadata = _observe(clean, config, rng)
    feature_types = np.zeros(features.shape[1], dtype=np.uint8)
    arrays = (
        features.astype(np.float32),
        clean.astype(np.float32),
        missing.astype(np.uint8),
        feature_types,
        labels.astype(np.int16),
        oracle.astype(np.float32),
    )
    artifact = _artifact_hash(config, task_index, split, arrays)
    task_id = f"dfh2-{split[:2]}-{task_index:08d}-{artifact[:12]}"
    metadata = {
        "generator": config.name,
        "generator_protocol_version": config.protocol_version,
        "task_id": task_id,
        "artifact_sha256": artifact,
        "seed": seed,
        "split": split,
        "task_index": task_index,
        "n_samples": n_samples,
        "intrinsic_dim": dimension,
        "n_features": 2 * dimension,
        "n_clusters": n_clusters,
        "labels_isolated": True,
        "acceptance_forbidden_fields": ["Y", "K", "ARI", "NMI", "CLM"],
        **clean_metadata,
        **observation_metadata,
    }
    features32, clean32, missing8, types8, labels16, oracle32 = arrays
    if not all(np.isfinite(value).all() for value in (features32, clean32, oracle32)):
        raise RuntimeError("v2 generator produced non-finite values")
    if not np.all(features32[missing8.astype(bool)] == 0.0):
        raise RuntimeError("masked cells must contain zero sentinel")
    return RecoverableV2Task(
        task_id=task_id,
        features=np.ascontiguousarray(features32),
        clean_signal=np.ascontiguousarray(clean32),
        missing_mask=np.ascontiguousarray(missing8),
        feature_types=np.ascontiguousarray(types8),
        labels=np.ascontiguousarray(labels16),
        oracle_recovered=np.ascontiguousarray(oracle32),
        metadata=metadata,
    )
