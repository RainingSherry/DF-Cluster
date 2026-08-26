"""Generator V4 P0/P1: audited label-free clean-geometry observation tasks.

This module is intentionally self-contained and separate from the legacy
``generator.py``/``hybrid_generator.py`` implementations.  The observation
adapter accepts only clean latent rows, nuisance roots, an immutable graph
specification, and stable row identifiers.  Labels are generated and kept in
an outer-audit field, never passed to the observation map.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .mechanisms import apply_label_free_mechanism
from .source_complexity_graph import (
    OBSERVATION_STRATA,
    INFORMATION_STRATA,
    SourceComplexityGraphSpec,
    build_source_complexity_graph,
    generate_source_complexity_observation,
)


FEATURE_TYPE_NAMES = ("numerical", "categorical", "ordinal", "count", "bounded")
FEATURE_TYPE_CODES = {name: index for index, name in enumerate(FEATURE_TYPE_NAMES)}
COLUMN_ROLE_NAMES = ("informative", "redundant", "nuisance", "irrelevant")
CLEAN_FAMILIES = ("gaussian", "student_t", "factor", "hierarchical")
OBSERVATION_FAMILIES = OBSERVATION_STRATA
MISSINGNESS_MODES = ("mcar", "mar", "mnar")
FORBIDDEN_TRAINING_KEYS = {
    "labels", "y", "Y", "K", "CLM", "ARI", "NMI", "AMI",
    "generator_family", "clean_family", "observation_family",
    "difficulty", "information_stratum", "generator_hidden_parameters",
}


def validate_training_payload(payload: Dict[str, Any]) -> None:
    """Fail closed if a training payload contains evaluation-only controls."""

    def visit(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if str(key) in FORBIDDEN_TRAINING_KEYS:
                    raise ValueError("forbidden training field: %s.%s" % (path, key))
                visit(child, "%s.%s" % (path, key))
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(child, "%s[%d]" % (path, index))

    visit(payload, "payload")


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _array_digest(digest: "hashlib._Hash", array: np.ndarray) -> None:
    value = np.ascontiguousarray(array)
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(_canonical(list(value.shape)))
    digest.update(value.tobytes(order="C"))


def _config_sha256(config: "V4Config") -> str:
    return hashlib.sha256(_canonical(asdict(config))).hexdigest()


GENERATOR_SOURCE_COMPONENTS = (
    "core.py",
    "mechanisms.py",
    "source_complexity_graph.py",
    "full_sampler.py",
    "provenance.yaml",
)


def source_sha256() -> str:
    """Hash only files that change the generated V4 task distribution.

    Audit runners, model definitions, qualification code and tests must not
    invalidate a frozen generator qualification if task-generation behavior is
    unchanged. This follows the plan's per-generator-component provenance rule.
    """

    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in GENERATOR_SOURCE_COMPONENTS:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError("missing generator source component: %s" % path)
        digest.update(name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class V4Config:
    """A resolved task configuration used by the P0/P1 contract."""

    n_samples: int = 256
    n_features: int = 32
    n_clusters: int = 4
    intrinsic_dim: int = 8
    ambient_dim: int = 128
    clean_family: str = "gaussian"
    observation_family: str = "tabicl_graph_mlp"
    information_stratum: str = "noisy_recoverable"
    missingness: str = "mcar"
    missing_rate: float = 0.10
    categorical_fraction: float = 0.20
    ordinal_fraction: float = 0.10
    count_fraction: float = 0.10
    bounded_fraction: float = 0.10
    graph_depth: int = 4
    max_parents: int = 3
    nuisance_dim: int = 8
    noise_scale: float = 0.05
    center_scale: float = 3.0
    split: str = "qualification"
    task_index: int = 0
    seed: int = 20260824

    def validate(self) -> None:
        if self.ambient_dim != 128:
            raise ValueError("Generator V4 requires ambient_dim=128")
        if not 128 <= self.n_samples <= 2048:
            raise ValueError("n_samples must be in [128, 2048]")
        if not 10 <= self.n_features <= 160:
            raise ValueError("n_features must be in [10, 160]")
        if not 2 <= self.n_clusters <= 20:
            raise ValueError("n_clusters must be in [2, 20]")
        min_size = max(8, int(math.ceil(0.01 * self.n_samples)))
        if self.n_samples < min_size * self.n_clusters:
            raise ValueError("n_samples cannot satisfy the minimum cluster-size rule")
        if self.split not in {"qualification", "train", "validation", "test"}:
            raise ValueError("split must be qualification, train, validation, or test")
        if self.task_index < 0:
            raise ValueError("task_index must be non-negative")
        if not 2 <= self.intrinsic_dim <= self.ambient_dim:
            raise ValueError("intrinsic_dim must be in [2, 128]")
        if self.n_samples * self.n_features > 2048 * 160:
            raise ValueError("task exceeds the Generator V4 cell envelope")
        if self.clean_family not in CLEAN_FAMILIES:
            raise ValueError("unknown clean_family")
        if self.observation_family not in OBSERVATION_FAMILIES:
            raise ValueError("unknown observation_family")
        if self.information_stratum not in INFORMATION_STRATA:
            raise ValueError("unknown information_stratum")
        if self.missingness not in MISSINGNESS_MODES:
            raise ValueError("unknown missingness mode")
        if not 0.0 <= self.missing_rate <= 0.30:
            raise ValueError("missing_rate must be in [0, 0.30]")
        fractions = (
            self.categorical_fraction,
            self.ordinal_fraction,
            self.count_fraction,
            self.bounded_fraction,
        )
        if any(not 0.0 <= value <= 1.0 for value in fractions):
            raise ValueError("feature-type fractions must be in [0, 1]")
        if sum(fractions) > 1.0 + 1e-12:
            raise ValueError("feature-type fractions must sum to at most one")
        if not 3 <= self.graph_depth <= 8:
            raise ValueError("graph_depth must be in [3, 8]")
        if not 1 <= self.max_parents <= 5:
            raise ValueError("max_parents must be in [1, 5]")
        if self.nuisance_dim < 1:
            raise ValueError("nuisance_dim must be positive")
        if self.noise_scale < 0.0 or self.center_scale <= 0.0:
            raise ValueError("noise_scale must be non-negative and center_scale positive")


@dataclass(frozen=True)
class ObservationGraphSpec:
    """Serializable row-wise observation graph used by ``generate_observation``."""

    root_dim: int
    family: str
    node_ops: Tuple[str, ...]
    node_parents: Tuple[Tuple[int, ...], ...]
    node_depths: Tuple[int, ...]
    leaf_nodes: Tuple[int, ...]
    feature_types: Tuple[str, ...]
    column_roles: Tuple[str, ...]
    missingness: str
    missing_rate: float
    noise_scale: float
    seed: int

    def validate(self, n_features: Optional[int] = None) -> None:
        if self.root_dim < 1:
            raise ValueError("root_dim must be positive")
        if self.family not in OBSERVATION_FAMILIES:
            raise ValueError("unknown observation graph family")
        if len(self.node_ops) != len(self.node_parents):
            raise ValueError("node_ops and node_parents must have equal length")
        if len(self.node_ops) != len(self.node_depths):
            raise ValueError("node_ops and node_depths must have equal length")
        if len(self.leaf_nodes) != len(self.feature_types):
            raise ValueError("one leaf and type are required per feature")
        if len(self.column_roles) != len(self.feature_types):
            raise ValueError("one column role is required per feature")
        if any(role not in COLUMN_ROLE_NAMES for role in self.column_roles):
            raise ValueError("unknown column role")
        if n_features is not None and len(self.feature_types) != n_features:
            raise ValueError("graph feature count mismatch")
        for index, parents in enumerate(self.node_parents):
            if not parents:
                raise ValueError("each graph node must have at least one parent")
            for parent in parents:
                if parent < 0 or parent >= self.root_dim + index:
                    raise ValueError("graph parent must be a root or earlier node")
        if self.missingness not in MISSINGNESS_MODES:
            raise ValueError("unknown graph missingness mode")
        if not 0.0 <= self.missing_rate <= 0.30:
            raise ValueError("graph missing_rate must be in [0, 0.30]")


@dataclass(frozen=True)
class ObservationResult:
    features: np.ndarray
    missing_mask: np.ndarray
    feature_types: np.ndarray
    certificate: Dict[str, Any]


@dataclass(frozen=True)
class V4Task:
    """A task with label-bearing fields available only to outer audit code."""

    features: np.ndarray
    clean_latent: np.ndarray
    missing_mask: np.ndarray
    feature_types: np.ndarray
    labels: np.ndarray
    nuisance_roots: np.ndarray
    observation_graph: Any
    row_ids: np.ndarray
    metadata: Dict[str, Any]

    def training_payload(self) -> Dict[str, Any]:
        """Return model inputs plus clean geometry target, never Y/K/CLM."""

        payload = {
            "task_id": self.metadata["task_id"],
            "model_input": {
                "features": self.features,
                "missing_mask": self.missing_mask,
                "feature_types": self.feature_types,
            },
            "geometry_target": {"clean_latent": self.clean_latent},
        }
        validate_training_payload(payload)
        return payload

    def inference_payload(self) -> Dict[str, Any]:
        """Return the deployment-shaped payload with no privileged geometry."""

        return {
            "task_id": self.metadata["task_id"],
            "model_input": {
                "features": self.features,
                "missing_mask": self.missing_mask,
                "feature_types": self.feature_types,
            },
        }

    def geometry_targets(self, pair_seed: int = 0, num_pairs: int = 4096) -> Dict[str, Any]:
        """Compute rotation-invariant clean targets without opening labels."""

        if num_pairs <= 0:
            raise ValueError("num_pairs must be positive")
        gram = centered_normalized_gram(self.clean_latent)
        rng = np.random.default_rng(pair_seed)
        first = rng.integers(0, self.clean_latent.shape[0], size=num_pairs)
        second = rng.integers(0, self.clean_latent.shape[0] - 1, size=num_pairs)
        second += second >= first
        distances = np.linalg.norm(
            self.clean_latent[first] - self.clean_latent[second], axis=1
        ).astype(np.float32)
        return {
            "clean_gram": np.ascontiguousarray(gram, dtype=np.float32),
            "pair_first": first.astype(np.int64),
            "pair_second": second.astype(np.int64),
            "clean_distances": distances,
        }

    def audit_payload(self) -> Dict[str, Any]:
        """Return the isolated outer-evaluation payload; never pass to training."""

        return {
            "task_id": self.metadata["task_id"],
            "labels": self.labels,
            "n_clusters": int(self.labels.max()) + 1,
            "metadata": self.metadata,
        }


def _orthogonal(rng: np.random.Generator, rows: int, cols: int) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(rows, cols)))
    signs = np.where(np.diag(r[:cols, :cols]) < 0.0, -1.0, 1.0)
    return q[:, :cols] * signs


def _cluster_counts(config: V4Config, rng: np.random.Generator) -> np.ndarray:
    minimum = max(8, int(math.ceil(0.01 * config.n_samples)))
    remaining = config.n_samples - minimum * config.n_clusters
    alpha = float(np.exp(rng.uniform(np.log(0.3), np.log(10.0))))
    weights = rng.dirichlet(np.full(config.n_clusters, alpha, dtype=np.float64))
    return minimum + rng.multinomial(remaining, weights)


def _clean_signal(config: V4Config, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    counts = _cluster_counts(config, rng)
    labels = np.repeat(np.arange(config.n_clusters, dtype=np.int16), counts)
    rng.shuffle(labels)

    prototypes = rng.normal(size=(config.n_clusters, config.intrinsic_dim))
    prototypes -= prototypes.mean(axis=0, keepdims=True)
    prototypes /= np.maximum(np.linalg.norm(prototypes, axis=1, keepdims=True), 1e-12)
    prototypes *= config.center_scale * math.sqrt(config.intrinsic_dim)

    signal = np.empty((config.n_samples, config.intrinsic_dim), dtype=np.float64)
    for cluster in range(config.n_clusters):
        rows = labels == cluster
        n_rows = int(rows.sum())
        if config.clean_family == "student_t":
            noise = rng.standard_t(df=3.0, size=(n_rows, config.intrinsic_dim))
        elif config.clean_family == "factor":
            rank = max(1, min(config.intrinsic_dim, config.intrinsic_dim // 3))
            factors = rng.normal(size=(n_rows, rank))
            loadings = rng.normal(size=(rank, config.intrinsic_dim)) / math.sqrt(rank)
            noise = factors @ loadings + 0.15 * rng.normal(
                size=(n_rows, config.intrinsic_dim)
            )
        elif config.clean_family == "hierarchical":
            sub = rng.normal(size=(2, config.intrinsic_dim))
            sub -= sub.mean(axis=0, keepdims=True)
            choices = rng.integers(0, 2, size=n_rows)
            noise = sub[choices] + 0.30 * rng.normal(
                size=(n_rows, config.intrinsic_dim)
            )
        else:
            noise = rng.normal(size=(n_rows, config.intrinsic_dim))
        scale = np.exp(rng.uniform(np.log(0.35), np.log(1.7)))
        signal[rows] = prototypes[cluster] + scale * noise

    signal -= signal.mean(axis=0, keepdims=True)
    rms = math.sqrt(float(np.mean(np.square(signal))))
    signal /= max(rms, 1e-12)
    basis = _orthogonal(rng, config.ambient_dim, config.intrinsic_dim)
    clean = signal @ basis.T
    clean -= clean.mean(axis=0, keepdims=True)
    clean /= max(math.sqrt(float(np.mean(np.square(clean)))), 1e-12)
    metadata = {
        "clean_family": config.clean_family,
        "clean_intrinsic_dim": config.intrinsic_dim,
        "clean_rank": int(np.linalg.matrix_rank(signal)),
        "minimum_cluster_size": int(counts.min()),
        "cluster_counts": counts.tolist(),
    }
    return np.ascontiguousarray(clean, dtype=np.float32), labels, metadata


def _nuisance_roots(config: V4Config, rng: np.random.Generator) -> np.ndarray:
    gaussian = rng.normal(size=(config.n_samples, config.nuisance_dim // 2 + config.nuisance_dim % 2))
    heavy = rng.standard_t(df=4.0, size=(config.n_samples, config.nuisance_dim // 2))
    roots = np.concatenate([gaussian, heavy], axis=1)[:, : config.nuisance_dim]
    roots = np.clip(roots, -8.0, 8.0)
    return np.ascontiguousarray(roots, dtype=np.float32)


def _feature_types(config: V4Config, rng: np.random.Generator) -> Tuple[str, ...]:
    names: List[str] = ["numerical"] * config.n_features
    remaining = list(range(config.n_features))
    rng.shuffle(remaining)
    cursor = 0
    for name, fraction in (
        ("categorical", config.categorical_fraction),
        ("ordinal", config.ordinal_fraction),
        ("count", config.count_fraction),
        ("bounded", config.bounded_fraction),
    ):
        count = int(round(fraction * config.n_features))
        for index in remaining[cursor : cursor + count]:
            names[index] = name
        cursor += count
    return tuple(names)


def _column_roles(config: V4Config) -> Tuple[str, ...]:
    """Resolve a deterministic role mixture without reading Y or geometry."""

    total = config.n_features
    counts = {
        "informative": max(1, int(round(0.50 * total))),
        "redundant": max(1, int(round(0.20 * total))),
        "nuisance": max(1, int(round(0.20 * total))),
        "irrelevant": max(1, int(round(0.10 * total))),
    }
    while sum(counts.values()) > total:
        candidate = max(counts, key=lambda name: counts[name])
        if counts[candidate] <= 1:
            break
        counts[candidate] -= 1
    while sum(counts.values()) < total:
        counts["informative"] += 1
    roles: List[str] = []
    for name in COLUMN_ROLE_NAMES:
        roles.extend([name] * counts[name])
    return tuple(roles)


def _operation_pool(family: str) -> Tuple[str, ...]:
    if family == "tabicl_mlp":
        return ("linear", "tabicl_mlp", "tanh", "signed_log1p")
    if family == "tabicl_tree":
        return ("linear", "tabicl_tree", "signed_log1p", "saturation")
    if family == "mitra_inspired_tbp":
        return ("linear", "mitra_tbp", "tabicl_tree", "tanh", "product")
    if family == "mixed":
        return (
            "linear", "tabicl_mlp", "tabicl_tree", "mitra_tbp", "tanh", "signed_log1p",
            "periodic", "product", "max", "min", "monotone", "saturation",
        )
    return (
        "linear", "tanh", "signed_log1p", "periodic", "product", "max",
        "min", "monotone", "saturation",
    )


def _build_early_prototype_graph(config: V4Config, rng: np.random.Generator) -> ObservationGraphSpec:
    types = _feature_types(config, rng)
    roles = _column_roles(config)
    root_dim = config.ambient_dim + config.nuisance_dim
    pools = _operation_pool(config.observation_family)
    ops: List[str] = []
    parents: List[Tuple[int, ...]] = []
    depths: List[int] = []
    leaves: List[int] = []
    informative_nodes: List[int] = []
    nuisance_root_pool = np.arange(config.ambient_dim, root_dim, dtype=np.int64)
    clean_root_pool = np.arange(0, config.ambient_dim, dtype=np.int64)
    for feature, role in enumerate(roles):
        if role == "informative":
            parent_pool = clean_root_pool
        elif role == "redundant" and informative_nodes:
            reference = root_dim + informative_nodes[feature % len(informative_nodes)]
            optional = np.asarray(
                [value for value in nuisance_root_pool if value != reference], dtype=np.int64
            )
            parent_pool = np.concatenate(([reference], optional))
        else:
            parent_pool = nuisance_root_pool
        max_parent_count = min(config.max_parents, len(parent_pool))
        parent_count = int(rng.integers(1, max_parent_count + 1))
        if role == "redundant" and informative_nodes:
            reference = root_dim + informative_nodes[feature % len(informative_nodes)]
            extra_count = min(parent_count - 1, len(parent_pool) - 1)
            if extra_count > 0:
                extra_pool = np.asarray([value for value in parent_pool if value != reference])
                extra = rng.choice(extra_pool, size=extra_count, replace=False)
                chosen = (int(reference),) + tuple(int(value) for value in extra)
            else:
                chosen = (int(reference),)
        else:
            chosen = tuple(
                int(value) for value in rng.choice(parent_pool, size=parent_count, replace=False)
            )
        ops.append(str(pools[feature % len(pools)]))
        parents.append(chosen)
        depths.append(int(config.graph_depth + (feature % max(1, 9 - config.graph_depth))))
        leaves.append(feature)
        if role == "informative":
            informative_nodes.append(feature)
    graph = ObservationGraphSpec(
        root_dim=root_dim,
        family=config.observation_family,
        node_ops=tuple(ops),
        node_parents=tuple(parents),
        node_depths=tuple(depths),
        leaf_nodes=tuple(leaves),
        feature_types=types,
        column_roles=roles,
        missingness=config.missingness,
        missing_rate=config.missing_rate,
        noise_scale=config.noise_scale,
        seed=int(config.seed),
    )
    graph.validate(config.n_features)
    return graph


def build_observation_graph(config: V4Config, rng: Optional[np.random.Generator] = None) -> SourceComplexityGraphSpec:
    """Build the active full source-complexity DAG from independent RNG streams."""

    config.validate()
    seed_sequence = np.random.SeedSequence(config.seed)
    topology_seed, mechanism_seed, data_seed = (
        int(value)
        for value in seed_sequence.generate_state(3, dtype=np.uint64)
    )
    return build_source_complexity_graph(
        n_features=config.n_features,
        ambient_dim=config.ambient_dim,
        nuisance_dim=config.nuisance_dim,
        family=config.observation_family,
        information_stratum=config.information_stratum,
        missingness=config.missingness,
        missing_rate=config.missing_rate,
        noise_scale=config.noise_scale,
        max_parents=config.max_parents,
        categorical_fraction=config.categorical_fraction,
        ordinal_fraction=config.ordinal_fraction,
        count_fraction=config.count_fraction,
        bounded_fraction=config.bounded_fraction,
        topology_seed=topology_seed,
        mechanism_seed=mechanism_seed,
        data_seed=data_seed,
    )


def _splitmix64(values: np.ndarray) -> np.ndarray:
    value = np.asarray(values, dtype=np.uint64)
    value = value + np.uint64(0x9E3779B97F4A7C15)
    value = (value ^ (value >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    value = (value ^ (value >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return value ^ (value >> np.uint64(31))


def _stable_uniform(row_ids: np.ndarray, column: int, seed: int) -> np.ndarray:
    base = np.asarray(row_ids, dtype=np.uint64)
    # These are intentional uint64 modular-mixing operations.  Suppress the
    # diagnostic overflow warning while retaining the exact wraparound.
    with np.errstate(over="ignore"):
        keyed = base + np.uint64(seed) * np.uint64(0xD1342543DE82EF95)
        keyed += np.uint64(column + 1) * np.uint64(0xA24BAED4963EE407)
    value = _splitmix64(keyed)
    return (value >> np.uint64(11)).astype(np.float64) / 9007199254740992.0


def _stable_normal(row_ids: np.ndarray, column: int, seed: int) -> np.ndarray:
    u1 = np.maximum(_stable_uniform(row_ids, column * 2 + 1, seed), 1e-12)
    u2 = _stable_uniform(row_ids, column * 2 + 2, seed)
    return np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)


def _node_value(
    values: np.ndarray,
    operation: str,
    node_index: int,
    seed: int,
    row_ids: Optional[np.ndarray] = None,
) -> np.ndarray:
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError("node parents must form a non-empty matrix")
    if row_ids is None:
        row_ids = np.arange(values.shape[0], dtype=np.uint64)
    concrete = apply_label_free_mechanism(values, operation, row_ids, node_index, seed)
    if concrete is not None:
        return np.clip(np.nan_to_num(concrete, nan=0.0, posinf=20.0, neginf=-20.0), -20.0, 20.0)
    rng = np.random.default_rng(np.random.SeedSequence([seed, node_index + 1]))
    weights = rng.normal(size=values.shape[1]) / math.sqrt(values.shape[1])
    linear = values @ weights
    if operation == "linear":
        output = linear
    elif operation == "tanh":
        output = np.tanh(linear)
    elif operation == "signed_log1p":
        output = np.sign(linear) * np.log1p(np.abs(linear))
    elif operation == "periodic":
        output = np.sin(linear) + 0.25 * np.cos(2.0 * linear)
    elif operation == "product":
        output = np.prod(np.tanh(values[:, : min(3, values.shape[1])]), axis=1)
    elif operation == "max":
        output = np.max(values, axis=1)
    elif operation == "min":
        output = np.min(values, axis=1)
    elif operation == "monotone":
        output = linear + 0.25 * np.sin(linear)
    elif operation == "saturation":
        output = np.tanh(1.5 * linear)
    elif operation == "mlp":
        hidden = np.tanh(values @ rng.normal(size=(values.shape[1], 8)) / math.sqrt(values.shape[1]))
        output = hidden @ rng.normal(size=8) / math.sqrt(8)
    elif operation == "tree":
        threshold = float(np.median(values[:, 0]))
        output = np.where(values[:, 0] <= threshold, linear - 0.35, linear + 0.35)
    elif operation == "ensemble":
        threshold = float(np.median(values[:, 0]))
        branch = np.where(values[:, 0] <= threshold, -1.0, 1.0)
        output = 0.5 * linear + 0.5 * branch * np.tanh(linear)
    else:
        raise ValueError("unknown observation node operation: %s" % operation)
    return np.clip(np.nan_to_num(output, nan=0.0, posinf=20.0, neginf=-20.0), -20.0, 20.0)


def _mixed_head(values: np.ndarray, feature_type: str, row_ids: np.ndarray, feature: int, seed: int) -> np.ndarray:
    scale = float(np.std(values))
    normalized = (values - float(np.mean(values))) / max(scale, 1e-6)
    if feature_type == "bounded":
        return 2.0 / (1.0 + np.exp(-np.clip(normalized, -12.0, 12.0))) - 1.0
    if feature_type in {"categorical", "ordinal"}:
        bins = 4 if feature_type == "ordinal" else 5
        quantiles = np.quantile(normalized, np.linspace(0.0, 1.0, bins + 1))
        quantiles = np.maximum.accumulate(quantiles)
        return np.digitize(normalized, quantiles[1:-1], right=False).astype(np.float64)
    if feature_type == "count":
        rate = np.exp(np.clip(normalized, -2.0, 2.0))
        noise = _stable_normal(row_ids, feature, seed)
        return np.maximum(0.0, np.rint(rate + np.sqrt(rate) * noise))
    return normalized


def _apply_missingness(
    values: np.ndarray,
    graph: ObservationGraphSpec,
    row_ids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    mask = np.zeros(values.shape, dtype=np.uint8)
    if graph.missing_rate > 0.0:
        for feature in range(values.shape[1]):
            centered = values[:, feature] - float(np.mean(values[:, feature]))
            scale = max(float(np.std(values[:, feature])), 1e-6)
            standardized = centered / scale
            if graph.missingness == "mcar":
                probability = np.full(values.shape[0], graph.missing_rate)
            elif graph.missingness == "mar":
                driver = np.tanh(standardized)
                probability = graph.missing_rate * (0.55 + 0.45 / (1.0 + np.exp(-driver)))
            else:
                driver = np.abs(standardized)
                probability = graph.missing_rate * (0.50 + 0.50 / (1.0 + np.exp(-driver)))
            random = _stable_uniform(row_ids, 10_000 + feature, graph.seed)
            mask[:, feature] = (random < np.clip(probability, 0.0, 0.95)).astype(np.uint8)
        for row in range(mask.shape[0]):
            if bool(mask[row].all()):
                keep = int(np.argmax(np.abs(values[row])))
                mask[row, keep] = 0
    output = np.asarray(values, dtype=np.float64).copy()
    output[mask.astype(bool)] = 0.0
    return output, mask, {
        "missingness": graph.missingness,
        "missing_rate_target": graph.missing_rate,
        "missing_cell_count": int(mask.sum()),
        "all_missing_rows": int(np.sum(mask.all(axis=1))),
    }


def generate_observation(
    clean_latent: np.ndarray,
    nuisance_roots: np.ndarray,
    graph: ObservationGraphSpec,
    row_ids: Optional[np.ndarray] = None,
) -> ObservationResult:
    """Generate X from roots; this function has no label-bearing argument."""

    if isinstance(graph, SourceComplexityGraphSpec):
        source = generate_source_complexity_observation(
            clean_latent, nuisance_roots, graph, np.arange(clean_latent.shape[0], dtype=np.uint64) if row_ids is None else row_ids
        )
        return ObservationResult(
            features=source.features,
            missing_mask=source.missing_mask,
            feature_types=source.feature_types,
            certificate=source.certificate,
        )
    clean = np.asarray(clean_latent, dtype=np.float64)
    nuisance = np.asarray(nuisance_roots, dtype=np.float64)
    if clean.ndim != 2 or nuisance.ndim != 2 or clean.shape[0] != nuisance.shape[0]:
        raise ValueError("clean_latent and nuisance_roots must be row-aligned matrices")
    if clean.shape[1] != 128 or not np.isfinite(clean).all() or not np.isfinite(nuisance).all():
        raise ValueError("roots must be finite and clean_latent must have width 128")
    graph.validate(len(graph.feature_types))
    if graph.root_dim != clean.shape[1] + nuisance.shape[1]:
        raise ValueError("graph root_dim does not match roots")
    rows = np.arange(clean.shape[0], dtype=np.uint64) if row_ids is None else np.asarray(row_ids, dtype=np.uint64)
    if rows.shape != (clean.shape[0],):
        raise ValueError("row_ids must have one value per row")
    roots = np.concatenate([clean, nuisance], axis=1)
    node_values: List[np.ndarray] = []
    for node_index, (operation, parents) in enumerate(zip(graph.node_ops, graph.node_parents)):
        parent_values = []
        for parent in parents:
            if parent < graph.root_dim:
                parent_values.append(roots[:, parent])
            else:
                parent_values.append(node_values[parent - graph.root_dim])
        current = _node_value(
            np.column_stack(parent_values), operation, node_index, graph.seed, rows
        )
        node_values.append(current)
    raw = np.column_stack([node_values[index] for index in graph.leaf_nodes])
    typed = np.column_stack(
        [
            _mixed_head(raw[:, feature], graph.feature_types[feature], rows, feature, graph.seed)
            for feature in range(raw.shape[1])
        ]
    )
    if graph.noise_scale > 0.0:
        noise = np.column_stack(
            [_stable_normal(rows, feature + 20_000, graph.seed) for feature in range(raw.shape[1])]
        )
        typed += graph.noise_scale * noise
    typed, mask, missing_certificate = _apply_missingness(typed, graph, rows)
    typed = np.ascontiguousarray(np.nan_to_num(typed, nan=0.0, posinf=1e4, neginf=-1e4), dtype=np.float32)
    type_codes = np.asarray([FEATURE_TYPE_CODES[name] for name in graph.feature_types], dtype=np.uint8)
    if not np.isfinite(typed).all():
        raise RuntimeError("observation graph produced non-finite features")
    if not np.all(typed[mask.astype(bool)] == 0.0):
        raise RuntimeError("missing cells must use the zero sentinel")
    certificate = {
        "node_count": len(graph.node_ops),
        "edge_count": int(sum(len(parents) for parents in graph.node_parents)),
        "max_depth": int(max(graph.node_depths) if graph.node_depths else 0),
        "observation_family": graph.family,
        "observation_map_reads_labels": False,
        "feature_type_counts": {
            name: int(sum(value == name for value in graph.feature_types))
            for name in FEATURE_TYPE_NAMES
        },
        "column_role_counts": {
            name: int(sum(value == name for value in graph.column_roles))
            for name in COLUMN_ROLE_NAMES
        },
        "stochastic_node_count": int(sum(operation in {"mlp", "tabicl_mlp", "ensemble", "mitra_tbp", "tree", "tabicl_tree"} for operation in graph.node_ops)),
        "tree_node_count": int(sum(operation in {"mitra_tbp", "tabicl_tree", "tree"} for operation in graph.node_ops)),
        "many_to_one_node_count": int(sum(operation in {"tree", "tabicl_tree", "mitra_tbp", "saturation", "max", "min"} for operation in graph.node_ops)),
        "nonfinite_count": 0,
        **missing_certificate,
    }
    return ObservationResult(
        features=typed,
        missing_mask=np.ascontiguousarray(mask, dtype=np.uint8),
        feature_types=type_codes,
        certificate=certificate,
    )


def centered_normalized_gram(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    gram = centered @ centered.T
    norm = float(np.linalg.norm(gram, ord="fro"))
    if norm <= 1e-12:
        raise ValueError("clean geometry is degenerate")
    return gram / norm


def generate_v4_task(config: V4Config) -> V4Task:
    """Generate one P0/P1 task with isolated labels and replay metadata."""

    config.validate()
    seed_sequence = np.random.SeedSequence(config.seed)
    geometry_seed, nuisance_seed = seed_sequence.generate_state(2, dtype=np.uint64)
    geometry_rng = np.random.default_rng(int(geometry_seed))
    nuisance_rng = np.random.default_rng(int(nuisance_seed))
    clean, labels, clean_certificate = _clean_signal(config, geometry_rng)
    nuisance = _nuisance_roots(config, nuisance_rng)
    graph = build_observation_graph(config)
    row_ids = np.arange(config.n_samples, dtype=np.uint64)
    observed = generate_observation(clean, nuisance, graph, row_ids)
    config_hash = _config_sha256(config)
    source_hash = source_sha256()
    digest = hashlib.sha256()
    digest.update(_canonical(asdict(config)))
    digest.update(source_hash.encode("ascii"))
    for array in (observed.features, clean, observed.missing_mask, observed.feature_types, labels, nuisance):
        _array_digest(digest, array)
    artifact_hash = digest.hexdigest()
    metadata: Dict[str, Any] = {
        "generator": "dfcluster_generator_v4_p0_p1",
        "generator_version": 4,
        "task_id": "generator_v4/%s/%d/%d"
        % (config.split, config.seed, config.task_index),
        "task_fingerprint": artifact_hash[:16],
        "artifact_sha256": artifact_hash,
        "config_sha256": config_hash,
        "source_sha256": source_hash,
        "labels_isolated": True,
        "observation_map_reads_labels": False,
        "clean_signal_is_privileged_training_target": True,
        "observation_family": config.observation_family,
        "cell_count": config.n_samples * config.n_features,
        **clean_certificate,
        **observed.certificate,
    }
    return V4Task(
        features=observed.features,
        clean_latent=np.ascontiguousarray(clean, dtype=np.float32),
        missing_mask=observed.missing_mask,
        feature_types=observed.feature_types,
        labels=np.ascontiguousarray(labels, dtype=np.int16),
        nuisance_roots=nuisance,
        observation_graph=graph,
        row_ids=row_ids,
        metadata=metadata,
    )
