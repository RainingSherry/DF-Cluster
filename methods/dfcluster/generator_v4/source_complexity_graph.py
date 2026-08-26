"""Full source-complexity observation DAG for Generator V4.

This module is the planned replacement for the early P0/P1 one-node-per-column
prototype.  It uses actual layered DAGs: every informative leaf is reached
from clean roots through 3--8 intermediate descendant nodes.  Nuisance-only
branches, redundant leaves, mixed-parent interactions and all source-derived
mechanism strata are explicit and serializable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .mechanisms import apply_label_free_mechanism, stable_normal


FEATURE_TYPE_NAMES = ("numerical", "categorical", "ordinal", "count", "bounded")
FEATURE_TYPE_CODES = {name: index for index, name in enumerate(FEATURE_TYPE_NAMES)}
COLUMN_ROLE_NAMES = ("informative", "redundant", "nuisance", "irrelevant")
OBSERVATION_STRATA = (
    "tabicl_graph_mlp",
    "tabicl_tree",
    "mitra_inspired_tbp",
    "flow_analytic",
    "mixed",
)
# Section 12 of 0824思路详细版.md requires a second, independently
# scheduled axis describing how much information the observation map retains.
# It is graph/data behavior, never a model input or task-selection signal.
INFORMATION_STRATA = (
    "preserving",
    "noisy_recoverable",
    "controlled_lossy",
)


def _splitmix64(values: np.ndarray) -> np.ndarray:
    value = np.asarray(values, dtype=np.uint64)
    with np.errstate(over="ignore"):
        value = value + np.uint64(0x9E3779B97F4A7C15)
        value = (value ^ (value >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        value = (value ^ (value >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return value ^ (value >> np.uint64(31))


def _stable_uniform(row_ids: np.ndarray, column: int, seed: int) -> np.ndarray:
    rows = np.asarray(row_ids, dtype=np.uint64)
    with np.errstate(over="ignore"):
        keyed = rows + np.uint64(seed) * np.uint64(0xD1342543DE82EF95)
        keyed += np.uint64(column + 1) * np.uint64(0xA24BAED4963EE407)
    value = _splitmix64(keyed)
    return (value >> np.uint64(11)).astype(np.float64) / 9007199254740992.0


def _canonical_rng(seed: int, *parts: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(seed), *(int(v) for v in parts)]))


def _finite(values: np.ndarray, limit: float = 20.0) -> np.ndarray:
    return np.clip(np.nan_to_num(values, nan=0.0, posinf=limit, neginf=-limit), -limit, limit)


@dataclass(frozen=True)
class SourceComplexityGraphSpec:
    root_dim: int
    family: str
    information_stratum: str
    node_ops: Tuple[str, ...]
    node_parents: Tuple[Tuple[int, ...], ...]
    node_depths: Tuple[int, ...]
    node_roles: Tuple[str, ...]
    node_param_seeds: Tuple[int, ...]
    leaf_nodes: Tuple[int, ...]
    leaf_path_depths: Tuple[int, ...]
    feature_types: Tuple[str, ...]
    column_roles: Tuple[str, ...]
    column_permutation: Tuple[int, ...]
    missingness: str
    missing_rate: float
    noise_scale: float
    topology_seed: int
    mechanism_seed: int
    data_seed: int

    def validate(self, n_features: Optional[int] = None) -> None:
        if self.family not in OBSERVATION_STRATA:
            raise ValueError("unknown source-complexity observation stratum")
        if self.information_stratum not in INFORMATION_STRATA:
            raise ValueError("unknown information stratum")
        count = len(self.node_ops)
        if not count or not (count == len(self.node_parents) == len(self.node_depths) == len(self.node_roles) == len(self.node_param_seeds)):
            raise ValueError("node graph fields have inconsistent length")
        if len(self.leaf_nodes) != len(self.feature_types) or len(self.leaf_nodes) != len(self.column_roles):
            raise ValueError("leaf/type/role cardinalities differ")
        if n_features is not None and len(self.leaf_nodes) != n_features:
            raise ValueError("feature count mismatch")
        if sorted(self.column_permutation) != list(range(len(self.leaf_nodes))):
            raise ValueError("column permutation is invalid")
        if any(role not in COLUMN_ROLE_NAMES for role in self.column_roles):
            raise ValueError("unknown column role")
        actual_depths: List[int] = []
        for node_index, parents in enumerate(self.node_parents):
            if not parents:
                raise ValueError("node without parents")
            previous = []
            for parent in parents:
                if parent < 0 or parent >= self.root_dim + node_index:
                    raise ValueError("parent is not topologically earlier")
                previous.append(0 if parent < self.root_dim else actual_depths[parent - self.root_dim])
            actual_depths.append(1 + max(previous))
        if tuple(actual_depths) != self.node_depths:
            raise ValueError("stored node depths do not equal actual DAG depths")
        for feature, leaf in enumerate(self.leaf_nodes):
            if not 0 <= leaf < count:
                raise ValueError("invalid leaf")
            if self.leaf_path_depths[feature] != actual_depths[leaf]:
                raise ValueError("leaf depth mismatch")
            if self.column_roles[feature] == "informative" and not 3 <= actual_depths[leaf] <= 8:
                raise ValueError("informative leaf path depth must be in [3,8]")
        if not any(role == "nuisance" for role in self.node_roles):
            raise ValueError("task lacks nuisance-only branch")
        if not any(len(parents) >= 2 for parents in self.node_parents):
            raise ValueError("task lacks multi-parent interaction branch")
        if len(set(self.node_ops)) < 2:
            raise ValueError("task must use at least two node mechanisms")


@dataclass(frozen=True)
class SourceObservationResult:
    features: np.ndarray
    raw_before_missing: np.ndarray
    missing_mask: np.ndarray
    feature_types: np.ndarray
    certificate: Dict[str, Any]


def _feature_types(n_features: int, rng: np.random.Generator, fractions: Sequence[float]) -> Tuple[str, ...]:
    categorical, ordinal, count, bounded = fractions
    labels: List[str] = ["numerical"] * n_features
    indices = rng.permutation(n_features).tolist()
    offset = 0
    for name, fraction in (("categorical", categorical), ("ordinal", ordinal), ("count", count), ("bounded", bounded)):
        take = int(round(n_features * fraction))
        for column in indices[offset : offset + take]:
            labels[column] = name
        offset += take
    return tuple(labels)


def _column_roles(n_features: int, rng: np.random.Generator) -> Tuple[str, ...]:
    counts = {"informative": max(1, int(round(n_features * 0.50))), "redundant": max(1, int(round(n_features * 0.20))), "nuisance": max(1, int(round(n_features * 0.20))), "irrelevant": max(1, int(round(n_features * 0.10)))}
    while sum(counts.values()) > n_features:
        largest = max(counts, key=counts.get)
        counts[largest] -= 1
    while sum(counts.values()) < n_features:
        counts["informative"] += 1
    output: List[str] = []
    for name in COLUMN_ROLE_NAMES:
        output.extend([name] * counts[name])
    return tuple(np.asarray(output, dtype=object)[rng.permutation(n_features)].tolist())


def _operation_pool(family: str) -> Tuple[str, ...]:
    if family == "tabicl_graph_mlp":
        return ("tabicl_mlp", "tabicl_graph_gp", "tabicl_graph_disc", "tabicl_graph_quad", "tabicl_graph_em", "tabicl_graph_prod", "linear", "tanh")
    if family == "tabicl_tree":
        return ("tabicl_tree", "tabicl_graph_tree", "linear", "signed_log1p", "saturation")
    if family == "mitra_inspired_tbp":
        return ("tbp_dt", "tbp_et", "tbp_rf", "tbp_gb", "tbp_direct_rf", "linear", "product")
    if family == "flow_analytic":
        return ("flow_iresnet", "linear", "monotone", "periodic", "product", "max", "min", "saturation")
    return ("tabicl_mlp", "tabicl_graph_gp", "tabicl_tree", "tbp_dt", "tbp_et", "tbp_rf", "tbp_gb", "tbp_direct_rf", "flow_iresnet", "linear", "monotone", "periodic", "product", "max", "min", "saturation")


def _add_chain(
    *,
    role: str,
    feature: int,
    first_parents: Sequence[int],
    target_depth: int,
    root_dim: int,
    max_parents: int,
    operations: Sequence[str],
    node_ops: List[str],
    node_parents: List[Tuple[int, ...]],
    node_depths: List[int],
    node_roles: List[str],
    node_param_seeds: List[int],
    rng: np.random.Generator,
    mechanism_seed: int,
    eligible_refs: Sequence[int],
) -> int:
    previous: Optional[int] = None
    initial_parent_depth = max(
        0 if parent < root_dim else node_depths[parent - root_dim]
        for parent in first_parents
    )
    for relative_depth in range(1, target_depth + 1):
        absolute_depth = initial_parent_depth + relative_depth
        if relative_depth == 1:
            parents = list(first_parents)
        else:
            assert previous is not None
            parents = [previous]
            candidates = [
                ref
                for ref in eligible_refs
                if (ref < root_dim or node_depths[ref - root_dim] <= absolute_depth - 1)
                and ref != previous
            ]
            extra_max = min(max_parents - 1, len(candidates))
            # Force a multi-parent interaction in the first informative chain at relative depth 2.
            extra_count = 1 if (role == "informative" and feature == 0 and relative_depth == 2 and extra_max >= 1) else int(rng.integers(0, extra_max + 1))
            if extra_count:
                extra = rng.choice(np.asarray(candidates, dtype=np.int64), size=extra_count, replace=False)
                parents.extend(int(value) for value in extra)
            parents = list(dict.fromkeys(parents))
        index = len(node_ops)
        op = operations[(feature + relative_depth + index) % len(operations)]
        node_ops.append(op)
        node_parents.append(tuple(int(parent) for parent in parents))
        node_depths.append(absolute_depth)
        node_roles.append(role)
        node_param_seeds.append(int(np.random.SeedSequence([mechanism_seed, index, feature, absolute_depth]).generate_state(1, dtype=np.uint64)[0]))
        previous = root_dim + index
    assert previous is not None
    return previous - root_dim


def _mechanism_anchors(family: str) -> Tuple[str, ...]:
    if family == "tabicl_graph_mlp":
        return ("tabicl_mlp", "tabicl_graph_gp", "tabicl_graph_disc", "tabicl_graph_quad", "tabicl_graph_em", "tabicl_graph_prod")
    if family == "tabicl_tree":
        return ("tabicl_tree", "tabicl_graph_tree")
    if family == "mitra_inspired_tbp":
        return ("tbp_dt", "tbp_et", "tbp_rf", "tbp_gb", "tbp_direct_rf")
    if family == "flow_analytic":
        return ("flow_iresnet", "flow_iresnet")
    return ("tabicl_mlp", "tabicl_graph_gp", "tabicl_tree", "tbp_dt", "tbp_et", "tbp_rf", "tbp_gb", "tbp_direct_rf", "flow_iresnet")


def _analytic_cycle() -> Tuple[str, ...]:
    return ("linear", "tanh", "signed_log1p", "periodic", "product", "monotone", "max", "min", "saturation")


def _information_analytic_cycle(information_stratum: str) -> Tuple[str, ...]:
    """Frozen Section-12 analytic mechanism mix for one information regime."""

    if information_stratum == "preserving":
        # Dimension-preserving iResNet/monotone descendants plus redundant
        # expansion are represented by these node operations and leaf roles.
        return ("flow_iresnet", "monotone", "linear", "signed_log1p", "tanh")
    if information_stratum == "noisy_recoverable":
        return _analytic_cycle()
    if information_stratum == "controlled_lossy":
        # The subsequent deterministic low-rank/quantized head transform is
        # the principal lossy operation; this cycle additionally includes
        # finite-resolution / many-to-one analytic descendants.
        return ("linear", "saturation", "max", "min", "signed_log1p", "periodic")
    raise ValueError("unknown information stratum")


def build_source_complexity_graph(
    *,
    n_features: int,
    ambient_dim: int,
    nuisance_dim: int,
    family: str,
    information_stratum: str,
    missingness: str,
    missing_rate: float,
    noise_scale: float,
    max_parents: int,
    categorical_fraction: float,
    ordinal_fraction: float,
    count_fraction: float,
    bounded_fraction: float,
    topology_seed: int,
    mechanism_seed: int,
    data_seed: int,
) -> SourceComplexityGraphSpec:
    """Construct a shared, layered 3--8-depth observation DAG.

    Intermediate nodes are shared across feature heads rather than creating a
    toy independent chain per column.  This yields actual graph branching,
    multi-parent interactions, source-complexity mechanisms and full depth
    while keeping the number of expensive TBP fits bounded per task.
    """

    if family not in OBSERVATION_STRATA:
        raise ValueError("unknown observation family")
    if information_stratum not in INFORMATION_STRATA:
        raise ValueError("unknown information stratum")
    if not 2 <= max_parents <= 5:
        raise ValueError("source-complexity DAG requires max_parents in [2,5]")
    topology_rng = _canonical_rng(topology_seed, 1)
    root_dim = ambient_dim + nuisance_dim
    types = _feature_types(n_features, topology_rng, (categorical_fraction, ordinal_fraction, count_fraction, bounded_fraction))
    roles = _column_roles(n_features, topology_rng)
    anchors = list(_mechanism_anchors(family))
    analytic = _information_analytic_cycle(information_stratum)
    node_ops: List[str] = []
    node_parents: List[Tuple[int, ...]] = []
    node_depths: List[int] = []
    node_roles: List[str] = []
    node_param_seeds: List[int] = []
    clean_roots = list(range(ambient_dim))
    nuisance_roots = list(range(ambient_dim, root_dim))
    info_layers: Dict[int, List[int]] = {depth: [] for depth in range(1, 8)}
    nuisance_layers: Dict[int, List[int]] = {depth: [] for depth in range(1, 8)}
    anchor_cursor = 0
    analytic_cursor = 0

    def ref_depth(ref: int) -> int:
        return 0 if ref < root_dim else node_depths[ref - root_dim]

    def choose_parents(required_depth: int, roots: Sequence[int], available: Sequence[int], force_multi: bool = False) -> List[int]:
        if required_depth == 1:
            primary_pool = list(roots)
        else:
            primary_pool = [ref for ref in available if ref_depth(ref) == required_depth - 1]
        if not primary_pool:
            raise RuntimeError("layered DAG has no parent at required predecessor depth")
        parents = [int(topology_rng.choice(np.asarray(primary_pool, dtype=np.int64)))]
        candidates = [ref for ref in list(roots) + list(available) if ref_depth(ref) <= required_depth - 1 and ref not in parents]
        maximum_extra = min(max_parents - 1, len(candidates))
        minimum_extra = 1 if force_multi and maximum_extra else 0
        extra_count = int(topology_rng.integers(minimum_extra, maximum_extra + 1))
        if extra_count:
            parents.extend(int(value) for value in topology_rng.choice(np.asarray(candidates, dtype=np.int64), size=extra_count, replace=False))
        return parents

    def next_operation(force_anchor: bool = False) -> str:
        nonlocal anchor_cursor, analytic_cursor
        if anchor_cursor < len(anchors) and (force_anchor or anchor_cursor < len(anchors)):
            op = anchors[anchor_cursor]
            anchor_cursor += 1
            return op
        op = analytic[analytic_cursor % len(analytic)]
        analytic_cursor += 1
        return op

    def append_node(parents: Sequence[int], depth: int, role: str, operation: Optional[str] = None) -> int:
        index = len(node_ops)
        node_ops.append(operation or next_operation())
        node_parents.append(tuple(int(parent) for parent in parents))
        node_depths.append(int(depth))
        node_roles.append(role)
        node_param_seeds.append(int(np.random.SeedSequence([mechanism_seed, index, depth, len(parents)]).generate_state(1, dtype=np.uint64)[0]))
        return root_dim + index

    # Shared clean and nuisance trunks.  Each layer has several nodes and
    # therefore forms real branches rather than a column-wise chain.
    shared_width = max(2, min(8, int(math.ceil(math.sqrt(n_features)))))
    for depth in range(1, 8):
        for width_index in range(shared_width):
            parents = choose_parents(depth, clean_roots, [ref for refs in info_layers.values() for ref in refs], force_multi=(depth == 2 and width_index == 0))
            info_layers[depth].append(append_node(parents, depth, "informative"))
        for width_index in range(shared_width):
            parents = choose_parents(depth, nuisance_roots, [ref for refs in nuisance_layers.values() for ref in refs], force_multi=(depth == 2 and width_index == 0))
            nuisance_layers[depth].append(append_node(parents, depth, "nuisance"))

    leaves: List[int] = [-1] * n_features
    # Construct leaves in dependency order; final column permutation randomizes
    # their emitted order, so this does not leak graph construction order.
    informative_features = [index for index, role in enumerate(roles) if role == "informative"]
    first_informative_feature = informative_features[0]
    for feature in informative_features:
        depth = int(topology_rng.integers(3, 9))
        parents = choose_parents(
            depth,
            clean_roots,
            info_layers[depth - 1],
            force_multi=(feature == first_informative_feature),
        )
        leaves[feature] = append_node(parents, depth, "informative") - root_dim
    informative_leaf_refs = [root_dim + leaf for index, leaf in enumerate(leaves) if leaf >= 0 and roles[index] == "informative"]
    for feature in [index for index, role in enumerate(roles) if role == "redundant"]:
        ref = int(topology_rng.choice(np.asarray(informative_leaf_refs, dtype=np.int64)))
        depth = ref_depth(ref) + 1
        parents = [ref]
        if max_parents >= 2:
            parents.append(int(topology_rng.choice(np.asarray(nuisance_roots, dtype=np.int64))))
        leaves[feature] = append_node(parents, depth, "redundant", operation=analytic[analytic_cursor % len(analytic)]) - root_dim
        analytic_cursor += 1
    nuisance_features = [index for index, value in enumerate(roles) if value == "nuisance"]
    first_nuisance_feature = nuisance_features[0]
    for role in ("nuisance", "irrelevant"):
        for feature in [index for index, value in enumerate(roles) if value == role]:
            depth = int(topology_rng.integers(3, 9))
            parents = choose_parents(
                depth,
                nuisance_roots,
                nuisance_layers[depth - 1],
                force_multi=(role == "nuisance" and feature == first_nuisance_feature),
            )
            leaves[feature] = append_node(parents, depth, role, operation=analytic[analytic_cursor % len(analytic)]) - root_dim
            analytic_cursor += 1

    graph = SourceComplexityGraphSpec(
        root_dim=root_dim,
        family=family,
        information_stratum=information_stratum,
        node_ops=tuple(node_ops),
        node_parents=tuple(node_parents),
        node_depths=tuple(node_depths),
        node_roles=tuple(node_roles),
        node_param_seeds=tuple(node_param_seeds),
        leaf_nodes=tuple(leaves),
        leaf_path_depths=tuple(node_depths[leaf] for leaf in leaves),
        feature_types=types,
        column_roles=roles,
        column_permutation=tuple(topology_rng.permutation(n_features).astype(int).tolist()),
        missingness=missingness,
        missing_rate=float(missing_rate),
        noise_scale=float(noise_scale),
        topology_seed=int(topology_seed),
        mechanism_seed=int(mechanism_seed),
        data_seed=int(data_seed),
    )
    graph.validate(n_features)
    return graph

def _analytic_node(values: np.ndarray, operation: str, node_seed: int) -> np.ndarray:
    generator = _canonical_rng(node_seed, 97)
    weights = generator.normal(size=values.shape[1]) / math.sqrt(values.shape[1])
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
    else:
        raise ValueError("unknown analytic operation: %s" % operation)
    return _finite(output)


def _robust_scale(values: np.ndarray) -> tuple[np.ndarray, float]:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = max(1.4826 * mad, float(np.std(values)), 1e-6)
    normalized = (values - median) / scale
    clipped = np.clip(normalized, -20.0, 20.0)
    return clipped, float(np.mean(clipped != normalized))


def _information_stratum_transform(
    raw: np.ndarray,
    graph: SourceComplexityGraphSpec,
    node_values: Sequence[np.ndarray],
    row_ids: np.ndarray,
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Apply the Section-12 regime without opening labels or audit metadata.

    The transform is deterministic from serialized graph/data seeds and row
    identifiers.  It is applied before typed heads and the frozen MCAR/MAR/
    MNAR mechanism, so it cannot alter the declared missing-rate grid.
    """

    values = np.asarray(raw, dtype=np.float64).copy()
    n_features = values.shape[1]
    if graph.information_stratum == "preserving":
        return values, {
            "kind": "dimension_preserving_flow_monotone_redundant",
            "low_rank_rank": None,
            "quantization_levels": None,
            "pre_head_random_mask_rate": 0.0,
            "correlated_nuisance_applied": False,
            "random_scale_applied": False,
        }
    if graph.information_stratum == "noisy_recoverable":
        rng = _canonical_rng(graph.mechanism_seed, 73)
        random_scale = np.exp(rng.uniform(math.log(0.75), math.log(1.25), size=n_features))
        nuisance_nodes = [index for index, role in enumerate(graph.node_roles) if role == "nuisance"]
        if not nuisance_nodes:
            raise RuntimeError("noisy-recoverable graph lacks nuisance descendants")
        nuisance = np.column_stack(
            [node_values[nuisance_nodes[feature % len(nuisance_nodes)]] for feature in range(n_features)]
        )
        nuisance = (nuisance - nuisance.mean(axis=0, keepdims=True)) / np.maximum(
            nuisance.std(axis=0, keepdims=True), 1e-6
        )
        values = values * random_scale[None, :] + 0.75 * graph.noise_scale * nuisance
        return _finite(values), {
            "kind": "heteroscedastic_outlier_correlated_nuisance",
            "low_rank_rank": None,
            "quantization_levels": None,
            "pre_head_random_mask_rate": 0.0,
            "correlated_nuisance_applied": True,
            "random_scale_applied": True,
            "random_scale_min": float(random_scale.min()),
            "random_scale_max": float(random_scale.max()),
        }
    if graph.information_stratum == "controlled_lossy":
        rng = _canonical_rng(graph.mechanism_seed, 79)
        rank = max(1, min(n_features - 1, int(math.ceil(0.50 * n_features))))
        basis, _ = np.linalg.qr(rng.normal(size=(n_features, rank)), mode="reduced")
        mixing = rng.normal(size=(rank, n_features)) / math.sqrt(rank)
        projected = values @ basis @ mixing
        center = np.median(projected, axis=0, keepdims=True)
        scale = np.maximum(np.std(projected, axis=0, keepdims=True), 1e-6)
        normalized = np.clip((projected - center) / scale, -4.0, 4.0)
        levels = int(rng.integers(4, 9))
        step = 8.0 / float(levels - 1)
        quantized = np.round((normalized + 4.0) / step) * step - 4.0
        thresholds = rng.uniform(0.10, 0.45, size=n_features)
        random_mask = np.column_stack(
            [_stable_uniform(row_ids, 160_000 + feature, graph.data_seed) < 0.05 for feature in range(n_features)]
        )
        thresholded = np.where(np.abs(quantized) < thresholds[None, :], 0.0, quantized)
        values = np.where(random_mask, 0.0, thresholded)
        return _finite(values), {
            "kind": "low_rank_quantization_threshold_random_mask",
            "low_rank_rank": int(rank),
            "quantization_levels": int(levels),
            "pre_head_random_mask_rate": float(random_mask.mean()),
            "correlated_nuisance_applied": False,
            "random_scale_applied": False,
        }
    raise ValueError("unknown information stratum")


def _typed_head(values: np.ndarray, feature_type: str, row_ids: np.ndarray, feature: int, data_seed: int) -> tuple[np.ndarray, Dict[str, Any]]:
    generator = _canonical_rng(data_seed, feature, 131)
    normalized, clip_fraction = _robust_scale(values)
    metadata: Dict[str, Any] = {"feature_type": feature_type, "head_clip_fraction": clip_fraction}
    if feature_type == "bounded":
        lo, hi = sorted(generator.uniform(-5.0, 5.0, size=2).tolist())
        if hi - lo < 0.5:
            hi = lo + 0.5
        metadata.update({"lower": lo, "upper": hi})
        return lo + (hi - lo) / (1.0 + np.exp(-np.clip(normalized, -12.0, 12.0))), metadata
    if feature_type == "ordinal":
        levels = int(generator.integers(3, 9))
        thresholds = np.sort(generator.normal(size=levels - 1))
        metadata.update({"levels": levels, "thresholds": thresholds.tolist()})
        return np.digitize(normalized, thresholds).astype(np.float64), metadata
    if feature_type == "categorical":
        categories = int(generator.integers(3, 13))
        mode = ("quantile_bins", "category_merge", "multi_logit")[feature % 3]
        if mode == "multi_logit":
            logits = np.column_stack([normalized * generator.normal() + generator.normal(scale=0.5) for _ in range(categories)])
            result = logits.argmax(axis=1).astype(np.float64)
        else:
            thresholds = np.quantile(normalized, np.linspace(0.0, 1.0, categories + 1)[1:-1])
            result = np.digitize(normalized, thresholds).astype(np.float64)
            if mode == "category_merge" and categories > 3:
                result = np.minimum(result, categories - 2)
        metadata.update({"categories": categories, "mode": mode})
        return result, metadata
    if feature_type == "count":
        mean_scale = float(np.exp(generator.uniform(np.log(0.5), np.log(5.0))))
        rate = mean_scale * np.exp(np.clip(normalized, -2.5, 2.5))
        dispersion = float(generator.uniform(0.5, 3.0))
        # Gamma-Poisson approximation using deterministic normal perturbations.
        gamma_like = np.maximum(rate * (1.0 + stable_normal(row_ids, 80_000 + feature, data_seed) / math.sqrt(dispersion + 1.0)), 1e-4)
        count = np.maximum(0.0, np.rint(gamma_like + np.sqrt(gamma_like) * stable_normal(row_ids, 90_000 + feature, data_seed)))
        metadata.update({"mean_scale": mean_scale, "dispersion": dispersion, "distribution": "negative_binomial_style"})
        return count, metadata
    return normalized, metadata


def _apply_missingness(values: np.ndarray, graph: SourceComplexityGraphSpec, row_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    mask = np.zeros(values.shape, dtype=np.uint8)
    for feature in range(values.shape[1]):
        standardized, _ = _robust_scale(values[:, feature])
        if graph.missingness == "mcar":
            probability = np.full(values.shape[0], graph.missing_rate)
        elif graph.missingness == "mar":
            other = values[:, (feature + 1) % values.shape[1]]
            driver, _ = _robust_scale(other)
            probability = graph.missing_rate * (0.35 + 0.65 / (1.0 + np.exp(-driver)))
        else:
            probability = graph.missing_rate * (0.30 + 0.70 / (1.0 + np.exp(-np.abs(standardized))))
        mask[:, feature] = (_stable_uniform(row_ids, 100_000 + feature, graph.data_seed) < np.clip(probability, 0.0, 0.95)).astype(np.uint8)
    # Every row retains at least one observed feature.
    for row in range(mask.shape[0]):
        if bool(mask[row].all()):
            mask[row, int(np.argmax(np.abs(values[row])))] = 0
    output = np.asarray(values, dtype=np.float64).copy()
    output[mask.astype(bool)] = 0.0
    return output, mask, {"missingness": graph.missingness, "missing_rate_target": graph.missing_rate, "missing_cell_count": int(mask.sum()), "all_missing_rows": int(mask.all(axis=1).sum())}


def _pair_indices(n_samples: int, data_seed: int, count: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    generator = _canonical_rng(data_seed, 181)
    first = generator.integers(0, n_samples, size=count)
    second = generator.integers(0, n_samples - 1, size=count)
    second += second >= first
    return first, second


def _knn_preservation(clean: np.ndarray, observed: np.ndarray, k: int = 10) -> float:
    count = clean.shape[0]
    k = min(k, count - 1)
    def indices(values: np.ndarray) -> np.ndarray:
        squared = np.sum(values * values, axis=1, keepdims=True)
        distances = squared + squared.T - 2.0 * (values @ values.T)
        np.fill_diagonal(distances, np.inf)
        return np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
    clean_neighbors = indices(clean)
    observed_neighbors = indices(observed)
    return float(np.mean([len(set(clean_neighbors[row]) & set(observed_neighbors[row])) / k for row in range(count)]))


def _matrix_certificate(clean: np.ndarray, observed: np.ndarray, graph: SourceComplexityGraphSpec, clipping: Sequence[float], head_metadata: Sequence[Dict[str, Any],], missing: Dict[str, Any]) -> Dict[str, Any]:
    singular = np.linalg.svd(observed, compute_uv=False)
    positive = singular[singular > 1e-8]
    weights = positive / max(float(positive.sum()), 1e-12)
    effective_rank = float(np.exp(-np.sum(weights * np.log(np.maximum(weights, 1e-12))))) if positive.size else 0.0
    condition = float(positive.max() / positive.min()) if positive.size else float("inf")
    first, second = _pair_indices(clean.shape[0], graph.data_seed)
    clean_dist = np.linalg.norm(clean[first] - clean[second], axis=1)
    observed_dist = np.linalg.norm(observed[first] - observed[second], axis=1)
    correlation = float(np.corrcoef(clean_dist, observed_dist)[0, 1]) if np.std(clean_dist) > 1e-12 and np.std(observed_dist) > 1e-12 else 0.0
    feature_role_indices = {role: [index for index, value in enumerate(graph.column_roles) if value == role] for role in COLUMN_ROLE_NAMES}
    return {
        "node_count": len(graph.node_ops),
        "edge_count": int(sum(len(parent) for parent in graph.node_parents)),
        "node_depth_min": int(min(graph.node_depths)),
        "node_depth_max": int(max(graph.node_depths)),
        "max_depth": int(max(graph.node_depths)),
        "leaf_path_depths": list(graph.leaf_path_depths),
        "informative_leaf_path_depths": [graph.leaf_path_depths[index] for index, role in enumerate(graph.column_roles) if role == "informative"],
        "parent_count_summary": {"min": int(min(map(len, graph.node_parents))), "max": int(max(map(len, graph.node_parents))), "mean": float(np.mean([len(parent) for parent in graph.node_parents]))},
        "operation_counts": {op: int(graph.node_ops.count(op)) for op in sorted(set(graph.node_ops))},
        "node_role_counts": {role: int(graph.node_roles.count(role)) for role in COLUMN_ROLE_NAMES},
        "feature_type_counts": {name: int(graph.feature_types.count(name)) for name in FEATURE_TYPE_NAMES},
        "feature_role_indices": feature_role_indices,
        "column_role_counts": {role: int(graph.column_roles.count(role)) for role in COLUMN_ROLE_NAMES},
        "column_permutation": list(graph.column_permutation),
        "stochastic_node_count": int(sum(op.startswith("tbp_") or op.startswith("tabicl_") for op in graph.node_ops)),
        "tree_node_count": int(sum(op in {"tabicl_tree", "tabicl_graph_tree", "tbp_dt", "tbp_et", "tbp_rf", "tbp_gb", "tbp_direct_rf"} for op in graph.node_ops)),
        "flow_node_count": int(sum(op == "flow_iresnet" for op in graph.node_ops)),
        "many_to_one_node_count": int(sum(op in {"tabicl_tree", "tabicl_graph_tree", "tbp_dt", "tbp_et", "tbp_rf", "tbp_gb", "tbp_direct_rf", "max", "min", "saturation", "tabicl_graph_disc"} for op in graph.node_ops)),
        "node_clipping_fraction": float(np.mean(clipping)),
        "head_metadata": list(head_metadata),
        "effective_rank": effective_rank,
        "condition_number": condition,
        "duplicate_row_count": int(observed.shape[0] - np.unique(observed, axis=0).shape[0]),
        "duplicate_column_count": int(observed.shape[1] - np.unique(observed.T, axis=0).shape[0]),
        "clean_raw_distance_correlation": correlation,
        "knn_neighborhood_preservation": _knn_preservation(clean, observed),
        "nonfinite_count": int(np.size(observed) - np.isfinite(observed).sum()),
        **missing,
    }


def generate_source_complexity_observation(clean_latent: np.ndarray, nuisance_roots: np.ndarray, graph: SourceComplexityGraphSpec, row_ids: np.ndarray) -> SourceObservationResult:
    clean = np.asarray(clean_latent, dtype=np.float64)
    nuisance = np.asarray(nuisance_roots, dtype=np.float64)
    if clean.ndim != 2 or clean.shape[1] != 128 or nuisance.ndim != 2 or nuisance.shape[0] != clean.shape[0]:
        raise ValueError("invalid clean/nuisance roots")
    graph.validate(len(graph.leaf_nodes))
    roots = np.column_stack((clean, nuisance))
    if roots.shape[1] != graph.root_dim:
        raise ValueError("root dimension mismatch")
    node_values: List[np.ndarray] = []
    clipping: List[float] = []
    for index, (operation, parents, node_seed) in enumerate(zip(graph.node_ops, graph.node_parents, graph.node_param_seeds)):
        inputs = np.column_stack([roots[:, parent] if parent < graph.root_dim else node_values[parent - graph.root_dim] for parent in parents])
        output = apply_label_free_mechanism(inputs, operation, row_ids, index, node_seed)
        if output is None:
            output = _analytic_node(inputs, operation, node_seed)
        scaled, fraction = _robust_scale(output)
        # Canonicalize sub-ulp library/order differences before descendants consume
        # the value. This is a reproducibility transform, not a data-quality gate.
        scaled = np.round(scaled, decimals=12)
        node_values.append(scaled)
        clipping.append(fraction)
    raw = np.column_stack([node_values[index] for index in graph.leaf_nodes])
    raw = raw[:, np.asarray(graph.column_permutation, dtype=np.int64)]
    raw, information_metadata = _information_stratum_transform(raw, graph, node_values, row_ids)
    types = tuple(graph.feature_types[index] for index in graph.column_permutation)
    roles = tuple(graph.column_roles[index] for index in graph.column_permutation)
    typed_columns = []
    head_metadata = []
    for feature, feature_type in enumerate(types):
        value, metadata = _typed_head(raw[:, feature], feature_type, row_ids, feature, graph.data_seed)
        if graph.information_stratum == "noisy_recoverable":
            hetero = graph.noise_scale * (0.5 + np.abs(value)) * stable_normal(row_ids, 110_000 + feature, graph.data_seed)
            outlier = (_stable_uniform(row_ids, 120_000 + feature, graph.data_seed) < 0.01).astype(np.float64) * 5.0 * stable_normal(row_ids, 130_000 + feature, graph.data_seed)
            metadata.update({"heteroscedastic_noise": True, "outlier_rate": 0.01})
        else:
            hetero = np.zeros_like(value)
            outlier = np.zeros_like(value)
            metadata.update({"heteroscedastic_noise": False, "outlier_rate": 0.0})
        typed_columns.append(value + hetero + outlier)
        head_metadata.append(metadata)
    before_missing = np.column_stack(typed_columns)
    output, mask, missing = _apply_missingness(before_missing, graph, row_ids)
    output = np.ascontiguousarray(_finite(output, limit=1.0e4), dtype=np.float32)
    certificate = _matrix_certificate(clean, before_missing, graph, clipping, head_metadata, missing)
    certificate["observation_family"] = graph.family
    certificate["information_stratum"] = graph.information_stratum
    certificate["information_stratum_metadata"] = information_metadata
    certificate["observation_map_reads_labels"] = False
    certificate["topology_seed"] = graph.topology_seed
    certificate["mechanism_seed"] = graph.mechanism_seed
    certificate["data_seed"] = graph.data_seed
    # Roles/types follow the final column permutation, matching output columns.
    certificate["feature_role_indices"] = {role: [idx for idx, value in enumerate(roles) if value == role] for role in COLUMN_ROLE_NAMES}
    return SourceObservationResult(output, np.ascontiguousarray(before_missing, dtype=np.float32), np.ascontiguousarray(mask, dtype=np.uint8), np.asarray([FEATURE_TYPE_CODES[name] for name in types], dtype=np.uint8), certificate)
