"""Source-complexity, label-free observation-node mechanisms.

The code adapts audited upstream mechanisms to the V4 root-injection contract.
It never accepts labels, K, CLM, or generator difficulty metadata.

* TabICL Graph mechanisms follow the public graph library's random MLP, tree,
  RFF/GP, discretization, quadratic, EM-assignment and product families.
* TabICL MLP/Tree mechanisms follow the paper-era MLP-SCM/Tree-SCM idea of
  random transforms over supplied parent variables instead of sampled labels.
* MITRA-inspired TBP implements DT, ET, RF, GB and direct-RF with standard
  sklearn execution kernels and deterministic pseudo-targets. This is local
  code, not MITRA's unpublished generator implementation.
* ``flow_iresnet`` is a conditional one-dimensional iResNet block that is
  contractive/invertible with respect to its primary parent coordinate.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


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
    values = _splitmix64(keyed)
    return (values >> np.uint64(11)).astype(np.float64) / 9007199254740992.0


def stable_normal(row_ids: np.ndarray, column: int, seed: int) -> np.ndarray:
    first = np.maximum(_stable_uniform(row_ids, 2 * column + 1, seed), 1e-12)
    second = _stable_uniform(row_ids, 2 * column + 2, seed)
    return np.sqrt(-2.0 * np.log(first)) * np.cos(2.0 * np.pi * second)


def _rng(seed: int, node_index: int, salt: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(node_index), int(salt)]))


def _finite(values: np.ndarray) -> np.ndarray:
    return np.clip(np.nan_to_num(values, nan=0.0, posinf=20.0, neginf=-20.0), -20.0, 20.0)


def _canonical_fit_inputs(values: np.ndarray, row_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(np.asarray(row_ids, dtype=np.uint64), kind="stable")
    inverse = np.empty_like(order)
    inverse[order] = np.arange(order.size)
    return np.asarray(values[order], dtype=np.float64), np.asarray(row_ids)[order], inverse


def _sklearn_seed(seed: int, node_index: int, salt: int) -> int:
    return int((int(seed) + 1_000_003 * int(node_index) + int(salt)) % (2**32 - 1))


# ---- TabICL-inspired graph/MLP/tree mechanisms ---------------------------------

def tabicl_mlp_node(values: np.ndarray, row_ids: np.ndarray, node_index: int, seed: int) -> np.ndarray:
    """Block-sparse random MLP with sampled width/layers and layer noise."""

    generator = _rng(seed, node_index, 17)
    input_dim = values.shape[1]
    width = int(generator.integers(4, min(128, max(5, 4 * input_dim)) + 1))
    layers = int(generator.integers(1, 4))
    current = np.asarray(values, dtype=np.float64)
    for layer in range(layers):
        output_dim = 1 if layer == layers - 1 else width
        weights = generator.normal(size=(current.shape[1], output_dim)) / math.sqrt(current.shape[1])
        keep = generator.random(size=weights.shape) > 0.20
        current = current @ (weights * keep) + generator.normal(scale=0.03, size=output_dim)
        if layer < layers - 1:
            activation = ("tanh", "gelu", "sin")[layer % 3]
            if activation == "tanh":
                current = np.tanh(current)
            elif activation == "gelu":
                current = 0.5 * current * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (current + 0.044715 * current**3)))
            else:
                current = np.sin(current)
            current += 0.01 * stable_normal(row_ids, 20_000 + 10 * node_index + layer, seed)[:, None]
    return _finite(current.reshape(-1))


def tabicl_oblivious_tree_node(values: np.ndarray, row_ids: np.ndarray, node_index: int, seed: int) -> np.ndarray:
    """Random oblivious-tree ensemble adapted from TabICL's graph tree function."""

    generator = _rng(seed, node_index, 23)
    inputs, canonical_rows, inverse = _canonical_fit_inputs(values, row_ids)
    importance = np.maximum(inputs.std(axis=0), 1e-8)
    importance = importance / importance.sum()
    n_trees = int(generator.integers(1, 9))
    depth = int(generator.integers(1, 9))
    output = np.zeros(inputs.shape[0], dtype=np.float64)
    for tree in range(n_trees):
        split_dims = generator.choice(inputs.shape[1], size=depth, replace=True, p=importance)
        threshold_rows = generator.integers(0, inputs.shape[0], size=depth)
        thresholds = inputs[threshold_rows, split_dims]
        sides = inputs[:, split_dims] > thresholds[None, :]
        indices = sides.astype(np.int64) @ (2 ** np.arange(depth, dtype=np.int64))
        leaves = generator.normal(size=2**depth)
        output += leaves[indices]
    output /= float(n_trees)
    return _finite(output[inverse])


def graph_rff_node(values: np.ndarray, row_ids: np.ndarray, node_index: int, seed: int) -> np.ndarray:
    """Cauchy-spectrum random Fourier feature function, following GraphSCM GP family."""

    generator = _rng(seed, node_index, 31)
    frequencies = int(generator.integers(16, 65))
    cauchy = np.tan(np.pi * (generator.random(frequencies) - 0.5))
    directions = generator.normal(size=(values.shape[1], frequencies))
    directions /= np.maximum(np.linalg.norm(directions, axis=0, keepdims=True), 1e-8)
    phase = generator.uniform(0.0, 2.0 * np.pi, size=frequencies)
    weights = generator.normal(size=frequencies) / math.sqrt(frequencies)
    return _finite(np.cos(values @ (directions * cauchy[None, :]) + phase) @ weights)


def graph_discretization_node(values: np.ndarray, row_ids: np.ndarray, node_index: int, seed: int) -> np.ndarray:
    generator = _rng(seed, node_index, 37)
    inputs, _, inverse = _canonical_fit_inputs(values, row_ids)
    centers = inputs[generator.choice(inputs.shape[0], size=min(inputs.shape[0], int(generator.integers(2, 17))), replace=False)]
    sqdist = ((inputs[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    targets = generator.normal(size=centers.shape[0])
    return _finite(targets[sqdist.argmin(axis=1)][inverse])


def graph_quadratic_node(values: np.ndarray, row_ids: np.ndarray, node_index: int, seed: int) -> np.ndarray:
    generator = _rng(seed, node_index, 41)
    width = min(values.shape[1], 20)
    chosen = generator.choice(values.shape[1], size=width, replace=False)
    sub = values[:, chosen]
    matrix = generator.normal(size=(width, width)) / max(width, 1)
    return _finite(np.einsum("bi,ij,bj->b", sub, matrix, sub) + sub @ generator.normal(size=width) / math.sqrt(width))


def graph_em_node(values: np.ndarray, row_ids: np.ndarray, node_index: int, seed: int) -> np.ndarray:
    generator = _rng(seed, node_index, 43)
    components = int(generator.integers(2, 9))
    inputs, _, inverse = _canonical_fit_inputs(values, row_ids)
    anchors = inputs[generator.choice(inputs.shape[0], size=components, replace=False)]
    scale = np.exp(generator.uniform(np.log(0.3), np.log(2.0), size=components))
    distances = ((inputs[:, None, :] - anchors[None, :, :]) ** 2).sum(axis=2)
    weights = np.exp(-distances / (2.0 * scale[None, :] ** 2))
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    output = weights @ generator.normal(size=components)
    return _finite(output[inverse])


def graph_product_node(values: np.ndarray, row_ids: np.ndarray, node_index: int, seed: int) -> np.ndarray:
    return _finite(graph_rff_node(values, row_ids, node_index, seed) * graph_quadratic_node(values, row_ids, node_index, seed + 1))


def flow_iresnet_node(values: np.ndarray, row_ids: np.ndarray, node_index: int, seed: int) -> np.ndarray:
    """Conditional scalar iResNet: invertible in the first parent coordinate."""

    generator = _rng(seed, node_index, 47)
    primary = np.asarray(values[:, 0], dtype=np.float64)
    conditioning = values[:, 1:] @ (generator.normal(size=max(values.shape[1] - 1, 1))[: max(values.shape[1] - 1, 0)] / math.sqrt(max(values.shape[1] - 1, 1))) if values.shape[1] > 1 else 0.0
    beta = float(generator.uniform(0.20, 0.80))
    alpha = float(generator.uniform(-0.80 / beta, 0.80 / beta))
    bias = float(generator.normal(scale=0.10))
    # d/d primary = 1 + alpha*beta*sech^2(.) remains strictly positive.
    return _finite(primary + alpha * np.tanh(beta * primary + conditioning + bias))


# ---- MITRA-inspired tree prior ---------------------------------------------------

def _pseudo_target(values: np.ndarray, row_ids: np.ndarray, node_index: int, seed: int, family: str) -> np.ndarray:
    base = values[:, 0] + 0.25 * np.sum(np.tanh(values[:, 1 : min(values.shape[1], 4)]), axis=1)
    if family == "direct_rf":
        return np.where(base > np.median(base), 1.0, -1.0) + 0.03 * stable_normal(row_ids, 60_000 + node_index, seed)
    if family == "gb":
        return np.tanh(base) + 0.20 * np.sin(2.0 * base) + 0.03 * stable_normal(row_ids, 60_000 + node_index, seed)
    return np.sin(base) + 0.35 * np.prod(np.tanh(values[:, : min(3, values.shape[1])]), axis=1) + 0.05 * stable_normal(row_ids, 60_000 + node_index, seed)


def _fit_tbp(values: np.ndarray, row_ids: np.ndarray, node_index: int, seed: int, family: str) -> np.ndarray:
    from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
    from sklearn.tree import DecisionTreeRegressor

    inputs, canonical_rows, inverse = _canonical_fit_inputs(values, row_ids)
    target = _pseudo_target(inputs, canonical_rows, node_index, seed, family)
    random_state = _sklearn_seed(seed, node_index, {"dt": 1, "et": 2, "rf": 3, "gb": 4, "direct_rf": 5}[family])
    if family == "dt":
        model = DecisionTreeRegressor(max_depth=5, splitter="random", random_state=random_state)
    elif family == "et":
        model = ExtraTreesRegressor(n_estimators=8, max_depth=5, max_features="sqrt", random_state=random_state, n_jobs=1)
    elif family == "rf":
        model = RandomForestRegressor(n_estimators=8, max_depth=5, max_features="sqrt", bootstrap=True, random_state=random_state, n_jobs=1)
    elif family == "gb":
        model = GradientBoostingRegressor(n_estimators=16, max_depth=3, learning_rate=0.05, random_state=random_state, loss="huber")
    else:
        model = RandomForestRegressor(n_estimators=16, max_depth=4, max_features=1.0, bootstrap=False, random_state=random_state, n_jobs=1)
    model.fit(inputs, target)
    return _finite(np.asarray(model.predict(inputs), dtype=np.float64)[inverse])


def apply_label_free_mechanism(values: np.ndarray, operation: str, row_ids: np.ndarray, node_index: int, seed: int) -> Optional[np.ndarray]:
    """Return a source-complexity node output; analytic core ops return ``None``."""

    if operation == "tabicl_mlp":
        return tabicl_mlp_node(values, row_ids, node_index, seed)
    if operation in {"tabicl_tree", "tabicl_graph_tree"}:
        return tabicl_oblivious_tree_node(values, row_ids, node_index, seed)
    if operation == "tabicl_graph_gp":
        return graph_rff_node(values, row_ids, node_index, seed)
    if operation == "tabicl_graph_disc":
        return graph_discretization_node(values, row_ids, node_index, seed)
    if operation == "tabicl_graph_quad":
        return graph_quadratic_node(values, row_ids, node_index, seed)
    if operation == "tabicl_graph_em":
        return graph_em_node(values, row_ids, node_index, seed)
    if operation == "tabicl_graph_prod":
        return graph_product_node(values, row_ids, node_index, seed)
    if operation == "flow_iresnet":
        return flow_iresnet_node(values, row_ids, node_index, seed)
    if operation == "tbp_dt":
        return _fit_tbp(values, row_ids, node_index, seed, "dt")
    if operation == "tbp_et":
        return _fit_tbp(values, row_ids, node_index, seed, "et")
    if operation == "tbp_rf":
        return _fit_tbp(values, row_ids, node_index, seed, "rf")
    if operation == "tbp_gb":
        return _fit_tbp(values, row_ids, node_index, seed, "gb")
    if operation == "tbp_direct_rf":
        return _fit_tbp(values, row_ids, node_index, seed, "direct_rf")
    return None
