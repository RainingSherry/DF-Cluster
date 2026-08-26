"""Paired counterfactual observation-task contract for Generator V4."""

from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np

from .core import (
    OBSERVATION_FAMILIES,
    SourceComplexityGraphSpec,
    V4Config,
    V4Task,
    _array_digest,
    _canonical,
    _config_sha256,
    build_observation_graph,
    generate_observation,
    generate_v4_task,
    source_sha256,
)


def _parent_id(config: V4Config, clean: np.ndarray, nuisance: np.ndarray) -> str:
    digest = hashlib.sha256(_canonical({
        "split": config.split,
        "seed": config.seed,
        "task_index": config.task_index,
        "n_samples": config.n_samples,
        "n_clusters": config.n_clusters,
        "intrinsic_dim": config.intrinsic_dim,
    }))
    _array_digest(digest, clean)
    _array_digest(digest, nuisance)
    return "generator_v4_parent/%s/%d/%d/%s" % (
        config.split, config.seed, config.task_index, digest.hexdigest()[:16]
    )


def _graph_for_family(config: V4Config, family: str, family_index: int) -> SourceComplexityGraphSpec:
    graph_config = replace(config, observation_family=family, seed=int(np.random.SeedSequence([config.seed, family_index + 1, 0xC0FFEE]).generate_state(1, dtype=np.uint64)[0]))
    return build_observation_graph(graph_config)


def _assemble_counterfactual(
    config: V4Config,
    base: V4Task,
    family: str,
    family_index: int,
    parent_task_id: str,
) -> V4Task:
    graph = _graph_for_family(config, family, family_index)
    observed = generate_observation(
        base.clean_latent, base.nuisance_roots, graph, base.row_ids
    )
    config_for_family = replace(config, observation_family=family)
    source_hash = source_sha256()
    config_hash = _config_sha256(config_for_family)
    digest = hashlib.sha256(_canonical(asdict(config_for_family)))
    digest.update(parent_task_id.encode("utf-8"))
    for array in (
        observed.features,
        base.clean_latent,
        observed.missing_mask,
        observed.feature_types,
        base.labels,
        base.nuisance_roots,
    ):
        _array_digest(digest, array)
    artifact_hash = digest.hexdigest()
    metadata = {
        "generator": "dfcluster_generator_v4_p0_p1",
        "generator_version": 4,
        "task_id": "generator_v4/%s/%d/%d/%s" % (
            config.split, config.seed, config.task_index, family
        ),
        "parent_task_id": parent_task_id,
        "counterfactual_family": family,
        "task_fingerprint": artifact_hash[:16],
        "artifact_sha256": artifact_hash,
        "config_sha256": config_hash,
        "source_sha256": source_hash,
        "labels_isolated": True,
        "observation_map_reads_labels": False,
        "clean_signal_is_privileged_training_target": True,
        "observation_family": family,
        "cell_count": config.n_samples * config.n_features,
        "paired_counterfactual": True,
        "parent_roots_shared": True,
        **observed.certificate,
    }
    return V4Task(
        features=observed.features,
        clean_latent=base.clean_latent,
        missing_mask=observed.missing_mask,
        feature_types=observed.feature_types,
        labels=base.labels,
        nuisance_roots=base.nuisance_roots,
        observation_graph=graph,
        row_ids=base.row_ids,
        metadata=metadata,
    )


def generate_counterfactual_tasks(
    config: V4Config,
    observation_families: Sequence[str] | None = None,
) -> Dict[str, V4Task]:
    """Generate paired tasks sharing roots but not observation mechanisms."""

    config.validate()
    families = tuple(observation_families or OBSERVATION_FAMILIES)
    if not families:
        raise ValueError("at least one observation family is required")
    unknown = set(families) - set(OBSERVATION_FAMILIES)
    if unknown:
        raise ValueError("unknown observation families: %s" % sorted(unknown))
    base = generate_v4_task(config)
    parent_task_id = _parent_id(config, base.clean_latent, base.nuisance_roots)
    return {
        family: _assemble_counterfactual(config, base, family, index, parent_task_id)
        for index, family in enumerate(families)
    }
