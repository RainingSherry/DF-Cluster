"""Separated V4 input, privileged target, and audit artifacts.

The input artifact is strictly X-only. The privileged target artifact contains
clean geometry and the synthetic Y/A_pair fields required by the single
pre-registered offline meta-pretraining co-assignment loss allowed by the
project rules. General model/input loaders must not open those fields. The
independent audit sidecar remains the only artifact for audit metrics and
outer evaluation labels.
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import os
from typing import Any, Dict

import h5py
import numpy as np

from .core import V4Task


INPUT_DATASETS = ("features", "missing_mask", "feature_types", "row_ids")
TARGET_DATASETS = (
    "clean_latent",
    "row_ids",
    "labels",
    "clean_gram",
    "pair_first",
    "pair_second",
    "clean_distances",
    "a_pair_bits",
)
PAIR_TARGET_PROTOCOL = "offline_meta_pretraining_coassignment_loss_only"


def _atomic_h5(path: Path, writer) -> None:
    path = Path(path)
    partial = path.with_name(path.name + ".partial")
    if path.exists() or partial.exists():
        raise FileExistsError("artifact already exists: %s" % path)
    with h5py.File(partial, "w") as handle:
        writer(handle)
        handle.flush()
    os.replace(partial, path)


def _pair_bits(labels: np.ndarray) -> tuple[np.ndarray, int]:
    values = np.asarray(labels)
    if values.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    n_rows = int(values.shape[0])
    upper = np.fromiter(
        (bool(values[first] == values[second]) for first in range(n_rows) for second in range(first + 1, n_rows)),
        dtype=np.bool_,
        count=n_rows * (n_rows - 1) // 2,
    )
    return np.packbits(upper, bitorder="little"), int(upper.size)


def _pair_seed(task: V4Task) -> int:
    return int(hashlib.sha256(task.metadata["task_id"].encode("utf-8")).hexdigest()[:16], 16) % (2**32)


def _write_input(path: Path, task: V4Task) -> None:
    def writer(handle: h5py.File) -> None:
        handle.attrs["schema_version"] = "dfcluster.generator_v4.input.v2"
        handle.attrs["task_id"] = task.metadata["task_id"]
        handle.attrs["artifact_sha256"] = task.metadata["artifact_sha256"]
        handle.attrs["config_sha256"] = task.metadata["config_sha256"]
        handle.attrs["source_sha256"] = task.metadata["source_sha256"]
        handle.attrs["labels_opened"] = False
        handle.attrs["clean_target_present"] = False
        handle.attrs["audit_fields_present"] = False
        handle.create_dataset("features", data=task.features, compression="gzip")
        handle.create_dataset("missing_mask", data=task.missing_mask, compression="gzip")
        handle.create_dataset("feature_types", data=task.feature_types)
        handle.create_dataset("row_ids", data=task.row_ids)

    _atomic_h5(path, writer)


def _write_target(path: Path, task: V4Task) -> None:
    geometry = task.geometry_targets(pair_seed=_pair_seed(task), num_pairs=4096)
    bits, bit_count = _pair_bits(task.labels)

    def writer(handle: h5py.File) -> None:
        handle.attrs["schema_version"] = "dfcluster.generator_v4.target.v2"
        handle.attrs["task_id"] = task.metadata["task_id"]
        handle.attrs["artifact_sha256"] = task.metadata["artifact_sha256"]
        handle.attrs["config_sha256"] = task.metadata["config_sha256"]
        handle.attrs["source_sha256"] = task.metadata["source_sha256"]
        handle.attrs["labels_present"] = True
        handle.attrs["labels_opened"] = True
        handle.attrs["clean_target_present"] = True
        handle.attrs["audit_fields_present"] = False
        handle.attrs["label_use_protocol"] = PAIR_TARGET_PROTOCOL
        handle.attrs["general_loader_may_open_labels"] = False
        handle.attrs["a_pair_bit_count"] = bit_count
        handle.attrs["a_pair_bit_order"] = "little"
        handle.attrs["a_pair_row_count"] = int(task.labels.shape[0])
        handle.create_dataset("clean_latent", data=task.clean_latent, compression="gzip")
        handle.create_dataset("row_ids", data=task.row_ids)
        handle.create_dataset("labels", data=task.labels, compression="gzip")
        handle.create_dataset("clean_gram", data=geometry["clean_gram"], compression="gzip")
        handle.create_dataset("pair_first", data=geometry["pair_first"])
        handle.create_dataset("pair_second", data=geometry["pair_second"])
        handle.create_dataset("clean_distances", data=geometry["clean_distances"], compression="gzip")
        handle.create_dataset("a_pair_bits", data=bits, compression="gzip")

    _atomic_h5(path, writer)


def _write_audit(path: Path, task: V4Task) -> None:
    if path.exists():
        raise FileExistsError("audit artifact already exists: %s" % path)
    payload = {
        "schema_version": "dfcluster.generator_v4.audit.v1",
        "task_id": task.metadata["task_id"],
        "labels": task.labels.astype(int).tolist(),
        "K": int(np.unique(task.labels).size),
        "metadata": task.metadata,
        "audit_only": True,
        "training_loader_may_open": False,
        "target_label_use_protocol": PAIR_TARGET_PROTOCOL,
    }
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def write_task_artifacts(root: Path, task: V4Task) -> Dict[str, Any]:
    """Write one separated task artifact under the approved target protocol."""

    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    input_path = root / "input.h5"
    target_path = root / "target.h5"
    audit_path = root / "audit.json"
    _write_input(input_path, task)
    _write_target(target_path, task)
    _write_audit(audit_path, task)
    return {
        "task_id": task.metadata["task_id"],
        "input_path": str(input_path),
        "target_path": str(target_path),
        "audit_path": str(audit_path),
        "labels_in_input": False,
        "labels_in_target": True,
        "labels_in_privileged_target": True,
        "labels_in_audit_sidecar": True,
        "labels_in_audit_only": False,
        "target_label_use_protocol": PAIR_TARGET_PROTOCOL,
        "general_input_loader_may_open_target_labels": False,
        "source_sha256": task.metadata["source_sha256"],
        "artifact_sha256": task.metadata["artifact_sha256"],
    }


def validate_input_artifact(path: Path) -> Dict[str, Any]:
    with h5py.File(path, "r") as handle:
        if set(handle.keys()) != set(INPUT_DATASETS):
            raise ValueError("input artifact dataset set is not exact")
        if bool(handle.attrs.get("labels_opened", True)):
            raise ValueError("input artifact is marked labels_opened")
        if bool(handle.attrs.get("clean_target_present", True)):
            raise ValueError("input artifact contains a clean target")
        if bool(handle.attrs.get("audit_fields_present", True)):
            raise ValueError("input artifact contains audit fields")
        return {"task_id": str(handle.attrs["task_id"]), "labels_opened": False}


def validate_target_artifact(path: Path) -> Dict[str, Any]:
    with h5py.File(path, "r") as handle:
        if set(handle.keys()) != set(TARGET_DATASETS):
            raise ValueError("target artifact dataset set is not exact")
        if not bool(handle.attrs.get("labels_present", False)):
            raise ValueError("target artifact lacks synthetic labels for the approved pair-loss protocol")
        if not bool(handle.attrs.get("clean_target_present", False)):
            raise ValueError("target artifact lacks clean target marker")
        if handle.attrs.get("label_use_protocol") != PAIR_TARGET_PROTOCOL:
            raise ValueError("target artifact label-use protocol is not approved")
        if bool(handle.attrs.get("general_loader_may_open_labels", True)):
            raise ValueError("general loader may not open target labels")
        if int(handle.attrs.get("a_pair_bit_count", -1)) != int(handle.attrs["a_pair_row_count"]) * (int(handle.attrs["a_pair_row_count"]) - 1) // 2:
            raise ValueError("bit-packed A_pair length is inconsistent")
        return {
            "task_id": str(handle.attrs["task_id"]),
            "labels_present": True,
            "label_use_protocol": PAIR_TARGET_PROTOCOL,
        }


def validate_audit_artifact(path: Path) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("audit_only") is not True or payload.get("training_loader_may_open") is not False:
        raise ValueError("audit sidecar is not isolated")
    if not isinstance(payload.get("labels"), list) or not isinstance(payload.get("K"), int):
        raise ValueError("audit sidecar lacks isolated labels/K")
    return {"task_id": payload["task_id"], "audit_only": True}