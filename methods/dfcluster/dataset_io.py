"""Label-isolated, resumable HDF5 shard I/O for heterogeneous tasks.

The training and audit views of a task intentionally live in different files.
The feature file can therefore be handed to a training worker without giving
that worker a path to (or an attribute containing) the labels.  A task is one
logical record in one pair of files; its arrays may have different column
counts from every other task.

Only the public functions in this module should be used to create these
artifacts.  Files are written to ``*.partial`` names, validated after closing,
and then promoted with :func:`os.replace`.
"""

from __future__ import annotations

from base64 import b64encode
from dataclasses import dataclass, fields, is_dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import h5py
import numpy as np


SCHEMA_VERSION = "dfcluster.paired_shard.v1"
FEATURE_DATASETS = ("features", "clean_signal", "missing_mask", "feature_types")
LABEL_DATASETS = ("labels",)
_FORBIDDEN_NAMES = {
    "y",
    "k",
    "clm",
    "labelpath",
    "labelspath",
    "labelfile",
    "labelfilepath",
    "labelsfile",
    "labelsfilepath",
    "labelshard",
}
_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9_.-]+$")


class ShardError(RuntimeError):
    """Base class for shard I/O errors."""


class ShardValidationError(ShardError, ValueError):
    """Raised when a shard or a pair does not satisfy the schema."""


class ShardConflictError(ShardError, FileExistsError):
    """Raised when a completed or one-sided pair conflicts with a write."""


@dataclass(frozen=True)
class TaskPayload:
    """Arrays and audit-only values needed to write one task.

    ``clean_signal`` may have any number of columns (including zero), so tasks
    with different intrinsic dimensions can be written independently.  The
    two audit values are deliberately not accepted as feature-file metadata.
    """

    task_id: str
    features: np.ndarray
    clean_signal: np.ndarray
    missing_mask: np.ndarray
    feature_types: np.ndarray
    labels: np.ndarray
    K: int
    CLM: float
    feature_attrs: Mapping[str, Any] | None = None
    # Canonical CLM audit fields.  ``CLM`` remains a compatibility alias for
    # older callers, but new records should provide the explicit observed
    # value and (when available) the clean-control value.
    clm_cha_observed: float | None = None
    clm_cha_clean_control: float | None = None
    clm_status: str = "ok"
    clm_error: str | None = None
    audit_attrs: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class ShardPaths:
    """Final and temporary paths for one worker-owned task pair."""

    features: Path
    labels: Path

    @property
    def features_partial(self) -> Path:
        return Path(f"{self.features}.partial")

    @property
    def labels_partial(self) -> Path:
        return Path(f"{self.labels}.partial")


@dataclass(frozen=True)
class ShardValidation:
    """Facts obtained by validating a completed pair."""

    task_id: str
    task_count: int
    n_samples: int
    n_features: int
    clean_signal_dim: int
    feature_sha256: str
    labels_sha256: str
    K: int
    CLM: float
    clm_cha_observed: float | None = None
    clm_cha_clean_control: float | None = None
    clm_status: str | None = None
    clm_error: str | None = None


@dataclass(frozen=True)
class MultiShardValidation:
    """Validation facts for a worker file containing several task groups."""

    task_ids: tuple[str, ...]
    task_count: int
    records: tuple[ShardValidation, ...]

    def by_task_id(self) -> dict[str, ShardValidation]:
        return {record.task_id: record for record in self.records}


@dataclass(frozen=True)
class ShardWriteResult:
    """Result of a fresh or resumed write."""

    task_id: str
    worker_id: str
    paths: ShardPaths
    training_manifest: Mapping[str, Any]
    audit_manifest: Mapping[str, Any]
    resumed: bool = False

    @property
    def features_path(self) -> Path:
        return self.paths.features

    @property
    def labels_path(self) -> Path:
        return self.paths.labels


@dataclass(frozen=True)
class MultiShardWriteResult:
    """Result of writing a multi-task worker pair."""

    paths: ShardPaths
    task_ids: tuple[str, ...]
    training_manifest_records: tuple[Mapping[str, Any], ...]
    audit_manifest_records: tuple[Mapping[str, Any], ...]
    worker_id: str
    resumed: bool = False

    @property
    def features_path(self) -> Path:
        return self.paths.features

    @property
    def labels_path(self) -> Path:
        return self.paths.labels

    # Singular-looking aliases keep the return object convenient when a
    # caller treats a one-task multi shard uniformly with ShardWriteResult.
    @property
    def training_manifest(self) -> tuple[Mapping[str, Any], ...]:
        return self.training_manifest_records

    @property
    def audit_manifest(self) -> tuple[Mapping[str, Any], ...]:
        return self.audit_manifest_records


def _canonical_value(value: Any) -> Any:
    """Return a JSON-compatible, order-independent representation.

    In particular, mappings are sorted by canonicalized keys and ndarray
    bytes include dtype and shape.  This avoids Python's process-dependent
    ``repr(dict)`` and makes hashes reproducible across workers.
    """

    if is_dataclass(value) and not isinstance(value, type):
        return _canonical_value({field.name: getattr(value, field.name) for field in fields(value)})
    if isinstance(value, Mapping):
        pairs = [(_canonical_value(key), _canonical_value(item)) for key, item in value.items()]
        pairs.sort(key=lambda pair: _json_dumps(pair[0]))
        return {"__mapping__": [[key, item] for key, item in pairs]}
    if isinstance(value, (set, frozenset)):
        items = [_canonical_value(item) for item in value]
        items.sort(key=_json_dumps)
        return {"__set__": items}
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        if array.dtype.hasobject:
            return {"__ndarray__": {"dtype": "object", "shape": list(array.shape), "data": _canonical_value(array.tolist())}}
        contiguous = np.ascontiguousarray(array)
        return {
            "__ndarray__": {
                "dtype": contiguous.dtype.str,
                "shape": list(contiguous.shape),
                "data_b64": b64encode(contiguous.tobytes(order="C")).decode("ascii"),
            }
        }
    if isinstance(value, np.generic):
        return _canonical_value(value.item())
    if isinstance(value, bytes):
        return {"__bytes__": b64encode(value).decode("ascii")}
    if isinstance(value, Path):
        return {"__path__": str(value)}
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("stable hashing does not accept non-finite floats")
        # JSON preserves the distinction between -0.0 and 0.0, which is useful
        # for an artifact hash and is deterministic on Python 3.10+.
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, bytearray):
        return {"__bytes__": b64encode(bytes(value)).decode("ascii")}
    raise TypeError(f"cannot stably hash value of type {type(value).__name__}")


def _json_dumps(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def stable_sha256(value: Any) -> str:
    """Hash nested values deterministically, independent of mapping order."""

    return sha256(_json_dumps(_canonical_value(value)).encode("utf-8")).hexdigest()


# A short alias is convenient for callers that use ``stable_hash`` in a
# manifest builder.  Keep one implementation so the contract cannot diverge.
stable_hash = stable_sha256


def _safe_component(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not value or value in {".", ".."}:
        raise ValueError(f"{name} must be a non-empty path-safe string")
    if not _SAFE_TOKEN.fullmatch(value):
        raise ValueError(f"{name} contains path separators or unsafe characters")
    return value


def shard_paths(
    output_dir: str | os.PathLike[str],
    task_id: str,
    *,
    worker_id: str = "worker-0",
) -> ShardPaths:
    """Return worker-specific final paths for one task.

    The worker component prevents two workers writing the same task stem by
    accident.  The returned names are deterministic and contain no secrets.
    """

    task_token = _safe_component(task_id, name="task_id")
    worker_token = _safe_component(worker_id, name="worker_id")
    directory = Path(output_dir)
    stem = f"{task_token}__{worker_token}"
    return ShardPaths(
        features=directory / f"{stem}.features.h5",
        labels=directory / f"{stem}.labels.h5",
    )


# A descriptive alias for callers who prefer the pair terminology.
paired_shard_paths = shard_paths


def _get_field(source: Any, names: Sequence[str], default: Any = None) -> Any:
    if isinstance(source, Mapping):
        for name in names:
            if name in source:
                return source[name]
        return default
    for name in names:
        if hasattr(source, name):
            return getattr(source, name)
    return default


def _task_from_input(
    task: Any,
    *,
    task_id: str | None,
    features: Any,
    clean_signal: Any,
    missing_mask: Any,
    feature_types: Any,
    labels: Any,
    K: Any,
    CLM: Any,
    feature_attrs: Mapping[str, Any] | None,
    clm_cha_observed: Any = None,
    clm_cha_clean_control: Any = None,
    clm_status: Any = None,
    clm_error: Any = None,
    audit_attrs: Mapping[str, Any] | None = None,
) -> TaskPayload:
    """Accept a dataclass, mapping, or explicit keyword arrays."""

    if task is None:
        source: Any = {}
    else:
        source = task
    task_id_value = task_id if task_id is not None else _get_field(source, ("task_id", "id"))
    features_value = features if features is not None else _get_field(source, ("features", "X"))
    clean_value = clean_signal if clean_signal is not None else _get_field(source, ("clean_signal", "clean_latent", "Z"))
    missing_value = missing_mask if missing_mask is not None else _get_field(source, ("missing_mask", "mask"))
    types_value = feature_types if feature_types is not None else _get_field(source, ("feature_types", "types"))
    labels_value = labels if labels is not None else _get_field(source, ("labels", "Y", "y"))
    k_value = K if K is not None else _get_field(source, ("K", "k"))
    clm_value = CLM if CLM is not None else _get_field(source, ("CLM", "clm"))
    observed_value = (
        clm_cha_observed
        if clm_cha_observed is not None
        else _get_field(source, ("clm_cha_observed", "clm_observed", "cha_observed"))
    )
    clean_control_value = (
        clm_cha_clean_control
        if clm_cha_clean_control is not None
        else _get_field(source, ("clm_cha_clean_control", "clm_clean_control", "cha_clean_control"))
    )
    status_value = clm_status if clm_status is not None else _get_field(source, ("clm_status",), "ok")
    error_value = clm_error if clm_error is not None else _get_field(source, ("clm_error",), None)
    audit_attrs_value = audit_attrs if audit_attrs is not None else _get_field(source, ("audit_attrs",), {})
    attrs_value = feature_attrs if feature_attrs is not None else _get_field(source, ("feature_attrs",), {})

    metadata = _get_field(source, ("metadata",), {})
    if isinstance(metadata, Mapping):
        if task_id_value is None:
            task_id_value = metadata.get("task_id")
        if k_value is None:
            k_value = metadata.get("K", metadata.get("k"))
        if clm_value is None:
            clm_value = metadata.get("CLM", metadata.get("clm"))
        if observed_value is None:
            observed_value = metadata.get("clm_cha_observed")
        if clean_control_value is None:
            clean_control_value = metadata.get("clm_cha_clean_control")
        if clm_status is None:
            status_value = metadata.get("clm_status", status_value)
        if clm_error is None:
            error_value = metadata.get("clm_error", error_value)

    if task_id_value is None:
        raise ValueError("task_id is required")
    if features_value is None:
        raise ValueError("features is required")
    if clean_value is None:
        raise ValueError("clean_signal is required")
    if labels_value is None:
        raise ValueError("labels are required")
    if k_value is None:
        # K is audit-only but can be inferred without changing the feature
        # file.  Explicit K remains preferable for a frozen benchmark.
        raw_labels = np.asarray(labels_value)
        if raw_labels.size == 0:
            raise ValueError("K is required when labels is empty")
        k_value = int(np.unique(raw_labels).size)
    if clm_value is None and observed_value is not None:
        clm_value = observed_value
    if observed_value is None and clm_value is not None:
        observed_value = clm_value
    if clm_value is None:
        raise ValueError("CLM is required for the audit shard")

    raw_features = np.asarray(features_value)
    if missing_value is None:
        missing_value = np.zeros(raw_features.shape, dtype=np.bool_)
    if types_value is None:
        if raw_features.ndim != 2:
            raise ValueError("feature_types is required when features is not rank-2")
        types_value = np.zeros(raw_features.shape[1], dtype=np.uint8)

    return TaskPayload(
        task_id=str(task_id_value),
        features=np.asarray(features_value),
        clean_signal=np.asarray(clean_value),
        missing_mask=np.asarray(missing_value),
        feature_types=np.asarray(types_value),
        labels=np.asarray(labels_value),
        K=int(k_value),
        CLM=float(clm_value),
        feature_attrs=attrs_value or {},
        clm_cha_observed=float(observed_value),
        clm_cha_clean_control=(None if clean_control_value is None else float(clean_control_value)),
        clm_status=str(status_value or "ok"),
        clm_error=(None if error_value is None else str(error_value)),
        audit_attrs=audit_attrs_value or {},
    )


def _normalise_task(task: TaskPayload) -> TaskPayload:
    """Validate shapes and cast storage arrays to the frozen dtypes."""

    _safe_component(task.task_id, name="task_id")
    features = np.asarray(task.features, dtype=np.float32)
    clean_signal = np.asarray(task.clean_signal, dtype=np.float32)
    if features.ndim != 2:
        raise ShardValidationError(f"features must be rank-2, got {features.shape}")
    if clean_signal.ndim != 2:
        raise ShardValidationError(f"clean_signal must be rank-2, got {clean_signal.shape}")
    if features.shape[0] < 1:
        raise ShardValidationError("features must contain at least one sample")
    if not np.isfinite(features).all():
        raise ShardValidationError("features contains non-finite values")
    if not np.isfinite(clean_signal).all():
        raise ShardValidationError("clean_signal contains non-finite values")

    missing_mask = np.asarray(task.missing_mask)
    if missing_mask.shape != features.shape:
        raise ShardValidationError(
            f"missing_mask shape {missing_mask.shape} does not match features {features.shape}"
        )
    if missing_mask.dtype.kind not in "bui":
        raise ShardValidationError("missing_mask must be boolean or uint8-like")
    if missing_mask.dtype.kind == "i" and (
        np.any(missing_mask < 0) or np.any(missing_mask > 1)
    ):
        raise ShardValidationError("integer missing_mask values must be 0 or 1")
    missing_mask = missing_mask.astype(np.bool_, copy=False)

    feature_types = np.asarray(task.feature_types)
    if feature_types.shape != (features.shape[1],):
        raise ShardValidationError(
            f"feature_types shape {feature_types.shape} does not match feature width {features.shape[1]}"
        )
    if feature_types.dtype.kind not in "biu":
        raise ShardValidationError("feature_types must be an unsigned integer array")
    if np.any(feature_types < 0) or np.any(feature_types > 255):
        raise ShardValidationError("feature_types values must fit uint8")
    feature_types = feature_types.astype(np.uint8, copy=False)

    raw_labels = np.asarray(task.labels)
    if raw_labels.ndim != 1 or raw_labels.shape[0] != features.shape[0]:
        raise ShardValidationError(
            f"labels shape {raw_labels.shape} does not match sample count {features.shape[0]}"
        )
    if raw_labels.dtype.kind not in "biuf":
        raise ShardValidationError("labels must be integer-valued")
    if raw_labels.dtype.kind == "f" and (
        not np.isfinite(raw_labels).all() or not np.equal(raw_labels, np.floor(raw_labels)).all()
    ):
        raise ShardValidationError("labels must contain finite integer values")
    if np.any(raw_labels < np.iinfo(np.int32).min) or np.any(raw_labels > np.iinfo(np.int32).max):
        raise ShardValidationError("labels do not fit int32")
    labels = raw_labels.astype(np.int32, copy=False)

    if task.K <= 0 or task.K > np.iinfo(np.int32).max:
        raise ShardValidationError("K must be a positive int32-compatible value")
    observed_clm = task.clm_cha_observed if task.clm_cha_observed is not None else task.CLM
    if not math.isfinite(observed_clm):
        raise ShardValidationError("clm_cha_observed must be finite")
    clean_control = task.clm_cha_clean_control
    if clean_control is not None and not math.isfinite(clean_control):
        raise ShardValidationError("clm_cha_clean_control must be finite when present")
    if not isinstance(task.clm_status, str) or not task.clm_status:
        raise ShardValidationError("clm_status must be a non-empty string")
    _validate_feature_attrs(task.feature_attrs or {})
    _validate_audit_attrs(task.audit_attrs or {})
    return TaskPayload(
        task_id=task.task_id,
        features=np.ascontiguousarray(features),
        clean_signal=np.ascontiguousarray(clean_signal),
        missing_mask=np.ascontiguousarray(missing_mask),
        feature_types=np.ascontiguousarray(feature_types),
        labels=np.ascontiguousarray(labels),
        K=task.K,
        CLM=float(observed_clm),
        feature_attrs=dict(task.feature_attrs or {}),
        clm_cha_observed=float(observed_clm),
        clm_cha_clean_control=(None if clean_control is None else float(clean_control)),
        clm_status=task.clm_status,
        clm_error=(None if task.clm_error is None else str(task.clm_error)),
        audit_attrs=dict(task.audit_attrs or {}),
    )


def _name_token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _contains_forbidden_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if _name_token(key) in _FORBIDDEN_NAMES:
                return True
            if _contains_forbidden_key(item):
                return True
    elif isinstance(value, (list, tuple, set, frozenset)):
        return any(_contains_forbidden_key(item) for item in value)
    elif is_dataclass(value) and not isinstance(value, type):
        return _contains_forbidden_key({field.name: getattr(value, field.name) for field in fields(value)})
    return False


def _validate_feature_attrs(attrs: Mapping[str, Any]) -> None:
    if not isinstance(attrs, Mapping):
        raise TypeError("feature_attrs must be a mapping")
    if _contains_forbidden_key(attrs):
        raise ShardValidationError("feature attrs contain label/audit-only fields")
    for key in attrs:
        if not isinstance(key, str) or not key or "/" in key:
            raise ValueError("feature attribute names must be non-empty strings without '/'")


def _validate_audit_attrs(attrs: Mapping[str, Any]) -> None:
    """Validate optional label/audit metadata without permitting collisions."""

    if not isinstance(attrs, Mapping):
        raise TypeError("audit_attrs must be a mapping")
    reserved = {"k", "clm", "clm_cha_observed", "clm_cha_clean_control", "clm_status", "clm_error"}
    for key in attrs:
        if not isinstance(key, str) or not key or "/" in key:
            raise ValueError("audit attribute names must be non-empty strings without '/'")
        if _name_token(key) in reserved:
            raise ShardValidationError(f"audit_attrs cannot override reserved field {key!r}")


def _compression_kwargs(compression: str | None) -> dict[str, Any]:
    if compression is None or str(compression).lower() in {"none", "off", "false"}:
        return {}
    value = str(compression).lower()
    if value == "lzf":
        return {"compression": "lzf"}
    if value in {"gzip1", "gzip-1", "gzip"}:
        return {"compression": "gzip", "compression_opts": 1}
    raise ValueError("compression must be 'lzf', 'gzip1', or None")


def _attr_value(value: Any) -> Any:
    """Convert a metadata value to a h5py-compatible scalar."""

    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, bytes, bool, int, float)) or value is None:
        if value is None:
            return "null"
        return value
    # JSON is deterministic and keeps nested metadata readable.  It is also
    # safer than letting h5py infer an object dtype for arbitrary Python data.
    return _json_dumps(_canonical_value(value))


def _set_attrs(target: h5py.Group, attrs: Mapping[str, Any]) -> None:
    for key, value in attrs.items():
        target.attrs[key] = _attr_value(value)


def _feature_digest(task: TaskPayload) -> str:
    return stable_sha256(
        {
            "features": task.features,
            "clean_signal": task.clean_signal,
            "missing_mask": task.missing_mask,
            "feature_types": task.feature_types,
        }
    )


def _labels_digest(task: TaskPayload) -> str:
    return stable_sha256(
        {
            "labels": task.labels,
            "K": task.K,
            "clm_cha_observed": task.clm_cha_observed if task.clm_cha_observed is not None else task.CLM,
            "clm_cha_clean_control": task.clm_cha_clean_control,
            "clm_status": task.clm_status,
            "clm_error": task.clm_error,
        }
    )


def _write_feature_file(path: Path, task: TaskPayload, worker_id: str, compression: str | None) -> None:
    options = _compression_kwargs(compression)
    feature_digest = _feature_digest(task)
    with h5py.File(path, "w") as handle:
        handle.attrs["schema_version"] = SCHEMA_VERSION
        handle.attrs["task_id"] = task.task_id
        handle.attrs["task_count"] = 1
        handle.attrs["worker_id"] = worker_id
        handle.attrs["n_samples"] = task.features.shape[0]
        handle.attrs["n_features"] = task.features.shape[1]
        handle.attrs["feature_sha256"] = feature_digest
        group = handle.create_group("features")
        group.create_dataset("features", data=task.features, **options)
        group.create_dataset("clean_signal", data=task.clean_signal, **options)
        group.create_dataset("missing_mask", data=task.missing_mask, **options)
        group.create_dataset("feature_types", data=task.feature_types, **options)
        _set_attrs(group, {"feature_schema": "float32+clean_signal+mask+types", **(task.feature_attrs or {})})
        handle.flush()


def _write_label_file(path: Path, task: TaskPayload, worker_id: str, compression: str | None) -> None:
    options = _compression_kwargs(compression)
    labels_digest = _labels_digest(task)
    with h5py.File(path, "w") as handle:
        handle.attrs["schema_version"] = SCHEMA_VERSION
        handle.attrs["task_id"] = task.task_id
        handle.attrs["task_count"] = 1
        handle.attrs["worker_id"] = worker_id
        handle.attrs["n_samples"] = task.labels.shape[0]
        handle.attrs["labels_sha256"] = labels_digest
        group = handle.create_group("labels")
        group.create_dataset("labels", data=task.labels, **options)
        # These are intentionally confined to the audit file/group.  CLM is
        # retained as a compatibility alias; the explicit CHA fields are the
        # authoritative names for new records.
        group.attrs["K"] = task.K
        group.attrs["CLM"] = task.clm_cha_observed if task.clm_cha_observed is not None else task.CLM
        group.attrs["clm_cha_observed"] = task.clm_cha_observed if task.clm_cha_observed is not None else task.CLM
        if task.clm_cha_clean_control is not None:
            group.attrs["clm_cha_clean_control"] = task.clm_cha_clean_control
        group.attrs["clm_status"] = task.clm_status
        if task.clm_error is not None:
            group.attrs["clm_error"] = task.clm_error
        _set_attrs(group, task.audit_attrs or {})
        handle.flush()


def _fsync_file(path: Path) -> None:
    """Best-effort durability barrier after a temporary file is closed."""

    try:
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
    except OSError:
        # Some network filesystems do not expose fsync on read handles; the
        # atomic rename still provides the required visibility guarantee.
        pass


def _read_text_attr(attrs: h5py.AttributeManager, key: str) -> str:
    if key not in attrs:
        raise ShardValidationError(f"missing required attribute {key!r}")
    value = attrs[key]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return str(value)


def _assert_no_forbidden_names(name: str, values: Mapping[str, Any]) -> None:
    for key in values:
        if _name_token(key) in _FORBIDDEN_NAMES:
            raise ShardValidationError(f"{name} contains forbidden field {key!r}")


def _validate_feature_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ShardValidationError(f"feature shard does not exist: {path}")
    with h5py.File(path, "r") as handle:
        if set(handle.keys()) != {"features"}:
            raise ShardValidationError("feature shard must contain only a /features group")
        task_id = _read_text_attr(handle.attrs, "task_id")
        schema = _read_text_attr(handle.attrs, "schema_version")
        if schema != SCHEMA_VERSION:
            raise ShardValidationError(f"unsupported feature shard schema: {schema}")
        task_count = int(handle.attrs.get("task_count", -1))
        if task_count != 1:
            raise ShardValidationError(f"feature task_count must be 1, got {task_count}")
        group = handle["features"]
        if set(group.keys()) != set(FEATURE_DATASETS):
            raise ShardValidationError(
                f"feature group datasets must be {FEATURE_DATASETS}, got {tuple(group.keys())}"
            )
        _assert_no_forbidden_names("feature root attrs", dict(handle.attrs))
        _assert_no_forbidden_names("feature group attrs", dict(group.attrs))
        features = group["features"]
        clean_signal = group["clean_signal"]
        missing_mask = group["missing_mask"]
        feature_types = group["feature_types"]
        if features.dtype != np.dtype(np.float32) or features.ndim != 2:
            raise ShardValidationError("features must be a rank-2 float32 dataset")
        if clean_signal.dtype != np.dtype(np.float32) or clean_signal.ndim != 2:
            raise ShardValidationError("clean_signal must be a rank-2 float32 dataset")
        if missing_mask.dtype not in (np.dtype(np.bool_), np.dtype(np.uint8)):
            raise ShardValidationError("missing_mask must be bool or uint8")
        if missing_mask.shape != features.shape:
            raise ShardValidationError("missing_mask shape does not match features")
        if feature_types.dtype != np.dtype(np.uint8) or feature_types.shape != (features.shape[1],):
            raise ShardValidationError("feature_types must be uint8 with one value per feature")
        features_array = np.asarray(features)
        clean_array = np.asarray(clean_signal)
        if not np.isfinite(features_array).all() or not np.isfinite(clean_array).all():
            raise ShardValidationError("features and clean_signal must be finite")
        digest = stable_sha256(
            {
                "features": features_array,
                "clean_signal": clean_array,
                "missing_mask": np.asarray(missing_mask),
                "feature_types": np.asarray(feature_types),
            }
        )
        recorded_digest = _read_text_attr(handle.attrs, "feature_sha256")
        if digest != recorded_digest:
            raise ShardValidationError("feature_sha256 does not match feature datasets")
        return {
            "task_id": task_id,
            "task_count": task_count,
            "n_samples": int(features.shape[0]),
            "n_features": int(features.shape[1]),
            "clean_signal_dim": int(clean_signal.shape[1]),
            "feature_sha256": digest,
        }


def _validate_label_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ShardValidationError(f"label shard does not exist: {path}")
    with h5py.File(path, "r") as handle:
        if set(handle.keys()) != {"labels"}:
            raise ShardValidationError("label shard must contain only a /labels group")
        task_id = _read_text_attr(handle.attrs, "task_id")
        schema = _read_text_attr(handle.attrs, "schema_version")
        if schema != SCHEMA_VERSION:
            raise ShardValidationError(f"unsupported label shard schema: {schema}")
        task_count = int(handle.attrs.get("task_count", -1))
        if task_count != 1:
            raise ShardValidationError(f"label task_count must be 1, got {task_count}")
        group = handle["labels"]
        if set(group.keys()) != set(LABEL_DATASETS):
            raise ShardValidationError("label group must contain only a labels dataset")
        attrs = group.attrs
        if "K" not in attrs:
            raise ShardValidationError("label group attrs must include K")
        if "clm_cha_observed" not in attrs and "CLM" not in attrs:
            raise ShardValidationError("label group attrs must include clm_cha_observed")
        labels = group["labels"]
        if labels.dtype not in (np.dtype(np.int16), np.dtype(np.int32)) or labels.ndim != 1:
            raise ShardValidationError("labels must be a rank-1 int16 or int32 dataset")
        labels_array = np.asarray(labels)
        k_value = int(group.attrs["K"])
        observed_value = float(attrs.get("clm_cha_observed", attrs.get("CLM")))
        clean_control_value = (
            None if "clm_cha_clean_control" not in attrs else float(attrs["clm_cha_clean_control"])
        )
        status_value = str(attrs.get("clm_status", "ok"))
        error_value = None if "clm_error" not in attrs else str(attrs["clm_error"])
        if k_value <= 0 or not math.isfinite(observed_value):
            raise ShardValidationError("label K must be positive and clm_cha_observed finite")
        if clean_control_value is not None and not math.isfinite(clean_control_value):
            raise ShardValidationError("clm_cha_clean_control must be finite")
        digest = stable_sha256(
            {
                "labels": labels_array,
                "K": k_value,
                "clm_cha_observed": observed_value,
                "clm_cha_clean_control": clean_control_value,
                "clm_status": status_value,
                "clm_error": error_value,
            }
        )
        recorded_digest = _read_text_attr(handle.attrs, "labels_sha256")
        if digest != recorded_digest:
            raise ShardValidationError("labels_sha256 does not match label datasets")
        return {
            "task_id": task_id,
            "task_count": task_count,
            "n_samples": int(labels.shape[0]),
            "labels_sha256": digest,
            "K": k_value,
            "CLM": observed_value,
            "clm_cha_observed": observed_value,
            "clm_cha_clean_control": clean_control_value,
            "clm_status": status_value,
            "clm_error": error_value,
        }


def validate_paired_shards(
    features_path: str | os.PathLike[str],
    labels_path: str | os.PathLike[str],
    *,
    expected_task_id: str | None = None,
    expected_task_count: int | None = None,
) -> ShardValidation:
    """Validate both files, including count, alignment, finite and mask checks."""

    # Multi-task worker files have one safe task-id group per root child.  The
    # helper is defined below so the public pair validator can dispatch to it
    # without making callers care which shard granularity they received.
    with h5py.File(features_path, "r") as feature_handle:
        feature_root_keys = set(feature_handle.keys())
    if feature_root_keys != {"features"}:
        if expected_task_id is not None:
            raise ValueError("expected_task_id is only valid for a one-task shard")
        multi = validate_multi_shard_pair(
            features_path,
            labels_path,
            expected_task_count=expected_task_count,
        )
        # A multi result intentionally has a different return type.  Keeping
        # this annotation broad preserves the historical one-task API while
        # letting is_completed_pair and workers use the same validator.
        return multi  # type: ignore[return-value]
    if expected_task_count not in (None, 1):
        raise ValueError("one-task shard requires expected_task_count=1")
    feature_info = _validate_feature_file(Path(features_path))
    label_info = _validate_label_file(Path(labels_path))
    if feature_info["task_id"] != label_info["task_id"]:
        raise ShardValidationError(
            f"task_id mismatch: {feature_info['task_id']!r} vs {label_info['task_id']!r}"
        )
    if expected_task_id is not None and feature_info["task_id"] != expected_task_id:
        raise ShardValidationError(
            f"unexpected task_id {feature_info['task_id']!r}; expected {expected_task_id!r}"
        )
    if feature_info["task_count"] != label_info["task_count"]:
        raise ShardValidationError("feature and label task_count values differ")
    if feature_info["n_samples"] != label_info["n_samples"]:
        raise ShardValidationError("feature and label sample counts differ")
    return ShardValidation(
        task_id=feature_info["task_id"],
        task_count=feature_info["task_count"],
        n_samples=feature_info["n_samples"],
        n_features=feature_info["n_features"],
        clean_signal_dim=feature_info["clean_signal_dim"],
        feature_sha256=feature_info["feature_sha256"],
        labels_sha256=label_info["labels_sha256"],
        K=label_info["K"],
        CLM=label_info["CLM"],
        clm_cha_observed=label_info.get("clm_cha_observed"),
        clm_cha_clean_control=label_info.get("clm_cha_clean_control"),
        clm_status=label_info.get("clm_status"),
        clm_error=label_info.get("clm_error"),
    )


# Common alternate spelling used by callers.
validate_shard_pair = validate_paired_shards


def is_completed_pair(
    features_path: str | os.PathLike[str],
    labels_path: str | os.PathLike[str],
    *,
    expected_task_id: str | None = None,
) -> bool:
    """Return true only when both final files independently validate."""

    feature_file = Path(features_path)
    label_file = Path(labels_path)
    if not feature_file.is_file() or not label_file.is_file():
        return False
    try:
        validate_paired_shards(feature_file, label_file, expected_task_id=expected_task_id)
    except (OSError, ShardError, ValueError):
        return False
    return True


def _training_manifest(
    task: TaskPayload,
    paths: ShardPaths,
    worker_id: str,
    feature_digest: str,
) -> dict[str, Any]:
    record = {
        "schema_version": SCHEMA_VERSION,
        "task_id": task.task_id,
        "worker_id": worker_id,
        "features_path": str(paths.features),
        "n_samples": int(task.features.shape[0]),
        "n_features": int(task.features.shape[1]),
        "clean_signal_dim": int(task.clean_signal.shape[1]),
        "feature_sha256": feature_digest,
    }
    _assert_manifest_isolated(record)
    record["record_sha256"] = stable_sha256(record)
    return record


def _audit_manifest(
    task: TaskPayload,
    paths: ShardPaths,
    worker_id: str,
    feature_digest: str,
    labels_digest: str,
) -> dict[str, Any]:
    record = {
        "schema_version": SCHEMA_VERSION,
        "task_id": task.task_id,
        "worker_id": worker_id,
        "features_path": str(paths.features),
        "labels_path": str(paths.labels),
        "n_samples": int(task.features.shape[0]),
        "n_features": int(task.features.shape[1]),
        "feature_sha256": feature_digest,
        "labels_sha256": labels_digest,
        "K": int(task.K),
        "CLM": float(task.CLM),
        "clm_cha_observed": float(
            task.clm_cha_observed if task.clm_cha_observed is not None else task.CLM
        ),
        "clm_cha_clean_control": task.clm_cha_clean_control,
        "clm_status": task.clm_status,
        "clm_error": task.clm_error,
    }
    record["record_sha256"] = stable_sha256(record)
    return record


def _assert_manifest_isolated(record: Mapping[str, Any]) -> None:
    """Enforce the feature-only training manifest contract."""

    if _contains_forbidden_key(record):
        raise ShardValidationError("training manifest contains audit-only fields")
    # A label path must not sneak in as an arbitrary value under a harmless
    # key.  The feature path is the sole allowed path field.
    for key, value in record.items():
        if key != "features_path" and isinstance(value, (str, Path)):
            token = _name_token(value)
            if "label" in token or token in {"y", "k", "clm"}:
                raise ShardValidationError("training manifest contains a label/audit value")


def append_manifest_records(
    training_record: Mapping[str, Any],
    audit_record: Mapping[str, Any],
    manifest_dir: str | os.PathLike[str],
) -> tuple[Path, Path]:
    """Append one pair of JSONL records to training and audit manifests.

    This helper is opt-in so workers can keep manifests on their own shard
    path.  It never writes the audit path into the training record.
    """

    _assert_manifest_isolated(training_record)
    directory = Path(manifest_dir)
    directory.mkdir(parents=True, exist_ok=True)
    training_path = directory / "training_manifest.jsonl"
    audit_path = directory / "audit_manifest.jsonl"
    for path, record in ((training_path, training_record), (audit_path, audit_record)):
        line = _json_dumps(_canonical_value(record)) + "\n"
        with path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(line)
            stream.flush()
            try:
                os.fsync(stream.fileno())
            except OSError:
                pass
    return training_path, audit_path


def write_task_shards(
    task: Any = None,
    output_dir: str | os.PathLike[str] | None = None,
    *,
    task_id: str | None = None,
    features: Any = None,
    clean_signal: Any = None,
    missing_mask: Any = None,
    feature_types: Any = None,
    labels: Any = None,
    K: Any = None,
    CLM: Any = None,
    feature_attrs: Mapping[str, Any] | None = None,
    clm_cha_observed: Any = None,
    clm_cha_clean_control: Any = None,
    clm_status: Any = None,
    clm_error: Any = None,
    audit_attrs: Mapping[str, Any] | None = None,
    worker_id: str = "worker-0",
    compression: str | None = "lzf",
    resume: bool = True,
    overwrite: bool = False,
    feature_path: str | os.PathLike[str] | None = None,
    labels_path: str | os.PathLike[str] | None = None,
    manifest_dir: str | os.PathLike[str] | None = None,
) -> ShardWriteResult:
    """Write one heterogeneous task as an atomically promoted file pair.

    ``task`` can be a :class:`TaskPayload`, a mapping, or an object exposing
    the corresponding attributes.  Explicit keyword arrays override fields
    from ``task``.  ``feature_path`` and ``labels_path`` may be supplied when
    a scheduler has already allocated worker-specific names; they must be
    supplied together.
    """

    # A convenient ``write_task_shards(output_dir, task_id=..., ...)`` form is
    # supported without making the normal task-first form ambiguous.
    if output_dir is None and isinstance(task, (str, os.PathLike)):
        output_dir = task
        task = None
    if output_dir is None and (feature_path is None or labels_path is None):
        raise ValueError("output_dir is required unless both shard paths are supplied")
    if (feature_path is None) != (labels_path is None):
        raise ValueError("feature_path and labels_path must be supplied together")

    payload = _task_from_input(
        task,
        task_id=task_id,
        features=features,
        clean_signal=clean_signal,
        missing_mask=missing_mask,
        feature_types=feature_types,
        labels=labels,
        K=K,
        CLM=CLM,
        feature_attrs=feature_attrs,
        clm_cha_observed=clm_cha_observed,
        clm_cha_clean_control=clm_cha_clean_control,
        clm_status=clm_status,
        clm_error=clm_error,
        audit_attrs=audit_attrs,
    )
    payload = _normalise_task(payload)
    worker_id = _safe_component(worker_id, name="worker_id")
    if feature_path is None:
        assert output_dir is not None
        paths = shard_paths(output_dir, payload.task_id, worker_id=worker_id)
    else:
        paths = ShardPaths(Path(feature_path), Path(labels_path))
        if paths.features == paths.labels:
            raise ValueError("feature and label paths must differ")
    paths.features.parent.mkdir(parents=True, exist_ok=True)
    paths.labels.parent.mkdir(parents=True, exist_ok=True)
    feature_digest = _feature_digest(payload)
    labels_digest = _labels_digest(payload)

    final_feature_exists = paths.features.exists()
    final_labels_exists = paths.labels.exists()
    if final_feature_exists or final_labels_exists:
        if final_feature_exists and final_labels_exists:
            try:
                validation = validate_paired_shards(
                    paths.features, paths.labels, expected_task_id=payload.task_id
                )
            except (OSError, ShardError, ValueError) as exc:
                if not overwrite:
                    raise ShardConflictError(
                        f"existing shard pair is invalid; refusing implicit overwrite: {paths.features}, {paths.labels}"
                    ) from exc
            else:
                if validation.feature_sha256 != feature_digest or validation.labels_sha256 != labels_digest:
                    if not overwrite:
                        raise ShardConflictError(
                            "existing completed pair has different task content; refusing resume"
                        )
                elif resume:
                    training = _training_manifest(payload, paths, worker_id, feature_digest)
                    audit = _audit_manifest(payload, paths, worker_id, feature_digest, labels_digest)
                    if manifest_dir is not None:
                        append_manifest_records(training, audit, manifest_dir)
                    return ShardWriteResult(payload.task_id, worker_id, paths, training, audit, resumed=True)
                elif not overwrite:
                    raise ShardConflictError("completed shard pair exists and resume=False")
        elif not overwrite:
            raise ShardConflictError(
                "only one side of the shard pair exists; it is not a completed resumable pair"
            )

        if overwrite:
            # Explicit overwrite is the only path that removes an old final;
            # the default resume behavior never mistakes a one-sided pair for
            # completion and never silently destroys it.
            for old_path in (paths.features, paths.labels):
                if old_path.exists():
                    old_path.unlink()

    # A stale temporary pair is safe to replace.  If one final side remained,
    # the branch above raised before reaching here unless overwrite was set.
    for partial in (paths.features_partial, paths.labels_partial):
        if partial.exists():
            partial.unlink()
    try:
        _write_feature_file(paths.features_partial, payload, worker_id, compression)
        _write_label_file(paths.labels_partial, payload, worker_id, compression)
        _fsync_file(paths.features_partial)
        _fsync_file(paths.labels_partial)
        partial_validation = validate_paired_shards(
            paths.features_partial,
            paths.labels_partial,
            expected_task_id=payload.task_id,
        )
        if (
            partial_validation.feature_sha256 != feature_digest
            or partial_validation.labels_sha256 != labels_digest
        ):
            raise ShardValidationError("temporary shard hash does not match input task")
        # os.replace is atomic for each file.  Completion is defined as both
        # final paths validating; a crash between these replaces is therefore
        # detected as a one-sided incomplete pair on the next resume.
        os.replace(paths.features_partial, paths.features)
        os.replace(paths.labels_partial, paths.labels)
        validation = validate_paired_shards(
            paths.features,
            paths.labels,
            expected_task_id=payload.task_id,
        )
    except Exception:
        # Keep no misleading completed file.  Temporary files are deliberately
        # left in place when validation/write fails so a caller can inspect the
        # failure; a subsequent invocation replaces them safely.
        raise

    training = _training_manifest(payload, paths, worker_id, validation.feature_sha256)
    audit = _audit_manifest(payload, paths, worker_id, validation.feature_sha256, validation.labels_sha256)
    if manifest_dir is not None:
        append_manifest_records(training, audit, manifest_dir)
    return ShardWriteResult(payload.task_id, worker_id, paths, training, audit, resumed=False)


MULTI_SCHEMA_VERSION = "dfcluster.paired_multi_shard.v1"


def _multi_paths(
    feature_path: str | os.PathLike[str],
    labels_path: str | os.PathLike[str],
) -> ShardPaths:
    paths = ShardPaths(Path(feature_path), Path(labels_path))
    if paths.features == paths.labels:
        raise ValueError("feature and label paths must differ")
    return paths


def _write_multi_files(
    paths: ShardPaths,
    tasks: Sequence[TaskPayload],
    worker_id: str,
    compression: str | None,
) -> None:
    options = _compression_kwargs(compression)
    task_ids = [task.task_id for task in tasks]
    with h5py.File(paths.features_partial, "w") as feature_file, h5py.File(
        paths.labels_partial, "w"
    ) as label_file:
        for handle in (feature_file, label_file):
            handle.attrs["schema_version"] = MULTI_SCHEMA_VERSION
            handle.attrs["task_count"] = len(tasks)
            handle.attrs["worker_id"] = worker_id
            handle.attrs["task_ids_sha256"] = stable_sha256(task_ids)
            handle.create_group("tasks", track_order=True)

        feature_root = feature_file["tasks"]
        label_root = label_file["tasks"]
        for task in tasks:
            feature_digest = _feature_digest(task)
            labels_digest = _labels_digest(task)

            feature_group = feature_root.create_group(task.task_id)
            feature_group.attrs["feature_sha256"] = feature_digest
            feature_group.attrs["n_samples"] = task.features.shape[0]
            feature_group.attrs["n_features"] = task.features.shape[1]
            feature_group.attrs["clean_signal_dim"] = task.clean_signal.shape[1]
            _set_attrs(feature_group, task.feature_attrs or {})
            feature_group.create_dataset("features", data=task.features, **options)
            feature_group.create_dataset("clean_signal", data=task.clean_signal, **options)
            feature_group.create_dataset("missing_mask", data=task.missing_mask, **options)
            feature_group.create_dataset("feature_types", data=task.feature_types, **options)

            label_group = label_root.create_group(task.task_id)
            label_group.attrs["labels_sha256"] = labels_digest
            label_group.attrs["K"] = task.K
            label_group.attrs["clm_cha_observed"] = (
                task.clm_cha_observed
                if task.clm_cha_observed is not None
                else task.CLM
            )
            if task.clm_cha_clean_control is not None:
                label_group.attrs["clm_cha_clean_control"] = task.clm_cha_clean_control
            label_group.attrs["clm_status"] = task.clm_status
            if task.clm_error is not None:
                label_group.attrs["clm_error"] = task.clm_error
            _set_attrs(label_group, task.audit_attrs or {})
            label_group.create_dataset("labels", data=task.labels, **options)
        feature_file.attrs["shard_sha256"] = stable_sha256(
            [_feature_digest(task) for task in tasks]
        )
        label_file.attrs["shard_sha256"] = stable_sha256(
            [_labels_digest(task) for task in tasks]
        )
        feature_file.flush()
        label_file.flush()


def validate_multi_paired_shards(
    features_path: str | os.PathLike[str],
    labels_path: str | os.PathLike[str],
    *,
    expected_task_ids: Sequence[str] | None = None,
) -> MultiShardValidation:
    """Validate every group in a heterogeneous multi-task shard pair."""

    feature_path = Path(features_path)
    label_path = Path(labels_path)
    if not feature_path.is_file() or not label_path.is_file():
        raise ShardValidationError("both sides of the multi-task pair must exist")
    records: list[ShardValidation] = []
    with h5py.File(feature_path, "r") as feature_file, h5py.File(
        label_path, "r"
    ) as label_file:
        for handle, side in ((feature_file, "feature"), (label_file, "label")):
            if _read_text_attr(handle.attrs, "schema_version") != MULTI_SCHEMA_VERSION:
                raise ShardValidationError(f"unexpected {side} multi-shard schema")
            if set(handle.keys()) != {"tasks"}:
                raise ShardValidationError(f"{side} multi-shard must contain only /tasks")
        feature_ids = tuple(feature_file["tasks"].keys())
        label_ids = tuple(label_file["tasks"].keys())
        if feature_ids != label_ids:
            raise ShardValidationError("feature and label task_id order differs")
        if len(feature_ids) != int(feature_file.attrs.get("task_count", -1)):
            raise ShardValidationError("feature task_count does not match groups")
        if len(label_ids) != int(label_file.attrs.get("task_count", -1)):
            raise ShardValidationError("label task_count does not match groups")
        if expected_task_ids is not None and feature_ids != tuple(expected_task_ids):
            raise ShardValidationError("multi-shard task_ids differ from expected order")
        if _read_text_attr(feature_file.attrs, "task_ids_sha256") != stable_sha256(
            list(feature_ids)
        ):
            raise ShardValidationError("feature task_ids hash mismatch")
        if _read_text_attr(label_file.attrs, "task_ids_sha256") != stable_sha256(
            list(label_ids)
        ):
            raise ShardValidationError("label task_ids hash mismatch")

        feature_digests: list[str] = []
        label_digests: list[str] = []
        for task_id in feature_ids:
            feature_group = feature_file[f"tasks/{task_id}"]
            label_group = label_file[f"tasks/{task_id}"]
            if set(feature_group.keys()) != set(FEATURE_DATASETS):
                raise ShardValidationError(f"invalid feature datasets for {task_id}")
            if set(label_group.keys()) != {"labels"}:
                raise ShardValidationError(f"invalid label datasets for {task_id}")
            _assert_no_forbidden_names(
                f"feature attrs for {task_id}", dict(feature_group.attrs)
            )
            features = np.asarray(feature_group["features"])
            clean = np.asarray(feature_group["clean_signal"])
            mask = np.asarray(feature_group["missing_mask"])
            types = np.asarray(feature_group["feature_types"])
            labels = np.asarray(label_group["labels"])
            if features.dtype != np.float32 or clean.dtype != np.float32:
                raise ShardValidationError("multi-shard numeric targets must be float32")
            if features.ndim != 2 or clean.ndim != 2 or clean.shape[0] != features.shape[0]:
                raise ShardValidationError(f"invalid feature/clean shape for {task_id}")
            if mask.shape != features.shape or mask.dtype not in (np.bool_, np.uint8):
                raise ShardValidationError(f"invalid mask for {task_id}")
            if types.shape != (features.shape[1],) or types.dtype != np.uint8:
                raise ShardValidationError(f"invalid feature_types for {task_id}")
            if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
                raise ShardValidationError(f"invalid labels for {task_id}")
            if not np.isfinite(features).all() or not np.isfinite(clean).all():
                raise ShardValidationError(f"non-finite arrays for {task_id}")

            feature_digest = stable_sha256(
                {
                    "features": features,
                    "clean_signal": clean,
                    "missing_mask": mask,
                    "feature_types": types,
                }
            )
            clm_observed = float(label_group.attrs["clm_cha_observed"])
            clm_clean = (
                float(label_group.attrs["clm_cha_clean_control"])
                if "clm_cha_clean_control" in label_group.attrs
                else None
            )
            k_value = int(label_group.attrs["K"])
            clm_status = _read_text_attr(label_group.attrs, "clm_status")
            clm_error = (
                _read_text_attr(label_group.attrs, "clm_error")
                if "clm_error" in label_group.attrs
                else None
            )
            labels_digest = stable_sha256(
                {
                    "labels": labels,
                    "K": k_value,
                    "clm_cha_observed": clm_observed,
                    "clm_cha_clean_control": clm_clean,
                    "clm_status": clm_status,
                    "clm_error": clm_error,
                }
            )
            if feature_digest != _read_text_attr(feature_group.attrs, "feature_sha256"):
                raise ShardValidationError(f"feature hash mismatch for {task_id}")
            if labels_digest != _read_text_attr(label_group.attrs, "labels_sha256"):
                raise ShardValidationError(f"label hash mismatch for {task_id}")
            feature_digests.append(feature_digest)
            label_digests.append(labels_digest)
            records.append(
                ShardValidation(
                    task_id=task_id,
                    task_count=1,
                    n_samples=int(features.shape[0]),
                    n_features=int(features.shape[1]),
                    clean_signal_dim=int(clean.shape[1]),
                    feature_sha256=feature_digest,
                    labels_sha256=labels_digest,
                    K=k_value,
                    CLM=clm_observed,
                    clm_cha_observed=clm_observed,
                    clm_cha_clean_control=clm_clean,
                    clm_status=clm_status,
                    clm_error=clm_error,
                )
            )
        if _read_text_attr(feature_file.attrs, "shard_sha256") != stable_sha256(
            feature_digests
        ):
            raise ShardValidationError("feature shard hash mismatch")
        if _read_text_attr(label_file.attrs, "shard_sha256") != stable_sha256(
            label_digests
        ):
            raise ShardValidationError("label shard hash mismatch")
    return MultiShardValidation(tuple(feature_ids), len(feature_ids), tuple(records))


def _multi_manifest_records(
    tasks: Sequence[TaskPayload],
    paths: ShardPaths,
    worker_id: str,
) -> tuple[tuple[Mapping[str, Any], ...], tuple[Mapping[str, Any], ...]]:
    training: list[Mapping[str, Any]] = []
    audit: list[Mapping[str, Any]] = []
    for task in tasks:
        feature_digest = _feature_digest(task)
        label_digest = _labels_digest(task)
        train_record: dict[str, Any] = {
            "schema_version": MULTI_SCHEMA_VERSION,
            "task_id": task.task_id,
            "worker_id": worker_id,
            "features_path": str(paths.features),
            "group_path": f"/tasks/{task.task_id}",
            "n_samples": int(task.features.shape[0]),
            "n_features": int(task.features.shape[1]),
            "clean_signal_dim": int(task.clean_signal.shape[1]),
            "feature_sha256": feature_digest,
        }
        _assert_manifest_isolated(train_record)
        train_record["record_sha256"] = stable_sha256(train_record)
        audit_record: dict[str, Any] = {
            **train_record,
            "labels_path": str(paths.labels),
            "labels_group_path": f"/tasks/{task.task_id}",
            "labels_sha256": label_digest,
            "K": int(task.K),
            "clm_cha_observed": float(
                task.clm_cha_observed
                if task.clm_cha_observed is not None
                else task.CLM
            ),
            "clm_cha_clean_control": task.clm_cha_clean_control,
            "clm_status": task.clm_status,
            "clm_error": task.clm_error,
            "audit_metadata": dict(task.audit_attrs or {}),
        }
        audit_record.pop("record_sha256", None)
        audit_record["record_sha256"] = stable_sha256(audit_record)
        training.append(train_record)
        audit.append(audit_record)
    return tuple(training), tuple(audit)


def write_multi_task_shards(
    tasks: Sequence[TaskPayload],
    *,
    feature_path: str | os.PathLike[str],
    labels_path: str | os.PathLike[str],
    worker_id: str,
    compression: str | None = "lzf",
    resume: bool = True,
) -> MultiShardWriteResult:
    """Write one worker-owned pair containing many heterogeneous tasks."""

    if not tasks:
        raise ValueError("a multi-task shard cannot be empty")
    worker_id = _safe_component(worker_id, name="worker_id")
    normalized = tuple(_normalise_task(task) for task in tasks)
    task_ids = tuple(task.task_id for task in normalized)
    if len(task_ids) != len(set(task_ids)):
        raise ShardValidationError("task_ids within a shard must be unique")
    paths = _multi_paths(feature_path, labels_path)
    paths.features.parent.mkdir(parents=True, exist_ok=True)
    paths.labels.parent.mkdir(parents=True, exist_ok=True)

    final_exists = (paths.features.exists(), paths.labels.exists())
    if any(final_exists):
        if not all(final_exists):
            raise ShardConflictError("only one side of the multi-task pair exists")
        validation = validate_multi_paired_shards(
            paths.features, paths.labels, expected_task_ids=task_ids
        )
        expected_feature = [_feature_digest(task) for task in normalized]
        expected_labels = [_labels_digest(task) for task in normalized]
        if [item.feature_sha256 for item in validation.records] != expected_feature or [
            item.labels_sha256 for item in validation.records
        ] != expected_labels:
            raise ShardConflictError("existing multi-task pair has different content")
        if not resume:
            raise ShardConflictError("completed multi-task pair exists and resume=False")
        training, audit = _multi_manifest_records(normalized, paths, worker_id)
        return MultiShardWriteResult(paths, task_ids, training, audit, worker_id, True)

    for partial in (paths.features_partial, paths.labels_partial):
        if partial.exists():
            partial.unlink()
    _write_multi_files(paths, normalized, worker_id, compression)
    _fsync_file(paths.features_partial)
    _fsync_file(paths.labels_partial)
    validate_multi_paired_shards(
        paths.features_partial,
        paths.labels_partial,
        expected_task_ids=task_ids,
    )
    os.replace(paths.features_partial, paths.features)
    os.replace(paths.labels_partial, paths.labels)
    validate_multi_paired_shards(
        paths.features, paths.labels, expected_task_ids=task_ids
    )
    training, audit = _multi_manifest_records(normalized, paths, worker_id)
    return MultiShardWriteResult(paths, task_ids, training, audit, worker_id, False)


# Public aliases make the single-task API discoverable without shadowing the
# explicit multi-task shard writer used by the corpus generator.
write_paired_shard = write_task_shards
write_shard_pair = write_multi_task_shards


def resume_task_shards(
    task: Any,
    output_dir: str | os.PathLike[str],
    **kwargs: Any,
) -> ShardWriteResult:
    """Resume a task pair, requiring both final sides to validate."""

    kwargs["resume"] = True
    return write_task_shards(task, output_dir, **kwargs)


__all__ = [
    "SCHEMA_VERSION",
    "FEATURE_DATASETS",
    "LABEL_DATASETS",
    "ShardError",
    "ShardValidationError",
    "ShardConflictError",
    "TaskPayload",
    "ShardPaths",
    "ShardValidation",
    "ShardWriteResult",
    "MultiShardValidation",
    "MultiShardWriteResult",
    "stable_sha256",
    "stable_hash",
    "shard_paths",
    "paired_shard_paths",
    "validate_paired_shards",
    "validate_shard_pair",
    "is_completed_pair",
    "append_manifest_records",
    "write_task_shards",
    "write_paired_shard",
    "write_shard_pair",
    "write_multi_task_shards",
    "validate_multi_paired_shards",
    "resume_task_shards",
]
