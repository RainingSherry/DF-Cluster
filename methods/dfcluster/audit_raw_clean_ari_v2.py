# ⚠️  DEPRECATED: Part of terminated V1/V2/V3 research line. See DEPRECATION_NOTICE.md
"""Three-process raw-X versus clean-oracle ARI audit for dfhybrid-v2.

This audit is an outer corpus-necessity check on a frozen TRAIN development
slice.  It is intentionally split
into four invocations (``freeze-k``, ``export``, ``cluster`` and ``score``):

* ``freeze-k`` reads the audit manifest once and writes a hash-frozen file
  containing only ``task_id`` and ``K``;
* ``export`` reads only the feature-only training manifest and feature HDF5
  files, and never receives the audit or label manifest;
* ``cluster`` receives only exported embeddings and the frozen K file;
* ``score`` is the first stage allowed to open labels, and can open them only
  below the explicitly supplied label root.

The clean-ARI gate and the capacity-probe decision are pre-registered here.
Stratified summaries are descriptive and cannot change the decision.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import socket
import sys
import time
import traceback
from typing import Any, Iterable

import h5py
import numpy as np

from .pilot_data import FeatureOnlyTaskPool, standardize_task


DEFAULT_MANIFEST = Path(
    "/data/luolie/DF-Cluster/data/synthetic/dfhybrid_v2_recoverable/manifests/"
    "training_manifest.jsonl"
)
DEFAULT_AUDIT_MANIFEST = Path(
    "/data/luolie/DF-Cluster/data/synthetic/dfhybrid_v2_recoverable/manifests/"
    "audit_manifest.jsonl"
)
DEFAULT_FEATURE_ROOT = Path(
    "/data/luolie/DF-Cluster/data/synthetic/dfhybrid_v2_recoverable/features"
)
DEFAULT_LABEL_ROOT = Path(
    "/data/luolie/DF-Cluster/data/synthetic/dfhybrid_v2_recoverable/labels"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/data/luolie/DF-Cluster/outputs/data_audit/raw_clean_ari_v2"
)
DEFAULT_TASK_COUNT = 4096
TRAIN_START_INDEX = 512
SELECTION_SALT = "dfcluster-raw-clean-ari-train-development-v2"
KMEANS_PROTOCOL = {
    "algorithm": "lloyd",
    "init": "k-means++",
    "n_init": 20,
    "max_iter": 300,
    "tol": 1.0e-4,
    "random_state": 20260824,
    "metric": "euclidean",
    "K_source": "frozen_train_development_k_manifest",
}
K_MANIFEST_FIELDS = {"task_id", "K"}
FORBIDDEN_K_TOKENS = ("label", "clm", "ari", "nmi")


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    digest = sha256()
    digest.update(str(contiguous.dtype).encode())
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(f"{path}.partial")
    partial.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(f"{path}.partial")
    with partial.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path} line {line_number} is not an object")
            rows.append(value)
    return rows


def _status(path: Path, status: str, **extra: Any) -> None:
    _atomic_json(path, {"status": status, **extra})


def _environment() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
    }


def _selected_task_ids(rows: list[dict[str, Any]], task_count: int) -> list[str]:
    """Select fixed TRAIN indices 512..4607 without inspecting arrays or labels."""

    if task_count <= 0:
        raise ValueError("task_count must be positive")
    task_rows: dict[str, dict[str, Any]] = {}
    train_rows: list[tuple[int, int, str]] = []
    for row_order, row in enumerate(rows):
        task_id = str(row.get("task_id", ""))
        if not task_id:
            raise ValueError("manifest row is missing task_id")
        if task_id in task_rows:
            raise ValueError(f"duplicate task_id in manifest: {task_id}")
        metadata = row.get("audit_metadata", {})
        split = metadata.get("split") if isinstance(metadata, dict) else None
        if split == "train" or "-tr-" in task_id:
            task_rows[task_id] = row
            metadata_index = metadata.get("task_index") if isinstance(metadata, dict) else None
            try:
                order_key = int(metadata_index)
            except (TypeError, ValueError):
                order_key = row_order
            train_rows.append((order_key, row_order, task_id))
    train_rows.sort(key=lambda value: (value[0], value[1]))
    if TRAIN_START_INDEX + task_count > len(train_rows):
        raise ValueError(
            f"requested TRAIN development slice [{TRAIN_START_INDEX},"
            f" {TRAIN_START_INDEX + task_count}) but manifest has {len(train_rows)} train tasks"
        )
    return [
        task_id
        for _, _, task_id in train_rows[TRAIN_START_INDEX : TRAIN_START_INDEX + task_count]
    ]


def _selected_train_indices(pool: FeatureOnlyTaskPool, task_count: int) -> list[int]:
    rows = list(pool.records)
    selected = set(_selected_task_ids(rows, task_count))
    indices = [
        index
        for index, row in enumerate(rows)
        if str(row["task_id"]) in selected
    ]
    if len(indices) != task_count:
        raise ValueError("training manifest selection did not preserve task count")
    return indices


def _validate_k_rows(rows: list[dict[str, Any]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for index, row in enumerate(rows, start=1):
        if set(row) != K_MANIFEST_FIELDS:
            raise ValueError(
                f"frozen K manifest line {index} must contain only task_id and K"
            )
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError(f"frozen K manifest line {index} has invalid task_id")
        lowered = json.dumps(row, ensure_ascii=False).lower()
        if any(token in lowered for token in FORBIDDEN_K_TOKENS):
            raise ValueError("frozen K manifest contains forbidden audit tokens")
        try:
            k_value = int(row["K"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"frozen K manifest line {index} has invalid K") from exc
        if k_value < 2:
            raise ValueError(f"frozen K manifest line {index} has K < 2")
        if task_id in result:
            raise ValueError(f"duplicate task_id in frozen K manifest: {task_id}")
        result[task_id] = k_value
    if not result:
        raise ValueError("frozen K manifest is empty")
    return result


def freeze_train_k(
    audit_manifest: Path,
    output_root: Path,
    task_count: int = DEFAULT_TASK_COUNT,
) -> dict[str, Any]:
    """Freeze only ``task_id`` and ``K`` from the audit manifest."""

    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "freeze_k_status.json"
    _status(status_path, "running", error=None)
    started = time.perf_counter()
    target = output_root / "train_development_k_manifest.jsonl"
    partial = Path(f"{target}.partial")
    try:
        audit_rows = _read_jsonl(audit_manifest)
        selected = _selected_task_ids(audit_rows, task_count)
        by_id = {str(row["task_id"]): row for row in audit_rows}
        frozen_rows = [{"task_id": task_id, "K": int(by_id[task_id]["K"])} for task_id in selected]
        _validate_k_rows(frozen_rows)
        _atomic_jsonl(target, frozen_rows)
        protocol = {
            "schema_version": "dfcluster.raw_clean_k_freeze.v2",
            "purpose": "predeclared TRAIN development K protocol; no labels or CLM are exposed",
            "source_audit_manifest": str(audit_manifest.resolve()),
            "source_audit_manifest_sha256": _sha256(audit_manifest),
            "selection_salt": SELECTION_SALT,
            "train_start_index": TRAIN_START_INDEX,
            "train_end_index_exclusive": TRAIN_START_INDEX + len(frozen_rows),
            "task_count": len(frozen_rows),
            "source_split": "train",
            "fields": ["task_id", "K"],
            "labels_paths_written": False,
            "clm_values_written": False,
            "train_development_k_manifest": str(target),
            "train_development_k_manifest_sha256": _sha256(target),
            "code_sha256": _sha256(Path(__file__)),
            "environment": _environment(),
        }
        _atomic_json(output_root / "freeze_k_protocol.json", protocol)
        elapsed = time.perf_counter() - started
        _status(
            status_path,
            "completed",
            error=None,
            elapsed_seconds=elapsed,
            task_count=len(frozen_rows),
            train_development_k_manifest_sha256=protocol[
                "train_development_k_manifest_sha256"
            ],
        )
        return protocol
    except Exception as exc:
        if partial.exists():
            partial.unlink()
        _status(
            status_path,
            "incomplete_compute",
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
            elapsed_seconds=time.perf_counter() - started,
        )
        raise


def export_embeddings(
    manifest: Path,
    feature_root: Path,
    output_root: Path,
    task_count: int = DEFAULT_TASK_COUNT,
) -> dict[str, Any]:
    """TRAIN development feature-only stage; no K/Y/CLM or audit manifest is accepted."""

    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "export_status.json"
    _status(status_path, "running", error=None)
    started = time.perf_counter()
    target = output_root / "embeddings.h5"
    partial = Path(f"{target}.partial")
    try:
        pool = FeatureOnlyTaskPool(manifest, allowed_feature_root=feature_root)
        indices = _selected_train_indices(pool, task_count)
        records: list[dict[str, Any]] = []
        with h5py.File(partial, "w") as handle:
            handle.attrs["schema_version"] = "dfcluster.raw_clean_embeddings.v2"
            handle.attrs["audit_fields_opened"] = False
            for index in indices:
                task = pool.load_numpy(index)
                prepared = standardize_task(task, __import__("torch").device("cpu"))
                raw = prepared["values"].cpu().numpy().astype(np.float32, copy=False)
                clean = prepared["clean_signal"].cpu().numpy().astype(np.float32, copy=False)
                group = handle.create_group(f"tasks/{task['task_id']}")
                group.create_dataset("raw_x", data=raw, compression="gzip", compression_opts=1)
                group.create_dataset("clean_s", data=clean, compression="gzip", compression_opts=1)
                group.attrs["task_id"] = task["task_id"]
                records.append(
                    {
                        "task_id": task["task_id"],
                        "group_path": f"/tasks/{task['task_id']}",
                        "n_samples": int(raw.shape[0]),
                        "raw_dim": int(raw.shape[1]),
                        "clean_dim": int(clean.shape[1]),
                        "raw_sha256": _array_sha256(raw),
                        "clean_sha256": _array_sha256(clean),
                    }
                )
        os.replace(partial, target)
        manifest_record = {
            "schema_version": "dfcluster.raw_clean_embedding_manifest.v2",
            "purpose": "frozen TRAIN development corpus-necessity audit; not a method representation",
            "selection_salt": SELECTION_SALT,
            "task_count": len(records),
            "source_split": "train",
            "train_start_index": TRAIN_START_INDEX,
            "train_end_index_exclusive": TRAIN_START_INDEX + len(records),
            "training_manifest": str(manifest.resolve()),
            "training_manifest_sha256": _sha256(manifest),
            "feature_root": str(feature_root.resolve()),
            "embeddings_path": str(target),
            "embeddings_sha256": _sha256(target),
            "audit_fields_opened": False,
            "label_arrays_opened": False,
            "K_source": "not_used_in_export",
            "code_sha256": _sha256(Path(__file__)),
            "environment": _environment(),
            "records": records,
        }
        _atomic_json(output_root / "embedding_manifest.json", manifest_record)
        elapsed = time.perf_counter() - started
        _status(status_path, "completed", error=None, elapsed_seconds=elapsed)
        return manifest_record
    except Exception as exc:
        if partial.exists():
            partial.unlink()
        _status(
            status_path,
            "incomplete_compute",
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
            elapsed_seconds=time.perf_counter() - started,
        )
        raise


def _read_frozen_k_manifest(path: Path) -> dict[str, int]:
    rows = _read_jsonl(path)
    return _validate_k_rows(rows)


def _verify_frozen_k_artifact(k_manifest_path: Path) -> dict[str, Any]:
    """Require the dedicated freeze-stage certificate before clustering."""

    protocol_path = k_manifest_path.with_name("freeze_k_protocol.json")
    if not protocol_path.is_file():
        raise FileNotFoundError(f"freeze-stage protocol is missing: {protocol_path}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    expected = protocol.get("train_development_k_manifest_sha256")
    if expected != _sha256(k_manifest_path):
        raise ValueError("frozen TRAIN K manifest hash does not match freeze protocol")
    rows = _read_frozen_k_manifest(k_manifest_path)
    if len(rows) != int(protocol.get("task_count", -1)):
        raise ValueError("frozen TRAIN K manifest task count does not match freeze protocol")
    return protocol


def cluster_known_k(
    embedding_manifest_path: Path,
    k_manifest_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Known-K stage; only embeddings and the frozen TRAIN K manifest are inputs."""

    from sklearn.cluster import KMeans
    import sklearn

    status_path = output_root / "cluster_status.json"
    _status(status_path, "running", error=None)
    started = time.perf_counter()
    target = output_root / "predictions.h5"
    partial = Path(f"{target}.partial")
    try:
        embedding_manifest = json.loads(embedding_manifest_path.read_text(encoding="utf-8"))
        embeddings_path = Path(embedding_manifest["embeddings_path"]).resolve()
        if _sha256(embeddings_path) != embedding_manifest["embeddings_sha256"]:
            raise ValueError("frozen embedding artifact hash mismatch")
        freeze_protocol = _verify_frozen_k_artifact(k_manifest_path)
        k_by_task = _read_frozen_k_manifest(k_manifest_path)
        records_in = embedding_manifest["records"]
        selected = {str(row["task_id"]) for row in records_in}
        if selected != set(k_by_task):
            raise ValueError("frozen K manifest and embedding manifest task sets differ")
        records: list[dict[str, Any]] = []
        with h5py.File(embeddings_path, "r") as source, h5py.File(partial, "w") as target_file:
            target_file.attrs["schema_version"] = "dfcluster.raw_clean_predictions.v2"
            target_file.attrs["Y_arrays_opened"] = False
            for row in records_in:
                task_id = str(row["task_id"])
                group = source[row["group_path"]]
                k = k_by_task[task_id]
                predictions: dict[str, np.ndarray] = {}
                diagnostics: dict[str, Any] = {}
                for name, dataset in (("raw_x", group["raw_x"]), ("clean_s", group["clean_s"])):
                    embedding = np.asarray(dataset, dtype=np.float32)
                    expected_sha = row["raw_sha256" if name == "raw_x" else "clean_sha256"]
                    if _array_sha256(embedding) != expected_sha:
                        raise ValueError(f"embedding hash mismatch for {task_id}/{name}")
                    if embedding.ndim != 2 or embedding.shape[0] != int(row["n_samples"]):
                        raise ValueError(f"embedding shape mismatch for {task_id}/{name}")
                    model = KMeans(
                        n_clusters=k,
                        init=KMEANS_PROTOCOL["init"],
                        n_init=KMEANS_PROTOCOL["n_init"],
                        max_iter=KMEANS_PROTOCOL["max_iter"],
                        tol=KMEANS_PROTOCOL["tol"],
                        algorithm=KMEANS_PROTOCOL["algorithm"],
                        random_state=KMEANS_PROTOCOL["random_state"],
                    )
                    predictions[name] = model.fit_predict(embedding).astype(np.int32)
                    diagnostics[name] = {
                        "inertia": float(model.inertia_),
                        "n_iter": int(model.n_iter_),
                    }
                output_group = target_file.create_group(f"tasks/{task_id}")
                output_group.create_dataset("raw_x", data=predictions["raw_x"], compression="gzip", compression_opts=1)
                output_group.create_dataset("clean_s", data=predictions["clean_s"], compression="gzip", compression_opts=1)
                output_group.attrs["task_id"] = task_id
                output_group.attrs["K"] = k
                records.append(
                    {
                        "task_id": task_id,
                        "group_path": f"/tasks/{task_id}",
                        "n_samples": int(row["n_samples"]),
                        "K": k,
                        "raw_prediction_sha256": _array_sha256(predictions["raw_x"]),
                        "clean_prediction_sha256": _array_sha256(predictions["clean_s"]),
                        "diagnostics": diagnostics,
                    }
                )
        os.replace(partial, target)
        protocol = {
            "schema_version": "dfcluster.raw_clean_cluster_protocol.v2",
            "embedding_manifest": str(embedding_manifest_path.resolve()),
            "embedding_manifest_sha256": _sha256(embedding_manifest_path),
            "train_development_k_manifest": str(k_manifest_path.resolve()),
            "train_development_k_manifest_sha256": _sha256(k_manifest_path),
            "freeze_k_protocol": str(
                k_manifest_path.with_name("freeze_k_protocol.json").resolve()
            ),
            "freeze_k_protocol_sha256": _sha256(
                k_manifest_path.with_name("freeze_k_protocol.json")
            ),
            "freeze_k_protocol_task_count": int(freeze_protocol["task_count"]),
            "inputs": ["embedding_manifest", "train_development_k_manifest"],
            "Y_arrays_opened": False,
            "CLM_used": False,
            "predictions_path": str(target),
            "predictions_sha256": _sha256(target),
            "kmeans": KMEANS_PROTOCOL,
            "sklearn": sklearn.__version__,
            "python": sys.version,
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
            "code_sha256": _sha256(Path(__file__)),
            "records": records,
        }
        _atomic_json(output_root / "cluster_protocol.json", protocol)
        elapsed = time.perf_counter() - started
        _status(status_path, "completed", error=None, elapsed_seconds=elapsed)
        return protocol
    except Exception as exc:
        if partial.exists():
            partial.unlink()
        _status(
            status_path,
            "incomplete_compute",
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
            elapsed_seconds=time.perf_counter() - started,
        )
        raise


def _audit_label_map(path: Path, selected: set[str]) -> dict[str, dict[str, Any]]:
    """Read label locators for score only; callers still enforce label-root."""

    result: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        task_id = str(row.get("task_id", ""))
        if task_id not in selected:
            continue
        if task_id in result:
            raise ValueError(f"duplicate task_id in audit manifest: {task_id}")
        metadata = row.get("audit_metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        # The fallback keeps the scorer usable with the v1-style nested
        # metadata while never reading CLM or any label-derived statistic.
        spec = metadata.get("spec", {})
        if not isinstance(spec, dict):
            spec = {}
        stratum = metadata.get("observation_family") or metadata.get("observation_mode")
        if not stratum:
            stratum = spec.get("observation_mode", "unspecified")
        result[task_id] = {
            "task_id": task_id,
            "labels_path": str(row["labels_path"]),
            "labels_group_path": str(row["labels_group_path"]),
            "labels_sha256": str(row["labels_sha256"]),
            "stratum": str(stratum),
        }
    if set(result) != selected:
        raise ValueError("audit manifest does not cover every selected task")
    return result


def _resolve_label_path(label_root: Path, labels_path: str | Path) -> Path:
    """Resolve one label shard, rejecting symlink/path escapes."""

    root = label_root.resolve()
    candidate = Path(labels_path)
    candidate = (root / candidate if not candidate.is_absolute() else candidate).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"label path escapes restricted label root: {candidate}")
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def _centered_gram(values: np.ndarray) -> np.ndarray:
    centered = values.astype(np.float64) - values.mean(axis=0, keepdims=True)
    gram = centered @ centered.T
    norm = np.linalg.norm(gram)
    return gram / max(norm, 1.0e-12)


def _geometry_metrics(raw: np.ndarray, clean: np.ndarray, task_id: str) -> tuple[float, float]:
    count = min(256, raw.shape[0])
    seed = int.from_bytes(sha256(task_id.encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    rows = rng.choice(raw.shape[0], size=count, replace=False)
    raw_selected = raw[rows].astype(np.float64)
    clean_selected = clean[rows].astype(np.float64)
    raw_gram = _centered_gram(raw_selected)
    clean_gram = _centered_gram(clean_selected)
    cka = float(np.sum(raw_gram * clean_gram))
    pair_count = min(4096, count * (count - 1) // 2)
    first = rng.integers(0, count, size=pair_count)
    second = rng.integers(0, count, size=pair_count)
    keep = first != second
    raw_distance = np.linalg.norm(raw_selected[first[keep]] - raw_selected[second[keep]], axis=1)
    clean_distance = np.linalg.norm(clean_selected[first[keep]] - clean_selected[second[keep]], axis=1)
    raw_distance /= max(float(np.median(raw_distance)), 1.0e-8)
    clean_distance /= max(float(np.median(clean_distance)), 1.0e-8)
    stress = float(np.mean(np.abs(raw_distance - clean_distance)))
    return cka, stress


def _summary(values: list[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return {"count": 0, "median": None, "q10": None, "q90": None, "mean": None}
    return {
        "count": int(array.size),
        "median": float(np.median(array)),
        "q10": float(np.quantile(array, 0.10)),
        "q90": float(np.quantile(array, 0.90)),
        "mean": float(array.mean()),
    }


def score_predictions(
    embedding_manifest_path: Path,
    cluster_protocol_path: Path,
    audit_manifest: Path,
    label_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Score stage; the first stage that opens Y, only below ``label_root``."""

    from scipy.stats import spearmanr
    from sklearn.metrics import adjusted_rand_score

    status_path = output_root / "score_status.json"
    _status(status_path, "running", error=None)
    started = time.perf_counter()
    try:
        embedding_manifest = json.loads(embedding_manifest_path.read_text(encoding="utf-8"))
        protocol = json.loads(cluster_protocol_path.read_text(encoding="utf-8"))
        embeddings_path = Path(embedding_manifest["embeddings_path"]).resolve()
        predictions_path = Path(protocol["predictions_path"]).resolve()
        if _sha256(embedding_manifest_path) != protocol["embedding_manifest_sha256"]:
            raise ValueError("embedding manifest hash mismatch before scoring")
        if _sha256(embeddings_path) != embedding_manifest["embeddings_sha256"]:
            raise ValueError("embedding hash mismatch before scoring")
        if _sha256(predictions_path) != protocol["predictions_sha256"]:
            raise ValueError("prediction hash mismatch before scoring")
        records_in = embedding_manifest["records"]
        selected = {str(row["task_id"]) for row in records_in}
        audit = _audit_label_map(audit_manifest, selected)
        cluster_rows = {str(row["task_id"]): row for row in protocol["records"]}
        if set(cluster_rows) != selected:
            raise ValueError("cluster protocol and embedding task sets differ")
        records: list[dict[str, Any]] = []
        with h5py.File(embeddings_path, "r") as embeddings, h5py.File(predictions_path, "r") as predictions:
            for row in records_in:
                task_id = str(row["task_id"])
                metadata = audit[task_id]
                prediction_row = cluster_rows[task_id]
                embedding_group = embeddings[row["group_path"]]
                if str(embedding_group.attrs.get("task_id", "")) != task_id:
                    raise ValueError(f"embedding task_id mismatch for {task_id}")
                raw = np.asarray(embedding_group["raw_x"], dtype=np.float32)
                clean = np.asarray(embedding_group["clean_s"], dtype=np.float32)
                if _array_sha256(raw) != row["raw_sha256"] or _array_sha256(clean) != row["clean_sha256"]:
                    raise ValueError(f"embedding hash mismatch during scoring for {task_id}")
                prediction_group = predictions[prediction_row["group_path"]]
                raw_prediction = np.asarray(prediction_group["raw_x"], dtype=np.int32)
                clean_prediction = np.asarray(prediction_group["clean_s"], dtype=np.int32)
                if _array_sha256(raw_prediction) != prediction_row["raw_prediction_sha256"]:
                    raise ValueError(f"raw prediction hash mismatch for {task_id}")
                if _array_sha256(clean_prediction) != prediction_row["clean_prediction_sha256"]:
                    raise ValueError(f"clean prediction hash mismatch for {task_id}")
                if str(prediction_group.attrs.get("task_id", "")) != task_id:
                    raise ValueError(f"prediction task_id mismatch for {task_id}")
                if raw_prediction.ndim != 1 or clean_prediction.ndim != 1:
                    raise ValueError(f"prediction shape mismatch for {task_id}")
                if raw_prediction.shape[0] != int(row["n_samples"]) or clean_prediction.shape != raw_prediction.shape:
                    raise ValueError(f"prediction row count mismatch for {task_id}")
                label_path = _resolve_label_path(label_root, metadata["labels_path"])
                group_path = metadata["labels_group_path"]
                if group_path != f"/tasks/{task_id}":
                    raise ValueError(f"label group path/task mismatch for {task_id}")
                # This is the first Y open in the whole program.  Only the
                # labels dataset, its task id, stored digest and shape are
                # consumed; CLM attrs are never read.
                with h5py.File(label_path, "r") as labels_file:
                    if group_path not in labels_file:
                        raise ValueError(f"missing label group for {task_id}")
                    label_group = labels_file[group_path]
                    if str(label_group.attrs.get("task_id", "")) != task_id:
                        raise ValueError(f"label task_id mismatch for {task_id}")
                    if set(label_group.keys()) != {"labels"}:
                        raise ValueError(f"label schema mismatch for {task_id}")
                    labels_dataset = label_group["labels"]
                    if labels_dataset.ndim != 1 or labels_dataset.shape[0] != raw_prediction.shape[0]:
                        raise ValueError(f"label shape mismatch for {task_id}")
                    stored_labels_hash = str(label_group.attrs.get("labels_sha256", ""))
                    if stored_labels_hash != metadata["labels_sha256"]:
                        raise ValueError(f"label hash mismatch for {task_id}")
                    labels = np.asarray(labels_dataset, dtype=np.int32)
                raw_cka, raw_stress = _geometry_metrics(raw, clean, task_id)
                raw_ari = float(adjusted_rand_score(labels, raw_prediction))
                clean_ari = float(adjusted_rand_score(labels, clean_prediction))
                records.append(
                    {
                        "task_id": task_id,
                        "stratum": metadata["stratum"],
                        "raw_ari": raw_ari,
                        "clean_oracle_ari": clean_ari,
                        "recoverable_ari_gap": clean_ari - raw_ari,
                        "raw_cka": raw_cka,
                        "raw_stress": raw_stress,
                    }
                )

        def group_report(selected_rows: list[dict[str, Any]]) -> dict[str, Any]:
            raw_ari = [row["raw_ari"] for row in selected_rows]
            clean_ari = [row["clean_oracle_ari"] for row in selected_rows]
            gaps = [row["recoverable_ari_gap"] for row in selected_rows]
            cka = [row["raw_cka"] for row in selected_rows]
            stress = [row["raw_stress"] for row in selected_rows]

            def correlation(first: list[float], second: list[float]) -> float | None:
                if len(first) <= 2:
                    return None
                value = float(spearmanr(first, second).statistic)
                return value if np.isfinite(value) else None

            return {
                "raw_ari": _summary(raw_ari),
                "clean_oracle_ari": _summary(clean_ari),
                "recoverable_ari_gap": _summary(gaps),
                "raw_cka": _summary(cka),
                "raw_stress": _summary(stress),
                "spearman_raw_cka_vs_raw_ari": correlation(cka, raw_ari),
                "spearman_raw_stress_vs_raw_ari": correlation(stress, raw_ari),
            }

        strata = sorted({row["stratum"] for row in records})
        by_stratum = {
            stratum: group_report([row for row in records if row["stratum"] == stratum])
            for stratum in strata
        }
        overall = group_report(records)
        clean_summary = overall["clean_oracle_ari"]
        gap_summary = overall["recoverable_ari_gap"]
        if gap_summary["median"] is None:
            raise ValueError("cannot compute G* on an empty score set")
        g_star = float(gap_summary["median"])
        clean_gate = bool(
            clean_summary["median"] is not None
            and clean_summary["q10"] is not None
            and clean_summary["median"] >= 0.90
            and clean_summary["q10"] >= 0.80
        )
        if not clean_gate:
            decision = "invalid_corpus"
        elif g_star >= 0.05:
            decision = "authorize_one_capacity_probe"
        else:
            decision = "stop_geometry_recovery"
        report = {
            "status": "completed",
            "purpose": "TRAIN development outer corpus-necessity audit; forbidden for model or threshold selection",
            "task_count": len(records),
            "source_split": "train",
            "train_start_index": TRAIN_START_INDEX,
            "train_end_index_exclusive": TRAIN_START_INDEX + len(records),
            "embedding_manifest": str(embedding_manifest_path.resolve()),
            "embedding_manifest_sha256": _sha256(embedding_manifest_path),
            "cluster_protocol": str(cluster_protocol_path.resolve()),
            "cluster_protocol_sha256": _sha256(cluster_protocol_path),
            "audit_manifest": str(audit_manifest.resolve()),
            "label_root": str(label_root.resolve()),
            "Y_arrays_opened": True,
            "Y_first_opened_stage": "score",
            "K_used_only_by_cluster_stage": True,
            "CLM_used": False,
            "overall": overall,
            "by_stratum_descriptive_only": by_stratum,
            "predeclared_clean_ari_gate": {
                "median_min": 0.90,
                "q10_min": 0.80,
                "pass": clean_gate,
            },
            "G_star": g_star,
            "G_star_definition": "median of per-task paired (clean_oracle_ARI - raw_X_ARI) deltas",
            "G_star_threshold": 0.05,
            "predeclared_decision_rule": (
                "if clean_oracle_ARI median>=0.90 and q10>=0.80, authorize one capacity probe "
                "iff overall median(clean_oracle_ARI-raw_X_ARI)>=0.05; otherwise stop; "
                "strata are descriptive only"
            ),
            "decision": decision,
            "per_task": records,
            "elapsed_seconds": time.perf_counter() - started,
            "code_sha256": _sha256(Path(__file__)),
            "environment": _environment(),
        }
        _atomic_json(output_root / "report.json", report)
        _status(
            status_path,
            "completed",
            error=None,
            elapsed_seconds=report["elapsed_seconds"],
            decision=decision,
        )
        return report
    except Exception as exc:
        _status(
            status_path,
            "incomplete_compute",
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
            elapsed_seconds=time.perf_counter() - started,
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze-k", "export", "cluster", "score"))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--audit-manifest", type=Path, default=DEFAULT_AUDIT_MANIFEST)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--label-root", type=Path, default=DEFAULT_LABEL_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--task-count", type=int, default=DEFAULT_TASK_COUNT)
    args = parser.parse_args()
    root = args.output_root.resolve()
    if args.stage == "freeze-k":
        result = freeze_train_k(args.audit_manifest.resolve(), root, args.task_count)
    elif args.stage == "export":
        result = export_embeddings(
            args.manifest.resolve(), args.feature_root.resolve(), root, args.task_count
        )
    elif args.stage == "cluster":
        result = cluster_known_k(
            root / "embedding_manifest.json",
            root / "train_development_k_manifest.jsonl",
            root,
        )
    else:
        result = score_predictions(
            root / "embedding_manifest.json",
            root / "cluster_protocol.json",
            args.audit_manifest.resolve(),
            args.label_root.resolve(),
            root,
        )
    print(json.dumps({"stage": args.stage, "status": result.get("status", "completed")}))


if __name__ == "__main__":
    main()
