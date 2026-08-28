"""Geometry and clustering metrics used by the ZEUS target-only probe."""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


def centered_linear_cka(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Return centered linear CKA without detaching from autograd."""
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("CKA inputs must have shape [n_samples, n_features].")
    if x.shape[0] != y.shape[0]:
        raise ValueError("CKA inputs must have the same number of samples.")

    x_centered = x - x.mean(dim=0, keepdim=True)
    y_centered = y - y.mean(dim=0, keepdim=True)
    cross_covariance = x_centered.transpose(0, 1) @ y_centered
    x_covariance = x_centered.transpose(0, 1) @ x_centered
    y_covariance = y_centered.transpose(0, 1) @ y_centered

    numerator = torch.sum(cross_covariance.square())
    denominator = torch.linalg.matrix_norm(x_covariance, ord="fro")
    denominator = denominator * torch.linalg.matrix_norm(y_covariance, ord="fro")
    return numerator / denominator.clamp_min(eps)


def known_k_kmeans_ari(
    representation: torch.Tensor, labels: torch.Tensor, n_clusters: int, random_state: int
) -> float:
    """Cluster a representation with the true K and return adjusted Rand index."""
    values = representation.detach().cpu().numpy()
    target = labels.detach().cpu().numpy()
    assignments = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state).fit_predict(values)
    return float(adjusted_rand_score(target, assignments))


def distance_spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    """Spearman correlation of two representations' upper-triangle distances."""
    if x.shape[0] < 3:
        raise ValueError("At least three samples are needed for distance correlation.")
    indices = torch.triu_indices(x.shape[0], x.shape[0], offset=1, device=x.device)
    x_distances = torch.cdist(x, x)[indices[0], indices[1]].detach().cpu().numpy()
    y_distances = torch.cdist(y, y)[indices[0], indices[1]].detach().cpu().numpy()
    correlation = float(spearmanr(x_distances, y_distances).statistic)
    return 0.0 if math.isnan(correlation) else correlation


def mean_knn_overlap(x: torch.Tensor, y: torch.Tensor, k: int) -> float:
    """Mean fraction of each point's k nearest neighbours shared by x and y."""
    if x.shape[0] != y.shape[0]:
        raise ValueError("kNN inputs must have the same number of samples.")
    if not 0 < k < x.shape[0]:
        raise ValueError("k must be positive and less than the sample count.")

    x_distances = torch.cdist(x, x)
    y_distances = torch.cdist(y, y)
    x_distances.fill_diagonal_(float("inf"))
    y_distances.fill_diagonal_(float("inf"))
    x_neighbours = torch.topk(x_distances, k=k, largest=False, dim=1).indices
    y_neighbours = torch.topk(y_distances, k=k, largest=False, dim=1).indices
    overlap_count = (x_neighbours.unsqueeze(2) == y_neighbours.unsqueeze(1)).any(dim=2).sum(dim=1)
    return float((overlap_count.float() / k).mean().item())


def geometry_metrics(representation: torch.Tensor, reference: torch.Tensor, k: int) -> dict[str, float]:
    """Geometry-recovery metrics for one learned representation."""
    effective_k = min(k, representation.shape[0] - 1)
    return {
        "cka_to_x_ref": float(centered_linear_cka(representation, reference).detach().cpu().item()),
        "distance_spearman_to_x_ref": distance_spearman(representation, reference),
        "knn_overlap_to_x_ref": mean_knn_overlap(representation, reference, effective_k),
    }
