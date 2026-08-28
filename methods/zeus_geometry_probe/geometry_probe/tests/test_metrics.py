import torch

from geometry_probe.core import ProbeConfig, materialize_tasks
from geometry_probe.metrics import centered_linear_cka, distance_spearman, mean_knn_overlap


def test_cka_is_invariant_to_scalar_rescaling() -> None:
    torch.manual_seed(1)
    x = torch.randn(20, 5)
    y = torch.randn(20, 3)
    assert torch.allclose(centered_linear_cka(x, y), centered_linear_cka(x, 7.0 * y), atol=1e-6)


def test_cka_is_invariant_to_feature_rotation() -> None:
    torch.manual_seed(2)
    x = torch.randn(20, 5)
    y = torch.randn(20, 4)
    rotation, _ = torch.linalg.qr(torch.randn(4, 4))
    assert torch.allclose(centered_linear_cka(x, y), centered_linear_cka(x, y @ rotation), atol=1e-6)


def test_cka_has_finite_nonzero_gradient() -> None:
    torch.manual_seed(4)
    representation = torch.randn(12, 6, requires_grad=True)
    reference = torch.randn(12, 3)
    loss = 1.0 - centered_linear_cka(representation, reference)
    loss.backward()
    assert torch.isfinite(representation.grad).all()
    assert representation.grad.abs().sum().item() > 0.0


def test_geometry_metrics_are_identity_at_identical_nonconstant_inputs() -> None:
    torch.manual_seed(3)
    x = torch.randn(15, 4)
    assert torch.allclose(centered_linear_cka(x, x), torch.tensor(1.0), atol=1e-6)
    assert abs(distance_spearman(x, x) - 1.0) < 1e-6
    assert abs(mean_knn_overlap(x, x, k=4) - 1.0) < 1e-6


def test_materialized_tasks_pair_exactly_by_seed() -> None:
    config = ProbeConfig(
        num_gaussians=3,
        min_points=4,
        max_points=5,
        dim=5,
        categorical_chance=0.0,
        max_blocks=1,
    )
    first = materialize_tasks([101, 102], "transformed", config)
    second = materialize_tasks([101, 102], "transformed", config)
    for left, right in zip(first, second):
        assert left.seed == right.seed
        assert left.generator_mode == right.generator_mode
        assert left.source_generator_mode == right.source_generator_mode
        assert torch.equal(left.x_obs, right.x_obs)
        assert torch.equal(left.labels, right.labels)
        assert torch.equal(left.x_ref, right.x_ref)
        assert torch.equal(left.probabilities, right.probabilities)
