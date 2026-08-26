"""Label-free AE/DDBM training-step contract for Generator V4.

This module only provides a one-step engineering smoke path. It is not a
training launcher and makes no performance claim. Labels, K, CLM and
observation-family metadata are absent from the function signature.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

from .models import (
    DatasetContextDDBM,
    GlobalAE,
    ddbm_geometry_loss,
    mixed_type_reconstruction_loss,
)


def _linear_noise_scale(timestep: int, diffusion_steps: int, device: torch.device) -> torch.Tensor:
    if not 0 <= timestep < diffusion_steps:
        raise ValueError("timestep outside diffusion range")
    fraction = torch.tensor(float(timestep) / max(diffusion_steps - 1, 1), device=device)
    return torch.sqrt(torch.clamp(0.02 + 0.98 * fraction, min=1e-4))


def v4_train_step(
    ae: GlobalAE,
    ddbm: DatasetContextDDBM,
    values: torch.Tensor,
    missing_mask: torch.Tensor,
    feature_types: torch.Tensor,
    clean_latent: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    *,
    timestep: int = 128,
    generator: Optional[torch.Generator] = None,
) -> Dict[str, float]:
    """Run one label-free CPU/GPU engineering step and return scalar diagnostics."""

    if generator is None:
        generator = torch.Generator(device="cpu").manual_seed(20260824)
    optimizer.zero_grad(set_to_none=True)
    ae_output = ae(values, missing_mask, feature_types)
    reconstruction_loss, _ = mixed_type_reconstruction_loss(
        ae_output, values, missing_mask, feature_types
    )
    noise = torch.randn(
        clean_latent.shape,
        generator=generator,
        device=clean_latent.device,
        dtype=clean_latent.dtype,
    )
    scale = _linear_noise_scale(timestep, ddbm.config.diffusion_steps, clean_latent.device)
    noisy_state = clean_latent + scale * noise
    recovered = ddbm(noisy_state, ae_output.observation, timestep)
    geometry_loss, geometry_metrics = ddbm_geometry_loss(
        recovered,
        clean_latent,
        generator=generator,
        pair_count=min(256, clean_latent.shape[0] * 4),
    )
    total = reconstruction_loss + geometry_loss
    if not torch.isfinite(total):
        raise FloatingPointError("V4 training step produced non-finite loss")
    total.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        list(ae.parameters()) + list(ddbm.parameters()), max_norm=100.0
    )
    if not torch.isfinite(gradient_norm):
        raise FloatingPointError("V4 training step produced non-finite gradient")
    optimizer.step()
    return {
        "total_loss": float(total.detach()),
        "reconstruction_loss": float(reconstruction_loss.detach()),
        "geometry_loss": float(geometry_loss.detach()),
        "gram_loss": float(geometry_metrics["gram_loss"]),
        "distance_loss": float(geometry_metrics["distance_loss"]),
        "gradient_norm": float(gradient_norm),
    }
