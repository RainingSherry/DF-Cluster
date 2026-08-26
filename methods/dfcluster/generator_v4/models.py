"""Global AE and dataset-context diffusion bridge architecture contracts.

The defaults mirror the Generator V4 plan, while tests and CPU smoke callers
can pass much smaller widths/layer counts. No method here accepts labels, K,
CLM, generator family, or clean-generator hidden parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Optional, Tuple

import torch
from torch import nn

FEATURE_TYPE_COUNT = 5


def _masked_mean(values: torch.Tensor, valid: torch.Tensor, dim: int) -> torch.Tensor:
    weights = valid.to(values.dtype)
    numerator = (values * weights.unsqueeze(-1)).sum(dim=dim)
    denominator = weights.sum(dim=dim).clamp_min(1.0).unsqueeze(-1)
    return numerator / denominator


def _safe_valid(mask: torch.Tensor) -> torch.Tensor:
    valid = ~mask
    all_missing = valid.sum(dim=1) == 0
    if all_missing.any():
        valid = valid.clone()
        valid[all_missing, 0] = True
    return valid


@dataclass(frozen=True)
class GlobalAEConfig:
    cell_hidden_dim: int = 512
    heads: int = 8
    column_layers: int = 4
    row_layers: int = 8
    ffn_dim: int = 2048
    perceiver_queries: int = 8
    latent_dim: int = 128
    dropout: float = 0.0

    def validate(self) -> None:
        if self.heads <= 0 or self.cell_hidden_dim <= 0:
            raise ValueError("hidden dimensions and heads must be positive")
        if self.cell_hidden_dim % self.heads:
            raise ValueError("cell_hidden_dim must be divisible by heads")
        if self.column_layers < 1 or self.row_layers < 1:
            raise ValueError("column_layers and row_layers must be positive")
        if self.ffn_dim < self.cell_hidden_dim:
            raise ValueError("ffn_dim must be at least cell_hidden_dim")
        if self.perceiver_queries < 1 or self.latent_dim != 128:
            raise ValueError("latent_dim must be 128 and queries positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


@dataclass(frozen=True)
class DDBMConfig:
    latent_dim: int = 128
    hidden_dim: int = 768
    heads: int = 12
    layers: int = 16
    ffn_dim: int = 3072
    diffusion_steps: int = 512
    dropout: float = 0.0
    gradient_checkpointing: bool = True

    def validate(self) -> None:
        if self.latent_dim != 128:
            raise ValueError("DDBM latent_dim must be 128")
        if self.heads <= 0 or self.hidden_dim <= 0 or self.hidden_dim % self.heads:
            raise ValueError("hidden_dim must be positive and divisible by heads")
        if self.layers < 1 or self.ffn_dim < self.hidden_dim:
            raise ValueError("invalid DDBM layer/FFN dimensions")
        if self.diffusion_steps != 512:
            raise ValueError("DDBM diffusion_steps must be 512")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


@dataclass(frozen=True)
class GlobalAEOutput:
    observation: torch.Tensor
    schema_tokens: torch.Tensor
    reconstruction: torch.Tensor
    mask_logits: torch.Tensor
    category_logits: torch.Tensor
    scale: torch.Tensor


class GlobalAE(nn.Module):
    """Permutation-equivariant mixed-type table autoencoder contract."""

    def __init__(self, config: GlobalAEConfig | None = None) -> None:
        super().__init__()
        self.config = config or GlobalAEConfig()
        self.config.validate()
        h = self.config.cell_hidden_dim
        self.value_projection = nn.Sequential(nn.Linear(1, h), nn.GELU(), nn.Linear(h, h))
        self.type_embedding = nn.Embedding(FEATURE_TYPE_COUNT, h)
        self.mask_embedding = nn.Embedding(2, h)
        column_layer = nn.TransformerEncoderLayer(
            d_model=h, nhead=self.config.heads, dim_feedforward=self.config.ffn_dim,
            dropout=self.config.dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        row_layer = nn.TransformerEncoderLayer(
            d_model=h, nhead=self.config.heads, dim_feedforward=self.config.ffn_dim,
            dropout=self.config.dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.column_encoder = nn.TransformerEncoder(
            column_layer, self.config.column_layers, norm=nn.LayerNorm(h)
        )
        self.row_encoder = nn.TransformerEncoder(
            row_layer, self.config.row_layers, norm=nn.LayerNorm(h)
        )
        self.perceiver_queries = nn.Parameter(
            torch.randn(1, self.config.perceiver_queries, h) * 0.02
        )
        self.perceiver = nn.MultiheadAttention(
            h, self.config.heads, dropout=self.config.dropout, batch_first=True
        )
        self.observation_projection = nn.Sequential(
            nn.LayerNorm(self.config.perceiver_queries * h),
            nn.Linear(self.config.perceiver_queries * h, 2 * h),
            nn.GELU(), nn.Linear(2 * h, 128),
        )
        self.row_decoder = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, h), nn.GELU())
        self.reconstruction_head = nn.Sequential(
            nn.LayerNorm(2 * h), nn.Linear(2 * h, h), nn.GELU(), nn.Linear(h, 1)
        )
        self.mask_head = nn.Sequential(
            nn.LayerNorm(2 * h), nn.Linear(2 * h, h), nn.GELU(), nn.Linear(h, 1)
        )
        self.category_head = nn.Sequential(
            nn.LayerNorm(2 * h), nn.Linear(2 * h, h), nn.GELU(), nn.Linear(h, 16)
        )
        self.scale_head = nn.Sequential(
            nn.LayerNorm(2 * h), nn.Linear(2 * h, h), nn.GELU(), nn.Linear(h, 1)
        )

    def _validate_inputs(
        self, values: torch.Tensor, missing_mask: torch.Tensor, feature_types: torch.Tensor
    ) -> None:
        if values.ndim != 2 or missing_mask.shape != values.shape:
            raise ValueError("values and missing_mask must have matching [N,D] shapes")
        if feature_types.shape != (values.shape[1],):
            raise ValueError("feature_types must have shape [D]")
        if not torch.isfinite(values).all():
            raise ValueError("values must be finite")

    def encode(
        self, values: torch.Tensor, missing_mask: torch.Tensor, feature_types: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._validate_inputs(values, missing_mask, feature_types)
        safe_values = values.masked_fill(missing_mask, 0.0)
        cells = self.value_projection(safe_values.unsqueeze(-1))
        cells = cells + self.type_embedding(
            feature_types.long().clamp(0, FEATURE_TYPE_COUNT - 1)
        ).unsqueeze(0)
        cells = cells + self.mask_embedding(missing_mask.long())
        valid = _safe_valid(missing_mask)
        column_tokens = _masked_mean(cells, valid, dim=0)
        column_tokens = self.column_encoder(column_tokens.unsqueeze(0)).squeeze(0)
        contextual_cells = self.row_encoder(cells + column_tokens.unsqueeze(0))
        query = self.perceiver_queries.expand(values.shape[0], -1, -1)
        pooled, _ = self.perceiver(
            query, contextual_cells, contextual_cells,
            key_padding_mask=~valid, need_weights=False,
        )
        observation = self.observation_projection(pooled.reshape(values.shape[0], -1))
        return observation, column_tokens, contextual_cells

    def forward(
        self, values: torch.Tensor, missing_mask: torch.Tensor, feature_types: torch.Tensor
    ) -> GlobalAEOutput:
        observation, schema_tokens, _ = self.encode(values, missing_mask, feature_types)
        rows = self.row_decoder(observation).unsqueeze(1).expand(-1, values.shape[1], -1)
        schema = schema_tokens.unsqueeze(0).expand(values.shape[0], -1, -1)
        decoder_input = torch.cat((rows, schema), dim=-1)
        reconstruction = self.reconstruction_head(decoder_input).squeeze(-1)
        mask_logits = self.mask_head(decoder_input).squeeze(-1)
        category_logits = self.category_head(decoder_input)
        scale = torch.nn.functional.softplus(self.scale_head(decoder_input).squeeze(-1)) + 1e-4
        return GlobalAEOutput(
            observation, schema_tokens, reconstruction, mask_logits, category_logits, scale
        )


def mixed_type_reconstruction_loss(
    output: GlobalAEOutput,
    target: torch.Tensor,
    missing_mask: torch.Tensor,
    feature_types: torch.Tensor,
    *,
    target_missing_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Type-aware masked/denoising reconstruction objective.

    ``missing_mask`` is the mask presented to the encoder. When denoising
    corruption is active, ``target_missing_mask`` identifies only the native
    missing cells that must be excluded from reconstruction supervision. This
    keeps masked cells in the loss while preserving the original missing-mask
    target for the mask head.
    """

    if target.ndim != 2 or target.shape != output.reconstruction.shape:
        raise ValueError("target must match reconstruction shape [N,D]")
    if missing_mask.shape != target.shape or feature_types.shape != (target.shape[1],):
        raise ValueError("mask/types do not match target")
    if target_missing_mask is None:
        target_missing_mask = missing_mask
    if target_missing_mask.shape != target.shape:
        raise ValueError("target_missing_mask must match target")
    valid = ~target_missing_mask
    if not bool(valid.any()):
        raise ValueError("reconstruction requires at least one observed cell")
    losses = []
    numerical = (feature_types == 0) | (feature_types == 4)
    numerical_valid = valid & numerical.unsqueeze(0)
    if bool(numerical_valid.any()):
        losses.append(torch.nn.functional.smooth_l1_loss(
            output.reconstruction[numerical_valid], target[numerical_valid], reduction="mean"
        ))
    discrete = (feature_types == 1) | (feature_types == 2)
    discrete_valid = valid & discrete.unsqueeze(0)
    if bool(discrete_valid.any()):
        labels = target[discrete_valid].round().long().clamp(0, 15)
        losses.append(torch.nn.functional.cross_entropy(
            output.category_logits[discrete_valid], labels, reduction="mean"
        ))
    count_valid = valid & (feature_types == 3).unsqueeze(0)
    if bool(count_valid.any()):
        rates = torch.nn.functional.softplus(output.reconstruction[count_valid]) + 1e-4
        counts = target[count_valid].clamp_min(0.0)
        losses.append(torch.nn.functional.poisson_nll_loss(
            rates, counts, log_input=False, full=False, reduction="mean"
        ))
    mask_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        output.mask_logits, target_missing_mask.to(output.mask_logits.dtype), reduction="mean"
    )
    losses.append(0.10 * mask_loss)
    total = torch.stack(losses).sum()
    return total, {"reconstruction": total.detach(), "mask": mask_loss.detach()}


class _DDBMBlock(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(hidden_dim)
        self.cross_norm = nn.LayerNorm(hidden_dim)
        self.self_attention = nn.MultiheadAttention(
            hidden_dim, heads, dropout=dropout, batch_first=True
        )
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, ffn_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(ffn_dim, hidden_dim), nn.Dropout(dropout),
        )

    def forward(self, rows: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        update, _ = self.self_attention(
            self.self_norm(rows), self.self_norm(rows), self.self_norm(rows), need_weights=False
        )
        rows = rows + update
        update, _ = self.cross_attention(
            self.cross_norm(rows), self.cross_norm(context), self.cross_norm(context), need_weights=False
        )
        rows = rows + update
        return rows + self.ffn(rows)


class _TimeEmbedding(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        half = self.hidden_dim // 2
        frequency = torch.exp(
            -math.log(10_000.0)
            * torch.arange(half, device=timestep.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        angles = timestep.float().unsqueeze(-1) * frequency.unsqueeze(0)
        embedding = torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)
        if embedding.shape[-1] < self.hidden_dim:
            embedding = torch.nn.functional.pad(embedding, (0, self.hidden_dim - embedding.shape[-1]))
        return self.projection(embedding)


class DatasetContextDDBM(nn.Module):
    """Conditional dataset-context bridge with fixed T=512."""

    def __init__(self, config: DDBMConfig | None = None) -> None:
        super().__init__()
        self.config = config or DDBMConfig()
        self.config.validate()
        h = self.config.hidden_dim
        self.input_projection = nn.Linear(128, h)
        self.context_projection = nn.Linear(128, h)
        self.time_embedding = _TimeEmbedding(h)
        self.blocks = nn.ModuleList([
            _DDBMBlock(h, self.config.heads, self.config.ffn_dim, self.config.dropout)
            for _ in range(self.config.layers)
        ])
        self.output_projection = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, 128))

    def _validate_and_prepare(
        self, noisy_state: torch.Tensor, context: torch.Tensor, timestep: torch.Tensor | int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if noisy_state.ndim != 2 or noisy_state.shape[1] != 128:
            raise ValueError("noisy_state must have shape [N, 128]")
        if context.ndim != 2 or context.shape[1] != 128:
            raise ValueError("context must have shape [M, 128]")
        if not torch.isfinite(noisy_state).all() or not torch.isfinite(context).all():
            raise ValueError("DDBM inputs must be finite")
        if isinstance(timestep, int):
            time = torch.full((noisy_state.shape[0],), timestep, device=noisy_state.device)
        else:
            time = timestep.to(noisy_state.device)
            if time.ndim == 0:
                time = time.expand(noisy_state.shape[0])
            if time.shape != (noisy_state.shape[0],):
                raise ValueError("timestep must be scalar or shape [N]")
        if bool((time < 0).any()) or bool((time >= self.config.diffusion_steps).any()):
            raise ValueError("timestep outside [0, T)")
        return noisy_state, context, time

    def _raw_forward(
        self, noisy_state: torch.Tensor, context: torch.Tensor, timestep: torch.Tensor | int
    ) -> torch.Tensor:
        noisy_state, context, time = self._validate_and_prepare(noisy_state, context, timestep)
        rows = self.input_projection(noisy_state).unsqueeze(0)
        context_rows = self.context_projection(context).unsqueeze(0)
        rows = rows + self.time_embedding(time).unsqueeze(0)
        for block in self.blocks:
            if self.training and self.config.gradient_checkpointing:
                rows = torch.utils.checkpoint.checkpoint(
                    lambda state, layer=block: layer(state, context_rows),
                    rows,
                    use_reentrant=False,
                )
            else:
                rows = block(rows, context_rows)
        return self.output_projection(rows.squeeze(0))

    def predict_noise(
        self, noisy_state: torch.Tensor, context: torch.Tensor, timestep: torch.Tensor | int
    ) -> torch.Tensor:
        """Return the unnormalized bridge/noise prediction for Stage B."""

        output = self._raw_forward(noisy_state, context, timestep)
        if not torch.isfinite(output).all():
            raise FloatingPointError("DDBM noise prediction is non-finite")
        return output

    def forward(
        self, noisy_state: torch.Tensor, context: torch.Tensor, timestep: torch.Tensor | int
    ) -> torch.Tensor:
        """Return a centered/RMS-normalized clean-factor geometry output."""

        output = self._raw_forward(noisy_state, context, timestep)
        output = output - output.mean(dim=0, keepdim=True)
        output = output / torch.sqrt(output.square().mean().clamp_min(1e-6))
        return output


def centered_normalized_gram_torch(values: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    centered = values - values.mean(dim=0, keepdim=True)
    gram = centered @ centered.transpose(0, 1)
    return gram / gram.norm().clamp_min(eps)


def ddbm_geometry_loss(
    predicted: torch.Tensor,
    clean_latent: torch.Tensor,
    *,
    generator: Optional[torch.Generator] = None,
    pair_count: int = 4096,
    gram_weight: float = 1.0,
    distance_weight: float = 0.5,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute label-free Gram and sampled-distance bridge losses."""

    if predicted.shape != clean_latent.shape or predicted.ndim != 2 or predicted.shape[1] != 128:
        raise ValueError("predicted and clean_latent must both have shape [N, 128]")
    if generator is None:
        generator = torch.Generator(device="cpu").manual_seed(0)
    pred_gram = centered_normalized_gram_torch(predicted)
    clean_gram = centered_normalized_gram_torch(clean_latent)
    gram_loss = torch.mean((pred_gram - clean_gram) ** 2)
    count = predicted.shape[0]
    pairs = min(pair_count, max(1, count * 4))
    first = torch.randint(count, (pairs,), generator=generator, device="cpu").to(predicted.device)
    second = torch.randint(count, (pairs,), generator=generator, device="cpu").to(predicted.device)
    pred_distance = (predicted[first] - predicted[second]).norm(dim=-1)
    clean_distance = (clean_latent[first] - clean_latent[second]).norm(dim=-1)
    pred_scale = pred_distance.detach().median().clamp_min(1e-6)
    clean_scale = clean_distance.detach().median().clamp_min(1e-6)
    distance_loss = torch.mean(torch.abs(
        pred_distance / pred_scale - clean_distance / clean_scale
    ))
    total = gram_weight * gram_loss + distance_weight * distance_loss
    return total, {"gram_loss": gram_loss.detach(), "distance_loss": distance_loss.detach()}
