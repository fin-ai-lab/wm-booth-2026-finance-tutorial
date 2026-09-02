"""Standalone loader for the tutorial's LeJEPA transformer encoder."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from huggingface_hub import hf_hub_download
from torch import nn


@dataclass
class TransformerConfig:
    hidden_size: int = 384
    num_hidden_layers: int = 12
    num_attention_heads: int = 6
    intermediate_size: int = 1536
    patch_size: int = 8
    layer_norm_eps: float = 1e-12
    drop_path_rate: float = 0.1
    rescale_residual_init: bool = False
    recency_bias_window: int = 16
    patch_embed_identity: bool = False


class PatchEmbedding1D(nn.Module):
    def __init__(self, n_features: int, patch_size: int, hidden_size: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv1d(
            n_features,
            hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.norm = nn.Identity()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        padding = (-values.shape[-1]) % self.patch_size
        if padding:
            values = functional.pad(values, (0, padding))
        return self.norm(self.proj(values).transpose(1, 2))


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = nn.MultiheadAttention(
            config.hidden_size,
            config.num_attention_heads,
            batch_first=True,
        )
        self.drop_path1 = nn.Identity()
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size),
        )
        self.drop_path2 = nn.Identity()

    def forward(
        self,
        values: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None,
        attention_bias: torch.Tensor | None,
    ) -> torch.Tensor:
        normalized = self.norm1(values)
        attended, _ = self.attn(
            normalized,
            normalized,
            normalized,
            key_padding_mask=key_padding_mask,
            attn_mask=attention_bias,
            need_weights=False,
        )
        values = values + self.drop_path1(attended)
        return values + self.drop_path2(self.mlp(self.norm2(values)))


class LeJEPAEncoder(nn.Module):
    """The transformer backbone architecture stored in the demo checkpoint."""

    def __init__(
        self,
        config: TransformerConfig,
        *,
        market_features: int,
        information_features: int,
        embedding_size: int,
        max_sequence_length: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.n_features = market_features + information_features
        self.n_info_channels = information_features
        self.d_embedding = embedding_size
        self.patch_size = config.patch_size
        self.patch_embed = PatchEmbedding1D(
            market_features,
            config.patch_size,
            config.hidden_size,
        )
        self.cls_token = nn.Parameter(torch.empty(1, 1, config.hidden_size))
        self.position_embeddings = nn.Parameter(
            torch.empty(1, max_sequence_length + 1, config.hidden_size)
        )
        self.recency_slope_raw = nn.Parameter(torch.empty(config.num_hidden_layers))
        self.info_proj = nn.Linear(information_features, config.hidden_size)
        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.num_hidden_layers)
        )
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.head = nn.Linear(config.hidden_size, embedding_size)

    def _attention_bias(
        self,
        sequence_length: int,
        patch_count: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        positions = torch.full(
            (sequence_length,),
            float(max(patch_count - 1, 0)),
            device=device,
            dtype=torch.float32,
        )
        positions[1 : patch_count + 1] = torch.arange(
            patch_count,
            device=device,
            dtype=torch.float32,
        )
        distances = (positions[:, None] - positions[None, :]).abs()
        distances[-1, :] = 0
        distances[:, -1] = 0
        slopes = functional.softplus(self.recency_slope_raw)
        return (-slopes[:, None, None] * distances[None]).to(dtype)

    def forward(
        self,
        values: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = values.shape[0]
        information = self.info_proj(values[:, -self.n_info_channels :, -1]).unsqueeze(1)
        market_values = values[:, : -self.n_info_channels]
        patches = self.patch_embed(market_values)
        patch_count = patches.shape[1]
        cls = self.cls_token.expand(batch_size, -1, -1)
        hidden = torch.cat([cls, patches], dim=1)
        hidden = hidden + self.position_embeddings[:, : hidden.shape[1]]
        hidden = torch.cat([hidden, information], dim=1)

        padding_mask = None
        if lengths is not None:
            valid_patches = (lengths + self.patch_size - 1) // self.patch_size
            patch_indices = torch.arange(patch_count, device=values.device).unsqueeze(0)
            patch_mask = patch_indices >= valid_patches.unsqueeze(1)
            special = torch.zeros((batch_size, 1), dtype=torch.bool, device=values.device)
            padding_mask = torch.cat([special, patch_mask, special], dim=1)

        biases = self._attention_bias(
            hidden.shape[1],
            patch_count,
            device=hidden.device,
            dtype=hidden.dtype,
        )
        for index, block in enumerate(self.blocks):
            block_padding = padding_mask
            if block_padding is not None:
                block_padding = torch.zeros_like(block_padding, dtype=hidden.dtype).masked_fill(
                    block_padding, float("-inf")
                )
            hidden = block(
                hidden,
                key_padding_mask=block_padding,
                attention_bias=biases[index],
            )
        pooled = hidden[:, 0]
        return self.head(self.layernorm(pooled))


def _build_encoder(config: dict[str, Any], state: dict[str, torch.Tensor]) -> LeJEPAEncoder:
    backbone = config["backbone_config"]
    transformer_config = TransformerConfig(**backbone)
    prefix = "backbone."
    backbone_state = {
        key.removeprefix(prefix): value for key, value in state.items() if key.startswith(prefix)
    }
    market_features = int(backbone_state["patch_embed.proj.weight"].shape[1])
    information_features = int(backbone_state["info_proj.weight"].shape[1])
    max_sequence_length = int(backbone_state["position_embeddings"].shape[1]) - 1
    encoder = LeJEPAEncoder(
        transformer_config,
        market_features=market_features,
        information_features=information_features,
        embedding_size=int(config["d_embedding"]),
        max_sequence_length=max_sequence_length,
    )
    encoder.load_state_dict(backbone_state, strict=True)
    return encoder.eval()


def load_lejepa_encoder(
    repo_id: str | Path,
    *,
    token: str | None = None,
    revision: str | None = None,
) -> tuple[LeJEPAEncoder, dict[str, Any]]:
    """Load a Hub or local checkpoint and return its frozen encoder."""
    local = Path(repo_id)
    if local.is_dir():
        config_path = local / "config.json"
        weights_path = local / "model.pt"
        metadata_path = local / "train_meta.json"
    else:
        hub_id = str(repo_id)
        config_path = hf_hub_download(hub_id, "config.json", token=token, revision=revision)
        weights_path = hf_hub_download(hub_id, "model.pt", token=token, revision=revision)
        metadata_path = hf_hub_download(hub_id, "train_meta.json", token=token, revision=revision)
    with Path(config_path).open() as handle:
        config = json.load(handle)
    with Path(metadata_path).open() as handle:
        metadata = json.load(handle)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    return _build_encoder(config, state), metadata
