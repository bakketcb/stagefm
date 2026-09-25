"""The abdominal-CT vision-language encoder and its frozen loading path.

The encoder named by the manuscript is the released 2026 abdominal CT
vision-language foundation model. Its weights are distributed under a data-use
agreement and are not bundled here, so :func:`build_encoder` loads them when a local
checkpoint is pointed at by configuration and otherwise falls back to the compact
3D vision transformer implemented in this file. The fallback is a fully specified
encoder in its own right: it has the same patch embedding, the same
attention projections that carry the low-rank adapters, and the same token output
contract, so every downstream module is exercised identically with either backbone.
Which encoder was used is recorded in the run metadata and in the verification
report.

Ref: Methods Sec. 4.5 (encoder, frozen with low-rank adaptation); Sec. 4.4 (the
released abdominal CT dataset and the public auxiliary resources); Sec. 4.6
(frozen encoder rather than a task-specific backbone).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from ..utils.config import EncoderConfig
from ..utils.logging import get_logger

LOGGER = get_logger("models.encoder")


class EncoderUnavailable(RuntimeError):
    """Raised when a released encoder was requested and its weights are not present."""


class Attention(nn.Module):
    """Multi-head self-attention with separately named projections."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, tokens: Tensor) -> Tensor:
        batch, count, _ = tokens.shape
        query = self.q_proj(tokens).reshape(batch, count, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(tokens).reshape(batch, count, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(tokens).reshape(batch, count, self.num_heads, self.head_dim).transpose(1, 2)
        weights = torch.softmax(query @ key.transpose(-2, -1) * self.scale, dim=-1)
        weights = self.attn_dropout(weights)
        merged = (weights @ value).transpose(1, 2).reshape(batch, count, self.embed_dim)
        output: Tensor = self.proj_dropout(self.out_proj(merged))
        return output


class TransformerBlock(nn.Module):
    """Pre-norm transformer block."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = Attention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(embed_dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, embed_dim), nn.Dropout(dropout))

    def forward(self, tokens: Tensor) -> Tensor:
        tokens = tokens + self.attention(self.norm1(tokens))
        out: Tensor = tokens + self.mlp(self.norm2(tokens))
        return out


@dataclass(frozen=True)
class EncoderOutput:
    """Token sequence plus the grid shape it came from."""

    tokens: Tensor
    grid: tuple[int, int, int]


class CompactCTEncoder(nn.Module):
    """A 3D abdominal CT vision transformer with the projection names LoRA targets."""

    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size
        self.embed_dim = config.embed_dim
        self.patch_embed = nn.Conv3d(1, config.embed_dim, kernel_size=config.patch_size, stride=config.patch_size)
        self.positional = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        nn.init.trunc_normal_(self.positional, std=0.02)
        self.blocks = nn.ModuleList([TransformerBlock(config.embed_dim, config.num_heads) for _ in range(config.depth)])
        self.norm = nn.LayerNorm(config.embed_dim)

    def compute_grid(self, volume: Tensor) -> tuple[int, int, int]:
        depth, height, width = volume.shape[-3:]
        return (depth // self.patch_size, height // self.patch_size, width // self.patch_size)

    def forward(self, volume: Tensor) -> EncoderOutput:
        if volume.ndim != 5:
            raise ValueError(f"encoder expects (B, 1, D, H, W), got {tuple(volume.shape)}")
        grid = self.compute_grid(volume)
        embedded = self.patch_embed(volume)
        tokens = embedded.flatten(2).transpose(1, 2)
        tokens = tokens + self.positional
        for block in self.blocks:
            tokens = block(tokens)
        return EncoderOutput(tokens=self.norm(tokens), grid=grid)


class MerlinEncoder(nn.Module):
    """Wrapper around a locally available checkpoint of the released encoder.

    The checkpoint is expected to expose a ``vision_model``-style submodule whose
    attention projections carry the names the adapters bind to. Only the vision
    tower is used: the staging head reads image tokens, not generated text.
    """

    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        path = Path(config.weights_path or "")
        if not path.is_file():
            raise EncoderUnavailable(f"encoder checkpoint not found: {path}")
        try:
            payload: Any = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as error:  # noqa: BLE001 - the boundary is a file we do not control
            raise EncoderUnavailable(f"could not read encoder checkpoint {path}: {error}") from error
        model = payload.get("model") if isinstance(payload, dict) else payload
        if not isinstance(model, nn.Module):
            raise EncoderUnavailable(f"checkpoint {path} does not contain a torch module")
        self.backbone = model
        self.patch_size = config.patch_size
        self.embed_dim = int(getattr(model, "embed_dim", config.embed_dim))
        self._grid_hint = config.patch_size

    def forward(self, volume: Tensor) -> EncoderOutput:
        depth, height, width = volume.shape[-3:]
        grid = (depth // self.patch_size, height // self.patch_size, width // self.patch_size)
        output = self.backbone(pixel_values=volume)
        tokens = getattr(output, "last_hidden_state", None)
        if tokens is None and isinstance(output, dict):
            tokens = output.get("last_hidden_state")
        if tokens is None:
            raise EncoderUnavailable("encoder output did not expose last_hidden_state")
        return EncoderOutput(tokens=tokens, grid=grid)


def build_encoder(config: EncoderConfig) -> tuple[nn.Module, dict[str, Any]]:
    """Load the released encoder when its weights are present, otherwise the compact backbone."""
    if config.weights_path:
        try:
            encoder: nn.Module = MerlinEncoder(config)
            return encoder, {"encoder": "released-checkpoint", "weights_path": config.weights_path, "embed_dim": encoder.embed_dim}
        except EncoderUnavailable as error:
            LOGGER.warning("falling back to the compact backbone: %s", error)
    encoder = CompactCTEncoder(config)
    return encoder, {"encoder": "compact-3d-vit", "weights_path": None, "embed_dim": config.embed_dim}


def encoder_token_count(grid: tuple[int, int, int]) -> int:
    """Number of patch tokens for a token grid."""
    return int(grid[0] * grid[1] * grid[2])


def cap_tokens(tokens: Tensor, cap: int) -> Tensor:
    """Truncate the token sequence to at most ``cap`` tokens.

    Token order follows the patch grid, so truncation keeps a contiguous region of
    the volume rather than a random subset of positions.
    """
    if tokens.shape[1] <= cap:
        return tokens
    return tokens[:, :cap, :]
