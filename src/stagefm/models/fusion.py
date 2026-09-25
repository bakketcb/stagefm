"""Cross-attention fusion of the image tokens and the non-imaging streams.

Three fusion strategies are implemented because the manuscript's justification
names two rejected alternatives: early concatenation and late averaging were both
considered and cross-attention was chosen because the streams differ in
heterogeneity and missingness and because averaging cannot express an interaction
between peritumoral texture and nodal sampling practice. All three sit behind the
same token contract, so the justification can be evaluated rather than asserted.

Absent streams are represented by a learned absence token combined with a presence
embedding; the fusion block never receives an imputed value.

Ref: Methods Sec. 4.5 (fusion block); Sec. 4.6 (cross-attention over early
concatenation and late averaging).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ..data.schema import NON_IMAGING_STREAMS, Stream
from ..utils.config import FusionConfig


@dataclass(frozen=True)
class FusionInputs:
    """The tensors the fusion block consumes for one batch."""

    image_tokens: Tensor
    radiomics: Tensor
    clinical: Tensor
    endoscopy: Tensor
    pathology: Tensor
    presence: Tensor


@dataclass(frozen=True)
class FusionOutput:
    """A fused record embedding plus the contextualised token sequence."""

    fused: Tensor
    context: Tensor
    stream_tokens: Tensor


class StreamEncoder(nn.Module):
    """Map one dense stream to a single token."""

    def __init__(self, input_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, values: Tensor) -> Tensor:
        projected: Tensor = self.projection(values)
        return projected


class TermStreamEncoder(nn.Module):
    """Embed a term list by averaging its term vectors."""

    def __init__(self, vocabulary_size: int, embed_dim: int, term_dim: int = 32) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, term_dim, padding_idx=0)
        self.projection = nn.Sequential(nn.Linear(term_dim, embed_dim), nn.GELU(), nn.LayerNorm(embed_dim))

    def forward(self, indices: Tensor) -> Tensor:
        vectors = self.embedding(indices)
        weights = (indices != 0).float().unsqueeze(-1)
        pooled = (vectors * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
        projected: Tensor = self.projection(pooled)
        return projected


class CrossAttentionFusion(nn.Module):
    """Self-attention over image and stream tokens, then attention pooling."""

    def __init__(
        self,
        config: FusionConfig,
        radiomics_dim: int,
        clinical_dim: int,
        endoscopy_vocab: int,
        pathology_vocab: int,
        active_streams: tuple[Stream, ...],
        use_image_tokens: bool = True,
    ) -> None:
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        self.active_streams = tuple(active_streams)
        self.use_image_tokens = use_image_tokens

        self.radiomics_encoder = StreamEncoder(radiomics_dim, config.embed_dim)
        self.clinical_encoder = StreamEncoder(clinical_dim, config.embed_dim)
        self.endoscopy_encoder = TermStreamEncoder(endoscopy_vocab, config.embed_dim)
        self.pathology_encoder = TermStreamEncoder(pathology_vocab, config.embed_dim)

        self.image_projection = nn.Identity()
        self.stream_type = nn.Parameter(torch.zeros(len(NON_IMAGING_STREAMS), config.embed_dim))
        self.presence_embedding = nn.Embedding(2, config.embed_dim) if config.presence_embedding else None
        self.absence_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.pool_query = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        nn.init.trunc_normal_(self.stream_type, std=0.02)
        nn.init.trunc_normal_(self.absence_token, std=0.02)
        nn.init.trunc_normal_(self.pool_query, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=4 * config.embed_dim,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=config.layers)
        self.input_norm = nn.LayerNorm(config.embed_dim)
        self.output_norm = nn.LayerNorm(config.embed_dim)

    def _stream_token(self, stream: Stream, inputs: FusionInputs) -> Tensor:
        token: Tensor
        if stream is Stream.RADIOMICS:
            token = self.radiomics_encoder(inputs.radiomics)
        elif stream is Stream.CLINICAL:
            token = self.clinical_encoder(inputs.clinical)
        elif stream is Stream.ENDOSCOPY:
            token = self.endoscopy_encoder(inputs.endoscopy)
        else:
            token = self.pathology_encoder(inputs.pathology)
        return token

    def _presence(self, inputs: FusionInputs, stream: Stream) -> Tensor:
        """Presence flag of one stream, or 1 when the arm drops it entirely from dropout."""
        position = NON_IMAGING_STREAMS.index(stream)
        if inputs.presence.shape[1] <= position:
            return torch.ones(inputs.presence.shape[0], device=inputs.presence.device, dtype=inputs.presence.dtype)
        return inputs.presence[:, position]

    def forward(self, inputs: FusionInputs) -> FusionOutput:
        batch = inputs.radiomics.shape[0]
        tokens: list[Tensor] = []
        if self.use_image_tokens:
            tokens.append(self.image_projection(inputs.image_tokens))

        stream_tokens: list[Tensor] = []
        for position, stream in enumerate(NON_IMAGING_STREAMS):
            if stream not in self.active_streams:
                continue
            present = self._presence(inputs, stream)
            encoded = self._stream_token(stream, inputs)
            # The presence flag is shaped (batch, 1, 1) so it broadcasts along the token
            # axis of a (batch, 1, dim) token rather than against it.
            flag = present.unsqueeze(-1).unsqueeze(-1)
            absence = self.absence_token.expand(batch, 1, self.embed_dim)
            token = encoded.unsqueeze(1) * flag + absence * (1.0 - flag)
            token = token + self.stream_type[position].view(1, 1, self.embed_dim)
            if self.presence_embedding is not None:
                token = token + self.presence_embedding(present.long()).unsqueeze(1)
            tokens.append(token)
            stream_tokens.append(token.squeeze(1))

        context = torch.cat(tokens, dim=1)
        contextualised = self.blocks(self.input_norm(context))
        query = self.pool_query.expand(batch, 1, self.embed_dim)
        weights = torch.softmax(query @ contextualised.transpose(1, 2) / (self.embed_dim**0.5), dim=-1)
        fused = (weights @ contextualised).squeeze(1)
        stacked = torch.stack(stream_tokens, dim=1) if stream_tokens else torch.zeros(batch, 0, self.embed_dim, device=context.device)
        return FusionOutput(fused=self.output_norm(fused), context=contextualised, stream_tokens=stacked)


class EarlyConcatFusion(CrossAttentionFusion):
    """Rejected alternative: concatenate every stream's feature vector, then project.

    Kept so the justification in Methods Sec. 4.6 is an evaluated comparison.
    """

    def __init__(self, *args: object, stream_dims: tuple[int, ...] = (), **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        if not stream_dims:
            raise ValueError("EarlyConcatFusion needs the concatenated stream dimensions")
        self.concat_projection = nn.Sequential(
            nn.Linear(sum(stream_dims), self.embed_dim),
            nn.GELU(),
            nn.LayerNorm(self.embed_dim),
        )
        self.concat_streams = stream_dims

    def forward(self, inputs: FusionInputs) -> FusionOutput:
        pieces = self._concat_pieces(inputs)
        vector: Tensor = self.concat_projection(torch.cat(pieces, dim=-1))
        context = vector.unsqueeze(1)
        return FusionOutput(fused=vector, context=context, stream_tokens=context)

    def _concat_pieces(self, inputs: FusionInputs) -> list[Tensor]:
        """Raw per-stream vectors, with a zero vector standing in for an absent stream."""
        pieces: list[Tensor] = []
        for stream in NON_IMAGING_STREAMS:
            if stream is Stream.RADIOMICS:
                value = inputs.radiomics
            elif stream is Stream.CLINICAL:
                value = inputs.clinical
            elif stream is Stream.ENDOSCOPY:
                value = inputs.endoscopy.float().mean(dim=1, keepdim=True).expand(-1, inputs.clinical.shape[1])
            else:
                value = inputs.pathology.float().mean(dim=1, keepdim=True).expand(-1, inputs.clinical.shape[1])
            present = self._presence(inputs, stream).unsqueeze(-1)
            pieces.append(value * present)
        return pieces


class LateAverageFusion(CrossAttentionFusion):
    """Rejected alternative: average each stream's own prediction after its own head.

    The block still encodes every stream into its own token, but performs no
    cross-stream attention: the tokens reach independent heads and the logits are
    averaged in :class:`stagefm.models.stagefm.STAGEFM`. That is exactly the design
    the manuscript rejects, since averaging cannot express an interaction between
    streams.
    """

    def forward(self, inputs: FusionInputs) -> FusionOutput:
        batch = inputs.radiomics.shape[0]
        stream_tokens: list[Tensor] = []
        for position, stream in enumerate(NON_IMAGING_STREAMS):
            if stream not in self.active_streams:
                continue
            present = self._presence(inputs, stream)
            encoded = self._stream_token(stream, inputs)
            flag = present.unsqueeze(-1)
            absence = self.absence_token.expand(batch, 1, self.embed_dim).squeeze(1)
            token = encoded * flag + absence * (1.0 - flag)
            stream_tokens.append(token + self.stream_type[position].view(1, self.embed_dim))
        if not stream_tokens:
            raise ValueError("late-average fusion needs at least one active stream")
        stacked = torch.stack(stream_tokens, dim=1)
        return FusionOutput(fused=stacked.mean(dim=1), context=stacked, stream_tokens=stacked)


def build_fusion(
    strategy: str,
    config: FusionConfig,
    radiomics_dim: int,
    clinical_dim: int,
    endoscopy_vocab: int,
    pathology_vocab: int,
    active_streams: tuple[Stream, ...],
    use_image_tokens: bool = True,
) -> nn.Module:
    """Construct the fusion block named by ``strategy``."""
    common = {
        "config": config,
        "radiomics_dim": radiomics_dim,
        "clinical_dim": clinical_dim,
        "endoscopy_vocab": endoscopy_vocab,
        "pathology_vocab": pathology_vocab,
        "active_streams": active_streams,
        "use_image_tokens": use_image_tokens,
    }
    if strategy == "cross_attention":
        return CrossAttentionFusion(**common)  # type: ignore[arg-type]
    if strategy == "early_concat":
        return EarlyConcatFusion(stream_dims=(radiomics_dim, clinical_dim, clinical_dim, clinical_dim), **common)
    if strategy == "late_average":
        return LateAverageFusion(**common)  # type: ignore[arg-type]
    raise ValueError(f"unknown fusion strategy: {strategy}")
