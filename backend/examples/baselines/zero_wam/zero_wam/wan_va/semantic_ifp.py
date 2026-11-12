"""Train-only predictions of future DINO features from causal video states."""

import torch
from torch import nn
from torch.nn import functional as F


class SemanticIFPHead(nn.Module):
    def __init__(self, hidden_dim, horizons, target_dim, grid, width=256):
        super().__init__()
        self.grid = grid
        self.bottleneck = nn.Sequential(nn.Linear(hidden_dim, width), nn.GELU())
        self.predictions = nn.ModuleList(nn.Linear(width, target_dim) for _ in range(horizons))

    def pool(self, states, frames, height, width):
        batch, _, channels = states.shape
        maps = states.reshape(batch * frames, height, width, channels).permute(0, 3, 1, 2)
        pooled = F.adaptive_avg_pool2d(maps, self.grid)
        return pooled.flatten(2).transpose(1, 2).reshape(batch, frames, self.grid**2, channels)

    def forward(self, states):
        compact = self.bottleneck(states)
        return [head(compact) for head in self.predictions]


def future_anchor_pairs(valid, horizons, chunk_size, future_chunk_stride):
    """Yield a shift and mask for real source and destination anchors."""
    for horizon in range(horizons):
        shift = (1 + horizon * future_chunk_stride) * chunk_size
        count = max(0, valid.shape[1] - shift)
        yield shift, valid[:, :count] & valid[:, shift:shift + count]


def semantic_ifp_losses(predictions, features, valid, chunk_size, future_chunk_stride):
    """Cosine loss on real current/future anchors, averaged over valid patches."""
    losses = []
    pairs = future_anchor_pairs(valid, len(predictions), chunk_size, future_chunk_stride)
    for prediction, (shift, mask) in zip(predictions, pairs):
        count = mask.shape[1]
        predicted = prediction[:, :count].float()
        target = features[:, shift:shift + count].detach().float()
        errors = 1 - F.cosine_similarity(predicted, target, dim=-1)
        mask = mask.unsqueeze(-1).expand_as(errors)
        losses.append(errors.masked_fill(~mask, 0).sum() / mask.sum().clamp_min(1))
    return losses
