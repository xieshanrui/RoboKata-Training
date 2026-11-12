"""Shared-vision Demo frame attention pool: four tokens per frame and view."""

import math

import torch
from torch import nn


class DemoAttentionPool(nn.Module):
    def __init__(self, width: int, frames: int = 10, views: int = 2, tokens_per_view: int = 4, heads: int = 8, valid_only: bool = False):
        super().__init__()
        if min(width, frames, views, tokens_per_view, heads) < 1 or width % heads:
            raise ValueError("Positive dimensions and width divisible by heads required")
        self.width = width
        self.frames = frames
        self.views = views
        self.tokens_per_view = tokens_per_view
        self.heads = heads
        self.valid_only = valid_only
        self.norm = nn.LayerNorm(width)
        self.queries = nn.Parameter(torch.randn(tokens_per_view, width) / math.sqrt(width))
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.output = nn.Linear(width, width)
        self.mlp = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, width))

    def forward(self, patches: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # patches: [batch, frame, view, patch, width]; valid: [batch, frame, view]
        if patches.ndim != 5 or patches.shape[1] < 1 or patches.shape[2] != self.views or patches.shape[-1] != self.width:
            raise ValueError("Demo patch features have an incompatible shape")
        if patches.shape[3] < 1 or valid.shape != patches.shape[:3] or valid.dtype != torch.bool:
            raise ValueError("Expected nonempty patches and a boolean per-frame/view validity mask")
        # Suppress invalid payloads before attention; missing views may contain NaN.
        patches = torch.where(valid[..., None, None], patches, 0).to(self.key.weight.dtype)
        if self.valid_only:
            patches = patches[valid]
        normalized = self.norm(patches)
        count, dim = patches.shape[-2], self.width // self.heads
        key = self.key(normalized).reshape(-1, count, self.heads, dim).transpose(1, 2)
        value = self.value(normalized).reshape(-1, count, self.heads, dim).transpose(1, 2)
        query = self.queries.reshape(self.tokens_per_view, self.heads, dim).transpose(0, 1)
        scores = torch.einsum("hkd,nhpd->nhkp", query.float(), key.float()) / math.sqrt(dim)
        tokens = torch.einsum("nhkp,nhpd->nhkd", scores.softmax(-1).to(value.dtype), value)
        tokens = tokens.transpose(1, 2).reshape(-1, self.tokens_per_view, self.width)
        tokens = self.output(tokens) + self.queries[None]
        tokens = tokens + self.mlp(tokens)
        if self.valid_only:
            restored = tokens.new_zeros((*valid.shape, self.tokens_per_view, self.width))
            restored[valid] = tokens
            tokens = restored
        else:
            tokens = tokens.reshape(*valid.shape, self.tokens_per_view, self.width)
        mask = valid[..., None].expand(*valid.shape, self.tokens_per_view)
        tokens = torch.where(mask[..., None], tokens, torch.zeros_like(tokens))
        return tokens.flatten(1, 3), mask.flatten(1, 3)
