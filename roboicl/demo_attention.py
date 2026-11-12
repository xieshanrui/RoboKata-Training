"""Narrow, zero-gated action-to-demo attention for the RoboICL A variant."""

import torch
from torch import nn
from torch.nn import functional as F

from demo_features import DUAL_SCHEMA, FEATURE_DIM, FRAMES, GRID, SCHEMA, feature_shape


SPEC = {
    "feature_schema": SCHEMA,
    "layers": [7, 15, 23],  # zero-based backbone block indices
    "width": 256,
    "heads": 4,
}


def spec_for_views(views):
    return {**SPEC, "feature_schema": DUAL_SCHEMA} if views == 2 else SPEC


class DemoAttention(nn.Module):
    def __init__(self, action_dim, width, heads, views=1):
        super().__init__()
        self.heads = heads
        self.views = views
        self.norm = nn.LayerNorm(action_dim)
        self.query = nn.Linear(action_dim, width, bias=False)
        self.key_value = nn.Linear(feature_shape(views)[1], 2 * width, bias=False)
        self.out = nn.Linear(width, action_dim, bias=False)
        # FSDP2 requires every managed parameter to have a sharding dimension.
        self.gate = nn.Parameter(torch.zeros(1))

    def split_heads(self, value):
        return value.unflatten(-1, (self.heads, -1)).transpose(1, 2)

    def forward(self, action, demo, log_index=None):
        query = self.split_heads(self.query(self.norm(action)))
        key, value = self.key_value(demo.to(action.dtype)).chunk(2, dim=-1)
        bias = None
        if log_index is not None:
            frames = log_index.shape[1]
            if action.shape[1] % frames:
                raise ValueError("Action tokens do not align with foresight index frames")
            bias = log_index.repeat_interleave(action.shape[1] // frames, dim=1)
            bias = bias.repeat_interleave(self.views * GRID**2, dim=-1).unsqueeze(1)
            bias = (self.index_strength.float() * bias).to(query.dtype).contiguous()
        attended = F.scaled_dot_product_attention(
            query, self.split_heads(key), self.split_heads(value), attn_mask=bias,
        )
        residual = self.out(attended.transpose(1, 2).flatten(2))
        return action + self.gate.tanh().to(action.dtype) * residual


def attach_demo_attention(model, spec):
    if spec not in (SPEC, spec_for_views(2)):
        raise ValueError(f"Unsupported Robot-ICL + A architecture: {spec}")
    if not getattr(model, "supports_demo_attention", False):
        raise RuntimeError("Apply demo_attention.patch to the isolated IG checkout")
    views = 2 if spec["feature_schema"] == DUAL_SCHEMA else 1
    for index in spec["layers"]:
        block = model.blocks[index]
        block.demo_attention = DemoAttention(model.action_inner_dim, spec["width"], spec["heads"], views)


def add_to_adapter_report(model, report):
    shapes = {name: list(param.shape) for name, param in model.named_parameters()
              if ".demo_attention." in name and param.requires_grad}
    if not shapes:
        raise RuntimeError("No trainable demo attention parameters")
    count = sum(param.numel() for name, param in model.named_parameters() if name in shapes)
    report["trainable_shapes"].update(shapes)
    report["trainable_parameters"] += count
    report["demo_attention_parameters"] = count


def attach_foresight_indexing(model, schema):
    from examples.baselines.zero_wam.zero_wam.wan_va.foresight_indexing import (
        KEY_LAYERS, SCHEMA as INDEX_SCHEMA, ForesightIndexing,
    )

    if schema != INDEX_SCHEMA or set(KEY_LAYERS.values()) != set(SPEC["layers"]):
        raise ValueError(f"Unsupported foresight indexing architecture: {schema}")
    if model.patch_size[0] != 1:
        raise ValueError("Foresight indexing requires one temporal frame per video patch")
    views = model.blocks[SPEC["layers"][0]].demo_attention.views
    model.foresight_indexing = ForesightIndexing(
        model.inner_dim, FEATURE_DIM - FRAMES - GRID**2, GRID, FRAMES, views=views,
    )
    for layer in KEY_LAYERS.values():
        model.blocks[layer].demo_attention.index_strength = nn.Parameter(torch.zeros(1))
