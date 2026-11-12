"""Narrow, zero-gated action-to-demo attention for the RoboICL A variant."""

import torch
from torch import nn
from torch.nn import functional as F

from demo_features import FEATURE_DIM, SCHEMA


SPEC = {
    "feature_schema": SCHEMA,
    "layers": [7, 15, 23],  # zero-based backbone block indices
    "width": 256,
    "heads": 4,
}


class DemoAttention(nn.Module):
    def __init__(self, action_dim, width, heads):
        super().__init__()
        self.heads = heads
        self.norm = nn.LayerNorm(action_dim)
        self.query = nn.Linear(action_dim, width, bias=False)
        self.key_value = nn.Linear(FEATURE_DIM, 2 * width, bias=False)
        self.out = nn.Linear(width, action_dim, bias=False)
        # FSDP2 requires every managed parameter to have a sharding dimension.
        self.gate = nn.Parameter(torch.zeros(1))

    def split_heads(self, value):
        return value.unflatten(-1, (self.heads, -1)).transpose(1, 2)

    def forward(self, action, demo):
        query = self.split_heads(self.query(self.norm(action)))
        key, value = self.key_value(demo.to(action.dtype)).chunk(2, dim=-1)
        attended = F.scaled_dot_product_attention(query, self.split_heads(key), self.split_heads(value))
        residual = self.out(attended.transpose(1, 2).flatten(2))
        return action + self.gate.tanh().to(action.dtype) * residual


def attach_demo_attention(model, spec):
    if spec != SPEC:
        raise ValueError(f"Unsupported Robot-ICL + A architecture: {spec}")
    if not getattr(model, "supports_demo_attention", False):
        raise RuntimeError("Apply demo_attention.patch to the isolated IG checkout")
    for index in spec["layers"]:
        block = model.blocks[index]
        block.demo_attention = DemoAttention(model.action_inner_dim, spec["width"], spec["heads"])


def add_to_adapter_report(model, report):
    shapes = {name: list(param.shape) for name, param in model.named_parameters()
              if ".demo_attention." in name and param.requires_grad}
    if not shapes:
        raise RuntimeError("No trainable demo attention parameters")
    count = sum(param.numel() for name, param in model.named_parameters() if name in shapes)
    report["trainable_shapes"].update(shapes)
    report["trainable_parameters"] += count
    report["demo_attention_parameters"] = count
