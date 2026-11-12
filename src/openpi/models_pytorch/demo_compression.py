"""Parameter-free, timestamp-window Demo compression and per-layer layouts."""
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PrefixCompressionPlan:
    compression_layer: int
    demo_token_count: int
    token_window_ids: torch.Tensor
    early_valid: torch.Tensor
    early_blocks: torch.Tensor
    late_valid: torch.Tensor
    late_blocks: torch.Tensor

    def stage(self, layer_index):
        return int(layer_index >= self.compression_layer)


@dataclass(frozen=True)
class PrefixKVBundle:
    cache: object
    compression_layer: int
    early_valid: torch.Tensor
    late_valid: torch.Tensor

    def valid_at(self, layer_index):
        return self.early_valid if layer_index < self.compression_layer else self.late_valid


def build_compression_plan(timestamps, prefix_valid, prefix_blocks, *, frames, views,
                           tokens_per_view, window_count, window_seconds, compression_layer):
    batch = prefix_valid.shape[0]
    count = frames * views * tokens_per_view
    if timestamps is None or timestamps.shape != (batch, frames):
        raise ValueError("Compression requires Demo timestamps [batch, frames]")
    if prefix_valid.dtype != torch.bool or prefix_blocks.shape != prefix_valid.shape:
        raise ValueError("Expected matching boolean prefix validity/block layouts")
    if count >= prefix_valid.shape[1] or window_count < 1 or window_seconds <= 0:
        raise ValueError("Invalid Demo boundary or temporal window configuration")
    frame_valid = prefix_valid[:, :count].reshape(batch, frames, -1).any(-1)
    safe_times = torch.where(frame_valid, timestamps, 0)
    if not torch.isfinite(safe_times).all() or (safe_times < 0).any():
        raise ValueError("Valid Demo timestamps must be finite nonnegative seconds")
    window_ids = torch.floor(safe_times / window_seconds).long()
    if (window_ids[frame_valid] >= window_count).any():
        raise ValueError("Demo exceeds the configured temporal windows; never truncate it")
    for times, valid in zip(safe_times, frame_valid, strict=True):
        if (torch.diff(times[valid]) <= 0).any():
            raise ValueError("Valid Demo timestamps must increase strictly")
    window_ids = window_ids.repeat_interleave(views * tokens_per_view, dim=1)
    token_valid = prefix_valid[:, :count]
    window_ids = torch.where(token_valid, window_ids, -1)
    counts = torch.zeros(batch, window_count, device=prefix_valid.device, dtype=torch.long)
    counts.scatter_add_(1, window_ids.clamp_min(0), token_valid.long())
    late_valid = torch.cat((counts > 0, prefix_valid[:, count:]), dim=1)
    late_blocks = torch.zeros_like(late_valid)
    late_blocks[:, :window_count + 1] = True  # Every summary, then source/query block.
    return PrefixCompressionPlan(compression_layer, count, window_ids, prefix_valid,
                                 prefix_blocks, late_valid, late_blocks)


def compress_demo_hidden(prefix_hidden, plan):
    count = plan.demo_token_count
    windows = plan.late_valid.shape[1] - (prefix_hidden.shape[1] - count)
    valid = plan.early_valid[:, :count]
    payload = torch.where(valid[..., None], prefix_hidden[:, :count], 0).float()
    ids = plan.token_window_ids.clamp_min(0)
    sums = payload.new_zeros(payload.shape[0], windows, payload.shape[-1])
    sums = sums.scatter_add(1, ids[..., None].expand_as(payload), payload)
    counts = payload.new_zeros(payload.shape[0], windows)
    counts = counts.scatter_add(1, ids, valid.to(payload.dtype))
    summaries = sums / counts.clamp_min(1)[..., None]
    summaries = torch.where(counts[..., None] > 0, summaries, 0).to(prefix_hidden.dtype)
    # Query hidden states already conditioned on dense Demo remain unchanged.
    return torch.cat((summaries, prefix_hidden[:, count:]), dim=1)


def prefix_layout(valid, blocks):
    order = blocks.cumsum(1)
    allowed = (order[:, None, :] <= order[:, :, None]) & valid[:, None, :] & valid[:, :, None]
    return allowed, valid.long().cumsum(1) - 1


def joint_layout(prefix_valid, prefix_blocks, suffix_valid, suffix_blocks):
    return prefix_layout(torch.cat((prefix_valid, suffix_valid), 1),
                         torch.cat((prefix_blocks, suffix_blocks), 1))


def suffix_layout(prefix_valid, suffix_valid, suffix_blocks):
    own, _ = prefix_layout(suffix_valid, suffix_blocks)
    previous = suffix_valid[:, :, None] & prefix_valid[:, None, :]
    positions = prefix_valid.sum(1)[:, None] + suffix_valid.long().cumsum(1) - 1
    return torch.cat((previous, own), -1), positions
