"""Franka adaptation of the released action diffusion interface.

The spatial/quaternion convention is unchanged. Both diffusion input and output
must have the same eight channels; the video and action Transformer blocks keep
their original widths and are fully trainable during adaptation.
"""
import torch


LEFT_CHANNELS = [0, 1, 2, 3, 4, 5, 6, 28]


def resize_action_interface(model):
    if model.action_embedder.in_features != 30 or model.action_proj_out.out_features != 30:
        raise ValueError('Expected the released 30-channel checkpoint')
    old_input, old_output = model.action_embedder, model.action_proj_out
    new_input = torch.nn.Linear(8, old_input.out_features, bias=old_input.bias is not None,
                                device=old_input.weight.device, dtype=old_input.weight.dtype)
    new_output = torch.nn.Linear(old_output.in_features, 8, bias=old_output.bias is not None,
                                 device=old_output.weight.device, dtype=old_output.weight.dtype)
    with torch.no_grad():
        new_input.weight.copy_(old_input.weight[:, LEFT_CHANNELS])
        if new_input.bias is not None:
            new_input.bias.copy_(old_input.bias)
        new_output.weight.copy_(old_output.weight[LEFT_CHANNELS])
        if new_output.bias is not None:
            new_output.bias.copy_(old_output.bias[LEFT_CHANNELS])
    model.action_embedder = new_input
    model.action_proj_out = new_output
    model.register_to_config(action_dim=8)


def configure_single_arm(config, adaptation):
    if adaptation['action_dim'] != 8 or adaptation['original_channels'] != LEFT_CHANNELS:
        raise ValueError('Unsupported action adaptation')
    config.action_dim = 8
    config.used_action_channel_ids = list(range(8))
    config.inverse_used_action_channel_ids = list(range(8))
    config.norm_stat = adaptation['norm_stat']
    if any(len(config.norm_stat[key]) != 8 for key in ['q01', 'q99']):
        raise ValueError('Expected training-only eight-channel action statistics')
